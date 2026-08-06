"""HELICS federate that uses distopf (ENAPP per-area) for OPF.

Each instance of this federate represents ONE spatial decomposition area.
On each HELICS iteration it performs a single local OPF solve on its own
sub-network, publishes boundary variables (S_up, V_dn) to the hub
aggregators, and reads the aggregated boundary data from the hubs to apply
as boundary conditions for the next iteration.

Run via the installed entry point:
    distopf-federate-sim
"""

import json
import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import helics as h
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    Injection,
    MeasurementArray,
    PowersAngle,
    PowersImaginary,
    PowersMagnitude,
    PowersReal,
    Topology,
    VoltagesImaginary,
    VoltagesMagnitude,
    VoltagesReal,
)

import distopf as opf
from distopf.distributed.spatial.decompose import decompose

from distopf_federate.importer import (
    apply_s_up_to_sub_case,
    apply_v_dn_to_sub_case,
    topology_to_case,
    update_case_from_measurements,
)
from distopf_federate.exporter import (
    enapp_s_up_to_pq,
    enapp_v_dn_to_vmag,
    result_to_commands,
    result_to_controls_pq,
    result_to_power_angle,
    result_to_power_mag,
    result_to_solver_stats,
    result_to_voltage_mag,
)

logger = logging.getLogger(__name__)
# Libraries should not configure the root logger; callers decide handler/level.
logger.addHandler(logging.NullHandler())

OBJECTIVES: dict[str, Callable] = {
    "cp_obj_loss": opf.cp_obj_loss,
    "cp_obj_curtail": opf.cp_obj_curtail,
    "cp_obj_curtail_lp": opf.cp_obj_curtail_lp,
    "cp_obj_target_p_total": opf.cp_obj_target_p_total,
    "cp_obj_target_q_total": opf.cp_obj_target_q_total,
    "cp_obj_none": opf.cp_obj_none,
}


@dataclass
class StaticConfig:
    name: str = ""
    deltat: float = 1.0
    switches: list = field(default_factory=list)
    source: str = ""
    objective: str = "cp_obj_none"
    tol: float = 1e-4
    max_iterations: int = 50
    number_of_timesteps: int = 1


@dataclass
class Subscriptions:
    topology: object = None
    injections: object = None
    voltages_real: object = None
    voltages_imag: object = None
    sub_v: object = None
    sub_p: object = None
    sub_q: object = None


class DistopfFederate:
    """HELICS value federate that runs one distopf OPF per HELICS iteration.

    Each instance handles a single spatial decomposition area.  After
    ``init_area()`` it holds a ``sub_case`` — the local sub-network extracted
    from the full topology via ``decompose()`` — and exchanges boundary
    variables (V_dn, S_up) with the hub aggregators on every HELICS iteration.
    """

    def __init__(self, broker_config: BrokerConfig) -> None:
        # Full-network case (built from topology once)
        self.case: Optional[opf.Case] = None
        # Per-area decomposed sub-network case (solved each iteration)
        self.sub_case: Optional[opf.Case] = None
        # This federate's area name (e.g. "area_152" or "area_150" for root)
        self.area_name: Optional[str] = None
        # Resolved source bus name (may differ from static.source if source is switch ID)
        self.source_bus: Optional[str] = None
        # Child area names (dummy PQ node names in sub_case)
        self.down_buses: list = []
        # Previous iteration boundary S_up values for convergence tracking
        self._prev_s_up_vals: list = []
        self.name_to_id: Optional[dict] = None
        self.v_ln_base_map: Optional[dict] = None
        self.gen_tags: Optional[dict] = None
        self._initialized: bool = False

        self.sub = Subscriptions()
        self.load_static_inputs()
        self.load_input_mapping()
        self.initialize(broker_config)
        self.register_subscription()
        self.register_publication()

    def load_static_inputs(self) -> None:
        path = Path(__file__).parent / "static_inputs.json"
        with open(path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        self.static = StaticConfig(
            name=config["name"],
            deltat=float(config.get("deltat", 1.0)),
            switches=config.get("switches", []),
            source=config["source"],
            objective=config.get("objective", "cp_obj_none"),
            tol=float(config.get("tol", 1e-4)),
            max_iterations=int(config.get("max_iterations", 50)),
            number_of_timesteps=int(config.get("number_of_timesteps", 1)),
        )

    def load_input_mapping(self) -> None:
        path = Path(__file__).parent / "input_mapping.json"
        with open(path, "r", encoding="utf-8") as fh:
            self.inputs = json.load(fh)

    def initialize(self, broker_config: BrokerConfig) -> None:
        self.info = h.helicsCreateFederateInfo()
        self.info.core_name = self.static.name
        self.info.core_type = h.HELICS_CORE_TYPE_ZMQ
        self.info.core_init = "--federates=1"
        h.helicsFederateInfoSetBroker(self.info, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(self.info, broker_config.broker_port)
        self.fed = h.helicsCreateValueFederate(self.static.name, self.info)
        h.helicsFederateSetTimeProperty(
            self.fed, h.HELICS_PROPERTY_TIME_PERIOD, int(self.static.deltat)
        )

    def register_subscription(self) -> None:
        self.sub.topology = self.fed.register_subscription(
            self.inputs["topology"], ""
        )
        self.sub.injections = self.fed.register_subscription(
            self.inputs["injections"], ""
        )
        self.sub.voltages_real = self.fed.register_subscription(
            self.inputs["voltages_real"], ""
        )
        self.sub.voltages_imag = self.fed.register_subscription(
            self.inputs["voltages_imag"], ""
        )
        self.sub.sub_v = self.fed.register_subscription(self.inputs["sub_v"], "")
        self.sub.sub_p = self.fed.register_subscription(self.inputs["sub_p"], "")
        self.sub.sub_q = self.fed.register_subscription(self.inputs["sub_q"], "")

    def register_publication(self) -> None:
        self.pub_c = self.fed.register_publication(
            "pub_c", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_solver_stats = self.fed.register_publication(
            "solver_stats", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_controls_real = self.fed.register_publication(
            "controls_real", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_controls_imag = self.fed.register_publication(
            "controls_imag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_powers_mag = self.fed.register_publication(
            "powers_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_powers_ang = self.fed.register_publication(
            "powers_ang", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltages_mag = self.fed.register_publication(
            "voltages_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_v = self.fed.register_publication(
            "pub_v", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_p = self.fed.register_publication(
            "pub_p", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_q = self.fed.register_publication(
            "pub_q", h.HELICS_DATA_TYPE_STRING, ""
        )

    def _get_objective_fn(self) -> Optional[Callable]:
        return OBJECTIVES.get(self.static.objective)

    def init_area(self) -> None:
        """Parse topology subscription, build full Case, decompose to local sub_case."""
        topology: Topology = Topology.parse_obj(self.sub.topology.json)

        # Resolve the source bus (handles both direct bus names and switch IDs)
        self.source_bus = self._resolve_source_bus(topology)

        case, name_to_id, v_ln_base_map = topology_to_case(
            topology,
            source_bus=self.source_bus,
        )

        self.case = case
        self.name_to_id = name_to_id
        self.v_ln_base_map = v_ln_base_map

        # Collect equipment tags per bus for command publication
        self.gen_tags = self._collect_gen_tags(topology)

        # Decompose full case to extract this area's local sub_case
        self._build_sub_case(topology, case)

        self._initialized = True
        logger.info(
            "Area '%s' initialized: source_bus=%s, %d buses, %d branches, "
            "%d generators, down_buses=%s",
            self.area_name,
            self.source_bus,
            len(case.bus_data),
            len(case.branch_data),
            len(case.gen_data) if case.gen_data is not None else 0,
            self.down_buses,
        )

    def _resolve_source_bus(self, topology: Topology) -> str:
        """Resolve ``static.source`` to an actual bus name.

        ``static.source`` may be either:
        - A bus name directly (e.g. ``"150"`` for the root area).
        - A switch / equipment ID (e.g. ``"sw3"`` for a child area), in which
          case the downstream bus of that switch is returned.

        The resolution mirrors the approach used by ``admm_federate.opf_federate``
        which scans the network graph's boundary edges to find the source bus.

        Parameters
        ----------
        topology : Topology

        Returns
        -------
        str
            Resolved source bus name.
        """
        source = self.static.source
        incidences = topology.incidences

        # Collect all known bus names from the topology base voltages
        bus_names: set = set()
        for id_str in topology.base_voltage_magnitudes.ids:
            bus_names.add(id_str.split(".", 1)[0])

        if source in bus_names:
            # Already a bus name — no resolution needed (root area case)
            return source

        # Treat source as a switch / equipment ID and find its downstream bus
        for fr_eq, to_eq, eq_id in zip(
            incidences.from_equipment,
            incidences.to_equipment,
            incidences.ids,
        ):
            if eq_id == source:
                to_bus = to_eq.split(".", 1)[0]
                logger.debug(
                    "Resolved source '%s' (switch ID) → bus '%s'", source, to_bus
                )
                return to_bus

        logger.warning(
            "Could not resolve source '%s' to a bus name; using it verbatim.", source
        )
        return source

    def _collect_gen_tags(self, topology: Topology) -> dict:
        """Build bus_name → list[equipment_tag] for PVSystem generators."""
        gen_tags: dict = {}
        real = topology.injections.power_real
        for id_str, eq in zip(real.ids, real.equipment_ids):
            if "PVSystem" not in eq:
                continue
            name = id_str.split(".", 1)[0]
            if name not in gen_tags:
                gen_tags[name] = []
            if eq not in gen_tags[name]:
                gen_tags[name].append(eq)
        return gen_tags

    def _build_sub_case(self, topology: Topology, full_case: opf.Case) -> None:
        """Decompose the full network into per-area sub-cases and store this area's.

        Scans ``static.switches`` in the topology incidences to identify each
        child area's source bus, then calls ``decompose(full_case, sources)`` to
        produce properly wired sub-cases (with dummy SWING / dummy PQ boundary
        nodes).  Stores:

        - ``self.sub_case`` — the local sub-network for this federate.
        - ``self.area_name`` — identifier for this area (``"area_{source_bus}"``).
        - ``self.down_buses`` — child area names (dummy PQ node names in sub_case).

        Parameters
        ----------
        topology : Topology
        full_case : distopf.Case
            The full network case built from the same topology.
        """
        self.area_name = f"area_{self.source_bus}"
        incidences = topology.incidences

        # Build sources dict: { area_name: source_bus } for this area + children
        sources: dict = {self.area_name: self.source_bus}
        child_area_names: list = []

        for fr_eq, to_eq, eq_id in zip(
            incidences.from_equipment,
            incidences.to_equipment,
            incidences.ids,
        ):
            if eq_id not in self.static.switches:
                continue
            to_bus = to_eq.split(".", 1)[0]
            # Skip the switch that represents *this* area's own parent boundary
            if to_bus == self.source_bus:
                continue
            child_name = f"area_{to_bus}"
            sources[child_name] = to_bus
            child_area_names.append(child_name)

        self.down_buses = child_area_names

        if len(sources) == 1 and not self.static.switches:
            # No switches → single area; sub_case IS the full case
            self.sub_case = full_case
            logger.debug("No switches — sub_case equals full case for area '%s'", self.area_name)
            return

        try:
            area_cases = decompose(full_case, sources)
        except Exception:
            logger.exception(
                "decompose() failed for area '%s'; falling back to full case", self.area_name
            )
            self.sub_case = full_case
            return

        if self.area_name not in area_cases:
            logger.warning(
                "Area '%s' not found in decompose output (keys: %s); using full case",
                self.area_name,
                list(area_cases.keys()),
            )
            self.sub_case = full_case
        else:
            self.sub_case = area_cases[self.area_name]
            logger.debug(
                "Sub-case built for area '%s': %d buses, %d branches",
                self.area_name,
                len(self.sub_case.bus_data),
                len(self.sub_case.branch_data),
            )


    def _read_injection(self) -> Optional[Injection]:
        if self.sub.injections.is_updated():
            return Injection.parse_obj(self.sub.injections.json)
        return None

    def _read_voltages_mag(self) -> Optional[VoltagesMagnitude]:
        """Compute voltage magnitude from real/imag subscriptions if updated."""
        if not (
            self.sub.voltages_real.is_updated()
            and self.sub.voltages_imag.is_updated()
        ):
            return None
        vr = VoltagesReal.parse_obj(self.sub.voltages_real.json)
        vi = VoltagesImaginary.parse_obj(self.sub.voltages_imag.json)

        vr_dict = dict(zip(vr.ids, vr.values))
        vi_dict = dict(zip(vi.ids, vi.values))
        ids, values = [], []
        for id_str, vr_val in vr_dict.items():
            vi_val = vi_dict.get(id_str, 0.0)
            ids.append(id_str)
            values.append((vr_val**2 + vi_val**2) ** 0.5)

        time = getattr(vr, "time", 0) or 0
        return VoltagesMagnitude(ids=ids, values=values, time=time)

    def _publish_empty(self, t: int) -> None:
        """Publish empty/default messages when the OPF fails or topology is not ready."""
        empty_v = VoltagesMagnitude(ids=[], values=[], time=t)
        empty_p = PowersReal(ids=[], equipment_ids=[], values=[], time=t)
        empty_q = PowersImaginary(ids=[], equipment_ids=[], values=[], time=t)
        empty_mag = PowersMagnitude(ids=[], equipment_ids=[], values=[], time=t)
        empty_ang = PowersAngle(ids=[], equipment_ids=[], values=[], time=t)
        stats = result_to_solver_stats(
            converged=False,
            objective_value=None,
            iterations=0,
            solve_time=0.0,
            time=t,
        )
        self.pub_c.publish(json.dumps([]))
        self.pub_solver_stats.publish(stats.json())
        self.pub_controls_real.publish(empty_p.json())
        self.pub_controls_imag.publish(empty_q.json())
        self.pub_voltages_mag.publish(empty_v.json())
        self.pub_powers_mag.publish(empty_mag.json())
        self.pub_powers_ang.publish(empty_ang.json())
        self.pub_v.publish(empty_v.json())
        self.pub_p.publish(empty_p.json())
        self.pub_q.publish(empty_q.json())

    def _publish_results(self, result, t: int) -> None:
        """Publish all output topics from a sub-area PowerFlowResult.

        This publishes both the standard OEDISI outputs (voltage magnitudes,
        branch powers, generator setpoints) and the ENAPP boundary variables
        (S_up via pub_p/pub_q, V_dn via pub_v) that the hub aggregators
        forward to neighbouring area federates.
        """
        # Inverter setpoint commands
        commands = result_to_commands(result, self.gen_tags or {}, t)
        self.pub_c.publish(json.dumps(commands))

        # Solver diagnostics
        stats = result_to_solver_stats(
            converged=bool(getattr(result, "converged", False)),
            objective_value=getattr(result, "objective_value", None),
            iterations=int(getattr(result, "iterations", 0) or 0),
            solve_time=float(getattr(result, "solve_time", 0.0) or 0.0),
            time=t,
        )
        self.pub_solver_stats.publish(stats.json())

        # Generator P/Q control setpoints
        ctrl_p, ctrl_q = result_to_controls_pq(result, self.gen_tags or {}, t)
        self.pub_controls_real.publish(ctrl_p.json())
        self.pub_controls_imag.publish(ctrl_q.json())

        # Full-network voltage magnitude (for OEDISI recorders / displays)
        v_mag = result_to_voltage_mag(result, self.v_ln_base_map or {}, t)
        self.pub_voltages_mag.publish(v_mag.json())

        # Branch power magnitude and angle
        p_mag = result_to_power_mag(result, self.v_ln_base_map or {}, t)
        self.pub_powers_mag.publish(p_mag.json())

        p_ang = result_to_power_angle(result, t)
        self.pub_powers_ang.publish(p_ang.json())

        # ── ENAPP boundary variable publications ──────────────────────────
        # S_up (upstream power injection): parent area reads this to update
        # its dummy PQ node for the next iteration.
        sub_case = self.sub_case if self.sub_case is not None else self.case
        pub_p, pub_q = enapp_s_up_to_pq(sub_case, result, self.area_name or "", t)
        self.pub_p.publish(pub_p.json())
        self.pub_q.publish(pub_q.json())

        # V_dn (downstream boundary voltages): child areas read this to set
        # their swing-bus voltage for the next iteration.
        pub_v = enapp_v_dn_to_vmag(sub_case, result, self.down_buses, t)
        self.pub_v.publish(pub_v.json())

        # Update convergence tracking: compare S_up values to previous iter.
        curr_vals = pub_p.values + pub_q.values
        if self._prev_s_up_vals:
            if len(curr_vals) == len(self._prev_s_up_vals):
                import math as _math
                max_dev = max(
                    abs(c - p) for c, p in zip(curr_vals, self._prev_s_up_vals)
                )
                if max_dev <= self.static.tol:
                    logger.debug(
                        "Area '%s' boundary converged (dev=%.2e <= tol=%.2e)",
                        self.area_name, max_dev, self.static.tol,
                    )
                    self.converged = True
        self._prev_s_up_vals = list(curr_vals)

    def first_pub(self, t: float) -> None:
        """Publish empty initial values at the start of each timestep's iteration loop."""
        self._prev_s_up_vals = []
        self._publish_empty(int(t))

    def itr_pub(self) -> None:
        """Read subscriptions, apply boundary conditions, solve sub-area OPF, publish.

        Each call represents ONE ENAPP iteration for this area:

        1. Apply V_dn received from the hub (``sub_v``) to the local sub_case
           swing-bus schedule.
        2. Apply S_up from child areas received from the hub (``sub_p``,
           ``sub_q``) to the dummy PQ-node load schedules in sub_case.
        3. Update sub_case loads/generation from live feeder measurements.
        4. Solve ``sub_case.run_opf()`` for one iteration.
        5. Publish S_up (``pub_p``, ``pub_q``) and V_dn (``pub_v``) for the
           hub to forward to the parent/child areas.
        """
        if not self._initialized:
            if not self.sub.topology.is_updated():
                logger.warning("Topology not yet available during iteration; publishing empty")
                self._publish_empty(self._current_t)
                return
            self.init_area()

        objective_fn = self._get_objective_fn()
        sub_case = self.sub_case if self.sub_case is not None else self.case

        # ── Step 1 & 2: Apply boundary conditions from hub ─────────────────
        if self.sub.sub_v.is_updated() and self.area_name:
            sub_v_msg = VoltagesMagnitude.parse_obj(self.sub.sub_v.json)
            if sub_v_msg.values:
                apply_v_dn_to_sub_case(sub_case, sub_v_msg, self.area_name)

        if self.down_buses and self.sub.sub_p.is_updated() and self.sub.sub_q.is_updated():
            sub_p_msg = PowersReal.parse_obj(self.sub.sub_p.json)
            sub_q_msg = PowersImaginary.parse_obj(self.sub.sub_q.json)
            if sub_p_msg.values or sub_q_msg.values:
                apply_s_up_to_sub_case(sub_case, sub_p_msg, sub_q_msg, self.down_buses)

        # ── Step 3: Update live measurements ──────────────────────────────
        injection = self._read_injection()
        voltages_mag = self._read_voltages_mag()

        if injection is not None:
            update_case_from_measurements(
                sub_case,
                injection,
                self.name_to_id,
                voltages_mag=voltages_mag,
            )

        # ── Step 4: Solve local sub-area OPF ──────────────────────────────
        tic = _time.perf_counter()
        result = None
        try:
            if objective_fn is not None:
                result = sub_case.run_opf(objective_fn)
            else:
                result = sub_case.run_pf()
        except Exception:
            logger.exception("OPF solve failed for area '%s' at t=%d", self.area_name, self._current_t)

        elapsed = _time.perf_counter() - tic
        if result is not None:
            logger.info(
                "area=%s  t=%d  converged=%s  obj=%.4g  solve_time=%.2fs",
                self.area_name,
                self._current_t,
                getattr(result, "converged", "?"),
                getattr(result, "objective_value", float("nan")) or float("nan"),
                elapsed,
            )
            # ── Step 5: Publish results and boundary variables ─────────────
            self._publish_results(result, self._current_t)
        else:
            logger.error("No result for area '%s' at t=%d; publishing empty", self.area_name, self._current_t)
            self._publish_empty(self._current_t)

    def run(self) -> None:
        try:
            logger.info("Federate connected: %s", datetime.now())
            itr_need = h.helics_iteration_request_iterate_if_needed
            itr_stop = h.helics_iteration_request_no_iteration

            # Hybrid Fallback Initialization Pattern
            h.helicsFederateEnterInitializingMode(self.fed)
            try:
                topo_json = self.sub.topology.json
                if topo_json:
                    self.init_area()
                    logger.info("Configured area in initialization mode.")
            except Exception as e:
                logger.debug("Topology not available during initialization mode: %s", e)

            h.helicsFederateEnterExecutingMode(self.fed)
            logger.info("Federate executing: %s", datetime.now())

            granted_time = 0.0
            self._current_t = 0
            logger.debug("Starting time/iteration loop")

            while True:
                if (
                    self.static.number_of_timesteps > 0
                    and granted_time >= self.static.number_of_timesteps
                ):
                    logger.info(
                        "Reached end time %d. Exiting loop.",
                        self.static.number_of_timesteps,
                    )
                    break

                request_time = granted_time + 1.0
                itr_flag = itr_need
                self.converged = False
                self.itr = 0
                self.first_pub(granted_time)

                while True:
                    logger.debug("Requesting time %s with flag %s", request_time, itr_flag)
                    granted_time, itr_status = h.helicsFederateRequestTimeIterative(
                        self.fed, request_time, itr_flag
                    )
                    logger.info("\tgranted time = %s", granted_time)
                    logger.info("\titr status = %s", itr_status)

                    if granted_time >= h.HELICS_TIME_MAXTIME:
                        logger.info("HELICS Max Time reached. Exiting loop.")
                        break

                    if (
                        granted_time > 0.0
                        and self.itr == 0
                        and not self.sub.voltages_real.is_updated()
                    ):
                        logger.info("Feeder disconnected. Exiting loop.")
                        granted_time = h.HELICS_TIME_MAXTIME
                        break

                    if itr_status == h.helics_iteration_result_error:
                        logger.error("HELICS iteration request failed with error status.")
                        break

                    if itr_status == h.helics_iteration_result_next_step:
                        logger.debug("Advancing to next timestep at itr=%d", self.itr)
                        break

                    self.itr += 1
                    self._current_t = int(granted_time)
                    logger.info("\titr: %d", self.itr)
                    self.itr_pub()

                    if self.converged:
                        itr_flag = itr_stop
                    else:
                        itr_flag = itr_need

                if (
                    granted_time >= h.HELICS_TIME_MAXTIME
                    or itr_status == h.helics_iteration_result_error
                ):
                    break

        finally:
            self.stop()

    def stop(self) -> None:
        h.helicsFederateFinalize(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info("Federate finalized")


def run_simulator(broker_config: BrokerConfig) -> None:
    """Entry point for the OEDISI component framework."""
    federate = DistopfFederate(broker_config)
    federate.run()


def main() -> None:
    broker_config = BrokerConfig(
        broker_ip="127.0.0.1",
        broker_port=23404,
    )
    run_simulator(broker_config)


if __name__ == "__main__":
    main()
