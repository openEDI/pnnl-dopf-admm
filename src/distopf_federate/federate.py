"""HELICS federate that uses distopf (ENAPP or single-area) for OPF.

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
from distopf.distributed.spatial.enapp import solve_enapp

from distopf_federate.importer import topology_to_case, update_case_from_measurements
from distopf_federate.exporter import (
    result_to_commands,
    result_to_power_angle,
    result_to_power_mag,
    result_to_pub_pqv,
    result_to_solver_stats,
    result_to_voltage_angle,
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
    t_steps: int = 1


@dataclass
class Subscriptions:
    topology: object = None
    injections: object = None
    powers_real: object = None
    powers_imag: object = None
    voltages_real: object = None
    voltages_imag: object = None
    sub_v: object = None
    sub_p: object = None
    sub_q: object = None


class DistopfFederate:
    """HELICS value federate that runs distopf OPF each timestep."""

    def __init__(self, broker_config: BrokerConfig) -> None:
        self.case = None
        self.area_info: Optional[dict] = None
        self.name_to_id: Optional[dict] = None
        self.v_ln_base_map: Optional[dict] = None
        self.gen_tags: Optional[dict] = None
        self.boundary_buses: list = []
        self._initialized: bool = False

        self.sub = Subscriptions()
        self.load_static_inputs()
        self.load_input_mapping()
        self.initialize(broker_config)
        self.load_component_definition()
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
            t_steps=int(config.get("number_of_timesteps", 1)),
        )

    def load_input_mapping(self) -> None:
        path = Path(__file__).parent / "input_mapping.json"
        with open(path, "r", encoding="utf-8") as fh:
            self.inputs = json.load(fh)

    def load_component_definition(self) -> None:
        path = Path(__file__).parent / "component_definition.json"
        with open(path, "r", encoding="utf-8") as fh:
            self.component_config = json.load(fh)

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
        self.sub.powers_real = self.fed.register_subscription(
            self.inputs["power_real"], ""
        )
        self.sub.powers_imag = self.fed.register_subscription(
            self.inputs["power_imag"], ""
        )
        self.sub.voltages_real = self.fed.register_subscription(
            self.inputs["voltage_real"], ""
        )
        self.sub.voltages_imag = self.fed.register_subscription(
            self.inputs["voltage_imag"], ""
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
        self.pub_power_mag = self.fed.register_publication(
            "power_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_power_angle = self.fed.register_publication(
            "power_angle", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltage_mag = self.fed.register_publication(
            "voltage_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltage_angle = self.fed.register_publication(
            "voltage_angle", h.HELICS_DATA_TYPE_STRING, ""
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
        """Parse topology subscription, build Case and area_info."""
        topology: Topology = Topology.parse_obj(self.sub.topology.json)

        case, name_to_id, v_ln_base_map = topology_to_case(
            topology,
            source_bus=self.static.source,
        )

        self.case = case
        self.name_to_id = name_to_id
        self.v_ln_base_map = v_ln_base_map

        # Collect equipment tags per bus for command publication
        self.gen_tags = self._collect_gen_tags(topology)

        # Build area_info for ENAPP spatial decomposition
        if self.static.switches:
            self._build_area_info(topology)
        else:
            self.area_info = None
            self.boundary_buses = []

        self._initialized = True
        logger.info(
            "Area initialized: %d buses, %d branches, %d generators, %d switches",
            len(case.bus_data),
            len(case.branch_data),
            len(case.gen_data) if case.gen_data is not None else 0,
            len(self.static.switches),
        )

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

    def _build_area_info(self, topology: Topology) -> None:
        """Construct area_info dict for solve_enapp from switch boundaries.

        Each switch branch in self.static.switches defines a decomposition
        boundary.  The downstream bus becomes the SWING of a child area.
        The main area (containing the slack bus) is named "main".
        """
        slack_bus = self.static.source
        incidences = topology.incidences

        # Map switch_id → (fr_bus, to_bus) from topology incidences
        switch_areas: dict = {}
        for fr_eq, to_eq, eq_id in zip(
            incidences.from_equipment,
            incidences.to_equipment,
            incidences.ids,
        ):
            if eq_id not in self.static.switches:
                continue
            fr_bus = fr_eq.split(".", 1)[0]
            to_bus = to_eq.split(".", 1)[0]
            area_name = f"area_{to_bus}"
            switch_areas[area_name] = (fr_bus, to_bus)

        if not switch_areas:
            logger.warning(
                "Switches %s not found in topology incidences; "
                "falling back to single-area solve",
                self.static.switches,
            )
            self.area_info = None
            self.boundary_buses = []
            return

        area_info: dict = {
            "main": {
                "up_areas": [],
                "down_areas": list(switch_areas.keys()),
                "up_buses": [slack_bus],
            }
        }
        boundary_buses = []
        for area_name, (fr_bus, to_bus) in switch_areas.items():
            area_info[area_name] = {
                "up_areas": ["main"],
                "down_areas": [],
                "up_buses": [to_bus],
            }
            boundary_buses.append(to_bus)

        self.area_info = area_info
        self.boundary_buses = boundary_buses
        logger.debug(
            "ENAPP area_info: %s", {k: v["up_buses"] for k, v in area_info.items()}
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
        self.pub_voltage_mag.publish(empty_v.json())
        self.pub_voltage_angle.publish(empty_v.json())
        self.pub_power_mag.publish(empty_mag.json())
        self.pub_power_angle.publish(empty_ang.json())
        self.pub_v.publish(empty_v.json())
        self.pub_p.publish(empty_p.json())
        self.pub_q.publish(empty_q.json())

    def _publish_results(self, result, t: int) -> None:
        """Publish all output topics from a PowerFlowResult."""
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

        # Voltage magnitude
        v_mag = result_to_voltage_mag(result, self.v_ln_base_map or {}, t)
        self.pub_voltage_mag.publish(v_mag.json())

        # Voltage angle
        v_ang = result_to_voltage_angle(result, t)
        self.pub_voltage_angle.publish(v_ang.json())

        # Branch power magnitude and angle
        p_mag = result_to_power_mag(result, self.v_ln_base_map or {}, t)
        self.pub_power_mag.publish(p_mag.json())

        p_ang = result_to_power_angle(result, t)
        self.pub_power_angle.publish(p_ang.json())

        # Boundary bus publications (for inter-federate coordination)
        pub_p, pub_q, pub_v = result_to_pub_pqv(
            result,
            boundary_buses=self.boundary_buses,
            v_ln_base_map=self.v_ln_base_map or {},
            time=t,
        )
        self.pub_v.publish(pub_v.json())
        self.pub_p.publish(pub_p.json())
        self.pub_q.publish(pub_q.json())

    def run(self) -> None:
        h.helicsFederateEnterExecutingMode(self.fed)
        logger.info("Federate executing: %s", datetime.now())

        update_interval = int(self.static.deltat)
        total_time = self.static.t_steps * update_interval
        granted_time = 0
        objective_fn = self._get_objective_fn()

        while granted_time < total_time:
            request_time = granted_time + update_interval
            granted_time = h.helicsFederateRequestTime(self.fed, request_time)
            t = int(granted_time)
            logger.info("Granted time: %d", t)

            # ── Initialize on first timestep ──────────────────────────────
            if not self._initialized:
                if not self.sub.topology.is_updated():
                    logger.warning("Topology subscription not yet available at t=%d", t)
                    self._publish_empty(t)
                    continue
                self.init_area()

            # ── Read live measurements ────────────────────────────────────
            injection = self._read_injection()
            voltages_mag = self._read_voltages_mag()

            if injection is not None:
                update_case_from_measurements(
                    self.case,
                    injection,
                    self.name_to_id,
                    voltages_mag=voltages_mag,
                )

            # ── Solve OPF ─────────────────────────────────────────────────
            tic = _time.perf_counter()
            result = None
            try:
                if self.area_info is not None:
                    result = solve_enapp(
                        self.case,
                        self.area_info,
                        objective=objective_fn,
                        tol=self.static.tol,
                        max_iterations=self.static.max_iterations,
                        parallel=True,
                    )
                elif objective_fn is not None:
                    result = self.case.run_opf(objective_fn)
                else:
                    result = self.case.run_pf()
            except Exception:
                logger.exception("OPF solve failed at t=%d", t)

            elapsed = _time.perf_counter() - tic
            if result is not None:
                logger.info(
                    "t=%d  converged=%s  obj=%.4g  solve_time=%.2fs",
                    t,
                    getattr(result, "converged", "?"),
                    getattr(result, "objective_value", float("nan")) or float("nan"),
                    elapsed,
                )
                self._publish_results(result, t)
            else:
                logger.error("No result at t=%d; publishing empty messages", t)
                self._publish_empty(t)

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
