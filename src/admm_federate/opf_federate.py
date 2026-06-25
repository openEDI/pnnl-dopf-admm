import copy
import json
import logging
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import helics as h
import networkx as nx
import numpy as np
import xarray as xr
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    Command,
    CommandList,
    Injection,
    MeasurementArray,
    PowersImaginary,
    PowersReal,
    Topology,
    VoltagesAngle,
    VoltagesImaginary,
    VoltagesMagnitude,
    VoltagesReal,
)
from pydantic import BaseModel

from admm_federate import adapter, lindistflow

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def measurement_to_xarray(eq: MeasurementArray):
    return xr.DataArray(eq.values, coords={"ids": eq.ids})


def xarray_to_dict(data):
    """Convert xarray to dict with values and ids for JSON serialization."""
    coords = {key: list(data.coords[key].data) for key in data.coords.keys()}
    return {"values": list(data.data), **coords}


def xarray_to_voltages_pol(data, **kwargs):
    """Conveniently turn xarray into VoltagesMagnitude and VoltagesAngle."""
    mag = VoltagesMagnitude(**xarray_to_dict(np.abs(data)), **kwargs)
    ang = VoltagesAngle(**xarray_to_dict(np.arctan2(data.imag, data.real)), **kwargs)
    return mag, ang


class ComponentParameters(BaseModel):
    name: str
    t_steps: int
    max_itr: int
    control_type: str
    switches: list[str]
    source_bus: str
    source_line: str
    relaxed: bool
    rho_sup: float
    rho_vup: float
    rho_sdn: float
    rho_vdn: float
    vup_tol: float
    sdn_tol: float


class Subscriptions:
    voltages_real: VoltagesReal
    voltages_imag: VoltagesImaginary
    injections: Injection
    topology: Topology
    area_v: VoltagesMagnitude
    area_p: PowersReal
    area_q: PowersImaginary


class OPFFederate:
    converged: bool = False
    parent_bus: str = ""
    parent_line: str = ""
    child_buses: [str] = []
    shared_buses: [str] = []
    switch_buses: [str] = []
    shared_lines: {str, str} = {}
    area_graph: nx.Graph = None
    area_branch: adapter.BranchInfo = None
    area_bus: adapter.BusInfo = None
    parent_info: adapter.BusInfo = None
    child_info: adapter.BusInfo = None
    area_v: VoltagesMagnitude
    area_p: PowersReal
    area_q: PowersImaginary

    def __init__(self, broker_config) -> None:
        self.sub = Subscriptions()
        self.load_static_inputs()
        self.load_input_mapping()
        self.initilize(broker_config)
        self.load_component_definition()
        self.register_subscription()
        self.register_publication()

    def load_component_definition(self) -> None:
        path = Path("component_definition.json")
        with open(path, encoding="UTF-8") as file:
            self.component_config = json.load(file)

    def load_input_mapping(self):
        path = Path("input_mapping.json")
        with open(path, encoding="UTF-8") as file:
            self.inputs = json.load(file)

    def load_static_inputs(self):
        path = Path("static_inputs.json")
        with open(path, encoding="UTF-8") as file:
            config = json.load(file)

        self.static = ComponentParameters.model_validate(config)
        self.deltat = config["deltat"]

        self.admm_config = lindistflow.ADMMConfig()
        self.admm_config.relaxed = self.static.relaxed
        self.admm_config.rho_vup = self.static.rho_vup
        self.admm_config.rho_sup = self.static.rho_sup
        self.admm_config.rho_vdn = self.static.rho_vdn
        self.admm_config.rho_sdn = self.static.rho_sdn

    def initilize(self, broker_config) -> None:
        self.info = h.helicsCreateFederateInfo()
        self.info.core_name = self.static.name
        self.info.core_type = h.HELICS_CORE_TYPE_ZMQ
        self.info.core_init = "--federates=1"

        # h.helicsFederateInfoSetTimeProperty(self.info, h.helics_property_time_delta, self.deltat)

        h.helicsFederateInfoSetBroker(self.info, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(self.info, broker_config.broker_port)

        self.fed = h.helicsCreateValueFederate(self.static.name, self.info)
        # h.helicsFederateSetFlagOption(self.fed, h.helics_flag_slow_responding, True)
        h.helicsFederateSetTimeProperty(
            self.fed, h.HELICS_PROPERTY_TIME_PERIOD, 1
        )

        # h.helicsFederateSetTimeProperty(self.fed, h.HELICS_PROPERTY_TIME_OFFSET, 0.1)
        # h.helicsFederateSetFlagOption(self.fed, h.HELICS_FLAG_UNINTERRUPTIBLE, True)

    def register_subscription(self) -> None:
        self.sub.topology = self.fed.register_subscription(self.inputs["topology"], "")
        self.sub.topology.option[h.HELICS_HANDLE_OPTION_IGNORE_INTERRUPTS] = True
        self.sub.injections = self.fed.register_subscription(
            self.inputs["injections"], ""
        )
        self.sub.injections.option[h.HELICS_HANDLE_OPTION_IGNORE_INTERRUPTS] = True
        self.sub.voltages_imag = self.fed.register_subscription(
            self.inputs["voltages_imag"], ""
        )
        self.sub.voltages_imag.option[h.HELICS_HANDLE_OPTION_IGNORE_INTERRUPTS] = True
        self.sub.voltages_real = self.fed.register_subscription(
            self.inputs["voltages_real"], ""
        )
        self.sub.voltages_real.option[h.HELICS_HANDLE_OPTION_IGNORE_INTERRUPTS] = True
        self.sub.area_v = self.fed.register_subscription(self.inputs["sub_v"], "")
        self.sub.area_p = self.fed.register_subscription(self.inputs["sub_p"], "")
        self.sub.area_q = self.fed.register_subscription(self.inputs["sub_q"], "")

    def register_publication(self) -> None:
        self.pub_pv_set = self.fed.register_publication(
            "pub_c", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_solver_stats = self.fed.register_publication(
            "solver_stats", h.HELICS_DATA_TYPE_STRING, ""
        )
        #        self.pub_powers_mag = self.fed.register_publication(
        #            "power_mag", h.HELICS_DATA_TYPE_STRING, ""
        #        )
        #        self.pub_powers_angle = self.fed.register_publication(
        #            "power_angle", h.HELICS_DATA_TYPE_STRING, ""
        #        )
        #        self.pub_voltages_mag = self.fed.register_publication(
        #            "voltage_mag", h.HELICS_DATA_TYPE_STRING, ""
        #        )
        #        self.pub_voltages_angle = self.fed.register_publication(
        #            "voltage_angle", h.HELICS_DATA_TYPE_STRING, ""
        #        )
        self.pub_admm_v = self.fed.register_publication(
            "pub_v", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_admm_p = self.fed.register_publication(
            "pub_p", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_admm_q = self.fed.register_publication(
            "pub_q", h.HELICS_DATA_TYPE_STRING, ""
        )

    def bus_to_branch_power(self, buses: dict) -> dict:
        branches = {}
        for k, v in buses.items():
            bus, phase = k.split(".", 1)
            if bus in self.shared_lines.keys():
                name = f"{self.shared_lines[bus]}.{phase}"
                branches[name] = v
        return branches

    def init_area(self):
        topology: Topology = Topology.model_validate(self.sub.topology.json)
        branch_info, bus_info, slack_bus = adapter.extract_info(topology)
        self.admm_config.slack = slack_bus

        G = adapter.generate_graph(topology.incidences, slack_bus)
        graph = copy.deepcopy(G)
        graph2 = copy.deepcopy(G)
        boundaries = adapter.area_disconnects(graph)

        if self.static.source_bus == slack_bus:
            self.parent_bus = slack_bus
            self.shared_buses.append(slack_bus)

        boundary = []
        for u, v, a in boundaries:
            if (
                a["id"] in self.static.switches
                and not a["id"] == self.static.source_line
            ):
                boundary.append((u, v, a))
                self.child_buses.append(v)
                self.shared_buses.append(v)
                self.switch_buses.append(u)
                self.switch_buses.append(v)
                self.shared_lines[u] = f"{u}_{v}"
                self.shared_lines[v] = f"{v}_{u}"

            if a["id"] == self.static.source_line:
                boundary.append((u, v, a))
                self.parent_bus = u
                self.shared_buses.append(u)
                self.switch_buses.append(u)
                self.switch_buses.append(v)
                self.parent_line = f"{u}_{v}"
                self.shared_lines[u] = f"{u}_{v}"
                self.shared_lines[v] = f"{v}_{u}"

        areas = adapter.disconnect_areas(graph2, boundary)
        areas = adapter.reconnect_area_switches(areas, boundary)

        ids = [a["id"] for _, _, a in boundary]
        for area in areas:
            area_branch, area_bus = adapter.generate_area_info(
                area, topology, self.parent_bus, ids
            )
            if area_branch is not None and area_bus is not None:
                for u, v in area.edges(self.parent_bus):
                    if self.parent_line == "":
                        self.parent_line = f"{u}_{v}"
                self.area_branch = area_branch
                self.area_bus = area_bus
                self.area_graph = area

        self.child_info = adapter.BusInfo()
        for k in self.child_buses:
            if k in self.area_bus.buses:
                self.child_info.buses[k] = copy.deepcopy(self.area_bus.buses[k])
                self.child_info.buses[k].pv = np.zeros((3, 2)).tolist()

        self.parent_info = adapter.BusInfo()
        self.parent_info.buses[self.parent_bus] = copy.deepcopy(
            self.area_bus.buses[self.parent_bus]
        )
        self.parent_info.buses[self.parent_bus].pv = np.zeros((3, 2)).tolist()

        logger.debug("Parent Info")
        logger.debug(f"\tbus = {self.parent_bus}")
        logger.debug(f"\tline = {self.parent_line}")
        logger.debug("Child Info")
        logger.debug(f"\tbuses = {self.child_buses}")
        logger.debug("Shared Info")
        logger.debug(f"\tbuses = {self.shared_buses}")

        # HELICS Synchronization Deadlock Prevention Mechanism:
        #
        # In a co-simulation involving an iterative federate (like this ADMM solver/Hubs)
        # and a non-iterative federate (like the Feeder), a circular startup deadlock can occur.
        # This happens because the iterative federates must iterate at the start time 0.0,
        # requesting iterative times from HELICS, while the non-iterative Feeder expects to
        # progress time linearly. If the iterative federate blocks waiting for initial
        # measurements (voltages, injections) from the Feeder, and the Feeder is blocked
        # waiting for the first step to progress or to receive control commands, neither can proceed.
        #
        # To resolve this circular dependency, we query nominal/base voltages and nominal
        # injections from the static topology and establish them as default values on their
        # respective subscriptions using `set_default()`.
        # This allows HELICS to immediately return these nominal values to the ADMM area on startup
        # without waiting for the Feeder to publish them, thus breaking the circular deadlock.
        # We use a timezone-naive fallback timestamp (e.g. datetime.now() if not present) for compatibility.
        ids = topology.base_voltage_magnitudes.ids
        mags = topology.base_voltage_magnitudes.values
        angles_map = dict(zip(topology.base_voltage_angles.ids, topology.base_voltage_angles.values))
        
        real_vals = []
        imag_vals = []
        for i, node_id in enumerate(ids):
            mag = mags[i]
            ang = angles_map.get(node_id, 0.0)
            v = mag * np.exp(1j * ang)
            real_vals.append(v.real)
            imag_vals.append(v.imag)
            
        time_val = topology.injections.power_real.time or datetime.now()
        default_v_real = VoltagesReal(ids=ids, values=real_vals, time=time_val)
        default_v_imag = VoltagesImaginary(ids=ids, values=imag_vals, time=time_val)
        
        self.sub.voltages_real.set_default(default_v_real.model_dump_json())
        self.sub.voltages_imag.set_default(default_v_imag.model_dump_json())
        self.sub.injections.set_default(topology.injections.model_dump_json())

        # Extract nominal PV capacities from topology injections
        # Used later to translate active power kW commands to %Pmpp setpoints
        self.pv_capacities = {}
        for val, eq_id in zip(topology.injections.power_real.values, topology.injections.power_real.equipment_ids):
            if eq_id.lower().startswith("pvsystem."):
                self.pv_capacities[eq_id] = self.pv_capacities.get(eq_id, 0.0) + float(val)

    def get_set_points(self, control: dict, bus_info: adapter.BusInfo) -> dict[complex]:
        setpoints = {}
        for key, val in control.items():
            if key in bus_info.buses:
                bus = bus_info.buses[key]
                for tag in set(bus.tags):
                    if "PVSystem" in tag:
                        p = max([p for p in val["Pdg_gen"].values()])
                        q = max([q for q in val["Qdg_gen"].values()])
                        setpoints[tag] = p + 1j * q
        return setpoints

    def first_pub(self, t):
        self.area_v = VoltagesMagnitude(ids=[], values=[], time=t)
        self.area_p = PowersReal(ids=[], equipment_ids=[], values=[], time=t)
        self.area_q = PowersImaginary(ids=[], equipment_ids=[], values=[], time=t)

        self.pub_admm_v.publish(self.area_v.json())
        self.pub_admm_p.publish(self.area_p.json())
        self.pub_admm_q.publish(self.area_q.json())

    def itr_pub(self):
        if self.area_graph is None:
            self.init_area()

        voltages_real = VoltagesReal.model_validate(self.sub.voltages_real.json)
        voltages_imag = VoltagesImaginary.model_validate(self.sub.voltages_imag.json)
        voltages = measurement_to_xarray(voltages_real) + 1j * measurement_to_xarray(
            voltages_imag
        )

        voltages_mag, voltages_ang = xarray_to_voltages_pol(voltages)
        t = voltages_real.time
        voltages_mag.time = t
        voltages_ang.time = t

        injections = Injection.model_validate(self.sub.injections.json)
        bus_info = adapter.extract_injection(copy.deepcopy(self.area_bus), injections)

        bus_info = adapter.extract_voltages(bus_info, voltages_mag)

        branch_info, bus_info = adapter.map_secondaries(
            copy.deepcopy(self.area_branch), bus_info
        )

        with open("bus_info.json", "w") as outfile:
            outfile.write(json.dumps(asdict(bus_info)))

        with open("branch_info.json", "w") as outfile:
            outfile.write(json.dumps(asdict(branch_info)))

        p_err = 0
        q_err = 0
        v_err = 0
        p = PowersReal.model_validate(self.sub.area_p.json)
        if p.values and self.area_p.values:
            logger.debug("Updating Area Active Power")
            p = adapter.filter_boundary_power_real(self.shared_buses, p)
            p, p_err = adapter.update_boundary_power_real(p, self.static.name)
            self.parent_info = adapter.extract_powers_real(self.parent_info, p, True)
            self.child_info = adapter.extract_powers_real(self.child_info, p, True)

        q = PowersImaginary.model_validate(self.sub.area_q.json)
        if q.values and self.area_q.values:
            logger.debug("Updating Area Reactive Power")
            q = adapter.filter_boundary_power_imag(self.shared_buses, q)
            q, q_err = adapter.update_boundary_power_imag(q, self.static.name)
            self.parent_info = adapter.extract_powers_imag(self.parent_info, q, True)
            self.child_info = adapter.extract_powers_imag(self.child_info, q, True)

        vmag = VoltagesMagnitude.model_validate(self.sub.area_v.json)
        if vmag.values and self.area_v.values:
            logger.debug("Updating Area Voltages")
            vmag = adapter.filter_boundary_voltage(self.switch_buses, vmag)
            vmag, v_err = adapter.update_boundary_voltage(self.area_v, vmag)
            self.parent_info = adapter.extract_voltages(self.parent_info, vmag)
            self.child_info = adapter.extract_voltages(self.child_info, vmag)

        with open("bus_info_updated.json", "w") as outfile:
            outfile.write(json.dumps(asdict(bus_info)))

        with open("branch_info_updated.json", "w") as outfile:
            outfile.write(json.dumps(asdict(branch_info)))

        if not adapter.check_radiality(branch_info, bus_info):
            logger.warning("Network radiality constraint violated on current topology!")

        self.admm_config.source_bus = self.parent_bus
        self.admm_config.source_line = adapter.get_edge_name(
            self.area_graph, self.parent_bus
        )
        self.admm_config.relaxed = self.static.relaxed
        v_mag, branch_pq, aux_pq, control, stats = lindistflow.solve(
            branch_info, bus_info, self.child_info, self.parent_info, self.admm_config
        )
        real_setpts = self.get_set_points(control, bus_info)

        bp = {k: p[0] for k, p in branch_pq.items()}
        bq = {k: p[1] for k, p in branch_pq.items()}

        p = {k: p[0] for k, p in aux_pq.items()}
        p = self.bus_to_branch_power(p)

        q = {k: p[1] for k, p in aux_pq.items()}
        q = self.bus_to_branch_power(q)

        # replace branch flows with aux loads if they exist
        for k, v in bp.items():
            if k in p:
                bp[k] = p[k]
                bq[k] = q[k]

        # CAPTURE STATS FOR PUB
        power_real = PowersReal(
            ids=list(bp.keys()),
            values=list(bp.values()),
            equipment_ids=list(bp.keys()),
            time=t,
        )
        self.area_p = copy.deepcopy(
            adapter.filter_line_power_real(
                self.shared_buses, power_real, self.static.name
            )
        )

        power_imag = PowersImaginary(
            ids=list(bq.keys()),
            values=list(bq.values()),
            equipment_ids=list(bq.keys()),
            time=t,
        )
        self.area_q = copy.deepcopy(
            adapter.filter_line_power_imag(
                self.shared_buses, power_imag, self.static.name
            )
        )

        vmag = adapter.pack_voltages(v_mag, bus_info, t)
        self.area_v = copy.deepcopy(
            adapter.filter_boundary_voltage(self.switch_buses, vmag)
        )

        self.pub_admm_p.publish(self.area_p.json())
        self.pub_admm_q.publish(self.area_q.json())
        self.pub_admm_v.publish(self.area_v.json())

        # SET COMMANDS FOR PUB
        #
        # PVSystem Command Translation Logic:
        # OpenDSS PVSystem objects receive active power commands in terms of percentage
        # of their nominal capacity (%Pmpp), rather than in kW. If we send kW directly,
        # OpenDSS will not command them correctly. Thus, for any device prefix starting with
        # "pvsystem.", we scale the desired real power setpoint by the nominal capacity
        # extracted from the topology injections to obtain %Pmpp. We also send %Cutout=0 and
        # %Cutin=0 commands to ensure the PV system remains connected.
        #
        # Other device types (like Storage):
        # We pass through kW properties directly.
        #
        # Reactive power (kvar) properties for all devices:
        # Passed through directly.
        commands = []
        for eq, val in real_setpts.items():
            if abs(val) < 1e-6:
                continue

            if eq.lower().startswith("pvsystem."):
                max_pv = self.pv_capacities.get(eq, 50.0)
                p = val.real
                if max_pv <= 0:
                    obj_val = 100.0
                elif p == 0:
                    obj_val = 0.0
                elif p < max_pv:
                    obj_val = p / float(max_pv) * 100.0
                else:
                    obj_val = 100.0
                
                commands.append(Command(obj_name=eq, obj_property="%Pmpp", val=str(obj_val)))
                commands.append(Command(obj_name=eq, obj_property="%Cutout", val="0"))
                commands.append(Command(obj_name=eq, obj_property="%Cutin", val="0"))
            else:
                commands.append(Command(obj_name=eq, obj_property="kW", val=str(val.real)))

            commands.append(
                Command(obj_name=eq, obj_property="kvar", val=str(val.imag))
            )
        cmd_list = CommandList(root=commands)
        self.pub_pv_set.publish(cmd_list.model_dump_json())

        # CAPTURE STATS FOR PUB
        stats["admm_iteration"] = self.itr
        stats["vup"] = v_err
        stats["sdn"] = math.sqrt(p_err**2 + q_err**2)
        logger.debug(f"Errors : {stats['vup']}, {stats['sdn']}")

        if self.itr > 1:
            v_settled = stats["vup"] <= self.static.vup_tol
            p_settled = stats["sdn"] <= self.static.sdn_tol
            if v_settled and p_settled:
                logger.debug("Converged")
                self.converged = True

        solver_stats = MeasurementArray(
            ids=list(stats.keys()),
            values=list(stats.values()),
            time=t,
            units="s",
        )
        self.pub_solver_stats.publish(solver_stats.json())

        # self.pub_voltages_mag.publish(v_mag.json())
        # self.pub_voltages_angle.publish(voltages_ang.json())
        # self.pub_powers_mag.publish(power_mag.json())
        # self.pub_powers_angle.publish(power_ang.json())

    def run(self) -> None:
        try:
            logger.info(f"Federate connected: {datetime.now()}")
            itr_need = h.helics_iteration_request_iterate_if_needed
            itr_stop = h.helics_iteration_request_no_iteration
            h.helicsFederateEnterExecutingMode(self.fed)
            logger.info(f"Federate executing: {datetime.now()}")

            # setting up time properties
            update_interval = int(
                h.helicsFederateGetTimeProperty(self.fed, h.HELICS_PROPERTY_TIME_PERIOD)
            )

            granted_time = 0
            logger.debug("Step 0: Starting Time/Iter loop")
            while granted_time < self.static.t_steps * self.deltat:
                request_time = granted_time + update_interval
                logger.debug("Step 1: published initial values for iteration")
                itr_flag = itr_need
                self.first_pub(granted_time)
                self.converged = False
                self.itr = 0
                while True:
                    logger.debug(f"Step 2: Requesting time {request_time}")
                    granted_time, itr_status = h.helicsFederateRequestTimeIterative(
                        self.fed, request_time, itr_flag
                    )
                    logger.info(f"\tgranted time = {granted_time}")
                    logger.info(f"\titr status = {itr_status}")

                    if itr_status == h.helics_iteration_result_next_step:
                        logger.debug(f"\titr next: {self.itr}")
                        break

                    self.itr += 1
                    logger.debug("Step 4: update iteration")
                    logger.info(f"\titr: {self.itr}")

                    logger.debug("Step 6: run solution")
                    self.itr_pub()

                    if self.converged or self.itr >= self.static.max_itr:
                        itr_flag = itr_stop
                    else:
                        itr_flag = itr_need

            logger.debug("FINISHED")
        finally:
            self.stop()

    def stop(self) -> None:
        h.helicsFederateDisconnect(self.fed)
        h.helicsFederateFree(self.fed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


def run_simulator(broker_config: BrokerConfig) -> None:
    schema = json.dumps(ComponentParameters.model_json_schema(), indent=2)
    with open("./schema.json", "w") as f:
        f.write(schema)

    sfed = OPFFederate(broker_config)
    sfed.run()


def main() -> None:
    broker_config = BrokerConfig(
        broker_ip="127.0.0.1",
        broker_port=23404,
    )
    run_simulator(broker_config)


if __name__ == "__main__":
    main()
