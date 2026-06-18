import copy
import json
import os

from oedisi.componentframework.system_configuration import (
    Component,
    Link,
    WiringDiagram,
)
from oedisi.types.data_types import Topology

from admm_federate.adapter import (
    area_disconnects,
    disconnect_areas,
    generate_graph,
    get_area_source,
    reconnect_area_switches,
)

ROOT = os.getcwd()
ALGO = "pnnl_dopf_admm"
NAME = ""
OUTPUTS = ""
SCENARIOS = ""


T_STEPS = 1
DELTA_T = 60 * 60  # minutes * seconds per hour


def generate_feeder(OUTPUTS: str) -> Component:
    smart_ds = False
    base = "gadal_ieee123"
    profiles = f"{base}/profiles"
    opendss = f"{base}/qsts"
    file = "opendss/master.dss"

    return Component(
        name="feeder",
        type="Feeder",
        host="feeder",
        container_port=5600,
        parameters={
            "use_smartds": smart_ds,
            "use_sparse_admittance": True,
            "profile_location": profiles,
            "opendss_location": opendss,
            "feeder_file": file,
            "start_date": "2018-05-01 00:00:00",
            "number_of_timesteps": T_STEPS,
            "run_freq_sec": DELTA_T,
            "topology_output": f"{OUTPUTS}/topology.json",
            "buscoords_output": f"{OUTPUTS}/Buscoords.dat",
        },
    )


def generate_recorder(port: str, src: str, OUTPUTS: str) -> tuple[Component, Link]:
    name = f"recorder_{port}_{src}"
    file = f"{port}_{src}"
    if src == "feeder":
        name = f"recorder_{port}"
        file = port

    component = Component(
        name=name,
        type="Recorder",
        host="recorder",
        container_port=None,
        parameters={
            "feather_filename": f"{OUTPUTS}/{file}.feather",
            "csv_filename": f"{OUTPUTS}/{file}.csv",
            "number_of_timesteps": T_STEPS,
            "deltat": DELTA_T,
        },
    )

    link = Link(
        source=src, source_port=port, target=component.name, target_port="subscription"
    )
    return (component, link)


def generate_sensor(port: str, src: str) -> tuple[Component, Link]:
    """Generates a Sensor component and its Link. Note: Currently unused but kept for potential future use."""
    if "power_real" in port:
        file = "sensors/real_ids.json"
    elif "power_imag" in port:
        file = "sensors/reactive_ids.json"
    elif "voltage" in port:
        file = "sensors/voltage_ids.json"
    else:
        print("Need sensor file")
        exit(1)

    component = Component(
        name=f"sensor_{port}",
        type="Sensor",
        host="sensor",
        container_port=None,
        parameters={
            "additive_noise_stddev": 0.01,
            "multiplicative_noise_stddev": 0.001,
            "measurement_file": f"../{src}/{file}",
            "number_of_timesteps": T_STEPS,
            "deltat": DELTA_T,
        },
    )

    link = Link(
        source=src, source_port=port, target=component.name, target_port="subscription"
    )
    return (component, link)


def link_feeder(system: WiringDiagram, feeder: Component) -> None:
    port = "voltage_real"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "voltage_imag"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "power_real"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "power_imag"
    component, link = generate_recorder(port, feeder.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)


def link_feeder_voltage(system: WiringDiagram, feeder: Component, src: int) -> None:
    system.links.append(
        Link(
            source=f"{feeder.name}",
            source_port="voltage_real",
            target=f"{ALGO}_{src}",
            target_port="sub_v",
        )
    )


def link_feeder_power(system: WiringDiagram, feeder: Component, src: int) -> None:
    system.links.append(
        Link(
            source=f"{feeder.name}",
            source_port="power_real",
            target=f"{ALGO}_{src}",
            target_port="sub_p",
        )
    )
    system.links.append(
        Link(
            source=f"{feeder.name}",
            source_port="power_imag",
            target=f"{ALGO}_{src}",
            target_port="sub_q",
        )
    )


def link_hub_voltage(system: WiringDiagram, hub: Component, src: int) -> None:
    system.links.append(
        Link(
            source=f"{ALGO}_{src}",
            source_port="pub_v",
            target=f"{hub.name}",
            target_port=f"sub_v{src}",
        )
    )
    system.links.append(
        Link(
            source=f"{hub.name}",
            source_port=f"pub_v{src}",
            target=f"{ALGO}_{src}",
            target_port="sub_v",
        )
    )


def link_hub_power(system: WiringDiagram, hub: Component, src: int) -> None:
    system.links.append(
        Link(
            source=f"{ALGO}_{src}",
            source_port="pub_p",
            target=f"{hub.name}",
            target_port=f"sub_p{src}",
        )
    )
    system.links.append(
        Link(
            source=f"{hub.name}",
            source_port=f"pub_p{src}",
            target=f"{ALGO}_{src}",
            target_port="sub_p",
        )
    )
    system.links.append(
        Link(
            source=f"{ALGO}_{src}",
            source_port="pub_q",
            target=f"{hub.name}",
            target_port=f"sub_q{src}",
        )
    )
    system.links.append(
        Link(
            source=f"{hub.name}",
            source_port=f"pub_q{src}",
            target=f"{ALGO}_{src}",
            target_port="sub_q",
        )
    )


def link_hub_control(system: WiringDiagram, hub: Component, src: int) -> None:
    system.links.append(
        Link(
            source=f"{ALGO}_{src}",
            source_port="pub_c",
            target=f"{hub.name}",
            target_port=f"sub_c{src}",
        )
    )


def link_algo(system: WiringDiagram, algo: Component, feeder: Component) -> None:
    port = "voltage_real"
    system.links.append(
        Link(source=feeder.name, source_port=port, target=algo.name, target_port=port)
    )

    port = "voltage_imag"
    system.links.append(
        Link(source=feeder.name, source_port=port, target=algo.name, target_port=port)
    )

    port = "power_real"
    system.links.append(
        Link(source=feeder.name, source_port=port, target=algo.name, target_port=port)
    )

    port = "power_imag"
    system.links.append(
        Link(source=feeder.name, source_port=port, target=algo.name, target_port=port)
    )

    #    port = "voltage_mag"
    #    component, link = generate_recorder(port, algo.name, OUTPUTS)
    #    system.components.append(component)
    #    system.links.append(link)
    #
    #    port = "voltage_angle"
    #    component, link = generate_recorder(port, algo.name, OUTPUTS)
    #    system.components.append(component)
    #    system.links.append(link)
    #
    #    port = "power_mag"
    #    component, link = generate_recorder(port, algo.name, OUTPUTS)
    #    system.components.append(component)
    #    system.links.append(link)
    #
    #    port = "power_angle"
    #    component, link = generate_recorder(port, algo.name, OUTPUTS)
    #    system.components.append(component)
    #    system.links.append(link)

    port = "solver_stats"
    component, link = generate_recorder(port, algo.name, OUTPUTS)
    system.components.append(component)
    system.links.append(link)

    port = "injections"
    system.links.append(
        Link(source=feeder.name, source_port=port, target=algo.name, target_port=port)
    )

    port = "topology"
    system.links.append(
        Link(source=feeder.name, source_port=port, target=algo.name, target_port=port)
    )


def generate() -> None:
    global OUTPUTS
    TOPOLOGY = f"{ROOT}/scenarios/ieee123/topology.json"
    OUTPUTS = "../../outputs"
    SCENARIOS = f"{ROOT}/scenarios"
    os.makedirs(SCENARIOS, exist_ok=True)

    topology = get_topology(TOPOLOGY)
    slack_bus, _ = topology.slack_bus[0].split(".", 1)

    if topology.incidences is None:
        print("topology must have incidences")
        exit(1)

    G = generate_graph(topology.incidences, slack_bus)
    print("Total: ", G)
    graph = copy.deepcopy(G)
    graph2 = copy.deepcopy(G)
    boundaries = area_disconnects(graph)
    areas = disconnect_areas(graph2, boundaries)
    areas = reconnect_area_switches(areas, boundaries)

    system = WiringDiagram(name=f"{ALGO}_ieee123", components=[], links=[])
    feeder = generate_feeder(OUTPUTS)
    system.components.append(feeder)

    link_feeder(system, feeder)

    switch_map = {}
    source_bus_map = {}
    source_line_map = {}
    for i, area in enumerate(areas):
        src = []
        switches = []
        for u, v, a in boundaries:
            if area.has_edge(u, v):
                switches.append(a["id"])
                src.append((u, v, a))

        if area.has_node(slack_bus):
            source_bus_map[i] = slack_bus
            source_line_map[i] = ""
        else:

            su, sv, sa = get_area_source(G, slack_bus, src)
            source_bus_map[i] = su
            source_line_map[i] = sa["id"]

        switch_map[i] = switches

    sub_areas = {}
    for area, switches in switch_map.items():
        area_set = set()
        for a, s in switch_map.items():
            if any(switch in s for switch in switches):
                if a != area:
                    area_set.add(a)
        sub_areas[area] = area_set

    max_itr = 10
    hub_voltage = Component(
        name="hub_voltage",
        type="VoltageHub",
        host="hub_voltage",
        container_port=None,
        parameters={
            "name": "hub_voltage",
            "max_itr": max_itr,
            "t_steps": T_STEPS,
        },
    )
    system.components.append(hub_voltage)

    hub_power = Component(
        name="hub_power",
        type="PowerHub",
        host="hub_voltage",
        container_port=None,
        parameters={
            "max_itr": max_itr,
            "number_of_timesteps": T_STEPS,
            "deltat": DELTA_T,
        },
    )
    system.components.append(hub_power)

    hub_control = Component(
        name="hub_control",
        type="ControlHub",
        host="hub_control",
        container_port=None,
        parameters={
            "max_itr": max_itr,
            "number_of_timesteps": T_STEPS,
            "deltat": DELTA_T,
        },
    )
    system.components.append(hub_control)

    port = "pv_set"
    system.links.append(
        Link(
            source=hub_control.name,
            source_port=port,
            target=feeder.name,
            target_port=port,
        )
    )

    for k, v in sub_areas.items():
        print(k, v)
        link_feeder_voltage(system, feeder, k)
        link_feeder_power(system, feeder, k)
        link_hub_control(system, hub_control, k)
        link_hub_power(system, hub_power, k)
        link_hub_voltage(system, hub_voltage, k)

    rho_vup = [1e3] * len(sub_areas)
    rho_sdn = [1e3] * len(sub_areas)
    for k, v in sub_areas.items():
        algo = Component(
            name=f"{ALGO}_{k}",
            type="OptimalPowerFlow",
            host=f"admm_{k}",
            container_port=None,
            parameters={
                "vup_tol": 0.01,
                "sdn_tol": 0.01,
                "max_itr": max_itr,
                "t_steps": T_STEPS,
                "deltat": DELTA_T,
                "relaxed": False,
                "control_type": "real",
                "switches": switch_map[k],
                "source_bus": source_bus_map[k],
                "source_line": source_line_map[k],
                "rho_vup": rho_vup[k],
                "rho_sup": 0,
                "rho_vdn": 0,
                "rho_sdn": rho_sdn[k],
            },
        )
        system.components.append(algo)
        link_algo(system, algo, feeder)

    with open(f"{SCENARIOS}/{system.name}.json", "w") as f:
        f.write(system.json())

    check = WiringDiagram.parse_file(f"{SCENARIOS}/{system.name}.json")

    components = {}
    for c in system.components:
        name = c.name
        print("Linking Component: ", name)
        if "hub" in name:
            components[c.type] = f"{name}/component_definition.json"
        elif "_" in name:
            base_name, _ = name.split("_", 1)
            components[c.type] = f"{base_name}_federate/component_definition.json"
        else:
            components[c.type] = f"{name}_federate/component_definition.json"

    with open(f"{SCENARIOS}/components.json", "w") as f:
        f.write(json.dumps(components))


def get_topology(path: str) -> Topology:
    assert os.path.exists(path), "need to generate topology from base scenario"

    with open(path) as f:
        topology = Topology.parse_obj(json.load(f))

    return topology


if __name__ == "__main__":
    print("generating: IEEE 123 ADMM")
    generate()
