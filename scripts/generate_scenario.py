import copy
import csv
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
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

SMART_DS = {
    "SFO/P1U": "p1uhs0_1247/p1uhs0_1247--p1udt942",
}

T_STEPS = 1
DELTA_T = 60 * 60  # minutes * seconds per hour


def parse_model_dir(model_dir: str) -> tuple[str, str]:
    parts = model_dir.split("-")
    if len(parts) >= 2:
        level = parts[-1]
        model_part = "-".join(parts[:-1])  # "sfo-p1u"
        model_parts = model_part.split("-")
        model = "/".join([mp.upper() for mp in model_parts])  # "SFO/P1U"
        return model, level
    return "", ""


def load_coordinates(coords_dir: str) -> dict[str, tuple[float, float]]:
    for filename in ["Buscoords.dat", "Buscoords.dss"]:
        path = os.path.join(coords_dir, filename)
        if os.path.exists(path):
            coords = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("//") or line.startswith("!"):
                        continue
                    if "," in line:
                        parts = [p.strip() for p in line.split(",")]
                    else:
                        parts = line.split()
                    if len(parts) >= 3:
                        bus = parts[0].strip("'\"")
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            coords[bus] = (x, y)
                        except ValueError:
                            pass
            if coords:
                return coords
    return {}


def plot_network(
    G: nx.Graph,
    boundaries: list,
    areas_clean: list[nx.Graph],
    slack_bus: str,
    coords_dir: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(12, 10))

    coords = load_coordinates(coords_dir)
    if coords:
        coords_upper = {k.upper(): v for k, v in coords.items()}
        pos = {
            node: coords_upper[node.upper()]
            for node in G.nodes()
            if node.upper() in coords_upper
        }
        missing_nodes = [n for n in G.nodes() if n not in pos]
        if missing_nodes:
            if len(pos) > 0:
                temp_pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), seed=42)
                pos.update({n: temp_pos[n] for n in missing_nodes})
            else:
                pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)

    node_to_area = {}
    for idx, area in enumerate(areas_clean):
        for node in area.nodes():
            node_to_area[node] = idx

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    node_colors = [colors[node_to_area[node] % len(colors)] for node in G.nodes()]

    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=1.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)

    y_vals = [p[1] for p in pos.values()]
    y_range = max(y_vals) - min(y_vals) if y_vals else 1.0
    offset_y = y_range * 0.02

    for u, v, a in boundaries:
        if u in pos and v in pos:
            mid_x = (pos[u][0] + pos[v][0]) / 2.0
            mid_y = (pos[u][1] + pos[v][1]) / 2.0
            plt.plot(
                mid_x,
                mid_y,
                marker="s",
                color="red",
                markersize=8,
                markeredgecolor="black",
                zorder=5,
            )
            plt.text(
                mid_x,
                mid_y + offset_y,
                a["id"],
                color="darkred",
                fontsize=8,
                weight="bold",
                ha="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

    if slack_bus in G.nodes():
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[slack_bus],
            node_shape="*",
            node_color="gold",
            node_size=200,
            edgecolors="black",
        )

    legend_elements = []
    for idx, area in enumerate(areas_clean):
        color = colors[idx % len(colors)]
        num_nodes = area.number_of_nodes()
        legend_elements.append(
            mpatches.Patch(color=color, label=f"Area {idx} ({num_nodes} nodes)")
        )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=8,
            label="Boundary Switch Location",
        )
    )
    if slack_bus in G.nodes():
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="gold",
                markeredgecolor="black",
                markersize=12,
                label="Slack Bus",
            )
        )

    plt.legend(handles=legend_elements, loc="best", fontsize=9, framealpha=0.9)
    model_name = (
        os.path.basename(output_path).replace(f"{ALGO}_", "").replace(".png", "")
    )
    plt.title(
        f"{model_name.upper()} Distribution Grid ADMM Area Partition",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("off")

    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close()


def generate_feeder_ieee(OUTPUTS: str) -> Component:
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


def generate_feeder_smartds(MODEL: str, LEVEL: str, OUTPUTS: str) -> Component:
    smart_ds = True
    base = f"SMART-DS/v1.0/2018/{MODEL}"
    scenario = f"scenarios/solar_{LEVEL}_batteries_none_timeseries"
    profiles = f"{base}/profiles"
    opendss = f"{base}/{scenario}/opendss/{SMART_DS[MODEL]}"
    file = "opendss/Master.dss"

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


def generate_feeder(MODEL: str, LEVEL: str, OUTPUTS: str) -> Component:
    if "ieee" in MODEL.lower():
        return generate_feeder_ieee(OUTPUTS)
    else:
        return generate_feeder_smartds(MODEL, LEVEL, OUTPUTS)


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


def generate_for_model(model_dir: str, topology_path: str, SCENARIOS: str) -> None:
    topology = get_topology(topology_path)
    slack_bus, _ = topology.slack_bus[0].split(".", 1)

    if topology.incidences is None:
        print(f"topology in {model_dir} must have incidences")
        return

    G = generate_graph(topology.incidences, slack_bus)
    print(f"Total graph for {model_dir}: ", G)
    graph = copy.deepcopy(G)
    graph2 = copy.deepcopy(G)
    boundaries = area_disconnects(graph)
    areas_clean = disconnect_areas(graph2, boundaries)
    areas = reconnect_area_switches(copy.deepcopy(areas_clean), boundaries)

    system = WiringDiagram(name=f"{ALGO}_{model_dir}", components=[], links=[])

    if "ieee" in model_dir.lower():
        feeder = generate_feeder("ieee123", "", OUTPUTS)
    else:
        model, level = parse_model_dir(model_dir)
        if not model:
            print(f"Skipping SMART-DS generation for unrecognized folder: {model_dir}")
            return
        feeder = generate_feeder(model, level, OUTPUTS)

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
        f.write(system.model_dump_json())

    with open(f"{SCENARIOS}/{system.name}.json") as f:
        check = WiringDiagram.model_validate_json(f.read())

    plot_network(
        G,
        boundaries,
        areas_clean,
        slack_bus,
        f"{SCENARIOS}/{model_dir}",
        f"{SCENARIOS}/{system.name}.png",
    )

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


def generate() -> None:
    global OUTPUTS
    OUTPUTS = "../../outputs"
    SCENARIOS = f"{ROOT}/scenarios"
    os.makedirs(SCENARIOS, exist_ok=True)

    for item in os.listdir(SCENARIOS):
        dir_path = os.path.join(SCENARIOS, item)
        if not os.path.isdir(dir_path):
            continue
        topology_path = os.path.join(dir_path, "topology.json")
        if not os.path.exists(topology_path):
            continue
        generate_for_model(item, topology_path, SCENARIOS)


def get_topology(path: str) -> Topology:
    assert os.path.exists(path), "need to generate topology from base scenario"

    with open(path) as f:
        topology = Topology.model_validate(json.load(f))

    return topology


if __name__ == "__main__":
    print("generating ADMM scenarios...")
    generate()
