import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from oedisi.types.data_types import IncidenceList, Topology


def generate_graph(inc: IncidenceList, slack_bus: str) -> nx.Graph:
    graph = nx.Graph()
    for src, dst, id in zip(inc.from_equipment, inc.to_equipment, inc.ids):
        if "OPEN" in src or "OPEN" in dst:
            continue
        if src == dst:
            continue

        if "." in src:
            src = src.split(".", 1)[0]
        if "." in dst:
            dst = dst.split(".", 1)[0]

        eq = "LINE"
        if ("sw" in id or "fuse" in id) and "padswitch" not in id:
            eq = "SWITCH"
        if "tr" in id or "reg" in id or "xfm" in id:
            eq = "XFMR"
        graph.add_edge(src, dst, name=f"{src}_{dst}", tag=eq, id=f"{id}")

    for c in nx.connected_components(graph):
        if slack_bus in c:
            return graph.subgraph(c).copy()
    return graph


def get_switches(graph: nx.Graph) -> list:
    switches = []
    for u, v, a in graph.edges(data=True):
        if a.get("tag") == "SWITCH":
            switches.append((u, v, a))
    return switches


def area_disconnects(graph: nx.Graph, n_max: int = 5) -> list:
    switches = get_switches(graph)
    if not switches:
        return []

    switch_weights = []
    for u, v, a in switches:
        temp_graph = graph.copy()
        temp_graph.remove_edge(u, v)
        components = list(nx.connected_components(temp_graph))
        if len(components) >= 2:
            sizes = [len(c) for c in components]
            min_size = min(sizes)
        else:
            min_size = 0
        switch_weights.append((min_size, (u, v, a)))

    switch_weights.sort(key=lambda x: x[0], reverse=True)

    n_open = min(n_max - 1, len(switches))
    if n_open <= 0:
        return []

    open_sw = [sw for _, sw in switch_weights[:n_open]]
    return open_sw


def disconnect_areas(graph: nx.Graph, switches: list) -> list[nx.Graph]:
    graph.remove_edges_from(switches)
    areas = []
    for c in nx.connected_components(graph):
        areas.append(graph.subgraph(c).copy())
    return areas


def reconnect_area_switches(areas: list[nx.Graph], switches: list) -> list[nx.Graph]:
    for area in areas:
        for u, v, a in switches:
            if area.has_node(u) or area.has_node(v):
                area.add_edge(u, v, **a)
    return areas


def get_area_source(graph: nx.Graph, slack_bus: str, switches: list) -> tuple:
    paths = {}
    for u, v, a in switches:
        paths[len(nx.shortest_path(graph, slack_bus, u))] = (u, v, a)
    source = min(paths, key=paths.get)
    return paths[source]


logger = logging.getLogger(__name__)

# Global color mapping/palette for areas to ensure consistency across plots
AREA_COLORS = [
    "#1f77b4",  # Area 0
    "#ff7f0e",  # Area 1
    "#2ca02c",  # Area 2
    "#d62728",  # Area 3
    "#9467bd",  # Area 4
    "#8c564b",  # Area 5
    "#e377c2",  # Area 6
    "#7f7f7f",  # Area 7
    "#bcbd22",  # Area 8
    "#17becf",  # Area 9
]

OBJECTIVE_SHORT_NAMES: dict[str, str] = {
    "maximize_gen": "Gen Max",
    "minimize_loss": "Loss Min",
    "minimize_curtail": "Curtail Min",
    "minimize_load": "Load Min",
}


def format_area_label(
    aid: int,
    area_params: dict[int, dict[str, Any]] | None = None,
    multiline: bool = False,
) -> str:
    """Format area label with compact objective name, e.g. 'Area 0 (Loss Min)' or 'Area 0\n(Loss Min)'."""
    if not area_params or aid not in area_params:
        return f"Area {aid}"
    obj = str(area_params[aid].get("objective", ""))
    short = OBJECTIVE_SHORT_NAMES.get(obj, obj)
    if not short:
        return f"Area {aid}"
    return f"Area {aid}\n({short})" if multiline else f"Area {aid} ({short})"


def format_time_val(time_val: Any) -> str:
    """Format time value to HH:MM format."""
    try:
        dt = pd.to_datetime(str(time_val))
        return dt.strftime("%H:%M")
    except Exception:
        return str(time_val)


def set_ieee_style() -> None:
    """Apply IEEE publication-quality style settings."""
    mpl.rcParams.update(
        {
            # ---- Figure ----
            "figure.figsize": (3.5, 2.5),  # Single column width (inches)
            "figure.dpi": 300,  # High resolution
            "figure.autolayout": True,  # Auto tight_layout
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # ---- Fonts (IEEE uses Times) ----
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,  # Base font size
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            # ---- Math text (matches serif) ----
            "mathtext.fontset": "stix",  # Times-like math font
            # ---- Lines and markers ----
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            # ---- Axes ----
            "axes.linewidth": 0.5,
            "axes.grid": True,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.5,
            # ---- Ticks ----
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # ---- Legend ----
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "legend.fancybox": False,
            # ---- Vector Font Export Settings ----
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def configure_publication_style(font_family: str = "serif", base_font_size: float = 8.0) -> None:
    """Set global Matplotlib rcParams for publication-quality figures."""
    set_ieee_style()


def get_publication_figsize(
    width_type: str | float = "single",
    aspect_ratio: str | float = "golden",
) -> tuple[float, float]:
    """Calculate figure size in inches based on publication columns and aspect ratios.

    Args:
        width_type: 'single' (3.5"), 'double' (7.0"), or a custom float width in inches.
        aspect_ratio: 'golden' (0.618), 'square' (1.0), or a custom float ratio (height/width).
    """
    if width_type == "single":
        width = 3.5
    elif width_type == "double":
        width = 7.0
    elif isinstance(width_type, (int, float)):
        width = float(width_type)
    else:
        raise ValueError(f"Invalid width_type: {width_type}")

    if aspect_ratio == "golden":
        ratio = (5**0.5 - 1) / 2
    elif aspect_ratio == "square":
        ratio = 1.0
    elif isinstance(aspect_ratio, (int, float)):
        ratio = float(aspect_ratio)
    else:
        raise ValueError(f"Invalid aspect_ratio: {aspect_ratio}")

    return (width, width * ratio)


def load_scenario_parameters(
    scenario: Path | dict,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    """Load scenario configuration and extract area IDs and component parameters
    for each area.
    """
    if isinstance(scenario, (str, Path)):
        try:
            with open(scenario, encoding="utf-8") as f:
                scenario_dict = json.load(f)
        except Exception as e:
            logger.error(f"Error loading scenario JSON file {scenario}: {e}")
            return [], {}
    else:
        scenario_dict = scenario

    area_ids: list[int] = []
    area_params: dict[int, dict[str, Any]] = {}
    for comp in scenario_dict.get("components", []):
        comp_name = comp.get("name", "")
        comp_type = comp.get("type", "")
        if (
            comp_type in ("DOPFADMMComponent", "PnnlDopfAdmmComponent")
            or comp_name.startswith("pnnl_dopf_admm_")
            or re.match(r"^area\d+$", comp_name)
        ):
            m = re.search(r"\d+$", comp_name)
            if m:
                area_id = int(m.group())
                area_ids.append(area_id)
                params = comp.get("parameters", {})
                area_params[area_id] = {
                    "source_bus": params.get("source_bus"),
                    "source_line": params.get("source_line"),
                    "switches": params.get("switches", []),
                    "objective": params.get("objective", "maximize_gen"),
                }
    area_ids.sort()
    return area_ids, area_params


def get_boundary_branch_name(G: nx.Graph, source_bus: str, source_line: str | None) -> str | None:
    """Find the branch name (u_v) representing the boundary of the area
    in the full graph.
    """
    if source_line:
        # Search for edge with matching id (switch name) in the full graph G
        for u, v, data in G.edges(data=True):
            if data.get("id") == source_line or data.get("name") == source_line:
                return f"{u}_{v}"

    # Fallback to slack/substation connection:
    # find any edge connected to source_bus in G
    if source_bus and G.has_node(source_bus):
        edges = list(G.edges(source_bus))
        if edges:
            u, v = edges[0]
            return f"{u}_{v}"

    return None


def get_der_mapping(topology_path: Path) -> dict[str, list[str]]:
    """Build a mapping from DER equipment ID to its connected bus.phase IDs."""
    try:
        with open(topology_path, encoding="utf-8") as f:
            topology = Topology.model_validate(json.load(f))
    except Exception as e:
        logger.error(f"Failed to load topology for DER mapping: {e}")
        return {}

    der_map: dict[str, list[str]] = {}
    real_inj = topology.injections.power_real
    for bus_phase, eq_id in zip(real_inj.ids, real_inj.equipment_ids):
        if eq_id.lower().startswith("pvsystem."):
            der_map.setdefault(eq_id, []).append(bus_phase)
    return der_map


def load_recorder_data(data_dir: Path, scenario: Path | dict) -> dict[str, pd.DataFrame]:
    """Ingest feeder and ADMM area recorder feather files into pandas dataframes
    using the scenario configuration.
    """
    if isinstance(scenario, (str, Path)):
        try:
            with open(scenario, encoding="utf-8") as f:
                scenario_dict = json.load(f)
        except Exception as e:
            logger.error(f"Error loading scenario JSON file {scenario}: {e}")
            return {}
    else:
        scenario_dict = scenario

    # Map component name to its parameters and type
    components = {comp["name"]: comp for comp in scenario_dict.get("components", [])}
    feeder_names = [name for name, comp in components.items() if comp.get("type") in ["Feeder", "LocalFeeder"]]
    control_feeder_name = next(
        (name for name in feeder_names if "control" in name.lower() or "local" in name.lower()), None
    )
    reference_feeder_name = next(
        (name for name in feeder_names if "reference" in name.lower() or "ref" in name.lower()), None
    )
    if not control_feeder_name and feeder_names:
        control_feeder_name = feeder_names[0]

    # Build a lookup of target component name to its incoming link details (source, source_port)
    incoming_links = {}
    for link in scenario_dict.get("links", []):
        target = link.get("target")
        if target:
            incoming_links[target] = (link.get("source"), link.get("source_port"))

    data: dict[str, pd.DataFrame] = {}

    for comp in scenario_dict.get("components", []):
        if comp.get("type") != "Recorder":
            continue

        name = comp.get("name", "")
        params = comp.get("parameters", {})
        feather_filename = params.get("feather_filename")
        if not feather_filename:
            continue

        filename = Path(feather_filename).name
        file_path = data_dir / filename
        run_dir = data_dir.parent if data_dir.name == "outputs" else data_dir
        if not file_path.exists():
            file_path = run_dir / "build" / name / filename
        if not file_path.exists():
            matches = list(run_dir.rglob(filename))
            if matches:
                file_path = matches[0]

        key = None
        link_info = incoming_links.get(name)
        if link_info:
            source, source_port = link_info
            if source == control_feeder_name:
                if source_port == "voltages_real":
                    key = "feeder_v_real"
                elif source_port == "voltages_imag":
                    key = "feeder_v_imag"
                elif source_port == "powers_real":
                    key = "feeder_p_real"
                elif source_port == "powers_imag":
                    key = "feeder_p_imag"
            elif source == reference_feeder_name:
                if source_port == "voltages_real":
                    key = "reference_v_real"
                elif source_port == "voltages_imag":
                    key = "reference_v_imag"
                elif source_port == "powers_real":
                    key = "reference_p_real"
                elif source_port == "powers_imag":
                    key = "reference_p_imag"
            elif source == "feeder" and not reference_feeder_name:
                # Fallback for single feeder named "feeder"
                if source_port == "voltages_real":
                    key = "feeder_v_real"
                elif source_port == "voltages_imag":
                    key = "feeder_v_imag"
                elif source_port == "powers_real":
                    key = "feeder_p_real"
                elif source_port == "powers_imag":
                    key = "feeder_p_imag"
            elif source and (
                source.startswith("pnnl_dopf_admm_") or source.startswith("area") or source.startswith("stats")
            ):
                m = re.search(r"\d+$", source)
                if m:
                    aid = int(m.group())
                    if source_port == "voltages_mag":
                        key = f"area_{aid}_v_mag"
                    elif source_port == "powers_mag":
                        key = f"area_{aid}_p_mag"
                    elif source_port == "powers_ang":
                        key = f"area_{aid}_p_ang"
                    elif source_port == "controls_real":
                        key = f"area_{aid}_ctrl_real"
                    elif source_port == "controls_imag":
                        key = f"area_{aid}_ctrl_imag"
                    elif source_port == "solver_stats":
                        key = f"area_{aid}_stats"

        if key:
            if file_path.exists():
                data[key] = pd.read_feather(file_path)
                logger.info(f"Loaded {filename} for {key} with shape {data[key].shape}")
            else:
                logger.warning(f"Required recorder file not found: {filename} (expected for {key})")

    # Align control feeder records to reference timestamps (selecting post-control state for each step)
    if "reference_v_real" in data and "time" in data["reference_v_real"].columns:
        ref_times = data["reference_v_real"]["time"].tolist()
        ctrl_keys = ["feeder_v_real", "feeder_v_imag", "feeder_p_real", "feeder_p_imag"]
        for ckey in ctrl_keys:
            if ckey in data and "time" in data[ckey].columns:
                cdf = data[ckey]
                aligned_rows = []
                for r_t in ref_times:
                    prefix = str(r_t)[:13]
                    matches = cdf[cdf["time"].astype(str).str.startswith(prefix)]
                    if len(matches) > 0:
                        row = matches.iloc[-1].to_dict()
                        row["time"] = r_t
                        aligned_rows.append(row)
                if len(aligned_rows) == len(ref_times):
                    data[ckey] = pd.DataFrame(aligned_rows)
                    logger.info(f"Aligned {ckey} with post-control timesteps (shape {data[ckey].shape})")

    return data


def process_voltages(
    data: dict[str, pd.DataFrame],
    area_ids: list[int],
    area_buses: list[list[str]],
    topology: Topology,
) -> dict[int, pd.DataFrame]:
    """Calculate voltage magnitudes from both reference and control feeders
    for all buses in each area and compare them."""
    voltage_comparisons: dict[int, pd.DataFrame] = {}

    # Need both control and reference feeder voltage data
    has_control = "feeder_v_real" in data and "feeder_v_imag" in data
    has_reference = "reference_v_real" in data and "reference_v_imag" in data

    if not has_control:
        logger.error("Control feeder voltage real/imag data missing. Skipping voltage processing.")
        return {}

    ctrl_real_df = data["feeder_v_real"].set_index("time")
    ctrl_imag_df = data["feeder_v_imag"].set_index("time")
    ctrl_v_mag = (ctrl_real_df**2 + ctrl_imag_df**2) ** 0.5

    ref_v_mag = None
    if has_reference:
        ref_real_df = data["reference_v_real"].set_index("time")
        ref_imag_df = data["reference_v_imag"].set_index("time")
        ref_v_mag = (ref_real_df**2 + ref_imag_df**2) ** 0.5

    # Map bus_phase to its nominal base voltage magnitude
    base_voltages = {}
    if topology.base_voltage_magnitudes:
        base_voltages = dict(
            zip(
                topology.base_voltage_magnitudes.ids,
                topology.base_voltage_magnitudes.values,
            )
        )

    for aid in area_ids:
        buses_in_area = area_buses[aid]
        comparison_records = []

        # Determine common timestamps between control and reference
        if ref_v_mag is not None:
            common_times = ctrl_v_mag.index.intersection(ref_v_mag.index)
        else:
            common_times = ctrl_v_mag.index

        for col in ctrl_v_mag.columns:
            if col == "time":
                continue
            bus_name = col.split(".", 1)[0]
            if bus_name in buses_in_area:
                base_v = base_voltages.get(col, 1.0)
                if base_v <= 0:
                    base_v = 1.0

                for t in common_times:
                    v_ctrl_val = float(ctrl_v_mag.loc[t, col]) / base_v

                    v_ref_val = None
                    if ref_v_mag is not None and col in ref_v_mag.columns:
                        v_ref_val = float(ref_v_mag.loc[t, col]) / base_v

                    comparison_records.append(
                        {
                            "time": t,
                            "bus_phase": col,
                            "area_id": aid,
                            "v_reference": v_ref_val,
                            "v_control": v_ctrl_val,
                        }
                    )

        if comparison_records:
            voltage_comparisons[aid] = pd.DataFrame(comparison_records)

    return voltage_comparisons


def get_descendants(G: nx.Graph, root: str, node: str) -> set[str]:
    """Find all nodes downstream of `node` in the tree rooted at `root`."""
    if root == node:
        return set(G.nodes())
    try:
        path = nx.shortest_path(G, source=root, target=node)
        parent = path[-2] if len(path) > 1 else None
    except nx.NetworkXNoPath:
        parent = None

    descendants = {node}
    queue = [node]
    while queue:
        curr = queue.pop(0)
        for neighbor in G.neighbors(curr):
            if neighbor != parent and neighbor not in descendants:
                descendants.add(neighbor)
                queue.append(neighbor)
    return descendants


def process_power_flows(
    data: dict[str, pd.DataFrame],
    area_ids: list[int],
    area_params: dict[int, dict[str, Any]],
    G: nx.Graph,
    area_buses: list[list[str]],
    der_map: dict[str, list[str]],
    slack_bus: str,
) -> dict[str, Any]:
    """Process boundary power flows and DER active/reactive power controls."""
    results: dict[str, Any] = {
        "boundary_flows": {},
        "der_comparisons": {},
    }

    # 1. Compare Boundary Power Flow
    for aid in area_ids:
        params = area_params.get(aid)
        if not params:
            continue

        # Check if control feeder power data is available
        if "feeder_p_real" not in data or "feeder_p_imag" not in data:
            continue

        feeder_p = data["feeder_p_real"].set_index("time")
        feeder_q = data["feeder_p_imag"].set_index("time")

        # Check reference feeder power data
        has_reference = "reference_p_real" in data and "reference_p_imag" in data
        if has_reference:
            ref_p = data["reference_p_real"].set_index("time")
            ref_q = data["reference_p_imag"].set_index("time")
            common_times = feeder_p.index.intersection(ref_p.index)
        else:
            ref_p, ref_q = None, None
            common_times = feeder_p.index

        buses_in_area = area_buses[aid]
        buses_in_area_set = set(buses_in_area)
        area_cols = [c for c in feeder_p.columns if c != "time" and c.split(".")[0] in buses_in_area_set]
        if not area_cols:
            continue

        boundary_records = []
        for t in common_times:
            p_ctrl_sum = 0.0
            q_ctrl_sum = 0.0
            p_ref_sum = 0.0
            q_ref_sum = 0.0
            for col in area_cols:
                p_ctrl_sum += float(feeder_p.loc[t, col])
                q_ctrl_sum += float(feeder_q.loc[t, col])
                if ref_p is not None and col in ref_p.columns:
                    p_ref_sum += float(ref_p.loc[t, col])
                if ref_q is not None and col in ref_q.columns:
                    q_ref_sum += float(ref_q.loc[t, col])
            boundary_records.append(
                {
                    "time": t,
                    "p_control_net_import": -p_ctrl_sum,
                    "q_control_net_import": -q_ctrl_sum,
                    "p_reference_net_import": -p_ref_sum,
                    "q_reference_net_import": -q_ref_sum,
                }
            )

        if boundary_records:
            results["boundary_flows"][aid] = pd.DataFrame(boundary_records)

    # 2. Highlight DER Injections (Controls)
    for aid in area_ids:
        ctrl_real_key = f"area_{aid}_ctrl_real"
        ctrl_imag_key = f"area_{aid}_ctrl_imag"

        if ctrl_real_key not in data or ctrl_imag_key not in data:
            continue

        ctrl_real_df = data[ctrl_real_key].set_index("time")
        ctrl_imag_df = data[ctrl_imag_key].set_index("time")

        der_records = []
        for der_id in ctrl_real_df.columns:
            if der_id == "time":
                continue
            connected_phases = der_map.get(der_id, [])
            if not connected_phases:
                continue

            num_phases = len(connected_phases)
            for bus_phase in connected_phases:
                common_times = ctrl_real_df.index
                for t in common_times:
                    p_admm_ctrl = float(ctrl_real_df.loc[t, der_id]) / num_phases
                    q_admm_ctrl = float(ctrl_imag_df.loc[t, der_id]) / num_phases

                    der_records.append(
                        {
                            "time": t,
                            "der_id": der_id,
                            "bus_phase": bus_phase,
                            "p_admm_ctrl": p_admm_ctrl,
                            "q_admm_ctrl": q_admm_ctrl,
                        }
                    )

        if der_records:
            results["der_comparisons"][aid] = pd.DataFrame(der_records)

    return results


def get_edge_flow(u: str, v: str, p_mag_df: pd.DataFrame, p_ang_df: pd.DataFrame, t: Any) -> float:
    """Calculate the active power flow on edge (u, v) at time t from the area data."""
    flow_sum = 0.0
    for col in p_mag_df.columns:
        if col.startswith(f"{u}_{v}."):
            mag = float(p_mag_df.loc[t, col])
            ang = float(p_ang_df.loc[t, col])
            flow_sum += mag * math.cos(ang)
        elif col.startswith(f"{v}_{u}."):
            mag = float(p_mag_df.loc[t, col])
            ang = float(p_ang_df.loc[t, col])
            flow_sum -= mag * math.cos(ang)
    return flow_sum


def process_self_sufficiency(
    data: dict[str, pd.DataFrame],
    area_ids: list[int],
    area_buses: list[list[str]],
    der_map: dict[str, list[str]],
    G: nx.Graph,
    slack_bus: str,
) -> dict[int, pd.DataFrame]:
    """Process area internal load, local generation, self-sufficiency,
    and boundary flows over time.
    """
    self_sufficiency: dict[int, pd.DataFrame] = {}

    for aid in area_ids:
        ctrl_real_key = f"area_{aid}_ctrl_real"
        if ctrl_real_key not in data:
            continue
        ctrl_real_df = data[ctrl_real_key].set_index("time")

        buses_in_area = area_buses[aid]
        buses_in_area_set = set(buses_in_area)
        records = []

        # Find DERs located in this area
        area_ders = []
        for der_id, bus_phases in der_map.items():
            if bus_phases and bus_phases[0].split(".")[0] in buses_in_area_set:
                area_ders.append(der_id)

        # Identify upstream and downstream boundary edges
        boundary_edges = []
        for u in buses_in_area:
            for v in G.neighbors(u):
                if v not in buses_in_area_set:
                    boundary_edges.append((u, v))

        upstream_edges = []
        downstream_edges = []
        for u, v in boundary_edges:
            try:
                len_u = nx.shortest_path_length(G, slack_bus, u)
                len_v = nx.shortest_path_length(G, slack_bus, v)
                if len_v < len_u:
                    upstream_edges.append((u, v))
                else:
                    downstream_edges.append((u, v))
            except nx.NetworkXNoPath:
                upstream_edges.append((u, v))

        p_mag_key = f"area_{aid}_p_mag"
        p_ang_key = f"area_{aid}_p_ang"
        has_boundary_data = p_mag_key in data and p_ang_key in data

        p_mag_df = data.get(p_mag_key, pd.DataFrame()).set_index("time") if has_boundary_data else None
        p_ang_df = data.get(p_ang_key, pd.DataFrame()).set_index("time") if has_boundary_data else None

        for t in ctrl_real_df.index:
            # 1. Total Generation = sum of DER active power controls
            p_gen = 0.0
            for der_id in area_ders:
                if der_id in ctrl_real_df.columns:
                    p_gen += float(ctrl_real_df.loc[t, der_id])

            # 2. Net Injection = sum of all active power injections in
            # the area from feeder
            p_net_inj = 0.0
            if "feeder_p_real" in data:
                feeder_p = data["feeder_p_real"].set_index("time")
                cols = [c for c in feeder_p.columns if c != "time" and c.split(".")[0] in buses_in_area_set]
                if t in feeder_p.index:
                    p_net_inj = float(feeder_p.loc[t, cols].sum())

            # 3. Total Load = Generation - Net Injection
            p_load = p_gen - p_net_inj
            p_import = -p_net_inj

            # 4. Calculate Upstream Import and Downstream Export
            p_upstream_import = 0.0
            p_downstream_export = 0.0
            if has_boundary_data:
                for u, v in upstream_edges:
                    p_upstream_import += get_edge_flow(v, u, p_mag_df, p_ang_df, t)
                for u, v in downstream_edges:
                    p_downstream_export += get_edge_flow(u, v, p_mag_df, p_ang_df, t)

            # Self Sufficiency Index (SSI) = Local Gen / Load (capped at 100%)
            ssi = (p_gen / p_load * 100.0) if p_load > 0 else 100.0
            ssi = min(max(ssi, 0.0), 100.0)

            records.append(
                {
                    "time": t,
                    "p_generation": p_gen,
                    "p_load": p_load,
                    "p_import": p_import,
                    "p_upstream_import": p_upstream_import,
                    "p_downstream_export": p_downstream_export,
                    "self_sufficiency_pct": ssi,
                }
            )

        if records:
            self_sufficiency[aid] = pd.DataFrame(records)

    return self_sufficiency


def process_convergence(
    data: dict[str, pd.DataFrame],
    area_ids: list[int],
) -> dict[int, pd.DataFrame]:
    """Extract convergence metrics (optimality and feasibility gaps) over ADMM iterations."""
    convergence_data = {}
    for aid in area_ids:
        stats_key = f"area_{aid}_stats"
        if stats_key in data:
            df = data[stats_key]
            if "time" in df.columns and "admm_iteration" in df.columns:
                convergence_data[aid] = df.sort_values(by=["time", "admm_iteration"])
    return convergence_data


def process_generation_adequacy(
    topology: Topology,
    area_ids: list[int],
    area_buses: list[list[str]],
    area_params: dict[int, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Calculate the aggregated rated generation capacity and rated load for each area from grid network models."""
    # Create area map: bus_name -> area_id
    bus_area_map = {}
    for aid in area_ids:
        for bus in area_buses[aid]:
            bus_area_map[bus] = aid

    # Initialize rated generation and rated load per area
    rated_gen = {aid: 0.0 for aid in area_ids}
    rated_load = {aid: 0.0 for aid in area_ids}

    # Iterate over injections in topology
    real_inj = topology.injections.power_real
    for bus_phase, eq_id, val in zip(real_inj.ids, real_inj.equipment_ids, real_inj.values):
        bus = bus_phase.split(".")[0]
        aid = bus_area_map.get(bus)
        if aid is None:
            continue

        eq_id_lower = eq_id.lower()
        if "pvsystem" in eq_id_lower:
            rated_gen[aid] += val
        elif "load" in eq_id_lower:
            rated_load[aid] += abs(val)

    records = []
    for aid in area_ids:
        area_label = format_area_label(aid, area_params, multiline=True)
        records.append(
            {
                "Area": area_label,
                "Power Capacity (kW)": rated_gen[aid],
                "Metric": "Rated Generation",
            }
        )
        records.append(
            {
                "Area": area_label,
                "Power Capacity (kW)": rated_load[aid],
                "Metric": "Rated Load",
            }
        )

    return pd.DataFrame(records)


# ──── Plotting functions returning matplotlib.figure.Figure ───────────


def get_max_diff_timestep(data: dict[str, pd.DataFrame], topology: Topology) -> Any:
    """Find the common timestamp where the control and reference feeder voltages differ the most."""
    if not all(k in data for k in ["feeder_v_real", "feeder_v_imag", "reference_v_real", "reference_v_imag"]):
        if "feeder_v_real" in data:
            return data["feeder_v_real"]["time"].max()
        return None

    c_real = data["feeder_v_real"]
    c_imag = data["feeder_v_imag"]
    r_real = data["reference_v_real"]
    r_imag = data["reference_v_imag"]

    time_col = "time" if "time" in c_real.columns else c_real.columns[0]
    c_times = c_real[time_col].unique()
    r_times = r_real[time_col].unique()
    common_times = [t for t in c_times if t in r_times]
    if not common_times:
        return c_times[-1] if len(c_times) > 0 else None

    # Common columns excluding time
    cols = [c for c in c_real.columns if c != time_col and c in r_real.columns]
    if not cols:
        return common_times[-1]

    # Calculate L1 voltage magnitude difference across all nodes for each common timestep
    max_diff = -1.0
    best_time = common_times[-1]

    for t in common_times:
        cr = c_real[c_real[time_col] == t][cols].values
        ci = c_imag[c_imag[time_col] == t][cols].values
        rr = r_real[r_real[time_col] == t][cols].values
        ri = r_imag[r_imag[time_col] == t][cols].values

        if cr.size == 0 or rr.size == 0:
            continue

        c_mag = np.sqrt(cr**2 + ci**2)
        r_mag = np.sqrt(rr**2 + ri**2)
        diff = np.abs(c_mag - r_mag).sum()

        if diff > max_diff:
            max_diff = diff
            best_time = t

    return best_time


def plot_voltage_comparison(
    voltage_data: dict[int, pd.DataFrame],
    timestep: Any = None,
    area_params: dict[int, dict[str, Any]] | None = None,
    figsize: tuple[float, float] | None = None,
    target: str = "paper",
) -> plt.Figure | None:
    """Generate a split violin plot comparing Reference vs Control feeder voltages per area."""
    records = []
    for aid, df in voltage_data.items():
        if df.empty:
            continue
        if timestep is not None:
            df_latest = df[df["time"] == timestep]
        else:
            latest_time = df["time"].max()
            df_latest = df[df["time"] == latest_time]

        area_label = format_area_label(aid, area_params, multiline=True)
        for _, row in df_latest.iterrows():
            if row.get("v_reference") is not None:
                records.append({"Voltage (p.u.)": row["v_reference"], "Area": area_label, "Case": "Reference"})
            records.append(
                {
                    "Voltage (p.u.)": row["v_control"],
                    "Area": area_label,
                    "Case": "Control",
                }
            )

    if not records:
        logger.warning("No voltage data to plot. Skipping plot.")
        return None

    df_volt = pd.DataFrame(records)
    df_volt = df_volt.sort_values(by="Area")

    if figsize is None:
        if target == "notebook":
            figsize = (7.5, 4.0)
        else:
            figsize = get_publication_figsize("single", "golden")

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=df_volt,
        x="Area",
        y="Voltage (p.u.)",
        hue="Case",
        hue_order=["Reference", "Control"],
        split=True,
        inner="quart",
        ax=ax,
    )

    # Recolor: even indices (left, Reference) are grey, odd indices (right, Control) are Area Colors
    for idx, coll in enumerate(ax.collections):
        if idx % 2 == 0:
            coll.set_facecolor("#b0bec5")  # Grey for Reference
        else:
            area_idx = idx // 2
            coll.set_facecolor(AREA_COLORS[area_idx % len(AREA_COLORS)])

    limit_line = ax.axhline(1.05, color="r", linestyle="--", label="Voltage Limits")
    ax.axhline(0.95, color="r", linestyle="--")

    ax.set_ylabel("Voltage Magnitude (p.u.)")
    ax.set_xlabel("Control Area")
    if timestep is not None:
        ax.set_title(f"Feeder Voltage Distributions ({format_time_val(timestep)})")

    # Custom legend
    legend_elements = [
        mpatches.Patch(color="#b0bec5", label="Reference"),
        mpatches.Patch(color="#7f7f7f", label="Control (Colored by Area)"),
        limit_line,
    ]
    if target == "notebook":
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            framealpha=0.95,
        )
    else:
        ax.legend(handles=legend_elements, loc="best")
    ax.tick_params(axis="x", labelsize=6.5)
    return fig


def plot_power_flow_comparison(
    flow_data: dict[str, Any],
    timestep: Any = None,
    area_params: dict[int, dict[str, Any]] | None = None,
    figsize: tuple[float, float] | None = None,
    target: str = "paper",
) -> plt.Figure | None:
    """Generate a high-quality grouped bar chart comparing ADMM vs Feeder boundary flows."""
    boundary_flows = flow_data["boundary_flows"]
    if not boundary_flows:
        logger.warning("No boundary flow data to plot.")
        return None

    records = []
    for aid, df in boundary_flows.items():
        if df.empty:
            continue
        if timestep is not None:
            df_latest = df[df["time"] == timestep]
        else:
            latest_time = df["time"].max()
            df_latest = df[df["time"] == latest_time]

        area_label = format_area_label(aid, area_params, multiline=True)
        for _, row in df_latest.iterrows():
            p_control = row.get("p_control_net_import", 0.0)
            p_reference = row.get("p_reference_net_import", 0.0)

            records.append(
                {
                    "Area": area_label,
                    "Real Power (kW)": p_control,
                    "Case": "Control",
                }
            )
            records.append(
                {
                    "Area": area_label,
                    "Real Power (kW)": p_reference,
                    "Case": "Reference",
                }
            )

    if not records:
        logger.warning("No boundary flow records to plot.")
        return None

    df_plot = pd.DataFrame(records)
    df_plot = df_plot.sort_values(by="Area")

    if figsize is None:
        if target == "notebook":
            figsize = (7.5, 4.0)
        else:
            figsize = get_publication_figsize("single", "golden")

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df_plot,
        x="Area",
        y="Real Power (kW)",
        hue="Case",
        hue_order=["Reference", "Control"],
        ax=ax,
    )

    # Recolor: first group (Reference) is grey, second group (Control) is Area Colors
    N = len(df_plot["Area"].unique())
    for idx, patch in enumerate(ax.patches):
        if idx < N:
            patch.set_facecolor("#b0bec5")  # Grey for Reference
        else:
            area_idx = idx - N
            patch.set_facecolor(AREA_COLORS[area_idx % len(AREA_COLORS)])

    ax.set_xlabel("Control Area")
    ax.set_ylabel("Real Power Exchange (kW)")

    # Custom legend
    legend_elements = [
        mpatches.Patch(color="#b0bec5", label="Reference"),
        mpatches.Patch(color="#7f7f7f", label="Control (Colored by Area)"),
    ]
    if target == "notebook":
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            framealpha=0.95,
        )
    else:
        ax.legend(handles=legend_elements, loc="best")
    ax.tick_params(axis="x", labelsize=6.5)
    plt.xticks(rotation=0)
    return fig


def plot_generation_adequacy(
    adequacy_df: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
    target: str = "paper",
) -> plt.Figure | None:
    """Generate a high-quality side-by-side bar chart of Rated Generation vs Rated Load per area."""
    if adequacy_df.empty:
        return None

    # Sort to match color mapping
    adequacy_df = adequacy_df.sort_values(by="Area")

    if figsize is None:
        if target == "notebook":
            figsize = (7.5, 4.0)
        else:
            figsize = get_publication_figsize("single", "golden")

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=adequacy_df,
        x="Area",
        y="Power Capacity (kW)",
        hue="Metric",
        hue_order=["Rated Load", "Rated Generation"],
        ax=ax,
    )

    # Recolor: first group (Rated Load) is grey, second group (Rated Generation) is Area Colors
    N = len(adequacy_df["Area"].unique())
    for idx, patch in enumerate(ax.patches):
        if idx < N:
            patch.set_facecolor("#b0bec5")  # Grey for Rated Load
        else:
            area_idx = idx - N
            patch.set_facecolor(AREA_COLORS[area_idx % len(AREA_COLORS)])

    ax.set_ylabel("Power Capacity (kW)")
    ax.set_xlabel("Control Area")

    # Custom legend
    legend_elements = [
        mpatches.Patch(color="#b0bec5", label="Rated Load"),
        mpatches.Patch(color="#7f7f7f", label="Rated Generation (Colored by Area)"),
    ]
    if target == "notebook":
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            framealpha=0.95,
        )
    else:
        ax.legend(handles=legend_elements, loc="best")
    ax.tick_params(axis="x", labelsize=6.5)
    return fig


def plot_algorithmic_convergence(
    convergence_data: dict[int, pd.DataFrame],
    area_params: dict[int, dict[str, Any]] | None = None,
    figsize: tuple[float, float] | None = None,
    target: str = "paper",
) -> plt.Figure | None:
    """Generate a high-quality semi-log plot of ADMM convergence history at each timestep."""
    if not convergence_data:
        logger.warning("No convergence data to plot.")
        return None

    if figsize is None:
        if target == "notebook":
            figsize = (7.5, 4.5)
        else:
            figsize = get_publication_figsize("single", 1.2)

    records = []
    for aid, df in convergence_data.items():
        if df.empty:
            continue
        idx = df.groupby("time")["admm_iteration"].idxmax()
        df_final = df.loc[idx]
        area_label = format_area_label(aid, area_params)
        for _, row in df_final.iterrows():
            records.append(
                {
                    "time": format_time_val(row["time"]),
                    "Area": area_label,
                    "Optimality Gap": abs(row["optimality_gap"]),
                    "Feasibility Gap": abs(row["feasibility_gap"]),
                }
            )

    if not records:
        logger.warning("No final iteration gaps to plot.")
        return None

    df_plot = pd.DataFrame(records)
    df_plot = df_plot.sort_values(by="time")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    colors = AREA_COLORS

    def _extract_aid(area_str: str) -> int:
        m = re.search(r"\d+", area_str)
        return int(m.group()) if m else 0

    areas = sorted(df_plot["Area"].unique(), key=_extract_aid)

    legend_handles = []
    legend_labels = []

    for area_name in areas:
        df_area = df_plot[df_plot["Area"] == area_name]
        aid = _extract_aid(area_name)
        color = colors[aid % len(colors)]

        # Plot Optimality Gap (Top)
        line_opt = ax1.semilogy(
            df_area["time"],
            df_area["Optimality Gap"],
            "o-",
            color=color,
            markersize=3,
            label=f"{area_name} Opt Gap",
        )[0]
        legend_handles.append(line_opt)
        legend_labels.append(area_name)

        # Plot Feasibility Gap (Bottom)
        ax2.semilogy(
            df_area["time"],
            df_area["Feasibility Gap"],
            "s--",
            color=color,
            markersize=3,
            label=f"{area_name} Feas Gap",
        )

    # Add standard convergence tolerance line (1e-3)
    tol_line = ax1.axhline(1e-3, color="gray", linestyle=":")
    ax2.axhline(1e-3, color="gray", linestyle=":")
    legend_handles.append(tol_line)
    legend_labels.append("Tolerance (1e-3)")

    ax1.set_ylabel("Optimality Gap")
    ax2.set_ylabel("Feasibility Gap")
    ax2.set_xlabel("Time")

    if target == "notebook":
        ax1.legend(
            legend_handles,
            legend_labels,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            framealpha=0.95,
        )
    else:
        ax1.legend(legend_handles, legend_labels, loc="best", ncol=2)

    return fig


def load_coordinates(
    coords_dir: str | Path | None = None,
    scenario_path: str | Path | None = None,
) -> dict[str, tuple[float, float]]:
    """Search for bus_coords.csv in standard locations or scenario folder."""
    search_dirs = []
    if coords_dir:
        search_dirs.append(Path(coords_dir))
    if scenario_path:
        search_dirs.append(Path(scenario_path).parent)

    # Search known dataset directories
    known_data_dirs = [
        Path("tests/test_data"),
        Path("data"),
        Path("../tests/test_data"),
        Path("../../data"),
    ]
    search_dirs.extend(known_data_dirs)

    # Also inspect component parameters in scenario JSON for opendss_location / profile_location
    if scenario_path and Path(scenario_path).exists():
        try:
            with open(scenario_path, encoding="utf-8") as f:
                scen = json.load(f)
            for comp in scen.get("components", []):
                for k in ["opendss_location", "profile_location"]:
                    val = comp.get("parameters", {}).get(k)
                    if val:
                        search_dirs.append(Path(val))
        except Exception:
            pass

    for d in search_dirs:
        if not d.exists():
            continue
        # Search for bus_coords.csv
        matches = list(d.glob("**/bus_coords.csv"))
        if not matches:
            matches = list(d.glob("**/BusCoords.dat")) + list(d.glob("**/Buscoords.dat"))
        if matches:
            f_path = matches[0]
            coords = {}
            with open(f_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("//") or line.startswith("!"):
                        continue
                    parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
                    if len(parts) >= 3:
                        bus = parts[0].strip("'\"")
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            coords[bus] = (x, y)
                        except ValueError:
                            pass
            if coords:
                logger.info(f"Loaded {len(coords)} bus coordinates from {f_path}")
                return coords

    return {}


def plot_network_partition(
    G: nx.Graph,
    boundaries: list,
    areas_clean: list[nx.Graph],
    slack_bus: str,
    coords_dir: str | Path | None = None,
    scenario_path: str | Path | None = None,
    area_params: dict[int, dict[str, Any]] | None = None,
    figsize: tuple[float, float] | None = None,
    target: str = "paper",
) -> plt.Figure:
    """Generate the network partition map showing control areas and boundary switches."""
    if figsize is None:
        if target == "notebook":
            figsize = (7.5, 5.5)
        else:
            figsize = get_publication_figsize("single", "square")

    fig, ax = plt.subplots(figsize=figsize)

    coords = load_coordinates(coords_dir, scenario_path=scenario_path)
    if coords:
        coords_upper = {k.upper(): v for k, v in coords.items()}
        pos = {node: coords_upper[node.upper()] for node in G.nodes() if node.upper() in coords_upper}
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

    colors = AREA_COLORS
    node_colors = [colors[node_to_area.get(node, 0) % len(colors)] for node in G.nodes()]

    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=12, ax=ax)

    for u, v, a in boundaries:
        if u in pos and v in pos:
            mid_x = (pos[u][0] + pos[v][0]) / 2.0
            mid_y = (pos[u][1] + pos[v][1]) / 2.0
            ax.plot(
                mid_x,
                mid_y,
                marker="s",
                color="red",
                markersize=5,
                markeredgecolor="black",
                zorder=5,
            )

    if slack_bus in G.nodes():
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[slack_bus],
            node_shape="*",
            node_color="gold",
            node_size=60,
            edgecolors="black",
            ax=ax,
        )

    legend_elements = []
    for idx, area in enumerate(areas_clean):
        color = colors[idx % len(colors)]
        label = format_area_label(idx, area_params)
        legend_elements.append(mpatches.Patch(color=color, label=label))
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=5,
            label="Boundary",
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
                markersize=8,
                label="Slack Bus",
            )
        )

    if target == "notebook":
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            framealpha=0.95,
        )
    else:
        ax.legend(handles=legend_elements, loc="best")
    ax.axis("off")
    return fig


def _get_base_voltages(topology: Any, common_cols: list[str]) -> dict[str, float]:
    """Extract base voltage dictionary for common_cols from the topology."""
    if not topology or not hasattr(topology, "base_voltage_magnitudes"):
        return {}
    try:
        info = topology.base_voltage_magnitudes
        return dict(zip(info.ids, info.values))
    except Exception:
        return {}


def plot_voltage_scatter_at_timestep(
    data: dict[str, pd.DataFrame],
    topology: Topology,
    timestep_idx: int = -1,
    timestep_val: Any = None,
    figsize: tuple[float, float] | None = None,
    target: str = "paper",
) -> plt.Figure | None:
    """Generate a scatter plot comparing individual bus voltage magnitudes (control vs reference)
    at a single timestep.
    """
    if not all(k in data for k in ["feeder_v_real", "feeder_v_imag", "reference_v_real", "reference_v_imag"]):
        logger.warning("Missing voltage data for voltage scatter plot. Skipping.")
        return None

    c_real = data["feeder_v_real"]
    c_imag = data["feeder_v_imag"]
    r_real = data["reference_v_real"]
    r_imag = data["reference_v_imag"]

    time_col = "time" if "time" in c_real.columns else c_real.columns[0]

    # Align by common times
    c_times = c_real[time_col].unique()
    r_times = r_real[time_col].unique()
    common_times = np.intersect1d(c_times, r_times)

    if len(common_times) == 0:
        logger.warning("No common timestamps found for voltage scatter plot.")
        return None

    c_r_df = c_real[c_real[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    c_i_df = c_imag[c_imag[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    r_r_df = r_real[r_real[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    r_i_df = r_imag[r_imag[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)

    # Compute magnitudes
    common_cols = [c for c in c_r_df.columns if c in r_r_df.columns]
    if not common_cols:
        logger.warning("No common bus columns found for voltage scatter plot.")
        return None

    base_voltages = _get_base_voltages(topology, common_cols)
    if not any(col in base_voltages for col in common_cols):
        logger.error("Voltage scatter failed: Recorded bus columns do not match topology base voltage IDs.")
        return None

    if timestep_val is not None:
        t_val = timestep_val
    else:
        if timestep_idx == -1 or timestep_idx is None:
            max_diff = -1.0
            best_idx = 0
            for idx, t in enumerate(common_times):
                diffs = []
                for col in common_cols:
                    v_r_ref = r_r_df.loc[t, col]
                    v_i_ref = r_i_df.loc[t, col]
                    v_ref_mag = (v_r_ref**2 + v_i_ref**2) ** 0.5
                    v_r_ctrl = c_r_df.loc[t, col]
                    v_i_ctrl = c_i_df.loc[t, col]
                    v_ctrl_mag = (v_r_ctrl**2 + v_i_ctrl**2) ** 0.5
                    base_v = base_voltages.get(col, 1.0)
                    if base_v <= 0:
                        base_v = 1.0
                    diffs.append(abs(v_ref_mag - v_ctrl_mag) / base_v)
                mean_diff = float(np.mean(diffs))
                if mean_diff > max_diff:
                    max_diff = mean_diff
                    best_idx = idx
            timestep_idx = best_idx
            logger.info(
                f"Selected timestep index {timestep_idx} ({common_times[timestep_idx]}) with maximum mean voltage difference of {max_diff:.5f} p.u. for the scatter plot."
            )
        t_val = common_times[timestep_idx]

    v_ref_list = []
    v_ctrl_list = []

    for col in common_cols:
        v_r_ref = r_r_df.loc[t_val, col]
        v_i_ref = r_i_df.loc[t_val, col]
        v_ref_mag = (v_r_ref**2 + v_i_ref**2) ** 0.5

        v_r_ctrl = c_r_df.loc[t_val, col]
        v_i_ctrl = c_i_df.loc[t_val, col]
        v_ctrl_mag = (v_r_ctrl**2 + v_i_ctrl**2) ** 0.5

        base_v = base_voltages.get(col, 1.0)
        if base_v <= 0:
            base_v = 1.0

        v_ref_list.append(v_ref_mag / base_v)
        v_ctrl_list.append(v_ctrl_mag / base_v)

    v_ref = np.array(v_ref_list)
    v_ctrl = np.array(v_ctrl_list)

    if figsize is None:
        if target == "notebook":
            figsize = (6.5, 5.0)
        else:
            figsize = get_publication_figsize("single", "square")

    fig, ax = plt.subplots(figsize=figsize)

    # Spans (Draw first as background)
    ax.axhspan(0.95, 1.05, color="#edf7ed", label="ANSI C84.1 Range", zorder=0)
    ax.axvspan(0.95, 1.05, color="#edf7ed", zorder=0)

    # Diagonal y=x line
    min_v = min(v_ref.min(), v_ctrl.min(), 0.94)
    max_v = max(v_ref.max(), v_ctrl.max(), 1.06)
    ax.plot([min_v, max_v], [min_v, max_v], color="#5f6368", linestyle="--", label="No Change (y=x)", zorder=2)

    # Scatter points (Draw on top of spans)
    ax.scatter(v_ref, v_ctrl, color="#1a73e8", edgecolors="none", s=50, label="Buses", zorder=3)

    ax.set_xlabel("Reference Voltage (p.u.)")
    ax.set_ylabel("Control Voltage (p.u.)")

    ax.grid(True, linestyle=":", zorder=1)
    if target == "notebook":
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            framealpha=0.95,
        )
    else:
        ax.legend(loc="best")

    return fig


def plot_power_scatter_at_timestep(
    data: dict[str, pd.DataFrame],
    timestep_idx: int = -1,
    timestep_val: Any = None,
    figsize: tuple[float, float] | None = None,
    target: str = "paper",
) -> plt.Figure | None:
    """Generate scatter plots comparing individual bus active and reactive power injections
    at a single timestep.
    """
    if not all(k in data for k in ["feeder_p_real", "feeder_p_imag", "reference_p_real", "reference_p_imag"]):
        logger.warning("Missing power data for power scatter plot. Skipping.")
        return None

    c_p = data["feeder_p_real"]
    c_q = data["feeder_p_imag"]
    r_p = data["reference_p_real"]
    r_q = data["reference_p_imag"]

    time_col = "time" if "time" in c_p.columns else c_p.columns[0]

    c_times = c_p[time_col].unique()
    r_times = r_p[time_col].unique()
    common_times = np.intersect1d(c_times, r_times)

    if len(common_times) == 0:
        logger.warning("No common timestamps found for power scatter plot.")
        return None

    c_p_df = c_p[c_p[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    c_q_df = c_q[c_q[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    r_p_df = r_p[r_p[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    r_q_df = r_q[r_q[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)

    common_cols = [c for c in c_p_df.columns if c in r_p_df.columns]
    if not common_cols:
        logger.warning("No common bus columns found for power scatter plot.")
        return None

    if timestep_val is not None:
        t_val = timestep_val
    else:
        if timestep_idx == -1 or timestep_idx is None:
            max_diff = -1.0
            best_idx = 0
            for idx, t in enumerate(common_times):
                diffs = []
                for col in common_cols:
                    p_ref_val = float(r_p_df.loc[t, col])
                    p_ctrl_val = float(c_p_df.loc[t, col])
                    diffs.append(abs(p_ref_val - p_ctrl_val))
                mean_diff = float(np.mean(diffs))
                if mean_diff > max_diff:
                    max_diff = mean_diff
                    best_idx = idx
            timestep_idx = best_idx
            logger.info(
                f"Selected timestep index {timestep_idx} ({common_times[timestep_idx]}) with maximum mean power injection difference of {max_diff:.3f} kW for the scatter plot."
            )
        t_val = common_times[timestep_idx]

    p_ref_list = []
    p_ctrl_list = []
    q_ref_list = []
    q_ctrl_list = []

    for col in common_cols:
        p_ref_list.append(float(r_p_df.loc[t_val, col]))
        p_ctrl_list.append(float(c_p_df.loc[t_val, col]))
        q_ref_list.append(float(r_q_df.loc[t_val, col]))
        q_ctrl_list.append(float(c_q_df.loc[t_val, col]))

    p_ref = np.array(p_ref_list)
    p_ctrl = np.array(p_ctrl_list)
    q_ref = np.array(q_ref_list)
    q_ctrl = np.array(q_ctrl_list)

    if figsize is None:
        if target == "notebook":
            figsize = (8.5, 4.0)
        else:
            figsize = get_publication_figsize("double", 0.55)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Real Power
    ax1.scatter(p_ref, p_ctrl, color="#ea4335", edgecolors="none", s=40, label="Buses")
    min_p = min(p_ref.min(), p_ctrl.min())
    max_p = max(p_ref.max(), p_ctrl.max())
    ax1.plot([min_p, max_p], [min_p, max_p], color="#5f6368", linestyle="--", label="y=x")
    ax1.set_xlabel("Reference Injection (kW)")
    ax1.set_ylabel("Control Injection (kW)")
    ax1.legend(loc="best")

    # Reactive Power
    ax2.scatter(q_ref, q_ctrl, color="#f9ab00", edgecolors="none", s=40, label="Buses")
    min_q = min(q_ref.min(), q_ctrl.min())
    max_q = max(q_ref.max(), q_ctrl.max())
    ax2.plot([min_q, max_q], [min_q, max_q], color="#5f6368", linestyle="--", label="y=x")
    ax2.set_xlabel("Reference Injection (kVar)")
    ax2.set_ylabel("Control Injection (kVar)")
    ax2.grid(True, linestyle=":")
    if target == "notebook":
        ax2.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            framealpha=0.95,
        )
    else:
        ax2.legend(loc="best")
    fig.subplots_adjust(wspace=0.35)
    return fig


def generate_objective_scorecard(
    data: dict[str, pd.DataFrame],
    topology: Topology,
    area_ids: list[int],
    area_params: dict[int, dict[str, Any]],
    area_buses: list[list[str]],
    flow_data: dict[str, Any],
    voltage_data: dict[int, pd.DataFrame],
    timestep: Any = None,
    figsize: tuple[float, float] | None = None,
    target: str = "paper",
) -> tuple[plt.Figure | None, pd.DataFrame]:
    """Generate an objective performance scorecard table comparing Control vs Reference metrics per area."""
    if not area_ids:
        return None, pd.DataFrame()

    boundary_flows = flow_data.get("boundary_flows", {})
    bus_area_map = {}
    for aid in area_ids:
        if aid < len(area_buses):
            for bus in area_buses[aid]:
                bus_area_map[bus] = aid

    # Map rated DER generation per area from topology
    rated_gen: dict[int, float] = {aid: 0.0 for aid in area_ids}
    real_inj = topology.injections.power_real
    for bus_phase, eq_id, val in zip(real_inj.ids, real_inj.equipment_ids, real_inj.values):
        bus = bus_phase.split(".")[0]
        aid = bus_area_map.get(bus)
        if aid is not None and "pvsystem" in eq_id.lower():
            rated_gen[aid] += float(val)

    # Actual DER generation at timestep if available from der_comparisons
    time_col = "time"
    der_comparisons = flow_data.get("der_comparisons", {})
    der_actual_gen: dict[int, float] = {aid: 0.0 for aid in area_ids}
    for aid in area_ids:
        if aid in der_comparisons and not der_comparisons[aid].empty:
            df_der = der_comparisons[aid]
            t_col = time_col if time_col in df_der.columns else df_der.columns[0]
            if timestep is not None and timestep in df_der[t_col].values:
                der_sub = df_der[df_der[t_col] == timestep]
            else:
                der_sub = df_der[df_der[t_col] == df_der[t_col].max()]
            der_actual_gen[aid] = float(der_sub["p_admm_ctrl"].sum())
        else:
            der_actual_gen[aid] = rated_gen[aid]

    rows = []
    for aid in area_ids:
        label = format_area_label(aid, area_params)
        obj_name = area_params.get(aid, {}).get("objective", "maximize_gen")

        # Boundary flows
        p_ctrl = 0.0
        p_ref = 0.0
        if aid in boundary_flows and not boundary_flows[aid].empty:
            df_b = boundary_flows[aid]
            t_col = time_col if time_col in df_b.columns else df_b.columns[0]
            if timestep is not None and timestep in df_b[t_col].values:
                b_row = df_b[df_b[t_col] == timestep].iloc[0]
            else:
                b_row = df_b.iloc[-1]
            p_ctrl = float(b_row.get("p_control_net_import", 0.0))
            p_ref = float(b_row.get("p_reference_net_import", 0.0))
        delta_p = p_ctrl - p_ref

        # Voltage stats
        v_ctrl_mean = 1.0
        v_ref_mean = 1.0
        v_ctrl_dev = 0.0
        v_ref_dev = 0.0
        if aid in voltage_data and not voltage_data[aid].empty:
            df_v = voltage_data[aid]
            t_col = time_col if time_col in df_v.columns else df_v.columns[0]
            if timestep is not None and timestep in df_v[t_col].values:
                v_sub = df_v[df_v[t_col] == timestep]
            else:
                v_sub = df_v[df_v[t_col] == df_v[t_col].max()]
            if not v_sub.empty:
                v_ctrl_mean = float(v_sub["v_control"].mean())
                v_ctrl_dev = float((v_sub["v_control"] - 1.0).abs().mean())
                if "v_reference" in v_sub and v_sub["v_reference"].notna().any():
                    v_ref_mean = float(v_sub["v_reference"].mean())
                    v_ref_dev = float((v_sub["v_reference"] - 1.0).abs().mean())

        r_gen = rated_gen[aid]
        c_gen = der_actual_gen[aid] if der_actual_gen[aid] > 0 else r_gen
        curtail_pct = max(0.0, (1.0 - c_gen / r_gen) * 100.0) if r_gen > 0 else 0.0

        rows.append(
            {
                "Area": label,
                "Objective": obj_name,
                "Net Import (kW)\nRef / Ctrl (Δ)": f"{p_ref:.1f} / {p_ctrl:.1f} ({delta_p:+.1f})",
                "DER Gen (kW)\nRated / Ctrl": f"{r_gen:.1f} / {c_gen:.1f}",
                "Curtailment\n(%)": f"{curtail_pct:.1f}%",
                "Mean Voltage\nRef / Ctrl (p.u.)": f"{v_ref_mean:.3f} / {v_ctrl_mean:.3f}",
                "Mean Dev |V-1|\nRef / Ctrl (p.u.)": f"{v_ref_dev:.3f} / {v_ctrl_dev:.3f}",
            }
        )

    scorecard_df = pd.DataFrame(rows)

    if figsize is None:
        if target == "notebook":
            figsize = (9.5, 3.2)
        else:
            figsize = get_publication_figsize("double", 0.42)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    col_labels = list(scorecard_df.columns)
    cell_text = scorecard_df.values.tolist()
    col_widths = [0.15, 0.16, 0.21, 0.14, 0.10, 0.12, 0.12]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.7)

    # Style header and alternating row colors
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#37474f")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f8f9fa")
        else:
            cell.set_facecolor("#ffffff")

    return fig, scorecard_df

