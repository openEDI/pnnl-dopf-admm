import os
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

from oedisi.types.data_types import Topology
from .adapter import area_disconnects, disconnect_areas, generate_graph

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


def format_time_val(time_val: Any) -> str:
    """Format time value to HH:MM format."""
    try:
        dt = pd.to_datetime(str(time_val))
        return dt.strftime("%H:%M")
    except Exception:
        return str(time_val)


def configure_publication_style(font_family: str = "serif", base_font_size: float = 9.0) -> None:
    """Set global Matplotlib rcParams for publication-quality figures."""
    import matplotlib as mpl
    mpl.rcParams.update({
        # High baseline resolution for previews/saves
        "figure.dpi": 300,
        "figure.constrained_layout.use": True,
        
        # Typography (Match your journal's body font)
        "font.family": font_family,
        "font.size": base_font_size,                     # Typically 8pt-10pt for journals
        "axes.labelsize": base_font_size,
        "axes.titlesize": base_font_size + 1.0,
        "legend.fontsize": base_font_size - 1.0,
        "xtick.labelsize": base_font_size - 1.0,
        "ytick.labelsize": base_font_size - 1.0,
        
        # Vector Font Export Settings
        "pdf.fonttype": 42,                 # Embeds true fonts into PDF output
        "ps.fonttype": 42,                  # Embeds true fonts into PostScript output
        "text.usetex": False,               # Set True if you have a local LaTeX engine
        
        # Line Weights & Geometries
        "axes.linewidth": 0.5,              # Thin crisp borders
        "lines.linewidth": 1.0,             # Clear data tracking lines
        "lines.markersize": 3.0,            # Legible, uncrowded data markers
        "patch.linewidth": 0.5,
        
        # Ticks Placement
        "xtick.direction": "in",            # Ticks point inward or outward cleanly
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "xtick.major.width": 0.5,
        "ytick.major.size": 3,
        "ytick.major.width": 0.5,
    })


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
            comp_type == "PnnlDopfAdmmComponent"
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
                }
    area_ids.sort()
    return area_ids, area_params


def get_boundary_branch_name(
    G: nx.Graph, source_bus: str, source_line: str | None
) -> str | None:
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
    control_feeder_name = next((name for name in feeder_names if "control" in name.lower() or "local" in name.lower()), None)
    reference_feeder_name = next((name for name in feeder_names if "reference" in name.lower() or "ref" in name.lower()), None)
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
                source.startswith("pnnl_dopf_admm_")
                or source.startswith("area")
                or source.startswith("stats")
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
                logger.warning(
                    f"Required recorder file not found: {filename} (expected for {key})"
                )

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
        logger.error(
            "Control feeder voltage real/imag data missing. Skipping voltage processing."
        )
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


def get_edge_flow(
    u: str, v: str, p_mag_df: pd.DataFrame, p_ang_df: pd.DataFrame, t: Any
) -> float:
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

        p_mag_df = (
            data.get(p_mag_key, pd.DataFrame()).set_index("time")
            if has_boundary_data
            else None
        )
        p_ang_df = (
            data.get(p_ang_key, pd.DataFrame()).set_index("time")
            if has_boundary_data
            else None
        )

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
                cols = [
                    c
                    for c in feeder_p.columns
                    if c != "time" and c.split(".")[0] in buses_in_area_set
                ]
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
    for bus_phase, eq_id, val in zip(
        real_inj.ids, real_inj.equipment_ids, real_inj.values
    ):
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
        records.append(
            {
                "Area": f"Area {aid}",
                "Power Capacity (kW)": rated_gen[aid],
                "Metric": "Rated Generation",
            }
        )
        records.append(
            {
                "Area": f"Area {aid}",
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
    common_times = np.intersect1d(c_times, r_times)

    if len(common_times) == 0:
        if len(c_times) > 0:
            return c_times[-1]
        return None

    c_r_df = c_real[c_real[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    c_i_df = c_imag[c_imag[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    r_r_df = r_real[r_real[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    r_i_df = r_imag[r_imag[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)

    common_cols = [c for c in c_r_df.columns if c in r_r_df.columns]
    if not common_cols:
        return common_times[-1]

    try:
        base_volts_info = topology.base_voltage_magnitudes
        ids = base_volts_info.ids
        values = base_volts_info.values
        base_voltages = dict(zip(ids, values))
    except Exception:
        base_voltages = {}

    max_diff = -1.0
    best_t = common_times[-1]

    for t in common_times:
        diffs = []
        for col in common_cols:
            v_r_ref = r_r_df.loc[t, col]
            v_i_ref = r_i_df.loc[t, col]
            v_ref_mag = (v_r_ref**2 + v_i_ref**2)**0.5

            v_r_ctrl = c_r_df.loc[t, col]
            v_i_ctrl = c_i_df.loc[t, col]
            v_ctrl_mag = (v_r_ctrl**2 + v_i_ctrl**2)**0.5

            base_v = base_voltages.get(col, 1.0)
            if base_v <= 0:
                base_v = 1.0
            diffs.append(abs(v_ref_mag - v_ctrl_mag) / base_v)
        mean_diff = float(np.mean(diffs))
        if mean_diff > max_diff:
            max_diff = mean_diff
            best_t = t

    logger.info(f"Selected consistent comparison timestep: {best_t} (mean voltage diff = {max_diff:.5f} p.u.)")
    return best_t


def plot_voltage_comparison(
    voltage_data: dict[int, pd.DataFrame],
    timestep: Any = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure | None:
    """Generate a split violin plot comparing Reference vs Control feeder voltages per area."""
    import seaborn as sns

    records = []
    for aid, df in voltage_data.items():
        if df.empty:
            continue
        if timestep is not None:
            df_latest = df[df["time"] == timestep]
        else:
            latest_time = df["time"].max()
            df_latest = df[df["time"] == latest_time]

        for _, row in df_latest.iterrows():
            if row.get("v_reference") is not None:
                records.append(
                    {"Voltage (p.u.)": row["v_reference"], "Area": f"Area {aid}", "Case": "Reference"}
                )
            records.append(
                {
                    "Voltage (p.u.)": row["v_control"],
                    "Area": f"Area {aid}",
                    "Case": "Control",
                }
            )

    if not records:
        logger.warning("No voltage data to plot. Skipping plot.")
        return None

    df_volt = pd.DataFrame(records)
    df_volt = df_volt.sort_values(by="Area")

    if figsize is None:
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

    ax.axhline(
        1.05, color="r", linestyle="--", label="Upper Limit (1.05 p.u.)"
    )
    ax.axhline(
        0.95, color="r", linestyle="--", label="Lower Limit (0.95 p.u.)"
    )

    ax.set_xlabel("Control Area")
    ax.set_ylabel("Voltage Magnitude (p.u.)")

    # Custom legend
    legend_elements = [
        mpatches.Patch(color="#b0bec5", label="Reference Feeder"),
        mpatches.Patch(color="#7f7f7f", label="Control Feeder (Colored by Area)"),
    ]
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#DDE3EC",
    )

    return fig


def plot_power_flow_comparison(
    flow_data: dict[str, Any],
    timestep: Any = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure | None:
    """Generate a high-quality grouped bar chart comparing ADMM vs Feeder boundary flows."""
    import seaborn as sns

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

        for _, row in df_latest.iterrows():
            p_control = row.get("p_control_net_import", 0.0)
            p_reference = row.get("p_reference_net_import", 0.0)

            records.append(
                {
                    "Area": f"Area {aid}",
                    "Real Power (kW)": p_control,
                    "Case": "Control",
                }
            )
            records.append(
                {
                    "Area": f"Area {aid}",
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
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#DDE3EC",
    )
    plt.xticks(rotation=45, ha="right")
    return fig



def plot_generation_adequacy(
    adequacy_df: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure | None:
    """Generate a high-quality side-by-side bar chart of Rated Generation vs Rated Load per area."""
    import seaborn as sns

    if adequacy_df.empty:
        return None

    # Sort to match color mapping
    adequacy_df = adequacy_df.sort_values(by="Area")

    if figsize is None:
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
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#DDE3EC",
    )
    return fig


def plot_algorithmic_convergence(
    convergence_data: dict[int, pd.DataFrame],
    figsize: tuple[float, float] | None = None,
) -> plt.Figure | None:
    """Generate a high-quality semi-log plot of ADMM convergence history at each timestep."""
    if not convergence_data:
        logger.warning("No convergence data to plot.")
        return None

    if figsize is None:
        figsize = get_publication_figsize("double", 0.45)

    records = []
    for aid, df in convergence_data.items():
        if df.empty:
            continue
        idx = df.groupby("time")["admm_iteration"].idxmax()
        df_final = df.loc[idx]
        for _, row in df_final.iterrows():
            records.append(
                {
                    "time": format_time_val(row["time"]),
                    "Area": f"Area {aid}",
                    "Optimality Gap": abs(row["optimality_gap"]),
                    "Feasibility Gap": abs(row["feasibility_gap"]),
                }
            )

    if not records:
        logger.warning("No final iteration gaps to plot.")
        return None

    df_plot = pd.DataFrame(records)
    df_plot = df_plot.sort_values(by="time")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True)
    colors = AREA_COLORS

    areas = sorted(df_plot["Area"].unique(), key=lambda x: int(x.split()[-1]))
    
    legend_handles = []
    legend_labels = []

    for area_name in areas:
        df_area = df_plot[df_plot["Area"] == area_name]
        try:
            aid = int(area_name.split()[-1])
        except (ValueError, IndexError):
            aid = 0
        color = colors[aid % len(colors)]
        
        # Plot Optimality Gap (Left)
        line_opt = ax1.semilogy(
            df_area["time"],
            df_area["Optimality Gap"],
            "o-",
            color=color,
            linewidth=1.5,
        )
        
        # Plot Feasibility Gap (Right)
        ax2.semilogy(
            df_area["time"],
            df_area["Feasibility Gap"],
            "s-",
            color=color,
            linewidth=1.5,
        )
        
        legend_handles.append(line_opt[0])
        legend_labels.append(f"Area {aid}")

    # Tolerance lines
    tol_line = ax1.axhline(1e-3, color="gray", linestyle=":")
    ax2.axhline(1e-3, color="gray", linestyle=":")
    
    legend_handles.append(tol_line)
    legend_labels.append("Tolerance")

    ax1.set_title("Optimality Gap")
    ax2.set_title("Feasibility Gap")

    ax1.set_xlabel("Simulation Time (HH:MM)")
    ax2.set_xlabel("Simulation Time (HH:MM)")
    ax1.set_ylabel("Optimality Gap")
    ax2.set_ylabel("Feasibility Gap")

    for ax in [ax1, ax2]:
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(True, which="both", linestyle=":")

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(areas) + 1,
        frameon=True,
        facecolor="white",
        edgecolor="#DDE3EC",
    )
    return fig


def load_coordinates(coords_dir: str | Path) -> dict[str, tuple[float, float]]:
    """Load coordinates from standard OpenDSS files in the coords directory."""
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


def plot_network_partition(
    G: nx.Graph,
    boundaries: list,
    areas_clean: list[nx.Graph],
    slack_bus: str,
    coords_dir: str | Path,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Generate the network partition map showing control areas and boundary switches."""
    if figsize is None:
        figsize = get_publication_figsize("double", "square")

    fig, ax = plt.subplots(figsize=figsize)

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

    colors = AREA_COLORS
    node_colors = [colors[node_to_area.get(node, 0) % len(colors)] for node in G.nodes()]

    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=1.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, ax=ax)

    y_vals = [p[1] for p in pos.values()]
    y_range = max(y_vals) - min(y_vals) if y_vals else 1.0
    offset_y = y_range * 0.02

    for u, v, a in boundaries:
        if u in pos and v in pos:
            mid_x = (pos[u][0] + pos[v][0]) / 2.0
            mid_y = (pos[u][1] + pos[v][1]) / 2.0
            ax.plot(
                mid_x,
                mid_y,
                marker="s",
                color="red",
                markersize=8,
                markeredgecolor="black",
                zorder=5,
            )
            ax.text(
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
            ax=ax,
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

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#DDE3EC",
    )
    ax.axis("off")
    return fig


def plot_voltage_scatter_at_timestep(
    data: dict[str, pd.DataFrame],
    topology: Topology,
    timestep_idx: int = -1,
    timestep_val: Any = None,
    figsize: tuple[float, float] | None = None,
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

    if timestep_val is not None:
        t_val = timestep_val
    else:
        if timestep_idx == -1 or timestep_idx is None:
            max_diff = -1.0
            best_idx = 0
            try:
                base_volts_info = topology.base_voltage_magnitudes
                ids = base_volts_info.ids
                values = base_volts_info.values
                base_voltages = dict(zip(ids, values))
            except Exception:
                base_voltages = {}

            for idx, t in enumerate(common_times):
                diffs = []
                for col in common_cols:
                    v_r_ref = r_r_df.loc[t, col]
                    v_i_ref = r_i_df.loc[t, col]
                    v_ref_mag = (v_r_ref**2 + v_i_ref**2)**0.5
                    v_r_ctrl = c_r_df.loc[t, col]
                    v_i_ctrl = c_i_df.loc[t, col]
                    v_ctrl_mag = (v_r_ctrl**2 + v_i_ctrl**2)**0.5
                    base_v = base_voltages.get(col, 1.0)
                    if base_v <= 0:
                        base_v = 1.0
                    diffs.append(abs(v_ref_mag - v_ctrl_mag) / base_v)
                mean_diff = float(np.mean(diffs))
                if mean_diff > max_diff:
                    max_diff = mean_diff
                    best_idx = idx
            timestep_idx = best_idx
            logger.info(f"Selected timestep index {timestep_idx} ({common_times[timestep_idx]}) with maximum mean voltage difference of {max_diff:.5f} p.u. for the scatter plot.")
        t_val = common_times[timestep_idx]
    
    try:
        # Load base voltages from topology
        base_volts_info = topology.base_voltage_magnitudes
        ids = base_volts_info.ids
        values = base_volts_info.values
        base_voltages = dict(zip(ids, values))
    except Exception as e:
        logger.warning(f"Could not parse topology for base voltages: {e}")
        base_voltages = {}

    v_ref_list = []
    v_ctrl_list = []

    for col in common_cols:
        v_r_ref = r_r_df.loc[t_val, col]
        v_i_ref = r_i_df.loc[t_val, col]
        v_ref_mag = (v_r_ref**2 + v_i_ref**2)**0.5

        v_r_ctrl = c_r_df.loc[t_val, col]
        v_i_ctrl = c_i_df.loc[t_val, col]
        v_ctrl_mag = (v_r_ctrl**2 + v_i_ctrl**2)**0.5

        base_v = base_voltages.get(col, 1.0)
        if base_v <= 0:
            base_v = 1.0

        v_ref_list.append(v_ref_mag / base_v)
        v_ctrl_list.append(v_ctrl_mag / base_v)

    v_ref = np.array(v_ref_list)
    v_ctrl = np.array(v_ctrl_list)

    if figsize is None:
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
    ax.legend(loc="lower right")

    return fig


def plot_power_scatter_at_timestep(
    data: dict[str, pd.DataFrame],
    timestep_idx: int = -1,
    timestep_val: Any = None,
    figsize: tuple[float, float] | None = None,
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
            logger.info(f"Selected timestep index {timestep_idx} ({common_times[timestep_idx]}) with maximum mean power injection difference of {max_diff:.3f} kW for the scatter plot.")
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
        figsize = get_publication_figsize("double", 0.55)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Real Power
    ax1.scatter(p_ref, p_ctrl, color="#ea4335", edgecolors="none", s=40, label="Buses")
    min_p = min(p_ref.min(), p_ctrl.min())
    max_p = max(p_ref.max(), p_ctrl.max())
    ax1.plot([min_p, max_p], [min_p, max_p], color="#5f6368", linestyle="--", label="y=x")
    ax1.set_xlabel("Reference Injection (kW)")
    ax1.set_ylabel("Control Injection (kW)")
    ax1.grid(True, linestyle=":")
    ax1.legend(loc="lower right")

    # Reactive Power
    ax2.scatter(q_ref, q_ctrl, color="#f9ab00", edgecolors="none", s=40, label="Buses")
    min_q = min(q_ref.min(), q_ctrl.min())
    max_q = max(q_ref.max(), q_ctrl.max())
    ax2.plot([min_q, max_q], [min_q, max_q], color="#5f6368", linestyle="--", label="y=x")
    ax2.set_xlabel("Reference Injection (kVar)")
    ax2.set_ylabel("Control Injection (kVar)")
    ax2.grid(True, linestyle=":")
    ax2.legend(loc="lower right")

    return fig
