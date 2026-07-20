import os
import json
import logging
import math
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
        if comp_name.startswith("pnnl_dopf_admm_"):
            try:
                area_id = int(comp_name.split("_")[-1])
                area_ids.append(area_id)
                params = comp.get("parameters", {})
                area_params[area_id] = {
                    "source_bus": params.get("source_bus"),
                    "source_line": params.get("source_line"),
                    "switches": params.get("switches", []),
                }
            except ValueError:
                continue
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
            elif source and source.startswith("pnnl_dopf_admm_"):
                try:
                    aid = int(source.split("_")[-1])
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
                except ValueError:
                    pass

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

        source_bus = params.get("source_bus", "")
        if not source_bus:
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

        # Find columns corresponding to source_bus
        bus_cols = [c for c in feeder_p.columns if c.startswith(f"{source_bus}.")]
        if not bus_cols:
            continue

        boundary_records = []
        for col in bus_cols:
            phase = col.split(".")[-1]
            for t in common_times:
                p_ctrl_val = float(feeder_p.loc[t, col])
                q_ctrl_val = float(feeder_q.loc[t, col])

                p_ref_val = 0.0
                q_ref_val = 0.0
                if ref_p is not None and col in ref_p.columns:
                    p_ref_val = float(ref_p.loc[t, col])
                if ref_q is not None and col in ref_q.columns:
                    q_ref_val = float(ref_q.loc[t, col])

                boundary_records.append(
                    {
                        "time": t,
                        "phase": phase,
                        "p_control_net_import": -p_ctrl_val,
                        "q_control_net_import": -q_ctrl_val,
                        "p_reference_net_import": -p_ref_val,
                        "q_reference_net_import": -q_ref_val,
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


def plot_voltage_comparison(
    voltage_data: dict[int, pd.DataFrame]
) -> plt.Figure | None:
    """Generate a split violin plot comparing Reference vs Control feeder voltages per area."""
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    records = []
    for aid, df in voltage_data.items():
        if df.empty:
            continue
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

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=df_volt,
        x="Area",
        y="Voltage (p.u.)",
        hue="Case",
        split=True,
        inner="quart",
        ax=ax,
        palette="muted",
    )
    ax.axhline(
        1.05, color="r", linestyle="--", alpha=0.6, label="Upper Limit (1.05 p.u.)"
    )
    ax.axhline(
        0.95, color="r", linestyle="--", alpha=0.6, label="Lower Limit (0.95 p.u.)"
    )

    ax.set_title(
        "Bus Voltage Profile Distribution per Area (Reference vs Control)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Control Area", fontsize=11)
    ax.set_ylabel("Voltage Magnitude (p.u.)", fontsize=11)
    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig


def plot_power_flow_comparison(flow_data: dict[str, Any]) -> plt.Figure | None:
    """Generate a high-quality grouped bar chart comparing ADMM vs Feeder boundary flows."""
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    boundary_flows = flow_data["boundary_flows"]
    if not boundary_flows:
        logger.warning("No boundary flow data to plot.")
        return None

    records = []
    for aid, df in boundary_flows.items():
        if df.empty:
            continue
        latest_time = df["time"].max()
        df_latest = df[df["time"] == latest_time]

        for _, row in df_latest.iterrows():
            phase = row["phase"]
            p_control = row.get("p_control_net_import", 0.0)
            p_reference = row.get("p_reference_net_import", 0.0)

            label = f"Area {aid} P{phase}"

            records.append(
                {
                    "Boundary Line": label,
                    "Real Power (kW)": p_control,
                    "Case": "Control",
                }
            )
            records.append(
                {
                    "Boundary Line": label,
                    "Real Power (kW)": p_reference,
                    "Case": "Reference",
                }
            )

    if not records:
        logger.warning("No boundary flow records to plot.")
        return None

    df_plot = pd.DataFrame(records)
    df_plot["Abs_Power"] = df_plot["Real Power (kW)"].abs()
    ref_powers = df_plot[df_plot["Case"] == "Reference"]
    sorted_labels = ref_powers.sort_values(by="Abs_Power", ascending=False)[
        "Boundary Line"
    ].unique()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df_plot,
        x="Boundary Line",
        y="Real Power (kW)",
        hue="Case",
        ax=ax,
        palette="Set2",
        order=sorted_labels,
    )
    ax.set_title(
        "Boundary Power Flow Comparison: Reference vs. Control",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Area Boundary Connection & Phase", fontsize=11)
    ax.set_ylabel("Real Power Exchange (kW)", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig



def plot_generation_adequacy(adequacy_df: pd.DataFrame) -> plt.Figure | None:
    """Generate a high-quality side-by-side bar chart of Rated Generation vs Rated Load per area."""
    import seaborn as sns

    if adequacy_df.empty:
        return None

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=adequacy_df,
        x="Area",
        y="Power Capacity (kW)",
        hue="Metric",
        ax=ax,
        palette="Set1",
    )
    ax.set_ylabel("Power Capacity (kW)", fontsize=11)
    ax.set_xlabel("Control Area", fontsize=11)
    ax.set_title(
        "Generation Adequacy: Rated Capacity vs. Rated Load",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_algorithmic_convergence(
    convergence_data: dict[int, pd.DataFrame]
) -> plt.Figure | None:
    """Generate a high-quality semi-log plot of ADMM convergence history at each timestep."""
    if not convergence_data:
        logger.warning("No convergence data to plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    records = []
    for aid, df in convergence_data.items():
        if df.empty:
            continue
        idx = df.groupby("time")["admm_iteration"].idxmax()
        df_final = df.loc[idx]
        for _, row in df_final.iterrows():
            records.append(
                {
                    "time": str(row["time"]),
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

    import seaborn as sns

    sns.set_theme(style="whitegrid")

    areas = df_plot["Area"].unique()
    for idx, area_name in enumerate(areas):
        df_area = df_plot[df_plot["Area"] == area_name]
        color = colors[idx % len(colors)]
        ax.semilogy(
            df_area["time"],
            df_area["Optimality Gap"],
            "o-",
            label=f"{area_name} Optimality Gap",
            color=color,
            linewidth=2,
        )
        ax.semilogy(
            df_area["time"],
            df_area["Feasibility Gap"],
            "s--",
            label=f"{area_name} Feasibility Gap",
            color=color,
            alpha=0.7,
            linewidth=2,
        )

    ax.axhline(
        1e-3, color="gray", linestyle=":", label=r"Tolerance ($\epsilon = 10^{-3}$)"
    )
    ax.set_title(
        "Decentralized Algorithm Convergence Profile over Timesteps",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Simulation Timestep", fontsize=11)
    ax.set_ylabel("Final Residual Error (Log Scale)", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
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
) -> plt.Figure:
    """Generate the network partition map showing control areas and boundary switches."""
    fig, ax = plt.subplots(figsize=(10, 8))

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

    ax.legend(handles=legend_elements, loc="best", fontsize=9, framealpha=0.9)
    ax.set_title("Distribution Grid ADMM Area Partition", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_voltage_scatter_at_timestep(
    data: dict[str, pd.DataFrame],
    topology: Topology,
    output_path: Path,
    timestep_idx: int = -1,
) -> None:
    """Generate a scatter plot comparing individual bus voltage magnitudes (control vs reference)
    at a single timestep.
    """
    if not all(k in data for k in ["feeder_v_real", "feeder_v_imag", "reference_v_real", "reference_v_imag"]):
        logger.warning("Missing voltage data for voltage scatter plot. Skipping.")
        return

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
        return

    c_r_df = c_real[c_real[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    c_i_df = c_imag[c_imag[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    r_r_df = r_real[r_real[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    r_i_df = r_imag[r_imag[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)

    # Compute magnitudes
    common_cols = [c for c in c_r_df.columns if c in r_r_df.columns]
    if not common_cols:
        logger.warning("No common bus columns found for voltage scatter plot.")
        return

    t_val = common_times[timestep_idx]
    
    try:
        # Load base voltages from topology
        base_volts_info = topology.base_voltage_magnitudes
        ids = base_volts_info.ids
        values = base_volts_info.values
        base_voltages = dict(zip(ids, values, strict=True))
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

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(v_ref, v_ctrl, color="#1a73e8", alpha=0.7, edgecolors="none", s=50, label="Buses")
    
    min_v = min(v_ref.min(), v_ctrl.min(), 0.94)
    max_v = max(v_ref.max(), v_ctrl.max(), 1.06)
    ax.plot([min_v, max_v], [min_v, max_v], color="#5f6368", linestyle="--", alpha=0.7, label="No Change (y=x)")
    
    ax.axhspan(0.95, 1.05, color="#34a853", alpha=0.08, label="ANSI C84.1 Range")
    ax.axvspan(0.95, 1.05, color="#34a853", alpha=0.08)
    
    ax.set_xlabel("Reference Voltage (p.u.)", fontsize=11)
    ax.set_ylabel("Control Voltage (p.u.)", fontsize=11)
    
    try:
        t_str = pd.to_datetime(str(t_val)).strftime("%H:%M")
    except Exception:
        t_str = str(t_val)
        
    ax.set_title(f"Individual Bus Voltages at Timestep {t_str}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved voltage scatter plot to: {output_path}")


def plot_power_scatter_at_timestep(
    data: dict[str, pd.DataFrame],
    output_path: Path,
    timestep_idx: int = -1,
) -> None:
    """Generate scatter plots comparing individual bus active and reactive power injections
    at a single timestep.
    """
    if not all(k in data for k in ["feeder_p_real", "feeder_p_imag", "reference_p_real", "reference_p_imag"]):
        logger.warning("Missing power data for power scatter plot. Skipping.")
        return

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
        return

    c_p_df = c_p[c_p[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    c_q_df = c_q[c_q[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    r_p_df = r_p[r_p[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)
    r_q_df = r_q[r_q[time_col].isin(common_times)].sort_values(by=time_col).set_index(time_col)

    common_cols = [c for c in c_p_df.columns if c in r_p_df.columns]
    if not common_cols:
        logger.warning("No common bus columns found for power scatter plot.")
        return

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Real Power
    ax1.scatter(p_ref, p_ctrl, color="#ea4335", alpha=0.7, edgecolors="none", s=40, label="Buses")
    min_p = min(p_ref.min(), p_ctrl.min())
    max_p = max(p_ref.max(), p_ctrl.max())
    ax1.plot([min_p, max_p], [min_p, max_p], color="#5f6368", linestyle="--", alpha=0.7, label="y=x")
    ax1.set_xlabel("Reference Injection (kW)", fontsize=11)
    ax1.set_ylabel("Control Injection (kW)", fontsize=11)
    ax1.set_title("Real Power Injection Comparison", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Reactive Power
    ax2.scatter(q_ref, q_ctrl, color="#f9ab00", alpha=0.7, edgecolors="none", s=40, label="Buses")
    min_q = min(q_ref.min(), q_ctrl.min())
    max_q = max(q_ref.max(), q_ctrl.max())
    ax2.plot([min_q, max_q], [min_q, max_q], color="#5f6368", linestyle="--", alpha=0.7, label="y=x")
    ax2.set_xlabel("Reference Injection (kVar)", fontsize=11)
    ax2.set_ylabel("Control Injection (kVar)", fontsize=11)
    ax2.set_title("Reactive Power Injection Comparison", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    try:
        t_str = pd.to_datetime(str(t_val)).strftime("%H:%M")
    except Exception:
        t_str = str(t_val)

    plt.suptitle(f"Individual Bus Power Comparison at Timestep {t_str}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved power scatter plot to: {output_path}")
