"""Convert oedisi Topology and live measurements to a distopf Case."""

import logging
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

import distopf as opf
from distopf.api import Case
from oedisi.types.data_types import Injection, Topology, VoltagesMagnitude

from distopf_federate.constants import MIN_GEN_SA_PU, S_BASE

logger = logging.getLogger(__name__)


def _phases_to_str(phases: list) -> str:
    """Convert phase list [1,2,3] → 'abc', [1,0,0] → 'a', [0,2,3] → 'bc', etc."""
    phase_map = {1: "a", 2: "b", 3: "c"}
    return "".join(phase_map[p] for p in phases if p != 0)


def _build_graph(topology: Topology) -> tuple:
    """Build undirected graph then return BFS-directed graph + slack_bus name.

    Returns
    -------
    DG : nx.DiGraph
        Directed graph rooted at slack_bus (edges point away from root).
    slack_bus : str
        Slack bus name (without phase suffix).
    """
    slack_bus = topology.slack_bus[0].split(".", 1)[0]

    G = nx.Graph()
    for src, dst, branch_id in zip(
        topology.incidences.from_equipment,
        topology.incidences.to_equipment,
        topology.incidences.ids,
    ):
        if "OPEN" in src or "OPEN" in dst or src == dst:
            continue

        src_bus = src.split(".", 1)[0] if "." in src else src
        dst_bus = dst.split(".", 1)[0] if "." in dst else dst

        lid = branch_id.lower()
        if ("sw" in lid or "fuse" in lid) and "padswitch" not in lid:
            tag = "SWITCH"
        elif "xfm" in lid or "xfmr" in lid or "tr" in lid:
            tag = "XFMR"
        elif "reg" in lid:
            tag = "REG"
        else:
            tag = "LINE"

        G.add_edge(src_bus, dst_bus, id=branch_id, tag=tag)

    # Keep only the component containing the slack bus
    for component in nx.connected_components(G):
        if slack_bus in component:
            G = G.subgraph(component).copy()
            break

    # Direct edges away from the slack bus
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    for u, v in nx.bfs_edges(G, slack_bus):
        DG.add_edge(u, v, **G.edges[u, v])

    return DG, slack_bus


def _extract_admittances(DG: nx.DiGraph, topology: Topology) -> dict:
    """Build per-branch 3×3 admittance matrices from the sparse admittance.

    Returns
    -------
    dict mapping "{fr_bus}_{to_bus}" → 3×3 complex numpy array (Y values,
    i.e. negated branch admittance off-diagonal nodal entries).
    """
    branch_admittances = {}
    for u, v in DG.edges():
        branch_admittances[f"{u}_{v}"] = np.zeros((3, 3), dtype=complex)

    for src, dst, v in zip(
        topology.admittance.from_equipment,
        topology.admittance.to_equipment,
        topology.admittance.admittance_list,
    ):
        src_bus, src_ph = src.split(".", 1)
        dst_bus, dst_ph = dst.split(".", 1)
        row = int(src_ph) - 1
        col = int(dst_ph) - 1

        if src_bus == dst_bus:
            continue  # skip diagonal (self-admittance)

        key = f"{src_bus}_{dst_bus}"
        rev_key = f"{dst_bus}_{src_bus}"
        if key in branch_admittances:
            branch_admittances[key][row, col] = complex(v[0], v[1])
        elif rev_key in branch_admittances:
            branch_admittances[rev_key][col, row] = complex(v[0], v[1])

    return branch_admittances


def _admittance_to_zprim(y_matrix: np.ndarray) -> np.ndarray:
    """Convert 3×3 nodal off-diagonal admittance to branch impedance.

    The nodal admittance Y_ij = -y_branch, so z_branch = -pinv(Y).

    Returns
    -------
    zprim : ndarray, shape (3, 3, 2)
        zprim[i][j] = [real_ohms, imag_ohms] for each matrix entry.
    """
    if np.allclose(y_matrix, 0):
        return np.zeros((3, 3, 2))
    try:
        z = -np.linalg.pinv(y_matrix)
    except np.linalg.LinAlgError:
        logger.warning("pinv failed for admittance matrix; using zeros")
        return np.zeros((3, 3, 2))

    zprim = np.zeros((3, 3, 2))
    for i in range(3):
        for j in range(3):
            zprim[i, j, 0] = float(z[i, j].real)
            zprim[i, j, 1] = float(z[i, j].imag)
    return zprim


def _active_phases_from_y(y_matrix: np.ndarray) -> list:
    """Determine which phases (1-indexed) are active in a branch admittance.

    A phase is active if any element in that row or column is non-zero.
    """
    phases = [0, 0, 0]
    for ph in range(3):
        if np.any(np.abs(y_matrix[ph, :]) > 1e-12) or np.any(
            np.abs(y_matrix[:, ph]) > 1e-12
        ):
            phases[ph] = ph + 1
    return phases


def _extract_base_voltages(topology: Topology) -> dict:
    """Extract base voltages from topology.

    Returns
    -------
    dict mapping bus_name → (base_kv: float, phases: list[int])
        base_kv is line-to-neutral in kV; phases is 3-element list [1/0, 2/0, 3/0].
    """
    result = {}
    for id_str, voltage in zip(
        topology.base_voltage_magnitudes.ids,
        topology.base_voltage_magnitudes.values,
    ):
        name, phase_str = id_str.split(".", 1)
        phase = int(phase_str) - 1
        if name not in result:
            result[name] = [0.0, [0, 0, 0]]
        if result[name][1][phase] == 0:  # first phase sets the kV
            result[name][0] = voltage / 1000.0  # V → kV
        result[name][1][phase] = phase + 1
    return result


def _parse_injections(real, imag) -> dict:
    """Parse oedisi PowersReal/PowersImaginary into per-bus P/Q arrays.

    Separates PVSystem generation from load injections.

    Parameters
    ----------
    real : PowersReal
        Real-power injection data with ids, equipment_ids, and values in kW.
    imag : PowersImaginary
        Reactive-power injection data with ids, equipment_ids, and values in kVAR.

    Returns
    -------
    dict mapping bus_name → {
        "pq": ndarray shape (3, 2)  — load [P_W, Q_VAR] per phase (positive = consuming),
        "pv": ndarray shape (3, 2)  — PV generation [P_W, Q_VAR] per phase,
        "tags": list[str]           — equipment tag strings for PVSystem entries,
    }
    """
    result: dict = {}

    def _ensure(name: str) -> None:
        if name not in result:
            result[name] = {"pq": np.zeros((3, 2)), "pv": np.zeros((3, 2)), "tags": []}

    for id_str, eq, power in zip(real.ids, real.equipment_ids, real.values):
        name, phase_str = id_str.split(".", 1)
        phase = int(phase_str) - 1
        _ensure(name)
        # kW → W (stored in Watts; divided by S_BASE when written to per-unit DataFrames)
        if "PVSystem" in eq:
            result[name]["pv"][phase, 0] += power * 1000.0
            if eq not in result[name]["tags"]:
                result[name]["tags"].append(eq)
        else:
            result[name]["pq"][phase, 0] -= power * 1000.0  # injection sign → load positive

    for id_str, eq, power in zip(imag.ids, imag.equipment_ids, imag.values):
        name, phase_str = id_str.split(".", 1)
        phase = int(phase_str) - 1
        _ensure(name)
        # kVAR → VAR
        if "PVSystem" in eq:
            result[name]["pv"][phase, 1] += power * 1000.0
        else:
            result[name]["pq"][phase, 1] -= power * 1000.0

    return result


def _extract_base_injections(topology: Topology) -> dict:
    """Extract base PQ loads and PV generation from topology injections.

    Returns
    -------
    dict mapping bus_name → {"pq": ndarray (3,2) in W, "pv": ndarray (3,2) in W, "tags": list}
    """
    return _parse_injections(
        topology.injections.power_real,
        topology.injections.power_imaginary,
    )


def topology_to_case(
    topology: Topology,
    source_bus: str,
) -> tuple:
    """Convert an oedisi Topology to a distopf Case.

    Parameters
    ----------
    topology : Topology
        OEDISI network topology (static data).
    source_bus : str
        Name of the slack/swing bus.

    Returns
    -------
    case : distopf.Case
    name_to_id : dict[str, int]
        Bus name → integer bus ID (1-indexed, BFS order from source_bus).
    v_ln_base_map : dict[str, float]
        Bus name → line-to-neutral base voltage in Volts.
    """
    DG, slack_bus = _build_graph(topology)
    admittances = _extract_admittances(DG, topology)
    base_voltages = _extract_base_voltages(topology)
    base_injections = _extract_base_injections(topology)

    # BFS order is already encoded in DG (edges directed away from slack_bus).
    # Walk nodes in BFS order starting from source_bus and assign 1-indexed IDs.
    bfs_nodes = list(nx.bfs_tree(DG, source_bus).nodes())
    for node in DG.nodes():
        if node not in bfs_nodes:
            bfs_nodes.append(node)
    name_to_id = {name: i + 1 for i, name in enumerate(bfs_nodes)}

    # Bus name → line-to-neutral base voltage in Volts
    v_ln_base_map = {name: info[0] * 1000.0 for name, info in base_voltages.items()}

    # ── bus_data ──────────────────────────────────────────────────────────
    bus_rows = []
    for bus_name in bfs_nodes:
        bus_id = name_to_id[bus_name]
        bus_type = opf.SWING_BUS if bus_name == source_bus else opf.PQ_BUS

        base_kv, phases = base_voltages.get(bus_name, [0.0, [0, 0, 0]])
        v_ln_base = base_kv * 1000.0

        inj = base_injections.get(bus_name, {})
        pq = inj.get("pq", np.zeros((3, 2)))
        pv = inj.get("pv", np.zeros((3, 2)))

        phases_str = _phases_to_str(phases) or "abc"

        bus_rows.append(
            {
                "id": bus_id,
                "name": bus_name,
                "pl_a": max(0.0, pq[0, 0]) / S_BASE,
                "ql_a": max(0.0, pq[0, 1]) / S_BASE,
                "pl_b": max(0.0, pq[1, 0]) / S_BASE,
                "ql_b": max(0.0, pq[1, 1]) / S_BASE,
                "pl_c": max(0.0, pq[2, 0]) / S_BASE,
                "ql_c": max(0.0, pq[2, 1]) / S_BASE,
                "bus_type": bus_type,
                "v_a": 1.0,
                "v_b": 1.0,
                "v_c": 1.0,
                "v_ln_base": v_ln_base if v_ln_base > 0 else 1.0,  # 1.0 V fallback; OPF will warn
                "s_base": S_BASE,
                "v_min": 0.95,
                "v_max": 1.05,
                "phases": phases_str,
                "has_gen": bool(np.any(pv[:, 0] > 0)),
                "has_load": bool(np.any(pq[:, 0] > 0)),
                "has_cap": False,
                "load_shape": "",  # empty → no schedule multiplier applied
            }
        )
    bus_data = pd.DataFrame(bus_rows)

    # ── branch_data ───────────────────────────────────────────────────────
    branch_rows = []
    for u, v in DG.edges():
        edge = DG.edges[u, v]
        branch_id = edge.get("id", f"{u}_{v}")

        key = f"{u}_{v}"
        y_matrix = admittances.get(key, np.zeros((3, 3), dtype=complex))
        zprim = _admittance_to_zprim(y_matrix)

        fr_kv = base_voltages.get(u, [0.0, [0, 0, 0]])[0]
        v_ln_base_fr = fr_kv * 1000.0
        z_base = (v_ln_base_fr**2) / S_BASE if v_ln_base_fr > 1.0 else 1.0

        active_phases = _active_phases_from_y(y_matrix)
        if sum(active_phases) == 0:
            active_phases = base_voltages.get(u, [0.0, [0, 0, 0]])[1][:]
        phases_str = _phases_to_str(active_phases) or "abc"

        branch_rows.append(
            {
                "fb": name_to_id[u],
                "tb": name_to_id[v],
                "name": branch_id,
                "type": edge.get("tag", "LINE").lower(),
                "status": "CLOSED",
                "phases": phases_str,
                "s_base": S_BASE,
                "v_ln_base": v_ln_base_fr,
                "z_base": z_base,
                "raa": zprim[0, 0, 0] / z_base,
                "rab": zprim[0, 1, 0] / z_base,
                "rac": zprim[0, 2, 0] / z_base,
                "rbb": zprim[1, 1, 0] / z_base,
                "rbc": zprim[1, 2, 0] / z_base,
                "rcc": zprim[2, 2, 0] / z_base,
                "xaa": zprim[0, 0, 1] / z_base,
                "xab": zprim[0, 1, 1] / z_base,
                "xac": zprim[0, 2, 1] / z_base,
                "xbb": zprim[1, 1, 1] / z_base,
                "xbc": zprim[1, 2, 1] / z_base,
                "xcc": zprim[2, 2, 1] / z_base,
            }
        )
    branch_data = pd.DataFrame(branch_rows)

    # ── gen_data ──────────────────────────────────────────────────────────
    gen_rows = []
    for bus_name in bfs_nodes:
        inj = base_injections.get(bus_name, {})
        pv = inj.get("pv", np.zeros((3, 2)))
        if not np.any(pv[:, 0] > 0):
            continue

        bus_id = name_to_id[bus_name]
        _, phases = base_voltages.get(bus_name, [0.0, [0, 0, 0]])
        phases_str = _phases_to_str(phases) or "abc"

        total_pv_pu = pv[:, 0].sum() / S_BASE
        # Use MIN_GEN_SA_PU as a floor to keep the OPF well-conditioned even
        # when measured output is near zero.
        sa_max_pu = max(total_pv_pu, MIN_GEN_SA_PU)

        gen_rows.append(
            {
                "id": bus_id,
                "name": bus_name,
                "pa": pv[0, 0] / S_BASE,
                "pb": pv[1, 0] / S_BASE,
                "pc": pv[2, 0] / S_BASE,
                "qa": pv[0, 1] / S_BASE,
                "qb": pv[1, 1] / S_BASE,
                "qc": pv[2, 1] / S_BASE,
                "sa_max": sa_max_pu,
                "sb_max": sa_max_pu,
                "sc_max": sa_max_pu,
                "qa_max": sa_max_pu,
                "qb_max": sa_max_pu,
                "qc_max": sa_max_pu,
                "qa_min": -sa_max_pu,
                "qb_min": -sa_max_pu,
                "qc_min": -sa_max_pu,
                "phases": phases_str,
                "control_variable": opf.CONTROL_PQ,
                "gen_shape": "",  # empty → no schedule multiplier
            }
        )
    gen_data = pd.DataFrame(gen_rows) if gen_rows else None

    # Minimal schedules: single step at t=0 with swing bus voltage = 1.0 pu
    schedules = pd.DataFrame({"time": [0], "v_a": [1.0], "v_b": [1.0], "v_c": [1.0]})

    case = Case(
        branch_data=branch_data,
        bus_data=bus_data,
        gen_data=gen_data,
        schedules=schedules,
        start_step=0,
        n_steps=1,
    )

    return case, name_to_id, v_ln_base_map


def update_case_from_measurements(
    case: Case,
    injection: Injection,
    name_to_id: dict,
    voltages_mag: Optional[VoltagesMagnitude] = None,
) -> Case:
    """Update bus_data loads and gen_data generation from live OEDISI measurements.

    Modifies *case* in-place and returns it for convenience.

    Parameters
    ----------
    case : distopf.Case
        The case to update.
    injection : Injection
        Live injection data (PVSystem = generation; others = loads), values in kW/kVAR.
    name_to_id : dict[str, int]
        Bus name → integer bus ID.
    voltages_mag : VoltagesMagnitude, optional
        Live nodal voltage magnitudes in Volts used to update the swing-bus schedule.

    Returns
    -------
    case : distopf.Case
        Same object, modified in-place.
    """
    parsed = _parse_injections(injection.power_real, injection.power_imaginary)

    # Update bus_data loads (clip to non-negative; distopf does not model load generation)
    for bus_name, data in parsed.items():
        if bus_name not in name_to_id:
            continue
        bus_id = name_to_id[bus_name]
        mask = case.bus_data["id"] == bus_id
        if not mask.any():
            continue
        pq = data["pq"]
        case.bus_data.loc[mask, "pl_a"] = max(0.0, pq[0, 0]) / S_BASE
        case.bus_data.loc[mask, "ql_a"] = max(0.0, pq[0, 1]) / S_BASE
        case.bus_data.loc[mask, "pl_b"] = max(0.0, pq[1, 0]) / S_BASE
        case.bus_data.loc[mask, "ql_b"] = max(0.0, pq[1, 1]) / S_BASE
        case.bus_data.loc[mask, "pl_c"] = max(0.0, pq[2, 0]) / S_BASE
        case.bus_data.loc[mask, "ql_c"] = max(0.0, pq[2, 1]) / S_BASE

    # Update gen_data setpoints and capacity bounds
    if case.gen_data is not None and not case.gen_data.empty:
        for bus_name, data in parsed.items():
            if bus_name not in name_to_id:
                continue
            bus_id = name_to_id[bus_name]
            mask = case.gen_data["id"] == bus_id
            if not mask.any():
                continue
            pv = data["pv"]
            total_p_pu = pv[:, 0].sum() / S_BASE
            sa_max_pu = max(total_p_pu, 0.0)
            case.gen_data.loc[mask, "pa"] = pv[0, 0] / S_BASE
            case.gen_data.loc[mask, "pb"] = pv[1, 0] / S_BASE
            case.gen_data.loc[mask, "pc"] = pv[2, 0] / S_BASE
            case.gen_data.loc[mask, "qa"] = pv[0, 1] / S_BASE
            case.gen_data.loc[mask, "qb"] = pv[1, 1] / S_BASE
            case.gen_data.loc[mask, "qc"] = pv[2, 1] / S_BASE
            case.gen_data.loc[mask, ["sa_max", "sb_max", "sc_max"]] = sa_max_pu
            case.gen_data.loc[mask, ["qa_max", "qb_max", "qc_max"]] = sa_max_pu
            case.gen_data.loc[mask, ["qa_min", "qb_min", "qc_min"]] = -sa_max_pu

    # Update swing bus voltage from live measurements
    if voltages_mag is not None:
        swing_mask = case.bus_data["bus_type"] == opf.SWING_BUS
        if swing_mask.any():
            swing_name = case.bus_data.loc[swing_mask, "name"].iloc[0]
            v_base = case.bus_data.loc[swing_mask, "v_ln_base"].iloc[0]
            if v_base > 0:
                phase_v = {1: None, 2: None, 3: None}
                for id_str, voltage in zip(
                    voltages_mag.ids, voltages_mag.values
                ):
                    name, ph_str = id_str.split(".", 1)
                    if name == swing_name:
                        phase_v[int(ph_str)] = voltage / v_base
                ph_map = {1: "v_a", 2: "v_b", 3: "v_c"}
                for ph, col in ph_map.items():
                    if phase_v[ph] is not None and "time" in case.schedules.columns:
                        case.schedules.loc[:, col] = float(phase_v[ph])

    return case
