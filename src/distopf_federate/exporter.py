"""Convert distopf PowerFlowResult to oedisi measurement types."""

import logging
import math
from typing import Iterator, Optional

import pandas as pd

from oedisi.types.data_types import (
    MeasurementArray,
    PowersAngle,
    PowersImaginary,
    PowersMagnitude,
    PowersReal,
    VoltagesAngle,
    VoltagesMagnitude,
)

from distopf_federate.constants import COMMAND_THRESHOLD_W, PHASE_COLS, S_BASE

logger = logging.getLogger(__name__)


def _iter_bus_phases(df: pd.DataFrame, v_base_map: dict) -> Iterator[tuple]:
    """Yield (bus_name, ph_num, value, v_base) for non-zero, non-NaN phase columns.

    Used by voltage magnitude functions to iterate over per-bus phase values.
    """
    for _, row in df.iterrows():
        bus_name = str(row.get("name", ""))
        v_base = v_base_map.get(bus_name, 1.0)
        for col, ph_num in PHASE_COLS:
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            yield bus_name, ph_num, float(val), v_base


def _iter_branch_pq(result) -> Iterator[tuple]:
    """Yield (fr_name, to_name, ph_str, p_pu, q_pu) for each branch-phase flow.

    Looks up the matching reactive-power row from *result.reactive_power_flows*
    by (fb, tb, t) index.  Skips entries where P is None or NaN.
    """
    p_df: Optional[pd.DataFrame] = getattr(result, "active_power_flows", None)
    q_df: Optional[pd.DataFrame] = getattr(result, "reactive_power_flows", None)
    if p_df is None:
        return

    q_indexed = q_df.set_index(["fb", "tb", "t"]) if (q_df is not None and "fb" in q_df.columns) else None

    for _, p_row in p_df.iterrows():
        fr_name = str(p_row.get("from_name", ""))
        to_name = str(p_row.get("to_name", ""))
        t_val = p_row.get("t", 0)
        fb = p_row.get("fb")
        tb = p_row.get("tb")

        for col, ph_str in PHASE_COLS:
            p_pu = p_row.get(col)
            if p_pu is None or (isinstance(p_pu, float) and math.isnan(p_pu)):
                continue
            q_pu = 0.0
            if q_indexed is not None:
                try:
                    q_pu = float(q_indexed.loc[(fb, tb, t_val), col])
                except (KeyError, TypeError):
                    pass
            yield fr_name, to_name, ph_str, float(p_pu), q_pu


def result_to_voltage_mag(
    result,
    v_ln_base_map: dict,
    time: int,
) -> VoltagesMagnitude:
    """Convert per-unit voltage magnitudes to Volts.

    Parameters
    ----------
    result : PowerFlowResult
    v_ln_base_map : dict[str, float]
        Bus name → line-to-neutral base voltage in Volts.
    time : int
        Simulation timestep.

    Returns
    -------
    VoltagesMagnitude
        ids in format "BUSNAME.1/2/3", values in Volts.  Zero-valued phases omitted.
    """
    ids, values = [], []
    if result.voltages is None:
        return VoltagesMagnitude(ids=ids, values=values, time=time)

    for bus_name, ph_num, v_pu, v_base in _iter_bus_phases(result.voltages, v_ln_base_map):
        if v_pu == 0.0:
            continue
        ids.append(f"{bus_name}.{ph_num}")
        values.append(v_pu * v_base)

    return VoltagesMagnitude(ids=ids, values=values, time=time)


def result_to_voltage_angle(
    result,
    time: int,
) -> VoltagesAngle:
    """Convert voltage angle results to a VoltagesAngle object.

    Parameters
    ----------
    result : PowerFlowResult
    time : int

    Returns
    -------
    VoltagesAngle
        ids in format "BUSNAME.1/2/3", values in degrees (units match distopf output).
    """
    ids, values = [], []
    angle_df: Optional[pd.DataFrame] = getattr(result, "voltage_angles", None)
    if angle_df is None:
        return VoltagesAngle(ids=ids, values=values, time=time)

    # v_base_map not needed for angles — pass empty dict, v_base is unused in the loop
    for bus_name, ph_num, angle, _ in _iter_bus_phases(angle_df, {}):
        ids.append(f"{bus_name}.{ph_num}")
        values.append(angle)

    return VoltagesAngle(ids=ids, values=values, time=time)


def result_to_power_mag(
    result,
    v_ln_base_map: dict,
    time: int,
) -> PowersMagnitude:
    """Convert per-unit branch apparent power flows to VA.

    Parameters
    ----------
    result : PowerFlowResult
    v_ln_base_map : dict[str, float]
        Unused; kept for API consistency with result_to_voltage_mag.
    time : int

    Returns
    -------
    PowersMagnitude
        ids in format "FRBUS_TOBUS.a/b/c", values in VA.
    """
    ids, equipment_ids, values = [], [], []

    for fr_name, to_name, ph_str, p_pu, q_pu in _iter_branch_pq(result):
        branch_key = f"{fr_name}_{to_name}"
        s_mag_va = math.sqrt(p_pu**2 + q_pu**2) * S_BASE
        ids.append(f"{branch_key}.{ph_str}")
        equipment_ids.append(branch_key)
        values.append(s_mag_va)

    return PowersMagnitude(ids=ids, equipment_ids=equipment_ids, values=values, time=time)


def result_to_power_angle(
    result,
    time: int,
) -> PowersAngle:
    """Convert per-unit branch flows to branch power angle (arctan2(Q, P)).

    Parameters
    ----------
    result : PowerFlowResult
    time : int

    Returns
    -------
    PowersAngle
        ids in format "FRBUS_TOBUS.a/b/c", values in radians.
    """
    ids, equipment_ids, values = [], [], []

    for fr_name, to_name, ph_str, p_pu, q_pu in _iter_branch_pq(result):
        branch_key = f"{fr_name}_{to_name}"
        ids.append(f"{branch_key}.{ph_str}")
        equipment_ids.append(branch_key)
        values.append(math.atan2(q_pu, p_pu))

    return PowersAngle(ids=ids, equipment_ids=equipment_ids, values=values, time=time)


def result_to_pub_pqv(
    result,
    boundary_buses: list,
    v_ln_base_map: dict,
    time: int,
) -> tuple:
    """Extract boundary bus voltages and injected powers for inter-federate publication.

    Parameters
    ----------
    result : PowerFlowResult
    boundary_buses : list[str]
        Names of the spatial-decomposition boundary buses to publish.
    v_ln_base_map : dict[str, float]
    time : int

    Returns
    -------
    (pub_p, pub_q, pub_v) : tuple of (PowersReal, PowersImaginary, VoltagesMagnitude)
    """
    v_ids, v_vals = [], []
    p_ids, p_eqids, p_vals = [], [], []
    q_ids, q_eqids, q_vals = [], [], []

    # Boundary voltages
    if result.voltages is not None:
        for bus_name, ph_num, v_pu, v_base in _iter_bus_phases(result.voltages, v_ln_base_map):
            if bus_name not in boundary_buses or v_pu == 0.0:
                continue
            v_ids.append(f"{bus_name}.{ph_num}")
            v_vals.append(v_pu * v_base)

    # Boundary branch power flows (flows into boundary buses)
    p_df: Optional[pd.DataFrame] = getattr(result, "active_power_flows", None)
    q_df: Optional[pd.DataFrame] = getattr(result, "reactive_power_flows", None)

    if p_df is not None:
        p_boundary = p_df.loc[p_df["to_name"].astype(str).isin(boundary_buses)]
        q_lookup: dict = {}
        if q_df is not None:
            q_boundary = q_df.loc[q_df["to_name"].astype(str).isin(boundary_buses)]
            for _, qrow in q_boundary.iterrows():
                key = (str(qrow.get("from_name", "")), str(qrow.get("to_name", "")))
                q_lookup[key] = qrow

        for _, prow in p_boundary.iterrows():
            fr = str(prow.get("from_name", ""))
            to = str(prow.get("to_name", ""))
            branch_key = f"{fr}_{to}"
            qrow = q_lookup.get((fr, to))

            for col, ph_num in PHASE_COLS:
                p_pu = prow.get(col)
                if p_pu is None or (isinstance(p_pu, float) and math.isnan(p_pu)):
                    continue
                flow_id = f"{branch_key}.{ph_num}"
                p_ids.append(flow_id)
                p_eqids.append(branch_key)
                p_vals.append(float(p_pu) * S_BASE)

                q_pu = 0.0
                if qrow is not None:
                    q_raw = qrow.get(col)
                    if q_raw is not None and not (isinstance(q_raw, float) and math.isnan(q_raw)):
                        q_pu = float(q_raw)
                q_ids.append(flow_id)
                q_eqids.append(branch_key)
                q_vals.append(q_pu * S_BASE)

    pub_p = PowersReal(ids=p_ids, equipment_ids=p_eqids, values=p_vals, time=time)
    pub_q = PowersImaginary(ids=q_ids, equipment_ids=q_eqids, values=q_vals, time=time)
    pub_v = VoltagesMagnitude(ids=v_ids, values=v_vals, time=time)

    return pub_p, pub_q, pub_v


def result_to_commands(
    result,
    gen_tags: dict,
    time: int,
) -> list:
    """Convert OPF generator setpoints to a list of (equipment_id, P_W, Q_VAR) tuples.

    This format matches what the OEDISI DER actuator federates expect.

    Parameters
    ----------
    result : PowerFlowResult
    gen_tags : dict[str, list[str]]
        Bus name → list of equipment tag strings (e.g. ["PVSystem.pv1"]).
    time : int

    Returns
    -------
    list of (equipment_id, P_watts, Q_vars) tuples
    """
    commands = []

    p_gen: Optional[pd.DataFrame] = getattr(result, "active_power_generation", None)
    q_gen: Optional[pd.DataFrame] = getattr(result, "reactive_power_generation", None)

    if p_gen is None:
        return commands

    q_lookup: dict = {}
    if q_gen is not None:
        for _, qrow in q_gen.iterrows():
            q_lookup[str(qrow.get("name", ""))] = qrow

    for _, prow in p_gen.iterrows():
        bus_name = str(prow.get("name", ""))
        if bus_name not in gen_tags:
            continue

        p_w = (
            (prow.get("a", 0.0) or 0.0)
            + (prow.get("b", 0.0) or 0.0)
            + (prow.get("c", 0.0) or 0.0)
        ) * S_BASE

        q_var = 0.0
        if bus_name in q_lookup:
            qrow = q_lookup[bus_name]
            q_var = (
                (qrow.get("a", 0.0) or 0.0)
                + (qrow.get("b", 0.0) or 0.0)
                + (qrow.get("c", 0.0) or 0.0)
            ) * S_BASE

        for eq_tag in gen_tags[bus_name]:
            if abs(p_w) < COMMAND_THRESHOLD_W and abs(q_var) < COMMAND_THRESHOLD_W:
                continue
            commands.append((eq_tag, float(p_w), float(q_var)))

    return commands


def result_to_solver_stats(
    converged: bool,
    objective_value: Optional[float],
    iterations: int,
    solve_time: float,
    time: int,
) -> MeasurementArray:
    """Build a MeasurementArray of solver diagnostics.

    Parameters
    ----------
    converged : bool
    objective_value : float or None
    iterations : int
    solve_time : float
        Wall-clock solve time in seconds.
    time : int

    Returns
    -------
    MeasurementArray
    """
    stats = {
        "converged": float(converged),
        "objective_value": float(objective_value) if objective_value is not None else float("nan"),
        "iterations": float(iterations),
        "solve_time": float(solve_time),
    }
    return MeasurementArray(
        ids=list(stats.keys()),
        values=list(stats.values()),
        time=time,
        units="mixed",
    )
