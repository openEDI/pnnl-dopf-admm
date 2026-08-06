"""Unit tests for distopf_federate.exporter."""

import math
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oedisi.types.data_types import MeasurementArray
from distopf_federate.exporter import (
    enapp_s_up_to_pq,
    enapp_v_dn_to_vmag,
    result_to_commands,
    result_to_power_angle,
    result_to_power_mag,
    result_to_pub_pqv,
    result_to_solver_stats,
    result_to_voltage_angle,
    result_to_voltage_mag,
)


# ---------------------------------------------------------------------------
# Minimal PowerFlowResult stub so tests have no distopf/HELICS dependency
# ---------------------------------------------------------------------------


class _FakeResult:
    """Minimal stub of distopf.PowerFlowResult."""

    def __init__(
        self,
        voltages=None,
        voltage_angles=None,
        active_power_flows=None,
        reactive_power_flows=None,
        active_power_generation=None,
        reactive_power_generation=None,
        converged=True,
        objective_value=None,
    ):
        self.voltages = voltages
        self.voltage_angles = voltage_angles
        self.active_power_flows = active_power_flows
        self.reactive_power_flows = reactive_power_flows
        self.active_power_generation = active_power_generation
        self.reactive_power_generation = reactive_power_generation
        self.converged = converged
        self.objective_value = objective_value


def _make_voltage_df(buses=("bus1", "bus2"), v_a=1.05, v_b=1.04, v_c=1.03):
    """Return a minimal voltages DataFrame."""
    rows = []
    for i, name in enumerate(buses):
        rows.append({"id": i + 1, "name": name, "t": 0, "a": v_a, "b": v_b, "c": v_c})
    return pd.DataFrame(rows)


def _make_flow_df(from_name="bus1", to_name="bus2", fb=1, tb=2, p=0.05, q=0.02):
    return pd.DataFrame(
        [{"from_name": from_name, "to_name": to_name, "fb": fb, "tb": tb, "t": 0,
          "a": p, "b": p, "c": p}]
    )


# ---------------------------------------------------------------------------
# result_to_voltage_mag
# ---------------------------------------------------------------------------


def test_result_to_voltage_mag_basic():
    result = _FakeResult(voltages=_make_voltage_df(["bus1"], v_a=1.05, v_b=0.0, v_c=0.0))
    v_base_map = {"bus1": 2401.77}

    vmag = result_to_voltage_mag(result, v_base_map, time=0)

    assert vmag.time.timestamp() == 0
    # Only phase a should appear (b and c are zero)
    assert "bus1.1" in vmag.ids
    assert len(vmag.ids) == 1
    assert abs(vmag.values[0] - 1.05 * 2401.77) < 0.01


def test_result_to_voltage_mag_three_phases():
    result = _FakeResult(voltages=_make_voltage_df(["busA"], v_a=1.0, v_b=1.0, v_c=1.0))
    v_base_map = {"busA": 1000.0}

    vmag = result_to_voltage_mag(result, v_base_map, time=5)

    assert vmag.time.timestamp() == 5
    assert len(vmag.ids) == 3
    for ph_num in [1, 2, 3]:
        assert f"busA.{ph_num}" in vmag.ids
    assert all(abs(v - 1000.0) < 1e-6 for v in vmag.values)


def test_result_to_voltage_mag_empty_result():
    result = _FakeResult(voltages=None)
    vmag = result_to_voltage_mag(result, {}, time=0)
    assert vmag.ids == []
    assert vmag.values == []


def test_result_to_voltage_mag_nan_skipped():
    df = pd.DataFrame([{"id": 1, "name": "bus1", "t": 0,
                        "a": float("nan"), "b": 1.0, "c": float("nan")}])
    result = _FakeResult(voltages=df)
    vmag = result_to_voltage_mag(result, {"bus1": 1000.0}, time=0)
    assert len(vmag.ids) == 1
    assert "bus1.2" in vmag.ids


# ---------------------------------------------------------------------------
# result_to_voltage_angle
# ---------------------------------------------------------------------------


def test_result_to_voltage_angle_basic():
    df = pd.DataFrame([{"id": 1, "name": "bus1", "t": 0, "a": 0.1, "b": -0.1, "c": 0.0}])
    result = _FakeResult(voltage_angles=df)
    vang = result_to_voltage_angle(result, time=3)

    assert vang.time.timestamp() == 3
    assert "bus1.1" in vang.ids
    idx = vang.ids.index("bus1.1")
    assert abs(vang.values[idx] - 0.1) < 1e-9


def test_result_to_voltage_angle_none():
    result = _FakeResult()
    vang = result_to_voltage_angle(result, time=0)
    assert vang.ids == []


# ---------------------------------------------------------------------------
# result_to_power_mag
# ---------------------------------------------------------------------------


def test_result_to_power_mag_basic():
    p_df = _make_flow_df(p=0.1)
    q_df = _make_flow_df(p=0.05)
    result = _FakeResult(active_power_flows=p_df, reactive_power_flows=q_df)

    pmag = result_to_power_mag(result, {"bus1": 1000.0, "bus2": 1000.0}, time=0)

    assert len(pmag.ids) > 0
    # |S| = sqrt(P^2 + Q^2) * S_BASE (all three phases with p=0.1, q=0.05)
    expected = math.sqrt(0.1**2 + 0.05**2) * 1e6
    for v in pmag.values:
        assert abs(v - expected) < 1.0


def test_result_to_power_mag_no_flows():
    result = _FakeResult()
    pmag = result_to_power_mag(result, {}, time=0)
    assert pmag.ids == []
    assert pmag.values == []


# ---------------------------------------------------------------------------
# result_to_power_angle
# ---------------------------------------------------------------------------


def test_result_to_power_angle_basic():
    p_df = _make_flow_df(p=1.0)
    q_df = _make_flow_df(p=0.0)
    result = _FakeResult(active_power_flows=p_df, reactive_power_flows=q_df)

    pang = result_to_power_angle(result, time=0)
    # atan2(0, 1) = 0
    assert all(abs(v - 0.0) < 1e-9 for v in pang.values)


# ---------------------------------------------------------------------------
# result_to_pub_pqv
# ---------------------------------------------------------------------------


def test_result_to_pub_pqv_filters_to_boundary():
    p_df = _make_flow_df(from_name="bus1", to_name="bus2")
    q_df = _make_flow_df(from_name="bus1", to_name="bus2", p=0.01)
    v_df = _make_voltage_df(["bus1", "bus2"], v_a=1.0, v_b=1.0, v_c=1.0)
    result = _FakeResult(voltages=v_df, active_power_flows=p_df, reactive_power_flows=q_df)

    pub_p, pub_q, pub_v = result_to_pub_pqv(
        result,
        boundary_buses=["bus2"],
        v_ln_base_map={"bus1": 1000.0, "bus2": 1000.0},
        time=0,
    )

    # Voltages: only bus2 entries
    assert all("bus2" in id_str for id_str in pub_v.ids)
    # Powers: only flows TO bus2
    assert all("bus2" in id_str for id_str in pub_p.ids)


def test_result_to_pub_pqv_no_boundary():
    result = _FakeResult()
    pub_p, pub_q, pub_v = result_to_pub_pqv(result, [], {}, time=0)
    assert pub_p.ids == []
    assert pub_v.ids == []


# ---------------------------------------------------------------------------
# result_to_commands
# ---------------------------------------------------------------------------


def test_result_to_commands_emits_for_known_bus():
    p_gen = pd.DataFrame([{"id": 1, "name": "bus3", "t": 0, "a": 0.05, "b": 0.0, "c": 0.0}])
    q_gen = pd.DataFrame([{"id": 1, "name": "bus3", "t": 0, "a": 0.01, "b": 0.0, "c": 0.0}])
    result = _FakeResult(active_power_generation=p_gen, reactive_power_generation=q_gen)

    gen_tags = {"bus3": ["PVSystem.pv3"]}
    commands = result_to_commands(result, gen_tags, time=0)

    assert len(commands) == 1
    eq_id, p_w, q_var = commands[0]
    assert eq_id == "PVSystem.pv3"
    assert abs(p_w - 0.05 * 1e6) < 1.0
    assert abs(q_var - 0.01 * 1e6) < 1.0


def test_result_to_commands_skips_zero_setpoints():
    p_gen = pd.DataFrame([{"id": 1, "name": "bus3", "t": 0, "a": 0.0, "b": 0.0, "c": 0.0}])
    result = _FakeResult(active_power_generation=p_gen)
    gen_tags = {"bus3": ["PVSystem.pv3"]}
    commands = result_to_commands(result, gen_tags, time=0)
    assert commands == []


def test_result_to_commands_unknown_bus_ignored():
    p_gen = pd.DataFrame([{"id": 99, "name": "ghostbus", "t": 0, "a": 0.5, "b": 0.0, "c": 0.0}])
    result = _FakeResult(active_power_generation=p_gen)
    gen_tags = {"bus3": ["PVSystem.pv3"]}
    commands = result_to_commands(result, gen_tags, time=0)
    assert commands == []


def test_result_to_commands_no_gen_data():
    result = _FakeResult()
    commands = result_to_commands(result, {"bus3": ["PVSystem.pv3"]}, time=0)
    assert commands == []


# ---------------------------------------------------------------------------
# result_to_solver_stats
# ---------------------------------------------------------------------------


def test_result_to_solver_stats_converged():
    stats = result_to_solver_stats(True, 1.23, 10, 0.5, time=7)
    assert stats.time.timestamp() == 7
    assert "converged" in stats.ids
    idx = stats.ids.index("converged")
    assert stats.values[idx] == 1.0


def test_result_to_solver_stats_not_converged():
    stats = result_to_solver_stats(False, None, 50, 2.1, time=1)
    idx = stats.ids.index("converged")
    assert stats.values[idx] == 0.0
    idx_obj = stats.ids.index("objective_value")
    assert stats.values[idx_obj] == 0.0


def test_result_to_solver_stats_has_all_keys():
    stats = result_to_solver_stats(True, 0.5, 5, 0.1, time=0)
    expected = {
        "converged",
        "objective_value",
        "iterations",
        "num_iters",
        "solve_time",
        "optimality_gap",
        "feasibility_gap",
        "admm_iteration",
        "vup",
        "sdn",
    }
    assert set(stats.ids) == expected


def test_result_to_solver_stats_pydantic_roundtrip():
    stats = result_to_solver_stats(False, None, 50, 2.1, time=1, admm_iteration=3, vup=0.01)
    json_str = stats.model_dump_json()
    deserialized = MeasurementArray.model_validate_json(json_str)
    assert None not in deserialized.values
    assert deserialized.time.timestamp() == 1
    assert deserialized.values[deserialized.ids.index("objective_value")] == 0.0
    assert deserialized.values[deserialized.ids.index("admm_iteration")] == 3.0
    assert deserialized.values[deserialized.ids.index("vup")] == 0.01




# ---------------------------------------------------------------------------
# ENAPP boundary encoding helpers
# ---------------------------------------------------------------------------

def _make_s_up_df(name="area_150", p_a=0.5, p_b=0.4, p_c=0.3, q_a=0.1, q_b=0.08, q_c=0.06):
    """Return a minimal parse_s_up-style DataFrame."""
    return pd.DataFrame([{
        "name": name, "t": 0,
        "a": complex(p_a, q_a),
        "b": complex(p_b, q_b),
        "c": complex(p_c, q_c),
    }])


def _make_v_dn_df(area_names=("area_152",), v=1.02):
    """Return a minimal parse_v_dn-style DataFrame."""
    rows = [{"name": n, "t": 0, "a": v, "b": v, "c": v} for n in area_names]
    return pd.DataFrame(rows)


class _FakeCase:
    pass


class _FakeResultEnapp:
    pass


@patch("distopf.distributed.spatial.enapp.parse_s_up")
def test_enapp_s_up_to_pq_encodes_area_name(mock_parse_s_up):
    """S_up is published with the area's own name as bus ID, not the SWING bus name."""
    s_up_df = _make_s_up_df(name="area_150", p_a=0.5, q_a=0.1)
    mock_parse_s_up.return_value = s_up_df

    from distopf_federate.constants import S_BASE
    pub_p, pub_q = enapp_s_up_to_pq(_FakeCase(), _FakeResultEnapp(), "area_152", time=0)

    # IDs should use the publishing area name, not the SWING bus name "area_150"
    assert all(id_.startswith("area_152.") for id_ in pub_p.ids), (
        f"Expected ids to start with 'area_152.', got {pub_p.ids}"
    )
    # Values should be in Watts (per-unit × S_BASE)
    assert any(abs(v) > 0 for v in pub_p.values), "Expected non-zero P values"
    # P and Q should have matching IDs
    assert set(pub_p.ids) == set(pub_q.ids)


@patch("distopf.distributed.spatial.enapp.parse_s_up")
def test_enapp_s_up_to_pq_empty_result(mock_parse_s_up):
    """Empty parse_s_up result produces empty PowersReal/Imaginary."""
    mock_parse_s_up.return_value = pd.DataFrame(columns=["name", "t", "a", "b", "c"])

    pub_p, pub_q = enapp_s_up_to_pq(_FakeCase(), _FakeResultEnapp(), "area_152", time=5)

    assert pub_p.ids == []
    assert pub_q.ids == []
    assert pub_p.values == []


@patch("distopf.distributed.spatial.enapp.parse_v_dn")
def test_enapp_v_dn_to_vmag_encodes_child_names(mock_parse_v_dn):
    """V_dn is published with child area names as bus IDs."""
    v_df = _make_v_dn_df(area_names=("area_152", "area_135"), v=1.03)
    mock_parse_v_dn.return_value = v_df

    vmag = enapp_v_dn_to_vmag(
        _FakeCase(), _FakeResultEnapp(),
        down_buses=["area_152", "area_135"],
        time=0,
    )

    assert any(id_.startswith("area_152.") for id_ in vmag.ids)
    assert any(id_.startswith("area_135.") for id_ in vmag.ids)
    assert all(abs(v - 1.03) < 1e-9 for v in vmag.values)


@patch("distopf.distributed.spatial.enapp.parse_v_dn")
def test_enapp_v_dn_to_vmag_no_down_buses(mock_parse_v_dn):
    """Empty down_buses returns an empty VoltagesMagnitude."""
    mock_parse_v_dn.return_value = pd.DataFrame(columns=["name", "t", "a", "b", "c"])

    vmag = enapp_v_dn_to_vmag(_FakeCase(), _FakeResultEnapp(), down_buses=[], time=3)

    assert vmag.ids == []
    assert vmag.values == []
