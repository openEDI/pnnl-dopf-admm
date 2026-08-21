"""Unit tests for ENAPP boundary helpers in distopf_federate.importer and federate.

These tests mock ``distopf`` so that the suite runs without an installed
distopf package, matching the approach used in ``test_importer.py``.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from oedisi.types.data_types import PowersImaginary, PowersReal, VoltagesMagnitude

from distopf_federate.constants import S_BASE
from distopf_federate.federate import DistopfFederate
from distopf_federate.importer import apply_s_up_to_sub_case, apply_v_dn_to_sub_case


def test_resolve_source_bus_direct_bus_name():
    """If static.source_bus is already a bus name, return it unchanged."""
    fed = object.__new__(DistopfFederate)
    fed.static = MagicMock()
    fed.static.source_bus = "150"

    topology = _make_mock_topology(
        bus_names=["150", "13", "18"],
        switch_incidences=[("150", "13", "sw2"), ("150", "18", "sw3")],
    )

    result = fed._resolve_source_bus(topology)
    assert result == "150"


def test_resolve_source_bus_switch_id_resolves_to_downstream_bus():
    """If static.source_bus is a switch ID, return the downstream bus."""
    fed = object.__new__(DistopfFederate)
    fed.static = MagicMock()
    fed.static.source_bus = "sw3"

    topology = _make_mock_topology(
        bus_names=["150", "13", "18"],
        switch_incidences=[("150", "13", "sw2"), ("150", "18", "sw3")],
    )

    result = fed._resolve_source_bus(topology)
    assert result == "18"


def test_resolve_source_bus_unknown_falls_back_to_verbatim():
    """If source cannot be matched, it is returned verbatim with a warning."""
    fed = object.__new__(DistopfFederate)
    fed.static = MagicMock()
    fed.static.source_bus = "unknown_eq"

    topology = _make_mock_topology(
        bus_names=["150", "13"],
        switch_incidences=[("150", "13", "sw2")],
    )

    result = fed._resolve_source_bus(topology)
    assert result == "unknown_eq"


# ---------------------------------------------------------------------------
# Tests for apply_v_dn_to_sub_case
# ---------------------------------------------------------------------------


def _make_schedules():
    return pd.DataFrame({"time": [0], "v_a": [1.0], "v_b": [1.0], "v_c": [1.0]})


class _FakeSubCase:
    """Minimal stand-in for distopf.Case with a schedules attribute."""

    def __init__(self):
        self.schedules = _make_schedules()


def test_apply_v_dn_no_matching_area_is_noop():
    """If no entry for area_name exists in vmag, schedules are unchanged."""
    sub_case = _FakeSubCase()
    vmag = VoltagesMagnitude(ids=["area_999.a", "area_999.b"], values=[1.05, 1.05], time=0)
    before = sub_case.schedules.copy()
    apply_v_dn_to_sub_case(sub_case, vmag, "area_152")
    pd.testing.assert_frame_equal(sub_case.schedules, before)


def test_apply_v_dn_calls_add_v_swing_to_schedules():
    """apply_v_dn_to_sub_case calls distopf's add_v_swing_to_schedules with parsed voltage."""
    sub_case = _FakeSubCase()
    vmag = VoltagesMagnitude(
        ids=["area_152.a", "area_152.b", "area_152.c"],
        values=[1.03, 1.02, 1.04],
        time=0,
    )
    with patch("distopf.distributed.spatial.enapp.add_v_swing_to_schedules") as add_v_fn:
        apply_v_dn_to_sub_case(sub_case, vmag, "area_152")
        add_v_fn.assert_called_once()
        _call_args = add_v_fn.call_args
        v_df = _call_args[0][1]  # second positional arg is the v DataFrame
        assert v_df.iloc[0]["a"] == pytest.approx(1.03)
        assert v_df.iloc[0]["b"] == pytest.approx(1.02)
        assert v_df.iloc[0]["c"] == pytest.approx(1.04)
        assert v_df.iloc[0]["name"] == "area_152"


# ---------------------------------------------------------------------------
# Tests for apply_s_up_to_sub_case
# ---------------------------------------------------------------------------


def test_apply_s_up_no_child_areas_is_noop():
    """Empty child_area_names list does nothing."""
    sub_case = _FakeSubCase()
    pub_p = PowersReal(ids=[], equipment_ids=[], values=[], time=0)
    pub_q = PowersImaginary(ids=[], equipment_ids=[], values=[], time=0)
    with patch("distopf.distributed.spatial.enapp.add_s_to_schedules") as add_s_fn:
        apply_s_up_to_sub_case(sub_case, pub_p, pub_q, [])
        add_s_fn.assert_not_called()


def test_apply_s_up_calls_add_s_to_schedules_per_child():
    """apply_s_up_to_sub_case calls add_s_to_schedules once per child area."""
    sub_case = _FakeSubCase()
    pub_p = PowersReal(
        ids=["area_152.a", "area_152.b", "area_152.c"],
        equipment_ids=["area_152.a", "area_152.b", "area_152.c"],
        values=[0.5 * S_BASE, 0.4 * S_BASE, 0.3 * S_BASE],
        time=0,
    )
    pub_q = PowersImaginary(
        ids=["area_152.a", "area_152.b", "area_152.c"],
        equipment_ids=["area_152.a", "area_152.b", "area_152.c"],
        values=[0.1 * S_BASE, 0.08 * S_BASE, 0.06 * S_BASE],
        time=0,
    )
    with patch("distopf.distributed.spatial.enapp.add_s_to_schedules") as add_s_fn:
        apply_s_up_to_sub_case(sub_case, pub_p, pub_q, ["area_152"])
        add_s_fn.assert_called_once()

        s_df = add_s_fn.call_args[0][1]  # second positional arg is the s DataFrame
        assert s_df.iloc[0]["name"] == "area_152"
        # Phase-a should be (0.5 + 0.1j) per-unit
        assert abs(s_df.iloc[0]["a"] - complex(0.5, 0.1)) < 1e-9


# ---------------------------------------------------------------------------
# Tests for _resolve_source_bus (federate.py)
# ---------------------------------------------------------------------------


def _make_mock_topology(bus_names, switch_incidences):
    """Build a minimal Topology-like mock.

    Parameters
    ----------
    bus_names : list[str]
    switch_incidences : list[tuple[str, str, str]]
        List of (from_eq, to_eq, eq_id) tuples.
    """
    bvm = MagicMock()
    bvm.ids = [f"{b}.1" for b in bus_names]

    inc = MagicMock()
    inc.from_equipment = [f"{fr}.1" for fr, _, _ in switch_incidences]
    inc.to_equipment = [f"{to}.1" for _, to, _ in switch_incidences]
    inc.ids = [eq_id for _, _, eq_id in switch_incidences]

    topology = MagicMock()
    topology.base_voltage_magnitudes = bvm
    topology.incidences = inc
    return topology
