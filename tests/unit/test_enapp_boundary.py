"""Unit tests for ENAPP boundary helpers in distopf_federate.importer and federate.

These tests mock ``distopf`` so that the suite runs without an installed
distopf package, matching the approach used in ``test_importer.py``.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub the distopf package so that the module-level ``import distopf as opf``
# in importer.py does not raise ModuleNotFoundError.
# ---------------------------------------------------------------------------

def _make_distopf_stub():
    """Build a minimal sys.modules stub for distopf and sub-modules."""
    distopf_mod = types.ModuleType("distopf")
    distopf_mod.SWING_BUS = "SWING"
    distopf_mod.SWING_FREE = "SWING_FREE"
    distopf_mod.PQ_BUS = "PQ"
    distopf_mod.CONTROL_PQ = "PQ"
    distopf_mod.cp_obj_loss = MagicMock()
    distopf_mod.cp_obj_curtail = MagicMock()
    distopf_mod.cp_obj_curtail_lp = MagicMock()
    distopf_mod.cp_obj_target_p_total = MagicMock()
    distopf_mod.cp_obj_target_q_total = MagicMock()
    distopf_mod.cp_obj_none = None

    api_mod = types.ModuleType("distopf.api")
    api_mod.Case = MagicMock()
    distopf_mod.api = api_mod
    distopf_mod.Case = api_mod.Case  # also expose at top level

    spatial_mod = types.ModuleType("distopf.distributed")
    spatial_sub = types.ModuleType("distopf.distributed.spatial")
    enapp_mod = types.ModuleType("distopf.distributed.spatial.enapp")
    decompose_mod = types.ModuleType("distopf.distributed.spatial.decompose")
    enapp_mod.add_v_swing_to_schedules = MagicMock(side_effect=lambda sched, v, name: sched)
    enapp_mod.add_s_to_schedules = MagicMock(side_effect=lambda sched, s, name: sched)
    enapp_mod.parse_s_up = MagicMock(return_value=pd.DataFrame(columns=["name", "t", "a", "b", "c"]))
    enapp_mod.parse_v_dn = MagicMock(return_value=pd.DataFrame(columns=["name", "t", "a", "b", "c"]))
    decompose_mod.decompose = MagicMock(return_value={})

    distopf_mod.distributed = spatial_mod
    spatial_mod.spatial = spatial_sub
    spatial_sub.enapp = enapp_mod
    spatial_sub.decompose = decompose_mod

    return {
        "distopf": distopf_mod,
        "distopf.api": api_mod,
        "distopf.distributed": spatial_mod,
        "distopf.distributed.spatial": spatial_sub,
        "distopf.distributed.spatial.enapp": enapp_mod,
        "distopf.distributed.spatial.decompose": decompose_mod,
    }


_DISTOPF_STUBS = _make_distopf_stub()
for _name, _mod in _DISTOPF_STUBS.items():
    sys.modules.setdefault(_name, _mod)

# Make sure helics is also stubbed (needed transitively by federate imports)
if "helics" not in sys.modules:
    _helics_stub = types.ModuleType("helics")
    _helics_stub.HELICS_CORE_TYPE_ZMQ = 0
    _helics_stub.HELICS_PROPERTY_TIME_PERIOD = 0
    _helics_stub.HELICS_DATA_TYPE_STRING = 0
    _helics_stub.HELICS_TIME_MAXTIME = 1e300
    _helics_stub.helics_iteration_request_iterate_if_needed = 0
    _helics_stub.helics_iteration_request_no_iteration = 1
    _helics_stub.helics_iteration_result_next_step = 0
    _helics_stub.helics_iteration_result_error = 2
    _helics_stub.helicsCreateFederateInfo = MagicMock(return_value=MagicMock())
    _helics_stub.helicsCreateValueFederate = MagicMock(return_value=MagicMock())
    _helics_stub.helicsFederateSetTimeProperty = MagicMock()
    _helics_stub.helicsFederateInfoSetBroker = MagicMock()
    _helics_stub.helicsFederateInfoSetBrokerPort = MagicMock()
    _helics_stub.helicsFederateEnterInitializingMode = MagicMock()
    _helics_stub.helicsFederateEnterExecutingMode = MagicMock()
    _helics_stub.helicsFederateRequestTimeIterative = MagicMock(return_value=(0, 0))
    _helics_stub.helicsFederateFinalize = MagicMock()
    _helics_stub.helicsFederateFree = MagicMock()
    _helics_stub.helicsCloseLibrary = MagicMock()
    _helics_stub.helicsFederateInfoSetTimeDelta = MagicMock()
    sys.modules["helics"] = _helics_stub

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from oedisi.types.data_types import PowersImaginary, PowersReal, VoltagesMagnitude

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
    from distopf_federate.importer import apply_v_dn_to_sub_case

    sub_case = _FakeSubCase()
    vmag = VoltagesMagnitude(ids=["area_999.a", "area_999.b"], values=[1.05, 1.05], time=0)
    before = sub_case.schedules.copy()
    apply_v_dn_to_sub_case(sub_case, vmag, "area_152")
    pd.testing.assert_frame_equal(sub_case.schedules, before)


def test_apply_v_dn_calls_add_v_swing_to_schedules():
    """apply_v_dn_to_sub_case calls distopf's add_v_swing_to_schedules with parsed voltage."""
    from distopf_federate.importer import apply_v_dn_to_sub_case

    add_v_fn = _DISTOPF_STUBS["distopf.distributed.spatial.enapp"].add_v_swing_to_schedules
    add_v_fn.reset_mock()

    sub_case = _FakeSubCase()
    vmag = VoltagesMagnitude(
        ids=["area_152.a", "area_152.b", "area_152.c"],
        values=[1.03, 1.02, 1.04],
        time=0,
    )
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
    from distopf_federate.importer import apply_s_up_to_sub_case

    add_s_fn = _DISTOPF_STUBS["distopf.distributed.spatial.enapp"].add_s_to_schedules
    add_s_fn.reset_mock()

    sub_case = _FakeSubCase()
    pub_p = PowersReal(ids=[], equipment_ids=[], values=[], time=0)
    pub_q = PowersImaginary(ids=[], equipment_ids=[], values=[], time=0)
    apply_s_up_to_sub_case(sub_case, pub_p, pub_q, [])

    add_s_fn.assert_not_called()


def test_apply_s_up_calls_add_s_to_schedules_per_child():
    """apply_s_up_to_sub_case calls add_s_to_schedules once per child area."""
    from distopf_federate.importer import apply_s_up_to_sub_case
    from distopf_federate.constants import S_BASE

    add_s_fn = _DISTOPF_STUBS["distopf.distributed.spatial.enapp"].add_s_to_schedules
    add_s_fn.reset_mock()

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


def test_resolve_source_bus_direct_bus_name():
    """If static.source is already a bus name, return it unchanged."""
    from distopf_federate.federate import DistopfFederate

    fed = object.__new__(DistopfFederate)
    fed.static = MagicMock()
    fed.static.source = "150"

    topology = _make_mock_topology(
        bus_names=["150", "13", "18"],
        switch_incidences=[("150", "13", "sw2"), ("150", "18", "sw3")],
    )

    result = fed._resolve_source_bus(topology)
    assert result == "150"


def test_resolve_source_bus_switch_id_resolves_to_downstream_bus():
    """If static.source is a switch ID, return the downstream bus."""
    from distopf_federate.federate import DistopfFederate

    fed = object.__new__(DistopfFederate)
    fed.static = MagicMock()
    fed.static.source = "sw3"

    topology = _make_mock_topology(
        bus_names=["150", "13", "18"],
        switch_incidences=[("150", "13", "sw2"), ("150", "18", "sw3")],
    )

    result = fed._resolve_source_bus(topology)
    assert result == "18"


def test_resolve_source_bus_unknown_falls_back_to_verbatim():
    """If source cannot be matched, it is returned verbatim with a warning."""
    from distopf_federate.federate import DistopfFederate

    fed = object.__new__(DistopfFederate)
    fed.static = MagicMock()
    fed.static.source = "unknown_eq"

    topology = _make_mock_topology(
        bus_names=["150", "13"],
        switch_incidences=[("150", "13", "sw2")],
    )

    result = fed._resolve_source_bus(topology)
    assert result == "unknown_eq"
