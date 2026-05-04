"""Unit tests for distopf_federate.importer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from oedisi.types.data_types import (
    AdmittanceSparse,
    IncidenceList,
    Injection,
    PowersImaginary,
    PowersReal,
    Topology,
    VoltagesMagnitude,
)

from distopf_federate.importer import (
    _phases_to_str,
    topology_to_case,
    update_case_from_measurements,
)

# ---------------------------------------------------------------------------
# Helpers to build a minimal 3-bus test topology:
#
#   sourcebus ─── line1 ─── bus2 ─── line2 ─── bus3
#
# All three-phase, 4.16 kV (line-to-line), v_ln = 2401.77 V.
# line1 has a simple resistive admittance y = 1 S (diagonal).
# line2 has the same.
# bus3 has a 100 kW load phase A and a PVSystem with 50 kW on phase A.
# ---------------------------------------------------------------------------

V_LN_BASE = 2401.77  # V, line-to-neutral for 4.16 kV system
# Off-diagonal nodal admittance Y_ij = -y_branch (negative by convention).
# With y_branch = 100 S (low-R line), Y_ij = -100 S.
Y_DIAG = -100.0


def _make_topology() -> Topology:
    """Minimal 3-bus Topology fixture for testing."""
    # Incidences: sourcebus→bus2 (line1), bus2→bus3 (line2)
    incidences = IncidenceList(
        from_equipment=["sourcebus", "bus2"],
        to_equipment=["bus2", "bus3"],
        ids=["line1", "line2"],
    )

    # Admittance: diagonal 3×3 for each branch (off-diagonal of nodal Y)
    # from/to format: "BUS.phase", entries for each (i, j) pair in the matrix
    from_equip = []
    to_equip = []
    adm_list = []
    for fr_bus, to_bus in [("sourcebus", "bus2"), ("bus2", "bus3")]:
        for ph in range(1, 4):
            from_equip.append(f"{fr_bus}.{ph}")
            to_equip.append(f"{to_bus}.{ph}")
            adm_list.append((Y_DIAG, 0.0))

    admittance = AdmittanceSparse(
        from_equipment=from_equip,
        to_equipment=to_equip,
        admittance_list=adm_list,
    )

    # Base voltages: all three phases at each bus
    bv_ids = []
    bv_values = []
    for bus in ["sourcebus", "bus2", "bus3"]:
        for ph in range(1, 4):
            bv_ids.append(f"{bus}.{ph}")
            bv_values.append(V_LN_BASE)

    base_voltages = VoltagesMagnitude(ids=bv_ids, values=bv_values, time=0)

    # Injections: 100 kW load at bus3.1, 50 kW PVSystem at bus3.1
    real_ids = ["bus3.1", "bus3.1"]
    real_equipment = ["Load.load3", "PVSystem.pv3"]
    # Load is a negative injection (consuming), PV is positive
    real_values = [-100.0, 50.0]  # kW

    imag_ids = ["bus3.1", "bus3.1"]
    imag_equipment = ["Load.load3", "PVSystem.pv3"]
    imag_values = [-20.0, 0.0]  # kVAR

    power_real = PowersReal(
        ids=real_ids, equipment_ids=real_equipment, values=real_values, time=0
    )
    power_imag = PowersImaginary(
        ids=imag_ids, equipment_ids=imag_equipment, values=imag_values, time=0
    )
    injections = Injection(power_real=power_real, power_imaginary=power_imag)

    return Topology(
        admittance=admittance,
        injections=injections,
        incidences=incidences,
        base_voltage_magnitudes=base_voltages,
        slack_bus=["sourcebus.1"],
    )


# ---------------------------------------------------------------------------
# _phases_to_str
# ---------------------------------------------------------------------------


def test_phases_to_str_all_phases():
    assert _phases_to_str([1, 2, 3]) == "abc"


def test_phases_to_str_phase_a_only():
    assert _phases_to_str([1, 0, 0]) == "a"


def test_phases_to_str_phases_bc():
    assert _phases_to_str([0, 2, 3]) == "bc"


def test_phases_to_str_empty():
    assert _phases_to_str([0, 0, 0]) == ""


# ---------------------------------------------------------------------------
# topology_to_case
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_case():
    topology = _make_topology()
    case, name_to_id, v_ln_base_map = topology_to_case(topology, source_bus="sourcebus")
    return case, name_to_id, v_ln_base_map


def test_topology_to_case_returns_correct_types(minimal_case):
    case, name_to_id, v_ln_base_map = minimal_case
    import distopf as opf

    assert isinstance(case, opf.Case)
    assert isinstance(name_to_id, dict)
    assert isinstance(v_ln_base_map, dict)


def test_topology_to_case_bus_count(minimal_case):
    case, name_to_id, _ = minimal_case
    assert len(case.bus_data) == 3
    assert set(name_to_id.keys()) == {"sourcebus", "bus2", "bus3"}


def test_topology_to_case_bus_ids_are_unique_integers(minimal_case):
    case, name_to_id, _ = minimal_case
    ids = case.bus_data["id"].tolist()
    assert len(ids) == len(set(ids)), "bus IDs are not unique"
    assert all(isinstance(i, (int, np.integer)) for i in ids)


def test_topology_to_case_swing_bus(minimal_case):
    case, _, _ = minimal_case
    import distopf as opf

    swing_rows = case.bus_data[case.bus_data["bus_type"] == opf.SWING_BUS]
    assert len(swing_rows) == 1
    assert swing_rows.iloc[0]["name"] == "sourcebus"


def test_topology_to_case_branch_count(minimal_case):
    case, _, _ = minimal_case
    assert len(case.branch_data) == 2


def test_topology_to_case_branch_topology(minimal_case):
    case, name_to_id, _ = minimal_case
    src_id = name_to_id["sourcebus"]
    bus2_id = name_to_id["bus2"]
    bus3_id = name_to_id["bus3"]
    fb_list = case.branch_data["fb"].tolist()
    tb_list = case.branch_data["tb"].tolist()
    assert src_id in fb_list
    assert bus2_id in tb_list
    assert bus2_id in fb_list
    assert bus3_id in tb_list


def test_topology_to_case_branch_impedance_positive(minimal_case):
    case, _, _ = minimal_case
    # With Y_DIAG = 100 S, R = 1/100 = 0.01 Ω → small but positive raa
    assert (case.branch_data["raa"] >= 0.0).all()
    assert (case.branch_data["xaa"] >= 0.0).all()


def test_topology_to_case_load_at_bus3(minimal_case):
    case, name_to_id, _ = minimal_case
    bus3_id = name_to_id["bus3"]
    bus3_row = case.bus_data[case.bus_data["id"] == bus3_id].iloc[0]
    # 100 kW = 0.1 per-unit with S_BASE = 1 MVA
    assert abs(bus3_row["pl_a"] - 0.1) < 1e-9, f"pl_a = {bus3_row['pl_a']}"


def test_topology_to_case_no_load_at_sourcebus(minimal_case):
    case, name_to_id, _ = minimal_case
    src_id = name_to_id["sourcebus"]
    src_row = case.bus_data[case.bus_data["id"] == src_id].iloc[0]
    assert src_row["pl_a"] == 0.0
    assert src_row["pl_b"] == 0.0
    assert src_row["pl_c"] == 0.0


def test_topology_to_case_gen_data_has_pvsystem(minimal_case):
    case, name_to_id, _ = minimal_case
    assert case.gen_data is not None
    assert len(case.gen_data) == 1
    assert case.gen_data.iloc[0]["name"] == "bus3"
    # 50 kW = 0.05 per-unit
    assert abs(case.gen_data.iloc[0]["pa"] - 0.05) < 1e-9


def test_topology_to_case_v_ln_base_map(minimal_case):
    _, _, v_ln_base_map = minimal_case
    assert "sourcebus" in v_ln_base_map
    assert abs(v_ln_base_map["sourcebus"] - V_LN_BASE) < 1.0


# ---------------------------------------------------------------------------
# update_case_from_measurements
# ---------------------------------------------------------------------------


def test_update_case_load_changes(minimal_case):
    case, name_to_id, _ = minimal_case

    # New live injection: bus3 load is now 200 kW (double)
    real = PowersReal(
        ids=["bus3.1"],
        equipment_ids=["Load.load3"],
        values=[-200.0],
        time=1,
    )
    imag = PowersImaginary(
        ids=["bus3.1"],
        equipment_ids=["Load.load3"],
        values=[-40.0],
        time=1,
    )
    injection = Injection(power_real=real, power_imaginary=imag)

    case = update_case_from_measurements(case, injection, name_to_id)

    bus3_id = name_to_id["bus3"]
    bus3_row = case.bus_data[case.bus_data["id"] == bus3_id].iloc[0]
    assert abs(bus3_row["pl_a"] - 0.2) < 1e-9, f"pl_a after update = {bus3_row['pl_a']}"


def test_update_case_pv_generation_changes(minimal_case):
    case, name_to_id, _ = minimal_case

    # PV ramps to 80 kW
    real = PowersReal(
        ids=["bus3.1"],
        equipment_ids=["PVSystem.pv3"],
        values=[80.0],
        time=1,
    )
    imag = PowersImaginary(
        ids=["bus3.1"],
        equipment_ids=["PVSystem.pv3"],
        values=[0.0],
        time=1,
    )
    injection = Injection(power_real=real, power_imaginary=imag)

    case = update_case_from_measurements(case, injection, name_to_id)

    bus3_id = name_to_id["bus3"]
    gen_row = case.gen_data[case.gen_data["id"] == bus3_id].iloc[0]
    assert abs(gen_row["pa"] - 0.08) < 1e-9, f"pa after update = {gen_row['pa']}"


def test_update_case_unknown_bus_is_ignored(minimal_case):
    """Measurements for buses not in name_to_id should not raise."""
    case, name_to_id, _ = minimal_case

    real = PowersReal(
        ids=["ghost_bus.1"],
        equipment_ids=["Load.ghost"],
        values=[-100.0],
        time=1,
    )
    imag = PowersImaginary(
        ids=["ghost_bus.1"],
        equipment_ids=["Load.ghost"],
        values=[0.0],
        time=1,
    )
    injection = Injection(power_real=real, power_imaginary=imag)

    # Should not raise
    update_case_from_measurements(case, injection, name_to_id)
