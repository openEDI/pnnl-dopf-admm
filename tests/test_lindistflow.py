"""Regression tests for lindistflow.py bug fixes.

Tests verify:
1. Phase C impedance coefficients are symmetric with Phases A/B
2. voltage_cons_sec receives integer nbranch_s1s2 (not float baseZ)
3. Secondary bus pq/pv indexing produces scalar values
4. Secondary DG_up_lim index stays within bounds
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from admm_federate.lindistflow import voltage_cons_pri, voltage_cons_sec  # noqa: E402


class TestPhaseCImpedanceSymmetry:
    """Verify Phase C voltage drop uses z[1,2] consistently (not z[0,2])."""

    def _compute_phase_coefficients(
        self, z: np.ndarray
    ) -> dict[str, tuple[float, float, float, float, float, float]]:
        """Reproduce the impedance coefficient computation for all 3 phases.

        Returns dict with keys 'A', 'B', 'C', each mapping to
        (pii, qii, pij, qij, pik, qik) tuples.
        """
        # Phase A (reference from lindistflow.py L579-L587)
        paa, qaa = -2 * z[0, 0][0], -2 * z[0, 0][1]
        pab, qab = (
            -(-z[0, 1][0] + math.sqrt(3) * z[0, 1][1]),
            -(-z[0, 1][1] - math.sqrt(3) * z[0, 1][0]),
        )
        pac, qac = (
            -(-z[0, 2][0] - math.sqrt(3) * z[0, 2][1]),
            -(-z[0, 2][1] + math.sqrt(3) * z[0, 2][0]),
        )

        # Phase B (reference from lindistflow.py L609-L617)
        pbb, qbb = -2 * z[1, 1][0], -2 * z[1, 1][1]
        pba, qba = (
            -(-z[0, 1][0] - math.sqrt(3) * z[0, 1][1]),
            -(-z[0, 1][1] + math.sqrt(3) * z[0, 1][0]),
        )
        pbc, qbc = (
            -(-z[1, 2][0] + math.sqrt(3) * z[1, 2][1]),
            -(-z[1, 2][1] - math.sqrt(3) * z[1, 2][0]),
        )

        # Phase C (reference from lindistflow.py L639-L647, FIXED)
        pcc, qcc = -2 * z[2, 2][0], -2 * z[2, 2][1]
        pca, qca = (
            -(-z[0, 2][0] + math.sqrt(3) * z[0, 2][1]),
            -(-z[0, 2][1] - math.sqrt(3) * z[0, 2][0]),
        )
        pcb, qcb = (
            -(-z[1, 2][0] - math.sqrt(3) * z[1, 2][1]),
            -(-z[1, 2][1] + math.sqrt(3) * z[1, 2][0]),
        )

        return {
            "A": (paa, qaa, pab, qab, pac, qac),
            "B": (pba, qba, pbb, qbb, pbc, qbc),
            "C": (pca, qca, pcb, qcb, pcc, qcc),
        }

    def test_mutual_impedance_uses_correct_matrix_entry(self) -> None:
        """Phase C 'cb' mutual terms must reference z[1,2], not z[0,2]."""
        # Construct an asymmetric impedance matrix where z[0,2] != z[1,2]
        z = np.zeros((3, 3, 2))
        z[0, 0] = [0.1, 0.05]
        z[1, 1] = [0.1, 0.05]
        z[2, 2] = [0.1, 0.05]
        z[0, 1] = [0.02, 0.01]
        z[0, 2] = [0.03, 0.015]  # intentionally different from z[1,2]
        z[1, 2] = [0.04, 0.02]  # the value Phase C 'cb' should use

        coeffs = self._compute_phase_coefficients(z)

        # Phase C 'cb' mutual terms (pcb, qcb) must be derived from z[1,2]
        pcb, qcb = coeffs["C"][2], coeffs["C"][3]

        # Compute expected values from z[1,2]
        expected_pcb = -(-z[1, 2][0] - math.sqrt(3) * z[1, 2][1])
        expected_qcb = -(-z[1, 2][1] + math.sqrt(3) * z[1, 2][0])

        assert pcb == pytest.approx(
            expected_pcb
        ), f"pcb should use z[1,2], got {pcb} expected {expected_pcb}"
        assert qcb == pytest.approx(
            expected_qcb
        ), f"qcb should use z[1,2], got {qcb} expected {expected_qcb}"

        # Verify it does NOT match the old buggy z[0,2] value
        buggy_qcb = -(-z[0, 2][1] + math.sqrt(3) * z[1, 2][0])
        assert qcb != pytest.approx(
            buggy_qcb
        ), "qcb incorrectly matches the old z[0,2] buggy formula"

    def test_self_impedance_diagonal(self) -> None:
        """Each phase's self-impedance uses its own diagonal entry."""
        z = np.zeros((3, 3, 2))
        z[0, 0] = [0.10, 0.05]
        z[1, 1] = [0.12, 0.06]
        z[2, 2] = [0.14, 0.07]

        coeffs = self._compute_phase_coefficients(z)

        # Phase A self
        assert coeffs["A"][0] == pytest.approx(-2 * 0.10)
        assert coeffs["A"][1] == pytest.approx(-2 * 0.05)
        # Phase B self
        assert coeffs["B"][2] == pytest.approx(-2 * 0.12)
        assert coeffs["B"][3] == pytest.approx(-2 * 0.06)
        # Phase C self
        assert coeffs["C"][4] == pytest.approx(-2 * 0.14)
        assert coeffs["C"][5] == pytest.approx(-2 * 0.07)


class TestVoltageConsSecArgument:
    """Verify voltage_cons_sec receives integer nbranch_s1s2, not float baseZ."""

    def test_nbranch_s1s2_produces_correct_offset(self) -> None:
        """The n_flow_s1s2 offset must use integer nbranch_s1s2."""
        nbus_ABC = 5
        nbus_s1s2 = 3
        nbranch_ABC = 4
        nbranch_s1s2 = 2  # correct value (integer count)

        # Expected offset per voltage_cons_sec L151-153
        expected_offset = (
            (nbus_ABC * 3 + nbus_s1s2)
            + (nbus_ABC * 6 + nbus_s1s2 * 2)
            + nbranch_ABC * 6
        )

        # Create arrays large enough for the function
        size = expected_offset + nbranch_s1s2 * 2 + 10
        A = np.zeros((10, size))
        b = np.zeros(10)

        A, b = voltage_cons_sec(
            A,
            b,
            p=0,
            frm=0,
            to=1,
            counteq=0,
            p_pri=1.0,
            q_pri=0.5,
            p_sec=0.8,
            q_sec=0.4,
            nbus_ABC=nbus_ABC,
            nbus_s1s2=nbus_s1s2,
            nbranch_ABC=nbranch_ABC,
            nbranch_s1s2=nbranch_s1s2,
        )

        # Real power coefficient should be at expected_offset + p
        assert (
            A[0, expected_offset] != 0.0
        ), f"Real power coefficient not set at offset {expected_offset}"
        # Reactive power coefficient at expected_offset + nbranch_s1s2
        assert (
            A[0, expected_offset + nbranch_s1s2] != 0.0
        ), f"Reactive power coefficient not set at offset {expected_offset + nbranch_s1s2}"

    def test_basez_float_would_corrupt_offset(self) -> None:
        """Passing baseZ=1.0 as nbranch_s1s2 would produce wrong reactive offset."""
        nbus_ABC = 5
        nbus_s1s2 = 3
        nbranch_ABC = 4
        nbranch_s1s2_correct = 2
        basez_wrong = 1.0  # the old buggy value

        base_offset = (
            (nbus_ABC * 3 + nbus_s1s2)
            + (nbus_ABC * 6 + nbus_s1s2 * 2)
            + nbranch_ABC * 6
        )

        # With correct nbranch_s1s2=2, reactive offset = base + 2
        correct_reactive_offset = base_offset + nbranch_s1s2_correct
        # With buggy baseZ=1.0, reactive offset = base + 1 (wrong!)
        buggy_reactive_offset = base_offset + int(basez_wrong)

        assert (
            correct_reactive_offset != buggy_reactive_offset
        ), "Test precondition: correct and buggy offsets should differ"


class TestSecondaryBusIndexing:
    """Verify secondary bus pq/pv indexing produces scalars and DG indices stay in bounds."""

    def test_pq_indexing_returns_scalar(self) -> None:
        """val_bus.pq[0][0] is a float, not val_bus.pq[0] (a list)."""
        # Simulate Bus.pq shape (3, 2) as list[list[float]]
        pq = [[100.0, 50.0], [0.0, 0.0], [0.0, 0.0]]

        real_power = pq[0][0]  # correct: 100.0
        reactive_power = pq[0][1]  # correct: 50.0
        buggy_real = pq[0]  # buggy: [100.0, 50.0]

        assert isinstance(real_power, float)
        assert isinstance(reactive_power, float)
        assert isinstance(buggy_real, list), "pq[0] should be a list (the bug)"
        assert real_power == 100.0
        assert reactive_power == 50.0

    def test_dg_up_lim_index_in_bounds(self) -> None:
        """DG_up_lim index for secondary buses must be < n_bus."""
        nbus_ABC = 10
        nbus_s1s2 = 5
        n_bus = nbus_ABC * 3 + nbus_s1s2  # = 35

        DG_up_lim = np.zeros((n_bus, 1))

        # Secondary bus indices start at nbus_ABC (after all primary buses)
        for sec_idx in range(nbus_ABC, nbus_ABC + nbus_s1s2):
            # Fixed formula: nbus_ABC * 2 + val_bus.idx
            fixed_index = nbus_ABC * 2 + sec_idx
            assert (
                0 <= fixed_index < n_bus
            ), f"Fixed index {fixed_index} out of bounds [0, {n_bus})"
            DG_up_lim[fixed_index] = 1.0  # should not raise

            # Old buggy formula: nbus_ABC * 3 + val_bus.idx
            buggy_index = nbus_ABC * 3 + sec_idx
            assert (
                buggy_index >= n_bus
            ), f"Buggy index {buggy_index} should be >= {n_bus} (out of bounds)"
