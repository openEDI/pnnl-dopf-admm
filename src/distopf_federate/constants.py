"""Shared constants for the distopf_federate package."""

# Per-unit power base (1 MVA)
S_BASE: float = 1.0e6  # VA

# Phase column names and 1-indexed phase numbers used by distopf result DataFrames
PHASE_COLS: list = [("a", 1), ("b", 2), ("c", 3)]

# Minimum apparent power capacity assigned to generators when the measured
# output is near zero (prevents an ill-conditioned OPF problem).  Units: pu.
MIN_GEN_SA_PU: float = 0.1  # 0.1 pu = 100 kW at S_BASE=1 MVA

# Commands below this threshold (Watts / VARs) are not published to avoid
# sending noise to DER actuators.
COMMAND_THRESHOLD_W: float = 1.0  # W
