"""Pydantic configuration models for the distopf federate."""

from pydantic import BaseModel, Field


class StaticInputs(BaseModel):
    """Validated configuration written by /configure and read by the federate."""

    name: str
    # Source bus or switch ID for this area's upstream boundary
    source_bus: str = ""
    # Source line/switch ID (used for ADMM boundary identification)
    source_line: str = ""
    # Switch IDs that define child-area boundaries
    switches: list[str] = Field(default_factory=list)
    # ADMM convergence tolerances
    vup_tol: float = 1e-3
    sdn_tol: float = 1e-3
    # ADMM rho penalty parameters
    rho_vup: float = 1000.0
    rho_sup: float = 0.0
    rho_vdn: float = 0.0
    rho_sdn: float = 1000.0
    # ADMM iteration limit
    max_itr: int = 100
    # Whether to use relaxed ADMM
    relaxed: bool = False
    # Control type: "real", "reactive", or "inverter"
    control_type: str = "real"
    # Simulation time steps
    number_of_timesteps: int = 1
    # HELICS time period in seconds (not in scenario — use feeder step size)
    deltat: float = 3600.0
    # OPF objective function name (see federate.OBJECTIVES)
    objective: str = "cp_obj_none"
