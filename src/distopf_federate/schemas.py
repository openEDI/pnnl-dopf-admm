"""Pydantic configuration models for the distopf federate."""

import json
from pathlib import Path
from typing import Any

from oedisi.types.common import DefaultFileNames
from pydantic import BaseModel, Field, ValidationError


class StaticInputs(BaseModel):
    """Validated static configuration written by /configure and read by the federate."""

    name: str = Field(..., description="Unique identifier for the federate name")
    source_bus: str = Field("", description="ID of the boundary/source bus")
    source_line: str = Field("", description="ID of the boundary/source line")
    switches: list[str] = Field(default_factory=list, description="List of switches/controllable lines in the area")
    vup_tol: float = Field(1e-3, description="Convergence tolerance for upstream voltage mismatch")
    sdn_tol: float = Field(1e-3, description="Convergence tolerance for power mismatch")
    rho_vup: float = Field(1000.0, description="ADMM penalty parameter for upstream voltage")
    rho_sup: float = Field(0.0, description="ADMM penalty parameter for active power injection")
    rho_vdn: float = Field(0.0, description="ADMM penalty parameter for downstream voltage discrepancy")
    rho_sdn: float = Field(1000.0, description="ADMM penalty parameter for active/reactive power flow discrepancy")
    max_itr: int = Field(100, description="Maximum number of ADMM iterations per step")
    relaxed: bool = Field(False, description="Boolean flag to enable relaxed model formulation")
    control_type: str = Field("real", description="Control mode (e.g. 'real', 'reactive')")
    number_of_timesteps: int = Field(1, description="Total number of simulation timesteps")
    deltat: float = Field(3600.0, description="Co-simulation time step interval in seconds")
    objective: str = Field("cp_obj_none", description="OPF objective function name")


class DynamicInputs(BaseModel):
    """Dynamic input subscriptions required for federate execution.

    All dynamic inputs are strictly required and must be specified in the input mapping.
    """

    voltages_real: str = Field(..., description="Subscription topic for real voltages")
    voltages_imag: str = Field(..., description="Subscription topic for imaginary voltages")
    topology: str = Field(..., description="Subscription topic for network topology")
    injections: str = Field(..., description="Subscription topic for power injections")
    sub_p: str = Field(..., description="Subscription topic for boundary sub_p from hub_power")
    sub_q: str = Field(..., description="Subscription topic for boundary sub_q from hub_power")
    sub_v: str = Field(..., description="Subscription topic for boundary sub_v from hub_voltage")


class DynamicOutputs(BaseModel):
    """Dynamic output publications produced by the federate.

    Publications are optional. If a publication topic is set to None, the federate
    will skip registering and publishing to that topic.
    """

    controls_real: str | None = Field("controls_real", description="Publication topic for real control commands")
    controls_imag: str | None = Field("controls_imag", description="Publication topic for imaginary control commands")
    solver_stats: str | None = Field("solver_stats", description="Publication topic for solver metrics")
    powers_mag: str | None = Field("powers_mag", description="Publication topic for power magnitudes")
    powers_ang: str | None = Field("powers_ang", description="Publication topic for power angles")
    voltages_mag: str | None = Field("voltages_mag", description="Publication topic for voltage magnitudes")
    pub_p: str | None = Field("pub_p", description="Publication topic for boundary pub_p")
    pub_q: str | None = Field("pub_q", description="Publication topic for boundary pub_q")
    pub_v: str | None = Field("pub_v", description="Publication topic for boundary pub_v")
    pub_c: str | None = Field("pub_c", description="Publication topic for boundary pub_c")



class ComponentDefinition(BaseModel):
    """Unified component definition containing static inputs, dynamic inputs, and dynamic outputs."""

    static_inputs: StaticInputs
    dynamic_inputs: DynamicInputs
    dynamic_outputs: DynamicOutputs = Field(default_factory=DynamicOutputs)

    @classmethod
    def from_build_files(
        cls,
        static_inputs_path: str | Path = DefaultFileNames.STATIC_INPUTS.value,
        input_mapping_path: str | Path = DefaultFileNames.INPUT_MAPPING.value,
    ) -> "ComponentDefinition":
        """Construct and validate ComponentDefinition from static_inputs.json and input_mapping.json.

        Raises ValidationError if any required static or dynamic input is missing or invalid.
        """
        static_path = Path(static_inputs_path)
        mapping_path = Path(input_mapping_path)

        if not static_path.exists():
            raise FileNotFoundError(f"Static inputs file not found: {static_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Input mapping file not found: {mapping_path}")

        with open(static_path, "r", encoding="utf-8") as fh:
            raw_static = json.load(fh)

        with open(mapping_path, "r", encoding="utf-8") as fh:
            raw_mapping: dict[str, Any] = json.load(fh)

        static_inputs = StaticInputs.model_validate(raw_static)
        dynamic_inputs = DynamicInputs.model_validate(raw_mapping)
        dynamic_outputs = DynamicOutputs.model_validate(raw_mapping)

        return cls(
            static_inputs=static_inputs,
            dynamic_inputs=dynamic_inputs,
            dynamic_outputs=dynamic_outputs,
        )

