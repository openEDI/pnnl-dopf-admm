import json
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

# Import module directly from source tree
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distopf_federate.schemas import (
    ComponentDefinition,
    DynamicInputs,
    DynamicOutputs,
    StaticInputs,
)


def test_component_definition_from_build_files(tmp_path) -> None:
    static_inputs = {
        "name": "test_admm",
        "vup_tol": 0.01,
        "sdn_tol": 0.01,
        "max_itr": 10,
        "deltat": 3600.0,
        "relaxed": False,
        "control_type": "real",
        "switches": ["sw2", "sw3"],
        "source_bus": "150",
        "source_line": "",
        "rho_vup": 1000.0,
        "rho_sup": 0.0,
        "rho_vdn": 0.0,
        "rho_sdn": 1000.0,
    }

    input_mapping = {
        "voltages_real": "feeder/voltages_real",
        "voltages_imag": "feeder/voltages_imag",
        "topology": "feeder/topology",
        "injections": "feeder/injections",
        "sub_p": "hub_power/pub_p",
        "sub_q": "hub_power/pub_q",
        "sub_v": "hub_voltage/pub_v",
        "pub_c": "control_feeder/change_commands",
        "solver_stats": "recorder/solver_stats",
    }

    static_file = tmp_path / "static_inputs.json"
    mapping_file = tmp_path / "input_mapping.json"
    static_file.write_text(json.dumps(static_inputs))
    mapping_file.write_text(json.dumps(input_mapping))

    comp_def = ComponentDefinition.from_build_files(
        static_inputs_path=static_file,
        input_mapping_path=mapping_file,
    )

    assert isinstance(comp_def.static_inputs, StaticInputs)
    assert comp_def.static_inputs.name == "test_admm"
    assert comp_def.static_inputs.source_bus == "150"

    assert isinstance(comp_def.dynamic_inputs, DynamicInputs)
    assert comp_def.dynamic_inputs.voltages_real == "feeder/voltages_real"
    assert comp_def.dynamic_inputs.sub_p == "hub_power/pub_p"

    assert isinstance(comp_def.dynamic_outputs, DynamicOutputs)
    assert comp_def.dynamic_outputs.pub_c == "control_feeder/change_commands"
    assert comp_def.dynamic_outputs.solver_stats == "recorder/solver_stats"
    # Unmapped publication defaults to standard port name "pub_v"
    assert comp_def.dynamic_outputs.pub_v == "pub_v"



def test_missing_required_dynamic_input_raises_validation_error(tmp_path) -> None:
    static_inputs = {"name": "test_admm"}
    # Missing required 'topology' and 'injections'
    incomplete_mapping = {
        "voltages_real": "feeder/voltages_real",
        "voltages_imag": "feeder/voltages_imag",
        "sub_p": "hub_power/pub_p",
        "sub_q": "hub_power/pub_q",
        "sub_v": "hub_voltage/pub_v",
    }

    static_file = tmp_path / "static_inputs.json"
    mapping_file = tmp_path / "input_mapping.json"
    static_file.write_text(json.dumps(static_inputs))
    mapping_file.write_text(json.dumps(incomplete_mapping))

    with pytest.raises(ValidationError) as exc_info:
        ComponentDefinition.from_build_files(
            static_inputs_path=static_file,
            input_mapping_path=mapping_file,
        )
    assert "topology" in str(exc_info.value) or "injections" in str(exc_info.value)


def test_schema_and_component_definition_sync() -> None:
    # 1. Generate schema dict from StaticInputs model
    model_schema = StaticInputs.model_json_schema()

    # Save to schema.json to ensure sync
    schema_path = Path(__file__).resolve().parents[1] / "schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(model_schema, indent=2) + "\n")

    # 2. Verify component_definition.json static_inputs match StaticInputs schema properties
    comp_def_path = Path(__file__).resolve().parents[1] / "component_definition.json"
    with open(comp_def_path, encoding="utf-8") as f:
        comp_def = json.load(f)

    static_inputs = comp_def.get("static_inputs", [])
    static_input_names = {item["port_id"] for item in static_inputs}
    schema_properties = set(model_schema.get("properties", {}).keys())

    assert static_input_names == schema_properties, (
        "Mismatch between component_definition.json static_inputs and StaticInputs schema properties.\n"
        f"Missing in component_definition.json: {schema_properties - static_input_names}\n"
        f"Extra in component_definition.json: {static_input_names - schema_properties}"
    )

    # 3. Verify component_definition.json dynamic_inputs match DynamicInputs fields
    dynamic_inputs = comp_def.get("dynamic_inputs", [])
    comp_def_dyn_inputs = {item["port_id"] for item in dynamic_inputs}
    model_dyn_inputs = set(DynamicInputs.model_fields.keys())
    assert comp_def_dyn_inputs == model_dyn_inputs

    # 4. Verify component_definition.json dynamic_outputs match DynamicOutputs fields
    dynamic_outputs = comp_def.get("dynamic_outputs", [])
    comp_def_dyn_outputs = {item["port_id"] for item in dynamic_outputs}
    model_dyn_outputs = set(DynamicOutputs.model_fields.keys())
    assert comp_def_dyn_outputs == model_dyn_outputs

