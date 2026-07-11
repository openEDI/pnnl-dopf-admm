import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import module directly from source tree to avoid heavy package side effects.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from admm_federate.opf_federate import ComponentParameters, OPFFederate  # noqa: E402


def test_load_static_inputs(tmp_path) -> None:
    static_inputs = {
        "name": "test_admm",
        "vup_tol": 0.01,
        "sdn_tol": 0.01,
        "max_itr": 10,
        "deltat": 3600,
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

    # We patch open in builtins so that when it looks for static_inputs.json, it reads from our tmp_path
    mock_file = tmp_path / "static_inputs.json"
    mock_file.write_text(json.dumps(static_inputs))

    original_open = open

    def mock_open(file, *args, **kwargs):
        if "static_inputs.json" in str(file):
            return original_open(mock_file, *args, **kwargs)
        return original_open(file, *args, **kwargs)

    with patch("builtins.open", mock_open):
        with (
            patch.object(OPFFederate, "initilize"),
            patch.object(OPFFederate, "load_input_mapping"),
            patch.object(OPFFederate, "load_component_definition"),
            patch.object(OPFFederate, "register_subscription"),
            patch.object(OPFFederate, "register_publication"),
        ):
            broker_config = MagicMock()
            fed = OPFFederate(broker_config)

            # Assertions on loaded parameters
            assert isinstance(fed.static, ComponentParameters)
            assert fed.static.name == "test_admm"
            assert fed.static.source_bus == "150"
            assert fed.static.source_line == ""
            assert fed.deltat == 3600
            assert fed.admm_config.rho_vup == 1000.0
            assert fed.admm_config.relaxed is False


def test_generate_area_info_missing_slack_bus() -> None:
    import networkx as nx

    from admm_federate import adapter

    # Create a simple graph that does not contain slack bus "150"
    graph = nx.Graph()
    graph.add_edge("1", "2", id="sw2", tag="SWITCH", name="1_2")

    # Mock topology
    topology = MagicMock()

    # Call generate_area_info with a slack_bus that is not in the graph
    # Boundary ids match the edge to prevent early return on boundary check
    res_branch, res_bus = adapter.generate_area_info(
        graph, topology, slack_bus="150", boundary=["sw2"]
    )

    # Should return None, None instead of crashing with NodeNotFound
    assert res_branch is None
    assert res_bus is None


def test_schema_and_component_definition() -> None:
    # 1. Load schema.json
    schema_path = Path(__file__).resolve().parents[1] / "schema.json"
    with open(schema_path, encoding="utf-8") as f:
        schema_json = json.load(f)

    # 2. Get current model schema
    model_schema = ComponentParameters.model_json_schema()

    # Verify they align
    assert model_schema == schema_json

    # 3. Load component_definition.json
    comp_def_path = Path(__file__).resolve().parents[1] / "component_definition.json"
    with open(comp_def_path, encoding="utf-8") as f:
        comp_def = json.load(f)

    static_inputs = comp_def.get("static_inputs", [])
    static_input_names = {item["port_id"] for item in static_inputs}

    schema_properties = set(model_schema.get("properties", {}).keys())

    # Verify static_inputs matches the schema properties exactly
    assert static_input_names == schema_properties
