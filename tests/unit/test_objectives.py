"""Unit tests for OPF objective configuration, validation, and execution."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from distopf_federate.federate import OBJECTIVES, DistopfFederate
from distopf_federate.importer import topology_to_case
from distopf_federate.schemas import ObjectiveType, StaticInputs

from .test_importer import _make_topology


def test_default_objective_is_maximize_gen() -> None:
    """StaticInputs should default objective to maximize_gen."""
    config = StaticInputs(name="test_area")
    assert config.objective == ObjectiveType.MAXIMIZE_GEN
    assert config.objective == "maximize_gen"


@pytest.mark.parametrize(
    "obj_str, expected_enum",
    [
        ("maximize_gen", ObjectiveType.MAXIMIZE_GEN),
        ("minimize_loss", ObjectiveType.MINIMIZE_LOSS),
        ("minimize_curtail", ObjectiveType.MINIMIZE_CURTAIL),
        ("minimize_load", ObjectiveType.MINIMIZE_LOAD),
    ],
)
def test_valid_objectives_accepted(obj_str: str, expected_enum: ObjectiveType) -> None:
    """All supported objective strings should validate to their ObjectiveType enum."""
    config = StaticInputs(name="test_area", objective=obj_str)
    assert config.objective == expected_enum


@pytest.mark.parametrize(
    "invalid_obj",
    [
        "none",
        "cp_obj_none",
        "",
        "loss_min",
        "curtail_min",
        "random_objective",
        "gen_max",
    ],
)
def test_invalid_objectives_raise_validation_error(invalid_obj: str) -> None:
    """Unrecognized or disallowed objective strings (including 'none') must raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        StaticInputs(name="test_area", objective=invalid_obj)
    assert "objective" in str(exc_info.value)


@pytest.mark.parametrize("obj_type", list(ObjectiveType))
def test_get_objective_fn_resolves_all_supported_objectives(obj_type: ObjectiveType) -> None:
    """DistopfFederate._get_objective_fn() should return a valid callable for every ObjectiveType."""
    fed = object.__new__(DistopfFederate)
    fed.static = MagicMock()
    fed.static.objective = obj_type

    fn = fed._get_objective_fn()
    assert callable(fn)
    assert fn == OBJECTIVES[obj_type.value]


def test_get_objective_fn_raises_on_unsupported_objective() -> None:
    """DistopfFederate._get_objective_fn() should raise ValueError for unsupported objective values."""
    fed = object.__new__(DistopfFederate)
    fed.static = MagicMock()
    fed.static.objective = "unsupported_obj"

    with pytest.raises(ValueError, match="Unsupported objective"):
        fed._get_objective_fn()


@pytest.mark.parametrize("obj_type", list(ObjectiveType))
def test_run_opf_execution_for_all_objectives(obj_type: ObjectiveType) -> None:
    """All 4 objectives should successfully solve on a representative network case."""
    topology = _make_topology()
    case, _, _ = topology_to_case(topology, source_bus="sourcebus")

    objective_fn = OBJECTIVES[obj_type.value]
    result = case.run_opf(objective=objective_fn, wrapper="matrix")

    assert result is not None
    assert result.converged is True
    assert result.objective_value is not None
    assert len(result.voltages) > 0
    assert len(result.active_power_flows) > 0


def test_format_area_label() -> None:
    """format_area_label should produce compact bracketed labels."""
    from distopf_federate.plotting import format_area_label

    area_params = {
        0: {"objective": "minimize_loss"},
        1: {"objective": "minimize_curtail"},
        2: {"objective": "maximize_gen"},
        3: {"objective": "minimize_load"},
        4: {},
    }
    assert format_area_label(0, area_params) == "Area 0 (Loss Min)"
    assert format_area_label(1, area_params) == "Area 1 (Curtail Min)"
    assert format_area_label(2, area_params) == "Area 2 (Gen Max)"
    assert format_area_label(3, area_params) == "Area 3 (Load Min)"
    assert format_area_label(0, area_params, multiline=True) == "Area 0\n(Loss Min)"
    assert format_area_label(4, area_params) == "Area 4"
    assert format_area_label(5, None) == "Area 5"


def test_load_scenario_parameters_extracts_objectives() -> None:
    """load_scenario_parameters should correctly extract objective strings for all areas."""
    from distopf_federate.plotting import load_scenario_parameters

    sample_scenario = {
        "components": [
            {
                "name": "pnnl_dopf_admm_0",
                "type": "DOPFADMMComponent",
                "parameters": {"source_bus": "150", "objective": "minimize_loss"},
            },
            {
                "name": "pnnl_dopf_admm_1",
                "type": "DOPFADMMComponent",
                "parameters": {"source_bus": "18", "objective": "minimize_curtail"},
            },
            {
                "name": "pnnl_dopf_admm_2",
                "type": "DOPFADMMComponent",
                "parameters": {"source_bus": "13"},  # Default
            },
        ]
    }
    area_ids, area_params = load_scenario_parameters(sample_scenario)
    assert area_ids == [0, 1, 2]
    assert area_params[0]["objective"] == "minimize_loss"
    assert area_params[1]["objective"] == "minimize_curtail"
    assert area_params[2]["objective"] == "maximize_gen"

