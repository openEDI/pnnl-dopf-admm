import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import module directly from source tree to avoid heavy package side effects.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "admm_federate"))

from opf_federate import ComponentParameters, OPFFederate  # noqa: E402


def test_load_static_inputs(tmp_path) -> None:
    static_inputs = {
        "name": "test_admm",
        "vup_tol": 0.01,
        "sdn_tol": 0.01,
        "max_itr": 10,
        "t_steps": 24,
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
            assert fed.static.t_steps == 24
            assert fed.static.source_bus == "150"
            assert fed.static.source_line == ""
            assert fed.deltat == 3600
            assert fed.admm_config.rho_vup == 1000.0
            assert fed.admm_config.relaxed is False
