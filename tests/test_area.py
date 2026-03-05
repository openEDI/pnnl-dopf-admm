import sys
from pathlib import Path

# Import module directly from source tree to avoid heavy package side effects.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "admm_federate"))

from area import check_network_radiality, graph_process  # noqa: E402


def test_check_network_radiality_true_for_tree() -> None:
    bus = {"a": {}, "b": {}, "c": {}}
    branch = {"ab": {}, "bc": {}}

    assert check_network_radiality(branch, bus) is True


def test_check_network_radiality_false_for_cycle() -> None:
    bus = {"a": {}, "b": {}, "c": {}}
    branch = {"ab": {}, "bc": {}, "ca": {}}

    assert check_network_radiality(branch, bus) is False


def test_graph_process_tracks_switch_edges() -> None:
    branch_info = {
        "l1": {"fr_bus": "a", "to_bus": "b", "type": "LINE"},
        "sw1": {"fr_bus": "b", "to_bus": "c", "type": "SWITCH"},
    }

    graph, open_switches = graph_process(branch_info)

    assert set(graph.nodes()) == {"a", "b"}
    assert set(graph.edges()) == {("a", "b")}
    assert open_switches == [["b", "c"]]
