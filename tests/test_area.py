import sys
from pathlib import Path

# Import module directly from source tree to avoid heavy package side effects.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "admm_federate"))

from area import area_info, check_network_radiality, graph_process  # noqa: E402


def test_check_network_radiality_true_for_tree() -> None:
    bus = {"a": {}, "b": {}, "c": {}}
    branch = {"ab": {}, "bc": {}}

    assert check_network_radiality(branch, bus) is True


def test_check_network_radiality_false_for_cycle() -> None:
    bus = {"a": {}, "b": {}, "c": {}}
    branch = {"ab": {}, "bc": {}, "ca": {}}

    assert check_network_radiality(branch, bus) is False


def test_check_network_radiality_disconnected_cyclic() -> None:
    # A graph with 4 nodes and 3 edges that contains a cycle and a disconnected component.
    # Nodes: a, b, c (connected in cycle), d (disconnected).
    # Since the heuristic only checks len(bus) - len(branch) == 1, it will evaluate to True.
    bus = {"a": {}, "b": {}, "c": {}, "d": {}}
    branch = {"ab": {}, "bc": {}, "ca": {}}

    assert check_network_radiality(branch, bus) is True


def test_graph_process_tracks_switch_edges() -> None:
    branch_info = {
        "l1": {"fr_bus": "a", "to_bus": "b", "type": "LINE"},
        "sw1": {"fr_bus": "b", "to_bus": "c", "type": "SWITCH"},
    }

    graph, open_switches = graph_process(branch_info)

    assert set(graph.nodes()) == {"a", "b"}
    assert {tuple(sorted(e)) for e in graph.edges()} == {("a", "b")}
    assert open_switches == [["b", "c"]]


def test_area_info_processes_buses_and_branches() -> None:
    # Set up bus_info:
    # "a" is primary (> 0.12 kv), connected to source
    # "b" is secondary (< 0.12 kv), connected to "a" via line
    # "c" is secondary, connected to "b" via split-phase branch
    # "d" is secondary, connected to "c" via tpx-line branch
    # "e" is disconnected primary
    bus_info = {
        "a": {
            "kv": 12.47,
            "phases": [1, 2, 3],
            "pv": [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
            "pq": [[0.5, 0.05], [0.5, 0.05], [0.5, 0.05]],
        },
        "b": {
            "kv": 0.08,
            "phases": [1, 2],
            "pv": [1.0, 0.1],
            "pq": [0.5, 0.05],
        },
        "c": {
            "kv": 0.08,
            "phases": [1, 2],
            "pv": [1.0, 0.1],
            "pq": [0.5, 0.05],
        },
        "d": {
            "kv": 0.08,
            "phases": [1, 2],
            "pv": [1.0, 0.1],
            "pq": [0.5, 0.05],
        },
        "e": {
            "kv": 12.47,
            "phases": [1, 2, 3],
            "pv": [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]],
            "pq": [[0.5, 0.05], [0.5, 0.05], [0.5, 0.05]],
        },
    }

    # Set up branch_info
    branch_info = {
        "l1": {
            "fr_bus": "a",
            "to_bus": "b",
            "type": "LINE",
            "phases": [1, 2, 3],
            "zprim": [[0.1, 0.0], [0.0, 0.1], [0.0, 0.0]],
        },
        "l2": {
            "fr_bus": "b",
            "to_bus": "c",
            "type": "SPLIT_PHASE",
            "phases": [1, 2],
            "impedance": [0.05, 0.01],
            "impedance1": [0.05, 0.01],
        },
        "l3": {
            "fr_bus": "c",
            "to_bus": "d",
            "type": "TPX_LINE",
            "phases": [1, 2],
            "zprim": [0.05, 0.01],
        },
        # switch to a disconnected node, should be open
        "sw_open": {
            "fr_bus": "a",
            "to_bus": "e",
            "type": "SWITCH",
        },
    }

    # Execute area_info (source_bus is "a")
    branch_data, bus_data = area_info(branch_info, bus_info, source_bus="a")

    # Assertions on bus_data
    # "a", "b", "c", "d" should be in bus_data, but "e" should not (since it's disconnected via the SWITCH)
    assert "a" in bus_data
    assert "b" in bus_data
    assert "c" in bus_data
    assert "d" in bus_data
    assert "e" not in bus_data

    # Check indices are sequential
    assert bus_data["a"]["idx"] == 0
    assert bus_data["b"]["idx"] == 1
    assert bus_data["c"]["idx"] == 2
    assert bus_data["d"]["idx"] == 3

    # Assert primary bus scaling and fields
    assert bus_data["a"]["kv"] == 12.47
    assert bus_data["a"]["s_rated"] == 3.0
    assert bus_data["a"]["pv"] == [[1.0, 0.1], [1.0, 0.1], [1.0, 0.1]]

    # Assert secondary bus scaling and fields
    assert bus_data["b"]["kv"] == 0.08
    assert bus_data["b"]["s_rated"] == 1.0
    assert bus_data["b"]["pv"] == [1.0, 0.1]

    # Assertions on branch_data
    # "l1" (LINE), "l2" (SPLIT_PHASE), "l3" (TPX_LINE) should be present.
    # "sw_open" (SWITCH) should not be present.
    assert "l1" in branch_data
    assert "l2" in branch_data
    assert "l3" in branch_data
    assert "sw_open" not in branch_data

    # Check indices (primary and secondary loops index separately)
    assert branch_data["l1"]["idx"] == 0
    assert branch_data["l2"]["idx"] == 0
    assert branch_data["l3"]["idx"] == 1

    assert branch_data["l1"]["zprim"] == [[0.1, 0.0], [0.0, 0.1], [0.0, 0.0]]
    assert branch_data["l2"]["impedance"] == [0.05, 0.01]
    assert branch_data["l2"]["impedance1"] == [0.05, 0.01]
    assert branch_data["l3"]["impedance"] == [0.05, 0.01]
