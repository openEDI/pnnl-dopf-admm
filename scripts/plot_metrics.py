#!/usr/bin/env python3
"""Script to analyze and plot ADMM power flow metrics from OEDISI co-simulation outputs.

This script compares power flows, voltages, and boundary exchange metrics between
the feeder and ADMM areas to validate the performance and convergence of the ADMM OPF.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the component's src directory to sys.path so we can import admm_federate modules
SCRIPT_DIR = Path(__file__).resolve().parent
COMPONENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(COMPONENT_DIR / "src"))

try:
    import matplotlib.pyplot as plt
    from oedisi.types.data_types import Topology
    from admm_federate.adapter import area_disconnects, disconnect_areas, generate_graph
    from admm_federate.plotting import (
        configure_publication_style,
        load_scenario_parameters,
        get_der_mapping,
        load_recorder_data,
        process_voltages,
        process_power_flows,
        process_generation_adequacy,
        process_convergence,
        get_max_diff_timestep,
        plot_voltage_comparison,
        plot_power_flow_comparison,
        plot_generation_adequacy,
        plot_algorithmic_convergence,
        plot_voltage_scatter_at_timestep,
        plot_power_scatter_at_timestep,
        plot_network_partition,
    )
except ImportError as e:
    print(
        f"Error importing admm_federate or oedisi modules: {e}. "
        "Ensure the script is executed within the project virtual environment.",
        file=sys.stderr,
    )
    sys.exit(1)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ADMM power flow metrics.")
    parser.add_argument(
        "scenario_path",
        type=str,
        help="Path to the scenario JSON file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(COMPONENT_DIR.parent.parent / "outputs"),
        help="Path to the directory where recorders saved feather files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(COMPONENT_DIR.parent.parent / "outputs"),
        help="Path to save the generated plots",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    scenario_path = Path(args.scenario_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    if not scenario_path.exists():
        logger.error(f"Scenario file does not exist: {scenario_path}")
        sys.exit(1)

    # Extract model name and configure publication styles
    stem = scenario_path.stem
    if stem.startswith("pnnl_dopf_admm_"):
        model = stem[len("pnnl_dopf_admm_"):]
    elif stem == "pnnl_dopf_admm":
        model = "default"
    else:
        model = stem

    import seaborn as sns
    sns.set_theme(style="whitegrid")
    configure_publication_style()

    # 1. Load configuration and input models
    logger.info(f"Loading scenario configuration from: {scenario_path}")
    area_ids, area_params = load_scenario_parameters(scenario_path)
    if not area_ids:
        logger.error("No ADMM area components found in the scenario configuration.")
        sys.exit(1)

    logger.info(f"Discovered ADMM areas from scenario: {area_ids}")

    topology_path = data_dir / "topology.json"
    if not topology_path.exists():
        # Fallback to check scenario folder structure
        topology_path = (
            scenario_path.parent
            / scenario_path.stem.replace("pnnl_dopf_admm_", "").split("_")[0]
            / "topology.json"
        )

    if not topology_path.exists():
        logger.error(
            f"Topology file topology.json not found in {data_dir} or scenario folder."
        )
        sys.exit(1)

    logger.info(f"Loading grid network model from: {topology_path}")
    try:
        topology = Topology.model_validate_json(topology_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load grid network model: {e}")
        sys.exit(1)

    # 2. Preprocess data (splitting areas, finding boundaries, mapping DERs)
    logger.info("Preprocessing grid network and partitioning areas...")
    try:
        slack_bus = topology.slack_bus[0].split(".", 1)[0]
        G = generate_graph(topology.incidences, slack_bus)

        # Partition grid
        graph_for_partition = G.copy()
        graph_for_split = G.copy()
        boundaries = area_disconnects(graph_for_partition, n_max=len(area_ids))
        areas_clean = disconnect_areas(graph_for_split, boundaries)

        # Sort areas_clean to align with area_ids by matching source_bus
        sorted_areas = [None] * len(area_ids)
        unmatched = []
        for area in areas_clean:
            matched_aid = -1
            for aid in area_ids:
                params = area_params.get(aid, {})
                source_bus = params.get("source_bus")
                if source_bus and source_bus in area.nodes():
                    matched_aid = aid
                    break
            if matched_aid != -1:
                sorted_areas[matched_aid] = area
            else:
                unmatched.append(area)

        for i in range(len(area_ids)):
            if sorted_areas[i] is None and unmatched:
                sorted_areas[i] = unmatched.pop(0)
        areas_clean = [a for a in sorted_areas if a is not None] + unmatched

        area_buses = [list(area.nodes()) for area in areas_clean]
        der_map = get_der_mapping(topology_path)
    except Exception as e:
        logger.error(f"Failed to preprocess grid partition or DER mapping: {e}")
        sys.exit(1)

    # 3. Load simulation results (ingesting recorder data)
    logger.info("Ingesting feather data files...")
    data = load_recorder_data(data_dir, scenario_path)

    # 4. Process metrics and evaluate results
    logger.info("Processing metrics...")
    voltage_data = process_voltages(data, area_ids, area_buses, topology)
    flow_data = process_power_flows(
        data, area_ids, area_params, G, area_buses, der_map, slack_bus
    )
    adequacy_df = process_generation_adequacy(topology, area_ids, area_buses)
    convergence_data = process_convergence(data, area_ids)

    # 5. Plot and save outputs (only those matching the updated example)
    logger.info("Generating and saving plots...")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_timestep = get_max_diff_timestep(data, topology)

    fig_volt = plot_voltage_comparison(voltage_data, timestep=comparison_timestep)
    if fig_volt:
        fig_volt.savefig(output_dir / f"admm_{model}_voltage_comparison.png", dpi=300)
        fig_volt.savefig(output_dir / f"admm_{model}_voltage_comparison.eps")
        plt.close(fig_volt)

    fig_flow = plot_power_flow_comparison(flow_data, timestep=comparison_timestep)
    if fig_flow:
        fig_flow.savefig(output_dir / f"admm_{model}_power_flow_comparison.png", dpi=300)
        fig_flow.savefig(output_dir / f"admm_{model}_power_flow_comparison.eps")
        plt.close(fig_flow)

    fig_adeq = plot_generation_adequacy(adequacy_df)
    if fig_adeq:
        fig_adeq.savefig(output_dir / f"admm_{model}_generation_adequacy.png", dpi=300)
        fig_adeq.savefig(output_dir / f"admm_{model}_generation_adequacy.eps")
        plt.close(fig_adeq)

    fig_conv = plot_algorithmic_convergence(convergence_data)
    if fig_conv:
        fig_conv.savefig(output_dir / f"admm_{model}_convergence.png", dpi=300, bbox_inches="tight")
        fig_conv.savefig(output_dir / f"admm_{model}_convergence.eps", bbox_inches="tight")
        plt.close(fig_conv)

    # 5. Save the scatter plots if reference data is available
    fig_volt_scatter = plot_voltage_scatter_at_timestep(
        data,
        topology,
        timestep_val=comparison_timestep
    )
    if fig_volt_scatter:
        fig_volt_scatter.savefig(output_dir / f"admm_{model}_voltage_scatter.png", dpi=300)
        fig_volt_scatter.savefig(output_dir / f"admm_{model}_voltage_scatter.eps")
        plt.close(fig_volt_scatter)

    fig_power_scatter = plot_power_scatter_at_timestep(
        data,
        timestep_val=comparison_timestep
    )
    if fig_power_scatter:
        fig_power_scatter.savefig(output_dir / f"admm_{model}_power_scatter.png", dpi=300)
        fig_power_scatter.savefig(output_dir / f"admm_{model}_power_scatter.eps")
        plt.close(fig_power_scatter)

    # 6. Generate and save the network partition plot
    fig_partition = plot_network_partition(
        G,
        boundaries,
        areas_clean,
        slack_bus,
        coords_dir=topology_path.parent
    )
    if fig_partition:
        fig_partition.savefig(output_dir / f"admm_{model}_network_partition.png", dpi=300)
        fig_partition.savefig(output_dir / f"admm_{model}_network_partition.eps")
        plt.close(fig_partition)

    print(f"\nAll timestep-dependent plots were generated for timestep: {comparison_timestep}")


if __name__ == "__main__":
    main()
