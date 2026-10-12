"""Simulate 1000 different particle sizes for lithium-ion batteries using PyBaMM."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybamm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Realistic particle size range for lithium-ion batteries: 0.1 - 100 micrometers
# Logarithmically spaced for better coverage
MIN_PARTICLE_SIZE = 0.1e-6  # 0.1 micrometers in meters
MAX_PARTICLE_SIZE = 100e-6  # 100 micrometers in meters
NUM_SIMULATIONS = 1000

# Particle sizes to test (logarithmically spaced)
particle_sizes = np.logspace(
    np.log10(MIN_PARTICLE_SIZE), np.log10(MAX_PARTICLE_SIZE), NUM_SIMULATIONS
)

# Output file for results
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "particle_size_sweep_results.csv"
PLOT_FILE = OUTPUT_DIR / "particle_size_analysis.png"


def run_simulation_for_particle_size(
    particle_size: float,
    simulation_index: int,
    total_simulations: int,
) -> dict:
    """Run a single simulation with a specific particle size.

    Parameters
    ----------
    particle_size : float
        The particle size in meters.
    simulation_index : int
        Current simulation index (for logging).
    total_simulations : int
        Total number of simulations.

    Returns:
    -------
    dict
        Dictionary containing simulation results.

    """
    try:
        # Create model
        model = pybamm.lithium_ion.DFN()

        # Use Chen2020 parameters as base
        parameter_values = pybamm.ParameterValues("Chen2020")

        # Update particle sizes for both negative and positive electrodes
        # PyBaMM uses a parameter for particle radius
        parameter_values["Negative particle radius [m]"] = particle_size
        parameter_values["Positive particle radius [m]"] = particle_size

        # Create simple discharge experiment at 1C rate
        experiment = pybamm.Experiment(
            ["Discharge at 1C until 2.5 V"],
            termination="voltage",
        )

        # Create and solve simulation
        simulation = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
        )

        logger.info(
            f"Running simulation {simulation_index}/{total_simulations} "
            f"for particle size {particle_size * 1e6:.4f} μm"
        )

        # Solve the simulation
        solution = simulation.solve(
            save_at_cycles=None,
            calc_esoh=False,
        )

        # Extract key performance metrics
        time = solution.t
        voltage = solution["Terminal voltage [V]"].entries
        capacity = solution["Capacity [A.h]"].entries

        # Calculate metrics
        discharge_time = time[-1]
        final_capacity = capacity[-1] if len(capacity) > 0 else 0
        min_voltage = float(np.min(voltage)) if len(voltage) > 0 else 0
        avg_voltage = float(np.mean(voltage)) if len(voltage) > 0 else 0

        result = {
            "particle_size_m": particle_size,
            "particle_size_um": particle_size * 1e6,
            "discharge_time_s": discharge_time,
            "final_capacity_Ah": final_capacity,
            "min_voltage_V": min_voltage,
            "avg_voltage_V": avg_voltage,
            "success": True,
            "error_message": "",
        }

    except Exception as e:  # noqa: BLE001 -- record failure, keep sweeping
        logger.warning(
            f"Simulation {simulation_index} failed for "
            f"particle size {particle_size * 1e6:.4f} μm: {e!s}"
        )
        return {
            "particle_size_m": particle_size,
            "particle_size_um": particle_size * 1e6,
            "discharge_time_s": None,
            "final_capacity_Ah": None,
            "min_voltage_V": None,
            "avg_voltage_V": None,
            "success": False,
            "error_message": str(e),
        }
    else:
        logger.debug(f"Simulation successful: {result}")
        return result


def run_particle_size_sweep() -> list[dict]:
    """Run simulations for all particle sizes.

    Returns:
    -------
    list[dict]
        List of results for each simulation.

    """
    logger.info(
        f"Starting particle size sweep with {NUM_SIMULATIONS} simulations"
    )
    logger.info(
        f"Particle size range: {MIN_PARTICLE_SIZE * 1e6:.4f} - "
        f"{MAX_PARTICLE_SIZE * 1e6:.4f} μm"
    )

    results = []

    for i, particle_size in enumerate(particle_sizes, 1):
        result = run_simulation_for_particle_size(
            particle_size, i, NUM_SIMULATIONS
        )
        results.append(result)

        # Progress indicator
        if i % 100 == 0:
            logger.info(
                f"Progress: {i}/{NUM_SIMULATIONS} simulations completed"
            )

    logger.info("Particle size sweep completed!")
    return results


def save_results(results: list[dict]) -> None:
    """Save simulation results to CSV file.

    Parameters
    ----------
    results : list[dict]
        List of simulation results.

    """
    logger.info(f"Saving results to {RESULTS_FILE}")

    with RESULTS_FILE.open("w", newline="") as f:
        fieldnames = [
            "particle_size_m",
            "particle_size_um",
            "discharge_time_s",
            "final_capacity_Ah",
            "min_voltage_V",
            "avg_voltage_V",
            "success",
            "error_message",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to {RESULTS_FILE}")


def plot_results(results: list[dict]) -> None:
    """Create plots analyzing the particle size effects.

    Parameters
    ----------
    results : list[dict]
        List of simulation results.

    """
    logger.info(f"Creating plots and saving to {PLOT_FILE}")

    # Extract successful results
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        logger.warning("No successful simulations to plot!")
        return

    particle_sizes_um = [r["particle_size_um"] for r in successful_results]
    discharge_times = [r["discharge_time_s"] for r in successful_results]
    final_capacities = [r["final_capacity_Ah"] for r in successful_results]
    min_voltages = [r["min_voltage_V"] for r in successful_results]
    avg_voltages = [r["avg_voltage_V"] for r in successful_results]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Particle Size Sweep Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Discharge time vs particle size
    axes[0, 0].semilogx(particle_sizes_um, discharge_times, "b-", linewidth=2)
    axes[0, 0].set_xlabel("Particle Size (μm)", fontsize=11)
    axes[0, 0].set_ylabel("Discharge Time (s)", fontsize=11)
    axes[0, 0].set_title("Discharge Time vs Particle Size")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Final capacity vs particle size
    axes[0, 1].semilogx(particle_sizes_um, final_capacities, "g-", linewidth=2)
    axes[0, 1].set_xlabel("Particle Size (μm)", fontsize=11)
    axes[0, 1].set_ylabel("Final Capacity (A·h)", fontsize=11)
    axes[0, 1].set_title("Final Capacity vs Particle Size")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Minimum voltage vs particle size
    axes[1, 0].semilogx(particle_sizes_um, min_voltages, "r-", linewidth=2)
    axes[1, 0].set_xlabel("Particle Size (μm)", fontsize=11)
    axes[1, 0].set_ylabel("Minimum Voltage (V)", fontsize=11)
    axes[1, 0].set_title("Minimum Voltage vs Particle Size")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Average voltage vs particle size
    axes[1, 1].semilogx(particle_sizes_um, avg_voltages, "m-", linewidth=2)
    axes[1, 1].set_xlabel("Particle Size (μm)", fontsize=11)
    axes[1, 1].set_ylabel("Average Voltage (V)", fontsize=11)
    axes[1, 1].set_title("Average Voltage vs Particle Size")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    logger.info(f"Plots saved to {PLOT_FILE}")
    plt.close()


def print_summary(results: list[dict]) -> None:
    """Print summary statistics of the simulation results.

    Parameters
    ----------
    results : list[dict]
        List of simulation results.

    """
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print("\n" + "=" * 70)
    print("PARTICLE SIZE SWEEP SUMMARY")
    print("=" * 70)
    print(f"Total simulations: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100 * successful / len(results):.1f}%")
    print()

    if successful > 0:
        successful_results = [r for r in results if r["success"]]
        capacities = [r["final_capacity_Ah"] for r in successful_results]
        times = [r["discharge_time_s"] for r in successful_results]
        voltages_min = [r["min_voltage_V"] for r in successful_results]
        voltages_avg = [r["avg_voltage_V"] for r in successful_results]

        print("Capacity Statistics (A·h):")
        print(f"  Min: {min(capacities):.6f}")
        print(f"  Max: {max(capacities):.6f}")
        print(f"  Mean: {np.mean(capacities):.6f}")
        print()

        print("Discharge Time Statistics (s):")
        print(f"  Min: {min(times):.2f}")
        print(f"  Max: {max(times):.2f}")
        print(f"  Mean: {np.mean(times):.2f}")
        print()

        print("Minimum Voltage Statistics (V):")
        print(f"  Min: {min(voltages_min):.4f}")
        print(f"  Max: {max(voltages_min):.4f}")
        print(f"  Mean: {np.mean(voltages_min):.4f}")
        print()

        print("Average Voltage Statistics (V):")
        print(f"  Min: {min(voltages_avg):.4f}")
        print(f"  Max: {max(voltages_avg):.4f}")
        print(f"  Mean: {np.mean(voltages_avg):.4f}")

    print("=" * 70)
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Plots saved to: {PLOT_FILE}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run the particle size sweep
    results = run_particle_size_sweep()

    # Save results to CSV
    save_results(results)

    # Create plots
    plot_results(results)

    # Print summary
    print_summary(results)
