#!/usr/bin/env python
"""
Script to run the complete simple federated learning experiment:
1. Generate synthetic data
2. Run federated learning simulation
"""

import os
import sys
import argparse
import subprocess

def run_experiment(num_households=100, days=30, num_clients=5, num_rounds=5, output_dir='results'):
    """
    Run a complete experiment using simplified federated learning
    """
    print("=== Privacy-Preserving Smart Grid Experiment (Simplified) ===")
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic smart meter data...")
    data_path = f"data/raw/synthetic_smart_meter_data_{num_households}_{days}days.csv"
    
    data_gen_cmd = [
        sys.executable, 
        "data/simulation/scripts/generate_synthetic_data.py",
        "--households", str(num_households),
        "--days", str(days),
        "--output", data_path
    ]
    
    subprocess.run(data_gen_cmd, check=True)
    
    # Step 2: Run federated learning simulation
    print("\n2. Running simplified federated learning simulation...")
    
    fl_sim_cmd = [
        sys.executable,
        "scripts/run/simple_fl.py",
        "--data", data_path,
        "--clients", str(num_clients),
        "--rounds", str(num_rounds),
        "--output", output_dir
    ]
    
    subprocess.run(fl_sim_cmd, check=True)
    
    print(f"\nExperiment completed! Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a simplified smart grid privacy experiment')
    parser.add_argument('--households', type=int, default=100,
                       help='Number of households to simulate')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days of data')
    parser.add_argument('--clients', type=int, default=5,
                       help='Number of clients for federated learning')
    parser.add_argument('--rounds', type=int, default=5,
                       help='Number of federated learning rounds')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run the experiment
    run_experiment(
        args.households,
        args.days,
        args.clients,
        args.rounds,
        args.output
    )
