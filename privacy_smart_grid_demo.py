#!/usr/bin/env python
"""
Integrated demo of privacy-preserving federated learning for smart grids
with differential privacy, secure aggregation, and blockchain
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Import our modules
# Make sure to copy the necessary functions from the previous files
# or import them if you've structured them as proper modules

# Output directory
os.makedirs("results_integrated", exist_ok=True)

def main():
    """Run the integrated privacy-preserving smart grid demo"""
    print("=== Privacy-Preserving Smart Grid Integrated Demo ===")
    
    # Generate data
    df = generate_synthetic_data(num_households=15, days=7)
    
    # Initialize blockchain
    blockchain = Blockchain()
    
    # Privacy parameters
    epsilon = 1.0  # Differential privacy parameter
    
    # Run federated learning with privacy
    results = run_federated_learning_with_privacy(
        df, 
        num_clients=5, 
        num_rounds=5, 
        epsilon=epsilon,
        blockchain=blockchain
    )
    
    # Print blockchain
    print("\nBlockchain Record:")
    blockchain.print_chain()
    
    print("\nDemo completed successfully!")
    print(f"Final test MAE: {results['test_metrics'][-1]['mae']:.4f}")

if __name__ == "__main__":
    main()
