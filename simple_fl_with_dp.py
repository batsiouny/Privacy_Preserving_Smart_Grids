#!/usr/bin/env python
"""
Federated learning with differential privacy for smart grid data
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Create output directory
os.makedirs("results_dp", exist_ok=True)

# Function to add Laplace noise for differential privacy
def add_laplace_noise(parameters, epsilon=1.0):
    """
    Add Laplace noise to model parameters for differential privacy
    
    Args:
        parameters: Dictionary with model parameters
        epsilon: Privacy parameter (smaller = more privacy)
        
    Returns:
        Dictionary with noisy parameters
    """
    sensitivity = 1.0  # Assumed sensitivity for demonstration
    
    # Calculate scale of Laplace noise
    scale = sensitivity / epsilon
    
    # Add noise to coefficients
    noisy_coef = parameters["coef"] + np.random.laplace(0, scale, parameters["coef"].shape)
    
    # Add noise to intercept
    noisy_intercept = parameters["intercept"] + np.random.laplace(0, scale)
    
    return {
        "coef": noisy_coef,
        "intercept": noisy_intercept
    }

# The rest of the code can remain the same as in simple_fl_demo.py, 
# but modify the aggregate_parameters function to include DP:

def aggregate_parameters_with_dp(clients, epsilon=1.0):
    """Aggregate model parameters from multiple clients with differential privacy"""
    # Collect parameters from all clients
    all_parameters = [client.get_parameters() for client in clients]
    
    # Calculate average coefficients and intercept
    avg_coef = np.mean([params["coef"] for params in all_parameters], axis=0)
    avg_intercept = np.mean([params["intercept"] for params in all_parameters])
    
    aggregated_params = {
        "coef": avg_coef,
        "intercept": avg_intercept
    }
    
    # Add Laplace noise for differential privacy
    noisy_params = add_laplace_noise(aggregated_params, epsilon)
    
    return noisy_params

# Modify the run_federated_learning function to use the DP version
# by replacing the aggregate_parameters call with aggregate_parameters_with_dp
