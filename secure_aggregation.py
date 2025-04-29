#!/usr/bin/env python
"""
Simulated secure aggregation for federated learning
"""
import numpy as np

def secure_aggregation(client_parameters, encryption_key=42):
    """
    Simulate secure aggregation of model parameters without revealing individual values
    
    This is a simplified simulation of secure aggregation. In practice, secure aggregation
    would use more sophisticated cryptographic methods.
    
    Args:
        client_parameters: List of parameter dictionaries from clients
        encryption_key: Simulated encryption key
        
    Returns:
        Aggregated parameters
    """
    # Number of clients
    n_clients = len(client_parameters)
    
    # Generate random masks for each client
    np.random.seed(encryption_key)
    client_masks = []
    
    # Step 1: Each client adds a random mask to their parameters
    masked_parameters = []
    for i, params in enumerate(client_parameters):
        # Create a random mask (sum of all masks should be zero)
        if i < n_clients - 1:
            coef_mask = np.random.normal(0, 1, params["coef"].shape)
            intercept_mask = np.random.normal(0, 1)
            client_masks.append({"coef": coef_mask, "intercept": intercept_mask})
        else:
            # The last client gets a mask that ensures all masks sum to zero
            coef_mask = -np.sum([mask["coef"] for mask in client_masks], axis=0)
            intercept_mask = -np.sum([mask["intercept"] for mask in client_masks])
            client_masks.append({"coef": coef_mask, "intercept": intercept_mask})
        
        # Apply mask
        masked_coef = params["coef"] + coef_mask
        masked_intercept = params["intercept"] + intercept_mask
        
        masked_parameters.append({
            "coef": masked_coef,
            "intercept": masked_intercept
        })
    
    # Step 2: Server aggregates the masked parameters
    # The masks cancel out during aggregation
    agg_coef = np.mean([params["coef"] for params in masked_parameters], axis=0)
    agg_intercept = np.mean([params["intercept"] for params in masked_parameters])
    
    return {
        "coef": agg_coef,
        "intercept": agg_intercept
    }
