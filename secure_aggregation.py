#!/usr/bin/env python
"""
Simplified secure aggregation for federated learning

This demonstrates the concept of secure aggregation without revealing
individual model updates to the server.
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
os.makedirs("results_secure_agg", exist_ok=True)

def generate_synthetic_data(num_households=10, days=5):
    """Generate synthetic smart meter data"""
    print(f"Generating synthetic data for {num_households} households over {days} days...")
    
    # Create timestamp range
    start_date = datetime(2023, 1, 1)
    end_date = start_date + timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    data = []
    
    for household_id in range(num_households):
        # Generate household characteristics
        base_consumption = np.random.uniform(0.2, 0.8)
        morning_peak = np.random.uniform(1.0, 3.0)
        evening_peak = np.random.uniform(2.0, 5.0)
        weekend_factor = np.random.uniform(1.1, 1.5)
        
        for ts in timestamps:
            hour = ts.hour
            is_weekend = ts.dayofweek >= 5  # Saturday or Sunday
            
            # Base load
            consumption = base_consumption
            
            # Add time-of-day pattern
            if 6 <= hour <= 9:  # Morning peak
                consumption += morning_peak * np.random.normal(1.0, 0.1)
            elif 17 <= hour <= 22:  # Evening peak
                consumption += evening_peak * np.random.normal(1.0, 0.1)
            elif 23 <= hour or hour <= 5:  # Night valley
                consumption *= 0.5 * np.random.normal(1.0, 0.05)
                
            # Weekend adjustment
            if is_weekend:
                consumption *= weekend_factor
                
            # Add some noise
            consumption += np.random.normal(0, 0.05)
            consumption = max(0.05, consumption)  # Ensure positive consumption
            
            data.append({
                'timestamp': ts,
                'household_id': f'H{household_id:03d}',
                'energy_consumption_kwh': consumption,
                'hour': hour,
                'day_of_week': ts.dayofweek,
                'month': ts.month,
                'is_weekend': is_weekend
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

class SimpleClient:
    """A simple federated learning client using LinearRegression"""
    
    def __init__(self, data, client_id):
        """Initialize client with dataset"""
        self.data = data
        self.client_id = client_id
        self.model = LinearRegression()
        self.prev_parameters = None
        
    def fit(self):
        """Train the model on local data"""
        self.model.fit(self.data['x_train'], self.data['y_train'])
        
    def evaluate(self):
        """Evaluate the model on local test data"""
        predictions = self.model.predict(self.data['x_test'])
        mse = np.mean((predictions - self.data['y_test']) ** 2)
        mae = mean_absolute_error(self.data['y_test'], predictions)
        return {"mse": mse, "mae": mae}
        
    def get_parameters(self):
        """Get model parameters"""
        return {
            "coef": self.model.coef_,
            "intercept": self.model.intercept_
        }
    
    def get_parameter_update(self):
        """Get parameter update (difference from previous parameters)"""
        current_params = self.get_parameters()
        
        if self.prev_parameters is None:
            # First round, return current parameters
            update = current_params
        else:
            # Calculate update (difference from previous parameters)
            update = {
                "coef": current_params["coef"] - self.prev_parameters["coef"],
                "intercept": current_params["intercept"] - self.prev_parameters["intercept"]
            }
        
        # Store current parameters for next round
        self.prev_parameters = {
            "coef": current_params["coef"].copy(),
            "intercept": current_params["intercept"]
        }
        
        return update
        
    def set_parameters(self, parameters):
        """Set model parameters"""
        self.model.coef_ = parameters["coef"]
        self.model.intercept_ = parameters["intercept"]
        
    def __str__(self):
        return f"Client {self.client_id}"

def prepare_data_for_client(df, household_id):
    """Prepare data for a client with the given household ID"""
    # Filter data for the household
    household_df = df[df['household_id'] == household_id].copy()
    
    # Features and target
    features = ['hour', 'day_of_week', 'month', 'is_weekend']
    
    X = household_df[features].values
    y = household_df['energy_consumption_kwh'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return {
        'x_train': X_train,
        'y_train': y_train,
        'x_test': X_test,
        'y_test': y_test
    }

def secure_aggregation(client_updates, encryption_key=42):
    """
    Simulate secure aggregation of model updates without revealing individual updates
    
    This is a simplified simulation of secure aggregation. In production, this would use
    cryptographic techniques like homomorphic encryption or secure multi-party computation.
    
    Args:
        client_updates: List of parameter updates from clients
        encryption_key: Seed for random mask generation (simulated encryption)
        
    Returns:
        Aggregated updates
    """
    # Number of clients
    n_clients = len(client_updates)
    print(f"  Securely aggregating updates from {n_clients} clients...")
    
    # Initialize random number generator for reproducibility
    np.random.seed(encryption_key)
    
    # Step 1: Generate random masks for each client
    #         These masks will sum to zero, so they cancel out during aggregation
    coef_shape = client_updates[0]["coef"].shape
    
    # For each client, generate a random mask
    client_masks = []
    for i in range(n_clients):
        if i < n_clients - 1:
            # Generate random mask for this client
            coef_mask = np.random.normal(0, 1, coef_shape)
            intercept_mask = np.random.normal(0, 1)
            client_masks.append({"coef": coef_mask, "intercept": intercept_mask})
        else:
            # Last client gets mask that ensures sum of all masks is zero
            coef_mask = -np.sum([mask["coef"] for mask in client_masks], axis=0)
            intercept_mask = -np.sum([mask["intercept"] for mask in client_masks])
            client_masks.append({"coef": coef_mask, "intercept": intercept_mask})
    
    # Step 2: Clients add masks to their updates
    masked_updates = []
    for i, update in enumerate(client_updates):
        masked_coef = update["coef"] + client_masks[i]["coef"]
        masked_intercept = update["intercept"] + client_masks[i]["intercept"]
        masked_updates.append({
            "coef": masked_coef,
            "intercept": masked_intercept
        })
    
    # Step 3: Server aggregates masked updates
    # Since masks sum to zero, they cancel out in aggregation
    agg_coef = np.sum([update["coef"] for update in masked_updates], axis=0) / n_clients
    agg_intercept = np.sum([update["intercept"] for update in masked_updates]) / n_clients
    
    return {
        "coef": agg_coef,
        "intercept": agg_intercept
    }

def run_federated_learning_with_secure_agg(df, num_clients=5, num_rounds=5):
    """Run federated learning with secure aggregation"""
    print(f"\nRunning federated learning with secure aggregation...")
    
    # Get unique households
    households = df['household_id'].unique()
    
    # Sample households for clients
    selected_households = np.random.choice(households, num_clients, replace=False)
    
    print(f"Selected households: {selected_households}")
    
    # Create clients
    clients = []
    for i, household in enumerate(selected_households):
        data = prepare_data_for_client(df, household)
        clients.append(SimpleClient(data, f"Client-{i+1}"))
    
    # Metrics for tracking
    rounds = []
    test_metrics = []
    
    # Initialize global model parameters (average of initial client models)
    print("Initializing global model...")
    for client in clients:
        client.fit()  # Initial training
    
    initial_parameters = [client.get_parameters() for client in clients]
    avg_coef = np.mean([params["coef"] for params in initial_parameters], axis=0)
    avg_intercept = np.mean([params["intercept"] for params in initial_parameters])
    
    global_parameters = {
        "coef": avg_coef,
        "intercept": avg_intercept
    }
    
    # Set initial parameters for all clients
    for client in clients:
        client.set_parameters(global_parameters)
    
    # Run federated learning
    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}/{num_rounds}")
        rounds.append(round_num)
        
        # Train each client locally
        for client in clients:
            client.fit()
            
        # Collect parameter updates from clients
        client_updates = [client.get_parameter_update() for client in clients]
        
        # Securely aggregate updates
        aggregated_update = secure_aggregation(client_updates)
        
        # Update global parameters
        global_parameters = {
            "coef": global_parameters["coef"] + aggregated_update["coef"],
            "intercept": global_parameters["intercept"] + aggregated_update["intercept"]
        }
        
        # Update each client with global parameters
        for client in clients:
            client.set_parameters(global_parameters)
        
        # Evaluate performance
        round_metrics = []
        for client in clients:
            metrics = client.evaluate()
            round_metrics.append(metrics)
            print(f"  {client} - MAE: {metrics['mae']:.4f}")
        
        # Average metrics across clients
        avg_metrics = {
            "mae": np.mean([m["mae"] for m in round_metrics]),
            "mse": np.mean([m["mse"] for m in round_metrics])
        }
        
        test_metrics.append(avg_metrics)
        print(f"  Average MAE: {avg_metrics['mae']:.4f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, [m["mae"] for m in test_metrics], marker='o')
    plt.title('Federated Learning with Secure Aggregation')
    plt.xlabel('Round')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    plt.savefig('results_secure_agg/secure_agg_performance.png')
    
    print("\nResults saved to 'results_secure_agg' directory")
    
    return {
        "rounds": rounds,
        "test_metrics": test_metrics,
        "households": selected_households,
        "final_parameters": global_parameters
    }

if __name__ == "__main__":
    print("=== Federated Learning with Secure Aggregation ===")
    
    # Generate synthetic data
    df = generate_synthetic_data(num_households=15, days=7)
    
    # Run federated learning with secure aggregation
    results = run_federated_learning_with_secure_agg(df, num_clients=5, num_rounds=5)
    
    print("\nDemo completed successfully!")
    print(f"Final test MAE: {results['test_metrics'][-1]['mae']:.4f}")