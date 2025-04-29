#!/usr/bin/env python
"""
Simple federated learning implementation for smart grid data
without dependencies on Flower or TensorFlow
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, household_id=None, test_size=0.2, random_state=42):
    """Load and preprocess data for a specific household"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for specific household if provided
    if household_id is not None:
        df = df[df['household_id'] == household_id]
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Features and target
    features = ['hour', 'day_of_week', 'month', 'is_weekend']
    if 'power_factor' in df.columns:
        features.append('power_factor')
    
    X = df[features].values
    y = df['energy_consumption_kwh'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'x_train': X_train,
        'y_train': y_train,
        'x_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': features
    }

class SimpleClient:
    """A simple federated learning client using LinearRegression"""
    
    def __init__(self, data, client_id):
        """Initialize client with dataset"""
        self.data = data
        self.client_id = client_id
        self.model = LinearRegression()
        
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
        
    def set_parameters(self, parameters):
        """Set model parameters"""
        self.model.coef_ = parameters["coef"]
        self.model.intercept_ = parameters["intercept"]
        
    def __str__(self):
        return f"Client {self.client_id}"

def aggregate_parameters(clients):
    """Aggregate model parameters from multiple clients using weighted averaging"""
    all_parameters = [client.get_parameters() for client in clients]
    
    # Calculate average coefficients and intercept
    avg_coef = np.mean([params["coef"] for params in all_parameters], axis=0)
    avg_intercept = np.mean([params["intercept"] for params in all_parameters])
    
    return {
        "coef": avg_coef,
        "intercept": avg_intercept
    }

def run_federated_learning(data_path, num_clients=5, num_rounds=5, output_dir="results"):
    """Run a simple federated learning simulation"""
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Get unique households
    households = df['household_id'].unique()
    
    # Sample households for clients
    selected_households = np.random.choice(households, num_clients, replace=False)
    
    print(f"Selected households for FL simulation: {selected_households}")
    
    # Create clients
    clients = []
    for i, household in enumerate(selected_households):
        data = load_and_preprocess_data(data_path, household_id=household)
        clients.append(SimpleClient(data, f"Client-{i+1}"))
    
    # Metrics for tracking
    rounds = []
    train_metrics = []
    test_metrics = []
    
    # Run federated learning
    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}/{num_rounds}")
        rounds.append(round_num)
        
        # Train each client locally
        for client in clients:
            client.fit()
            print(f"  {client} trained")
        
        # Collect and aggregate parameters
        global_parameters = aggregate_parameters(clients)
        
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
    plt.title('Federated Learning: Test MAE Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'fl_test_loss.png'))
    
    # Save results to CSV
    results_df = pd.DataFrame({
        "round": rounds,
        "test_mae": [m["mae"] for m in test_metrics],
        "test_mse": [m["mse"] for m in test_metrics]
    })
    
    results_df.to_csv(os.path.join(output_dir, 'fl_metrics.csv'), index=False)
    
    # Save simulation info
    with open(os.path.join(output_dir, 'simulation_info.txt'), 'w') as f:
        f.write(f"Federated Learning Simulation\n")
        f.write(f"Number of clients: {len(clients)}\n")
        f.write(f"Selected households: {', '.join(selected_households)}\n")
        f.write(f"Final test MAE: {test_metrics[-1]['mae']:.4f}\n")
    
    print(f"\nResults saved to {output_dir}")
    
    return {
        "rounds": rounds,
        "test_metrics": test_metrics,
        "households": selected_households
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simplified federated learning simulation")
    parser.add_argument("--data", type=str, default="data/raw/synthetic_smart_meter_data.csv",
                      help="Path to the CSV data file")
    parser.add_argument("--clients", type=int, default=5,
                      help="Number of clients to simulate")
    parser.add_argument("--rounds", type=int, default=5,
                      help="Number of federated learning rounds")
    parser.add_argument("--output", type=str, default="results",
                      help="Output directory for results")
    
    args = parser.parse_args()
    
    run_federated_learning(args.data, args.clients, args.rounds, args.output)
