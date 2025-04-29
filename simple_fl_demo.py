#!/usr/bin/env python
"""
Self-contained federated learning demo for smart grids with privacy
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
os.makedirs("results", exist_ok=True)

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

def run_federated_learning(df, num_clients=5, num_rounds=5):
    """Run a simple federated learning simulation"""
    print(f"\nRunning federated learning with {num_clients} clients for {num_rounds} rounds...")
    
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
    plt.savefig('results/fl_test_loss.png')
    
    # Save results to CSV
    results_df = pd.DataFrame({
        "round": rounds,
        "test_mae": [m["mae"] for m in test_metrics],
        "test_mse": [m["mse"] for m in test_metrics]
    })
    
    results_df.to_csv('results/fl_metrics.csv', index=False)
    
    print(f"\nResults saved to 'results' directory")
    
    return {
        "rounds": rounds,
        "test_metrics": test_metrics,
        "households": selected_households
    }

def main():
    """Run the complete demo"""
    print("=== Privacy-Preserving Smart Grid Federated Learning Demo ===")
    
    # Generate synthetic data
    df = generate_synthetic_data(num_households=15, days=7)
    
    # Run federated learning
    results = run_federated_learning(df, num_clients=5, num_rounds=5)
    
    print("\nDemo completed successfully!")
    print(f"Final test MAE: {results['test_metrics'][-1]['mae']:.4f}")

if __name__ == "__main__":
    main()
