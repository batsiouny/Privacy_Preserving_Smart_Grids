#!/usr/bin/env python
"""
Blockchain integration for federated learning in smart grids

This demonstrates how blockchain can add transparency and auditability
to the federated learning process.
"""
import os
import time
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Create output directories
os.makedirs("results_blockchain", exist_ok=True)

class Block:
    """Simple block implementation for blockchain"""
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()
        
    def calculate_hash(self):
        """Calculate hash of the block contents"""
        # Convert data to a string representation
        if isinstance(self.data, dict):
            data_str = json.dumps(self.data, sort_keys=True, default=str)
        else:
            data_str = str(self.data)
            
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": data_str,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty=2):
        """Mine the block (find a hash with the specified difficulty)"""
        target = '0' * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
            
        print(f"  Block #{self.index} mined: {self.hash}")

class Blockchain:
    """Simple blockchain implementation"""
    def __init__(self, difficulty=2):
        """Initialize the blockchain with the genesis block"""
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty
        
    def create_genesis_block(self):
        """Create the first block in the chain"""
        return Block(0, datetime.now().isoformat(), "Genesis Block", "0")
    
    def get_latest_block(self):
        """Get the most recent block in the chain"""
        return self.chain[-1]
    
    def add_block(self, data):
        """Add a new block to the chain"""
        index = len(self.chain)
        timestamp = datetime.now().isoformat()
        previous_hash = self.get_latest_block().hash
        
        new_block = Block(index, timestamp, data, previous_hash)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        
        return new_block
    
    def is_chain_valid(self):
        """Validate the blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if the current hash is valid
            if current_block.hash != current_block.calculate_hash():
                print(f"Invalid hash in block #{current_block.index}")
                return False
            
            # Check if the previous hash reference is correct
            if current_block.previous_hash != previous_block.hash:
                print(f"Invalid previous hash reference in block #{current_block.index}")
                return False
        
        return True
    
    def print_chain(self):
        """Print the blockchain"""
        for block in self.chain:
            print(f"Block #{block.index}")
            print(f"Timestamp: {block.timestamp}")
            print(f"Data: {block.data}")
            print(f"Hash: {block.hash[:10]}...") # Truncate for readability
            print(f"Previous Hash: {block.previous_hash[:10]}...")
            print("-" * 50)
    
    def export_to_file(self, filename="results_blockchain/blockchain.json"):
        """Export the blockchain to a JSON file"""
        blockchain_data = []
        for block in self.chain:
            block_data = {
                "index": block.index,
                "timestamp": block.timestamp,
                "data": block.data,
                "hash": block.hash,
                "previous_hash": block.previous_hash,
                "nonce": block.nonce
            }
            blockchain_data.append(block_data)
        
        with open(filename, 'w') as f:
            json.dump(blockchain_data, f, indent=2, default=str)
        
        print(f"Blockchain exported to {filename}")

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
    """Aggregate model parameters from multiple clients"""
    all_parameters = [client.get_parameters() for client in clients]
    
    # Calculate average coefficients and intercept
    avg_coef = np.mean([params["coef"] for params in all_parameters], axis=0)
    avg_intercept = np.mean([params["intercept"] for params in all_parameters])
    
    return {
        "coef": avg_coef,
        "intercept": avg_intercept
    }

def record_fl_round(blockchain, round_num, global_parameters, metrics, participating_clients):
    """Record federated learning round in the blockchain"""
    # Simplify parameters for storage (full parameters would be too large)
    simplified_params = {
        "coef_mean": float(np.mean(global_parameters["coef"])),
        "coef_std": float(np.std(global_parameters["coef"])),
        "intercept": float(global_parameters["intercept"])
    }
    
    # Create data to store
    data = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "parameters_summary": simplified_params,
        "metrics": {
            "mae": metrics["mae"],
            "mse": metrics["mse"]
        },
        "num_clients": len(participating_clients),
        "client_ids": [client.client_id for client in participating_clients]
    }
    
    # Add to blockchain
    print(f"  Recording round {round_num} data to blockchain...")
    new_block = blockchain.add_block(data)
    
    return new_block

def run_federated_learning_with_blockchain(df, num_clients=5, num_rounds=5):
    """Run federated learning with blockchain for auditability"""
    print(f"\nRunning federated learning with blockchain integration...")
    
    # Initialize blockchain with mining difficulty of 2
    blockchain = Blockchain(difficulty=2)
    
    # Record initialization in blockchain
    init_data = {
        "event": "FL_INITIALIZATION",
        "timestamp": datetime.now().isoformat(),
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "privacy_mechanism": "None"  # Could be "DP" for differential privacy
    }
    blockchain.add_block(init_data)
    
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
    
    # Global model parameters
    global_parameters = None
    
    # Run federated learning
    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}/{num_rounds}")
        rounds.append(round_num)
        
        # Train each client locally
        for client in clients:
            client.fit()
            
        # Aggregate parameters
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
        
        # Record round in blockchain
        record_fl_round(blockchain, round_num, global_parameters, avg_metrics, clients)
    
    # Record completion in blockchain
    completion_data = {
        "event": "FL_COMPLETION",
        "timestamp": datetime.now().isoformat(),
        "final_metrics": test_metrics[-1],
        "num_blocks": len(blockchain.chain)
    }
    blockchain.add_block(completion_data)
    
    # Export blockchain to file
    blockchain.export_to_file()
    
    # Verify blockchain integrity
    is_valid = blockchain.is_chain_valid()
    print(f"\nBlockchain integrity check: {'Valid' if is_valid else 'Invalid'}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, [m["mae"] for m in test_metrics], marker='o')
    plt.title('Federated Learning with Blockchain Integration')
    plt.xlabel('Round')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    plt.savefig('results_blockchain/blockchain_fl_performance.png')
    
    # Return results
    return {
        "rounds": rounds,
        "test_metrics": test_metrics,
        "households": selected_households,
        "final_parameters": global_parameters,
        "blockchain": blockchain
    }

if __name__ == "__main__":
    print("=== Federated Learning with Blockchain Integration ===")
    
    # Generate synthetic data
    df = generate_synthetic_data(num_households=15, days=7)
    
    # Run federated learning with blockchain
    results = run_federated_learning_with_blockchain(df, num_clients=5, num_rounds=5)
    
    # Print blockchain
    print("\nBlockchain Contents:")
    results["blockchain"].print_chain()
    
    print("\nDemo completed successfully!")
    print(f"Final test MAE: {results['test_metrics'][-1]['mae']:.4f}")