#!/usr/bin/env python
"""
Integrated Privacy-Preserving Smart Grid System

This script demonstrates a complete privacy-preserving solution for smart grids with:
1. Federated Learning: Decentralized model training without sharing raw data
2. Differential Privacy: Adding noise to protect against inference attacks
3. Secure Aggregation: Masking individual updates for aggregation privacy
4. Blockchain: Recording the learning process for auditability and transparency
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
os.makedirs("results_integrated", exist_ok=True)
os.makedirs("results_integrated/visualizations", exist_ok=True)
os.makedirs("results_integrated/metrics", exist_ok=True)

# Performance tracking
performance_metrics = {
    "computation_time": [],
    "communication_overhead": [],
    "privacy_budget_spent": [],
    "rounds": []
}

class Block:
    """Block for the blockchain"""
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
        
        mining_start = time.time()
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
            
        mining_time = time.time() - mining_start
        print(f"  Block #{self.index} mined in {mining_time:.2f}s: {self.hash[:10]}...")
        return mining_time

class Blockchain:
    """Blockchain implementation"""
    def __init__(self, difficulty=2):
        """Initialize the blockchain with the genesis block"""
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty
        self.mining_times = []
        
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
        mining_time = new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        self.mining_times.append(mining_time)
        
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
    
    def export_to_file(self, filename="results_integrated/blockchain.json"):
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
        
        # Also export mining times
        with open("results_integrated/metrics/mining_times.csv", 'w') as f:
            f.write("block_index,mining_time_seconds\n")
            for i, time in enumerate(self.mining_times):
                f.write(f"{i+1},{time:.4f}\n")

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

class SmartMeterClient:
    """A client in the federated learning system"""
    
    def __init__(self, data, client_id):
        """Initialize client with dataset"""
        self.data = data
        self.client_id = client_id
        self.model = LinearRegression()
        self.prev_parameters = None
        
    def fit(self):
        """Train the model on local data"""
        start_time = time.time()
        self.model.fit(self.data['x_train'], self.data['y_train'])
        training_time = time.time() - start_time
        return training_time
        
    def evaluate(self):
        """Evaluate the model on local test data"""
        start_time = time.time()
        predictions = self.model.predict(self.data['x_test'])
        mse = np.mean((predictions - self.data['y_test']) ** 2)
        mae = mean_absolute_error(self.data['y_test'], predictions)
        evaluation_time = time.time() - start_time
        
        return {"mse": mse, "mae": mae, "evaluation_time": evaluation_time}
        
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
        
        # Calculate update size in bytes (simplified)
        update_size = (current_params["coef"].size * 8) + 8  # 8 bytes per float64
        
        return update, update_size
        
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

def add_laplace_noise(parameters, epsilon=1.0):
    """
    Add Laplace noise to model parameters for differential privacy
    
    Args:
        parameters: Dictionary with model parameters
        epsilon: Privacy parameter (smaller = more privacy)
        
    Returns:
        Dictionary with noisy parameters
    """
    if epsilon is None:  # No privacy
        return parameters.copy(), 0
        
    sensitivity = 1.0  # Assumed sensitivity for demonstration
    
    # Calculate scale of Laplace noise
    scale = sensitivity / epsilon
    
    # Add noise to coefficients
    noise_coef = np.random.laplace(0, scale, parameters["coef"].shape)
    noisy_coef = parameters["coef"] + noise_coef
    
    # Add noise to intercept
    noise_intercept = np.random.laplace(0, scale)
    noisy_intercept = parameters["intercept"] + noise_intercept
    
    # Calculate privacy budget spent (simplified)
    privacy_cost = 1.0 / epsilon
    
    return {
        "coef": noisy_coef,
        "intercept": noisy_intercept
    }, privacy_cost

def secure_aggregation(client_updates, encryption_key=42):
    """
    Simulate secure aggregation of model updates without revealing individual updates
    
    Args:
        client_updates: List of parameter updates from clients
        encryption_key: Seed for random mask generation (simulated encryption)
        
    Returns:
        Aggregated updates
    """
    # Number of clients
    n_clients = len(client_updates)
    print(f"  Securely aggregating updates from {n_clients} clients...")
    
    start_time = time.time()
    
    # Initialize random number generator for reproducibility
    np.random.seed(encryption_key)
    
    # Extract updates from the tuple (update, size)
    updates = [u[0] for u in client_updates]
    
    # Step 1: Generate random masks for each client
    coef_shape = updates[0]["coef"].shape
    
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
    for i, update in enumerate(updates):
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
    
    secure_agg_time = time.time() - start_time
    
    # Calculate secure aggregation overhead (simplified)
    # Each client sends masked update and receives masks
    communication_overhead = sum([u[1] for u in client_updates]) * 2  # bytes
    
    return {
        "coef": agg_coef,
        "intercept": agg_intercept
    }, secure_agg_time, communication_overhead

def record_fl_round(blockchain, round_num, global_parameters, metrics, 
                   participating_clients, privacy_params=None):
    """Record federated learning round in the blockchain"""
    # Simplify parameters for storage (full parameters would be too large)
    simplified_params = {
        "coef_mean": float(np.mean(global_parameters["coef"])),
        "coef_std": float(np.std(global_parameters["coef"])),
        "intercept": float(global_parameters["intercept"])
    }
    
    # Create privacy info
    privacy_info = None
    if privacy_params:
        privacy_info = {
            "mechanism": privacy_params["mechanism"],
            "epsilon": privacy_params["epsilon"],
            "budget_spent": privacy_params["budget_spent"]
        }
    
    # Create data to store
    data = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "parameters_summary": simplified_params,
        "metrics": {
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "computation_time": metrics["computation_time"],
            "communication_overhead": metrics["communication_overhead"]
        },
        "num_clients": len(participating_clients),
        "client_ids": [client.client_id for client in participating_clients],
        "privacy": privacy_info
    }
    
    # Add to blockchain
    print(f"  Recording round {round_num} data to blockchain...")
    new_block = blockchain.add_block(data)
    
    return new_block

def run_integrated_fl(df, num_clients=5, num_rounds=5, epsilon=1.0, 
                     use_secure_agg=True, use_blockchain=True, use_dp=True,
                     blockchain_difficulty=2):
    """Run integrated federated learning with privacy and blockchain"""
    print(f"\nRunning integrated federated learning with:")
    print(f"  - {'Differential Privacy (ε=' + str(epsilon) + ')' if use_dp else 'No Differential Privacy'}")
    print(f"  - {'Secure Aggregation' if use_secure_agg else 'Standard Aggregation'}")
    print(f"  - {'Blockchain Recording' if use_blockchain else 'No Blockchain'}")
    
    # Initialize blockchain if needed
    blockchain = None
    if use_blockchain:
        blockchain = Blockchain(difficulty=blockchain_difficulty)
        
        # Record initialization in blockchain
        init_data = {
            "event": "FL_INITIALIZATION",
            "timestamp": datetime.now().isoformat(),
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "privacy_mechanism": "DP" if use_dp else "None",
            "differential_privacy_epsilon": epsilon if use_dp else None,
            "secure_aggregation": use_secure_agg
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
        clients.append(SmartMeterClient(data, f"Client-{i+1}"))
    
    # Metrics for tracking
    rounds = []
    test_metrics = []
    total_privacy_budget_spent = 0
    
    # Global model parameters
    global_parameters = None
    
    # Run federated learning
    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}/{num_rounds}")
        rounds.append(round_num)
        
        round_start_time = time.time()
        
        # Train each client locally
        training_times = []
        for client in clients:
            training_time = client.fit()
            training_times.append(training_time)
        
        # Collect parameter updates from clients
        client_updates = []
        for client in clients:
            update, update_size = client.get_parameter_update()
            client_updates.append((update, update_size))
        
        # Aggregate parameters (with or without secure aggregation)
        if use_secure_agg:
            aggregated_parameters, secure_agg_time, comm_overhead = secure_aggregation(client_updates)
        else:
            # Standard aggregation
            start_time = time.time()
            all_params = [update[0] for update in client_updates]
            avg_coef = np.mean([params["coef"] for params in all_params], axis=0)
            avg_intercept = np.mean([params["intercept"] for params in all_params])
            aggregated_parameters = {
                "coef": avg_coef,
                "intercept": avg_intercept
            }
            secure_agg_time = time.time() - start_time
            comm_overhead = sum([u[1] for u in client_updates])  # bytes
            
        # Apply differential privacy if needed
        privacy_budget_spent = 0
        if use_dp:
            global_parameters, privacy_budget_spent = add_laplace_noise(aggregated_parameters, epsilon)
            total_privacy_budget_spent += privacy_budget_spent
        else:
            global_parameters = aggregated_parameters
        
        # Update each client with global parameters
        for client in clients:
            client.set_parameters(global_parameters)
        
        # Evaluate performance
        evaluation_metrics = []
        for client in clients:
            metrics = client.evaluate()
            evaluation_metrics.append(metrics)
        
        # Average metrics across clients
        avg_metrics = {
            "mae": np.mean([m["mae"] for m in evaluation_metrics]),
            "mse": np.mean([m["mse"] for m in evaluation_metrics]),
            "computation_time": np.mean(training_times) + np.mean([m["evaluation_time"] for m in evaluation_metrics]),
            "communication_overhead": comm_overhead
        }
        
        test_metrics.append(avg_metrics)
        print(f"  Average MAE: {avg_metrics['mae']:.4f}")
        
        # Record performance metrics
        performance_metrics["computation_time"].append(avg_metrics["computation_time"])
        performance_metrics["communication_overhead"].append(avg_metrics["communication_overhead"])
        performance_metrics["privacy_budget_spent"].append(privacy_budget_spent)
        performance_metrics["rounds"].append(round_num)
        
        # Record round in blockchain if enabled
        if use_blockchain:
            privacy_params = None
            if use_dp:
                privacy_params = {
                    "mechanism": "Laplace",
                    "epsilon": epsilon,
                    "budget_spent": privacy_budget_spent
                }
            
            record_fl_round(blockchain, round_num, global_parameters, 
                           avg_metrics, clients, privacy_params)
        
        round_end_time = time.time()
        print(f"  Round completed in {round_end_time - round_start_time:.2f} seconds")
    
    # Record completion in blockchain if enabled
    if use_blockchain:
        completion_data = {
            "event": "FL_COMPLETION",
            "timestamp": datetime.now().isoformat(),
            "final_metrics": test_metrics[-1],
            "total_privacy_budget_spent": total_privacy_budget_spent,
            "num_blocks": len(blockchain.chain)
        }
        blockchain.add_block(completion_data)
        
        # Export blockchain to file
        blockchain.export_to_file()
        
        # Verify blockchain integrity
        is_valid = blockchain.is_chain_valid()
        print(f"\nBlockchain integrity check: {'Valid' if is_valid else 'Invalid'}")
    
    # Save performance metrics to file
    df_metrics = pd.DataFrame({
        "round": performance_metrics["rounds"],
        "computation_time": performance_metrics["computation_time"],
        "communication_overhead": performance_metrics["communication_overhead"],
        "privacy_budget_spent": performance_metrics["privacy_budget_spent"]
    })
    df_metrics.to_csv("results_integrated/metrics/performance_metrics.csv", index=False)
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, [m["mae"] for m in test_metrics], marker='o')
    title = "Integrated Privacy-Preserving Federated Learning"
    if use_dp:
        title += f" with DP (ε={epsilon})"
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    plt.savefig('results_integrated/visualizations/integrated_performance.png')
    
    # Return results
    return {
        "rounds": rounds,
        "test_metrics": test_metrics,
        "households": selected_households,
        "final_parameters": global_parameters,
        "blockchain": blockchain if use_blockchain else None,
        "total_privacy_budget_spent": total_privacy_budget_spent
    }

def run_comparison(df, num_clients=5, num_rounds=5):
    """Run a comparison of different privacy configurations"""
    print("\n=== Running Privacy Configuration Comparison ===")
    
    # Configuration variants to compare
    configs = [
        {"name": "No Privacy", "use_dp": False, "use_secure_agg": False, "epsilon": None},
        {"name": "DP Only (ε=1.0)", "use_dp": True, "use_secure_agg": False, "epsilon": 1.0},
        {"name": "Secure Agg Only", "use_dp": False, "use_secure_agg": True, "epsilon": None},
        {"name": "DP (ε=1.0) + Secure Agg", "use_dp": True, "use_secure_agg": True, "epsilon": 1.0},
        {"name": "DP (ε=5.0) + Secure Agg", "use_dp": True, "use_secure_agg": True, "epsilon": 5.0}
    ]
    
    results = {}
    for config in configs:
        print(f"\n--- Testing configuration: {config['name']} ---")
        
        # Disable blockchain for comparison to speed up execution
        result = run_integrated_fl(
            df, 
            num_clients=num_clients, 
            num_rounds=num_rounds,
            epsilon=config["epsilon"],
            use_secure_agg=config["use_secure_agg"],
            use_dp=config["use_dp"],
            use_blockchain=False,  # Disable blockchain for comparison
            blockchain_difficulty=1  # Lower difficulty for faster runs
        )
        
        results[config["name"]] = result
    
    # Create comparison visualization
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        plt.plot(result["rounds"], [m["mae"] for m in result["test_metrics"]], 
                marker='o', label=name)
    
    plt.title('Privacy-Utility Tradeoff Comparison')
    plt.xlabel('Round')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_integrated/visualizations/privacy_comparison.png')
    
    # Create performance comparison
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        mae_values = [m["mae"] for m in result["test_metrics"]]
        final_mae = mae_values[-1]
        
        # Get computation time and communication overhead
        computation_times = performance_metrics["computation_time"][-num_rounds:]
        communication_overheads = performance_metrics["communication_overhead"][-num_rounds:]
        
        avg_computation = np.mean(computation_times)
        avg_communication = np.mean(communication_overheads)
        
        plt.scatter(avg_computation, final_mae, label=name, s=100)
    
    plt.title('Computation Time vs. Accuracy')
    plt.xlabel('Average Computation Time per Round (seconds)')
    plt.ylabel('Final Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_integrated/visualizations/computation_vs_accuracy.png')
    
    # Create comprehensive comparison table
    comparison_data = []
    for name, result in results.items():
        final_mae = result["test_metrics"][-1]["mae"]
        privacy_level = "High" if name.startswith("DP (ε=1.0)") else \
                       "Medium" if name.startswith("DP (ε=5.0)") else \
                       "None" if name == "No Privacy" else "Low"
                       
        comparison_data.append({
            "Configuration": name,
            "Final MAE": final_mae,
            "Privacy Level": privacy_level,
            "Total Privacy Budget": result.get("total_privacy_budget_spent", 0),
            "Secure Aggregation": "Yes" if "Secure Agg" in name else "No"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("results_integrated/metrics/configuration_comparison.csv", index=False)
    
    print("\nComparison complete!")
    print(f"Results saved to results_integrated/visualizations/")
    
    return results

if __name__ == "__main__":
    print("=== Integrated Privacy-Preserving Smart Grid System ===")
    
    # Generate synthetic data
    df = generate_synthetic_data(num_households=15, days=7)
    
    # Run the integrated federated learning (full version with all privacy mechanisms)
    results = run_integrated_fl(
        df, 
        num_clients=5, 
        num_rounds=5,
        epsilon=1.0,
        use_secure_agg=True,
        use_blockchain=True,
        use_dp=True,
        blockchain_difficulty=2
    )
    
    # Run comparison of different privacy configurations
    comparison_results = run_comparison(df, num_clients=5, num_rounds=3)
    
    print("\nIntegrated demo completed successfully!")
    print(f"Final test MAE: {results['test_metrics'][-1]['mae']:.4f}")
    if results['blockchain']:
        print(f"Blockchain length: {len(results['blockchain'].chain)} blocks")
    
    print("\nCheck the results_integrated directory for all outputs and visualizations.")