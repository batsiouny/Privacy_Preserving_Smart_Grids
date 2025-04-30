#!/usr/bin/env python
"""
Create a visualization dashboard for the privacy-preserving smart grid system.
This script generates an HTML dashboard with interactive visualizations.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Create directory for dashboard
os.makedirs("dashboard", exist_ok=True)

def load_data():
    """Load result data from the integrated system"""
    data = {}
    
    # Load performance metrics
    try:
        data['performance'] = pd.read_csv("results_integrated/metrics/performance_metrics.csv")
    except FileNotFoundError:
        print("Warning: Performance metrics file not found")
        data['performance'] = None
    
    # Load configuration comparison
    try:
        data['configurations'] = pd.read_csv("results_integrated/metrics/configuration_comparison.csv")
    except FileNotFoundError:
        print("Warning: Configuration comparison file not found")
        data['configurations'] = None
    
    # Load blockchain data
    try:
        with open("results_integrated/blockchain.json", "r") as f:
            data['blockchain'] = json.load(f)
    except FileNotFoundError:
        print("Warning: Blockchain data file not found")
        data['blockchain'] = None
    
    # Load mining times
    try:
        data['mining_times'] = pd.read_csv("results_integrated/metrics/mining_times.csv")
    except FileNotFoundError:
        print("Warning: Mining times file not found")
        data['mining_times'] = None
    
    return data

def create_privacy_utility_plot(configurations_df):
    """Create privacy-utility tradeoff visualization"""
    if configurations_df is None:
        return None
        
    plt.figure(figsize=(10, 6))
    
    # Try to sort configurations by privacy level if possible
    try:
        privacy_order = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
        configurations_df['privacy_order'] = configurations_df['Privacy Level'].map(privacy_order)
        sorted_df = configurations_df.sort_values('privacy_order')
    except KeyError:
        sorted_df = configurations_df  # Use as is if columns not found
    
    # Plot bars
    try:
        bars = plt.bar(sorted_df['Configuration'], sorted_df['Final MAE'], color=sns.color_palette("viridis", len(sorted_df)))
        
        # Add privacy level as color gradient if possible
        try:
            for i, privacy in enumerate(sorted_df['Privacy Level']):
                if privacy == "High":
                    bars[i].set_color('#d62728')  # Red for high privacy
                elif privacy == "Medium":
                    bars[i].set_color('#ff7f0e')  # Orange for medium
                elif privacy == "Low":
                    bars[i].set_color('#2ca02c')  # Green for low
                else:
                    bars[i].set_color('#1f77b4')  # Blue for none
        except:
            pass  # Skip coloring if there's an issue
    except:
        # Fallback if the expected columns aren't available
        plt.text(0.5, 0.5, "Privacy utility data not in expected format",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    
    plt.xlabel('Privacy Configuration')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Privacy-Utility Tradeoff')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('dashboard/privacy_utility_tradeoff.png', dpi=300)
    return 'privacy_utility_tradeoff.png'

def create_performance_comparison(performance_df):
    """Create performance metrics visualization"""
    if performance_df is None:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    try:
        # Computation time
        ax1.plot(performance_df['round'], performance_df['computation_time'], 
                marker='o', linewidth=2, label='Computation Time')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Computation Time (seconds)')
        ax1.set_title('Computation Time per Round')
        ax1.grid(True)
        
        # Communication overhead
        comm_overhead = performance_df['communication_overhead'] / 1024  # Convert to KB
        ax2.plot(performance_df['round'], comm_overhead, 
                marker='o', linewidth=2, color='orange', label='Communication Overhead')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Communication Overhead (KB)')
        ax2.set_title('Communication Overhead per Round')
        ax2.grid(True)
    except:
        # Fallback if columns don't exist
        ax1.text(0.5, 0.5, "Computation time data not available",
                horizontalalignment='center', verticalalignment='center')
        ax2.text(0.5, 0.5, "Communication overhead data not available",
                horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    plt.savefig('dashboard/performance_metrics.png', dpi=300)
    return 'performance_metrics.png'

def create_privacy_budget_plot(performance_df):
    """Create privacy budget visualization"""
    if performance_df is None or 'privacy_budget_spent' not in performance_df.columns:
        return None
    
    plt.figure(figsize=(10, 5))
    
    try:
        # Cumulative privacy budget
        cumulative_budget = np.cumsum(performance_df['privacy_budget_spent'])
        
        plt.plot(performance_df['round'], cumulative_budget, 
                marker='o', linewidth=2, color='purple')
        plt.xlabel('Round')
        plt.ylabel('Cumulative Privacy Budget Spent')
        plt.title('Privacy Budget Expenditure Over Time')
        plt.grid(True)
    except:
        plt.text(0.5, 0.5, "Privacy budget data not available",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    plt.savefig('dashboard/privacy_budget.png', dpi=300)
    return 'privacy_budget.png'

def create_blockchain_visualizations(blockchain_data):
    """Create blockchain-related visualizations"""
    if blockchain_data is None:
        return None
    
    # Extract round metrics from blockchain
    rounds = []
    mae_values = []
    timestamps = []
    
    try:
        for block in blockchain_data:
            if isinstance(block['data'], dict) and 'round' in block['data']:
                rounds.append(block['data']['round'])
                mae_values.append(block['data']['metrics']['mae'])
                timestamps.append(block['timestamp'])
        
        if not rounds:
            return None
        
        plt.figure(figsize=(10, 5))
        
        plt.plot(rounds, mae_values, marker='o', linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title('Model Performance Recorded in Blockchain')
        plt.grid(True)
    except:
        plt.text(0.5, 0.5, "Blockchain performance data not available",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    plt.savefig('dashboard/blockchain_performance.png', dpi=300)
    return 'blockchain_performance.png'

def create_mining_times_plot(mining_times_df):
    """Create mining times visualization"""
    if mining_times_df is None:
        return None
    
    plt.figure(figsize=(10, 5))
    
    try:
        plt.bar(mining_times_df['block_index'], mining_times_df['mining_time_seconds'], color='teal')
        plt.xlabel('Block Index')
        plt.ylabel('Mining Time (seconds)')
        plt.title('Blockchain Mining Time per Block')
        plt.grid(True, axis='y')
    except:
        plt.text(0.5, 0.5, "Mining times data not available",
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    plt.savefig('dashboard/mining_times.png', dpi=300)
    return 'mining_times.png'

def generate_html_dashboard(visualizations):
    """Generate HTML dashboard from visualizations"""
    # Use double braces to escape CSS style braces
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Privacy-Preserving Smart Grid Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                text-align: center;
                padding: 20px 0;
                margin-bottom: 30px;
                background-color: #4b6cb7;
                color: white;
                border-radius: 5px;
            }}
            .section {{
                background-color: white;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
                color: #4b6cb7;
            }}
            .visualization {{
                text-align: center;
                margin: 20px 0;
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }}
            .visualization-description {{
                margin-top: 10px;
                font-style: italic;
                color: #666;
            }}
            footer {{
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                color: #666;
                font-size: 0.8em;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                grid-gap: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Privacy-Preserving Smart Grid Dashboard</h1>
            <p>Visualizing the performance and privacy tradeoffs in smart grid federated learning</p>
        </div>
        
        <div class="section">
            <h2>Privacy-Utility Tradeoff</h2>
            <div class="visualization">
                <img src="{privacy_utility_tradeoff}" alt="Privacy-Utility Tradeoff">
                <div class="visualization-description">
                    This chart shows how different privacy configurations affect model accuracy.
                    Higher privacy typically results in lower utility (higher error).
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            <div class="visualization">
                <img src="{performance_metrics}" alt="Performance Metrics">
                <div class="visualization-description">
                    These charts show computation time and communication overhead per round.
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Privacy Budget</h2>
            <div class="visualization">
                <img src="{privacy_budget}" alt="Privacy Budget">
                <div class="visualization-description">
                    This chart shows how the privacy budget is spent over time with differential privacy.
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Blockchain Integration</h2>
            <div class="grid">
                <div class="visualization">
                    <img src="{blockchain_performance}" alt="Blockchain Performance">
                    <div class="visualization-description">
                        Model performance metrics recorded in the blockchain.
                    </div>
                </div>
                <div class="visualization">
                    <img src="{mining_times}" alt="Mining Times">
                    <div class="visualization-description">
                        Time required to mine each block in the blockchain.
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Generated on: {date}</p>
            <p>Privacy-Preserving Smart Grid Project</p>
        </footer>
    </body>
    </html>
    """
    
    # Fill in visualization paths
    html_content = html_template.format(
        privacy_utility_tradeoff=visualizations.get('privacy_utility_tradeoff', ''),
        performance_metrics=visualizations.get('performance_metrics', ''),
        privacy_budget=visualizations.get('privacy_budget', ''),
        blockchain_performance=visualizations.get('blockchain_performance', ''),
        mining_times=visualizations.get('mining_times', ''),
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Write to file
    with open("dashboard/index.html", "w") as f:
        f.write(html_content)
    
    print(f"Dashboard generated at dashboard/index.html")

if __name__ == "__main__":
    print("Creating Privacy-Preserving Smart Grid Dashboard...")
    
    # Load data
    data = load_data()
    
    # Create visualizations
    visualizations = {}
    
    if data['configurations'] is not None:
        visualizations['privacy_utility_tradeoff'] = create_privacy_utility_plot(data['configurations'])
    
    if data['performance'] is not None:
        visualizations['performance_metrics'] = create_performance_comparison(data['performance'])
        visualizations['privacy_budget'] = create_privacy_budget_plot(data['performance'])
    
    if data['blockchain'] is not None:
        visualizations['blockchain_performance'] = create_blockchain_visualizations(data['blockchain'])
    
    if data['mining_times'] is not None:
        visualizations['mining_times'] = create_mining_times_plot(data['mining_times'])
    
    # Generate HTML dashboard
    generate_html_dashboard(visualizations)
    
    print("Dashboard creation complete!")
