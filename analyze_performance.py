#!/usr/bin/env python
"""
Analyze performance metrics for the privacy-preserving smart grid system.
This script calculates detailed metrics and generates a comprehensive report.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
os.makedirs("performance_analysis", exist_ok=True)

def load_data():
    """Load result data from the integrated system"""
    data = {}
    
    # Load performance metrics
    try:
        data['performance'] = pd.read_csv("results_integrated/metrics/performance_metrics.csv")
        print("Loaded performance metrics")
    except FileNotFoundError:
        print("Warning: Performance metrics file not found")
        data['performance'] = None
    
    # Load configuration comparison
    try:
        data['configurations'] = pd.read_csv("results_integrated/metrics/configuration_comparison.csv")
        print("Loaded configuration comparison")
    except FileNotFoundError:
        print("Warning: Configuration comparison file not found")
        data['configurations'] = None
    
    # Load blockchain data
    try:
        with open("results_integrated/blockchain.json", "r") as f:
            data['blockchain'] = json.load(f)
            print("Loaded blockchain data")
    except FileNotFoundError:
        print("Warning: Blockchain data file not found")
        data['blockchain'] = None
    
    return data

def analyze_performance(data):
    """Analyze performance metrics"""
    performance_df = data['performance']
    
    if performance_df is None:
        print("No performance data available.")
        return {}
    
    # Check which columns are available
    available_columns = performance_df.columns.tolist()
    print(f"Available performance columns: {available_columns}")
    
    # Calculate statistics
    stats = {}
    
    if 'computation_time' in available_columns:
        stats["avg_computation_time"] = performance_df['computation_time'].mean()
        stats["max_computation_time"] = performance_df['computation_time'].max()
        stats["total_computation_time"] = performance_df['computation_time'].sum()
    
    if 'communication_overhead' in available_columns:
        stats["avg_communication_overhead_kb"] = (performance_df['communication_overhead'] / 1024).mean()
        stats["total_communication_overhead_mb"] = (performance_df['communication_overhead'].sum() / (1024 * 1024))
    
    if 'privacy_budget_spent' in available_columns:
        stats["total_privacy_budget"] = performance_df['privacy_budget_spent'].sum()
        stats["avg_privacy_budget_per_round"] = performance_df['privacy_budget_spent'].mean()
    
    # Print statistics
    print("\nPerformance Analysis:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create detailed plots
    create_performance_plots(performance_df, stats)
    
    # Save statistics to CSV
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv("performance_analysis/performance_statistics.csv", index=False)
    
    return stats

def analyze_privacy_utility(data):
    """Analyze privacy-utility tradeoff"""
    config_df = data['configurations']
    
    if config_df is None:
        print("No configuration comparison data available.")
        return None
    
    # Check which columns are available
    available_columns = config_df.columns.tolist()
    print(f"Available configuration columns: {available_columns}")
    
    if not all(col in available_columns for col in ['Configuration', 'Final MAE', 'Privacy Level']):
        print("Missing required columns in configuration data")
        return None
    
    # Print privacy-utility analysis
    print("\nPrivacy-Utility Analysis:")
    for _, row in config_df.iterrows():
        print(f"{row['Configuration']}: MAE = {row['Final MAE']:.4f}, Privacy Level = {row['Privacy Level']}")
    
    # Calculate privacy cost
    if 'No Privacy' in config_df['Configuration'].values:
        baseline_mae = config_df[config_df['Configuration'] == 'No Privacy']['Final MAE'].values[0]
        
        # Calculate utility loss for each configuration
        config_df['Utility Loss (%)'] = (config_df['Final MAE'] - baseline_mae) / baseline_mae * 100
        
        print("\nUtility Loss Analysis:")
        for _, row in config_df.iterrows():
            if row['Configuration'] != 'No Privacy':
                print(f"{row['Configuration']}: Utility loss = {row['Utility Loss (%)']:.2f}%")
    
    # Create privacy-utility tradeoff plot
    create_privacy_utility_plot(config_df)
    
    # Save to CSV
    config_df.to_csv("performance_analysis/privacy_utility_analysis.csv", index=False)
    
    return config_df

def analyze_blockchain(data):
    """Analyze blockchain performance"""
    blockchain_data = data['blockchain']
    
    if blockchain_data is None:
        print("No blockchain data available.")
        return {}
    
    # Calculate statistics
    stats = {}
    stats["total_blocks"] = len(blockchain_data)
    
    if all('nonce' in block for block in blockchain_data):
        stats["avg_nonce"] = np.mean([block['nonce'] for block in blockchain_data])
        stats["max_nonce"] = np.max([block['nonce'] for block in blockchain_data])
    
    # Extract timestamps and calculate average block time
    timestamps = []
    for block in blockchain_data:
        if 'timestamp' in block:
            try:
                timestamps.append(pd.to_datetime(block['timestamp']))
            except:
                pass
    
    if len(timestamps) > 1:
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                      for i in range(len(timestamps)-1)]
        stats["avg_block_time"] = np.mean(time_diffs)
    
    # Print statistics
    print("\nBlockchain Analysis:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create blockchain metrics plot
    create_blockchain_plot(blockchain_data)
    
    # Save statistics to CSV
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv("performance_analysis/blockchain_statistics.csv", index=False)
    
    return stats

def create_performance_plots(performance_df, stats):
    """Create detailed performance plots"""
    available_columns = performance_df.columns.tolist()
    
    # Computation vs Communication plot
    if all(col in available_columns for col in ['computation_time', 'communication_overhead', 'round']):
        plt.figure(figsize=(10, 6))
        plt.scatter(performance_df['computation_time'], 
                   performance_df['communication_overhead'] / 1024,
                   s=100, alpha=0.7)
        
        for i, round_num in enumerate(performance_df['round']):
            plt.annotate(f"Round {round_num}", 
                        (performance_df['computation_time'].iloc[i], 
                         performance_df['communication_overhead'].iloc[i] / 1024),
                        xytext=(10, 5), textcoords='offset points')
        
        plt.xlabel('Computation Time (seconds)')
        plt.ylabel('Communication Overhead (KB)')
        plt.title('Computation Time vs. Communication Overhead')
        plt.grid(True)
        
        plt.savefig('performance_analysis/computation_vs_communication.png', dpi=300)
    
    # Privacy budget vs Round
    if all(col in available_columns for col in ['privacy_budget_spent', 'round']):
        plt.figure(figsize=(10, 6))
        
        plt.bar(performance_df['round'], performance_df['privacy_budget_spent'], 
               alpha=0.7, color='purple')
        
        plt.plot(performance_df['round'], np.cumsum(performance_df['privacy_budget_spent']),
                 marker='o', color='red', linewidth=2, label='Cumulative Budget')
        
        plt.xlabel('Round')
        plt.ylabel('Privacy Budget Spent')
        plt.title('Privacy Budget Expenditure by Round')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('performance_analysis/privacy_budget_by_round.png', dpi=300)

def create_privacy_utility_plot(config_df):
    """Create detailed privacy-utility tradeoff plot"""
    if 'Utility Loss (%)' not in config_df.columns:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create categories based on privacy level
    privacy_categories = config_df['Privacy Level'].unique()
    colors = sns.color_palette("viridis", len(privacy_categories))
    color_map = {level: color for level, color in zip(privacy_categories, colors)}
    
    # Bar plot
    bars = []
    for _, row in config_df.iterrows():
        if row['Configuration'] != 'No Privacy':
            bar = plt.bar(row['Configuration'], row['Utility Loss (%)'],
                        color=color_map[row['Privacy Level']], alpha=0.7)
            bars.append(bar)
    
    plt.xlabel('Configuration')
    plt.ylabel('Utility Loss (%)')
    plt.title('Privacy-Utility Tradeoff: Utility Loss by Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    
    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, color=color_map[level]) 
                      for level in privacy_categories if level != 'None']
    legend_labels = [level for level in privacy_categories if level != 'None']
    plt.legend(legend_elements, legend_labels, title="Privacy Level")
    
    plt.tight_layout()
    
    plt.savefig('performance_analysis/utility_loss_by_config.png', dpi=300)

def create_blockchain_plot(blockchain_data):
    """Create blockchain metrics plot"""
    if not all('nonce' in block for block in blockchain_data):
        return
        
    nonces = [block['nonce'] for block in blockchain_data]
    indices = range(len(blockchain_data))
    
    plt.figure(figsize=(10, 6))
    
    plt.bar(indices, nonces, alpha=0.7, color='teal')
    
    plt.xlabel('Block Index')
    plt.ylabel('Nonce (Mining Difficulty)')
    plt.title('Mining Difficulty by Block')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    
    plt.savefig('performance_analysis/blockchain_mining_difficulty.png', dpi=300)

def generate_report(perf_stats, privacy_utility_df, blockchain_stats):
    """Generate comprehensive performance report"""
    report_content = """
# Performance Analysis Report
## Privacy-Preserving Smart Grid System

This report provides a comprehensive analysis of the performance metrics for the privacy-preserving smart grid system.

## 1. Computational Performance

- **Average computation time per round**: {avg_comp_time:.4f} seconds
- **Maximum computation time**: {max_comp_time:.4f} seconds
- **Total computation time**: {total_comp_time:.4f} seconds

## 2. Communication Overhead

- **Average communication overhead**: {avg_comm_overhead:.2f} KB per round
- **Total communication overhead**: {total_comm_overhead:.4f} MB

## 3. Privacy Budget Expenditure

- **Total privacy budget spent**: {total_privacy_budget:.4f}
- **Average privacy budget per round**: {avg_privacy_budget:.4f}

## 4. Privacy-Utility Tradeoff

| Configuration | Final MAE | Privacy Level | Utility Loss (%) |
|---------------|-----------|--------------|------------------|
{privacy_utility_table}

## 5. Blockchain Performance

- **Total blocks**: {total_blocks}
- **Average nonce (mining difficulty)**: {avg_nonce:.2f}
- **Maximum nonce**: {max_nonce}
- **Average block time**: {avg_block_time:.4f} seconds

## 6. Conclusions

The analysis demonstrates the tradeoffs between privacy, utility, and performance in the smart grid system:

1. **Privacy vs. Utility**: Higher privacy levels (lower epsilon values) result in greater utility loss.
2. **Performance Impact**: Privacy mechanisms add computational overhead and increase communication costs.
3. **Blockchain Overhead**: The blockchain integration adds auditability at the cost of additional computation.

## 7. Recommendations

Based on the analysis, we recommend:

1. Using DP with ε=5.0 for a good balance between privacy and utility
2. Implementing secure aggregation to protect individual updates
3. Adjusting blockchain difficulty based on deployment requirements
"""
    
    # Format privacy utility table
    privacy_utility_table = "No data available"
    
    if privacy_utility_df is not None and 'Utility Loss (%)' in privacy_utility_df.columns:
        privacy_table_rows = []
        for _, row in privacy_utility_df.iterrows():
            config = row['Configuration']
            mae = row['Final MAE']
            privacy_level = row['Privacy Level']
            utility_loss = row['Utility Loss (%)'] if config != 'No Privacy' else 0
            
            privacy_table_rows.append(f"| {config} | {mae:.4f} | {privacy_level} | {utility_loss:.2f} |")
        
        if privacy_table_rows:
            privacy_utility_table = "\n".join(privacy_table_rows)
    
    # Fill in the template with defaults if values are missing
    report_content = report_content.format(
        avg_comp_time=perf_stats.get("avg_computation_time", 0),
        max_comp_time=perf_stats.get("max_computation_time", 0),
        total_comp_time=perf_stats.get("total_computation_time", 0),
        avg_comm_overhead=perf_stats.get("avg_communication_overhead_kb", 0),
        total_comm_overhead=perf_stats.get("total_communication_overhead_mb", 0),
        total_privacy_budget=perf_stats.get("total_privacy_budget", 0),
        avg_privacy_budget=perf_stats.get("avg_privacy_budget_per_round", 0),
        privacy_utility_table=privacy_utility_table,
        total_blocks=blockchain_stats.get("total_blocks", 0),
        avg_nonce=blockchain_stats.get("avg_nonce", 0),
        max_nonce=blockchain_stats.get("max_nonce", 0),
        avg_block_time=blockchain_stats.get("avg_block_time", 0)
    )
    
    # Write to file
    with open("performance_analysis/performance_report.md", "w") as f:
        f.write(report_content)
    
    print(f"Performance report generated at performance_analysis/performance_report.md")

if __name__ == "__main__":
    print("Analyzing Performance of Privacy-Preserving Smart Grid System...")
    
    # Load data
    data = load_data()
    
    # Analyze performance metrics
    perf_stats = analyze_performance(data)
    
    # Analyze privacy-utility tradeoff
    privacy_utility_df = analyze_privacy_utility(data)
    
    # Analyze blockchain performance
    blockchain_stats = analyze_blockchain(data)
    
    # Generate comprehensive report
    generate_report(perf_stats, privacy_utility_df, blockchain_stats)
    
    print("Performance analysis complete!")
