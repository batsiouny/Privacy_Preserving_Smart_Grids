# data/simulation/scripts/generate_synthetic_data.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def generate_synthetic_smart_meter_data(num_households=100, days=30, 
                                       start_date=datetime(2023, 1, 1),
                                       sampling_rate='15min',
                                       output_dir='data/raw',
                                       seed=42):
    """
    Generate synthetic smart meter data simulating energy consumption patterns
    
    Args:
        num_households: Number of households to simulate
        days: Number of days of data
        start_date: Starting date for the simulation
        sampling_rate: Time between consecutive readings
        output_dir: Directory to save output data
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic smart meter data
    """
    np.random.seed(seed)
    
    # Create timestamp range
    end_date = start_date + timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq=sampling_rate)
    
    # Create empty dataframe
    data = []
    
    # Base load patterns with daily and weekly seasonality
    for household_id in range(num_households):
        # Generate household characteristics
        base_consumption = np.random.uniform(0.2, 0.8)  # Base power in kW
        morning_peak = np.random.uniform(1.0, 3.0)      # Morning peak additional power
        evening_peak = np.random.uniform(2.0, 5.0)      # Evening peak additional power
        weekend_factor = np.random.uniform(1.1, 1.5)    # Increased usage on weekends
        
        # Household occupancy patterns
        num_occupants = np.random.randint(1, 6)  # 1 to 5 occupants
        has_electric_vehicle = np.random.random() < 0.2  # 20% chance of EV
        has_solar_panels = np.random.random() < 0.15     # 15% chance of solar panels
        
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
                
            # Electric vehicle charging (typically evening)
            if has_electric_vehicle and 19 <= hour <= 23:
                consumption += np.random.uniform(3.0, 7.0) * np.random.binomial(1, 0.3)  # 30% chance of charging
                
            # Solar panel generation (daytime)
            if has_solar_panels and 9 <= hour <= 16:
                solar_generation = np.random.uniform(1.0, 2.0) * np.random.normal(1.0, 0.3)
                consumption = max(0.05, consumption - solar_generation)  # Ensure positive consumption
                
            # Add some noise
            consumption += np.random.normal(0, 0.05)
            consumption = max(0.05, consumption)  # Ensure positive consumption
            
            # Calculate energy from power (kWh)
            time_delta_hours = pd.Timedelta(sampling_rate).total_seconds() / 3600
            energy_kwh = consumption * time_delta_hours
            
            data.append({
                'timestamp': ts,
                'household_id': f'H{household_id:03d}',
                'energy_consumption_kwh': energy_kwh,
                'power_kw': consumption,
                'voltage': np.random.normal(120, 0.5),
                'current_amp': consumption * 1000 / 120,  # I = P/V
                'power_factor': np.random.uniform(0.85, 0.98),
                'has_ev': has_electric_vehicle,
                'has_solar': has_solar_panels,
                'num_occupants': num_occupants
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def save_synthetic_data(num_households=100, days=30, output_dir='data/raw', filename='synthetic_smart_meter_data.csv'):
    """Generate and save synthetic data"""
    os.makedirs(output_dir, exist_ok=True)
    
    df = generate_synthetic_smart_meter_data(num_households=num_households, days=days)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    
    print(f'Synthetic data saved to {output_path}')
    print(f'Generated {len(df)} records for {df["household_id"].nunique()} households')
    
    return df

def visualize_data(df, output_dir='data/processed'):
    """Create visualization of the synthetic data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot daily consumption patterns for a few households
    plt.figure(figsize=(12, 6))
    
    # Select 5 random households
    sample_households = np.random.choice(df['household_id'].unique(), 5, replace=False)
    
    # Filter for a single day
    start_date = df['timestamp'].min()
    end_date = start_date + pd.Timedelta(days=1)
    day_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)]
    
    for household in sample_households:
        household_df = day_df[day_df['household_id'] == household]
        plt.plot(household_df['timestamp'], household_df['power_kw'], label=household)
    
    plt.title('Daily Power Consumption Patterns')
    plt.xlabel('Time')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'daily_consumption_patterns.png'))
    
    # 2. Plot weekday vs weekend patterns for a household
    plt.figure(figsize=(12, 6))
    
    # Select a random household
    household = np.random.choice(df['household_id'].unique())
    household_df = df[df['household_id'] == household]
    
    # Group by hour and weekday/weekend
    household_df['hour'] = household_df['timestamp'].dt.hour
    household_df['is_weekend'] = household_df['timestamp'].dt.dayofweek >= 5
    
    # Aggregate by hour and weekend status
    hourly_avg = household_df.groupby(['hour', 'is_weekend'])['power_kw'].mean().unstack()
    
    hourly_avg.plot(figsize=(12, 6))
    plt.title(f'Weekday vs Weekend Power Consumption for {household}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Power (kW)')
    plt.legend(['Weekday', 'Weekend'])
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'weekday_weekend_comparison.png'))
    
    print(f'Visualizations saved to {output_dir}')
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic smart meter data')
    parser.add_argument('--households', type=int, default=100,
                       help='Number of households to simulate')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days of data')
    parser.add_argument('--output', type=str, default='data/raw/synthetic_smart_meter_data.csv',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Generate data
    df = save_synthetic_data(
        num_households=args.households,
        days=args.days,
        output_dir=os.path.dirname(args.output),
        filename=os.path.basename(args.output)
    )
    
    # Create visualizations
    visualize_data(df)