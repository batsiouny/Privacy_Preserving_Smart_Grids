# federated_learning/client/utils/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, household_id=None, test_size=0.2, random_state=42):
    """
    Load and preprocess data for a single household or all households
    
    Args:
        file_path: Path to the CSV file
        household_id: Optional ID of a specific household to filter
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing preprocessed data
    """
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
    
    # Create lagged features (past 24 hours)
    for i in range(1, 25):
        df[f'energy_lag_{i}'] = df.groupby('household_id')['energy_consumption_kwh'].shift(i)
    
    # Drop rows with NaN values (from lag creation)
    df = df.dropna()
    
    # Features and target
    features = ['hour', 'day_of_week', 'month', 'is_weekend', 'power_factor']
    features.extend([f'energy_lag_{i}' for i in range(1, 25)])
    
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

def create_time_series_data(file_path, household_id=None, look_back=24, test_size=0.2, random_state=42):
    """
    Create time series data for LSTM model
    
    Args:
        file_path: Path to the CSV file
        household_id: Optional ID of a specific household to filter
        look_back: Number of time steps to look back
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing preprocessed time series data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for specific household if provided
    if household_id is not None:
        df = df[df['household_id'] == household_id]
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Features to include
    features = ['energy_consumption_kwh', 'power_kw', 'power_factor']
    
    # Scale features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Create sequences
    sequences = []
    targets = []
    
    for household in df['household_id'].unique():
        household_df = df[df['household_id'] == household][features].values
        
        for i in range(len(household_df) - look_back):
            sequences.append(household_df[i:i+look_back])
            targets.append(household_df[i+look_back, 0])  # energy_consumption_kwh is index 0
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'x_train': X_train,
        'y_train': y_train,
        'x_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': features
    }