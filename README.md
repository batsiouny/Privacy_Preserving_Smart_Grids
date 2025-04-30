# Privacy-Preserving Smart Grid System

A comprehensive implementation of privacy-preserving mechanisms for smart grid data analysis, combining federated learning, differential privacy, secure aggregation, and blockchain.

## Project Overview

This project implements innovative techniques to ensure data privacy in smart grids while enabling real-time monitoring, optimization, and decision-making. The focus is on implementing privacy-preserving mechanisms using federated learning for distributed data analysis and blockchain for secure data exchange and auditability.

### Key Components

1. **Federated Learning (FL)**: Enables collaborative model training without sharing raw data
2. **Differential Privacy (DP)**: Protects against inference attacks by adding calibrated noise
3. **Secure Aggregation**: Prevents the server from seeing individual model updates
4. **Blockchain Integration**: Provides auditability and transparency for the learning process

## Repository Structure
privacy-preserving-smart-grid/
├── simple_fl_demo.py                # Basic federated learning implementation
├── simple_fl_with_dp.py             # Federated learning with differential privacy
├── secure_aggregation.py            # Secure aggregation implementation
├── blockchain_integration.py        # Blockchain integration for FL
├── integrated_privacy_grid.py       # Complete integrated system
├── create_dashboard.py              # Dashboard creation script
├── analyze_performance.py           # Performance analysis script
├── dashboard/                       # Visualization dashboard
├── results_integrated/              # Results from integrated system
│   ├── metrics/                     # Performance metrics
│   ├── visualizations/              # Visualizations of results
│   └── blockchain.json              # Blockchain record
└── performance_analysis/            # Detailed performance analysis
## Features

- **Decentralized Data Analysis**: FL enables model training across distributed smart meters without centralizing raw data
- **Enhanced Privacy**: Multiple privacy mechanisms protect user data from different threats
- **Auditability**: Blockchain provides immutable records of model updates and training process
- **Visualization Dashboard**: Interactive dashboard for visualizing system performance and privacy-utility tradeoffs

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/privacy-preserving-smart-grid.git
cd privacy-preserving-smart-grid

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn matplotlib tensorflow seaborn
