#!/usr/bin/env python3
"""
Create sample data for the dashboard demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_data():
    """Create sample simulation data."""
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Generate time series data
    start_time = datetime.now() - timedelta(hours=1)
    time_points = []
    current_time = start_time
    
    # Generate 3600 data points (1 hour at 1-second intervals)
    for i in range(3600):
        time_points.append(current_time)
        current_time += timedelta(seconds=1)
    
    # Create realistic traffic patterns
    np.random.seed(42)
    
    # Simple simulation data
    simple_data = {
        'step': np.arange(3600),
        'vehicles': np.random.poisson(6, 3600) + np.sin(np.arange(3600) * 2 * np.pi / 3600) * 2,
        'waiting_vehicles': np.random.poisson(1, 3600),
        'avg_vehicle_wait': np.random.exponential(0.5, 3600)
    }
    
    # Fixed-time controller data (slightly worse performance)
    fixed_data = {
        'step': np.arange(3600),
        'vehicles': np.random.poisson(6, 3600) + np.sin(np.arange(3600) * 2 * np.pi / 3600) * 2,
        'pedestrians': np.random.poisson(3, 3600),
        'waiting_vehicles': np.random.poisson(2, 3600),
        'waiting_pedestrians': np.random.poisson(1, 3600),
        'avg_vehicle_wait': np.random.exponential(1.2, 3600),
        'avg_pedestrian_wait': np.random.exponential(2.5, 3600),
        'phase': np.random.choice([0, 1, 2, 3], 3600),
        'phase_duration': np.random.randint(20, 40, 3600),
        'time_in_phase': np.random.randint(0, 30, 3600)
    }
    
    # RL controller data (better performance)
    rl_data = {
        'step': np.arange(3600),
        'vehicles': np.random.poisson(6, 3600) + np.sin(np.arange(3600) * 2 * np.pi / 3600) * 2,
        'pedestrians': np.random.poisson(3, 3600),
        'waiting_vehicles': np.random.poisson(1, 3600),
        'waiting_pedestrians': np.random.poisson(0, 3600),
        'avg_vehicle_wait': np.random.exponential(0.8, 3600),
        'avg_pedestrian_wait': np.random.exponential(1.8, 3600),
        'phase': np.random.choice([0, 1, 2, 3], 3600),
        'phase_duration': np.random.randint(15, 35, 3600),
        'time_in_phase': np.random.randint(0, 25, 3600),
        'state': ['(2,1,1,0)'] * 3600,
        'q_table_size': np.arange(50, 3600 + 50),
        'epsilon': np.linspace(0.1, 0.05, 3600)
    }
    
    # Create DataFrames
    simple_df = pd.DataFrame(simple_data)
    fixed_df = pd.DataFrame(fixed_data)
    rl_df = pd.DataFrame(rl_data)
    
    # Save to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    simple_file = f"results/simple_results_{timestamp}.csv"
    fixed_file = f"results/fixed_time_results_{timestamp}.csv"
    rl_file = f"results/rl_results_{timestamp}.csv"
    
    simple_df.to_csv(simple_file, index=False)
    fixed_df.to_csv(fixed_file, index=False)
    rl_df.to_csv(rl_file, index=False)
    
    print(f"✓ Sample data created:")
    print(f"  - Simple simulation: {simple_file}")
    print(f"  - Fixed-time controller: {fixed_file}")
    print(f"  - RL controller: {rl_file}")
    
    return simple_df, fixed_df, rl_df

if __name__ == "__main__":
    create_sample_data()