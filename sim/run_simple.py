#!/usr/bin/env python3
"""
Simple traffic simulation for Adaptive Intersection AI Copilot.
This script runs a SUMO simulation without traffic light control for basic testing.
"""

import os
import sys
import time
import traci
import sumolib
import pandas as pd
from datetime import datetime

def collect_metrics():
    """Collect simulation metrics."""
    metrics = {
        'step': traci.simulation.getCurrentTime() / 1000.0,  # Convert to seconds
        'vehicles': traci.vehicle.getIDCount(),
        'waiting_vehicles': len([v for v in traci.vehicle.getIDList() 
                               if traci.vehicle.getWaitingTime(v) > 0])
    }
    
    # Calculate average waiting times
    if metrics['vehicles'] > 0:
        metrics['avg_vehicle_wait'] = sum(traci.vehicle.getWaitingTime(v) 
                                        for v in traci.vehicle.getIDList()) / metrics['vehicles']
    else:
        metrics['avg_vehicle_wait'] = 0
    
    return metrics

def run_simulation():
    """Run the simple simulation."""
    print("Starting Simple Traffic Simulation...")
    
    # SUMO configuration
    sumo_cmd = [
        "../venv/bin/sumo", "-c", "simple_grid_sim.sumocfg",
        "--start", "--quit-on-end",
        "--no-step-log", "--no-warnings"
    ]
    
    try:
        # Start TraCI connection
        traci.start(sumo_cmd)
        print("✓ Connected to SUMO simulation")
        
        # Initialize metrics collection
        metrics_data = []
        
        # Simulation loop
        step = 0
        max_steps = 36000  # 1 hour at 0.1s steps
        
        print(f"Running simulation for {max_steps} steps...")
        
        while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
            # Collect metrics
            metrics = collect_metrics()
            metrics_data.append(metrics)
            
            # Advance simulation
            traci.simulationStep()
            step += 1
            
            # Progress update
            if step % 3600 == 0:  # Every 6 minutes
                print(f"Step {step}/{max_steps} - Vehicles: {metrics['vehicles']}")
        
        # Close TraCI connection
        traci.close()
        print("✓ Simulation completed successfully")
        
        # Save results
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"../results/simple_results_{timestamp}.csv"
            df.to_csv(results_file, index=False)
            print(f"✓ Results saved to {results_file}")
            
            # Print summary statistics
            print("\n=== SIMULATION SUMMARY ===")
            print(f"Total simulation time: {df['step'].max():.1f} seconds")
            print(f"Average vehicles: {df['vehicles'].mean():.1f}")
            print(f"Average vehicle wait time: {df['avg_vehicle_wait'].mean():.2f}s")
            print(f"Max vehicle wait time: {df['avg_vehicle_wait'].max():.2f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        try:
            traci.close()
        except:
            pass
        return False

if __name__ == "__main__":
    # Change to simulation directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create results directory if it doesn't exist
    os.makedirs("../results", exist_ok=True)
    
    # Run simulation
    success = run_simulation()
    sys.exit(0 if success else 1)