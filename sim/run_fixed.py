#!/usr/bin/env python3
"""
Fixed-time traffic controller simulation for Adaptive Intersection AI Copilot.
This script runs a SUMO simulation with a simple fixed-time traffic light controller.
"""

import os
import sys
import time
import traci
import sumolib
import pandas as pd
from datetime import datetime

class FixedTimeController:
    """Simple fixed-time traffic light controller."""
    
    def __init__(self):
        self.phase_duration = 30  # 30 seconds per phase
        self.current_phase = 0
        self.phase_start_time = 0
        self.phases = [
            "GGrrrrGGrrrr",  # North-South green
            "yyyrrryyyrrr",  # North-South yellow
            "rrrGGGrrrGGG",  # East-West green
            "rrryyyrrryyy"   # East-West yellow
        ]
        
    def update(self, step):
        """Update traffic light based on fixed timing."""
        if step - self.phase_start_time >= self.phase_duration:
            self.current_phase = (self.current_phase + 1) % len(self.phases)
            self.phase_start_time = step
            
        # Set traffic light state
        traci.trafficlight.setRedYellowGreenState("J0", self.phases[self.current_phase])
        
        return {
            'phase': self.current_phase,
            'phase_duration': self.phase_duration,
            'time_in_phase': step - self.phase_start_time
        }

def collect_metrics():
    """Collect simulation metrics."""
    metrics = {
        'step': traci.simulation.getCurrentTime() / 1000.0,  # Convert to seconds
        'vehicles': traci.vehicle.getIDCount(),
        'pedestrians': traci.person.getIDCount(),
        'waiting_vehicles': len([v for v in traci.vehicle.getIDList() 
                               if traci.vehicle.getWaitingTime(v) > 0]),
        'waiting_pedestrians': len([p for p in traci.person.getIDList() 
                                  if traci.person.getWaitingTime(p) > 0])
    }
    
    # Calculate average waiting times
    if metrics['vehicles'] > 0:
        metrics['avg_vehicle_wait'] = sum(traci.vehicle.getWaitingTime(v) 
                                        for v in traci.vehicle.getIDList()) / metrics['vehicles']
    else:
        metrics['avg_vehicle_wait'] = 0
        
    if metrics['pedestrians'] > 0:
        metrics['avg_pedestrian_wait'] = sum(traci.person.getWaitingTime(p) 
                                           for p in traci.person.getIDList()) / metrics['pedestrians']
    else:
        metrics['avg_pedestrian_wait'] = 0
    
    return metrics

def run_simulation():
    """Run the fixed-time simulation."""
    print("Starting Fixed-Time Traffic Controller Simulation...")
    
    # SUMO configuration
    sumo_cmd = [
        "../venv/bin/sumo", "-c", "grid_sim.sumocfg",
        "--start", "--quit-on-end",
        "--no-step-log", "--no-warnings"
    ]
    
    try:
        # Start TraCI connection
        traci.start(sumo_cmd)
        print("✓ Connected to SUMO simulation")
        
        # Initialize controller
        controller = FixedTimeController()
        
        # Initialize metrics collection
        metrics_data = []
        
        # Simulation loop
        step = 0
        max_steps = 36000  # 1 hour at 0.1s steps
        
        print(f"Running simulation for {max_steps} steps...")
        
        while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
            # Update controller
            controller_info = controller.update(step)
            
            # Collect metrics
            metrics = collect_metrics()
            metrics.update(controller_info)
            metrics_data.append(metrics)
            
            # Advance simulation
            traci.simulationStep()
            step += 1
            
            # Progress update
            if step % 3600 == 0:  # Every 6 minutes
                print(f"Step {step}/{max_steps} - Vehicles: {metrics['vehicles']}, "
                      f"Pedestrians: {metrics['pedestrians']}")
        
        # Close TraCI connection
        traci.close()
        print("✓ Simulation completed successfully")
        
        # Save results
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"../results/fixed_time_results_{timestamp}.csv"
            df.to_csv(results_file, index=False)
            print(f"✓ Results saved to {results_file}")
            
            # Print summary statistics
            print("\n=== SIMULATION SUMMARY ===")
            print(f"Total simulation time: {df['step'].max():.1f} seconds")
            print(f"Average vehicles: {df['vehicles'].mean():.1f}")
            print(f"Average pedestrians: {df['pedestrians'].mean():.1f}")
            print(f"Average vehicle wait time: {df['avg_vehicle_wait'].mean():.2f}s")
            print(f"Average pedestrian wait time: {df['avg_pedestrian_wait'].mean():.2f}s")
            print(f"Max vehicle wait time: {df['avg_vehicle_wait'].max():.2f}s")
            print(f"Max pedestrian wait time: {df['avg_pedestrian_wait'].max():.2f}s")
        
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