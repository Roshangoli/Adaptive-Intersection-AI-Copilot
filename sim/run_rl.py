#!/usr/bin/env python3
"""
Reinforcement Learning traffic controller simulation for Adaptive Intersection AI Copilot.
This script runs a SUMO simulation with an RL-based traffic light controller.
"""

import os
import sys
import time
import numpy as np
import traci
import sumolib
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

class RLTrafficController:
    """RL-based traffic light controller using a simple Q-learning approach."""
    
    def __init__(self):
        # Traffic light phases
        self.phases = [
            "GGrrrrGGrrrr",  # North-South green
            "yyyrrryyyrrr",  # North-South yellow
            "rrrGGGrrrGGG",  # East-West green
            "rrryyyrrryyy"   # East-West yellow
        ]
        
        # RL parameters
        self.learning_rate = 0.1
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.9    # Discount factor
        self.min_phase_time = 5  # Minimum time per phase (seconds)
        
        # Q-table (state -> action -> value)
        self.q_table = {}
        
        # Current state
        self.current_phase = 0
        self.phase_start_time = 0
        self.last_action = 0
        
    def get_state(self) -> Tuple[int, int, int, int]:
        """Get current state representation."""
        # State: (vehicle_count_ns, vehicle_count_ew, pedestrian_count_ns, pedestrian_count_ew)
        vehicles_ns = len([v for v in traci.vehicle.getIDList() 
                          if self._is_vehicle_on_edge(v, ['E0', 'E1'])])
        vehicles_ew = len([v for v in traci.vehicle.getIDList() 
                          if self._is_vehicle_on_edge(v, ['E4', 'E5'])])
        
        pedestrians_ns = len([p for p in traci.person.getIDList() 
                             if self._is_pedestrian_on_edge(p, ['E0', 'E1'])])
        pedestrians_ew = len([p for p in traci.person.getIDList() 
                             if self._is_pedestrian_on_edge(p, ['E4', 'E5'])])
        
        # Discretize state (bin the counts)
        vehicles_ns_bin = min(vehicles_ns // 2, 4)  # 0-4 bins
        vehicles_ew_bin = min(vehicles_ew // 2, 4)
        pedestrians_ns_bin = min(pedestrians_ns // 2, 4)
        pedestrians_ew_bin = min(pedestrians_ew // 2, 4)
        
        return (vehicles_ns_bin, vehicles_ew_bin, pedestrians_ns_bin, pedestrians_ew_bin)
    
    def _is_vehicle_on_edge(self, vehicle_id: str, edges: List[str]) -> bool:
        """Check if vehicle is on any of the specified edges."""
        try:
            edge = traci.vehicle.getRoadID(vehicle_id)
            return edge in edges
        except:
            return False
    
    def _is_pedestrian_on_edge(self, pedestrian_id: str, edges: List[str]) -> bool:
        """Check if pedestrian is on any of the specified edges."""
        try:
            edge = traci.person.getRoadID(pedestrian_id)
            return edge in edges
        except:
            return False
    
    def get_action(self, state: Tuple[int, int, int, int]) -> int:
        """Get action (next phase) using epsilon-greedy policy."""
        state_key = str(state)
        
        # Initialize Q-table for new state
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * len(self.phases)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(0, len(self.phases))
        else:
            # Exploit: best action
            action = np.argmax(self.q_table[state_key])
        
        return action
    
    def calculate_reward(self, state: Tuple[int, int, int, int], action: int) -> float:
        """Calculate reward based on current state and action."""
        # Reward components
        reward = 0.0
        
        # Penalty for waiting vehicles
        waiting_vehicles = len([v for v in traci.vehicle.getIDList() 
                              if traci.vehicle.getWaitingTime(v) > 0])
        reward -= waiting_vehicles * 0.1
        
        # Penalty for waiting pedestrians
        waiting_pedestrians = len([p for p in traci.person.getIDList() 
                                 if traci.person.getWaitingTime(p) > 0])
        reward -= waiting_pedestrians * 0.2  # Higher penalty for pedestrians
        
        # Bonus for throughput
        total_vehicles = traci.vehicle.getIDCount()
        total_pedestrians = traci.person.getIDCount()
        reward += (total_vehicles + total_pedestrians) * 0.01
        
        # Penalty for phase changes (encourage stability)
        if action != self.current_phase:
            reward -= 0.5
        
        return reward
    
    def update_q_table(self, state: Tuple[int, int, int, int], action: int, 
                      reward: float, next_state: Tuple[int, int, int, int]):
        """Update Q-table using Q-learning."""
        state_key = str(state)
        next_state_key = str(next_state)
        
        # Initialize Q-table for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * len(self.phases)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * len(self.phases)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def update(self, step: int) -> Dict:
        """Update traffic light based on RL policy."""
        current_state = self.get_state()
        
        # Check if minimum phase time has elapsed
        if step - self.phase_start_time >= self.min_phase_time * 10:  # Convert to steps
            # Get action from RL policy
            action = self.get_action(current_state)
            
            # Calculate reward for previous action
            if hasattr(self, 'last_state'):
                reward = self.calculate_reward(self.last_state, self.last_action)
                self.update_q_table(self.last_state, self.last_action, reward, current_state)
            
            # Update phase
            self.current_phase = action
            self.phase_start_time = step
            self.last_action = action
            self.last_state = current_state
        
        # Set traffic light state
        traci.trafficlight.setRedYellowGreenState("J0", self.phases[self.current_phase])
        
        return {
            'phase': self.current_phase,
            'phase_duration': step - self.phase_start_time,
            'state': current_state,
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon
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
    """Run the RL-based simulation."""
    print("Starting RL-Based Traffic Controller Simulation...")
    
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
        controller = RLTrafficController()
        
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
                      f"Pedestrians: {metrics['pedestrians']}, "
                      f"Q-table size: {controller_info['q_table_size']}")
        
        # Close TraCI connection
        traci.close()
        print("✓ Simulation completed successfully")
        
        # Save results
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"../results/rl_results_{timestamp}.csv"
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
            print(f"Final Q-table size: {controller.q_table_size}")
        
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