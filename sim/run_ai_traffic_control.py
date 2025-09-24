#!/usr/bin/env python3
"""
AI Traffic Light Controller - Demonstrates AI controlling traffic lights.
This script shows how your AI can make decisions and control traffic lights.
"""

import os
import sys
import time
import traci
import numpy as np
import pandas as pd
from datetime import datetime

class AITrafficController:
    """AI Traffic Light Controller that makes smart decisions."""
    
    def __init__(self):
        """Initialize the AI controller."""
        self.traffic_light_id = "J0"  # We'll create this
        
        # Traffic light states
        self.phases = {
            'north_south_green': "GGrrrrGGrrrr",
            'north_south_yellow': "yyyrrryyyrrr", 
            'east_west_green': "rrrGGGrrrGGG",
            'east_west_yellow': "rrryyyrrryyy",
            'all_red': "rrrrrrrrrrrr"
        }
        
        self.current_phase = 'north_south_green'
        self.phase_start_time = 0
        self.min_phase_time = 5  # Minimum 5 seconds per phase
        
        # AI decision history
        self.decisions = []
        
    def analyze_traffic(self):
        """Analyze current traffic situation."""
        try:
            # Get vehicle counts
            vehicles = traci.vehicle.getIDList()
            pedestrians = traci.person.getIDList()
            
            # Count vehicles by direction (simplified)
            vehicles_ns = len([v for v in vehicles if self._is_vehicle_north_south(v)])
            vehicles_ew = len([v for v in vehicles if self._is_vehicle_east_west(v)])
            
            # Count pedestrians by direction (simplified)
            pedestrians_ns = len([p for p in pedestrians if self._is_pedestrian_north_south(p)])
            pedestrians_ew = len([p for p in pedestrians if self._is_pedestrian_east_west(p)])
            
            # Calculate wait times
            avg_vehicle_wait = sum(traci.vehicle.getWaitingTime(v) for v in vehicles) / max(len(vehicles), 1)
            avg_pedestrian_wait = sum(traci.person.getWaitingTime(p) for p in pedestrians) / max(len(pedestrians), 1)
            
            return {
                'vehicles_ns': vehicles_ns,
                'vehicles_ew': vehicles_ew,
                'pedestrians_ns': pedestrians_ns,
                'pedestrians_ew': pedestrians_ew,
                'total_vehicles': len(vehicles),
                'total_pedestrians': len(pedestrians),
                'avg_vehicle_wait': avg_vehicle_wait,
                'avg_pedestrian_wait': avg_pedestrian_wait
            }
        except:
            return {
                'vehicles_ns': 0, 'vehicles_ew': 0,
                'pedestrians_ns': 0, 'pedestrians_ew': 0,
                'total_vehicles': 0, 'total_pedestrians': 0,
                'avg_vehicle_wait': 0, 'avg_pedestrian_wait': 0
            }
    
    def _is_vehicle_north_south(self, vehicle_id):
        """Check if vehicle is going north-south."""
        try:
            edge = traci.vehicle.getRoadID(vehicle_id)
            return edge in ['e0', 'e1', 'e2', 'e3']
        except:
            return False
    
    def _is_vehicle_east_west(self, vehicle_id):
        """Check if vehicle is going east-west."""
        try:
            edge = traci.vehicle.getRoadID(vehicle_id)
            return edge in ['e4', 'e5', 'e6', 'e7']
        except:
            return False
    
    def _is_pedestrian_north_south(self, pedestrian_id):
        """Check if pedestrian is going north-south."""
        try:
            edge = traci.person.getRoadID(pedestrian_id)
            return edge in ['e0', 'e1', 'e2', 'e3']
        except:
            return False
    
    def _is_pedestrian_east_west(self, pedestrian_id):
        """Check if pedestrian is going east-west."""
        try:
            edge = traci.person.getRoadID(pedestrian_id)
            return edge in ['e4', 'e5', 'e6', 'e7']
        except:
            return False
    
    def make_ai_decision(self, traffic_data, current_time):
        """Make AI decision about traffic light control."""
        # AI Decision Logic
        
        # Priority 1: Safety - if pedestrians waiting too long
        if traffic_data['avg_pedestrian_wait'] > 30:
            if self.current_phase in ['east_west_green', 'east_west_yellow']:
                decision = "Switch to North-South for pedestrian safety"
                new_phase = 'north_south_green'
            else:
                decision = "Extend North-South green for pedestrian safety"
                new_phase = 'north_south_green'
        
        # Priority 2: Efficiency - if heavy vehicle traffic
        elif traffic_data['total_vehicles'] > 8:
            if traffic_data['vehicles_ns'] > traffic_data['vehicles_ew'] + 2:
                decision = "Prioritize North-South vehicle flow"
                new_phase = 'north_south_green'
            elif traffic_data['vehicles_ew'] > traffic_data['vehicles_ns'] + 2:
                decision = "Prioritize East-West vehicle flow"
                new_phase = 'east_west_green'
            else:
                decision = "Balanced vehicle flow"
                new_phase = self.current_phase
        
        # Priority 3: Fairness - balance both directions
        elif traffic_data['total_pedestrians'] > 3:
            if traffic_data['pedestrians_ns'] > traffic_data['pedestrians_ew']:
                decision = "Prioritize North-South pedestrians"
                new_phase = 'north_south_green'
            else:
                decision = "Prioritize East-West pedestrians"
                new_phase = 'east_west_green'
        
        # Default: Cycle through phases
        else:
            if current_time - self.phase_start_time > 30:  # 30 seconds max
                if self.current_phase == 'north_south_green':
                    decision = "Cycle to East-West"
                    new_phase = 'east_west_green'
                else:
                    decision = "Cycle to North-South"
                    new_phase = 'north_south_green'
            else:
                decision = "Continue current phase"
                new_phase = self.current_phase
        
        return {
            'decision': decision,
            'new_phase': new_phase,
            'reason': f"Vehicles: {traffic_data['total_vehicles']}, Pedestrians: {traffic_data['total_pedestrians']}, Wait: {traffic_data['avg_pedestrian_wait']:.1f}s"
        }
    
    def execute_decision(self, decision_data, current_time):
        """Execute the AI decision."""
        new_phase = decision_data['new_phase']
        
        # Check if minimum phase time has elapsed
        if current_time - self.phase_start_time >= self.min_phase_time:
            if new_phase != self.current_phase:
                # Change phase
                self.current_phase = new_phase
                self.phase_start_time = current_time
                
                # Set traffic light state
                try:
                    traci.trafficlight.setRedYellowGreenState(self.traffic_light_id, self.phases[new_phase])
                    print(f"🚦 AI Decision: {decision_data['decision']}")
                    print(f"   Reason: {decision_data['reason']}")
                    print(f"   New State: {self.phases[new_phase]}")
                except:
                    print(f"⚠️ Could not control traffic light (simulation mode)")
        
        # Store decision
        self.decisions.append({
            'time': current_time,
            'decision': decision_data['decision'],
            'phase': self.current_phase,
            'reason': decision_data['reason']
        })

def run_ai_traffic_control_demo():
    """Run AI traffic control demonstration."""
    print("🤖 AI Traffic Light Control Demo")
    print("=" * 50)
    
    # SUMO configuration
    sumo_cmd = [
        "../venv/bin/sumo", "-c", "simple_grid_sim.sumocfg",
        "--start", "--quit-on-end",
        "--no-step-log", "--no-warnings"
    ]
    
    try:
        # Start TraCI connection
        traci.start(sumo_cmd)
        print("✅ Connected to SUMO simulation")
        
        # Initialize AI controller
        controller = AITrafficController()
        
        # Check if we have traffic lights
        traffic_lights = traci.trafficlight.getIDList()
        if traffic_lights:
            controller.traffic_light_id = traffic_lights[0]
            print(f"✅ Found traffic light: {controller.traffic_light_id}")
        else:
            print("⚠️ No traffic lights found - running in simulation mode")
        
        # Simulation loop
        step = 0
        max_steps = 1800  # 3 minutes at 0.1s steps
        
        print(f"🔄 Running AI traffic control for {max_steps} steps...")
        
        while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
            current_time = step / 10.0  # Convert to seconds
            
            # Analyze traffic
            traffic_data = controller.analyze_traffic()
            
            # Make AI decision
            decision_data = controller.make_ai_decision(traffic_data, current_time)
            
            # Execute decision
            controller.execute_decision(decision_data, current_time)
            
            # Advance simulation
            traci.simulationStep()
            step += 1
            
            # Progress update
            if step % 300 == 0:  # Every 30 seconds
                print(f"Step {step}/{max_steps} - "
                      f"Vehicles: {traffic_data['total_vehicles']}, "
                      f"Pedestrians: {traffic_data['total_pedestrians']}, "
                      f"Phase: {controller.current_phase}")
        
        # Close TraCI connection
        traci.close()
        print("✅ AI traffic control demo completed")
        
        # Print AI decision summary
        print(f"\n🧠 AI Decision Summary:")
        print(f"Total decisions made: {len(controller.decisions)}")
        
        # Show sample decisions
        print(f"\n📋 Sample AI Decisions:")
        for i, decision in enumerate(controller.decisions[-5:]):  # Last 5 decisions
            print(f"  {i+1}. {decision['decision']}")
            print(f"     Reason: {decision['reason']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        try:
            traci.close()
        except:
            pass
        return False

if __name__ == "__main__":
    # Change to simulation directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run AI traffic control demo
    success = run_ai_traffic_control_demo()
    
    if success:
        print("\n🎉 AI Traffic Control Demo Completed!")
        print("Your AI can now make intelligent traffic decisions!")
    else:
        print("\n❌ Demo failed. Check the errors above.")