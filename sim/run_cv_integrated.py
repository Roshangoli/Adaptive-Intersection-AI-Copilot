#!/usr/bin/env python3
"""
Integrated SUMO simulation with Computer Vision for Adaptive Intersection AI Copilot.
This script runs SUMO simulation and uses YOLO to count pedestrians/vehicles in real-time.
"""

import os
import sys
import time
import traci
import numpy as np
import cv2
from datetime import datetime
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'cv'))
from pedestrian_detector import PedestrianDetector, IntersectionAnalyzer

class CVIntegratedController:
    """Controller that integrates computer vision with SUMO simulation."""
    
    def __init__(self):
        """Initialize the integrated controller."""
        self.detector = PedestrianDetector()
        self.analyzer = IntersectionAnalyzer(self.detector)
        
        # Define intersection zones (example coordinates)
        self.waiting_zones = {
            'north_waiting': [(100, 50), (200, 50), (200, 100), (100, 100)],
            'south_waiting': [(100, 150), (200, 150), (200, 200), (100, 200)],
            'east_waiting': [(50, 100), (100, 100), (100, 150), (50, 150)],
            'west_waiting': [(150, 100), (200, 100), (200, 150), (150, 150)]
        }
        
        self.crossing_zones = {
            'north_south': [(100, 100), (200, 100), (200, 150), (100, 150)],
            'east_west': [(100, 100), (150, 100), (150, 200), (100, 200)]
        }
        
        # Simulation state
        self.current_phase = 0
        self.phase_start_time = 0
        self.phases = [
            "GGrrrrGGrrrr",  # North-South green
            "yyyrrryyyrrr",  # North-South yellow
            "rrrGGGrrrGGG",  # East-West green
            "rrryyyrrryyy"   # East-West yellow
        ]
        
        # Metrics collection
        self.metrics_data = []
        
    def simulate_camera_feed(self, step):
        """Simulate camera feed from SUMO simulation."""
        # In a real implementation, this would capture video from cameras
        # For now, we'll simulate based on SUMO vehicle/pedestrian data
        
        # Get current simulation data
        vehicles = traci.vehicle.getIDList()
        pedestrians = traci.person.getIDList()
        
        # Create a simulated camera frame
        frame = self.create_simulated_frame(vehicles, pedestrians, step)
        
        return frame
    
    def create_simulated_frame(self, vehicles, pedestrians, step):
        """Create a simulated camera frame based on SUMO data."""
        # Create a simple frame representing the intersection
        frame = np.ones((300, 400, 3), dtype=np.uint8) * 128  # Gray background
        
        # Draw intersection
        cv2.rectangle(frame, (150, 100), (250, 200), (64, 64, 64), -1)  # Intersection area
        
        # Draw roads
        cv2.rectangle(frame, (0, 140), (400, 160), (96, 96, 96), -1)  # Horizontal road
        cv2.rectangle(frame, (190, 0), (210, 300), (96, 96, 96), -1)  # Vertical road
        
        # Simulate vehicles as rectangles
        for i, vehicle in enumerate(vehicles[:5]):  # Limit to 5 for visualization
            try:
                pos = traci.vehicle.getPosition(vehicle)
                x = int(pos[0] / 2)  # Scale down
                y = int(pos[1] / 2)
                
                # Draw vehicle
                cv2.rectangle(frame, (x-10, y-5), (x+10, y+5), (0, 0, 255), -1)
                cv2.putText(frame, f"V{i}", (x-5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            except:
                pass
        
        # Simulate pedestrians as circles
        for i, pedestrian in enumerate(pedestrians[:3]):  # Limit to 3 for visualization
            try:
                pos = traci.person.getPosition(pedestrian)
                x = int(pos[0] / 2)  # Scale down
                y = int(pos[1] / 2)
                
                # Draw pedestrian
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(frame, f"P{i}", (x-5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            except:
                pass
        
        # Add traffic light indicator
        light_color = (0, 255, 0) if self.current_phase in [0, 2] else (0, 255, 255)
        cv2.circle(frame, (200, 150), 15, light_color, -1)
        
        # Add step counter
        cv2.putText(frame, f"Step: {step}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def analyze_traffic_with_cv(self, frame):
        """Analyze traffic using computer vision."""
        try:
            # Run YOLO detection
            detections = self.detector.detect_objects(frame)
            
            # Count in zones
            waiting_counts = self.detector.get_waiting_counts(detections, self.waiting_zones)
            crossing_counts = self.detector.get_crossing_counts(detections, self.crossing_zones)
            
            # Apply privacy protection
            protected_frame = self.detector.apply_privacy_protection(frame, detections)
            
            return {
                'detections': detections,
                'waiting_counts': waiting_counts,
                'crossing_counts': crossing_counts,
                'protected_frame': protected_frame,
                'total_pedestrians': len(detections['pedestrians']),
                'total_vehicles': len(detections['vehicles'])
            }
            
        except Exception as e:
            print(f"CV analysis error: {e}")
            return {
                'detections': {'pedestrians': [], 'vehicles': [], 'all_detections': []},
                'waiting_counts': {},
                'crossing_counts': {},
                'protected_frame': frame,
                'total_pedestrians': 0,
                'total_vehicles': 0
            }
    
    def make_traffic_decision(self, cv_data, step):
        """Make traffic control decision based on CV data."""
        # Simple decision logic based on CV counts
        total_pedestrians = cv_data['total_pedestrians']
        total_vehicles = cv_data['total_vehicles']
        
        # Check if minimum phase time has elapsed
        if step - self.phase_start_time >= 50:  # 5 seconds at 0.1s steps
            # Simple logic: prioritize pedestrians if many are waiting
            if total_pedestrians > 3:
                self.current_phase = 0  # North-South green for pedestrians
                self.phase_start_time = step
            elif total_vehicles > 5:
                self.current_phase = 2  # East-West green for vehicles
                self.phase_start_time = step
            else:
                # Cycle through phases
                self.current_phase = (self.current_phase + 1) % len(self.phases)
                self.phase_start_time = step
        
        # Set traffic light state (if we had a real traffic light)
        # traci.trafficlight.setRedYellowGreenState("J0", self.phases[self.current_phase])
        
        return {
            'phase': self.current_phase,
            'phase_duration': step - self.phase_start_time,
            'decision_reason': f"Pedestrians: {total_pedestrians}, Vehicles: {total_vehicles}"
        }
    
    def collect_metrics(self, step, cv_data, decision_data):
        """Collect simulation metrics."""
        metrics = {
            'step': step / 10.0,  # Convert to seconds
            'vehicles': traci.vehicle.getIDCount(),
            'pedestrians': traci.person.getIDCount(),
            'cv_pedestrians': cv_data['total_pedestrians'],
            'cv_vehicles': cv_data['total_vehicles'],
            'waiting_vehicles': len([v for v in traci.vehicle.getIDList() 
                                   if traci.vehicle.getWaitingTime(v) > 0]),
            'avg_vehicle_wait': sum(traci.vehicle.getWaitingTime(v) 
                                  for v in traci.vehicle.getIDList()) / max(traci.vehicle.getIDCount(), 1),
            'current_phase': decision_data['phase'],
            'phase_duration': decision_data['phase_duration'],
            'decision_reason': decision_data['decision_reason']
        }
        
        self.metrics_data.append(metrics)
        return metrics

def run_cv_integrated_simulation():
    """Run the integrated CV-SUMO simulation."""
    print("🚦 Starting CV-Integrated Traffic Simulation...")
    
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
        
        # Initialize controller
        controller = CVIntegratedController()
        
        # Simulation loop
        step = 0
        max_steps = 3600  # 6 minutes at 0.1s steps
        
        print(f"🔄 Running simulation for {max_steps} steps...")
        
        while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
            # Simulate camera feed
            frame = controller.simulate_camera_feed(step)
            
            # Analyze with computer vision
            cv_data = controller.analyze_traffic_with_cv(frame)
            
            # Make traffic control decision
            decision_data = controller.make_traffic_decision(cv_data, step)
            
            # Collect metrics
            metrics = controller.collect_metrics(step, cv_data, decision_data)
            
            # Advance simulation
            traci.simulationStep()
            step += 1
            
            # Progress update
            if step % 600 == 0:  # Every minute
                print(f"Step {step}/{max_steps} - "
                      f"SUMO: {metrics['vehicles']}V/{metrics['pedestrians']}P, "
                      f"CV: {metrics['cv_vehicles']}V/{metrics['cv_pedestrians']}P, "
                      f"Phase: {metrics['current_phase']}")
        
        # Close TraCI connection
        traci.close()
        print("✅ Simulation completed successfully")
        
        # Save results
        if controller.metrics_data:
            df = pd.DataFrame(controller.metrics_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"../results/cv_integrated_results_{timestamp}.csv"
            df.to_csv(results_file, index=False)
            print(f"💾 Results saved to {results_file}")
            
            # Print summary statistics
            print("\n📊 SIMULATION SUMMARY:")
            print(f"Total simulation time: {df['step'].max():.1f} seconds")
            print(f"Average SUMO vehicles: {df['vehicles'].mean():.1f}")
            print(f"Average CV vehicles: {df['cv_vehicles'].mean():.1f}")
            print(f"Average SUMO pedestrians: {df['pedestrians'].mean():.1f}")
            print(f"Average CV pedestrians: {df['cv_pedestrians'].mean():.1f}")
            print(f"Average vehicle wait time: {df['avg_vehicle_wait'].mean():.2f}s")
            print(f"Total phase changes: {len(df[df['phase_duration'] < 5])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
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
    success = run_cv_integrated_simulation()
    sys.exit(0 if success else 1)