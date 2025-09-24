#!/usr/bin/env python3
"""
Quick test of CV integration with SUMO - short simulation for testing.
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
from pedestrian_detector import PedestrianDetector

def test_cv_integration():
    """Quick test of CV integration."""
    print("🚦 Quick CV Integration Test")
    print("=" * 40)
    
    # Initialize YOLO detector
    print("📥 Initializing YOLO detector...")
    detector = PedestrianDetector()
    print("✅ YOLO detector ready")
    
    # SUMO configuration
    sumo_cmd = [
        "../venv/bin/sumo", "-c", "simple_grid_sim.sumocfg",
        "--start", "--quit-on-end",
        "--no-step-log", "--no-warnings"
    ]
    
    try:
        # Start TraCI connection
        print("🔌 Connecting to SUMO...")
        traci.start(sumo_cmd)
        print("✅ Connected to SUMO simulation")
        
        # Test data collection
        metrics_data = []
        
        # Short simulation loop (only 60 steps = 6 seconds)
        max_steps = 60
        print(f"🔄 Running quick test for {max_steps} steps...")
        
        for step in range(max_steps):
            # Get current simulation data
            vehicles = traci.vehicle.getIDList()
            pedestrians = traci.person.getIDList()
            
            # Create simulated camera frame
            frame = create_test_frame(vehicles, pedestrians, step)
            
            # Run YOLO detection (only every 10 steps to speed up)
            if step % 10 == 0:
                detections = detector.detect_objects(frame)
                cv_pedestrians = len(detections['pedestrians'])
                cv_vehicles = len(detections['vehicles'])
            else:
                cv_pedestrians = 0
                cv_vehicles = 0
            
            # Collect metrics
            metrics = {
                'step': step / 10.0,  # Convert to seconds
                'vehicles': len(vehicles),
                'pedestrians': len(pedestrians),
                'cv_pedestrians': cv_pedestrians,
                'cv_vehicles': cv_vehicles,
                'avg_vehicle_wait': sum(traci.vehicle.getWaitingTime(v) 
                                      for v in vehicles) / max(len(vehicles), 1)
            }
            metrics_data.append(metrics)
            
            # Advance simulation
            traci.simulationStep()
            
            # Progress update
            if step % 20 == 0:
                print(f"  Step {step}/{max_steps} - "
                      f"SUMO: {len(vehicles)}V/{len(pedestrians)}P, "
                      f"CV: {cv_vehicles}V/{cv_pedestrians}P")
        
        # Close TraCI connection
        traci.close()
        print("✅ Simulation completed")
        
        # Save results
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"../results/cv_test_results_{timestamp}.csv"
            df.to_csv(results_file, index=False)
            print(f"💾 Results saved to {results_file}")
            
            # Print summary
            print("\n📊 TEST SUMMARY:")
            print(f"Total steps: {len(df)}")
            print(f"Average SUMO vehicles: {df['vehicles'].mean():.1f}")
            print(f"Average CV vehicles: {df['cv_vehicles'].mean():.1f}")
            print(f"Average SUMO pedestrians: {df['pedestrians'].mean():.1f}")
            print(f"Average CV pedestrians: {df['cv_pedestrians'].mean():.1f}")
            print(f"Average vehicle wait: {df['avg_vehicle_wait'].mean():.2f}s")
        
        print("\n🎉 CV Integration Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        try:
            traci.close()
        except:
            pass
        return False

def create_test_frame(vehicles, pedestrians, step):
    """Create a simple test frame."""
    # Create a simple frame
    frame = np.ones((200, 300, 3), dtype=np.uint8) * 128
    
    # Draw intersection
    cv2.rectangle(frame, (100, 80), (200, 120), (64, 64, 64), -1)
    
    # Draw roads
    cv2.rectangle(frame, (0, 95), (300, 105), (96, 96, 96), -1)
    cv2.rectangle(frame, (145, 0), (155, 200), (96, 96, 96), -1)
    
    # Draw vehicles as small rectangles
    for i, vehicle in enumerate(vehicles[:3]):
        x = 120 + i * 20
        y = 100
        cv2.rectangle(frame, (x-5, y-3), (x+5, y+3), (0, 0, 255), -1)
    
    # Draw pedestrians as small circles
    for i, pedestrian in enumerate(pedestrians[:2]):
        x = 130 + i * 15
        y = 100
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    
    # Add step counter
    cv2.putText(frame, f"Step: {step}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

if __name__ == "__main__":
    # Change to simulation directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create results directory if it doesn't exist
    os.makedirs("../results", exist_ok=True)
    
    # Run test
    success = test_cv_integration()
    sys.exit(0 if success else 1)