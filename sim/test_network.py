#!/usr/bin/env python3
"""
Test script to verify SUMO network loads correctly.
"""

import os
import sys
import traci

def test_network():
    """Test if the network loads correctly."""
    print("Testing SUMO network...")
    
    # SUMO configuration
    sumo_cmd = [
        "../venv/bin/sumo", "-c", "simple_sim.sumocfg",
        "--start", "--quit-on-end",
        "--no-step-log", "--no-warnings"
    ]
    
    try:
        # Start TraCI connection
        traci.start(sumo_cmd)
        print("✓ Connected to SUMO simulation")
        
        # Test basic functionality
        print(f"✓ Simulation time: {traci.simulation.getCurrentTime()}")
        print(f"✓ Vehicle count: {traci.vehicle.getIDCount()}")
        print(f"✓ Network loaded successfully")
        
        # Close TraCI connection
        traci.close()
        print("✓ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        try:
            traci.close()
        except:
            pass
        return False

if __name__ == "__main__":
    # Change to simulation directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    success = test_network()
    sys.exit(0 if success else 1)