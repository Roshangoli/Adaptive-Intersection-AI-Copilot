#!/usr/bin/env python3
"""
Create a realistic intersection network with working traffic lights.
"""

import subprocess
import os

def create_realistic_intersection():
    """Create a realistic intersection with traffic lights."""
    print("🚦 Creating Realistic Intersection with Traffic Lights")
    print("=" * 60)
    
    # Create a more realistic intersection using SUMO's tools
    try:
        # Create intersection with traffic lights
        result = subprocess.run([
            '../venv/bin/netgenerate',
            '--grid',
            '--grid.x-number', '3',
            '--grid.y-number', '3', 
            '--grid.length', '200',
            '--tls.guess', 'true',
            '--output-file', 'realistic_intersection.net.xml'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Realistic intersection created successfully!")
            print("Output:", result.stdout)
        else:
            print("❌ Failed to create intersection:")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Generate routes for the intersection
    try:
        print("\n🛣️ Generating realistic routes...")
        result = subprocess.run([
            '../venv/bin/python',
            '../venv/lib/python3.9/site-packages/sumo/tools/randomTrips.py',
            '-n', 'realistic_intersection.net.xml',
            '-r', 'realistic_routes.rou.xml',
            '--period', '1.5',  # More frequent vehicles
            '-e', '3600',
            '--seed', '42'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Routes generated successfully!")
        else:
            print("❌ Route generation failed:")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Create configuration file
    config_content = '''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="realistic_intersection.net.xml"/>
        <route-files value="realistic_routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="0.1"/>
    </time>
    <processing>
        <ignore-junction-blocker value="0"/>
        <collision.action value="warn"/>
    </processing>
    <routing>
        <device.rerouting.adaptation-interval value="10"/>
    </routing>
    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>'''
    
    with open('realistic_sim.sumocfg', 'w') as f:
        f.write(config_content)
    
    print("✅ Configuration file created")
    
    # Test the network
    print("\n🧪 Testing the realistic network...")
    try:
        result = subprocess.run([
            '../venv/bin/sumo',
            '-c', 'realistic_sim.sumocfg',
            '--start', '--quit-on-end',
            '--no-step-log', '--no-warnings'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Network test successful!")
            print("Simulation completed without errors")
        else:
            print("❌ Network test failed:")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    
    print("\n🎉 Realistic intersection setup completed!")
    print("\nFiles created:")
    print("  - realistic_intersection.net.xml (network)")
    print("  - realistic_routes.rou.xml (routes)")
    print("  - realistic_sim.sumocfg (configuration)")
    
    return True

def check_traffic_lights():
    """Check what traffic lights are available."""
    print("\n🔍 Checking available traffic lights...")
    
    try:
        import traci
        
        # SUMO configuration
        sumo_cmd = [
            '../venv/bin/sumo', '-c', 'realistic_sim.sumocfg',
            '--start', '--quit-on-end',
            '--no-step-log', '--no-warnings'
        ]
        
        traci.start(sumo_cmd)
        traffic_lights = traci.trafficlight.getIDList()
        traci.close()
        
        print(f"✅ Found {len(traffic_lights)} traffic lights:")
        for tl in traffic_lights:
            print(f"  - {tl}")
        
        return traffic_lights
        
    except Exception as e:
        print(f"❌ Error checking traffic lights: {e}")
        return []

if __name__ == "__main__":
    # Change to simulation directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create realistic intersection
    success = create_realistic_intersection()
    
    if success:
        # Check traffic lights
        traffic_lights = check_traffic_lights()
        
        if traffic_lights:
            print(f"\n🚀 Ready to use! You have {len(traffic_lights)} traffic lights to control.")
        else:
            print("\n⚠️ No traffic lights found, but network is working.")
    else:
        print("\n❌ Setup failed. Check the errors above.")