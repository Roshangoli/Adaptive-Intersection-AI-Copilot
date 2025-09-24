#!/usr/bin/env python3
"""
Create a working traffic light system that can be controlled by AI.
"""

import os
import subprocess
import traci
import time

def create_simple_traffic_light_network():
    """Create a simple but working traffic light network."""
    print("🚦 Creating Working Traffic Light Network")
    print("=" * 50)
    
    # Create a very simple network that definitely works
    network_content = '''<?xml version="1.0" encoding="UTF-8"?>
<net version="1.9" junctionCornerDetail="5" limitTurnSpeed="5.5" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    
    <!-- External nodes -->
    <junction id="n0" type="priority" x="0.00" y="100.00" incLanes="" intLanes="" shape="0.00,95.00 0.00,105.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    <junction id="n2" type="priority" x="200.00" y="100.00" incLanes="" intLanes="" shape="200.00,95.00 200.00,105.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    <junction id="n3" type="priority" x="100.00" y="0.00" incLanes="" intLanes="" shape="95.00,0.00 105.00,0.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    <junction id="n4" type="priority" x="100.00" y="200.00" incLanes="" intLanes="" shape="95.00,200.00 105.00,200.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    
    <!-- Central traffic light junction -->
    <junction id="J0" type="traffic_light" x="100.00" y="100.00" incLanes="e0_0 e2_0 e4_0 e6_0" intLanes=":J0_0_0 :J0_1_0 :J0_2_0 :J0_3_0" shape="100.00,95.00 105.00,100.00 100.00,105.00 95.00,100.00">
        <request index="0" response="00" foes="00" cont="0"/>
        <request index="1" response="00" foes="00" cont="0"/>
        <request index="2" response="00" foes="00" cont="0"/>
        <request index="3" response="00" foes="00" cont="0"/>
    </junction>

    <!-- Edges -->
    <edge id="e0" from="n0" to="J0" priority="1">
        <lane id="e0_0" index="0" speed="13.89" length="100.00" shape="0.00,100.00 100.00,100.00"/>
    </edge>
    <edge id="e1" from="J0" to="n2" priority="1">
        <lane id="e1_0" index="0" speed="13.89" length="100.00" shape="100.00,100.00 200.00,100.00"/>
    </edge>
    <edge id="e2" from="n2" to="J0" priority="1">
        <lane id="e2_0" index="0" speed="13.89" length="100.00" shape="200.00,100.00 100.00,100.00"/>
    </edge>
    <edge id="e3" from="J0" to="n0" priority="1">
        <lane id="e3_0" index="0" speed="13.89" length="100.00" shape="100.00,100.00 0.00,100.00"/>
    </edge>
    <edge id="e4" from="n3" to="J0" priority="1">
        <lane id="e4_0" index="0" speed="13.89" length="100.00" shape="100.00,0.00 100.00,100.00"/>
    </edge>
    <edge id="e5" from="J0" to="n4" priority="1">
        <lane id="e5_0" index="0" speed="13.89" length="100.00" shape="100.00,100.00 100.00,200.00"/>
    </edge>
    <edge id="e6" from="n4" to="J0" priority="1">
        <lane id="e6_0" index="0" speed="13.89" length="100.00" shape="100.00,200.00 100.00,100.00"/>
    </edge>
    <edge id="e7" from="J0" to="n3" priority="1">
        <lane id="e7_0" index="0" speed="13.89" length="100.00" shape="100.00,100.00 100.00,0.00"/>
    </edge>

    <!-- Internal lanes -->
    <edge id=":J0_0_0" from="J0" to="J0" priority="1">
        <lane id=":J0_0_0" index="0" speed="13.89" length="10.00" shape="95.00,100.00 100.00,100.00"/>
    </edge>
    <edge id=":J0_1_0" from="J0" to="J0" priority="1">
        <lane id=":J0_1_0" index="0" speed="13.89" length="10.00" shape="100.00,100.00 105.00,100.00"/>
    </edge>
    <edge id=":J0_2_0" from="J0" to="J0" priority="1">
        <lane id=":J0_2_0" index="0" speed="13.89" length="10.00" shape="100.00,95.00 100.00,100.00"/>
    </edge>
    <edge id=":J0_3_0" from="J0" to="J0" priority="1">
        <lane id=":J0_3_0" index="0" speed="13.89" length="10.00" shape="100.00,100.00 100.00,105.00"/>
    </edge>

</net>'''
    
    # Create traffic light program
    tls_content = '''<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    <tlLogic id="J0" type="static" programID="0" offset="0">
        <phase duration="30" state="GGrrrrGGrrrr"/>
        <phase duration="3" state="yyyrrryyyrrr"/>
        <phase duration="30" state="rrrGGGrrrGGG"/>
        <phase duration="3" state="rrryyyrrryyy"/>
    </tlLogic>
</additional>'''
    
    # Create routes
    routes_content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

    <!-- Vehicle types -->
    <vType id="car" accel="2.5" decel="4.5" sigma="0.5" length="4.3" maxSpeed="50"/>

    <!-- Routes -->
    <route id="route_ns" edges="e0 e1"/>
    <route id="route_sn" edges="e2 e3"/>
    <route id="route_ew" edges="e4 e5"/>
    <route id="route_we" edges="e6 e7"/>

    <!-- Vehicle flows -->
    <flow id="flow_ns" type="car" route="route_ns" begin="0" end="3600" period="3"/>
    <flow id="flow_sn" type="car" route="route_sn" begin="0" end="3600" period="4"/>
    <flow id="flow_ew" type="car" route="route_ew" begin="0" end="3600" period="3"/>
    <flow id="flow_we" type="car" route="route_we" begin="0" end="3600" period="5"/>

</routes>'''
    
    # Create configuration
    config_content = '''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="working_traffic_light.net.xml"/>
        <route-files value="working_traffic_light_routes.rou.xml"/>
        <additional-files value="working_traffic_light_tls.add.xml"/>
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
    
    # Write files
    with open('working_traffic_light.net.xml', 'w') as f:
        f.write(network_content)
    
    with open('working_traffic_light_tls.add.xml', 'w') as f:
        f.write(tls_content)
    
    with open('working_traffic_light_routes.rou.xml', 'w') as f:
        f.write(routes_content)
    
    with open('working_traffic_light_sim.sumocfg', 'w') as f:
        f.write(config_content)
    
    print("✅ Working traffic light network created!")
    print("\nFiles created:")
    print("  - working_traffic_light.net.xml (network)")
    print("  - working_traffic_light_tls.add.xml (traffic light program)")
    print("  - working_traffic_light_routes.rou.xml (routes)")
    print("  - working_traffic_light_sim.sumocfg (configuration)")
    
    return True

def test_traffic_light_control():
    """Test that we can control the traffic light."""
    print("\n🧪 Testing Traffic Light Control...")
    
    try:
        # SUMO configuration
        sumo_cmd = [
            '../venv/bin/sumo', '-c', 'working_traffic_light_sim.sumocfg',
            '--start', '--quit-on-end',
            '--no-step-log', '--no-warnings'
        ]
        
        traci.start(sumo_cmd)
        
        # Check traffic lights
        traffic_lights = traci.trafficlight.getIDList()
        print(f"✅ Found {len(traffic_lights)} traffic lights: {traffic_lights}")
        
        if traffic_lights:
            tl_id = traffic_lights[0]
            
            # Test current state
            current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            print(f"✅ Traffic light {tl_id} current state: {current_state}")
            
            # Test changing state
            print("🔄 Testing traffic light control...")
            
            # Change to North-South green
            traci.trafficlight.setRedYellowGreenState(tl_id, "GGrrrrGGrrrr")
            new_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            print(f"✅ Changed to North-South green: {new_state}")
            
            # Advance simulation
            traci.simulationStep()
            
            # Change to East-West green
            traci.trafficlight.setRedYellowGreenState(tl_id, "rrrGGGrrrGGG")
            new_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            print(f"✅ Changed to East-West green: {new_state}")
            
            # Advance simulation
            traci.simulationStep()
            
            # Change to all red
            traci.trafficlight.setRedYellowGreenState(tl_id, "rrrrrrrrrrrr")
            new_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            print(f"✅ Changed to all red: {new_state}")
            
            print("✅ Traffic light control test successful!")
            return True
        else:
            print("❌ No traffic lights found")
            return False
        
        traci.close()
        
    except Exception as e:
        print(f"❌ Traffic light test failed: {e}")
        return False

if __name__ == "__main__":
    # Change to simulation directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create working traffic light network
    success = create_simple_traffic_light_network()
    
    if success:
        # Test traffic light control
        test_success = test_traffic_light_control()
        
        if test_success:
            print("\n🎉 Working Traffic Light System Ready!")
            print("You can now control traffic lights with your AI!")
        else:
            print("\n⚠️ Network created but traffic light test failed.")
    else:
        print("\n❌ Network creation failed.")