#!/usr/bin/env python3
"""
Create a manual traffic light network that definitely works.
"""

import os

def create_manual_traffic_light_network():
    """Create a manual network with guaranteed traffic lights."""
    print("🚦 Creating Manual Traffic Light Network")
    print("=" * 50)
    
    # Create a simple but working traffic light network
    network_content = '''<?xml version="1.0" encoding="UTF-8"?>
<net version="1.9" junctionCornerDetail="5" limitTurnSpeed="5.5" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,300.00,300.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    
    <!-- External nodes -->
    <junction id="n0" type="priority" x="0.00" y="150.00" incLanes="" intLanes="" shape="0.00,145.00 0.00,155.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    <junction id="n2" type="priority" x="300.00" y="150.00" incLanes="" intLanes="" shape="300.00,145.00 300.00,155.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    <junction id="n3" type="priority" x="150.00" y="0.00" incLanes="" intLanes="" shape="145.00,0.00 155.00,0.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    <junction id="n4" type="priority" x="150.00" y="300.00" incLanes="" intLanes="" shape="145.00,300.00 155.00,300.00">
        <request index="0" response="00" foes="00" cont="0"/>
    </junction>
    
    <!-- Central traffic light junction -->
    <junction id="J0" type="traffic_light" x="150.00" y="150.00" incLanes="e0_0 e2_0 e4_0 e6_0" intLanes=":J0_0_0 :J0_1_0 :J0_2_0 :J0_3_0" shape="150.00,145.00 155.00,150.00 150.00,155.00 145.00,150.00">
        <request index="0" response="00" foes="00" cont="0"/>
        <request index="1" response="00" foes="00" cont="0"/>
        <request index="2" response="00" foes="00" cont="0"/>
        <request index="3" response="00" foes="00" cont="0"/>
    </junction>

    <!-- Edges -->
    <edge id="e0" from="n0" to="J0" priority="1">
        <lane id="e0_0" index="0" speed="13.89" length="150.00" shape="0.00,150.00 150.00,150.00"/>
    </edge>
    <edge id="e1" from="J0" to="n2" priority="1">
        <lane id="e1_0" index="0" speed="13.89" length="150.00" shape="150.00,150.00 300.00,150.00"/>
    </edge>
    <edge id="e2" from="n2" to="J0" priority="1">
        <lane id="e2_0" index="0" speed="13.89" length="150.00" shape="300.00,150.00 150.00,150.00"/>
    </edge>
    <edge id="e3" from="J0" to="n0" priority="1">
        <lane id="e3_0" index="0" speed="13.89" length="150.00" shape="150.00,150.00 0.00,150.00"/>
    </edge>
    <edge id="e4" from="n3" to="J0" priority="1">
        <lane id="e4_0" index="0" speed="13.89" length="150.00" shape="150.00,0.00 150.00,150.00"/>
    </edge>
    <edge id="e5" from="J0" to="n4" priority="1">
        <lane id="e5_0" index="0" speed="13.89" length="150.00" shape="150.00,150.00 150.00,300.00"/>
    </edge>
    <edge id="e6" from="n4" to="J0" priority="1">
        <lane id="e6_0" index="0" speed="13.89" length="150.00" shape="150.00,300.00 150.00,150.00"/>
    </edge>
    <edge id="e7" from="J0" to="n3" priority="1">
        <lane id="e7_0" index="0" speed="13.89" length="150.00" shape="150.00,150.00 150.00,0.00"/>
    </edge>

    <!-- Internal lanes -->
    <edge id=":J0_0_0" from="J0" to="J0" priority="1">
        <lane id=":J0_0_0" index="0" speed="13.89" length="10.00" shape="145.00,150.00 150.00,150.00"/>
    </edge>
    <edge id=":J0_1_0" from="J0" to="J0" priority="1">
        <lane id=":J0_1_0" index="0" speed="13.89" length="10.00" shape="150.00,150.00 155.00,150.00"/>
    </edge>
    <edge id=":J0_2_0" from="J0" to="J0" priority="1">
        <lane id=":J0_2_0" index="0" speed="13.89" length="10.00" shape="150.00,145.00 150.00,150.00"/>
    </edge>
    <edge id=":J0_3_0" from="J0" to="J0" priority="1">
        <lane id=":J0_3_0" index="0" speed="13.89" length="10.00" shape="150.00,150.00 150.00,155.00"/>
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
    <vType id="pedestrian" vClass="pedestrian" speed="1.34" accel="2.0" decel="2.0"/>

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

    <!-- Pedestrian flows -->
    <flow id="ped_flow_ns" type="pedestrian" route="route_ns" begin="0" end="3600" period="8"/>
    <flow id="ped_flow_sn" type="pedestrian" route="route_sn" begin="0" end="3600" period="10"/>
    <flow id="ped_flow_ew" type="pedestrian" route="route_ew" begin="0" end="3600" period="12"/>
    <flow id="ped_flow_we" type="pedestrian" route="route_we" begin="0" end="3600" period="15"/>

</routes>'''
    
    # Create configuration
    config_content = '''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="manual_traffic_light.net.xml"/>
        <route-files value="manual_traffic_light_routes.rou.xml"/>
        <additional-files value="manual_traffic_light_tls.add.xml"/>
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
    with open('manual_traffic_light.net.xml', 'w') as f:
        f.write(network_content)
    
    with open('manual_traffic_light_tls.add.xml', 'w') as f:
        f.write(tls_content)
    
    with open('manual_traffic_light_routes.rou.xml', 'w') as f:
        f.write(routes_content)
    
    with open('manual_traffic_light_sim.sumocfg', 'w') as f:
        f.write(config_content)
    
    print("✅ Manual traffic light network created!")
    print("\nFiles created:")
    print("  - manual_traffic_light.net.xml (network)")
    print("  - manual_traffic_light_tls.add.xml (traffic light program)")
    print("  - manual_traffic_light_routes.rou.xml (routes)")
    print("  - manual_traffic_light_sim.sumocfg (configuration)")
    
    return True

def test_traffic_lights():
    """Test the traffic light network."""
    print("\n🧪 Testing Traffic Light Network...")
    
    try:
        import traci
        
        # SUMO configuration
        sumo_cmd = [
            '../venv/bin/sumo', '-c', 'manual_traffic_light_sim.sumocfg',
            '--start', '--quit-on-end',
            '--no-step-log', '--no-warnings'
        ]
        
        traci.start(sumo_cmd)
        
        # Check traffic lights
        traffic_lights = traci.trafficlight.getIDList()
        print(f"✅ Found {len(traffic_lights)} traffic lights: {traffic_lights}")
        
        if traffic_lights:
            # Test traffic light control
            tl_id = traffic_lights[0]
            current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            print(f"✅ Traffic light {tl_id} current state: {current_state}")
            
            # Test changing state
            traci.trafficlight.setRedYellowGreenState(tl_id, "GGrrrrGGrrrr")
            new_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            print(f"✅ Changed state to: {new_state}")
        
        traci.close()
        print("✅ Traffic light test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Traffic light test failed: {e}")
        return False

if __name__ == "__main__":
    # Change to simulation directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create manual traffic light network
    success = create_manual_traffic_light_network()
    
    if success:
        # Test traffic lights
        test_success = test_traffic_lights()
        
        if test_success:
            print("\n🎉 Manual Traffic Light Network Ready!")
            print("You now have a working traffic light that you can control!")
        else:
            print("\n⚠️ Network created but traffic light test failed.")
    else:
        print("\n❌ Network creation failed.")