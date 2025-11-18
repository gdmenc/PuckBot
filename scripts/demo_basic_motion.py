import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from drake_implementation import AirHockeyDrakeEnv
from simple_motion_controller import SimpleMotionController

def demo_basic_motion():
    """
    Demonstrates basic paddle motion to multiple target positions.
    """
    print("="*60)
    print("BASIC MOTION DEMO")
    print("="*60)

    print("\nInitializing Drake Air Hockey Environment...")
    env = AirHockeyDrakeEnv(time_step=0.001, use_meshcat=True)
    env.reset()

    print("\nInitializing motion controllers...")
    env.initialize_motion_controllers()

    controller1 = env.motion_controller_1

    table_height = env.table_height

    print(f"\nTable height: {table_height:.4f}m")
    print("Starting demonstration...")
    print("\nWatch the robot move in the Meshcat visualization!")

    # Define waypoints for demonstration
    waypoints = [
        ("Center of Table", np.array([-0.6, 0.0, table_height + 0.08])),
        ("Left Side", np.array([-0.5, 0.3, table_height + 0.08])),
        ("Right Side", np.array([-0.5, -0.3, table_height + 0.08])),
        ("Near Goal", np.array([-0.9, 0.0, table_height + 0.08])),
        ("Home Position", np.array([-0.5, 0.0, table_height + 0.15])),
    ]

    for i, (name, target_pos) in enumerate(waypoints):
        print(f"\n[{i+1}/{len(waypoints)}] Moving to: {name}")
        print(f"    Target Position: {target_pos}")

        current_pos = controller1.get_paddle_position(env.plant_context)
        print(f"    Current Position: {current_pos}")

        # Plan Trajectory
        success, trajectory = controller1.move_to_position(
            target_pos, env.plant_context, duration=2.0, dt=0.01
        )

        if not success:
            print(f"    Failed to reach {name} (IK FAILED)")
            continue

        print(f"    Trajectory planned ({len(trajectory)} steps)")

        for step, q in enumerate(trajectory):
            env.set_robot_joint_positions(env.robot1_model, q)
            env.diagram.ForcedPublish(env.context)

            if step % 10 == 0:
                time.sleep(0.01)

        # Verify final position
        final_pos = controller1.get_paddle_position(env.plant_context)
        error = np.linalg.norm(final_pos - target_pos)
        print(f"    Final Position: {final_pos}")
        print(f"    Position Error: {error*1000:.2f}mm")
        time.sleep(0.5)

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nKeeping visualization open for 10 seconds...")
    print("(You can close the browser tab or press Ctrl+C to exit)")

    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("\nExiting...")



def test_ik_reachability():
    """
    Test IK solver on multiple table positions to check reachability.
    """
    print("\n" + "="*60)
    print("IK REACHABILITY TEST")
    print("="*60)
    
    env = AirHockeyDrakeEnv(time_step=0.001, use_meshcat=False)
    env.reset()
    env.initialize_motion_controllers()
    
    controller1 = env.motion_controller_1
    table_height = env.table_height
    
    # Test grid of points on robot's half of table
    test_points = []
    for x in np.linspace(-0.9, -0.3, 5):
        for y in np.linspace(-0.4, 0.4, 5):
            test_points.append(np.array([x, y, table_height + 0.08]))
    
    print(f"\nTesting {len(test_points)} points on table...")
    
    success_count = 0
    for i, point in enumerate(test_points):
        success, q = controller1.solve_ik_position(point, env.plant_context)
        
        if success:
            success_count += 1
            # Verify solution
            env.set_robot_joint_positions(env.robot1_model, q)
            achieved = controller1.get_paddle_position(env.plant_context)
            error = np.linalg.norm(achieved - point)
            
            status = "✓" if error < 0.01 else "⚠"
            print(f"{status} Point {i+1}: {point} (error: {error*1000:.1f}mm)")
        else:
            print(f"✗ Point {i+1}: {point} (unreachable)")
    
    success_rate = (success_count / len(test_points)) * 100
    print(f"\n{'='*60}")
    print(f"Reachability: {success_count}/{len(test_points)} ({success_rate:.1f}%)")
    print(f"{'='*60}")



def test_forward_kinematics():
    """
    Test forward kinematics by setting joint angles and reading paddle position.
    """
    print("\n" + "="*60)
    print("FORWARD KINEMATICS TEST")
    print("="*60)
    
    env = AirHockeyDrakeEnv(time_step=0.001, use_meshcat=False)
    env.reset()
    env.initialize_motion_controllers()
    
    controller1 = env.motion_controller_1
    
    # Test configurations
    test_configs = [
        ("Home", np.array([-1.57, 0.5, 0.0, -1.2, 0.0, 1.6, 0.0])),
        ("All zeros", np.zeros(7)),
        ("Random 1", np.array([-1.0, 0.3, 0.1, -1.5, 0.2, 1.8, 0.1])),
    ]
    
    for name, q_test in test_configs:
        print(f"\n{name} configuration:")
        print(f"  Joint angles: {q_test}")
        
        env.set_robot_joint_positions(env.robot1_model, q_test)
        paddle_pos = controller1.get_paddle_position(env.plant_context)
        
        print(f"  Paddle position: {paddle_pos}")
        print(f"  ✓ FK computed successfully")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# SIMPLE MOTION CONTROLLER - DEMONSTRATION")
    print("#"*60)
    
    # Run tests
    test_forward_kinematics()
    test_ik_reachability()
    
    # Run main demo
    demo_basic_motion()