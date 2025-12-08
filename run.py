"""
Main entry point for Air Hockey Robot Game
"""
from argparse import ArgumentParser
import time
import numpy as np
import sys

from scripts.env.configure import create_air_hockey_simulation, initialize_puck, initialize_paddles, initialize_robots

def get_args():
    parser = ArgumentParser(description="Air Hockey Robot Game")
    
    parser.add_argument(
        "--time_step",
        type=float,
        default=0.001,
        help="Simulation time step (seconds)"
    )

    parser.add_argument(
        "--game_duration",
        type=float,
        default=30.0,
        help="Game duration in seconds"
    )
    
    parser.add_argument(
        "--num_arms",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of robot arms in the simulation (1 or 2)"
    )

    parser.add_argument(
        "--no_meshcat",
        action="store_true",
        help="Disable Meshcat visualization"
    )
    parser.add_argument(
        "--skip-grasp",
        action="store_true",
        help="Skip grasping sequence and weld paddles directly to grippers"
    )

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    print("="*70)
    print("AIR HOCKEY ROBOT GAME")
    print("="*70)
    print(f"Configuration:")
    print(f"  Num Arms: {args.num_arms}")
    print(f"  Time step: {args.time_step}s")
    print(f"  Game duration: {args.game_duration}s")
    
    # Create game environment
    print("\n" + "="*70)
    print("Initializing Environment...")
    print("="*70)
    
    # Updated to unpack 4 values
    simulator, meshcat, plant, diagram = create_air_hockey_simulation(
        num_arms=args.num_arms,
        time_step=args.time_step,
        use_meshcat=not args.no_meshcat,
        skip_grasp=args.skip_grasp
    )
    
    # Positions are now initialized inside create_air_hockey_simulation()
    
    if meshcat:
        print(f"\nMeshcat visualization: {meshcat.web_url()}")
        print("Open this URL in your browser to view the simulation")
        time.sleep(1)  # Give user time to open browser
    
    # Grasping sequence (skip if --skip-grasp flag is set)
    if not args.skip_grasp:
        print("\n" + "="*70)
        print("Executing Grasp Sequence...")
        print("="*70)
    
        try:
            from scripts.env.sim_adapter import SimAdapter
            from scripts.grasp.paddle_grasper import PaddleGrasper
            
            adapter = SimAdapter(simulator, plant, diagram, meshcat=meshcat)
            
            robots_to_grasp = []
            if adapter.robot2_model:  # Right
                robots_to_grasp.append((adapter.robot2_model, "body", 2))
        
            if args.num_arms == 2 and adapter.robot1_model:  # Left
                robots_to_grasp.append((adapter.robot1_model, "body", 1))
                
            for model, frame_name, robot_id in robots_to_grasp:
                print(f"\n[GRASP] Starting grasp sequence for Robot {robot_id}...")
                adapter.set_active_paddle_for_robot(robot_id)
                
                gripper_name = "left_wsg" if robot_id == 1 else "right_wsg"
                try:
                    gripper_model = plant.GetModelInstanceByName(gripper_name)
                except:
                    gripper_model = None
        
                grasper = PaddleGrasper(
                    adapter,
                    model, 
                    tool_frame_name=frame_name,
                    tool_model_instance=gripper_model
                )
                
                success = grasper.execute_grasp_sequence()
                if not success:
                    print(f"[ERROR] Grasp failed for Robot {robot_id}!")
                    
        except ImportError as e:
            print(f"[WARNING] Skipping grasping sequence due to missing modules: {e}")
        except Exception as e:
            print(f"[ERROR] Grasping sequence encountered an error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[INFO] Grasping skipped (paddles welded to grippers)")

    # Run gameplay
    print("\n" + "="*70)
    print("Starting Simulation...")
    print("="*70)
    
    current_time = simulator.get_context().get_time()
    desired_end_time = current_time + args.game_duration
    print(f"[INFO] Running simulation from {current_time:.2f}s to {desired_end_time:.2f}s")
    
    try:
        simulator.AdvanceTo(desired_end_time)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    print("\n" + "="*70)
    print("Simulation Complete")
    print("="*70)
    
    if meshcat:
        print("\nMeshcat visualization will remain open.")
        print("Press Ctrl+C to exit.")
        try:
            # Keep alive for viewing
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main()
