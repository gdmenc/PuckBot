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
        "--mode",
        type=str,
        default="tournament",
        choices=["defend", "hit", "tournament"],
        help="Game mode: defend (block approaching puck), hit (strike stationary puck), or tournament (full game)"
    )

    parser.add_argument(
        "--no_meshcat",
        action="store_true",
        help="Disable Meshcat visualization"
    )
    parser.add_argument("--skip-grasp", action="store_true", default=True,
                        help="Skip grasping (paddles welded to grippers)")
    parser.add_argument("--max_puck_velocity", type=float, default=0.3,
                        help="Maximum puck velocity in m/s (default: 0.3)")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    print("="*70)
    print("AIR HOCKEY ROBOT GAME")
    print("="*70)
    print(f"Configuration:")
    print(f"  Num Arms: {args.num_arms}")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Time step: {args.time_step}s")
    print(f"  Game duration: {args.game_duration}s")
    
    # Create game environment
    print("\n" + "="*70)
    print("Initializing Environment...")
    print("="*70)
    
    # Updated to unpack 4 values
    robot_side = "right" if args.num_arms == 1 else "right"  # For single arm training
    
    simulator, meshcat, plant, diagram = create_air_hockey_simulation(
        num_arms=args.num_arms,
        time_step=args.time_step,
        use_meshcat=not args.no_meshcat,
        skip_grasp=args.skip_grasp,
        game_mode=args.mode,
        robot_side=robot_side
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
    print("\n" + "="*70) # Modified from "\n" + "="*70 to "="*70
    print("Starting Simulation...")
    print("="*70)
    print(f"[INFO] Running simulation from 0.00s to {args.game_duration}s")
    
    # Get puck for monitoring
    try:
        puck_model = plant.GetModelInstanceByName("puck")
        puck_body = plant.GetBodyByName("puck_body_link", puck_model)
        
        # Create predictive puck monitor for physics
        from scripts.env.puck_monitor import PredictivePuckMonitor
        
        # Determine robot side for single-arm modes
        robot_side = "right" if args.num_arms == 1 else "right"  # Default to right for training
        
        puck_monitor = PredictivePuckMonitor(
            plant, puck_body,
            x_bounds=(-1.0, 1.0),  # Goals
            y_bounds=(-0.52, 0.52),  # Walls (adjusted for buffer)
            restitution=1.0,  # Perfect elastic - conservation of momentum
            dt=0.01,
            max_velocity=args.max_puck_velocity,
            game_mode=args.mode,  # Pass game mode
            robot_side=robot_side
        )
        
        
        print("[INFO] Setting up actuator-based robot control...")
        
        # Create actuator-based controllers
        from scripts.kinematics.actuator_controller import ActuatorBasedController
        controllers = {}
        robot_models = {}
        gripper_models = {}
        
        if args.num_arms >= 1:
            robot_models['right'] = plant.GetModelInstanceByName("right_iiwa")
            gripper_models['right'] = plant.GetModelInstanceByName("right_wsg")
            controllers['right'] = ActuatorBasedController(
                plant, robot_models['right'], gripper_models['right']
            )
        
        if args.num_arms >= 2:
            robot_models['left'] = plant.GetModelInstanceByName("left_iiwa")
            gripper_models['left'] = plant.GetModelInstanceByName("left_wsg")
            controllers['left'] = ActuatorBasedController(
                plant, robot_models['left'], gripper_models['left']
            )
        
        # Monitor puck state during simulation
        current_time = 0.0
        last_print_time = 0.0
        last_control_time = 0.0
        print_interval = 0.5
        control_interval = 0.02  # 50 Hz control for faster reactions
        
        print("[INFO] Starting simulation with actuator control...")
        
        while current_time < args.game_duration:
            simulator.AdvanceTo(current_time + 0.01)
            current_time = simulator.get_context().get_time()
            
            context = simulator.get_mutable_context()
            plant_context = plant.GetMyMutableContextFromRoot(context)
            
            # Update puck physics
            puck_monitor.update(context, plant_context)
            
            # Get puck state
            pose = plant.EvalBodyPoseInWorld(plant_context, puck_body)
            vel = plant.EvalBodySpatialVelocityInWorld(plant_context, puck_body)
            puck_pos = np.array(pose.translation())
            puck_vel = np.array(vel.translational())
            
            # Robot actuator control
            if current_time - last_control_time >= control_interval:
                # Save puck state for validation
                puck_z_before = puck_pos[2]
                
                # Control right robot via velocities (simpler than actuators)
                if 'right' in controllers:
                    q_right = plant.GetPositions(plant_context, robot_models['right'])
                    qd_right = plant.GetVelocities(plant_context, robot_models['right'])
                    
                    has_cmd, torques = controllers['right'].update(
                        puck_pos, puck_vel, q_right, qd_right, 
                        plant_context, robot_x_side=1.0
                    )
                    
                    # Quick fix: Use velocity control instead of torques
                    if has_cmd and controllers['right'].target_q is not None:
                        # Compute desired velocity from position error
                        position_error = controllers['right'].target_q - q_right
                        desired_velocity = position_error * 3.5  # Increased from 2.0 for faster reaction
                        desired_velocity = np.clip(desired_velocity, -1.5, 1.5)  # Increased speed limit
                        
                        # Set velocity directly
                        plant.SetVelocities(plant_context, robot_models['right'], desired_velocity)
                
                # Control left robot
                if 'left' in controllers:
                    q_left = plant.GetPositions(plant_context, robot_models['left'])
                    qd_left = plant.GetVelocities(plant_context, robot_models['left'])
                    
                    has_cmd, torques = controllers['left'].update(
                        puck_pos, puck_vel, q_left, qd_left,
                        plant_context, robot_x_side=-1.0
                    )
                    
                    if has_cmd and controllers['left'].target_q is not None:
                        position_error = controllers['left'].target_q - q_left
                        desired_velocity = position_error * 3.5  # Faster reaction
                        desired_velocity = np.clip(desired_velocity, -1.5, 1.5)
                        plant.SetVelocities(plant_context, robot_models['left'], desired_velocity)
                
                # Validate puck wasn't affected
                pose_check = plant.EvalBodyPoseInWorld(plant_context, puck_body)
                puck_z_after = pose_check.translation()[2]
                if abs(puck_z_after - puck_z_before) > 0.01:
                    print(f"[WARNING] Puck Z corrupted: {puck_z_before:.3f} -> {puck_z_after:.3f}")
                
                last_control_time = current_time
            
            # Print puck state
            if current_time - last_print_time >= print_interval:
                # Print puck
                print(f"[t={current_time:.2f}s] Puck pos: [{puck_pos[0]:.3f}, {puck_pos[1]:.3f}, {puck_pos[2]:.3f}], "
                      f"vel: [{puck_vel[0]:.3f}, {puck_vel[1]:.3f}, {puck_vel[2]:.3f}]")
                
                # Print robot status
                if 'right' in controllers and controllers['right'].target_q is not None:
                    q_right = plant.GetPositions(plant_context, robot_models['right'])
                    error = np.linalg.norm(controllers['right'].target_q - q_right)
                    print(f"  Right robot: error={error:.3f}")
                
                if 'left' in controllers and controllers['left'].target_q is not None:
                    q_left = plant.GetPositions(plant_context, robot_models['left']) 
                    error = np.linalg.norm(controllers['left'].target_q - q_left)
                    print(f"  Left robot: error={error:.3f}")
                
                last_print_time = current_time
                
    except Exception as e:
        print(f"[ERROR] Simulation monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fall back to simple advance
        simulator.AdvanceTo(args.game_duration)
    
    print() # Added this print()
    
    # The original code's simulation advance logic is replaced by the monitoring loop above.
    # The following lines are removed as they are superseded by the new monitoring loop.
    # current_time = simulator.get_context().get_time()
    # desired_end_time = current_time + args.game_duration
    # print(f"[INFO] Running simulation from {current_time:.2f}s to {desired_end_time:.2f}s")
    
    # try:
    #     simulator.AdvanceTo(desired_end_time)
    # except KeyboardInterrupt:
    #     print("\n\nSimulation interrupted by user")
    
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
