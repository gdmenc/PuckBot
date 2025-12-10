"""
Main entry point for Air Hockey Robot Game
"""
from argparse import ArgumentParser
import time
import numpy as np
import sys
import pickle
from pathlib import Path
from datetime import datetime

from scripts.env.configure import create_air_hockey_simulation, initialize_puck, initialize_paddles, initialize_robots
from scripts.logging.match_logger import MatchLogger

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
    parser.add_argument("--max_puck_velocity", type=float, default=5.0,
                        help="Maximum puck velocity in m/s (default: 5.0)")
    parser.add_argument("--log", action="store_true", default=False,
                        help="Enable match logging (default: OFF for performance)")
    parser.add_argument("--record", action="store_true", default=False,
                        help="Record animation for replay (default: OFF for performance)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for match logs")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    # AUTO-SET num_arms based on mode:
    # - hit and defend are single-robot training modes
    # - tournament is 2-robot competitive mode
    if args.mode in ["hit", "defend"]:
        num_arms = 1
        robot_side = "right"  # Single training robot on right side
        if args.num_arms != 1:
            print(f"[INFO] Mode '{args.mode}' requires single robot - overriding num_arms to 1")
    else:  # tournament
        num_arms = args.num_arms
        robot_side = "right" if num_arms == 1 else "right"
    
    print("="*70)
    print("AIR HOCKEY ROBOT GAME")
    print("="*70)
    print(f"Configuration:")
    print(f"  Num Arms: {num_arms}")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Time step: {args.time_step}s")
    print(f"  Game duration: {args.game_duration}s")
    
    # Create game environment
    print("\n" + "="*70)
    print("Initializing Environment...")
    print("="*70)
    
    simulator, meshcat, plant, diagram = create_air_hockey_simulation(
        num_arms=num_arms,
        time_step=args.time_step,
        use_meshcat=not args.no_meshcat,
        skip_grasp=args.skip_grasp,
        game_mode=args.mode,
        robot_side=robot_side
    )
    
    # Positions are now initialized inside create_air_hockey_simulation()
    
    # Initialize match logger
    logger = None
    if args.log:
        logger = MatchLogger(log_dir=args.log_dir, mode=args.mode)
        print(f"[INFO] Match logging enabled - logs will be saved to {args.log_dir}/")
    
    # Start recording if enabled
    if meshcat and args.record:
        print("[INFO] Animation recording enabled")
        meshcat.StartRecording()
        print("[INFO] Recording started - use Animation tab to replay")
    
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
        
            if num_arms == 2 and adapter.robot1_model:  # Left
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
        
        # Robot side already set above based on num_arms and mode
        
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
        
        # Always create right robot controller (exists in all modes)
        robot_models['right'] = plant.GetModelInstanceByName("right_iiwa")
        gripper_models['right'] = plant.GetModelInstanceByName("right_wsg")
        controllers['right'] = ActuatorBasedController(
            plant, robot_models['right'], gripper_models['right']
        )
        print(f"[INFO] Initialized right robot controller")
        if logger:
            logger.add_robot('right')
        
        # Only create left robot controller if we have 2 arms (tournament mode)
        if num_arms >= 2:
            robot_models['left'] = plant.GetModelInstanceByName("left_iiwa")
            gripper_models['left'] = plant.GetModelInstanceByName("left_wsg")
            controllers['left'] = ActuatorBasedController(
                plant, robot_models['left'], gripper_models['left']
            )
            print(f"[INFO] Initialized left robot controller")
            if logger:
                logger.add_robot('left')
        
        # Monitor puck state during simulation
        current_time = 0.0
        last_print_time = 0.0
        last_control_time = 0.0
        print_interval = 0.5
        control_interval = 0.01  # 100 Hz control for real-time tracking (was 0.02/50Hz)
        
        # Track last puck position for smart replanning
        last_puck_pos = None
        replan_threshold = 0.15  # meters - increased from 0.05 to reduce excessive replanning
        
        print("[INFO] Starting simulation with actuator control...")
        print(f"[INFO] Control rate: {1.0/control_interval:.0f} Hz")
        
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
            
            # Update logger periodically (not every frame - too expensive!)
            # Only update every 0.1 seconds to avoid slowdown
            if logger and current_time - getattr(logger, '_last_update_time', 0) >= 0.1:
                # Cache gripper bodies if not already cached
                if not hasattr(logger, '_gripper_bodies'):
                    logger._gripper_bodies = {}
                    for name, gripper_model in gripper_models.items():
                        logger._gripper_bodies[name] = plant.GetBodyByName("body", gripper_model)
                
                # Get robot positions efficiently
                robot_positions = {}
                for name, gripper_body in logger._gripper_bodies.items():
                    gripper_pose = plant.EvalBodyPoseInWorld(plant_context, gripper_body)
                    robot_positions[name] = np.array(gripper_pose.translation())
                
                logger.update(current_time, puck_pos, puck_vel, robot_positions)
                logger._last_update_time = current_time
                
                # Detect puck contacts - ACCURATE detection (visual overlap only)
                # Paddle radius ~0.05m + puck radius 0.04m = 0.09m when touching
                # Contact = when center-to-center distance < 0.06m (accounting for gripper offset)
                for name, position in robot_positions.items():
                    distance_to_puck = np.linalg.norm(position[:2] - puck_pos[:2])
                    if distance_to_puck < 0.06:  # TRUE contact only (was 0.12 - too large!)
                        if not hasattr(logger, '_last_contact') or logger._last_contact.get(name, 0) < current_time - 0.5:
                            was_planned = current_time - getattr(controllers.get(name), 'last_intercept_time', 0) < 0.5
                            logger.robot_stats[name].record_puck_contact(was_planned)
                            if not hasattr(logger, '_last_contact'):
                                logger._last_contact = {}
                            logger._last_contact[name] = current_time
            
            # Goal detection (fast, check every frame)
            goal_scored = False
            scoring_robot = None
            opponent_robot = None
            
            # CRITICAL: Goal positions vs. which robot scores
            # - Left goal at X = -1.05 (negative) → RIGHT robot scores
            # - Right goal at X = +1.05 (positive) → LEFT robot scores
            if puck_pos[0] < -1.05:  # Puck entered left goal
                goal_scored = True
                scoring_robot = 'right'  # RIGHT robot scored! (was attacking left goal)
                opponent_robot = 'left'
            elif puck_pos[0] > 1.05:  # Puck entered right goal
                goal_scored = True
                scoring_robot = 'left'  # LEFT robot scored! (was attacking right goal)
                opponent_robot = 'right'
            
            if goal_scored:
                if logger:
                    logger.record_goal(scoring_robot, opponent_robot)
                
                # Reset puck to center
                initialize_puck(simulator, plant, plant.GetModelInstanceByName("puck"))
                print(f"\n[GOAL] {scoring_robot.upper()} scored!")
                print(f"Score: Right {logger.final_score['right'] if logger else '?'} - {logger.final_score['left'] if logger else '?'} Left\n")
            
            # Robot actuator control
            if current_time - last_control_time >= control_interval:
                # Save puck state for validation
                puck_z_before = puck_pos[2]
                
                # Check if we need to replan (puck moved significantly)
                should_replan = False
                if last_puck_pos is None:
                    should_replan = True
                else:
                    puck_moved = np.linalg.norm(puck_pos[:2] - last_puck_pos[:2])
                    if puck_moved > replan_threshold:
                        should_replan = True
                
                # Control right robot via velocities (simpler than actuators)
                if 'right' in controllers:
                    q_right = plant.GetPositions(plant_context, robot_models['right'])
                    qd_right = plant.GetVelocities(plant_context, robot_models['right'])
                    
                    # Only call update if we should replan
                    if should_replan:
                        # Track strike attempt for logging
                        if logger:
                            logger.robot_stats['right'].record_strike_attempt()
                        
                        has_cmd, torques = controllers['right'].update(
                            puck_pos, puck_vel, q_right, qd_right, 
                            plant_context, robot_x_side=1.0
                        )
                    else:
                        has_cmd = controllers['right'].target_q is not None
                    
                    # Use high-gain velocity control for fast, responsive motion
                    if has_cmd and controllers['right'].target_q is not None:
                        # Compute desired velocity from position error
                        position_error = controllers['right'].target_q - q_right
                        # MUCH higher gain for very fast response (was 6.0, now 8.0)
                        desired_velocity = position_error * 8.0
                        # Very high velocity limit to intercept fast pucks (was 2.5, now 4.0)
                        desired_velocity = np.clip(desired_velocity, -4.0, 4.0)
                        
                        # Set velocity directly
                        plant.SetVelocities(plant_context, robot_models['right'], desired_velocity)
                    
                    # Gripper fingers are now WELDED - no position control needed!
                    # DON'T zero velocity - paddle needs momentum to hit puck!
                    # Gripper inherits arm velocity for proper force transfer
                
                # Control left robot
                if 'left' in controllers:
                    q_left = plant.GetPositions(plant_context, robot_models['left'])
                    qd_left = plant.GetVelocities(plant_context, robot_models['left'])
                    
                    if should_replan:
                        # Track strike attempt for logging
                        if logger:
                            logger.robot_stats['left'].record_strike_attempt()
                        
                        has_cmd, torques = controllers['left'].update(
                            puck_pos, puck_vel, q_left, qd_left,
                            plant_context, robot_x_side=-1.0
                        )
                    else:
                        has_cmd = controllers['left'].target_q is not None
                    
                    if has_cmd and controllers['left'].target_q is not None:
                        position_error = controllers['left'].target_q - q_left
                        desired_velocity = position_error * 6.0  # Increased from 5.0 for faster response
                        desired_velocity = np.clip(desired_velocity, -2.5, 2.5)
                        plant.SetVelocities(plant_context, robot_models['left'], desired_velocity)
                    
                    # Gripper fingers are now WELDED - no position control needed!
                    # DON'T zero velocity - paddle needs momentum to hit puck!
                    # Gripper inherits arm velocity for proper force transfer
                
                # Update last puck position for next replan check
                if should_replan:
                    last_puck_pos = puck_pos.copy()
                
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
    
    # Save match log and print summary
    if logger:
        print("\n" + "="*70)
        print("SAVING MATCH DATA")
        print("="*70)
        
        logger.print_summary()
        log_file = logger.save_match_log()
        print(f"\n✅ Match log saved: {log_file}")
    
    # Save animation recording
    if meshcat and args.record:
        try:
            print(f"\n[RECORDING] Saving animation...")
            
            # Create recordings directory with mode-specific subdirectory
            mode_dir_name = f"{args.mode.lower()}_mode"
            recordings_dir = Path("recordings") / mode_dir_name
            recordings_dir.mkdir(parents=True, exist_ok=True)
            
            # Save recording
            match_id = logger.match_id if logger else datetime.now().strftime("%Y%m%d_%H%M%S")
            recording_file = recordings_dir / f"match_{match_id}.html"
            
            # Stop and publish recording
            meshcat.StopRecording()
            meshcat.PublishRecording()
            
            # Also save to file
            html = meshcat.StaticHtml()
            recording_file.write_text(html)
            
            print(f"✅ Animation saved: {recording_file}")
            print(f"✅ Animation available in Meshcat Animation tab (use Controls)")
            print(f"   Recording directory: {recordings_dir}")
            
        except Exception as e:
            print(f"[WARNING] Could not save animation: {e}")
    
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
