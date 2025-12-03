"""
Main entry point for Air Hockey Robot Game
Integrates paddle grasping and gameplay using Drake simulation.
"""
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, BooleanOptionalAction
import time

from scripts.env.game_env import AirHockeyGameEnv
from scripts.kinematics.game_controller import GameController


def get_args():
    parser = ArgumentParser(description="Air Hockey Robot Game")
    arg_test = parser.add_argument_group("simulation parameters")

    arg_test.add_argument(
        "--time_step",
        type=float,
        default=0.001,
        help="Simulation time step (seconds)"
    )

    arg_test.add_argument(
        "--game_duration",
        type=float,
        default=30.0,
        help="Game duration in seconds"
    )

    arg_test.add_argument(
        "--control_dt",
        type=float,
        default=0.02,
        help="Control loop time step (seconds)"
    )

    arg_test.add_argument(
        "--robot_id",
        type=int,
        default=1,
        choices=[1, 2],
        help="Which robot to control (1 or 2)"
    )

    arg_test.add_argument(
        "--skip_grasp",
        action="store_true",
        help="Skip paddle grasping (assume paddle already grasped)"
    )

    arg_test.add_argument(
        "--no_meshcat",
        action="store_true",
        help="Disable Meshcat visualization"
    )

    arg_test.add_argument(
        "--random_puck",
        action=BooleanOptionalAction,
        default=True,
        help="Start puck with random velocity (use --no-random_puck to disable)"
    )

    arg_test.add_argument(
        "--enable_paddle",
        action="store_true",
        help="Include paddle + side table (defaults to disabled for testing)"
    )

    args = vars(parser.parse_args())
    return args


def main():
    args = get_args()
    
    print("="*70)
    print("AIR HOCKEY ROBOT GAME")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Robot ID: {args['robot_id']}")
    print(f"  Time step: {args['time_step']}s")
    print(f"  Game duration: {args['game_duration']}s")
    print(f"  Control dt: {args['control_dt']}s")
    print(f"  Skip grasp: {args['skip_grasp']}")
    print(f"  Random puck: {args['random_puck']}")
    print(f"  Include paddle: {args['enable_paddle']}")
    
    # Create game environment
    print("\n" + "="*70)
    print("Initializing Environment...")
    print("="*70)
    
    env = AirHockeyGameEnv(
        time_step=args['time_step'],
        use_meshcat=not args['no_meshcat'],
        include_paddle=args['enable_paddle']
    )
    
    # Reset environment
    env.reset(random_velocity=args['random_puck'], reset_paddle=True)
    
    if not args['no_meshcat']:
        meshcat_url = env.meshcat.web_url()
        print(f"\nMeshcat visualization: {meshcat_url}")
        print("Open this URL in your browser to view the simulation")
        time.sleep(1)  # Give user time to open browser
    
    # Create game controller
    print("\n" + "="*70)
    print("Creating Game Controller...")
    print("="*70)
    
    controller = GameController(
        env=env,
        robot_id=args['robot_id'],
        auto_grasp=not args['skip_grasp']
    )
    
    # Initialize (grasp paddle if needed)
    if not args['skip_grasp']:
        print("\n" + "="*70)
        print("Grasping Paddle...")
        print("="*70)
        
        success = controller.initialize()
        if not success:
            print("\nERROR: Failed to initialize robot!")
            print("Exiting...")
            return
        
        # Brief pause after grasping
        print("\nPaddle grasped! Preparing for gameplay...")
        time.sleep(1)
    else:
        # Assume paddle is already grasped
        controller.paddle_grasped = True
        controller.game_started = True
        print("\nSkipping paddle grasp (assuming already grasped)")
    
    # Run gameplay
    print("\n" + "="*70)
    print("Starting Gameplay...")
    print("="*70)
    print("\nThe robot will now attempt to intercept the puck!")
    print("Watch the simulation in Meshcat.\n")
    
    try:
        controller.run_game(
            duration=args['game_duration'],
            control_dt=args['control_dt']
        )
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")
    
    # Final state
    print("\n" + "="*70)
    print("Game Complete")
    print("="*70)
    
    puck_state = controller.get_puck_state()
    robot_state = controller.get_robot_state()
    
    print(f"\nFinal Puck State:")
    print(f"  Position: {puck_state.position}")
    print(f"  Velocity: {puck_state.velocity}")
    print(f"  Speed: {puck_state.speed:.3f} m/s")
    
    print(f"\nFinal Robot State:")
    print(f"  Joint positions: {robot_state.q}")
    
    print("\n" + "="*70)
    print("Simulation complete!")
    print("="*70)
    
    if not args['no_meshcat']:
        print("\nMeshcat visualization will remain open.")
        print("Press Ctrl+C to exit, or close the browser window.")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    main()

