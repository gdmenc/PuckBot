"""
Game Controller
High-level controller that manages the complete air hockey game:
1. Paddle grasping sequence
2. Gameplay using intercept planning
"""
import numpy as np
from typing import Optional
from .motion_controller import MotionController
from .states import PuckState, RobotState


class GameController:
    """
    Manages the complete air hockey game flow:
    - Initialization and paddle grasping
    - Gameplay with intercept planning
    """
    
    def __init__(
        self,
        env,
        robot_id: int = 1,
        auto_grasp: bool = True
    ):
        """
        Args:
            env: AirHockeyGameEnv instance
            robot_id: Which robot to control (1 or 2)
            auto_grasp: If True, automatically execute grasp sequence on start
        """
        self.env = env
        self.robot_id = robot_id
        self.auto_grasp = auto_grasp
        
        # Create motion controller for gameplay
        self.motion_controller = MotionController(env, robot_id)
        
        # Paddle grasper (will be created when needed)
        self.paddle_grasper = None
        
        # Game state
        self.game_started = False
        self.paddle_grasped = False
    
    def initialize(self) -> bool:
        """
        Initialize the game: grasp paddle if needed.
        
        Returns:
            True if initialization successful
        """
        if self.env.is_paddle_grasped():
            print("Paddle already grasped!")
            self.paddle_grasped = True
            self.game_started = True
            return True
        
        if not self.auto_grasp:
            print("Auto-grasp disabled. Paddle must be grasped manually.")
            return False
        
        # Create paddle grasper
        from scripts.env.paddle_grasper import PaddleGrasper
        from scripts.kinematics.motion_controller import MotionController
        
        if self.robot_id == 1:
            robot_model = self.env.robot1_model
            tool_frame = "robot1/tool_body"
        else:
            robot_model = self.env.robot2_model
            tool_frame = "robot2/tool_body"
        
        motion_controller = MotionController(
            self.env,
            self.robot_id
        )
        
        self.paddle_grasper = PaddleGrasper(
            self.env,
            robot_model,
            tool_frame,
            motion_controller
        )
        
        # Execute grasp sequence
        print(f"\n=== Initializing Robot {self.robot_id} ===")
        success = self.paddle_grasper.execute_grasp_sequence(dt=0.01)
        
        if success:
            self.paddle_grasped = True
            self.game_started = True
            print(f"Robot {self.robot_id} ready for gameplay!")
            return True
        else:
            print(f"Failed to initialize robot {self.robot_id}")
            return False
    
    def update(self) -> bool:
        """
        Main game update loop.
        Plans and executes intercept trajectories.
        
        Returns:
            True if action was taken
        """
        if not self.game_started:
            return False
        
        # Update motion controller
        return self.motion_controller.update()
    
    def execute_step(self, dt: float = 0.01) -> None:
        """
        Execute one control step.
        
        Args:
            dt: Time step
        """
        if not self.game_started:
            return
        
        self.motion_controller.execute_step(dt)
    
    def run_game(self, duration: float, control_dt: float = 0.02) -> None:
        """
        Run the game for specified duration.
        
        Args:
            duration: Game duration in seconds
            control_dt: Control loop time step
        """
        if not self.game_started:
            print("Game not started! Call initialize() first.")
            return
        
        print(f"\n=== Starting Gameplay (Robot {self.robot_id}) ===")
        self.motion_controller.run_control_loop(duration, control_dt)
    
    def get_puck_state(self) -> PuckState:
        """Get current puck state."""
        return self.motion_controller.get_puck_state()
    
    def get_robot_state(self) -> RobotState:
        """Get current robot state."""
        return self.motion_controller.get_robot_state()
    
    def reset(self, reset_paddle: bool = True) -> None:
        """
        Reset the game.
        
        Args:
            reset_paddle: If True, reset paddle to side table
        """
        self.env.reset(reset_paddle=reset_paddle)
        self.game_started = False
        self.paddle_grasped = False
        self.motion_controller.current_trajectory = None

