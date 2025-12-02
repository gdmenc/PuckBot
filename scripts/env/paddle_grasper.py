"""
Paddle Grasping Module
Handles the sequence of grasping the paddle from the side table and positioning it for gameplay.
"""
import numpy as np
from pydrake.all import (
    RigidTransform,
    RollPitchYaw,
    SpatialVelocity,
)
from typing import Optional, Tuple
from scripts.simple_motion_controller import SimpleMotionController


class PaddleGrasper:
    """
    Manages the paddle grasping sequence:
    1. Move to pre-grasp position above paddle
    2. Move down to grasp position
    3. Grasp paddle (weld to robot)
    4. Lift paddle
    5. Move to table and orient paddle parallel to table
    """
    
    def __init__(
        self,
        env,
        robot_model,
        tool_frame_name: str,
        motion_controller: Optional[SimpleMotionController] = None
    ):
        """
        Args:
            env: AirHockeyGameEnv instance
            robot_model: Robot model instance
            tool_frame_name: Name of tool/end-effector frame
            motion_controller: Optional motion controller (will create if None)
        """
        self.env = env
        self.robot_model = robot_model
        self.tool_frame_name = tool_frame_name
        
        if motion_controller is None:
            self.motion_controller = SimpleMotionController(
                env.plant,
                robot_model,
                tool_frame_name
            )
        else:
            self.motion_controller = motion_controller
        
        self.tool_frame = env.plant.GetFrameByName(tool_frame_name, robot_model)
        self.world_frame = env.plant.world_frame()
    
    def execute_grasp_sequence(self, dt: float = 0.01) -> bool:
        """
        Execute the complete paddle grasping sequence.
        
        Args:
            dt: Time step for motion execution
            
        Returns:
            True if sequence completed successfully
        """
        if self.env.paddle_grasped:
            print("Paddle already grasped!")
            return True
        
        if self.env.paddle_body is None:
            print("No paddle object found!")
            return False
        
        print("\n=== Starting Paddle Grasp Sequence ===")
        
        # Step 1: Move to pre-grasp position
        print("Step 1: Moving to pre-grasp position...")
        paddle_pose = self.env.get_paddle_pose()
        if paddle_pose is None:
            print("Failed to get paddle pose!")
            return False
        
        pre_grasp_pos = paddle_pose.translation() + np.array([0, 0, 0.15])
        if not self._move_to_position(pre_grasp_pos, duration=2.0, dt=dt):
            print("Failed to reach pre-grasp position!")
            return False
        
        # Step 2: Move down to grasp position
        print("Step 2: Moving down to grasp position...")
        grasp_pos = paddle_pose.translation() + np.array([0, 0, 0.05])
        if not self._move_to_position(grasp_pos, duration=1.5, dt=dt):
            print("Failed to reach grasp position!")
            return False
        
        # Step 3: Grasp paddle (weld to robot)
        print("Step 3: Grasping paddle...")
        if not self._grasp_paddle():
            print("Failed to grasp paddle!")
            return False
        
        # Step 4: Lift paddle
        print("Step 4: Lifting paddle...")
        lift_pos = grasp_pos + np.array([0, 0, 0.2])
        if not self._move_to_position(lift_pos, duration=1.5, dt=dt):
            print("Warning: Failed to lift paddle, but continuing...")
        
        # Step 5: Move to table and orient paddle parallel
        print("Step 5: Moving to table and orienting paddle...")
        table_pos = np.array([-0.6, 0.0, self.env.table_height + 0.15])
        if not self._move_to_position(table_pos, duration=2.0, dt=dt):
            print("Warning: Failed to reach table position, but paddle is grasped")
        
        print("=== Paddle Grasp Sequence Complete ===\n")
        return True
    
    def _move_to_position(
        self, 
        target_position: np.ndarray, 
        duration: float = 2.0, 
        dt: float = 0.01
    ) -> bool:
        """
        Move robot to target position using motion controller.
        
        Args:
            target_position: Target 3D position [x, y, z]
            duration: Motion duration
            dt: Time step
            
        Returns:
            True if motion completed successfully
        """
        success, trajectory = self.motion_controller.move_to_position(
            target_position,
            self.env.plant_context,
            duration=duration,
            dt=dt
        )
        
        if not success:
            return False
        
        # Execute trajectory
        for q in trajectory:
            self.env.set_robot_joint_positions(self.robot_model, q)
            self.env.step(duration=dt)
            self.env.diagram.ForcedPublish(self.env.context)
        
        return True
    
    def _grasp_paddle(self) -> bool:
        """
        Weld the paddle to the robot's tool body to simulate grasping.
        
        Returns:
            True if grasp successful
        """
        if self.env.paddle_grasped:
            return True
        
        try:
            # Get current poses
            X_WT = self.env.plant.EvalBodyPoseInWorld(
                self.env.plant_context, 
                self.tool_frame.body()
            )
            X_WP = self.env.plant.EvalBodyPoseInWorld(
                self.env.plant_context,
                self.env.paddle_body
            )
            
            # Compute transform from tool to paddle
            X_TW = X_WT.inverse()
            X_TP = X_TW @ X_WP
            
            # Adjust: align paddle surface parallel to table (horizontal)
            # Rotate paddle so surface is horizontal (z-up)
            R_align = RollPitchYaw(0, np.pi/2, 0).ToRotationMatrix()
            X_TP_adjusted = RigidTransform(
                R_align @ X_TP.rotation(),
                X_TP.translation()
            )
            
            # Weld paddle to tool
            self.env.plant.WeldFrames(
                self.tool_frame,
                self.env.paddle_body.body_frame(),
                X_TP_adjusted
            )
            
            self.env.paddle_grasped = True
            return True
            
        except Exception as e:
            print(f"Failed to grasp paddle: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def plan_grasp_trajectory(
        self,
        paddle_pose: RigidTransform
    ) -> Tuple[bool, list]:
        """
        Plan trajectory for grasping sequence.
        
        Args:
            paddle_pose: Current paddle pose in world frame
            
        Returns:
            (success, waypoints): List of waypoint positions
        """
        waypoints = []
        
        # Pre-grasp position
        pre_grasp = paddle_pose.translation() + np.array([0, 0, 0.15])
        waypoints.append(pre_grasp)
        
        # Grasp position
        grasp = paddle_pose.translation() + np.array([0, 0, 0.05])
        waypoints.append(grasp)
        
        # Lift position
        lift = grasp + np.array([0, 0, 0.2])
        waypoints.append(lift)
        
        # Table position
        table_pos = np.array([-0.6, 0.0, self.env.table_height + 0.15])
        waypoints.append(table_pos)
        
        return True, waypoints

