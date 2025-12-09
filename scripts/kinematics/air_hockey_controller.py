"""
Air Hockey Robot Controller

Implements reactive robot control for air hockey:
1. Predict puck trajectory
2. Find optimal intercept point
3. Move paddle to intercept
4. Hit puck with momentum transfer
"""
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from dataclasses import dataclass

from .states import PuckState, RobotState, JointTrajectory
from .puck_predictor import PuckPredictor, TableBounds
from .intercept_planner import InterceptPlanner, RobotConfig


@dataclass
class HitStrategy:
    """Strategy for hitting the puck."""
    target_point: NDArray[np.float64]  # Where to aim puck toward
    desired_speed: float  # Desired puck speed after hit
    

class AirHockeyController:
    """
    High-level controller for air hockey robot.
    
    Predicts puck motion, plans intercepts, and executes hits
    while maintaining paddle parallel to table.
    """
    
    def __init__(
        self,
        plant,
        robot_model,
        paddle_frame_name: str,
        robot_config: RobotConfig,
        table_bounds: Optional[TableBounds] = None,
        goal_x: float = 0.0  # X coordinate of opponent goal
    ):
        """
        Args:
            plant: Drake MultibodyPlant
            robot_model: Robot model instance
            paddle_frame_name: Name of paddle frame
            robot_config: Robot workspace configuration
            table_bounds: Table dimensions
            goal_x: X coordinate to aim toward (opponent's goal)
        """
        self.plant = plant
        self.robot_model = robot_model
        self.robot_config = robot_config
        self.goal_x = goal_x
        
        # Create predictor and planner
        self.predictor = PuckPredictor(
            table_bounds=table_bounds or TableBounds(),
            restitution=0.95,
            friction_decel=0.1
        )
        
        self.planner = InterceptPlanner(
            plant=plant,
            robot_model=robot_model,
            paddle_frame_name=paddle_frame_name,
            robot_config=robot_config,
            puck_predictor=self.predictor
        )
        
    def compute_intercept(
        self,
        puck: PuckState,
        robot: RobotState,
        context
    ) -> Optional[Tuple[NDArray[np.float64], float]]:
        """
        Compute optimal intercept point and timing.
        
        Returns:
            (intercept_position_3d, intercept_time) or None if no feasible intercept
        """
        # Predict puck trajectory
        horizon = 3.0  # Look 3 seconds ahead
        trajectory = self.predictor.predict(puck, horizon, dt=0.02)
        
        if not trajectory:
            return None
        
        # Find first reachable point on trajectory
        for t, pos_3d in trajectory:
            # Check if within robot workspace
            if not self._is_in_workspace(pos_3d):
                continue
            
            # Estimate if robot can reach in time
            time_available = t - robot.t
            if time_available < 0.1:  # Need at least 100ms
                continue
            
            # Simple reachability check (more sophisticated IK check could be done)
            current_paddle_pos = self._get_paddle_position(robot, context)
            distance = np.linalg.norm(pos_3d[:2] - current_paddle_pos[:2])
            
            # Assume max paddle speed of ~1.0 m/s
            if distance / time_available < 1.0:
                return (pos_3d, t)
        
        return None
    
    def plan_defensive_intercept(
        self,
        puck: PuckState,
        robot: RobotState,
        context
    ) -> Optional[JointTrajectory]:
        """
        Plan to block puck from entering our goal.
        Simple strategy: move paddle to predicted Y position at our goal line.
        """
        # Predict where puck will be at our goal line
        trajectory = self.predictor.predict(puck, horizon=5.0, dt=0.01)
        
        goal_x = self.robot_config.x_min if self.goal_x < 0 else self.robot_config.x_max
        
        for t, pos_3d in trajectory:
            if abs(pos_3d[0] - goal_x) < 0.1:  # Near goal line
                # Move to block at this Y position
                block_pos = np.array([
                    goal_x + 0.2,  # Stand 20cm in front of goal
                    pos_3d[1],
                    self.robot_config.paddle_height
                ])
                
                # Plan trajectory to blocking position
                success, traj = self.planner.plan_to_position(
                    target_position=block_pos,
                    current_q=robot.joint_positions,
                    context=context,
                    duration=max(0.5, t - robot.t - 0.2)
                )
                
                if success:
                    return traj
        
        return None
    
    def plan_attacking_hit(
        self,
        puck: PuckState,
        robot: RobotState,
        context
    ) -> Optional[JointTrajectory]:
        """
        Plan to hit puck toward opponent goal.
        
        Strategy:
        1. Predict puck trajectory
        2. Find intercept point
        3. Calculate hit angle toward goal
        4. Plan trajectory with paddle velocity at intercept
        """
        result = self.compute_intercept(puck, robot, context)
        if not result:
            return None
        
        intercept_pos, intercept_time = result
        
        # Calculate desired hit direction (toward goal)
        goal_pos = np.array([self.goal_x, 0.0, self.robot_config.paddle_height])
        hit_direction = goal_pos[:2] - intercept_pos[:2]
        hit_direction = hit_direction / np.linalg.norm(hit_direction)
        
        # Plan trajectory to intercept with some paddle velocity
        # TODO: Include desired paddle velocity for momentum transfer
        success, traj = self.planner.plan_to_position(
            target_position=intercept_pos,
            current_q=robot.joint_positions,
            context=context,
            duration=max(0.3, intercept_time - robot.t - 0.1)
        )
        
        if success:
            return traj
        
        return None
    
    def _is_in_workspace(self, position_3d: NDArray[np.float64]) -> bool:
        """Check if position is within robot's workspace."""
        x, y, z = position_3d
        return (
            self.robot_config.x_min <= x <= self.robot_config.x_max and
            self.robot_config.y_min <= y <= self.robot_config.y_max
        )
    
    def _get_paddle_position(self, robot: RobotState, context) -> NDArray[np.float64]:
        """Get current paddle position from robot state."""
        # This would use forward kinematics - simplified for now
        # In real implementation, evaluate paddle frame pose
        return robot.paddle_position if hasattr(robot, 'paddle_position') else np.zeros(3)
