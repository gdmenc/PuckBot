import numpy as np
from pydrake.all import (
    MultibodyPlant,
    Context,
    RigidTransform,
)
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve


class SimpleMotionController:
    """
    Basic motion controller for IIWA paddle movement.
    Provides simple point-to-point motion using IK and linear interpolation.
    """

    def __init__(self, plant: MultibodyPlant, robot_model_instance, paddle_frame_name: str):
        """
        Initializes the motion controller.

        Args:
            plant: Drake MultibodyPlant containing the robot
            robot_model_instance: ModelInstance for the specific robot
            paddle_frame_name: Name of the paddle frame
        """
        self.plant = plant
        self.robot_model = robot_model_instance
        self.paddle_frame_name = paddle_frame_name

        # Get paddle frame
        try:
            self.paddle_frame = self.plant.GetFrameByName(paddle_frame_name, robot_model_instance)
        except:
            print(f"Warning: Could not find frame {paddle_frame_name}, trying without model instance.")
            self.paddle_frame = self.plant.GetFrameByName(paddle_frame_name)

        self.world_frame = self.plant.world_frame()
        self.n_joints = self.plant.num_positions(robot_model_instance)
        print(f"SimpleMotionController initialized for {paddle_frame_name} ({self.n_joints} joints)")

    def get_paddle_position(self, context: Context) -> np.ndarray:
        """
        Get current paddle position using forward kinematics.

        Args:
            context: Drake context with current plant state

        Returns:
            3D position array [x, y, z]
        """
        pose = self.plant.CalcRelativeTransform(
            context, self.world_frame, self.paddle_frame
        )
        return pose.translation()
        
    def solve_ik_position(self, target_position: np.ndarray, context: Context, position_tolerance: float = 0.001) -> tuple:
        """
        Solve inverse kinematics to reach target position.

        Args:
            target_position: Desired paddle position [x, y, z]
            context: Current plant context
            position_tolerance: Position error tolerance (meters)

        Returns:
            (success, joint_angles): tuple of bool and np.ndarray
        """
        ik = InverseKinematics(self.plant, context)
        ik.AddPositionConstraint(
            frameB=self.paddle_frame,
            p_BQ=[0, 0, 0],
            frameA=self.world_frame,
            p_AQ_lower=target_position - position_tolerance,
            p_AQ_upper=target_position + position_tolerance
        )

        q_current = self.plant.GetPositions(context, self.robot_model)
        prog = ik.prog()
        q_variables = ik.q()
        prog.SetInitialGuess(q_variables, q_current)
        result = Solve(prog)

        if result.is_success():
            q_solution = result.GetSolution(q_variables)
            return True, q_solution
        else:
            print(f"IK failed for target position: {target_position}")
            return False, q_current
    
    def interpolate_trajectory(self, q_start: np.ndarray, q_end: np.ndarray, num_steps: int) -> np.ndarray:
        """
        Generate a simple linear interpolation trajectory in joint space.

        Args:
            q_start: Starting joint configuration
            q_end: Ending joint configuration
            num_steps: Number of interpolation steps

        Returns:
            Array of shape (num_steps + 1, n_joints) with interpolated configurations
        """
        trajectory = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            q_interp = (1 - alpha) * q_start + alpha * q_end
            trajectory.append(q_interp)
        
        return np.array(trajectory)

    def move_to_position(self, target_position: np.ndarray, context: Context, duration: float = 2.0, dt: float = 0.01) -> tuple:
        """
        Plan and return trajectory to move paddle to target position.

        Args:
            target_position: Desired paddle position [x, y, z]
            context: Current plant context
            duration: Trajectory duration (in seconds)
            dt: Time step for trajectory sampling

        Returns:
            (success, trajectory): tuple of bool and array of joint configurations (or None)
        """
        q_current = self.plant.GetPositions(context, self.robot_model)
        success, q_target = self.solve_ik_position(target_position, context)

        if not success:
            return False, None

        num_steps = int(duration / dt)
        trajectory = self.interpolate_trajectory(q_current, q_target, num_steps)

        return True, trajectory