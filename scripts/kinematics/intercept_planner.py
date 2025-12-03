import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .states import PuckState, RobotState, JointTrajectory
from .puck_predictor import PuckPredictor, TableBounds

@dataclass
class RobotConfig:
    robot_id: int
    x_min: float
    x_max: float
    y_min: float = -0.5
    y_max: float = 0.5
    paddle_height: float = 0.15
    joint_velocity_limit: float = 1.5
    n_joints: int = 7


class InterceptPlanner:
    """
    Plans intercept trajectories for a robot to reach a predicted puck position.
    """

    def __init__(
        self,
        plant,
        robot_model,
        paddle_frame_name: str,
        robot_config: RobotConfig,
        puck_predictor: PuckPredictor | None = None
    ) -> None:
        """
        Args:
            plant: Drake MultibodyPlant
            robot_model: Robot model instance
            paddle_frame_name: Name of paddle/end-effector frame
            robot_config: Robot workspace and limits
            puck_predictor: Puck trajectory predictor
        """
        self.plant = plant
        self.robot_model = robot_model
        self.config = robot_config
        self.predictor = puck_predictor or PuckPredictor()

        self.paddle_frame = plant.GetFrameByName(paddle_frame_name, robot_model)
        self.world_frame = plant.world_frame()

    def plan_intercept(
        self,
        puck: PuckState,
        robot: RobotState,
        context,
        horizon: float = 2.0,
        n_candidates: int = 20,
        trajectory_dt: float = 0.02
    ) -> Optional[JointTrajectory]:
        """
        Plan a trajectory to intercept the puck.

        Args:
            puck: Current puck state
            robot: Current robot state
            context: Drake plant context
            horizon: How far ahead to look for intercepts (seconds)
            n_candidates: Number of intercept candidates to evaluate
            trajectory_dt: time resolution for output trajectory

        Returns:
            JointTrajectory if feasible intercept found, None otherwise
        """
        predicted_path = self.predictor.predict(puck, horizon, dt=0.02)
        if len(predicted_path) < 2:
            return None

        candidates = self._filter_reachable_candidates(predicted_path)
        if not candidates:
            return None

        step = max(1, len(candidates) // n_candidates)
        sampled = candidates[::step]

        for t_intercept, intercept_pos in sampled:
            available_time = t_intercept - puck.t
            if available_time < 0.1:
                continue

            target_pos = intercept_pos.copy()
            target_pos[2] = self.config.paddle_height + TableBounds().z

            success, q_target = self._solve_ik(target_pos, robot.q, context)
            if not success:
                continue

            if not self._is_time_feasible(robot.q, q_target, available_time):
                continue

            trajectory = self._generate_trajectory(
                q_start=robot.q,
                q_end=q_target,
                t_start=puck.t,
                t_end=t_intercept,
                dt=trajectory_dt
            )
            return trajectory

        return None

    def _filter_reachable_candidates(
        self,
        path: List[Tuple[float, NDArray[np.float64]]]
    ) -> List[Tuple[float, NDArray[np.float64]]]:
        """
        Filter trajectory to those within robot's workspace.
        """
        reachable = []
        for t, pos in path:
            x, y = pos[0], pos[1]
            if (self.config.x_min <= x <= self.config.x_max and
                self.config.y_min <= y <= self.config.y_max):
                reachable.append((t, pos))
        
        return reachable

    def _solve_ik(
        self,
        target_position: NDArray[np.float64],
        seed_q: NDArray[np.float64],
        context
    ) -> Tuple[bool, NDArray[np.float64]]:
        """
        Solve inverse kinematics for target end-effector position.

        Returns:
            (success, joint_positions)
        """
        from pydrake.multibody.inverse_kinematics import InverseKinematics
        from pydrake.solvers import Solve

        ik = InverseKinematics(self.plant, context)

        position_tolerance = 0.01
        ik.AddPositionConstraint(
            frameB=self.paddle_frame,
            p_BQ=np.zeros(3),
            frameA=self.world_frame,
            p_AQ_lower=target_position - position_tolerance,
            p_AQ_upper=target_position + position_tolerance
        )

        prog = ik.prog()
        q_vars = ik.q()

        # IK expects a guess vector for the entire plant configuration, not just this robot.
        full_q_guess = self.plant.GetPositions(context).copy()
        context_guess = self.plant.CreateDefaultContext()
        self.plant.SetPositions(context_guess, self.robot_model, seed_q)
        full_q_guess = self.plant.GetPositions(context_guess)
        prog.SetInitialGuess(q_vars, full_q_guess)

        result = Solve(prog)
        if result.is_success():
            q_solution_full = result.GetSolution(q_vars)
            context_solution = self.plant.CreateDefaultContext()
            self.plant.SetPositions(context_solution, q_solution_full)
            q_robot = self.plant.GetPositions(context_solution, self.robot_model)
            return True, q_robot
        else:
            return False, seed_q

    def _is_time_feasible(
        self,
        q_start: NDArray[np.float64],
        q_end: NDArray[np.float64],
        available_time: float
    ) -> bool:
        """
        Check if robot can move from q_start to q_end within available time.
        """
        q_diff = np.abs(q_end - q_start)
        required_velocities = q_diff / available_time
        max_required = np.max(required_velocities)
        return max_required <= self.config.joint_velocity_limit

    def _generate_trajectory(
        self,
        q_start: NDArray[np.float64],
        q_end: NDArray[np.float64],
        t_start: float,
        t_end: float,
        dt: float
    ) -> JointTrajectory:
        """
        Generate linear interpolation trajectory between two joint configs.
        """
        # TODO: Check this line
        n_steps = max(2, int((t_end - t_start) / dt) + 1)
        times = np.linspace(t_start, t_end, n_steps)

        alphas = np.linspace(0, 1, n_steps).reshape(-1, 1)
        positions = (1 - alphas) * q_start + alphas * q_end

        return JointTrajectory(times=times, positions=positions)

    def plan_to_home(
        self,
        robot: RobotState,
        home_q: NDArray[np.float64],
        duration: float = 1.0,
        dt: float = 0.02
    ) -> JointTrajectory:
        """
        Plan a trajectory to return to home position.
        """
        return self._generate_trajectory(
            q_start=robot.q,
            q_end=home_q,
            t_start=robot.t,
            t_end=robot.t + duration,
            dt=dt
        )

    def plan_to_position(
        self,
        target_position: NDArray[np.float64],
        current_q: NDArray[np.float64],
        context,
        duration: float = 2.0,
        dt: float = 0.01
    ) -> Tuple[bool, Optional[JointTrajectory]]:
        """
        Plan a trajectory to reach a specific 3D position.
        
        Args:
            target_position: [x, y, z] target in world frame
            current_q: Current joint positions
            context: Plant context for IK
            duration: Motion duration
            dt: Time step
            
        Returns:
            (success, trajectory)
        """
        success, q_target = self._solve_ik(target_position, current_q, context)
        
        if not success:
            return False, None
            
        # Simple check for timing
        if not self._is_time_feasible(current_q, q_target, duration):
            # Could potentially stretch duration here if needed
            print(f"Warning: Motion might exceed velocity limits for duration {duration}s")
            
        t_now = context.get_time()
        
        trajectory = self._generate_trajectory(
            q_start=current_q,
            q_end=q_target,
            t_start=t_now,
            t_end=t_now + duration,
            dt=dt
        )
        
        return True, trajectory