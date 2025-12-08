import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union

from typing import Optional, Union, Any

# from scripts.drake_implementation import AirHockeyDrakeEnv
# from scripts.env.game_env import AirHockeyGameEnv

from .states import PuckState, RobotState, JointTrajectory
from .intercept_planner import InterceptPlanner, RobotConfig
from .puck_predictor import PuckPredictor

class MotionController:
    """
    High-level controller that connects puck observation, intercept planning,
        and trajectory execution for a single robot.
    """

    def __init__(
        self,
        env: Any,
        robot_id: int,
        home_q: NDArray[np.float64] | None = None
    ) -> None:
        """
        Args:
            env: Drake air hockey environment (base or game)
            robot_id: Which robot
            home_q: Home joint config
        """
        self.env = env
        self.robot_id = robot_id

        # Determine models based on ID
        if robot_id == 1:
            self.robot_model = env.robot1_model
            # Look for left gripper
            try:
                self.gripper_model = env.plant.GetModelInstanceByName("left_wsg")
                paddle_frame = "body"
            except:
                 # Fallback if names differ or testing
                self.gripper_model = self.robot_model
                paddle_frame = "robot1/tool_body" # Fallback to old behavior
                
            robot_config = RobotConfig(
                robot_id=1,
                x_min=-1.0,
                x_max=-0.2,
                y_min=-0.5,
                y_max=0.5
            )
        else:
            self.robot_model = env.robot2_model
            try:
                self.gripper_model = env.plant.GetModelInstanceByName("right_wsg")
                paddle_frame = "body"
            except:
                self.gripper_model = self.robot_model
                paddle_frame = "robot2/tool_body"
                
            robot_config = RobotConfig(
                robot_id=2,
                x_min=0.2,
                x_max=1.0,
                y_min=-0.5,
                y_max=0.5
            )

        self.home_q = home_q if home_q is not None else np.array(
            [-1.57, 0.5, 0.0, -1.2, 0.0, 1.6, 0.0]
        )

        self.planner = InterceptPlanner(
            plant=env.plant,
            robot_model=self.robot_model,
            paddle_frame_name=paddle_frame,
            robot_config=robot_config,
            frame_model=self.gripper_model
        )

        self.current_trajectory: Optional[JointTrajectory] = None
        self.trajectory_start_time: float = 0.0

    def get_puck_state(self) -> PuckState:
        position, velocity = self.env.get_puck_state()
        t = self.env.simulator.get_context().get_time()
        return PuckState(position=position, velocity=velocity, t=t)

    def get_robot_state(self) -> RobotState:
        q = self.env.get_robot_joint_positions(self.robot_model)
        t = self.env.simulator.get_context().get_time()
        return RobotState(q=q, dq=np.zeros_like(q), t=t)

    def update(self) -> bool:
        """
        Main update loop, called each control cycle.

        Returns:
            True if a new trajectory was planned, False otherwise
        """
        puck = self.get_puck_state()
        robot = self.get_robot_state()
        t_now = puck.t

        if not self._should_intercept(puck):
            if not self._is_at_home(robot):
                self.current_trajectory = self.planner.plan_to_home(robot, self.home_q)
                return True
            return False
        
        trajectory = self.planner.plan_intercept(
            puck=puck,
            robot=robot,
            context=self.env.plant_context,
            horizon=2.0
        )

        if trajectory is not None:
            self.current_trajectory = trajectory
            self.trajectory_start_time = t_now
            return True

        return False

    def _should_intercept(self, puck: PuckState) -> bool:
        if self.robot_id == 1:
            return puck.velocity[0] < -0.05
        else:
            return puck.velocity[0] > 0.05

    def _is_at_home(self, robot: RobotState, tolerance: float = 0.1) -> bool:
        return np.allclose(robot.q, self.home_q, atol=tolerance)

    def execute_step(self, dt: float = 0.01) -> None:
        if self.current_trajectory is None:
            return

        t_now = self.env.simulator.get_context().get_time()
        q_desired = self.current_trajectory.at_time(t_now)
        self.env.set_robot_joint_positions(self.robot_model, q_desired)

        if t_now >= self.current_trajectory.times[-1]:
            self.current_trajectory = None

    def run_control_loop(self, duration: float, control_dt: float = 0.02) -> None:
        t_end = self.env.simulator.get_context().get_time() + duration

        while self.env.simulator.get_context().get_time() < t_end:
            self.update()
            self.execute_step()
            self.env.step(duration=control_dt)
            self.env.diagram.ForcedPublish(self.env.context)