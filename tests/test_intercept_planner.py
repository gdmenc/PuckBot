import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from movement.states import PuckState, RobotState, JointTrajectory
from movement.intercept_planner import RobotConfig


class TestRobotConfig:
    def test_default_values(self):
        """RobotConfig should have sensible defaults."""
        config = RobotConfig(robot_id=1, x_min=-1.0, x_max=-0.2)
        
        assert config.y_min == -0.5
        assert config.y_max == 0.5
        assert config.paddle_height == 0.15
        assert config.joint_velocity_limit == 1.5
        assert config.n_joints == 7

    def test_custom_values(self):
        """RobotConfig should accept custom values."""
        config = RobotConfig(
            robot_id=2,
            x_min=0.2,
            x_max=1.0,
            y_min=-0.4,
            y_max=0.4,
            joint_velocity_limit=2.0
        )
        
        assert config.robot_id == 2
        assert config.x_min == 0.2
        assert config.joint_velocity_limit == 2.0


class TestTimeFeasibility:
    """Test the time feasibility logic used by InterceptPlanner."""

    def test_feasible_slow_movement(self):
        """Slow movement should be feasible."""
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_end = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        available_time = 1.0
        joint_velocity_limit = 1.5
        
        q_diff = np.abs(q_end - q_start)
        required_velocities = q_diff / available_time
        max_required = np.max(required_velocities)
        is_feasible = max_required <= joint_velocity_limit
        
        assert is_feasible is True
        assert max_required == pytest.approx(0.1)

    def test_infeasible_fast_movement(self):
        """Fast movement should be infeasible."""
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_end = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        available_time = 0.5
        joint_velocity_limit = 1.5
        
        q_diff = np.abs(q_end - q_start)
        required_velocities = q_diff / available_time
        max_required = np.max(required_velocities)
        is_feasible = max_required <= joint_velocity_limit
        
        assert is_feasible is False
        assert max_required == pytest.approx(4.0)

    def test_edge_case_exactly_at_limit(self):
        """Movement exactly at velocity limit should be feasible."""
        q_start = np.array([0.0])
        q_end = np.array([1.5])
        available_time = 1.0
        joint_velocity_limit = 1.5
        
        q_diff = np.abs(q_end - q_start)
        required_velocities = q_diff / available_time
        max_required = np.max(required_velocities)
        is_feasible = max_required <= joint_velocity_limit
        
        assert is_feasible is True


class TestWorkspaceFiltering:
    """Test the workspace filtering logic used by InterceptPlanner."""

    def test_filter_inside_workspace(self):
        """Points inside workspace should be kept."""
        config = RobotConfig(robot_id=1, x_min=-1.0, x_max=-0.2, y_min=-0.5, y_max=0.5)
        
        candidates = [
            (0.1, np.array([-0.5, 0.0, 0.1])),
            (0.2, np.array([-0.8, 0.3, 0.1])),
            (0.3, np.array([-0.3, -0.4, 0.1])),
        ]
        
        reachable = []
        for t, pos in candidates:
            x, y = pos[0], pos[1]
            if (config.x_min <= x <= config.x_max and
                config.y_min <= y <= config.y_max):
                reachable.append((t, pos))
        
        assert len(reachable) == 3

    def test_filter_outside_x_bounds(self):
        """Points outside x bounds should be rejected."""
        config = RobotConfig(robot_id=1, x_min=-1.0, x_max=-0.2)
        
        candidates = [
            (0.1, np.array([-0.5, 0.0, 0.1])),   # Inside
            (0.2, np.array([0.0, 0.0, 0.1])),    # x too high
            (0.3, np.array([-1.5, 0.0, 0.1])),   # x too low
        ]
        
        reachable = []
        for t, pos in candidates:
            x, y = pos[0], pos[1]
            if (config.x_min <= x <= config.x_max and
                config.y_min <= y <= config.y_max):
                reachable.append((t, pos))
        
        assert len(reachable) == 1
        assert reachable[0][0] == 0.1

    def test_filter_outside_y_bounds(self):
        """Points outside y bounds should be rejected."""
        config = RobotConfig(robot_id=1, x_min=-1.0, x_max=-0.2, y_min=-0.5, y_max=0.5)
        
        candidates = [
            (0.1, np.array([-0.5, 0.0, 0.1])),   # Inside
            (0.2, np.array([-0.5, 0.6, 0.1])),   # y too high
            (0.3, np.array([-0.5, -0.7, 0.1])),  # y too low
        ]
        
        reachable = []
        for t, pos in candidates:
            x, y = pos[0], pos[1]
            if (config.x_min <= x <= config.x_max and
                config.y_min <= y <= config.y_max):
                reachable.append((t, pos))
        
        assert len(reachable) == 1


class TestTrajectoryGeneration:
    """Test the trajectory generation logic used by InterceptPlanner."""

    def test_generate_linear_trajectory(self):
        """Trajectory should linearly interpolate between start and end."""
        q_start = np.array([0.0, 0.0])
        q_end = np.array([1.0, 2.0])
        t_start = 0.0
        t_end = 1.0
        dt = 0.5
        
        n_steps = max(2, int((t_end - t_start) / dt) + 1)
        times = np.linspace(t_start, t_end, n_steps)
        alphas = np.linspace(0, 1, n_steps).reshape(-1, 1)
        positions = (1 - alphas) * q_start + alphas * q_end
        
        traj = JointTrajectory(times=times, positions=positions)
        
        assert traj.n_points == 3
        np.testing.assert_array_almost_equal(traj.positions[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(traj.positions[1], [0.5, 1.0])
        np.testing.assert_array_almost_equal(traj.positions[2], [1.0, 2.0])

    def test_trajectory_timing(self):
        """Trajectory times should span from t_start to t_end."""
        q_start = np.array([0.0])
        q_end = np.array([1.0])
        t_start = 2.0
        t_end = 4.0
        dt = 0.5
        
        n_steps = max(2, int((t_end - t_start) / dt) + 1)
        times = np.linspace(t_start, t_end, n_steps)
        
        assert times[0] == pytest.approx(2.0)
        assert times[-1] == pytest.approx(4.0)

    def test_minimum_two_points(self):
        """Trajectory should have at least 2 points even for short durations."""
        t_start = 0.0
        t_end = 0.001  # Very short
        dt = 0.1
        
        n_steps = max(2, int((t_end - t_start) / dt) + 1)
        
        assert n_steps >= 2


class TestInterceptPlannerIntegration:
    """
    Integration tests that require Drake environment.
    These are skipped by default and should be run separately.
    """

    @pytest.mark.skip(reason="Requires Drake environment")
    def test_plan_intercept_with_moving_puck(self):
        """InterceptPlanner should return trajectory for reachable puck."""
        pass

    @pytest.mark.skip(reason="Requires Drake environment")
    def test_plan_intercept_unreachable_puck(self):
        """InterceptPlanner should return None for unreachable puck."""
        pass

    @pytest.mark.skip(reason="Requires Drake environment")
    def test_plan_to_home(self):
        """plan_to_home should return valid trajectory."""
        pass

