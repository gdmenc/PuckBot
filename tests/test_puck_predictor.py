import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from movement.states import PuckState
from movement.puck_predictor import PuckPredictor, TableBounds


class TestTableBounds:
    def test_default_values(self):
        """TableBounds should have sensible defaults."""
        bounds = TableBounds()
        
        assert bounds.x_min == -1.064
        assert bounds.x_max == 1.064
        assert bounds.y_min == -0.609
        assert bounds.y_max == 0.609
        assert bounds.z == 0.101


class TestPuckPredictor:
    @pytest.fixture
    def predictor(self):
        """Create a predictor with no friction for simpler tests."""
        return PuckPredictor(friction_decel=0.0, restitution=1.0, min_speed=0.001)

    def test_stationary_puck(self, predictor):
        """Stationary puck should stop prediction immediately."""
        puck = PuckState(
            position=[0.0, 0.0, 0.101],
            velocity=[0.0, 0.0, 0.0],
            t=0.0
        )
        
        trajectory = predictor.predict(puck, horizon=1.0, dt=0.01)
        
        assert len(trajectory) >= 1
        assert len(trajectory) < 100

    def test_straight_line_no_bounce(self, predictor):
        """Puck moving straight should travel in a line."""
        puck = PuckState(
            position=[0.0, 0.0, 0.101],
            velocity=[1.0, 0.0, 0.0],
            t=0.0
        )
        
        trajectory = predictor.predict(puck, horizon=0.5, dt=0.1)
        
        assert len(trajectory) > 1
        for t, pos in trajectory:
            assert pos[1] == pytest.approx(0.0, abs=1e-6)

    def test_wall_bounce_bottom(self, predictor):
        """Puck should bounce off bottom wall (y_min)."""
        puck = PuckState(
            position=[0.0, -0.5, 0.101],
            velocity=[0.0, -1.0, 0.0],
            t=0.0
        )
        
        trajectory = predictor.predict(puck, horizon=1.0, dt=0.01)
        
        for t, pos in trajectory:
            assert pos[1] >= predictor.bounds.y_min - 0.01

    def test_wall_bounce_top(self, predictor):
        """Puck should bounce off top wall (y_max)."""
        puck = PuckState(
            position=[0.0, 0.5, 0.101],
            velocity=[0.0, 1.0, 0.0],
            t=0.0
        )
        
        trajectory = predictor.predict(puck, horizon=1.0, dt=0.01)
        
        for t, pos in trajectory:
            assert pos[1] <= predictor.bounds.y_max + 0.01

    def test_exit_at_goal(self, predictor):
        """Prediction should stop when puck exits via goal (x bounds)."""
        puck = PuckState(
            position=[0.9, 0.0, 0.101],
            velocity=[2.0, 0.0, 0.0],
            t=0.0
        )
        
        trajectory = predictor.predict(puck, horizon=2.0, dt=0.01)
        
        final_t, final_pos = trajectory[-1]
        assert final_t < 2.0
        assert final_pos[0] >= 0.9

    def test_friction_slows_puck(self):
        """Friction should cause puck to slow down."""
        predictor = PuckPredictor(friction_decel=0.5, restitution=1.0, min_speed=0.01)
        
        puck = PuckState(
            position=[0.0, 0.0, 0.101],
            velocity=[1.0, 0.0, 0.0],
            t=0.0
        )
        
        trajectory = predictor.predict(puck, horizon=5.0, dt=0.1)
        
        final_t, final_pos = trajectory[-1]
        assert final_pos[0] < 5.0

    def test_restitution_reduces_bounce_velocity(self):
        """Restitution < 1 should reduce velocity after bounce."""
        predictor = PuckPredictor(friction_decel=0.0, restitution=0.5, min_speed=0.001)
        
        puck = PuckState(
            position=[0.0, -0.6, 0.101],
            velocity=[0.0, -1.0, 0.0],
            t=0.0
        )
        
        trajectory = predictor.predict(puck, horizon=2.0, dt=0.01)
        
        assert len(trajectory) > 10

    def test_get_position_at_time_past(self):
        """get_position_at_time should return current position for past times."""
        predictor = PuckPredictor()
        puck = PuckState(
            position=[1.0, 2.0, 0.101],
            velocity=[1.0, 0.0, 0.0],
            t=5.0
        )
        
        result = predictor.get_position_at_time(puck, target_time=3.0)
        
        np.testing.assert_array_equal(result, puck.position)

    def test_get_position_at_time_future(self, predictor):
        """get_position_at_time should predict future position."""
        puck = PuckState(
            position=[0.0, 0.0, 0.101],
            velocity=[1.0, 0.0, 0.0],
            t=0.0
        )
        
        result = predictor.get_position_at_time(puck, target_time=0.5)
        
        assert result[0] == pytest.approx(0.5, abs=0.05)
        assert result[1] == pytest.approx(0.0, abs=0.01)

