import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from dataclasses import dataclass

from .states import PuckState

@dataclass
class TableBounds:
    x_min: float = -1.064
    x_max: float = 1.064
    y_min: float = -0.609
    y_max: float = 0.609
    z: float = 0.101


class PuckPredictor:
    """
    Predicts future puck trajectory using 2D kinematics with wall reflections.
    """

    def __init__(
        self,
        table_bounds: TableBounds | None = None,
        restitution: float = 0.9,
        friction_decel: float = 0.1,
        min_speed: float = 0.01
    ) -> None:
        """
        Args:
            table_bounds: Table geometry
            restitution: Velocity retained after wall bounce (0-1)
            friction_decel: Velocity reduction per second (m/s^2)
            min_speed: Stop prediction when speed falls below this
        """
        self.bounds = table_bounds or TableBounds()
        self.restitution = restitution
        self.friction_decel = friction_decel
        self.min_speed = min_speed

    def predict(
        self,
        puck: PuckState,
        horizon: float,
        dt: float = 0.01
    ) -> List[Tuple[float, NDArray[np.float64]]]:
        """
        Predict puck trajectory over a time horizon.

        Args:
            puck: Current puck state
            horizon: How far ahead to predict
            dt: Time step for prediction samples

        Returns:
            List of (time, position_3d) tuples along predicted path
        """
        trajectory: List[Tuple[float, NDArray[np.float64]]] = []

        position = puck.position_2d.copy()
        velocity = puck.velocity_2d.copy()
        t = puck.t
        z = self.bounds.z

        t_end = t + horizon

        while t < t_end:
            position_3d = np.array([position[0], position[1], z])
            trajectory.append((t, position_3d))

            speed = np.linalg.norm(velocity)
            if speed < self.min_speed:
                break

            if speed > 0:
                friction_factor = max(0, 1 - (self.friction_decel * dt) / speed)
                velocity = velocity * friction_factor

            next_position = position + velocity * dt

            # Check wall bounce
            if next_position[1] <= self.bounds.y_min:
                next_position[1] = 2 * self.bounds.y_min - next_position[1]
                velocity[1] = -velocity[1] * self.restitution
            elif next_position[1] >= self.bounds.y_max:
                next_position[1] = 2 * self.bounds.y_max - next_position[1]
                velocity[1] = -velocity[1] * self.restitution

            # Check goal scored
            if next_position[0] < self.bounds.x_min or next_position[0] > self.bounds.x_max:
                break

            position = next_position
            t += dt

        return trajectory

    def get_position_at_time(
        self,
        puck: PuckState,
        target_time: float
    ) -> NDArray[np.float64]:
        """
        Get predicted puck position at a specific future time.

        Args:
            puck: Current puck state
            target_time: Absolute time to predict position at

        Returns:
            Predicted 3D position, or None if puck leaves table before then
        """
        horizon = target_time - puck.t
        if horizon <= 0:
            # target_time is now or in the past
            return puck.position.copy()

        trajectory = self.predict(puck, horizon, dt=0.005)
        if not trajectory:
            return None

        return trajectory[-1][1]