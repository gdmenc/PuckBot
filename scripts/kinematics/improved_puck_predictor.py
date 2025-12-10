"""
IMPROVED Puck Predictor - Based on robust visualization code

Key improvements:
1. Smaller timestep (0.001s) for accuracy with fast pucks
2. Handles ALL 4 wall bounces (including X-axis)
3. Perfect elastic collisions (restitution=1.0, no friction)
4. Matches actual physics from collision-filtered simulation
"""
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from dataclasses import dataclass

from .states import PuckState

@dataclass
class TableBounds:
    x_min: float = -1.0  # Use play area bounds, not physical bounds
    x_max: float = 1.0
    y_min: float = -0.52
    y_max: float = 0.52
    z: float = 0.11  # Table height


class ImprovedPuckPredictor:
    """
    Predicts future puck trajectory with perfect elastic physics.
    
    This matches the actual simulation physics:
    - Zero friction (collision filtering prevents table contact)
    - Perfect elastic bounces (restitution = 1.0)
    - Bounces on all 4 walls
    - High accuracy timestep
    """

    def __init__(
        self,
        table_bounds: TableBounds | None = None,
        dt: float = 0.001  # 1ms timestep for accuracy
    ) -> None:
        """
        Args:
            table_bounds: Table geometry
            dt: Integration timestep (smaller = more accurate)
        """
        self.bounds = table_bounds or TableBounds()
        self.dt = dt

    def predict(
        self,
        puck: PuckState,
        horizon: float,
        dt: float | None = None
    ) -> List[Tuple[float, NDArray[np.float64]]]:
        """
        Predict puck trajectory over a time horizon.

        Args:
            puck: Current puck state
            horizon: How far ahead to predict (seconds)
            dt: Override default timestep if needed

        Returns:
            List of (time, position_3d) tuples along predicted path
        """
        if dt is None:
            dt = self.dt
            
        trajectory: List[Tuple[float, NDArray[np.float64]]] = []

        # Work in 2D
        pos = puck.position_2d.copy().astype(float)
        vel = puck.velocity_2d.copy().astype(float)
        t = puck.t
        z = self.bounds.z

        t_end = t + horizon

        while t < t_end:
            # Record current state
            position_3d = np.array([pos[0], pos[1], z])
            trajectory.append((t, position_3d))

            # Predict next position
            next_pos = pos + vel * dt

            # Check X-axis bounces (end walls)
            if next_pos[0] < self.bounds.x_min:
                vel[0] = abs(vel[0])  # Bounce right
                pos[0] = self.bounds.x_min
            elif next_pos[0] > self.bounds.x_max:
                vel[0] = -abs(vel[0])  # Bounce left
                pos[0] = self.bounds.x_max

            # Check Y-axis bounces (side walls)
            if next_pos[1] < self.bounds.y_min:
                vel[1] = abs(vel[1])  # Bounce up
                pos[1] = self.bounds.y_min
            elif next_pos[1] > self.bounds.y_max:
                vel[1] = -abs(vel[1])  # Bounce down
                pos[1] = self.bounds.y_max

            # Integrate position with boundary clamping
            pos = pos + vel * dt
            pos[0] = np.clip(pos[0], self.bounds.x_min, self.bounds.x_max)
            pos[1] = np.clip(pos[1], self.bounds.y_min, self.bounds.y_max)

            t += dt

        return trajectory

    def get_position_at_time(
        self,
        puck: PuckState,
        target_time: float
    ) -> NDArray[np.float64] | None:
        """
        Get predicted puck position at a specific future time.

        Args:
            puck: Current puck state
            target_time: Absolute time to predict position at

        Returns:
            Predicted 3D position
        """
        horizon = target_time - puck.t
        if horizon <= 0:
            # Target time is now or in the past
            return puck.position.copy()

        trajectory = self.predict(puck, horizon)
        if not trajectory:
            return None

        return trajectory[-1][1]
    
    def predict_next_intercept_in_zone(
        self,
        puck: PuckState,
        x_min: float,
        x_max: float,
        y_min: float = -0.5,
        y_max: float = 0.5,
        max_horizon: float = 3.0
    ) -> Tuple[float, NDArray[np.float64]] | None:
        """
        Find the next time puck enters a zone (e.g., robot workspace).
        
        Args:
            puck: Current puck state
            x_min, x_max, y_min, y_max: Zone boundaries
            max_horizon: Maximum lookahead time
            
        Returns:
            (time, position) of first entry into zone, or None
        """
        trajectory = self.predict(puck, max_horizon)
        
        for t, pos in trajectory:
            x, y = pos[0], pos[1]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return (t, pos)
        
        return None
