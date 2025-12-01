from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class PuckState:
    """
    Represents the current state of the puck.

    Fields:
        position: 3D position [x, y, z] in meters (world frame)
        velocity: 3D velocity [dx, dy, dz] in m/s (world frame)
        t: Timestamp in seconds (simulation time)
    """
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    t: float

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)

    @property
    def position_2d(self) -> NDArray[np.float64]:
        """
        Position projected in 2D: [x, y]
        """
        return self.position[:2]

    @property
    def velocity_2d(self) -> NDArray[np.float64]:
        """
        Velocity projected in 2D: [dx, dy]
        """
        return self.velocity[:2]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity_2d))


@dataclass
class RobotState:
    """
    Represents the current state of the robot arm.

    Attributes:
        q: Joint positions in radians
        dq: Joint velocities in radians/sec
        t: Timestamp in seconds (simulation time)
    """
    q: NDArray[np.float64]
    dq: NDArray[np.float64]
    t: float

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=np.float64)
        self.dq = np.asarray(self.dq, dtype=np.float64)

    @property
    def n_joints(self) -> int:
        return len(self.q)


@dataclass
class JointTrajectory:
    """
    Attributes:
        times: 1D array of timestamps
        positions: 2D array of joint positions
    """
    times: NDArray[np.float64]
    positions: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    @property
    def n_points(self) -> int:
        return len(self.times)

    def at_time(self, t: float) -> NDArray[np.float64]:
        """
        Interpolates the joint positions at time t.
        """
        if t <= self.times[0]:
            return self.positions[0]
        if t >= self.times[-1]:
            return self.positions[-1]

        index = np.searchsorted(self.times, t) - 1
        t0, t1 = self.times[index], self.times[index + 1]
        alpha = (t - t0) / (t1 - t0)
        return (1 - alpha) * self.positions[index] + alpha * self.positions[index + 1]