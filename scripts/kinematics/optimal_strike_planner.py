"""
Optimal Strike Planner - Predictive striking algorithm for air hockey.

Key features:
1. Samples multiple future puck positions with fine granularity
2. Evaluates intercept options based on:
   - Angle to goal
   - Robot reachability
   - Time to reach
   - Shot speed potential
3. For stationary pucks: uses curved approach paths to optimize strike angle
4. Always aims toward opponent goal with maximum velocity
"""
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

try:
    from scripts.kinematics.puck_predictor import PuckPredictor, TableBounds
    from scripts.kinematics.improved_puck_predictor import ImprovedPuckPredictor
    from scripts.kinematics.states import PuckState
except ModuleNotFoundError:
    from puck_predictor import PuckPredictor, TableBounds
    from improved_puck_predictor import ImprovedPuckPredictor
    from states import PuckState


@dataclass
class InterceptCandidate:
    """Represents a potential intercept point."""
    position: np.ndarray  # 3D position
    time: float  # Time to intercept
    puck_velocity: np.ndarray  # Puck velocity at this point
    score: float  # Quality score (higher is better)
    approach_angle: float  # Angle to approach from (radians)
    

class OptimalStrikePlanner:
    """
    Plans optimal strikes by sampling future trajectories and selecting
    the best intercept point based on goal angle and feasibility.
    """
    
    def __init__(
        self,
        table_bounds: Optional[dict] = None,
        puck_radius: float = 0.04,
        paddle_radius: float = 0.05,
        max_robot_speed: float = 1.5,  # m/s end-effector speed
        desired_shot_speed: float = 3.5  # m/s shot speed - INCREASED from 2.5 for more power
    ):
        """
        Args:
            table_bounds: {'x': (min, max), 'y': (min, max)}
            puck_radius: Puck radius in meters
            paddle_radius: Paddle radius in meters
            max_robot_speed: Maximum end-effector speed
            desired_shot_speed: Desired puck speed after hit
        """
        if table_bounds is None:
            table_bounds = {'x': (-1.0, 1.0), 'y': (-0.52, 0.52)}
        self.table_bounds = table_bounds
        self.puck_radius = puck_radius
        self.paddle_radius = paddle_radius
        self.max_robot_speed = max_robot_speed
        self.desired_shot_speed = desired_shot_speed
        
        # Create puck predictor
        self.predictor = ImprovedPuckPredictor()
        
    def get_goal_position(self, robot_side: float) -> np.ndarray:
        """
        Get opponent goal position.
        
        Args:
            robot_side: 1.0 for right robot, -1.0 for left robot
            
        Returns:
            [x, y] goal position
        """
        if robot_side > 0:
            return np.array([self.table_bounds['x'][0], 0.0])  # Left goal
        else:
            return np.array([self.table_bounds['x'][1], 0.0])  # Right goal
    
    def evaluate_intercept_quality(
        self,
        intercept_pos: np.ndarray,
        puck_vel: np.ndarray,
        robot_pos: np.ndarray,
        robot_side: float,
        time_available: float
    ) -> float:
        """
        Evaluate how good an intercept point is.
        
        Scoring criteria:
        1. Angle to goal (prefer straight shots)
        2. Robot reachability (prefer closer points)
        3. Puck speed (prefer slower for better control)
        4. Time available (prefer more time to set up)
        
        Returns:
            Quality score (0-100, higher is better)
        """
        goal_pos = self.get_goal_position(robot_side)
        
        # 1. Angle to goal score (0-40 points)
        # Best: straight line from intercept to goal
        puck_to_goal = goal_pos - intercept_pos[:2]
        puck_to_goal_norm = np.linalg.norm(puck_to_goal)
        
        if puck_to_goal_norm > 0.01:
            ideal_shot_dir = puck_to_goal / puck_to_goal_norm
            
            # For moving pucks, check if we can redirect velocity toward goal
            puck_speed = np.linalg.norm(puck_vel[:2])
            if puck_speed > 0.1:
                current_dir = puck_vel[:2] / puck_speed
                # Angle difference (smaller is better)
                angle_diff = np.arccos(np.clip(np.dot(current_dir, ideal_shot_dir), -1, 1))
                angle_score = 40 * (1.0 - angle_diff / np.pi)
            else:
                # Stationary puck - we have full control over direction
                angle_score = 40.0
        else:
            angle_score = 0.0
        
        # 2. Reachability score (0-30 points)
        # Closer is better, but penalize being too close (need time to accelerate)
        dist_to_intercept = np.linalg.norm(intercept_pos[:2] - robot_pos[:2])
        optimal_dist = 0.3  # Optimal distance in meters
        dist_score = 30 * np.exp(-((dist_to_intercept - optimal_dist) ** 2) / 0.2)
        
        # 3. Puck speed score (0-15 points)
        # Slower pucks are easier to control
        puck_speed = np.linalg.norm(puck_vel[:2])
        speed_score = 15 * np.exp(-puck_speed / 2.0)
        
        # 4. Time score (0-15 points)
        # More time is better (up to 0.5s)
        time_score = 15 * min(1.0, time_available / 0.5)
        
        total_score = angle_score + dist_score + speed_score + time_score
        return total_score
    
    def sample_future_trajectory(
        self,
        puck_state: PuckState,
        horizon: float = 3.5,  # INCREASED from 2.0 to 3.5 for more intercept options
        n_samples: int = 70  # INCREASED from 50 to 70 for finer resolution
    ) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Sample puck positions along predicted trajectory.
        
        Returns:
            List of (time, position, velocity) tuples
        """
        trajectory = self.predictor.predict(puck_state, horizon, dt=0.01)
        
        if len(trajectory) < 2:
            return []
        
        # Sample evenly along trajectory
        step = max(1, len(trajectory) // n_samples)
        sampled = []
        
        prev_pos = trajectory[0][1]
        prev_t = trajectory[0][0]
        
        for i in range(0, len(trajectory), step):
            t, pos = trajectory[i]
            
            # Estimate velocity from position change
            if i > 0:
                dt = t - prev_t
                if dt > 0:
                    vel = (pos - prev_pos) / dt
                else:
                    vel = puck_state.velocity
            else:
                vel = puck_state.velocity
            
            sampled.append((t, pos, vel))
            prev_pos = pos
            prev_t = t
        
        return sampled
    
    def compute_curved_approach(
        self,
        puck_pos: np.ndarray,
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
        arc_distance: float = 0.2
    ) -> Tuple[np.ndarray, float]:
        """
        Compute curved approach path for stationary puck.
        
        Creates an arc that allows the robot to strike the puck
        toward the goal from an optimal angle.
        
        Args:
            puck_pos: Puck position [x, y, z]
            robot_pos: Current robot end-effector position [x, y, z]
            goal_pos: Goal position [x, y]
            arc_distance: Distance to position robot from puck
            
        Returns:
            (approach_position, approach_angle) - where to position robot
        """
        # Direction from puck to goal
        puck_to_goal = goal_pos - puck_pos[:2]
        puck_to_goal_dist = np.linalg.norm(puck_to_goal)
        
        if puck_to_goal_dist < 0.01:
            # Fallback: approach from behind
            approach_dir = np.array([1.0, 0.0]) if robot_pos[0] > puck_pos[0] else np.array([-1.0, 0.0])
        else:
            # Approach from opposite direction of goal
            # This way, robot will strike THROUGH the puck TOWARD the goal
            approach_dir = -puck_to_goal / puck_to_goal_dist
        
        # Position robot offset from puck along approach direction
        # This is where robot needs to be positioned
        offset_distance = self.puck_radius + self.paddle_radius + arc_distance
        approach_pos_2d = puck_pos[:2] + approach_dir * offset_distance
        approach_pos_3d = np.array([approach_pos_2d[0], approach_pos_2d[1], puck_pos[2]])
        
        # Calculate approach angle (for later velocity direction)
        approach_angle = np.arctan2(-approach_dir[1], -approach_dir[0])
        
        return approach_pos_3d, approach_angle
    
    def plan_optimal_strike(
        self,
        puck_state: PuckState,
        robot_pos: np.ndarray,
        robot_side: float,
        workspace_bounds: Optional[dict] = None
    ) -> Optional[InterceptCandidate]:
        """
        Plan optimal strike by sampling and evaluating intercept candidates.
        
        Args:
            puck_state: Current puck state
            robot_pos: Current robot end-effector position [x, y, z]
            robot_side: 1.0 for right robot, -1.0 for left robot
            workspace_bounds: Robot workspace {'x': (min, max), 'y': (min, max)}
            
        Returns:
            Best InterceptCandidate or None if no good option
        """
        if workspace_bounds is None:
            if robot_side > 0:
                workspace_bounds = {'x': (0.0, 0.9), 'y': (-0.5, 0.5)}
            else:
                workspace_bounds = {'x': (-0.9, 0.0), 'y': (-0.5, 0.5)}
        
        goal_pos = self.get_goal_position(robot_side)
        puck_speed = np.linalg.norm(puck_state.velocity[:2])
        
        # Handle stationary puck differently
        if puck_speed < 0.05:
            # Stationary puck - use curved approach
            approach_pos, approach_angle = self.compute_curved_approach(
                puck_state.position,
                robot_pos,
                goal_pos,
                arc_distance=0.15
            )
            
            # Check if approach position is reachable
            if not (workspace_bounds['x'][0] <= approach_pos[0] <= workspace_bounds['x'][1] and
                    workspace_bounds['y'][0] <= approach_pos[1] <= workspace_bounds['y'][1]):
                # Try shorter arc
                approach_pos, approach_angle = self.compute_curved_approach(
                    puck_state.position,
                    robot_pos,
                    goal_pos,
                    arc_distance=0.08
                )
            
            # Time to reach approach position
            dist_to_approach = np.linalg.norm(approach_pos[:2] - robot_pos[:2])
            time_to_reach = dist_to_approach / (self.max_robot_speed * 0.7)  # Conservative
            
            # Desired strike velocity (toward goal)
            strike_dir = (goal_pos - puck_state.position[:2])
            strike_dir_norm = np.linalg.norm(strike_dir)
            if strike_dir_norm > 0.01:
                strike_dir = strike_dir / strike_dir_norm
            else:
                strike_dir = np.array([-1.0, 0.0]) if robot_side > 0 else np.array([1.0, 0.0])
            
            strike_velocity = strike_dir * self.desired_shot_speed
            
            return InterceptCandidate(
                position=approach_pos,
                time=puck_state.t + time_to_reach,
                puck_velocity=strike_velocity,  # Desired post-impact velocity
                score=100.0,  # Stationary pucks get perfect score
                approach_angle=approach_angle
            )
        
        # Moving puck - sample trajectory
        samples = self.sample_future_trajectory(puck_state, horizon=2.0, n_samples=50)
        
        if not samples:
            return None
        
        candidates = []
        
        for t, pos, vel in samples:
            time_available = t - puck_state.t
            
            # Check workspace bounds
            if not (workspace_bounds['x'][0] <= pos[0] <= workspace_bounds['x'][1] and
                    workspace_bounds['y'][0] <= pos[1] <= workspace_bounds['y'][1]):
                continue
            
            # Check if robot can reach in time
            dist_to_intercept = np.linalg.norm(pos[:2] - robot_pos[:2])
            min_time_needed = dist_to_intercept / self.max_robot_speed
            
            if min_time_needed > time_available:
                continue  # Can't reach in time
            
            # Evaluate this intercept
            score = self.evaluate_intercept_quality(
                pos, vel, robot_pos, robot_side, time_available
            )
            
            # Compute approach angle (opposite to puck velocity for best control)
            puck_speed_here = np.linalg.norm(vel[:2])
            if puck_speed_here > 0.1:
                approach_angle = np.arctan2(-vel[1], -vel[0])
            else:
                # Use goal direction
                to_goal = goal_pos - pos[:2]
                approach_angle = np.arctan2(-to_goal[1], -to_goal[0])
            
            candidates.append(InterceptCandidate(
                position=pos,
                time=t,
                puck_velocity=vel,
                score=score,
                approach_angle=approach_angle
            ))
        
        if not candidates:
            return None
        
        # Select best candidate
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[0]
    
    def compute_strike_target_position(
        self,
        intercept: InterceptCandidate,
        goal_pos: np.ndarray,
        follow_through: float = 0.25
    ) -> np.ndarray:
        """
        Compute where robot should target to strike through the intercept point.
        
        Args:
            intercept: Intercept candidate
            goal_pos: Goal position [x, y]
            follow_through: Distance past intercept to target
            
        Returns:
            Target position [x, y, z] for robot to aim for
        """
        # Direction from intercept to goal
        to_goal = goal_pos - intercept.position[:2]
        to_goal_norm = np.linalg.norm(to_goal)
        
        if to_goal_norm < 0.01:
            strike_dir = np.array([-1.0, 0.0])  # Fallback
        else:
            strike_dir = to_goal / to_goal_norm
        
        # Target position is PAST the intercept point
        # Robot will accelerate THROUGH the puck
        target_pos_2d = intercept.position[:2] + strike_dir * follow_through
        target_pos_3d = np.array([target_pos_2d[0], target_pos_2d[1], intercept.position[2]])
        
        return target_pos_3d
    
    def compute_required_velocity(
        self,
        intercept: InterceptCandidate,
        goal_pos: np.ndarray
    ) -> np.ndarray:
        """
        Compute required end-effector velocity for strike.
        
        Args:
            intercept: Intercept candidate
            goal_pos: Goal position [x, y]
            
        Returns:
            Required velocity vector [vx, vy] in m/s
        """
        # Direction toward goal
        to_goal = goal_pos - intercept.position[:2]
        to_goal_norm = np.linalg.norm(to_goal)
        
        if to_goal_norm < 0.01:
            strike_dir = np.array([-1.0, 0.0])
        else:
            strike_dir = to_goal / to_goal_norm
        
        # Velocity magnitude based on desired shot speed
        # Using momentum conservation: m_puck * v_puck = (m_puck + m_paddle) * v_final
        # For simplified model: v_paddle ≈ 1.5 * v_desired_puck
        required_speed = min(self.desired_shot_speed * 1.5, self.max_robot_speed)
        
        return strike_dir * required_speed


# ============================================================================
# Unit Tests
# ============================================================================

def test_optimal_strike_planner():
    """Test optimal strike planner with various scenarios."""
    print("Testing Optimal Strike Planner...")
    print("=" * 60)
    
    planner = OptimalStrikePlanner()
    
    # Test 1: Stationary puck
    print("\nTest 1: Stationary puck")
    puck_state = PuckState(
        position=np.array([0.3, 0.1, 0.11]),
        velocity=np.array([0.0, 0.0, 0.0]),
        t=0.0
    )
    robot_pos = np.array([0.6, 0.2, 0.11])
    robot_side = 1.0  # Right robot
    
    intercept = planner.plan_optimal_strike(puck_state, robot_pos, robot_side)
    
    if intercept:
        print(f"  Intercept position: {intercept.position[:2]}")
        print(f"  Approach angle: {np.degrees(intercept.approach_angle):.1f}°")
        print(f"  Score: {intercept.score:.1f}")
        
        goal = planner.get_goal_position(robot_side)
        target = planner.compute_strike_target_position(intercept, goal)
        print(f"  Strike target: {target[:2]}")
        
        velocity = planner.compute_required_velocity(intercept, goal)
        print(f"  Required velocity: {velocity} ({np.linalg.norm(velocity):.2f} m/s)")
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL - No intercept found")
    
    # Test 2: Moving puck
    print("\nTest 2: Moving puck")
    puck_state = PuckState(
        position=np.array([0.2, 0.0, 0.11]),
        velocity=np.array([0.8, 0.2, 0.0]),
        t=0.0
    )
    robot_pos = np.array([0.5, 0.1, 0.11])
    
    intercept = planner.plan_optimal_strike(puck_state, robot_pos, robot_side)
    
    if intercept:
        print(f"  Intercept position: {intercept.position[:2]}")
        print(f"  Intercept time: {intercept.time:.3f}s")
        print(f"  Score: {intercept.score:.1f}")
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL - No intercept found")
    
    print("\n" + "=" * 60)
    print("✅ TESTS COMPLETE")


if __name__ == "__main__":
    test_optimal_strike_planner()
