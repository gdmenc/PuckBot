"""
Shot planning for air hockey - compute desired puck trajectories toward goal.

Features:
- Direct shots toward opponent goal
- 1-bounce bank shots (future)
- Trajectory validation (stays in bounds)
- Shot type selection heuristics
"""
import numpy as np


class ShotPlanner:
    """
    Plans shot trajectories for air hockey.
    
    Computes desired post-impact puck velocity to score goals,
    considering direct paths and potential bank shots.
    """
    
    def __init__(self, table_bounds, goal_width=0.5):
        """
        Args:
            table_bounds: Dict with 'x' and 'y' bounds
                         e.g., {'x': (-1.0, 1.0), 'y': (-0.52, 0.52)}
            goal_width: Width of goal (centered at y=0)
        """
        self.x_min, self.x_max = table_bounds['x']
        self.y_min, self.y_max = table_bounds['y']
        self.goal_width = goal_width
        
    def plan_direct_shot(self, hit_pos, goal_center, speed=2.0):
        """
        Plan direct shot toward goal center.
        
        Args:
            hit_pos: Impact position [x, y]
            goal_center: Goal center position [x, y]
            speed: Desired puck speed after impact (m/s)
        
        Returns:
            v_puck_desired: Desired puck velocity [vx, vy]
        """
        hit_pos = np.asarray(hit_pos, dtype=float)
        goal_center = np.asarray(goal_center, dtype=float)
        
        # Direction vector
        direction = goal_center - hit_pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            # Degenerate case: already at goal
            # Shoot along X axis toward goal
            direction = np.array([np.sign(goal_center[0] - hit_pos[0]), 0.0])
        else:
            direction = direction / distance
        
        # Scale to desired speed
        v_puck_desired = speed * direction
        
        return v_puck_desired
    
    def plan_bounce_shot(self, hit_pos, goal_center, wall_side, speed=2.0):
        """
        Plan 1-bounce bank shot off side wall.
        
        Args:
            hit_pos: Impact position [x, y]
            goal_center: Goal center position [x, y]
            wall_side: Which wall to bounce off ('top' or 'bottom')
            speed: Desired puck speed
        
        Returns:
            v_puck_desired: Desired puck velocity [vx, vy]
        
        Method:
            Reflect goal across wall, then shoot toward reflected goal.
            This ensures puck bounces off wall and heads toward goal.
        """
        hit_pos = np.asarray(hit_pos, dtype=float)
        goal_center = np.asarray(goal_center, dtype=float)
        
        # Reflect goal across wall
        if wall_side == 'top':
            wall_y = self.y_max
            goal_reflected = np.array([goal_center[0], 2*wall_y - goal_center[1]])
        elif wall_side == 'bottom':
            wall_y = self.y_min
            goal_reflected = np.array([goal_center[0], 2*wall_y - goal_center[1]])
        else:
            raise ValueError(f"Invalid wall_side: {wall_side}. Use 'top' or 'bottom'")
        
        # Shoot toward reflected goal
        direction = goal_reflected - hit_pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            # Fallback to direct shot
            return self.plan_direct_shot(hit_pos, goal_center, speed)
        
        direction = direction / distance
        v_puck_desired = speed * direction
        
        return v_puck_desired
    
    def is_valid_direct_shot(self, hit_pos, v_puck, max_time=5.0):
        """
        Check if direct shot stays within table bounds.
        
        Args:
            hit_pos: Starting position [x, y]
            v_puck: Puck velocity [vx, vy]
            max_time: Maximum time to simulate (seconds)
        
        Returns:
            valid: True if shot stays in bounds until reaching goal
        """
        hit_pos = np.asarray(hit_pos, dtype=float)
        v_puck = np.asarray(v_puck, dtype=float)
        
        # Simulate trajectory
        dt = 0.01
        pos = hit_pos.copy()
        
        for _ in range(int(max_time / dt)):
            pos = pos + v_puck * dt
            
            # Check if reached goal
            if abs(pos[0]) >= abs(self.x_max):
                return True  # Reached end (goal)
            
            # Check if out of bounds (Y)
            if pos[1] < self.y_min or pos[1] > self.y_max:
                return False  # Hit side wall (not valid for direct shot check)
        
        return True  # Stayed in bounds
    
    def choose_best_shot(self, hit_pos, goal_center, speed=2.0):
        """
        Choose best shot type (direct or bounce).
        
        Args:
            hit_pos: Impact position [x, y]
            goal_center: Goal center [x, y]
            speed: Desired shot speed
        
        Returns:
            (shot_type, v_puck_desired)
            shot_type: 'direct', 'bounce_top', or 'bounce_bottom'
        """
        # Try direct shot first
        v_direct = self.plan_direct_shot(hit_pos, goal_center, speed)
        
        # For now, always prefer direct (Phase 2A)
        # TODO Phase 2B: Add bounce shot selection
        return ('direct', v_direct)


# ============================================================================
# Unit Tests  
# ============================================================================

def test_shot_planner():
    """Test suite for shot planning."""
    table_bounds = {'x': (-1.0, 1.0), 'y': (-0.52, 0.52)}
    planner = ShotPlanner(table_bounds, goal_width=0.5)
    
    print("Testing Shot Planner...")
    print("=" * 60)
    
    # Test 1: Direct shot from center toward right goal
    print("\nTest 1: Direct shot from center to right goal")
    hit_pos = np.array([0.0, 0.0])
    goal_center = np.array([1.0, 0.0])
    speed = 2.0
    
    v_shot = planner.plan_direct_shot(hit_pos, goal_center, speed)
    print(f"  Hit position:  {hit_pos}")
    print(f"  Goal center:   {goal_center}")
    print(f"  Shot velocity: {v_shot}")
    print(f"  Shot speed:    {np.linalg.norm(v_shot):.3f} m/s")
    
    assert np.abs(np.linalg.norm(v_shot) - speed) < 0.01
    assert v_shot[0] > 0  # Moving toward +X (right goal)
    assert np.abs(v_shot[1]) < 0.01  # Straight ahead
    print("  ✓ PASS")
    
    # Test 2: Angled shot
    print("\nTest 2: Angled shot toward goal corner")
    hit_pos = np.array([-0.5, 0.2])
    goal_center = np.array([1.0, 0.15])
    
    v_shot = planner.plan_direct_shot(hit_pos, goal_center, speed)
    print(f"  Hit position:  {hit_pos}")
    print(f"  Goal center:   {goal_center}")
    print(f"  Shot velocity: {v_shot}")
    
    # Verify direction
    expected_dir = (goal_center - hit_pos) / np.linalg.norm(goal_center - hit_pos)
    actual_dir = v_shot / np.linalg.norm(v_shot)
    direction_error = np.linalg.norm(actual_dir - expected_dir)
    print(f"  Direction error: {direction_error:.6f}")
    
    assert np.abs(np.linalg.norm(v_shot) - speed) < 0.01
    assert direction_error < 0.01
    print("  ✓ PASS")
    
    # Test 3: Bounce shot
    print("\nTest 3: Bounce shot off top wall")
    hit_pos = np.array([0.0, 0.0])
    goal_center = np.array([1.0, 0.0])
    
    v_bounce = planner.plan_bounce_shot(hit_pos, goal_center, 'top', speed)
    print(f"  Hit position:  {hit_pos}")
    print(f"  Goal (actual): {goal_center}")
    print(f"  Bounce velocity: {v_bounce}")
    
    # Should be angled upward (positive vy) to hit top wall
    assert v_bounce[1] > 0
    assert np.abs(np.linalg.norm(v_bounce) - speed) < 0.01
    print("  ✓ PASS")
    
    # Test 4: Shot selection
    print("\nTest 4: Best shot selection")
    hit_pos = np.array([0.2, 0.1])
    goal_center = np.array([1.0, 0.0])
    
    shot_type, v_shot = planner.choose_best_shot(hit_pos, goal_center, speed)
    print(f"  Hit position:   {hit_pos}")
    print(f"  Goal center:    {goal_center}")
    print(f"  Selected type:  {shot_type}")
    print(f"  Shot velocity:  {v_shot}")
    
    assert shot_type in ['direct', 'bounce_top', 'bounce_bottom']
    assert np.abs(np.linalg.norm(v_shot) - speed) < 0.01
    print("  ✓ PASS")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")


if __name__ == "__main__":
    test_shot_planner()
