"""
Workspace safety constraints for preventing robot collisions with table walls.
"""
import numpy as np


class WorkspaceSafety:
    """
    Enforces workspace constraints to prevent robot from hitting table walls.
    
    Safety zones:
    - Outer table walls at ±0.609m (Y) and ±1.064m (X)  
    - Safety margin added to prevent collisions
    - Separate zones for left/right robots
    """
    
    def __init__(self, robot_side: float, table_bounds: dict = None):
        """
        Args:
            robot_side: 1.0 for right robot, -1.0 for left robot
            table_bounds: {'x': (min, max), 'y': (min, max)}
        """
        if table_bounds is None:
            table_bounds = {
                'x': (-1.064, 1.064),  # Table X boundaries
                'y': (-0.609, 0.609)    # Table Y boundaries
            }
        
        self.table_bounds = table_bounds
        self.robot_side = robot_side
        
        # Safety margins to prevent wall collisions
        # INCREASED margins to prevent getting stuck on outer rim
        # Paddle radius ~0.0475m + robot arm length + uncertainty
        self.x_margin = 0.20  # 20cm safety margin from X boundaries (was 15cm)
        self.y_margin = 0.18  # 18cm safety margin from Y walls (was 12cm)
        
        # Extra margin for corners (diagonal movements need more space)
        self.corner_margin = 0.25  # 25cm from corners
        
        # Define safe workspace based on robot side
        if robot_side > 0:  # Right robot
            # Right robot workspace: stay on right side with margins
            self.x_min = 0.05  # Don't cross center line
            self.x_max = table_bounds['x'][1] - self.x_margin  # Stay away from right wall
        else:  # Left robot
            # Left robot workspace: stay on left side with margins  
            self.x_min = table_bounds['x'][0] + self.x_margin  # Stay away from left wall
            self.x_max = -0.05  # Don't cross center line
        
        # Y limits are same for both robots
        self.y_min = table_bounds['y'][0] + self.y_margin  # Stay away from bottom wall
        self.y_max = table_bounds['y'][1] - self.y_margin  # Stay away from top wall
        
    def clamp_to_workspace(self, position: np.ndarray) -> np.ndarray:
        """
        Clamp a 2D or 3D position to safe workspace.
        
        ENHANCED: Applies stricter limits near corners to prevent getting stuck.
        
        Args:
            position: [x, y] or [x, y, z]
            
        Returns:
            Clamped position (same shape as input)
        """
        pos = position.copy()
        
        # Standard clamping
        pos[0] = np.clip(pos[0], self.x_min, self.x_max)
        pos[1] = np.clip(pos[1], self.y_min, self.y_max)
        
        # CORNER DETECTION: Apply stricter limits near corners
        # Check distance to corners and pull inward if too close
        x_near_edge = (abs(pos[0] - self.x_min) < self.corner_margin or
                       abs(pos[0] - self.x_max) < self.corner_margin)
        y_near_edge = (abs(pos[1] - self.y_min) < self.corner_margin or
                       abs(pos[1] - self.y_max) < self.corner_margin)
        
        if x_near_edge and y_near_edge:
            # In corner zone - apply extra margin
            # Pull toward center
            if self.robot_side > 0:  # Right robot
                if pos[0] > (self.x_max - self.corner_margin):
                    pos[0] = self.x_max - self.corner_margin
            else:  # Left robot
                if pos[0] < (self.x_min + self.corner_margin):
                    pos[0] = self.x_min + self.corner_margin
            
            # Y direction - pull from edges
            if pos[1] > (self.y_max - self.corner_margin):
                pos[1] = self.y_max - self.corner_margin
            elif pos[1] < (self.y_min + self.corner_margin):
                pos[1] = self.y_min + self.corner_margin
        
        return pos
    
    def is_in_workspace(self, position: np.ndarray) -> bool:
        """
        Check if position is within safe workspace.
        
        Args:
            position: [x, y] or [x, y, z]
            
        Returns:
            True if position is safe
        """
        x, y = position[0], position[1]
        return (self.x_min <= x <= self.x_max and 
                self.y_min <= y <= self.y_max)
    
    def distance_to_boundary(self, position: np.ndarray) -> float:
        """
        Compute minimum distance from position to workspace boundary.
        
        Useful for adaptive control (slow down near walls).
        
        Args:
            position: [x, y] or [x, y, z]
            
        Returns:
            Minimum distance to any wall in meters
        """
        x, y = position[0], position[1]
        
        dist_x_min = abs(x - self.x_min)
        dist_x_max = abs(x - self.x_max)
        dist_y_min = abs(y - self.y_min)
        dist_y_max = abs(y - self.y_max)
        
        return min(dist_x_min, dist_x_max, dist_y_min, dist_y_max)
    
    def get_workspace_bounds(self) -> dict:
        """
        Get workspace bounds as dictionary.
        
        Returns:
            {'x': (min, max), 'y': (min, max)}
        """
        return {
            'x': (self.x_min, self.x_max),
            'y': (self.y_min, self.y_max)
        }
    
    def warn_if_close_to_wall(self, position: np.ndarray, threshold: float = 0.05) -> bool:
        """
        Check if robot is dangerously close to wall.
        
        Args:
            position: [x, y] or [x, y, z]
            threshold: Distance threshold for warning (meters)
            
        Returns:
            True if warning should be issued
        """
        dist = self.distance_to_boundary(position)
        if dist < threshold:
            print(f"[WARNING] Robot close to wall! Distance: {dist:.3f}m at pos [{position[0]:.3f}, {position[1]:.3f}]")
            return True
        return False


# Unit tests
if __name__ == "__main__":
    print("Testing WorkspaceSafety...")
    print("=" * 60)
    
    # Test right robot
    print("\nTest 1: Right robot workspace")
    safety_right = WorkspaceSafety(robot_side=1.0)
    bounds = safety_right.get_workspace_bounds()
    print(f"  X range: {bounds['x']}")
    print(f"  Y range: {bounds['y']}")
    
    # Test clamping
    test_pos = np.array([1.0, 0.6, 0.11])  # Too far right and top
    clamped = safety_right.clamp_to_workspace(test_pos)
    print(f"  Clamped [{test_pos[0]:.3f}, {test_pos[1]:.3f}] -> [{clamped[0]:.3f}, {clamped[1]:.3f}]")
    
    assert clamped[0] < 1.0, "Should clamp X"
    assert clamped[1] < 0.6, "Should clamp Y"
    print("  ✓ PASS")
    
    # Test left robot
    print("\nTest 2: Left robot workspace")
    safety_left = WorkspaceSafety(robot_side=-1.0)
    bounds = safety_left.get_workspace_bounds()
    print(f"  X range: {bounds['x']}")
    print(f"  Y range: {bounds['y']}")
    
    # Test distance to boundary
    safe_pos = np.array([0.5, 0.0, 0.11])
    dist = safety_right.distance_to_boundary(safe_pos)
    print(f"\nTest 3: Distance to boundary at [{safe_pos[0]:.3f}, {safe_pos[1]:.3f}]: {dist:.3f}m")
    print("  ✓ PASS")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
