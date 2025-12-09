"""
Game mode configurations for air hockey training.

Inspired by air_hockey_challenge task structure:
- defend: Train robot to block approaching puck
- hit: Train robot to strike stationary puck
- tournament: Full competitive play
"""
import numpy as np

class GameMode:
    """Game mode constants"""
    DEFEND = "defend"
    HIT = "hit"
    TOURNAMENT = "tournament"
    
    @staticmethod
    def all_modes():
        return [GameMode.DEFEND, GameMode.HIT, GameMode.TOURNAMENT]


class PuckInitializer:
    """
    Handles puck initialization for different game modes.
    
    Modes:
    - DEFEND: Puck spawns on opponent's side, always moving toward robot
    - HIT: Puck spawns stationary on robot's side for striking practice
    - TOURNAMENT: Random center spawn with random velocity (competitive play)
    """
    
    def __init__(self, mode, robot_side="right", table_bounds=None):
        """
        Args:
            mode: One of GameMode constants
            robot_side: "right" or "left" (which side is the training robot on)
            table_bounds: Dict with 'x' and 'y' limits
        """
        if mode not in GameMode.all_modes():
            raise ValueError(f"Invalid mode: {mode}. Must be one of {GameMode.all_modes()}")
        
        self.mode = mode
        self.robot_side = robot_side
        
        # Default table bounds
        if table_bounds is None:
            table_bounds = {
                'x': (-1.0, 1.0),
                'y': (-0.52, 0.52)
            }
        self.table_bounds = table_bounds
        
        # Constants
        self.table_height = 0.11
        self.puck_radius = 0.03165
        
    def get_initial_state(self):
        """
        Get initial puck position and velocity based on mode.
        
        Returns:
            tuple: (position, velocity) as numpy arrays
                position: [x, y, z]
                velocity: [vx, vy, vz]
        """
        if self.mode == GameMode.DEFEND:
            return self._defend_init()
        elif self.mode == GameMode.HIT:
            return self._hit_init()
        else:  # TOURNAMENT
            return self._tournament_init()
    
    def _defend_init(self):
        """
        Defend mode: Puck spawns on opponent's side, moving toward robot.
        
        Right robot is at +X (~0.4), so puck spawns at -X and moves with +vx
        Left robot is at -X (~-0.4), so puck spawns at +X and moves with -vx
        
        Returns:
            tuple: (position, velocity)
        """
        # Spawn on opponent's side (away from robot)
        if self.robot_side == "right":
            # Robot on right (+X), spawn puck on left (negative X)
            x = np.random.uniform(-0.75, -0.25)  # Opponent's side (left)
            sign = 1  # Move toward positive X (toward robot on right)
        else:  # left
            # Robot on left (-X), spawn puck on right (positive X)
            x = np.random.uniform(0.25, 0.75)  # Opponent's side (right)
            sign = -1  # Move toward negative X (toward robot on left)
        
        y = np.random.uniform(-0.4, 0.4)
        
        # Velocity: Always toward robot's goal
        speed = np.random.uniform(0.4, 1.0)  # m/s
        angle = np.random.uniform(-0.75, 0.75)  # radians (±43 degrees)
        
        vx = sign * np.cos(angle) * speed
        vy = np.sin(angle) * speed
        
        position = np.array([x, y, self.table_height])
        velocity = np.array([vx, vy, 0.0])
        
        print(f"[DEFEND MODE] Puck spawned at [{x:.3f}, {y:.3f}] moving toward robot (vx={vx:.3f})")
        
        return position, velocity
    
    def _hit_init(self):
        """
        Hit mode: Puck spawns stationary on robot's side for striking practice.
        
        Right robot is at +X (~0.4), so puck spawns at +X nearby
        Left robot is at -X (~-0.4), so puck spawns at -X nearby
        
        Returns:
            tuple: (position, velocity)
        """
        # Spawn on robot's side (stationary target)
        if self.robot_side == "right":
            # Robot on right (+X), spawn puck nearby on right side
            x = np.random.uniform(0.2, 0.7)
        else:  # left
            # Robot on left (-X), spawn puck nearby on left side
            x = np.random.uniform(-0.7, -0.2)
        
        y = np.random.uniform(-0.4, 0.4)
        
        # Stationary
        position = np.array([x, y, self.table_height])
        velocity = np.array([0.0, 0.0, 0.0])
        
        print(f"[HIT MODE] Puck spawned stationary at [{x:.3f}, {y:.3f}] near robot")
        
        return position, velocity
    
    def _tournament_init(self, min_speed=0.4, max_speed=0.6):
        """
        Tournament mode: Random center spawn with random velocity.
        
        Args:
            min_speed: Minimum puck speed (m/s)
            max_speed: Maximum puck speed (m/s)
            
        Returns:
            tuple: (position, velocity)
        """
        # Center spawn
        position = np.array([0.0, 0.0, self.table_height])
        
        # Random direction and speed
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(min_speed, max_speed)
        
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        velocity = np.array([vx, vy, 0.0])
        
        print(f"[TOURNAMENT MODE] Puck at center with velocity: [{vx:.3f}, {vy:.3f}] m/s")
        
        return position, velocity
