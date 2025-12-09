"""
Predictive air hockey puck physics based on paper methodology.
Predicts next position, detects boundary crossings, and applies physics before they occur.
"""
import numpy as np
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialVelocity
from scripts.env.game_modes import GameMode, PuckInitializer

class PredictivePuckMonitor:
    """
    Handles air hockey puck physics using predictive collision detection:
    - Predicts next position based on current velocity
    - Detects wall/goal boundary crossings BEFORE they occur
    - Applies elastic collision (conservation of momentum) for walls
    - Triggers goal scoring and reset for goal boundaries
    - Supports multiple game modes (defend, hit, tournament)
    """
    def __init__(self, plant, puck_body, 
                 x_bounds=(-1.0, 1.0), y_bounds=(-0.52, 0.52),
                 restitution=1.0, dt=0.01, max_velocity=0.3,
                 game_mode=GameMode.TOURNAMENT, robot_side="right"):
        """
        Args:
            restitution: 1.0 for perfect elastic (conservation of momentum)
            game_mode: GameMode constant (DEFEND, HIT, or TOURNAMENT)
            robot_side: "right" or "left" - which side the training robot is on
        """
        self.plant = plant
        self.puck_body = puck_body
        self.x_min, self.x_max = x_bounds  # Goal boundaries
        self.y_min, self.y_max = y_bounds  # Wall boundaries
        self.restitution = restitution  # Should be 1.0 for perfect elastic
        self.dt = dt
        self.table_height = 0.11
        self.max_velocity = max_velocity  # Max puck speed
        self.puck_radius = 0.03165  # Puck radius (from MuJoCo spec)
        
        # Game mode support
        self.game_mode = game_mode
        self.robot_side = robot_side
        self.puck_initializer = PuckInitializer(
            mode=game_mode,
            robot_side=robot_side,
            table_bounds={'x': x_bounds, 'y': y_bounds}
        )
        
        self.score = [0, 0]  # [home, away]
        self.goal_pause = 0.0  # Timer for goal celebration
        self.goal_pause_duration = 2.0  # seconds
        
    def update(self, context, plant_context):
        """
        Predictive physics update - called every timestep.
        
        1. Get current state
        2. Predict next position
        3. Check for boundary crossings
        4. Apply physics (bounce or goal)
        """
        # Handle goal pause (puck hidden)
        if self.goal_pause > 0:
            self.goal_pause -= self.dt
            if self.goal_pause <= 0:
                self._reset_puck(plant_context)
            return
        
        # Get current state
        pose = self.plant.EvalBodyPoseInWorld(plant_context, self.puck_body)
        vel = self.plant.EvalBodySpatialVelocityInWorld(plant_context, self.puck_body)
        
        pos = np.array(pose.translation())
        v = np.array(vel.translational())
        
        # Predict next position
        next_pos = pos + v * self.dt
        
        # Check for GOAL boundaries (X) - predict crossing
        # Goals only count if within goal posts (Y bounds)
        goal_post_width = 0.25  # Goal posts from -0.25 to +0.25 in Y
        
        if next_pos[0] < self.x_min or next_pos[0] > self.x_max:
            # Check if within goal posts
            if abs(pos[1]) <= goal_post_width:
                # GOAL!
                if next_pos[0] < self.x_min:
                    print(f"[GOAL!] Away team scores! Score: {self.score[0]} - {self.score[1]+1}")
                    self.score[1] += 1
                else:
                    print(f"[GOAL!] Home team scores! Score: {self.score[0]+1} - {self.score[1]}")
                    self.score[0] += 1
                
                self._hide_puck(plant_context)
                self.goal_pause = self.goal_pause_duration
                return
            else:
                # Hit the wall next to goal, bounce off end wall
                print(f"[WALL] Puck hit end wall outside goal posts (y={pos[1]:.3f})")
                v_new = v.copy()
                if next_pos[0] < self.x_min:
                    v_new[0] = abs(v[0])  # Bounce right
                else:
                    v_new[0] = -abs(v[0])  # Bounce left
                
                # Apply velocity
                v_new[2] = 0.0
                v_spatial = np.concatenate([np.zeros(3), v_new])
                self.plant.SetFreeBodySpatialVelocity(
                    plant_context, self.puck_body, SpatialVelocity(v_spatial)
                )
                return
        
        # Check for WALL boundaries (Y) - perfect elastic collision
        # Conservation of momentum: just flip the perpendicular component
        # Add buffer equal to puck radius to prevent visual clipping
        v_new = v.copy()
        bounced = False
        
        wall_buffer = self.puck_radius
        
        if next_pos[1] < (self.y_min + wall_buffer):
            # Will cross bottom wall - flip Y velocity (perfect elastic)
            v_new[1] = abs(v[1])  # Make positive (bounce upward)
            # Clamp position to prevent clipping
            pos[1] = max(pos[1], self.y_min + wall_buffer)
            bounced = True
            print(f"[BOUNCE] Bottom wall - vel flipped to +{abs(v[1]):.3f}")
        elif next_pos[1] > (self.y_max - wall_buffer):
            # Will cross top wall - flip Y velocity (perfect elastic)
            v_new[1] = -abs(v[1])  # Make negative (bounce downward)
            # Clamp position to prevent clipping
            pos[1] = min(pos[1], self.y_max - wall_buffer)
            bounced = True
            print(f"[BOUNCE] Top wall - vel flipped to -{abs(v[1]):.3f}")
        
        # Enforce 2D motion and velocity limits
        # NOTE: NO FRICTION - speed is only clamped, not reduced
        v_new[2] = 0.0  # Force Z velocity to zero
        
        # Clamp to max velocity (safety only, not friction)
        speed = np.linalg.norm(v_new[:2])
        if speed > self.max_velocity:
            v_new[:2] = v_new[:2] * (self.max_velocity / speed)
        
        # Update velocity if changed
        if bounced or np.linalg.norm(v_new - v) > 0.001:
            v_spatial = np.concatenate([np.zeros(3), v_new])
            self.plant.SetFreeBodySpatialVelocity(
                plant_context, self.puck_body, SpatialVelocity(v_spatial)
            )
        
        # Position correction if out of bounds (safety check)
        if pos[1] < self.y_min or pos[1] > self.y_max:
            pos[1] = np.clip(pos[1], self.y_min, self.y_max)
            pos[2] = self.table_height
            self.plant.SetFreeBodyPose(
                plant_context, self.puck_body, RigidTransform(pos)
            )
    
    def _hide_puck(self, plant_context):
        """Freeze puck at origin during goal celebration (don't move off table)."""
        self.plant.SetFreeBodyPose(
            plant_context, self.puck_body,
            RigidTransform([0, 0, self.table_height])  # Keep on table at center
        )
        self.plant.SetFreeBodySpatialVelocity(
            plant_context, self.puck_body, SpatialVelocity(np.zeros(6))
        )
    
    def _reset_puck(self, plant_context):
        """Reset puck based on game mode."""
        # Get mode-specific initialization
        position, velocity = self.puck_initializer.get_initial_state()
        
        # Set position
        self.plant.SetFreeBodyPose(
            plant_context, self.puck_body,
            RigidTransform(position)
        )
        
        # Set velocity
        v_spatial = np.concatenate([np.zeros(3), velocity])
        self.plant.SetFreeBodySpatialVelocity(
            plant_context, self.puck_body, SpatialVelocity(v_spatial)
        )
