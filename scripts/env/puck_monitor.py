"""
Predictive air hockey puck physics based on paper methodology.
Predicts next position, detects boundary crossings, and applies physics before they occur.
"""
import numpy as np
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialVelocity

class PredictivePuckMonitor:
    """
    Handles air hockey puck physics using predictive collision detection:
    - Predicts next position based on current velocity
    - Detects wall/goal boundary crossings BEFORE they occur
    - Applies elastic collision (conservation of momentum) for walls
    - Triggers goal scoring and reset for goal boundaries
    """
    def __init__(self, plant, puck_body, 
                 x_bounds=(-1.0, 1.0), y_bounds=(-0.55, 0.55),
                 restitution=0.95, dt=0.01, max_velocity=0.3):
        self.plant = plant
        self.puck_body = puck_body
        self.x_min, self.x_max = x_bounds  # Goal boundaries
        self.y_min, self.y_max = y_bounds  # Wall boundaries
        self.restitution = restitution
        self.dt = dt
        self.table_height = 0.11
        self.max_velocity = max_velocity  # Max puck speed
        
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
        if next_pos[0] < self.x_min or next_pos[0] > self.x_max:
            if next_pos[0] < self.x_min:
                print(f"[GOAL!] Away team scores! Score: {self.score[0]} - {self.score[1]+1}")
                self.score[1] += 1
            else:
                print(f"[GOAL!] Home team scores! Score: {self.score[0]+1} - {self.score[1]}")
                self.score[0] += 1
            
            self._hide_puck(plant_context)
            self.goal_pause = self.goal_pause_duration
            return
        
        # Check for WALL boundaries (Y) - predict crossing and bounce
        v_new = v.copy()
        bounced = False
        
        if next_pos[1] < self.y_min:
            # Will cross bottom wall - reflect Y velocity
            v_new[1] = abs(v[1]) * self.restitution
            bounced = True
            print(f"[BOUNCE] Bottom wall (y={next_pos[1]:.3f})")
        elif next_pos[1] > self.y_max:
            # Will cross top wall - reflect Y velocity
            v_new[1] = -abs(v[1]) * self.restitution
            bounced = True
            print(f"[BOUNCE] Top wall (y={next_pos[1]:.3f})")
        
        # Always clamp velocities and force Z=0 (2D motion)
        v_new[0] = np.clip(v_new[0], -self.max_velocity, self.max_velocity)
        v_new[1] = np.clip(v_new[1], -self.max_velocity, self.max_velocity)
        v_new[2] = 0.0
        
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
        """Reset puck to center with random velocity < 0.3 m/s."""
        # Position at center
        self.plant.SetFreeBodyPose(
            plant_context, self.puck_body,
            RigidTransform([0, 0, self.table_height])
        )
        
        # Random direction and speed
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0.1, self.max_velocity)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        v_spatial = np.array([0, 0, 0, vx, vy, 0])
        self.plant.SetFreeBodySpatialVelocity(
            plant_context, self.puck_body, SpatialVelocity(v_spatial)
        )
        
        print(f"[RESET] Puck at center with velocity: [{vx:.3f}, {vy:.3f}] m/s")
