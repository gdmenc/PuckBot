"""
Simple, physics-accurate puck predictor using environment's actual model.
"""
import numpy as np

class SimplePuckPredictor:
    """
    Predict puck trajectory using the EXACT physics from PredictivePuckMonitor.
    
    This matches the environment's physics perfectly:
    - Wall bounces (elastic, restitution=1.0)
    - Velocity clamping (max_velocity)
    - No friction
    - 2D motion only
    """
    
    def __init__(self,
                 x_bounds=(-1.0, 1.0),
                 y_bounds=(-0.52, 0.52),
                 puck_radius=0.03165,
                 max_velocity=0.3,
                 restitution=1.0):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.puck_radius = puck_radius
        self.max_velocity = max_velocity
        self.restitution = restitution
        
    def predict(self, pos, vel, horizon=1.0, dt=0.01):
        """
        Predict puck trajectory over time horizon.
        
        Uses EXACT physics from environment:
        - Perfect elastic wall bounces
        - Velocity clamping
        - No friction
        
        Args:
            pos: [x, y, z] current position
            vel: [vx, vy, vz] current velocity  
            horizon: prediction time in seconds
            dt: timestep
            
        Returns:
            List of (time, pos, vel) tuples
        """
        trajectory = []
        
        # Work in 2D
        p = np.array([pos[0], pos[1]])
        v = np.array([vel[0], vel[1]])
        
        # Clamp initial velocity
        speed = np.linalg.norm(v)
        if speed > self.max_velocity:
            v = v * (self.max_velocity / speed)
        
        t = 0.0
        while t < horizon:
            # Predict next position
            p_next = p + v * dt
            
            # Check wall bounces (Y boundaries)
            wall_buffer = self.puck_radius
            
            if p_next[1] < (self.y_min + wall_buffer):
                # Bottom wall bounce
                v[1] = abs(v[1])  # Flip to positive
                p[1] = max(p[1], self.y_min + wall_buffer)
            elif p_next[1] > (self.y_max - wall_buffer):
                # Top wall bounce  
                v[1] = -abs(v[1])  # Flip to negative
                p[1] = min(p[1], self.y_max - wall_buffer)
            
            # Check goal boundaries (X) - no bounce, just mark
            # We still track trajectory beyond goals for prediction
            
            # Update position
            p = p + v * dt
            
            # Clamp velocity (safety only, no friction!)
            speed = np.linalg.norm(v)
            if speed > self.max_velocity:
                v = v * (self.max_velocity / speed)
            
            # Store prediction
            trajectory.append((t, np.array([p[0], p[1], 0.11]), np.array([v[0], v[1], 0.0])))
            
            t += dt
        
        return trajectory


def predict_puck_simple(puck_pos, puck_vel, time_ahead=0.6, 
                        x_bounds=(-1.0, 1.0), y_bounds=(-0.52, 0.52)):
    """
    Quick helper: Where will puck be in 'time_ahead' seconds?
    
    Uses environment physics (bounces, clamping, no friction).
    """
    predictor = SimplePuckPredictor(x_bounds=x_bounds, y_bounds=y_bounds)
    trajectory = predictor.predict(puck_pos, puck_vel, horizon=time_ahead, dt=0.01)
    
    if not trajectory:
        return puck_pos, puck_vel
    
    # Return final predicted state
    _, final_pos, final_vel = trajectory[-1]
    return final_pos, final_vel
