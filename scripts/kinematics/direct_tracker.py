"""
SIMPLE DIRECT PUCK TRACKER

Strategy: Just follow the puck. That's it.
- No complex modes
- No defensive/offensive switching  
- Just put paddle where puck is (or will be)
- Stay on table
- Don't twist around

This is the simplest possible approach that should work.
"""
import numpy as np
from scripts.kinematics.simple_puck_predictor import predict_puck_simple

def get_paddle_target(puck_pos, puck_vel, robot_side, 
                      lookahead_time=0.4,
                      table_bounds={'x': (-1.0, 1.0), 'y': (-0.52, 0.52)}):
    """
    SUPER SIMPLE: Where should paddle go?
    
    Args:
        puck_pos: [x, y, z] current puck position
        puck_vel: [vx, vy, vz] current puck velocity
        robot_side: 1.0 for right, -1.0 for left
        lookahead_time: how far ahead to predict
        table_bounds: table dimensions
        
    Returns:
        target_pos: [x, y, z] where paddle should move to
    """
    # Predict where puck will be
    predicted_pos, _ = predict_puck_simple(
        puck_pos, puck_vel, 
        time_ahead=lookahead_time,
        x_bounds=table_bounds['x'],
        y_bounds=table_bounds['y']
    )
    
    # Target = predicted puck position
    target_x = predicted_pos[0]
    target_y = predicted_pos[1]
    
    # ONLY constraint: stay on table
    target_x = np.clip(target_x, table_bounds['x'][0] + 0.1, table_bounds['x'][1] - 0.1)
    target_y = np.clip(target_y, table_bounds['y'][0] + 0.1, table_bounds['y'][1] - 0.1)
    
    # Paddle height
    paddle_z = 0.105
    
    return np.array([target_x, target_y, paddle_z])


def validate_target_reachable(target_pos, robot_side, 
                               max_reach=1.0,
                               table_bounds={'x': (-1.0, 1.0), 'y': (-0.52, 0.52)}):
    """
    Make sure target is reachable and won't cause arm twist.
    
    Args:
        target_pos: [x, y, z] desired position
        robot_side: 1.0 for right (-1.2, 0), -1.0 for left (+1.2, 0)
        max_reach: robot's max reach distance
        
    Returns:
        validated_pos: [x, y, z] safe, reachable position
    """
    robot_base_x = 1.2 if robot_side > 0 else -1.2
    
    # Check distance from base
    dist = np.linalg.norm([target_pos[0] - robot_base_x, target_pos[1]])
    
    if dist > max_reach:
        # Clamp to reachable circle
        direction = np.array([target_pos[0] - robot_base_x, target_pos[1]])
        direction = direction / np.linalg.norm(direction)
        clamped = np.array([robot_base_x, 0.0]) + direction * (max_reach - 0.05)
        target_pos[0] = clamped[0]
        target_pos[1] = clamped[1]
        
    # Re-clamp to table (priority)
    target_pos[0] = np.clip(target_pos[0], table_bounds['x'][0] + 0.1, table_bounds['x'][1] - 0.1)
    target_pos[1] = np.clip(target_pos[1], table_bounds['y'][0] + 0.1, table_bounds['y'][1] - 0.1)
    
    # Check base angle (don't twist)
    dx = target_pos[0] - robot_base_x
    dy = target_pos[1]
    angle = np.arctan2(dy, dx)
    
    if abs(angle) > np.deg2rad(90):
        # Too much rotation - force to safe angle
        safe_angle = np.clip(angle, -np.deg2rad(90), np.deg2rad(90))
        reach_dist = min(dist, max_reach - 0.05)
        target_pos[0] = robot_base_x + reach_dist * np.cos(safe_angle)
        target_pos[1] = 0.0 + reach_dist * np.sin(safe_angle)
        
        # Final table clamp
        target_pos[0] = np.clip(target_pos[0], table_bounds['x'][0] + 0.1, table_bounds['x'][1] - 0.1)
        target_pos[1] = np.clip(target_pos[1], table_bounds['y'][0] + 0.1, table_bounds['y'][1] - 0.1)
    
    return target_pos
