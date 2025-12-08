"""
Simple manual grasp definitions for paddle cylinder stem
"""
import numpy as np
import pickle
from pydrake.math import RigidTransform, RotationMatrix

def create_simple_paddle_grasps(n_grasps: int = 12):
    """
    Create simple grasps around a cylinder axis.
    For a vertical cylinder (stem), grasp from different angles.
    
    Args:
        n_grasps: Number of grasps around the cylinder
    """
    grasps = []
    
    # Cylinder stem: radius ~0.015m
    stem_radius = 0.015
    
    # Gripper approaches from the side
    # x-axis points radially inward  
    # y-axis points down (finger closure direction)
    # z-axis tangent to cylinder
    
    for i in range(n_grasps):
        angle = 2 * np.pi * i / n_grasps
        
        # Position: at cylinder surface
        x = stem_radius * np.cos(angle)
        y = stem_radius * np.sin(angle)
        z = 0.0  # Middle of stem
        
        # Rotation: approach from radial direction
        # x-axis points toward center (-radial)
        x_axis = np.array([-np.cos(angle), -np.sin(angle), 0.0])
        # y-axis points down (for WSG gripper)
        y_axis = np.array([0.0, 0.0, -1.0])
        # z-axis = tangent
        z_axis = np.cross(x_axis, y_axis)
        
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # Add finger offset (gripper base is ~0.1m behind fingers)
        p = np.array([x, y, z]) + y_axis * -0.1
        
        X_OG = RigidTransform(RotationMatrix(R), p)
        
        grasps.append({
            'pose_matrix': X_OG.GetAsMatrix4(),
            'quality': 1.0,  # All equally good for symmetric cylinder
            'angle': angle
        })
    
    # Save to file
    output_path = '/Users/gdmen/MIT/F2025/6.4210/PuckBot/scripts/grasp/precomputed_paddle_grasps.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(grasps, f)
    
    print(f"Created {len(grasps)} manual grasps for paddle cylinder")
    print(f"Saved to: {output_path}")
    
    return grasps

if __name__ == "__main__":
    create_simple_paddle_grasps(n_grasps=16)
