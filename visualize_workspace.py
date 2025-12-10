#!/usr/bin/env python3
"""
Visualize Robot Workspace on Air Hockey Table

This script displays the robot's reachable workspace as visual indicators
in the simulation environment. Helps debug workspace safety bounds and 
understand where the robot can/cannot reach.

Usage:
    python visualize_workspace.py --mode hit    # Single robot (right)
    python visualize_workspace.py --mode tournament  # Both robots
"""

import numpy as np
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    MeshcatVisualizer,
    StartMeshcat,
    RigidTransform,
    Cylinder,
    Box,
    Rgba,
)
from scripts.env.configure import create_air_hockey_simulation
import argparse


def add_workspace_visualization(meshcat, plant, context, robot_model, robot_side, color_name="red"):
    """
    Add visual indicators for workspace - CIRCULAR based on arm reach.
    
    Args:
        meshcat: Meshcat visualizer instance
        plant: MultibodyPlant
        context: Context to get robot state
        robot_model: Robot model instance
        robot_side: 1.0 for right, -1.0 for left
        color_name: Color for visualization
    """
    from pydrake.all import Sphere
    
    # Get ACTUAL robot base position from plant
    try:
        plant_context = plant.GetMyContextFromRoot(context)
        base_body = plant.GetBodyByName("iiwa_link_0", robot_model)
        base_pose = plant.EvalBodyPoseInWorld(plant_context, base_body)
        robot_base_pos = base_pose.translation()
        robot_base_x = robot_base_pos[0]
        robot_base_y = robot_base_pos[1]
        robot_base_z = robot_base_pos[2]
    except:
        # Fallback to approximation
        if robot_side > 0:
            robot_base_x = 0.6
        else:
            robot_base_x = -0.6
        robot_base_y = 0.0
        robot_base_z = 0.0
    
    table_height = 0.11
    
    # IIWA arm reach characteristics
    # IIWA 7 has ~820mm reach from base
    max_reach = 0.82  # meters
    min_reach = 0.15  # Inner dead zone
    
    # Color mapping
    colors = {
        "red": Rgba(1.0, 0.0, 0.0, 0.25),  # More visible
        "blue": Rgba(0.0, 0.0, 1.0, 0.25),
        "green": Rgba(0.0, 1.0, 0.0, 0.25),
        "yellow": Rgba(1.0, 1.0, 0.0, 0.25),
    }
    color = colors.get(color_name, colors["red"])
    
    # Create circular workspace visualization using a thin cylinder
    # Made thicker for better visibility
    workspace_cylinder = Cylinder(max_reach, 0.01)  # 1cm thick (was 2mm)
    
    # Position at robot base, at table height
    transform = RigidTransform(
        p=[robot_base_x, robot_base_y, table_height]
    )
    
    # Add to meshcat
    side_name = "right" if robot_side > 0 else "left"
    path = f"/workspace/{side_name}_reach"
    meshcat.SetObject(path, workspace_cylinder, color)
    meshcat.SetTransform(path, transform)
    
    # Add inner dead zone (area too close to reach)
    if min_reach > 0.05:
        inner_color = Rgba(color.r(), color.g(), color.b(), 0.4)  # More opaque
        inner_cylinder = Cylinder(min_reach, 0.015)  # Thicker
        inner_transform = RigidTransform(
            p=[robot_base_x, robot_base_y, table_height + 0.005]
        )
        inner_path = f"/workspace/{side_name}_inner"
        meshcat.SetObject(inner_path, inner_cylinder, inner_color)
        meshcat.SetTransform(inner_path, inner_transform)
    
    # Add small sphere at robot base for reference
    base_sphere = Sphere(0.08)  # Bigger sphere
    base_color = Rgba(color.r(), color.g(), color.b(), 0.9)  # More visible
    base_transform = RigidTransform(
        p=[robot_base_x, robot_base_y, robot_base_z + 0.05]  # Above robot base
    )
    base_path = f"/workspace/{side_name}_base"
    meshcat.SetObject(base_path, base_sphere, base_color)
    meshcat.SetTransform(base_path, base_transform)
    
    # Calculate actual reachable area
    reachable_area = np.pi * (max_reach**2 - min_reach**2)
    
    # Print workspace info
    print(f"\n{'='*60}")
    print(f"{side_name.upper()} ROBOT WORKSPACE (Circular)")
    print(f"{'='*60}")
    print(f"Robot base: [{robot_base_x:.3f}, {robot_base_y:.3f}, {robot_base_z:.3f}]")
    print(f"Maximum reach: {max_reach:.3f} m")
    print(f"Minimum reach: {min_reach:.3f} m (inner dead zone)")
    print(f"Reachable area: {reachable_area:.3f} m²")
    print(f"{'='*60}\n")


def visualize_workspaces(mode="hit"):
    """
    Main visualization function.
    
    Args:
        mode: Game mode - determines number of robots
    """
    print("\n" + "="*60)
    print("ROBOT WORKSPACE VISUALIZATION")
    print("="*60)
    print(f"Mode: {mode.upper()}")
    print("="*60 + "\n")
    
    # Determine number of arms based on mode
    if mode.lower() in ["hit", "defend"]:
        num_arms = 1
        robot_side = "right"  # Single robot always on right
    else:  # tournament
        num_arms = 2
        robot_side = "right"  # Will show both
    
    # Create simulation environment
    print("Creating simulation environment...")
    
    # Call the function with correct signature
    simulator, meshcat_from_sim, plant, diagram = create_air_hockey_simulation(
        num_arms=num_arms,
        time_step=0.001,
        use_meshcat=True,
        skip_grasp=True,
        game_mode=mode.upper(),
        robot_side=robot_side
    )
    
    # Use the meshcat instance from simulation
    if meshcat_from_sim:
        meshcat = meshcat_from_sim
    
    # Get context
    context = simulator.get_mutable_context()
    
    # Find robot models
    robot_models = {}
    try:
        robot_models['right'] = plant.GetModelInstanceByName("right_iiwa")
    except:
        pass
    
    try:
        robot_models['left'] = plant.GetModelInstanceByName("left_iiwa")
    except:
        pass
    
    # Add workspace visualizations
    if robot_models.get('right'):
        print("\nVisualizing RIGHT robot workspace...")
        add_workspace_visualization(meshcat, plant, context, robot_models['right'], 1.0, "red")
    
    if robot_models.get('left'):
        print("\nVisualizing LEFT robot workspace...")
        add_workspace_visualization(meshcat, plant, context, robot_models['left'], -1.0, "blue")
    
    # Add table boundaries visualization
    print("\nAdding table boundaries...")
    table_bounds_x = (-1.064, 1.064)
    table_bounds_y = (-0.609, 0.609)
    table_height = 0.11
    
    # Thin lines for table edges
    edge_thickness = 0.01
    edge_height = 0.1
    edge_color = Rgba(0.5, 0.5, 0.5, 0.5)
    
    # X edges (along Y direction)
    x_edge_box = Box(edge_thickness, table_bounds_y[1] - table_bounds_y[0], edge_height)
    for x_pos, name in [(table_bounds_x[0], "left"), (table_bounds_x[1], "right")]:
        transform = RigidTransform(p=[x_pos, 0, table_height + edge_height/2])
        meshcat.SetObject(f"/table_bounds/edge_x_{name}", x_edge_box, edge_color)
        meshcat.SetTransform(f"/table_bounds/edge_x_{name}", transform)
    
    # Y edges (along X direction)
    y_edge_box = Box(table_bounds_x[1] - table_bounds_x[0], edge_thickness, edge_height)
    for y_pos, name in [(table_bounds_y[0], "bottom"), (table_bounds_y[1], "top")]:
        transform = RigidTransform(p=[0, y_pos, table_height + edge_height/2])
        meshcat.SetObject(f"/table_bounds/edge_y_{name}", y_edge_box, edge_color)
        meshcat.SetTransform(f"/table_bounds/edge_y_{name}", transform)
    
    print("\n" + "="*60)
    print("LEGEND")
    print("="*60)
    print("RED semi-transparent disk:  Right robot max reach (82cm radius)")
    if num_arms == 2:
        print("BLUE semi-transparent disk: Left robot max reach (82cm radius)")
    print("Darker inner disk:          Inner dead zone (15cm radius)")
    print("Small sphere:               Robot base position")
    print("GREY lines:                 Table physical boundaries")
    print("="*60)
    
    print("\n" + "="*60)
    print("VISUALIZATION READY")
    print("="*60)
    print("\nThe workspace is now visible in Meshcat!")
    print("- Semi-transparent colored boxes show robot reach")
    print("- Cylinders mark workspace corners")
    print("- Grey lines show table boundaries")
    print("\nVisualization will remain open.")
    print("Press Ctrl+C to exit.")
    print("="*60 + "\n")
    
    # Keep visualization open
    try:
        input("Press Enter to exit...")
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize robot workspace on air hockey table"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hit",
        choices=["hit", "defend", "tournament"],
        help="Game mode (determines number of robots)"
    )
    
    args = parser.parse_args()
    visualize_workspaces(mode=args.mode)


if __name__ == "__main__":
    main()
