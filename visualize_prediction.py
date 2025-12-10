"""
PUCK PREDICTION VISUALIZATION

Visualizes the puck prediction model by showing:
- Puck sliding on table (no robots)
- Cylinders at predicted future positions
- Updates in real-time every frame

Green cylinders = predicted positions over next 5 seconds
"""

import numpy as np
import time
from pydrake.all import (
    DiagramBuilder, Simulator, MultibodyPlant, SceneGraph,
    Parser, RigidTransform, SpatialVelocity,
    MeshcatVisualizer, StartMeshcat, Rgba,
    Cylinder, Sphere
)
from pydrake.geometry import GeometryInstance, MakePhongIllustrationProperties
from pathlib import Path
import sys

# Add scripts to path
sys.path.append(str(Path(__file__).parent))
from scripts.kinematics.simple_puck_predictor import predict_puck_simple


def predict_with_all_wall_bounces(puck_pos, puck_vel, time_ahead, x_bounds, y_bounds, dt=0.001):
    """
    Predict puck position accounting for bounces off ALL four walls.
    Uses smaller timestep for accuracy at high speeds.
    
    Args:
        puck_pos: [x, y, z] current position
        puck_vel: [vx, vy, vz] current velocity
        time_ahead: prediction horizon in seconds
        x_bounds: (x_min, x_max) tuple
        y_bounds: (y_min, y_max) tuple
        dt: timestep for integration (smaller = more accurate)
    
    Returns:
        (final_pos, final_vel) - predicted position and velocity
    """
    # Work in 2D
    pos = np.array([puck_pos[0], puck_pos[1]], dtype=float)
    vel = np.array([puck_vel[0], puck_vel[1]], dtype=float)
    
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    
    t = 0.0
    while t < time_ahead:
        # Predict next position
        next_pos = pos + vel * dt
        
        # Check and handle X-axis bounces (end walls)
        if next_pos[0] < x_min:
            vel[0] = abs(vel[0])  # Bounce right
            pos[0] = x_min  # Clamp to boundary
        elif next_pos[0] > x_max:
            vel[0] = -abs(vel[0])  # Bounce left
            pos[0] = x_max
        
        # Check and handle Y-axis bounces (side walls)
        if next_pos[1] < y_min:
            vel[1] = abs(vel[1])  # Bounce up
            pos[1] = y_min
        elif next_pos[1] > y_max:
            vel[1] = -abs(vel[1])  # Bounce down
            pos[1] = y_max
        
        # Integrate position (with clamped boundaries)
        pos = pos + vel * dt
        pos[0] = np.clip(pos[0], x_min, x_max)
        pos[1] = np.clip(pos[1], y_min, y_max)
        
        t += dt
    
    return np.array([pos[0], pos[1], puck_pos[2]]), np.array([vel[0], vel[1], 0.0])


def visualize_predictions(meshcat, puck_pos, puck_vel, num_predictions=10, max_time=5.0):
    """
    Visualize predicted puck positions as green cylinders.
    
    Args:
        meshcat: Meshcat instance
        puck_pos: Current puck position [x, y, z]
        puck_vel: Current puck velocity [vx, vy, vz]
        num_predictions: Number of prediction markers to show
        max_time: Maximum lookahead time (seconds)
    """
    table_bounds = {
        'x_min': -1.0, 'x_max': 1.0,
        'y_min': -0.52, 'y_max': 0.52
    }
    
    # Clear old predictions
    meshcat.Delete("predictions")
    
    # Create prediction markers at different times
    for i in range(num_predictions):
        t = (i + 1) * (max_time / num_predictions)
        
        # Predict position at time t with all wall bounces
        pred_pos, pred_vel = predict_with_all_wall_bounces(
            puck_pos, puck_vel,
            time_ahead=t,
            x_bounds=(table_bounds['x_min'], table_bounds['x_max']),
            y_bounds=(table_bounds['y_min'], table_bounds['y_max']),
            dt=0.001  # Small timestep for accuracy
        )
        
        # Create cylinder at predicted position
        # Color intensity based on time (darker = further in future)
        alpha = 1.0 - (i / num_predictions) * 0.7  # Fade from 1.0 to 0.3
        
        # Green for predictions
        color = Rgba(0.0, 1.0, 0.0, alpha)
        
        # Small cylinder marker
        cylinder_height = 0.01
        cylinder_radius = 0.02
        
        # Set transform (position + rotation to stand upright)
        X = RigidTransform([pred_pos[0], pred_pos[1], 0.11])
        
        # Add to meshcat
        path = f"predictions/marker_{i}"
        meshcat.SetObject(path, Cylinder(cylinder_radius, cylinder_height), color)
        meshcat.SetTransform(path, X)


def main():
    print("="*70)
    print("PUCK PREDICTION VISUALIZATION")
    print("="*70)
    print("Visualizing puck motion with 5-second lookahead predictions")
    print("Green cylinders = predicted future positions")
    print("="*70)
    print()
    
    # Paths to assets
    repo_dir = Path(__file__).parent
    assets_dir = repo_dir / "scripts" / "env" / "assets" / "models"
    
    # Build simulation
    builder = DiagramBuilder()
    plant = MultibodyPlant(time_step=0.001)
    
    # CRITICAL: Disable gravity for air hockey physics!
    plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    
    scene_graph = SceneGraph()
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    builder.AddSystem(plant)
    builder.AddSystem(scene_graph)
    
    parser = Parser(plant)
    
    # Load table
    print("[SETUP] Loading table...")
    table_sdf = assets_dir / "air_hockey_table" / "table.sdf"
    table = parser.AddModels(str(table_sdf))[0]
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("table_body", table),
        RigidTransform([0, 0, 0.10])
    )
    
    # Load puck
    print("[SETUP] Loading puck...")
    puck_sdf = assets_dir / "puck" / "puck.sdf"
    puck_model = parser.AddModels(str(puck_sdf))[0]
    puck_body = plant.GetBodyByName("puck_body_link")
    
    # Finalize
    plant.Finalize()
    
    # CRITICAL: Collision filtering BEFORE building diagram!
    # Exclude puck-table SURFACE collisions to achieve frictionless motion
    # But keep puck-wall collisions for bounces
    print("[SETUP] Filtering puck-table surface collisions (frictionless motion)...")
    try:
        from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
        
        # Get puck geometries
        puck_geom_ids = plant.GetCollisionGeometriesForBody(puck_body)
        
        # Get table surface geometry (just the main surface, not walls)
        table_body = plant.GetBodyByName("table_body", table)
        table_geom_ids = plant.GetCollisionGeometriesForBody(table_body)
        
        if len(puck_geom_ids) > 0 and len(table_geom_ids) > 0:
            puck_set = GeometrySet(puck_geom_ids)
            table_set = GeometrySet(table_geom_ids)
            
            filter_decl = CollisionFilterDeclaration()
            filter_decl.ExcludeBetween(puck_set, table_set)
            
            scene_graph.collision_filter_manager().Apply(filter_decl)
            print(f"[INFO] Filtered puck-table collisions: puck={len(puck_geom_ids)}, table={len(table_geom_ids)}")
        else:
            print(f"[WARNING] Could not apply filter: puck={len(puck_geom_ids)}, table={len(table_geom_ids)}")
    except Exception as e:
        print(f"[WARNING] Collision filtering failed: {e}")
    
    # Visualization
    print("[SETUP] Starting visualizer...")
    meshcat = StartMeshcat()
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    
    # CRITICAL: Wait for meshcat to fully initialize before proceeding
    print("[SETUP] Waiting for Meshcat to fully render...")
    import time
    time.sleep(2.0)  # Give Meshcat time to load and render
    
    # Connect ports
    builder.Connect(
        plant.get_geometry_pose_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id())
    )
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port()
    )
    
    # Build
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    
    # Get puck body
    puck_body = plant.GetBodyByName("puck_body_link")
    
    # Initialize puck with 1 m/s velocity
    print("[INIT] Launching puck at 1.0 m/s...")
    initial_pos = np.array([0.0, 0.0, 0.11])
    initial_vel = np.array([0.8, 0.6, 0.0])  # 1.0 m/s total
    
    plant.SetFreeBodyPose(plant_context, puck_body, RigidTransform(initial_pos))
    plant.SetFreeBodySpatialVelocity(plant_context, puck_body,
        SpatialVelocity(np.concatenate([np.zeros(3), initial_vel])))
    
    print(f"[INFO] Meshcat visualization: {meshcat.web_url()}")
    print("[INFO] Watch the green cylinders showing predicted positions!")
    print("[INFO] Predictions account for multiple wall bounces!")
    print()
    print("="*70)
    print("PUCK PREDICTION DEMO - 1.0 m/s for 30 seconds")
    print("="*70)
    print()
    
    # Simulation loop
    duration = 30.0  # 30 seconds total
    sim_time = 0.0
    last_update = 0.0
    update_interval = 0.1  # Update predictions every 0.1s
    
    while sim_time < duration:
        # Step simulation
        simulator.AdvanceTo(sim_time + 0.01)
        sim_time += 0.01
        
        # Get puck state AFTER step
        puck_pose = plant.EvalBodyPoseInWorld(plant_context, puck_body)
        puck_vel_spatial = plant.EvalBodySpatialVelocityInWorld(plant_context, puck_body)
        puck_pos = np.array(puck_pose.translation())
        puck_vel = np.array(puck_vel_spatial.translational())
        
        # PREDICTIVE PHYSICS: Check if NEXT position would cross wall
        next_pos = puck_pos + puck_vel * 0.01
        
        # Wall bounces - only update velocity if bouncing
        vel_changed = False
        new_vel = puck_vel.copy()
        
        if next_pos[0] < -1.0:  # Left wall
            new_vel[0] = abs(puck_vel[0])
            vel_changed = True
        elif next_pos[0] > 1.0:  # Right wall
            new_vel[0] = -abs(puck_vel[0])
            vel_changed = True
            
        if next_pos[1] < -0.52:  # Bottom wall
            new_vel[1] = abs(puck_vel[1])
            vel_changed = True
        elif next_pos[1] > 0.52:  # Top wall
            new_vel[1] = -abs(puck_vel[1])
            vel_changed = True
        
        # Force Z velocity to zero (2D motion)
        new_vel[2] = 0.0
        
        # Only set velocity if it changed
        if vel_changed or abs(puck_vel[2]) > 0.001:
            plant.SetFreeBodySpatialVelocity(plant_context, puck_body,
                SpatialVelocity(np.concatenate([np.zeros(3), new_vel])))
        
        # Fix position if out of bounds
        pos_fixed = False
        fixed_pos = puck_pos.copy()
        
        if fixed_pos[0] < -1.0:
            fixed_pos[0] = -1.0
            pos_fixed = True
        elif fixed_pos[0] > 1.0:
            fixed_pos[0] = 1.0
            pos_fixed = True
            
        if fixed_pos[1] < -0.52:
            fixed_pos[1] = -0.52
            pos_fixed = True
        elif fixed_pos[1] > 0.52:
            fixed_pos[1] = 0.52
            pos_fixed = True
        
        # Always keep on table
        if abs(fixed_pos[2] - 0.11) > 0.001:
            fixed_pos[2] = 0.11
            pos_fixed = True
        
        if pos_fixed:
            plant.SetFreeBodyPose(plant_context, puck_body, RigidTransform(fixed_pos))
        
        # Update prediction visualization
        if sim_time - last_update >= update_interval:
            visualize_predictions(meshcat, puck_pos, puck_vel, 
                                num_predictions=10, max_time=5.0)
            last_update = sim_time
        
        # Status every 2 seconds
        if int(sim_time) % 2 == 0 and sim_time - int(sim_time) < 0.01:
            speed = np.linalg.norm(puck_vel[:2])
            print(f"[{sim_time:.0f}s] Puck: [{puck_pos[0]:.2f}, {puck_pos[1]:.2f}] Speed: {speed:.2f} m/s")
    
    
    
    
    
    print()
    print("="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print("The green cylinders showed where the puck WILL BE in 0-5 seconds")
    print("Notice how they predict bounces off the walls!")
    print("="*70)


if __name__ == "__main__":
    main()
