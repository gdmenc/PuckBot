from pydrake.all import (
    DiagramBuilder,
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    Simulator,
    StartMeshcat,
    AddDefaultVisualization,
    RigidTransform,
    RotationMatrix,
    SpatialVelocity,
)
from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlantConfig
from pydrake.systems.analysis import ApplySimulatorConfig, SimulatorConfig
import numpy as np

from scripts.env.scenario_builder import generate_scenario_yaml

def initialize_puck(simulator, plant, puck_model, velocity=[0.5, 0.3, 0]):
    """
    Initializes the puck at the center of the table with a starting velocity.
    As per paper: puck motion is 2D (xy-plane), z velocity constrained to 0.
    
    Args:
        velocity: [vx, vy, vz] - initial velocity (vz should be 0)
    """
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    
    try:
        puck_body = plant.GetBodyByName("puck_body_link", puck_model)
        
        # Set position at center of table, slightly above surface
        plant.SetFreeBodyPose(
            plant_context,
            puck_body,
            RigidTransform([0.0, 0.0, 0.15])  # Center, at table height
        )
        
        # Set initial velocity (2D motion, no z component)
        velocity = np.array(velocity, dtype=float)
        velocity[2] = 0.0  # Force z velocity to zero
        
        # Split velocity into linear (xyz) and angular (000)
        v = np.concatenate([np.zeros(3), velocity])  # [wx, wy, wz, vx, vy, vz]
        
        plant.SetFreeBodySpatialVelocity(
            plant_context,
            puck_body,
            SpatialVelocity(v)
        )
        
    except Exception as e:
        print(f"[WARNING] Could not initialize puck: {e}")

def initialize_paddles(plant, plant_context, skip_if_welded=False):
    """
    Sets initial pose for paddles (free bodies only).
    If skip_if_welded=True, skip initialization for welded paddles.
    """
    # Right Paddle
    try:
        paddle_model = plant.GetModelInstanceByName("right_paddle")
        paddle_body = plant.GetBodyByName("paddle_body_link", paddle_model)
        
        # Check if body is free before setting pose
        if paddle_body.is_floating_base_body():
            plant.SetFreeBodyPose(
                plant_context, 
                paddle_body, 
                RigidTransform([0.4, 0.0, 0.15])  # Right side of table
            )
        elif not skip_if_welded:
            print(f"[WARNING] Could not initialize right paddle: Body 'paddle_body_link' is not a free body.")
    except RuntimeError as e:
        if not skip_if_welded:
            print(f"[ERROR] Failed to initialize right paddle: {e}")
        
    # Left Paddle (only exists in 2-arm mode)
    try:
        paddle_model = plant.GetModelInstanceByName("left_paddle")
        paddle_body = plant.GetBodyByName("paddle_body_link", paddle_model)
        
        # Check if body is free before setting pose
        if paddle_body.is_floating_base_body():
            plant.SetFreeBodyPose(
                plant_context, 
                paddle_body, 
                RigidTransform([-0.4, 0.0, 0.15])  # Left side of table
            )
        elif not skip_if_welded:
            print(f"[WARNING] Could not initialize left paddle: Body 'paddle_body_link' is not a free body.")
    except RuntimeError:
        # Expected in single-arm mode
        pass

def initialize_robots(plant, plant_context, num_arms=2):
    """
    Initialize robot joint positions so grippers hold paddles resting on table.
    """
    # Right robot - position so paddle base rests flat on table
    # With paddle welded at [0, 0.08, -0.05] and paddle head at z=0 in paddle frame,
    # we need gripper at appropriate height
    q_right = np.array([
        0.0,     # J1: base rotation
        -np.pi/3,     # J2: shoulder - angled to reach table
        0.0,     # J3: elbow rotation
        np.pi/2,    # J4: elbow bend
        0.0,     # J5: wrist rotation
        np.pi/3,     # J6: wrist bend to align paddle with table
        0.0      # J7: wrist rotation
    ])
    
    try:
        right_iiwa = plant.GetModelInstanceByName("right_iiwa")
        plant.SetPositions(plant_context, right_iiwa, q_right)
        
        # Gripper fingers are now WELDED (fixed joints) - no position setting needed!
    except Exception as e:
        print(f"[WARNING] Could not set right robot position: {e}")
    
    if num_arms == 2:
        # Left robot - mirror configuration
        q_left = np.array([
            0.0,   # J1: base rotation (facing opposite)
            -np.pi/3,     # J2: shoulder
            0.0,     # J3: elbow rotation
            np.pi/2,    # J4: elbow bend
            0.0,     # J5: wrist rotation
            np.pi/3,     # J6: wrist bend
            0.0      # J7: wrist rotation
        ])
        
        try:
            left_iiwa = plant.GetModelInstanceByName("left_iiwa")
            plant.SetPositions(plant_context, left_iiwa, q_left)
            
            # Gripper fingers are now WELDED (fixed joints) - no position setting needed!
        except Exception as e:
            print(f"[WARNING] Could not set left robot position: {e}")

def create_air_hockey_simulation(
    num_arms: int = 2,
    time_step: float = 0.001,
    use_meshcat: bool = True,
    skip_grasp: bool = False,  # Default to free paddles on table
    game_mode: str = "tournament",
    robot_side: str = "right"
):
    """
    Creates the Air Hockey simulation environment.
    
    Returns:
        simulator: Drake Simulator
        meshcat: Meshcat instance
        plant: MultibodyPlant
        diagram: Diagram
    """
    
    builder = DiagramBuilder()
    meshcat = None
    if use_meshcat:
        meshcat = StartMeshcat()
    
    # Use hydroelastic contact model for stable physics
    plant, scene_graph = AddMultibodyPlant(
        MultibodyPlantConfig(
            time_step=time_step,
            contact_model="hydroelastic_with_fallback",
            contact_surface_representation="polygon",
        ),
        builder,
    )
    
    # Disable gravity for 2D air hockey physics
    plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    
    parser = Parser(plant)
    
    # Build scenario from YAML
    generate_scenario_yaml(num_arms=num_arms, save_file=True, weld_paddles=skip_grasp)
    
    # Path to saved file
    import os
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filename = "single_arm.yaml" if num_arms == 1 else "two_arm.yaml"
    scenario_path = os.path.join(repo_dir, "scripts", "env", "scenario", filename)
    
    print(f"[INFO] Loading scenario from: {scenario_path}")
        
    directives = LoadModelDirectives(scenario_path)
    ProcessModelDirectives(directives, plant, parser)
    
    plant.Finalize()
    
    # ========================================
    # COLLISION FILTERING (inspired by MuJoCo air_hockey_challenge)
    # ========================================
    # Exclude puck-table surface collisions to achieve frictionless motion
    # Similar to MuJoCo's: <exclude body1="puck" body2="table_surface"/>
    try:
        from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
        
        # Get puck body and its geometries
        puck_model = plant.GetModelInstanceByName("puck")
        puck_body = plant.GetBodyByName("puck_body_link", puck_model)
        puck_geom_ids = plant.GetCollisionGeometriesForBody(puck_body)
        
        # Get table body and its geometries
        table_model = plant.GetModelInstanceByName("air_hockey_table")
        table_body = plant.GetBodyByName("table_body", table_model)
        table_geom_ids = plant.GetCollisionGeometriesForBody(table_body)
        
        if len(puck_geom_ids) > 0 and len(table_geom_ids) > 0:
            # Create filter declaration to exclude puck-table collisions
            puck_set = GeometrySet(puck_geom_ids)
            table_set = GeometrySet(table_geom_ids)
            
            filter_decl = CollisionFilterDeclaration()
            filter_decl.ExcludeBetween(puck_set, table_set)
            
            # Apply to SceneGraph
            scene_graph.collision_filter_manager().Apply(filter_decl)
            
            print("[INFO] Applied collision filter: puck-table surface collisions excluded")
            print(f"      Puck geometries: {len(puck_geom_ids)}, Table geometries: {len(table_geom_ids)}")
        else:
            print(f"[WARNING] Could not apply collision filter (puck geoms: {len(puck_geom_ids)}, table geoms: {len(table_geom_ids)})")
        
        # Also filter out puck collisions with robot ARM (not paddle!)
        # Only the PADDLE should hit the puck, not the arm links
        try:
            # Get robot models
            robot_models_to_filter = []
            try:
                robot_models_to_filter.append(plant.GetModelInstanceByName("right_iiwa"))
            except:
                pass
            try:
                robot_models_to_filter.append(plant.GetModelInstanceByName("left_iiwa"))
            except:
                pass
            
            for robot_model in robot_models_to_filter:
                # Get all robot arm link geometries (excluding paddle/gripper)
                robot_geom_ids = []
                for body_index in plant.GetBodyIndices(robot_model):
                    body = plant.get_body(body_index)
                    body_name = body.name()
                    
                    # EXCLUDE paddle - we WANT puck-paddle collisions!
                    # Only filter out the ARM links
                    if "iiwa_link" in body_name:  # Only arm links, not gripper/paddle
                        geoms = plant.GetCollisionGeometriesForBody(body)
                        robot_geom_ids.extend(geoms)
                
                if len(robot_geom_ids) > 0 and len(puck_geom_ids) > 0:
                    arm_set = GeometrySet(robot_geom_ids)
                    puck_set = GeometrySet(puck_geom_ids)
                    
                    filter_decl = CollisionFilterDeclaration()
                    filter_decl.ExcludeBetween(arm_set, puck_set)
                    scene_graph.collision_filter_manager().Apply(filter_decl)
                    
                    # Get model name for logging
                    model_name = "right_iiwa" if robot_model == robot_models_to_filter[0] else "left_iiwa"
                    print(f"[INFO] Filtered puck-arm collisions for {model_name}: {len(robot_geom_ids)} arm links")
        except Exception as e:
            print(f"[WARNING] Could not filter robot arm collisions: {e}")
        
        # ========================================================================
        # FILTER ARM-TABLE COLLISIONS (CRITICAL for level paddle!)
        # ========================================================================
        # The robot arm (especially elbow) can hit the table, preventing level paddle
        # Filter these collisions so IK can achieve flat orientation
        try:
            table_geom_ids = []
            try:
                table_model = plant.GetModelInstanceByName("air_hockey_table")
                for body_index in plant.GetBodyIndices(table_model):
                    geoms = plant.GetCollisionGeometriesForBody(plant.get_body(body_index))
                    table_geom_ids.extend(geoms)
            except:
                pass
            
            robot_arm_geom_ids = []
            for robot_model in robot_models_to_filter:
                for body_index in plant.GetBodyIndices(robot_model):
                    body = plant.get_body(body_index)
                    body_name = body.name()
                    # ONLY filter ARM links, NOT paddle/gripper
                    if "iiwa_link" in body_name:
                        geoms = plant.GetCollisionGeometriesForBody(body)
                        robot_arm_geom_ids.extend(geoms)
            
            if len(table_geom_ids) > 0 and len(robot_arm_geom_ids) > 0:
                arm_set = GeometrySet(robot_arm_geom_ids)
                table_set = GeometrySet(table_geom_ids)
                
                filter_decl = CollisionFilterDeclaration()
                filter_decl.ExcludeBetween(arm_set, table_set)
                scene_graph.collision_filter_manager().Apply(filter_decl)
                
                print(f"[INFO] Filtered arm-table collisions: {len(robot_arm_geom_ids)} arm links")
        except Exception as e:
            print(f"[WARNING] Could not filter arm-table collisions: {e}")
            
    except Exception as e:
        print(f"[WARNING] Failed to set up collision filtering: {e}")
    # ========================================
    
    # Game scoring system
    from scripts.env.game_systems import GameScoreSystem
    
    try:
        puck_model = plant.GetModelInstanceByName("puck")
        score_system = builder.AddSystem(GameScoreSystem(plant, puck_model))
        builder.Connect(
            plant.get_body_poses_output_port(),
            score_system.get_input_port(0)
        )
        builder.ExportOutput(score_system.get_output_port(0), "score")
    except Exception as e:
        print(f"[WARNING] Could not set up game systems: {e}")

    if use_meshcat:
        AddDefaultVisualization(builder, meshcat)
        print("[INFO] Waiting for Meshcat to initialize...")
        import time
        time.sleep(2.0)  # Give Meshcat time to load
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    
    # Simplified simulator configuration
    simulator.set_target_realtime_rate(1.0)
    
    # --- CRITICAL: Set initial positions BEFORE Initialize() ---
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    
    # Initialize puck (free body at center of table)
    try:
        puck_model = plant.GetModelInstanceByName("puck")
        puck_body = plant.GetBodyByName("puck_body_link", puck_model)
        
        # Set POSITION before Initialize (at table surface)
        plant.SetFreeBodyPose(
            plant_context,
            puck_body,
            RigidTransform([0.0, 0.0, 0.11])  # Table surface height
        )
    except Exception as e:
        print(f"[WARNING] Could not initialize puck position: {e}")
    
    # Initialize paddles (skip warning if welded in skip-grasp mode)
    initialize_paddles(plant, plant_context, skip_if_welded=skip_grasp)
    
    # Initialize robots to home positions
    initialize_robots(plant, plant_context, num_arms)
    
    simulator.Initialize()
    
    # Set puck POSITION and VELOCITY based on game mode
    try:
        from scripts.env.game_modes import PuckInitializer
        
        puck_model = plant.GetModelInstanceByName("puck")
        puck_body = plant.GetBodyByName("puck_body_link", puck_model)
        context = simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)
        
        # Get mode-specific initialization
        initializer = PuckInitializer(
            mode=game_mode,
            robot_side=robot_side,
            table_bounds={'x': (-1.0, 1.0), 'y': (-0.52, 0.52)}
        )
        puck_position, velocity = initializer.get_initial_state()
        
        # Set position
        plant.SetFreeBodyPose(
            plant_context,
            puck_body,
            RigidTransform(puck_position)
        )
        
        # Set velocity
        v = np.concatenate([np.zeros(3), velocity])  # [wx, wy, wz, vx, vy, vz]
        plant.SetFreeBodySpatialVelocity(
            plant_context,
            puck_body,
            SpatialVelocity(v)
        )
        
        # CRITICAL: Constrain puck to stay flat on table
        # Lock roll and pitch (only allow yaw rotation and XY translation)
        # This prevents puck from tipping/rotating when hit
        from pydrake.multibody.tree import QuaternionFloatingJoint, JointIndex
        
        # Get the puck's floating joint
        puck_joint = None
        for joint_index in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(joint_index))  # Convert int to JointIndex
            if "puck" in joint.name().lower() or joint.child_body() == puck_body:
                puck_joint = joint
                break
        
        if puck_joint is not None and isinstance(puck_joint, QuaternionFloatingJoint):
            # Lock the quaternion to upright orientation (no roll/pitch)
            # We'll do this by setting damping on angular velocities
            pass  # Damping is set via inertia in SDF
        
        # Alternative: Add constraint in simulator update loop
        # For now, rely on high rotational inertia in puck model
        
        # Validation: verify puck state was set correctly
        actual_pose = plant.EvalBodyPoseInWorld(plant_context, puck_body)
        actual_vel = plant.EvalBodySpatialVelocityInWorld(plant_context, puck_body)
        actual_pos = actual_pose.translation()
        actual_v = actual_vel.translational()
        
        # Mode-aware validation
        if game_mode == "hit":
            # Hit mode should be stationary
            assert np.linalg.norm(actual_v[:2]) < 0.01, f"Hit mode puck should be stationary: {actual_v[:2]}"
        # Don't enforce minimum velocity for other modes in case puck was stopped
        
        assert abs(actual_pos[2] - 0.11) < 0.02, f"Puck Z={actual_pos[2]:.3f}, expected ~0.11"
        
    except Exception as e:
        print(f"[ERROR] Could not initialize puck: {e}")
        import traceback
        traceback.print_exc()
    
    return simulator, meshcat, plant, diagram
