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

def initialize_puck(simulator: Simulator, plant, puck_model, velocity=None):
    """
    Sets the initial state of the puck (free body).
    """
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    
    # Get the puck body
    puck_body = plant.GetBodyByName("puck_body_link", puck_model)
    
    # Set initial position (center of table, above surface)
    # Using 0.15 to be safe (table surface at ~0.10)
    
    initial_pose = RigidTransform(
        RotationMatrix(),
        [0.0, 0.0, 0.3] 
    )
    
    plant.SetFreeBodyPose(plant_context, puck_body, initial_pose)
    
    if velocity is not None:
        # velocity is [vx, vy, vz]
        # Set spatial velocity
        v = np.zeros(6)
        v[3:6] = velocity # translational part
        plant.SetFreeBodySpatialVelocity(
            plant_context, 
            puck_body, 
            SpatialVelocity(v)
        )

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
    Sets initial joint positions for robots to a safe 'home' configuration.
    Avoids singularities encountered at [0,0,...].
    """
    # Standard home config (bent arm)
def initialize_robots(plant, plant_context, num_arms):
    """
    Initialize robot joint positions so grippers are positioned around paddle stems.
    """
    # Right robot - position gripper around paddle at [0.4, 0, 0.15]
    # Joint config to reach paddle on right side of table
    q_right = np.array([
        0.0,     # J1: base rotation
        0.3,     # J2: shoulder
        0.0,     # J3: elbow rotation
        -1.0,    # J4: elbow bend
        0.0,     # J5: wrist rotation
        1.2,     # J6: wrist bend
        0.0      # J7: wrist rotation
    ])
    
    try:
        right_iiwa = plant.GetModelInstanceByName("right_iiwa")
        plant.SetPositions(plant_context, right_iiwa, q_right)
    except Exception as e:
        print(f"[WARNING] Could not set right robot position: {e}")
    
    if num_arms == 2:
        # Left robot - position gripper around paddle at [-0.4, 0, 0.15]
        # Mirror configuration for left side
        q_left = np.array([
            np.pi,   # J1: base rotation (facing opposite direction)
            0.3,     # J2: shoulder
            0.0,     # J3: elbow rotation
            -1.0,    # J4: elbow bend
            0.0,     # J5: wrist rotation
            1.2,     # J6: wrist bend
            0.0      # J7: wrist rotation
        ])
        
        try:
            left_iiwa = plant.GetModelInstanceByName("left_iiwa")
            plant.SetPositions(plant_context, left_iiwa, q_left)
        except Exception as e:
            print(f"[WARNING] Could not set left robot position: {e}")

def create_air_hockey_simulation(
    num_arms: int = 2,
    time_step: float = 0.001,
    use_meshcat: bool = True,
    skip_grasp: bool = False  # Default to free paddles on table
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
    
    # --- Game Systems (DISABLED for debugging visualization) ---
    from scripts.env.game_systems import PuckDragSystem, GameScoreSystem
    #
    try:
        puck_model = plant.GetModelInstanceByName("puck")
        
        # 1. Drag System
        drag_system = builder.AddSystem(PuckDragSystem(plant, puck_model))
        builder.Connect(
            plant.get_body_spatial_velocities_output_port(),
            drag_system.get_input_port(1)
        )
        builder.Connect(
            drag_system.get_output_port(0),
            plant.get_applied_spatial_force_input_port()
        )
        
        # 2. Scoring System (prints to terminal)
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
        initialize_puck(simulator, plant, puck_model, velocity=[0, 0, 0])
    except Exception as e:
        print(f"[WARNING] Could not initialize puck: {e}")
    
    # Initialize paddles (skip warning if welded in skip-grasp mode)
    initialize_paddles(plant, plant_context, skip_if_welded=skip_grasp)
    
    # Initialize robots to home positions
    initialize_robots(plant, plant_context, num_arms)
    
    simulator.Initialize()
    
    return simulator, meshcat, plant, diagram
