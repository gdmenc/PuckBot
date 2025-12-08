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

def initialize_paddles(plant, plant_context):
    """
    Sets initial pose for paddles (free bodies).
    """
    # Spawn paddles at user-specified positions (+/- 0.2, Z=0.15)
    
    # Right Paddle
    try:
        paddle_model = plant.GetModelInstanceByName("right_paddle")
        paddle_body = plant.GetBodyByName("paddle_body_link", paddle_model)
        plant.SetFreeBodyPose(
            plant_context, 
            paddle_body, 
            RigidTransform([0.0, 0.0, 0.0]) 
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize right paddle: {e}")
        
    # Left Paddle (only exists in 2-arm mode)
    try:
        paddle_model = plant.GetModelInstanceByName("left_paddle")
        paddle_body = plant.GetBodyByName("paddle_body_link", paddle_model)
        plant.SetFreeBodyPose(
            plant_context, 
            paddle_body, 
            RigidTransform([-0.6, 0.0, 0.20]) 
        )
    except RuntimeError:
        # Expected in single-arm mode
        pass

def initialize_robots(plant, plant_context, num_arms=2):
    """
    Sets initial joint positions for robots to a safe 'home' configuration.
    Avoids singularities encountered at [0,0,...].
    """
    # Standard home config (bent arm)
    # q = [-1.57, 0.5, 0.0, -1.2, 0.0, 1.6, 0.0]
    home_q = np.array([-1.57, 0.5, 0.0, -1.2, 0.0, 1.6, 0.0])
    
    # Right Arm
    try:
        model = plant.GetModelInstanceByName("right_iiwa")
        plant.SetPositions(plant_context, model, home_q)
    except Exception as e:
        print(f"[WARNING] Could not set home for right_iiwa: {e}")

    if num_arms == 2:
        # Left Arm (Mirrored? Or same? -1.57 is -90 deg. Left might need +1.57?)
        # Let's use mirror for joint 0 (base rotation).
        # -1.57 puts Right Arm (X-forward) to -Y side? 
        # Check frame: IIWA base Z-up. Joint 1 rotates around Z.
        # If Right Arm is at Y=0. 
        # Actually, let's just use the same bent shape. 
        # If we want symmetry, J1 might need sign flip. 
        # But for now, let's just use the same "bent up and forward" pose.
        # Wait, [-1.57 ...] rotates it 90 deg.
        # Let's use the provided home_q for both and see.
        try:
            model = plant.GetModelInstanceByName("left_iiwa")
            # Mirror J1? 
            home_q_left = home_q.copy()
            home_q_left[0] = 1.57 # Rotate 90 deg the other way (if facing each other?)
            plant.SetPositions(plant_context, model, home_q_left)
        except Exception as e:
            print(f"[WARNING] Could not set home for left_iiwa: {e}")

def create_air_hockey_simulation(
    num_arms: int = 2,
    time_step: float = 0.001,
    use_meshcat: bool = True,
    skip_grasp: bool = False
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
    
    # Puck position
    try:
        puck_model = plant.GetModelInstanceByName("puck")
        puck_body = plant.GetBodyByName("puck_body_link", puck_model)
        plant.SetFreeBodyPose(
            plant_context, 
            puck_body, 
            RigidTransform([0.0, 0.0, 0.15])  # Center of table, above surface
        )
    except Exception as e:
        print(f"[WARNING] Could not initialize puck: {e}")
    
    # Right paddle position (close to right robot)
    try:
        paddle_model = plant.GetModelInstanceByName("right_paddle")
        paddle_body = plant.GetBodyByName("paddle_body_link", paddle_model)
        plant.SetFreeBodyPose(
            plant_context, 
            paddle_body, 
            RigidTransform([0.6, 0.0, 0.15])  # Near right robot
        )
    except Exception as e:
        print(f"[WARNING] Could not initialize right paddle: {e}")
    
    # Left paddle position (close to left robot, only in 2-arm mode)
    try:
        paddle_model = plant.GetModelInstanceByName("left_paddle")
        paddle_body = plant.GetBodyByName("paddle_body_link", paddle_model)
        plant.SetFreeBodyPose(
            plant_context, 
            paddle_body, 
            RigidTransform([-0.6, 0.0, 0.15])  # Near left robot
        )
    except RuntimeError:
        pass  # Expected in single-arm mode
    
    # Robot home positions
    home_q = np.array([-1.57, 0.5, 0.0, -1.2, 0.0, 1.6, 0.0])
    try:
        right_iiwa = plant.GetModelInstanceByName("right_iiwa")
        plant.SetPositions(plant_context, right_iiwa, home_q)
    except:
        pass
    try:
        left_iiwa = plant.GetModelInstanceByName("left_iiwa")
        home_q_left = home_q.copy()
        home_q_left[0] = 1.57  # Mirror for left arm
        plant.SetPositions(plant_context, left_iiwa, home_q_left)
    except:
        pass
    
    simulator.Initialize()
    
    return simulator, meshcat, plant, diagram
