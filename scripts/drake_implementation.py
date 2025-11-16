import numpy as np
import random
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    SceneGraph,
    Parser,
    Simulator,
    MeshcatVisualizer,
    StartMeshcat,
    RigidTransform,
    RollPitchYaw,
    Sphere,
    Box,
    Cylinder,
    ProximityProperties,
    CoulombFriction,
    AddMultibodyPlantSceneGraph,
    UnitInertia,
    SpatialInertia,
    SpatialVelocity,
    RevoluteJoint,
    PrismaticJoint,
    BallRpyJoint,
    LoadModelDirectives,
    ProcessModelDirectives,
)
import os
import sys

try:
    import air_hockey_challenge
    AHC_PACKAGE_PATH = os.path.dirname(air_hockey_challenge.__file__)
    DATA_PATH = os.path.join(AHC_PACKAGE_PATH, "environments", "data")
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(script_dir, "..", "air_hockey_challenge", 
                            "air_hockey_challenge", "environments", "data")
    if not os.path.exists(DATA_PATH):
        DATA_PATH = None
        print("Warning: Could not find air_hockey_challenge package. Using default models.")


class AirHockeyDrakeEnv:
    
    def __init__(self, time_step=0.001, use_meshcat=True):
        self.time_step = time_step
        self.builder = DiagramBuilder()
        
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=time_step
        )
        
        self.plant.mutable_gravity_field().set_gravity_vector([0, 0, -9.81])
        
        if use_meshcat:
            self.meshcat = StartMeshcat()
            self.visualizer = MeshcatVisualizer.AddToBuilder(
                self.builder, self.scene_graph, self.meshcat
            )
        
        self.parser = Parser(self.plant)
        
        self._add_base_table()
        self._add_table()
        self._add_robots()
        self._add_puck()
        
        self.puck_damping_linear = 0.005
        self.puck_damping_angular = 2e-6
        
        self.plant.Finalize()
        
        self.diagram = self.builder.Build()
        
        self.simulator = Simulator(self.diagram)
        self.context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        
        print("Drake Air Hockey Environment initialized")
        print(f"Number of bodies: {self.plant.num_bodies()}")
        print(f"Number of positions: {self.plant.num_positions()}")
        print(f"Number of velocities: {self.plant.num_velocities()}")
        
        if use_meshcat:
            meshcat_url = self.meshcat.web_url()
            print(f"\nMeshcat visualization available at: {meshcat_url}")
            print("Open this URL in your browser to view the simulation")
    
    def _add_base_table(self):
        base_length = 4.0
        base_width = 2.0
        base_height = 0.1
        
        base_shape = Box(base_length, base_width, base_height)
        base_props = ProximityProperties()
        base_props.AddProperty("material", "coulomb_friction", 
                              CoulombFriction(0.5, 0.3))
        
        X_base = RigidTransform(p=[0, 0, -base_height/2])
        
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(), X_base, base_shape, "base_table_visual",
            [0.6, 0.4, 0.2, 1.0]
        )
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(), X_base, base_shape, "base_table_collision",
            base_props
        )
    
    def _add_table(self):
        table_length = 1.064 * 2
        table_width = 0.609 * 2
        table_height = 0.0505 * 2
        
        self.table_height = table_height
        
        table_shape = Box(table_length, table_width, table_height)
        table_props = ProximityProperties()
        table_props.AddProperty("material", "coulomb_friction", 
                               CoulombFriction(0.01, 0.005))
        
        X_WT = RigidTransform(p=[0, 0, table_height/2])
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(), X_WT, table_shape, "table_surface_visual",
            [1.0, 1.0, 1.0, 1.0]
        )
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(), X_WT, table_shape, "table_surface_collision",
            table_props
        )
        
        rim_height = 0.015
        rim_thickness = 0.045
        
        wall_left = Box(table_length, rim_thickness, rim_height)
        X_wall_left = RigidTransform(p=[0, table_width/2 + rim_thickness/2, table_height + rim_height/2])
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(), X_wall_left, wall_left, "rim_left_visual",
            [0.8, 0.8, 0.8, 1.0]
        )
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(), X_wall_left, wall_left, "rim_left_collision",
            ProximityProperties()
        )
        
        wall_right = Box(table_length, rim_thickness, rim_height)
        X_wall_right = RigidTransform(p=[0, -table_width/2 - rim_thickness/2, table_height + rim_height/2])
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(), X_wall_right, wall_right, "rim_right_visual",
            [0.8, 0.8, 0.8, 1.0]
        )
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(), X_wall_right, wall_right, "rim_right_collision",
            ProximityProperties()
        )
        
        goal_home = Box(rim_thickness, 0.394, rim_height)
        X_goal_home = RigidTransform(p=[-table_length/2 - rim_thickness/2, 0, table_height + rim_height/2])
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(), X_goal_home, goal_home, "rim_home_visual",
            [0.8, 0.8, 0.8, 1.0]
        )
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(), X_goal_home, goal_home, "rim_home_collision",
            ProximityProperties()
        )
        
        goal_away = Box(rim_thickness, 0.394, rim_height)
        X_goal_away = RigidTransform(p=[table_length/2 + rim_thickness/2, 0, table_height + rim_height/2])
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(), X_goal_away, goal_away, "rim_away_visual",
            [0.8, 0.8, 0.8, 1.0]
        )
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(), X_goal_away, goal_away, "rim_away_collision",
            ProximityProperties()
        )
    
    def _add_puck(self):
        puck_model_instance = self.plant.AddModelInstance("puck_model")
        
        puck_radius = 0.03165
        puck_height = 0.01
        
        puck_shape = Cylinder(puck_radius, puck_height)
        
        puck_props = ProximityProperties()
        puck_props.AddProperty("material", "coulomb_friction",
                              CoulombFriction(0.01, 0.005))
        
        puck_mass = 0.01
        puck_inertia = np.diag([2.5e-6, 2.5e-6, 5e-6])
        
        self.puck_body = self.plant.AddRigidBody(
            model_instance=puck_model_instance,
            name="puck",
            M_BBo_B=SpatialInertia(
                mass=puck_mass,
                p_PScm_E=[0, 0, 0],
                G_SP_E=UnitInertia(
                    Ixx=puck_inertia[0, 0] / puck_mass,
                    Iyy=puck_inertia[1, 1] / puck_mass,
                    Izz=puck_inertia[2, 2] / puck_mass
                )
            )
        )
        
        self.plant.RegisterVisualGeometry(
            self.puck_body, RigidTransform(), puck_shape, "puck_visual",
            [1.0, 0.0, 0.0, 1.0]
        )
        self.plant.RegisterCollisionGeometry(
            self.puck_body, RigidTransform(), puck_shape, "puck_collision",
            puck_props
        )
    
    def _add_robot_from_specs(self, robot_name, base_position):
        link_masses = [
            0.0, 8.240527, 6.357896, 4.042756, 3.642249, 2.580896, 2.760564, 1.285417,
        ]
        
        link_positions = [
            [0, 0, 0], [0, 0, 0.1575], [0, 0, 0.2025], [0, 0.2045, 0],
            [0, 0, 0.2155], [0, 0.1845, 0], [0, 0, 0.2155], [0, 0.081, 0],
        ]
        
        joint_axes = [
            None, [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        ]
        
        joint_damping = [
            0.0, 0.33032, 0.21216, 0.1, 0.219041, 0.185923, 0.1, 0.1,
        ]
        
        joint_limits = [
            None, (-2.96706, 2.96706), (-2.0944, 2.0944), (-2.96706, 2.96706),
            (-2.0944, 2.0944), (-2.96706, 2.96706), (-2.0944, 2.0944), (-3.05433, 3.05433),
        ]
        
        base_body = self.plant.AddRigidBody(
            name=f"{robot_name}/base",
            M_BBo_B=SpatialInertia(mass=1.0, p_PScm_E=[0, 0, 0],
                                  G_SP_E=UnitInertia(1.0, 1.0, 1.0))
        )
        
        X_base = RigidTransform(p=base_position)
        self.plant.WeldFrames(
            self.plant.world_frame(),
            base_body.body_frame(),
            X_base
        )
        
        base_shape = Box(0.2, 0.2, 0.2)
        self.plant.RegisterVisualGeometry(
            base_body, RigidTransform(), base_shape, f"{robot_name}/base_visual",
            [0.4, 0.4, 0.4, 1.0]
        )
        
        bodies = [base_body]
        joints = [None]
        
        parent_body = base_body
        for i in range(1, 8):
            link_body = self.plant.AddRigidBody(
                name=f"{robot_name}/link_{i}",
                M_BBo_B=SpatialInertia(
                    mass=link_masses[i],
                    p_PScm_E=[0, 0, 0],
                    G_SP_E=UnitInertia(1.0, 1.0, 1.0)
                )
            )
            
            X_parent = RigidTransform(p=link_positions[i])
            if joint_limits[i] is not None:
                joint = self.plant.AddJoint(
                    RevoluteJoint(
                        name=f"{robot_name}/joint_{i}",
                        frame_on_parent=parent_body.body_frame(),
                        frame_on_child=link_body.body_frame(),
                        axis=np.array(joint_axes[i]),
                        pos_lower_limit=joint_limits[i][0],
                        pos_upper_limit=joint_limits[i][1],
                        damping=joint_damping[i]
                    )
                )
            else:
                joint = self.plant.AddJoint(
                    RevoluteJoint(
                        name=f"{robot_name}/joint_{i}",
                        frame_on_parent=parent_body.body_frame(),
                        frame_on_child=link_body.body_frame(),
                        axis=np.array(joint_axes[i]),
                        damping=joint_damping[i]
                    )
                )
            
            link_shape = Cylinder(0.05, 0.1)
            self.plant.RegisterVisualGeometry(
                link_body, RigidTransform(), link_shape, f"{robot_name}/link_{i}_visual",
                [1.0, 0.423529, 0.0392157, 1.0]
            )
            self.plant.RegisterCollisionGeometry(
                link_body, RigidTransform(), link_shape, f"{robot_name}/link_{i}_collision",
                ProximityProperties()
            )
            
            bodies.append(link_body)
            joints.append(joint)
            parent_body = link_body
        
        striker_offset = self.plant.AddRigidBody(
            name=f"{robot_name}/striker_offset",
            M_BBo_B=SpatialInertia(mass=0.0, p_PScm_E=[0, 0, 0],
                                  G_SP_E=UnitInertia(0.0, 0.0, 0.0))
        )
        
        X_striker = RigidTransform(p=[0, 0, 0.585])
        self.plant.WeldFrames(
            parent_body.body_frame(),
            striker_offset.body_frame(),
            X_striker
        )
        
        striker_link = self.plant.AddRigidBody(
            name=f"{robot_name}/striker_joint_link",
            M_BBo_B=SpatialInertia(mass=0.1, p_PScm_E=[0, 0, 0],
                                  G_SP_E=UnitInertia(0.01, 0.01, 0.01))
        )
        
        striker_joint = self.plant.AddJoint(
            RevoluteJoint(
                name=f"{robot_name}/striker_joint_1",
                frame_on_parent=striker_offset.body_frame(),
                frame_on_child=striker_link.body_frame(),
                axis=np.array([0, 1, 0]),
                pos_lower_limit=-1.5708,
                pos_upper_limit=1.5708,
                damping=0.0
            )
        )
        
        striker_mallet = self.plant.AddRigidBody(
            name=f"{robot_name}/striker_mallet",
            M_BBo_B=SpatialInertia(mass=0.283, p_PScm_E=[0, 0, 0.0682827],
                                  G_SP_E=UnitInertia(0.0177, 0.0177, 0.0177))
        )
        
        X_mallet = RigidTransform()
        mallet_joint = self.plant.AddJoint(
            RevoluteJoint(
                name=f"{robot_name}/striker_joint_2",
                frame_on_parent=striker_link.body_frame(),
                frame_on_child=striker_mallet.body_frame(),
                axis=np.array([1, 0, 0]),
                pos_lower_limit=-1.5708,
                pos_upper_limit=1.5708,
                damping=0.0
            )
        )
        
        mallet_shape = Cylinder(0.04815, 0.06)
        self.plant.RegisterVisualGeometry(
            striker_mallet, RigidTransform(p=[0, 0, 0.0505]), mallet_shape,
            f"{robot_name}/striker_mallet_visual",
            [0.3, 0.3, 0.3, 1.0]
        )
        self.plant.RegisterCollisionGeometry(
            striker_mallet, RigidTransform(p=[0, 0, 0.0505]), mallet_shape,
            f"{robot_name}/striker_mallet_collision",
            ProximityProperties()
        )
        
        return bodies, joints
    
    def _add_robots(self):
        try:
            robot_yaml = """
directives:
- add_model:
    name: iiwa_robot1
    file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.5]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa_robot1::iiwa_link_0
    X_PC:
        translation: [-1.51, 0, 0.0]
        rotation: !Rpy { deg: [0, 0, 180] }
- add_model:
    name: iiwa_robot2
    file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.5]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa_robot2::iiwa_link_0
    X_PC:
        translation: [1.51, 0, 0.0]
        rotation: !Rpy { deg: [0, 0, 0] }
"""
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(robot_yaml)
                yaml_file = f.name
            
            directives = LoadModelDirectives(yaml_file)
            ProcessModelDirectives(directives, self.plant, self.parser)
            
            os.unlink(yaml_file)
            
            self.robot1_model = self.plant.GetModelInstanceByName("iiwa_robot1")
            self.robot2_model = self.plant.GetModelInstanceByName("iiwa_robot2")
            
            try:
                ee_frame_1 = self.plant.GetFrameByName("iiwa_link_7", self.robot1_model)
                print(f"Found end effector frame for robot1: {ee_frame_1.name()}")
                self._add_gripper_and_paddle_to_robot(ee_frame_1, "robot1", self.robot1_model)
                
                ee_frame_2 = self.plant.GetFrameByName("iiwa_link_7", self.robot2_model)
                print(f"Found end effector frame for robot2: {ee_frame_2.name()}")
                self._add_gripper_and_paddle_to_robot(ee_frame_2, "robot2", self.robot2_model)
                
                print("Successfully loaded KUKA iiwa robots from Drake models with end effectors")
            except Exception as ee_error:
                print(f"Warning: Could not add end effectors: {ee_error}")
                import traceback
                traceback.print_exc()
                print("Robots loaded but without end effectors")
            
        except Exception as e:
            print(f"Could not load robot models: {e}")
            print("Continuing without robots - table and puck only")
            self.robot1_model = None
            self.robot2_model = None
    
    def _add_gripper_and_paddle_to_robot(self, parent_frame, robot_name, model_instance):
        tool_body = self.plant.AddRigidBody(
            model_instance=model_instance,
            name=f"{robot_name}/tool_body",
            M_BBo_B=SpatialInertia(
                mass=0.5,
                p_PScm_E=[0, 0, 0.05],
                G_SP_E=UnitInertia(0.02, 0.02, 0.02)
            )
        )
        
        X_weld = RigidTransform(p=[0, 0, 0.126])
        self.plant.WeldFrames(
            parent_frame,
            tool_body.body_frame(),
            X_weld
        )
        
        paddle_radius = 0.04815
        paddle_thickness = 0.01
        handle_radius = 0.015
        handle_length = 0.08
        
        paddle_props = ProximityProperties()
        paddle_props.AddProperty("material", "coulomb_friction",
                               CoulombFriction(0.3, 0.2))
        
        paddle_shape = Cylinder(paddle_radius, paddle_thickness)
        X_surface = RigidTransform(p=[0, 0, paddle_thickness / 2.0])
        self.plant.RegisterVisualGeometry(
            tool_body, X_surface, paddle_shape,
            f"{robot_name}/paddle_surface_visual",
            [0.2, 0.6, 1.0, 1.0]
        )
        self.plant.RegisterCollisionGeometry(
            tool_body, X_surface, paddle_shape,
            f"{robot_name}/paddle_surface_collision",
            paddle_props
        )
        
        handle_shape = Cylinder(handle_radius, handle_length)
        R_handle = RollPitchYaw(0, np.pi/2, 0)
        handle_z_center = paddle_thickness + handle_radius
        X_handle = RigidTransform(R_handle, p=[0, 0, handle_z_center])
        
        self.plant.RegisterVisualGeometry(
            tool_body, X_handle, handle_shape,
            f"{robot_name}/paddle_handle_visual",
            [0.8, 0.2, 0.2, 1.0]
        )
        self.plant.RegisterCollisionGeometry(
            tool_body, X_handle, handle_shape,
            f"{robot_name}/paddle_handle_collision",
            paddle_props
        )

        gripper_props = ProximityProperties()
        gripper_props.AddProperty("material", "coulomb_friction",
                                 CoulombFriction(0.5, 0.3))
        gripper_color = [0.5, 0.5, 0.5, 1.0]

        finger_height = 0.02
        finger_shape = Box(handle_length * 0.8, handle_length, finger_height)
        
        z_top_finger = handle_z_center + handle_radius + finger_height / 2.0
        z_bottom_finger = handle_z_center - handle_radius - finger_height / 2.0
        
        X_finger_top = RigidTransform(p=[0, 0, z_top_finger])
        X_finger_bottom = RigidTransform(p=[0, 0, z_bottom_finger])

        self.plant.RegisterVisualGeometry(
            tool_body, X_finger_top, finger_shape,
            f"{robot_name}/gripper_finger_top_visual", gripper_color
        )
        self.plant.RegisterCollisionGeometry(
            tool_body, X_finger_top, finger_shape,
            f"{robot_name}/gripper_finger_top_collision", gripper_props
        )
        self.plant.RegisterVisualGeometry(
            tool_body, X_finger_bottom, finger_shape,
            f"{robot_name}/gripper_finger_bottom_visual", gripper_color
        )
        self.plant.RegisterCollisionGeometry(
            tool_body, X_finger_bottom, finger_shape,
            f"{robot_name}/gripper_finger_bottom_collision", gripper_props
        )

        base_height = (z_top_finger + finger_height/2.0) - (z_bottom_finger - finger_height/2.0)
        base_shape = Box(0.03, handle_length, base_height)
        x_base = -(handle_length * 0.8 / 2.0) - (0.03 / 2.0)
        X_base = RigidTransform(p=[x_base, 0, handle_z_center])
        
        self.plant.RegisterVisualGeometry(
            tool_body, X_base, base_shape,
            f"{robot_name}/gripper_base_visual", gripper_color
        )
        self.plant.RegisterCollisionGeometry(
            tool_body, X_base, base_shape,
            f"{robot_name}/gripper_base_collision", gripper_props
        )
    
    def _generate_random_puck_velocity(self):
        speed = random.uniform(0.5, 2.0)
        
        max_attempts = 100
        for _ in range(max_attempts):
            angle = random.uniform(0, 2 * np.pi)
            angle_deg = np.degrees(angle)
            
            avoid_angles = [0, 45, 90, 135, 180, 225, 270, 315]
            tolerance = 20
            
            too_close = False
            for avoid_angle in avoid_angles:
                diff = abs(angle_deg - avoid_angle)
                if diff < tolerance or diff > (360 - tolerance):
                    too_close = True
                    break
            
            if not too_close:
                break
        
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        vz = 0.0
        
        return [vx, vy, vz]
    
    def reset(self, puck_pos=None, puck_vel=None, random_velocity=False):
        self.simulator.Initialize()
        self.context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        
        if puck_pos is None:
            puck_pos = [0.0, 0.0, self.table_height + 0.01]
        if puck_vel is None:
            if random_velocity:
                puck_vel = self._generate_random_puck_velocity()
            else:
                puck_vel = [0.0, 0.0, 0.0]
        
        puck_pose = RigidTransform(p=puck_pos)
        self.plant.SetFreeBodyPose(self.plant_context, self.puck_body, puck_pose)
        
        puck_spatial_vel = SpatialVelocity(
            w=np.array([0.0, 0.0, 0.0]),
            v=np.array(puck_vel)
        )
        self.plant.SetFreeBodySpatialVelocity(
            self.plant_context,
            self.puck_body,
            puck_spatial_vel
        )
        
        if hasattr(self, 'robot1_model') and self.robot1_model is not None:
            try:
                q_home = np.array([-1.57, 0.5, 0.0, -1.2, 0.0, 1.6, 0.0])
                from pydrake.multibody.plant import ModelInstanceIndex
                if isinstance(self.robot1_model, ModelInstanceIndex):
                    self.plant.SetPositions(self.plant_context, self.robot1_model, q_home)
                    self.plant.SetPositions(self.plant_context, self.robot2_model, q_home)
            except Exception as e:
                print(f"Warning: Could not set robot joint positions: {e}")
                pass
        
        self.diagram.ForcedPublish(self.context)
    
    def step(self, action=None, duration=None):
        if duration is None:
            duration = self.time_step
        
        self.simulator.AdvanceTo(self.simulator.get_context().get_time() + duration)
    
    def get_puck_state(self):
        pose = self.plant.GetFreeBodyPose(self.plant_context, self.puck_body)
        
        spatial_vel = self.plant.EvalBodySpatialVelocityInWorld(
            self.plant_context, self.puck_body
        )
        
        return pose.translation(), spatial_vel.translational()
    
    def _apply_puck_damping(self):
        spatial_vel = self.plant.EvalBodySpatialVelocityInWorld(
            self.plant_context, self.puck_body
        )
        
        v_trans = spatial_vel.translational()
        v_rot = spatial_vel.rotational()
        
        v_trans_damped = v_trans * (1.0 - self.puck_damping_linear * self.time_step)
        
        v_rot_damped = v_rot.copy()
        v_rot_damped[2] = v_rot[2] * (1.0 - self.puck_damping_angular * self.time_step)
        
        from pydrake.multibody.math import SpatialVelocity
        damped_spatial_vel = SpatialVelocity(
            w=v_rot_damped,
            v=v_trans_damped
        )
        
        self.plant.SetFreeBodySpatialVelocity(
            self.plant_context,
            self.puck_body,
            damped_spatial_vel
        )
    
    def run_simulation(self, duration=5.0):
        self.diagram.ForcedPublish(self.context)
        
        final_time = self.simulator.get_context().get_time() + duration
        step_size = 0.01
        
        current_time = self.simulator.get_context().get_time()
        while current_time < final_time:
            next_time = min(current_time + step_size, final_time)
            
            self._apply_puck_damping()
            
            self.simulator.AdvanceTo(next_time)
            current_time = self.simulator.get_context().get_time()
            
            self.diagram.ForcedPublish(self.context)
            
            if int(current_time) != int(current_time - step_size):
                print(f"  Simulation time: {current_time:.1f}s / {final_time:.1f}s")
        
        self.diagram.ForcedPublish(self.context)


def main():
    print("Creating Drake Air Hockey Environment...")
    
    env = AirHockeyDrakeEnv(time_step=0.001, use_meshcat=True)
    
    env.reset(puck_pos=[0.0, 0.0, env.table_height + 0.01], random_velocity=True)
    
    print("\nInitial puck state:")
    puck_pos, puck_vel = env.get_puck_state()
    print(f"Position: {puck_pos}")
    print(f"Velocity: {puck_vel}")
    
    print("\nRunning simulation for 10 seconds...")
    print("Watch the visualization in Meshcat!")
    print("The simulation will stay open for viewing...")
    
    env.run_simulation(duration=10.0)
    
    print("\nFinal puck state:")
    puck_pos, puck_vel = env.get_puck_state()
    print(f"Position: {puck_pos}")
    print(f"Velocity: {puck_vel}")
    
    print("\nSimulation complete! Meshcat visualization will remain open.")
    print("Press Ctrl+C to exit, or close the browser window.")
    
    try:
        import time
        print("Keeping simulation alive for 30 more seconds for viewing...")
        time.sleep(30)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()