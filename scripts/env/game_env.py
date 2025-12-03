"""
Air Hockey Game Environment.

Provides a Drake simulation setup that includes a detailed air-hockey table,
robots, and a planar puck constrained to the table surface. Paddle support can
optionally be enabled for grasping experiments.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    Cylinder,
    CoulombFriction,
    DiagramBuilder,
    MeshcatVisualizer,
    Mesh,
    Parser,
    PrismaticJoint,
    ProximityProperties,
    RevoluteJoint,
    RigidTransform,
    RollPitchYaw,
    SpatialInertia,
    SpatialVelocity,
    StartMeshcat,
    UnitInertia,
    Simulator,
)


class AirHockeyGameEnv:
    """
    Standalone Drake environment that includes:
    - Air hockey table / puck / robots (same as original environment)
    - A side table that holds a paddle
    - A free paddle object that can be grasped and welded to a robot tool
    """

    def __init__(
        self,
        time_step: float = 0.001,
        use_meshcat: bool = True,
        include_paddle: bool = False,
    ):
        self.time_step = time_step
        self.builder = DiagramBuilder()
        self.include_paddle = include_paddle
        self.paddle_body = None
        self.paddle_grasped = False
        self.assets_root = self._resolve_assets_root()

        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=time_step
        )

        self.plant.mutable_gravity_field().set_gravity_vector([0, 0, -9.81])

        self.meshcat = None
        if use_meshcat:
            self.meshcat = StartMeshcat()
            MeshcatVisualizer.AddToBuilder(
                self.builder,
                self.scene_graph,
                self.meshcat,
            )

        self.parser = Parser(self.plant)

        # Base environment pieces
        self._add_base_table()
        self._add_table_surface()
        self._add_table_rims()
        self._add_robots()
        self._add_puck()

        # Additional assets for gameplay setup
        if self.include_paddle:
            self._add_side_table()
            self._add_paddle_object()

        self.puck_damping_linear = 0.005
        self.puck_damping_angular = 2e-6

        self.plant.Finalize()
        self.diagram = self.builder.Build()

        self.simulator = Simulator(self.diagram)
        self.context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        print("Air Hockey Game Environment initialized (extended version)")
        if use_meshcat and self.meshcat:
            print(f"Meshcat available at: {self.meshcat.web_url()}")

    def _resolve_assets_root(self) -> Path:
        """Return the directory containing local Drake/MuJoCo assets."""
        return Path(__file__).parent / "assets"

    # ----------------------------------------------------------------- helpers
    def _add_base_table(self) -> None:
        base_length = 4.0
        base_width = 2.0
        base_height = 0.1
        base_shape = Box(base_length, base_width, base_height)

        props = ProximityProperties()
        props.AddProperty("material", "coulomb_friction", CoulombFriction(0.5, 0.3))

        X_base = RigidTransform(p=[0, 0, -base_height / 2])

        self.plant.RegisterVisualGeometry(
            self.plant.world_body(),
            X_base,
            base_shape,
            "base_table_visual",
            [0.6, 0.4, 0.2, 1.0],
        )
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(),
            X_base,
            base_shape,
            "base_table_collision",
            props,
        )

    def _add_table_surface(self) -> None:
        table_length = 1.064 * 2
        table_width = 0.609 * 2
        table_thickness = 0.0505 * 2

        self.table_height = table_thickness

        table_shape = Box(table_length, table_width, table_thickness)

        props = ProximityProperties()
        props.AddProperty("material", "coulomb_friction", CoulombFriction(1e-5, 1e-5))

        X_WT = RigidTransform(p=[0, 0, table_thickness / 2])
        color = [0.97, 0.97, 0.99, 1.0]
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(),
            X_WT,
            table_shape,
            "table_surface_visual",
            color,
        )
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(),
            X_WT,
            table_shape,
            "table_surface_collision",
            props,
        )

        # Center lines / markings for improved visuals
        marking_shape = Box(table_length * 0.95, 0.01, 0.002)
        X_mark = RigidTransform(p=[0, 0, self.table_height + 0.001])
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(),
            X_mark,
            marking_shape,
            "table_marking_center",
            [0.8, 0.0, 0.0, 1.0],
        )

    def _add_table_rims(self) -> None:
        rim_height = 0.03
        rim_thickness = 0.045
        rim_color = [0.78, 0.78, 0.78, 1.0]

        def add_rim_box(name, translation, size):
            shape = Box(*size)
            X = RigidTransform(p=translation)
            self.plant.RegisterVisualGeometry(
                self.plant.world_body(),
                X,
                shape,
                name + "_visual",
                rim_color,
            )
            self.plant.RegisterCollisionGeometry(
                self.plant.world_body(),
                X,
                shape,
                name + "_collision",
                ProximityProperties(),
            )

        table_length = 1.064 * 2
        table_width = 0.609 * 2
        top_z = self.table_height + rim_height / 2

        # side walls
        add_rim_box(
            "rim_left",
            [0, table_width / 2 + rim_thickness / 2, top_z],
            [table_length, rim_thickness, rim_height],
        )
        add_rim_box(
            "rim_right",
            [0, -table_width / 2 - rim_thickness / 2, top_z],
            [table_length, rim_thickness, rim_height],
        )

        # goal walls
        goal_width = 0.394
        add_rim_box(
            "rim_home",
            [-table_length / 2 - rim_thickness / 2, 0, top_z],
            [rim_thickness, goal_width, rim_height],
        )
        add_rim_box(
            "rim_away",
            [table_length / 2 + rim_thickness / 2, 0, top_z],
            [rim_thickness, goal_width, rim_height],
        )

        # Decorative rim meshes if available
        rim_mesh_path = self.assets_root / "mjx" / "assets" / "table_rim.stl"
        if rim_mesh_path.exists():
            mesh = Mesh(rim_mesh_path.as_posix(), scale=0.001)
            rim_positions = [
                ("rim_mesh_home_l", [-1.019, 0.322, 0.0]),
                ("rim_mesh_home_r", [-1.019, -0.322, 0.0]),
                ("rim_mesh_away_l", [1.019, 0.322, 0.0]),
                ("rim_mesh_away_r", [1.019, -0.322, 0.0]),
            ]
            mesh_color = [0.9, 0.9, 0.9, 1.0]
            for name, pos in rim_positions:
                X = RigidTransform(p=pos)
                self.plant.RegisterVisualGeometry(
                    self.plant.world_body(),
                    X,
                    mesh,
                    name + "_visual",
                    mesh_color,
                )

    def _add_robots(self) -> None:
        from pydrake.multibody.parsing import LoadModelDirectives, ProcessModelDirectives

        robot_yaml = """
directives:
- add_model:
    name: iiwa_robot1
    file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
- add_weld:
    parent: world
    child: iiwa_robot1::iiwa_link_0
    X_PC:
        translation: [-1.51, 0, 0.0]
        rotation: !Rpy { deg: [0, 0, 180] }
- add_model:
    name: iiwa_robot2
    file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
- add_weld:
    parent: world
    child: iiwa_robot2::iiwa_link_0
    X_PC:
        translation: [1.51, 0, 0.0]
        rotation: !Rpy { deg: [0, 0, 0] }
"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(robot_yaml)
            yaml_file = f.name

        directives = LoadModelDirectives(yaml_file)
        ProcessModelDirectives(directives, self.plant, self.parser)

        import os

        os.unlink(yaml_file)

        self.robot1_model = self.plant.GetModelInstanceByName("iiwa_robot1")
        self.robot2_model = self.plant.GetModelInstanceByName("iiwa_robot2")

        self.robot1_tool_frame = None
        self.robot2_tool_frame = None

        if self.robot1_model is not None:
            self.robot1_tool_frame = self._add_tool_body(self.robot1_model, "robot1")
        if self.robot2_model is not None:
            self.robot2_tool_frame = self._add_tool_body(self.robot2_model, "robot2")

    def _add_tool_body(self, robot_model, robot_name: str):
        """
        Adds a virtual tool body welded to the end effector so that downstream
        planners can reference `robotX/tool_body`.
        """
        ee_frame = self.plant.GetFrameByName("iiwa_link_7", robot_model)

        tool_body = self.plant.AddRigidBody(
            model_instance=robot_model,
            name=f"{robot_name}/tool_body",
            M_BBo_B=SpatialInertia(
                mass=0.5,
                p_PScm_E=[0, 0, 0.05],
                G_SP_E=UnitInertia(0.02, 0.02, 0.02),
            ),
        )

        X_weld = RigidTransform(p=[0, 0, 0.126])
        self.plant.WeldFrames(
            ee_frame,
            tool_body.body_frame(),
            X_weld,
        )

        return tool_body.body_frame()

    def _add_puck(self) -> None:
        """Creates a planar puck constrained to the air-hockey surface."""
        from pydrake.all import PrismaticJoint, RevoluteJoint

        model = self.plant.AddModelInstance("puck_model")
        eps_mass = 1e-6
        eps_inertia = SpatialInertia(
            mass=eps_mass,
            p_PScm_E=[0, 0, 0],
            G_SP_E=UnitInertia(1.0, 1.0, 1.0),
        )

        # Planar chain: world -> slider x -> slider y -> yaw -> puck body
        x_body = self.plant.AddRigidBody(
            name="puck_frame_x",
            model_instance=model,
            M_BBo_B=eps_inertia,
        )
        self.puck_x_joint = self.plant.AddJoint(
            PrismaticJoint(
                name="puck_x",
                frame_on_parent=self.plant.world_frame(),
                frame_on_child=x_body.body_frame(),
                axis=[1, 0, 0],
                damping=0.02,
            )
        )

        y_body = self.plant.AddRigidBody(
            name="puck_frame_y",
            model_instance=model,
            M_BBo_B=eps_inertia,
        )
        self.puck_y_joint = self.plant.AddJoint(
            PrismaticJoint(
                name="puck_y",
                frame_on_parent=x_body.body_frame(),
                frame_on_child=y_body.body_frame(),
                axis=[0, 1, 0],
                damping=0.02,
            )
        )

        yaw_body = self.plant.AddRigidBody(
            name="puck_yaw",
            model_instance=model,
            M_BBo_B=eps_inertia,
        )
        self.puck_yaw_joint = self.plant.AddJoint(
            RevoluteJoint(
                name="puck_yaw",
                frame_on_parent=y_body.body_frame(),
                frame_on_child=yaw_body.body_frame(),
                axis=[0, 0, 1],
                damping=1e-4,
            )
        )

        puck_radius = 0.03165
        puck_height = 0.01
        puck_mass = 0.01
        puck_inertia = SpatialInertia(
            mass=puck_mass,
            p_PScm_E=[0, 0, 0],
            G_SP_E=UnitInertia(0.5, 0.5, 0.5),
        )

        puck_shape = Cylinder(puck_radius, puck_height)
        puck_props = ProximityProperties()
        puck_props.AddProperty("material", "coulomb_friction", CoulombFriction(1e-5, 1e-5))

        self.puck_body = self.plant.AddRigidBody(
            name="puck",
            model_instance=model,
            M_BBo_B=puck_inertia,
        )

        puck_center_height = self.table_height + puck_height / 2.0
        self.plant.WeldFrames(
            yaw_body.body_frame(),
            self.puck_body.body_frame(),
            RigidTransform(p=[0, 0, puck_center_height]),
        )

        self.plant.RegisterVisualGeometry(
            self.puck_body,
            RigidTransform(),
            puck_shape,
            "puck_visual",
            [1.0, 0.0, 0.0, 1.0],
        )
        self.plant.RegisterCollisionGeometry(
            self.puck_body,
            RigidTransform(),
            puck_shape,
            "puck_collision",
            puck_props,
        )

    # ------------------------------------------------------ new environment add-ons
    def _add_side_table(self) -> None:
        side_table_length = 0.5
        side_table_width = 0.4
        side_table_height = 0.05
        side_table_z = 0.8

        side_table_x = -1.8
        side_table_y = 0.0

        table_shape = Box(side_table_length, side_table_width, side_table_height)
        props = ProximityProperties()
        props.AddProperty("material", "coulomb_friction", CoulombFriction(0.5, 0.3))

        X_side_table = RigidTransform(p=[side_table_x, side_table_y, side_table_z])

        self.plant.RegisterVisualGeometry(
            self.plant.world_body(),
            X_side_table,
            table_shape,
            "side_table_visual",
            [0.4, 0.3, 0.2, 1.0],
        )
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(),
            X_side_table,
            table_shape,
            "side_table_collision",
            props,
        )

        self.side_table_position = np.array([side_table_x, side_table_y, side_table_z])
        self.side_table_height = side_table_z

    def _add_paddle_object(self) -> None:
        paddle_model_instance = self.plant.AddModelInstance("paddle_object")

        paddle_radius = 0.04815
        paddle_thickness = 0.01
        handle_radius = 0.015
        handle_length = 0.08

        paddle_mass = 0.2
        paddle_inertia = np.diag([0.001, 0.001, 0.001])

        self.paddle_body = self.plant.AddRigidBody(
            model_instance=paddle_model_instance,
            name="paddle_object",
            M_BBo_B=SpatialInertia(
                mass=paddle_mass,
                p_PScm_E=[0, 0, 0],
                G_SP_E=UnitInertia(
                    Ixx=paddle_inertia[0, 0] / paddle_mass,
                    Iyy=paddle_inertia[1, 1] / paddle_mass,
                    Izz=paddle_inertia[2, 2] / paddle_mass,
                ),
            ),
        )

        props = ProximityProperties()
        props.AddProperty("material", "coulomb_friction", CoulombFriction(0.3, 0.2))

        paddle_shape = Cylinder(paddle_radius, paddle_thickness)
        R_paddle = RollPitchYaw(0, np.pi / 2, 0)
        X_surface = RigidTransform(R_paddle, p=[0, 0, 0])
        self.plant.RegisterVisualGeometry(
            self.paddle_body,
            X_surface,
            paddle_shape,
            "paddle_surface_visual",
            [0.2, 0.6, 1.0, 1.0],
        )
        self.plant.RegisterCollisionGeometry(
            self.paddle_body,
            X_surface,
            paddle_shape,
            "paddle_surface_collision",
            props,
        )

        handle_shape = Cylinder(handle_radius, handle_length)
        handle_z_center = paddle_thickness / 2.0 + handle_radius
        X_handle = RigidTransform(p=[0, 0, handle_z_center])
        self.plant.RegisterVisualGeometry(
            self.paddle_body,
            X_handle,
            handle_shape,
            "paddle_handle_visual",
            [0.8, 0.2, 0.2, 1.0],
        )
        self.plant.RegisterCollisionGeometry(
            self.paddle_body,
            X_handle,
            handle_shape,
            "paddle_handle_collision",
            props,
        )

        initial_paddle_z = self.side_table_height + handle_z_center + 0.01
        self.initial_paddle_pose = RigidTransform(
            p=[self.side_table_position[0], self.side_table_position[1], initial_paddle_z]
        )

    # ----------------------------------------------------------------- public API
    # ----------------------------------------------------------------- core API
    def _generate_random_puck_velocity(self):
        speed = np.random.uniform(0.5, 2.0)
        angle = np.random.uniform(0, 2 * np.pi)
        return [speed * np.cos(angle), speed * np.sin(angle), 0.0]

    def reset(self, puck_pos=None, puck_vel=None, random_velocity=False, reset_paddle=True):
        self.simulator.Initialize()
        self.context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        if puck_pos is None:
            puck_pos = np.array([0.0, 0.0, self.table_height])
        else:
            puck_pos = np.asarray(puck_pos)
        if puck_vel is None:
            puck_vel = (
                self._generate_random_puck_velocity() if random_velocity else [0.0, 0.0, 0.0]
            )

        self.puck_x_joint.set_translation(self.plant_context, puck_pos[0])
        self.puck_y_joint.set_translation(self.plant_context, puck_pos[1])
        self.puck_yaw_joint.set_angle(self.plant_context, 0.0)

        self.puck_x_joint.set_translation_rate(self.plant_context, puck_vel[0])
        self.puck_y_joint.set_translation_rate(self.plant_context, puck_vel[1])
        self.puck_yaw_joint.set_angular_rate(self.plant_context, 0.0)

        if reset_paddle and self.paddle_body is not None:
            self.paddle_grasped = False
            self.plant.SetFreeBodyPose(self.plant_context, self.paddle_body, self.initial_paddle_pose)
            self.plant.SetFreeBodySpatialVelocity(
                self.plant_context,
                self.paddle_body,
                SpatialVelocity(w=np.zeros(3), v=np.zeros(3)),
            )

        self.diagram.ForcedPublish(self.context)

    def step(self, action=None, duration=None):
        duration = duration or self.time_step
        target_time = self.simulator.get_context().get_time() + duration
        self.simulator.AdvanceTo(target_time)

    def get_puck_state(self):
        x = self.puck_x_joint.get_translation(self.plant_context)
        y = self.puck_y_joint.get_translation(self.plant_context)
        vx = self.puck_x_joint.get_translation_rate(self.plant_context)
        vy = self.puck_y_joint.get_translation_rate(self.plant_context)
        z = self.table_height + 0.005
        return np.array([x, y, z]), np.array([vx, vy, 0.0])

    def _apply_puck_damping(self):
        vx = self.puck_x_joint.get_translation_rate(self.plant_context)
        vy = self.puck_y_joint.get_translation_rate(self.plant_context)
        vx *= 1.0 - self.puck_damping_linear * self.time_step
        vy *= 1.0 - self.puck_damping_linear * self.time_step
        self.puck_x_joint.set_translation_rate(self.plant_context, vx)
        self.puck_y_joint.set_translation_rate(self.plant_context, vy)

    def run_simulation(self, duration=5.0):
        start = self.simulator.get_context().get_time()
        while self.simulator.get_context().get_time() - start < duration:
            self._apply_puck_damping()
            self.step(duration=0.01)
            self.diagram.ForcedPublish(self.context)

    def set_robot_joint_positions(self, robot_model, q):
        self.plant.SetPositions(self.plant_context, robot_model, q)

    def get_robot_joint_positions(self, robot_model):
        return self.plant.GetPositions(self.plant_context, robot_model)

    def get_paddle_pose(self):
        if self.paddle_body is None:
            return None
        return self.plant.EvalBodyPoseInWorld(self.plant_context, self.paddle_body)

    def is_paddle_grasped(self):
        return self.paddle_grasped

    def initialize_motion_controllers(self):
        from scripts.kinematics.motion_controller import MotionController

        if hasattr(self, 'robot1_model') and self.robot1_model is not None:
            self.motion_controller_1 = MotionController(
                self, robot_id=1
            )
        
        if hasattr(self, 'robot2_model') and self.robot2_model is not None:
            self.motion_controller_2 = MotionController(
                self, robot_id=2
            )


