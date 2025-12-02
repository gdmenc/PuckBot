"""
Air Hockey Game Environment.

Provides a Drake simulation setup that extends the base `AirHockeyDrakeEnv`
with a side table and a free paddle object that can be grasped by the robot
before playing air hockey.
"""

from __future__ import annotations

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    Cylinder,
    CoulombFriction,
    DiagramBuilder,
    MeshcatVisualizer,
    Parser,
    ProximityProperties,
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

    def __init__(self, time_step: float = 0.001, use_meshcat: bool = True):
        self.time_step = time_step
        self.builder = DiagramBuilder()

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
        self._add_table()
        self._add_robots()
        self._add_puck()

        # Additional assets for gameplay setup
        self._add_side_table()
        self._add_paddle_object()

        self.paddle_grasped = False

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

    def _add_table(self) -> None:
        table_length = 1.064 * 2
        table_width = 0.609 * 2
        table_height = 0.0505 * 2

        self.table_height = table_height

        table_shape = Box(table_length, table_width, table_height)

        props = ProximityProperties()
        props.AddProperty("material", "coulomb_friction", CoulombFriction(0.01, 0.005))

        X_WT = RigidTransform(p=[0, 0, table_height / 2])
        self.plant.RegisterVisualGeometry(
            self.plant.world_body(),
            X_WT,
            table_shape,
            "table_surface_visual",
            [1.0, 1.0, 1.0, 1.0],
        )
        self.plant.RegisterCollisionGeometry(
            self.plant.world_body(),
            X_WT,
            table_shape,
            "table_surface_collision",
            props,
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

        # Track tool frames for later use
        self.robot1_tool_frame = self.plant.GetFrameByName(
            "iiwa_link_7", self.robot1_model
        )
        self.robot2_tool_frame = self.plant.GetFrameByName(
            "iiwa_link_7", self.robot2_model
        )

    def _add_puck(self) -> None:
        puck_model_instance = self.plant.AddModelInstance("puck_model")

        puck_radius = 0.03165
        puck_height = 0.01

        puck_shape = Cylinder(puck_radius, puck_height)

        props = ProximityProperties()
        props.AddProperty("material", "coulomb_friction", CoulombFriction(0.01, 0.005))

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
                    Izz=puck_inertia[2, 2] / puck_mass,
                ),
            ),
        )

        self.plant.RegisterVisualGeometry(
            self.puck_body, RigidTransform(), puck_shape, "puck_visual", [1.0, 0.0, 0.0, 1.0]
        )
        self.plant.RegisterCollisionGeometry(
            self.puck_body,
            RigidTransform(),
            puck_shape,
            "puck_collision",
            props,
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
            puck_pos = [0.0, 0.0, self.table_height + 0.01]
        if puck_vel is None:
            puck_vel = self._generate_random_puck_velocity() if random_velocity else [0.0, 0.0, 0.0]

        self.plant.SetFreeBodyPose(self.plant_context, self.puck_body, RigidTransform(p=puck_pos))
        self.plant.SetFreeBodySpatialVelocity(
            self.plant_context,
            self.puck_body,
            SpatialVelocity(w=np.zeros(3), v=np.array(puck_vel)),
        )

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
        pose = self.plant.GetFreeBodyPose(self.plant_context, self.puck_body)
        spatial_vel = self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, self.puck_body)
        return pose.translation(), spatial_vel.translational()

    def _apply_puck_damping(self):
        spatial_vel = self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, self.puck_body)
        v_trans = spatial_vel.translational() * (1.0 - self.puck_damping_linear * self.time_step)
        v_rot = spatial_vel.rotational()
        v_rot[2] = v_rot[2] * (1.0 - self.puck_damping_angular * self.time_step)
        self.plant.SetFreeBodySpatialVelocity(
            self.plant_context,
            self.puck_body,
            SpatialVelocity(w=v_rot, v=v_trans),
        )

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
        from scripts.simple_motion_controller import SimpleMotionController

        self.motion_controller_1 = SimpleMotionController(
            self.plant, self.robot1_model, "robot1/tool_body"
        )
        self.motion_controller_2 = SimpleMotionController(
            self.plant, self.robot2_model, "robot2/tool_body"
        )


