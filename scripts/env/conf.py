import os
from pathlib import Path
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    JacobianWrtVariable,
)

class AirHockeyChallengeEnv:
    def __init__(self,
                 scenario_name: str = "air_hockey",
                 generated_dir: str = "generated_scene",
                 rim_stl_path: str = "mjx/assets/table_rim.stl",
                 use_wsg: bool = True,
                 meshcat: bool = True,
                 time_step: float = 0.0):
    
        self.generated_dir = Path(generated_dir)
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        self.rim_stl_path = rim_stl_path
        self.scenario_name = scenario_name
        self.use_wsg = use_wsg
        self.meshcat_flag = meshcat
        self.time_step = time_step

        self.table_sdf_path = str(self.generated_dir / "table.sdf")
        self.puck_sdf_path = str(self.generated_dir / "puck.sdf")
        self.yaml_path = str(self.generated_dir / f"{self.scenario_name}.yaml")

        self.builder = None
        self.plant = None
        self.scene_graph = None
        self.simulator = None
        self.diagram = None
        self.meshcat = None
        self.meshcat_visualizer = None

        self._write_table_sdf()
        self._write_puck_sdf()
        self._write_scenario_yaml()
        self._build_diagram()

    def _write_table_sdf(self):
        """
        A simplified SDF that reproduces:
        - table surface (box)
        - rim (boxes + references to rim STL twice)
        The rim STL path is used with a <mesh> visual/collision element.
        """
        rim_mesh = self.rim_stl_path.replace("\\", "/")
        # SDF coordinates: units meters. The MJCF used a scale 0.001 for the mesh; keep that.
        sdf = f"""<?xml version="1.0" ?>
                <sdf version="1.6">
                <model name="table">
                    <static>true</static>
                    <link name="table_link">
                    <!-- surface: 2*size dims in MJCF were 1.064 x 0.609 x 0.0505 -> SDF size half-extents -->
                    <visual name="surface_visual">
                        <geometry>
                        <box>
                            <size>2.128 1.218 0.101</size>
                        </box>
                        </geometry>
                        <material>
                        <ambient>1 1 1 1</ambient>
                        <diffuse>1 1 1 1</diffuse>
                        </material>
                        <pose>0 0 {-0.0505} 0 0 0</pose>
                    </visual>
                    <collision name="surface_collision">
                        <geometry>
                        <box>
                            <size>2.128 1.218 0.101</size>
                        </box>
                        </geometry>
                        <pose>0 0 {-0.0505} 0 0 0</pose>
                    </collision>

                    <!-- left/right rim as box approximations -->
                    <visual name="rim_left_box">
                        <geometry>
                        <box>
                            <size>2.128 0.09 0.03</size>
                        </box>
                        </geometry>
                        <pose>0 0.564 0.015 0 0 0</pose>
                        <material>
                        <ambient>0.8 0.8 0.8 1</ambient>
                        <diffuse>0.8 0.8 0.8 1</diffuse>
                        </material>
                    </visual>

                    <visual name="rim_right_box">
                        <geometry>
                        <box>
                            <size>2.128 0.09 0.03</size>
                        </box>
                        </geometry>
                        <pose>0 -0.564 0.015 0 0 0</pose>
                    </visual>

                    <!-- two copies of the rim STL (scaled as MJCF used 0.001) -->
                    <visual name="rim_mesh_home_l">
                        <geometry>
                        <mesh>
                            <uri>file://{rim_mesh}</uri>
                            <scale>0.001 0.001 0.001</scale>
                        </mesh>
                        </geometry>
                        <pose>1.019 0.322 0 0 0 0</pose>
                    </visual>

                    <visual name="rim_mesh_home_r">
                        <geometry>
                        <mesh>
                            <uri>file://{rim_mesh}</uri>
                            <scale>0.001 0.001 0.001</scale>
                        </mesh>
                        </geometry>
                        <pose>1.019 -0.322 0 0 0 0</pose>
                    </visual>

                    <pose>0 0 0 0 0 0</pose>
                    </link>
                </model>
                </sdf>
                """
        with open(self.table_sdf_path, "w") as f:
            f.write(sdf)

    def _write_puck_sdf(self):
        """
        The puck model is implemented as a 3-DOF kinematic chain:
        world -> puck_x (slider) -> puck_y (slider) -> puck_yaw (revolute) -> puck_link
        The puck visual is an ellipsoid/capsule approximation.
        """
        sdf = """<?xml version="1.0" ?>
                <sdf version="1.6">
                <model name="puck">
                    <static>false</static>

                    <!-- first joint: slide x -->
                    <link name="puck_frame_x">
                    <pose>0 0 0 0 0 0</pose>
                    <inertial><mass>1e-6</mass></inertial>
                    </link>
                    <joint name="puck_x" type="prismatic">
                    <parent>world</parent>
                    <child>puck_frame_x</child>
                    <axis>
                        <xyz>1 0 0</xyz>
                        <dynamics>
                        <damping>0.005</damping>
                        </dynamics>
                    </axis>
                    </joint>

                    <!-- second joint: slide y -->
                    <link name="puck_frame_y">
                    <pose>0 0 0 0 0 0</pose>
                    <inertial><mass>1e-6</mass></inertial>
                    </link>
                    <joint name="puck_y" type="prismatic">
                    <parent>puck_frame_x</parent>
                    <child>puck_frame_y</child>
                    <axis>
                        <xyz>0 1 0</xyz>
                        <dynamics>
                        <damping>0.005</damping>
                        </dynamics>
                    </axis>
                    </joint>

                    <!-- yaw joint -->
                    <link name="puck_base">
                    <pose>0 0 0.01 0 0 0</pose>
                    <inertial>
                        <mass>0.01</mass>
                        <inertia>
                        <ixx>2.5e-6</ixx>
                        <iyy>2.5e-6</iyy>
                        <izz>5e-6</izz>
                        </inertia>
                    </inertial>
                    <!-- visual: approximate puck as a short cylinder (capsule-like) -->
                    <visual name="puck_vis">
                        <geometry>
                        <cylinder>
                            <radius>0.03165</radius>
                            <length>0.006</length>
                        </cylinder>
                        </geometry>
                        <pose>0 0 0 0 0 0</pose>
                        <material>
                        <ambient>0 0 0 0</ambient>
                        <diffuse>0 0 0 1</diffuse>
                        </material>
                    </visual>
                    <collision name="puck_collision">
                        <geometry>
                        <cylinder>
                            <radius>0.03165</radius>
                            <length>0.006</length>
                        </cylinder>
                        </geometry>
                    </collision>
                    </link>

                    <joint name="puck_yaw" type="revolute">
                    <parent>puck_frame_y</parent>
                    <child>puck_base</child>
                    <axis>
                        <xyz>0 0 1</xyz>
                        <dynamics>
                        <damping>2e-6</damping>
                        </dynamics>
                    </axis>
                    </joint>

                </model>
                </sdf>
                """
        with open(self.puck_sdf_path, "w") as f:
            f.write(sdf)

    def _write_scenario_yaml(self):
        """
        Generates a Drake model directives (scenario) YAML file that:
         - Adds IIWA & WSG models (from package URIs)
         - Adds the generated table and puck
         - Welds table to world
         - Welds wsg to iiwa end if requested
        """
        iiwa_uri = "package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf"
        wsg_uri = "package://manipulation/hydro/schunk_wsg_50_with_tip.sdf"

        table_file = Path(self.table_sdf_path).absolute().as_uri()
        puck_file = Path(self.puck_sdf_path).absolute().as_uri()

        yaml = f"""
                directives:
                - add_model:
                    name: iiwa
                    file: {iiwa_uri}
                - add_model:
                    name: wsg
                    file: {wsg_uri}
                - add_model:
                    name: table
                    file: {table_file}
                - add_model:
                    name: puck
                    file: {puck_file}

                # Weld table to world at small z offset (to match MJCF pos)
                - add_weld:
                    parent: world
                    child: table::table_link
                    X_PC:
                        translation: [0.0, 0.0, -0.05]
                        rotation: !Rpy {{ deg: [0, 0, -90] }}

                # Put iiwa base a little back
                - add_weld:
                    parent: world
                    child: iiwa::iiwa_link_0
                    X_PC:
                        translation: [0, -0.5, 0]
                        rotation: !Rpy {{ deg: [0, 0, 180] }}

                # Weld the gripper to the end of the iiwa (optional)
                - add_weld:
                    parent: iiwa::iiwa_link_7
                    child: wsg::body
                    X_PC:
                        translation: [0, 0, 0.09]
                        rotation: !Rpy {{deg: [90, 0, 90] }}

                model_drivers:
                """
        yaml += """
                iiwa: !IiwaDriver
                    control_mode: position_only
                    hand_model_name: wsg
                wsg: !SchunkWsgDriver {}
        """
        with open(self.yaml_path, "w") as f:
            f.write(yaml)


    def _build_diagram(self):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=self.time_step)
        self.plant = plant
        self.scene_graph = scene_graph
        parser = Parser(self.plant)

        # load model directives
        directives = LoadModelDirectives(self.yaml_path)
        ProcessModelDirectives(directives, parser)

        # finalize plant
        self.plant.Finalize()

        # Optional Meshcat
        if self.meshcat_flag:
            self.meshcat = StartMeshcat()
            # add meshcat visualizer
            self.meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
                builder=builder,
                scene_graph=self.scene_graph,
                meshcat=self.meshcat,
                publish_period=0.05
            )
        # build diagram
        self.diagram = builder.Build()
        self.simulator = Simulator(self.diagram)
        # set simulator to use plant context as default
        self.simulator.set_publish_every_time_step(False)

        # Build references to common model instances
        # Use FindModelInstanceByName when you need ids
        try:
            self.iiwa_model = self.plant.GetModelInstanceByName("iiwa")
        except Exception:
            self.iiwa_model = None
        try:
            self.table_model = self.plant.GetModelInstanceByName("table")
        except Exception:
            self.table_model = None
        try:
            self.puck_model = self.plant.GetModelInstanceByName("puck")
        except Exception:
            self.puck_model = None

    def reset(self):
        """
        Reset the simulator and set reasonable defaults for IIWA if present.
        """
        # (re)create simulator context
        self.simulator.reset()
        self.simulator.set_target_realtime_rate(1.0)
        context = self.simulator.get_mutable_context()
        # set default iiwa joint positions (if iiwa present)
        if self.iiwa_model is not None:
            # iiwa has 7 joints named iiwa_joint_1 ... iiwa_joint_7 in the model
            q0 = [-1.57, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0]
            for i, value in enumerate(q0, start=1):
                joint_name = f"iiwa_joint_{i}"
                try:
                    joint = self.plant.GetJointByName(joint_name, self.iiwa_model)
                    joint.set_angle(context, value)
                except Exception:
                    # many Drake joints are handled via position indices; use SetPositions by index if needed
                    pass

    def step(self, dt: float = 0.001):
        """
        Advance the simulator by dt seconds.
        """
        context = self.simulator.get_mutable_context()
        t0 = context.get_time()
        self.simulator.AdvanceTo(t0 + dt)

    def get_puck_pose(self):
        """
        Returns a (x,y,yaw) of the puck model root.
        """
        if self.puck_model is None:
            return None
        # the SDF named the final link 'puck_base'
        try:
            # Get frame / body pose in world
            body = self.plant.GetBodyByName("puck_base", self.puck_model)
            X_WB = self.plant.EvalBodyPoseInWorld(self.simulator.get_context(), body)
            p = X_WB.translation()
            R = X_WB.rotation().ToQuaternion()  # quaternion
            # extract yaw (around z) from rotation matrix
            # convert quaternion to yaw
            rot = X_WB.rotation()
            # rotation matrix uses rpy extraction
            rpy = rot.ToEulerAngles() if hasattr(rot, "ToEulerAngles") else (0.0, 0.0, 0.0)
            # fallback: compute yaw from quaternion
            # But we'll compute yaw from rotation matrix directly
            try:
                yaw = rot.ToYawPitchRoll()[0]
            except Exception:
                # fallback numeric
                yaw = 0.0
            return float(p[0]), float(p[1]), float(yaw)
        except Exception:
            return None

