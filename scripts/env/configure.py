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
from pydrake.all import AddFrameTriadIllustration
from pydrake.multibody.tree import BodyIndex
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer

# # Clean up the Meshcat instance.
# meshcat.Delete()
# meshcat.DeleteAddedControls()
from pydrake.systems.framework import LeafSystem
import numpy as np

import pydrake.geometry as mut


# class AirHockeyChallengeEnv:
#     def __init__(self):
#         # Start a new Meshcat instance.
#         self.meshcat = StartMeshcat()

#         builder = DiagramBuilder()

meshcat = StartMeshcat()
builder = DiagramBuilder()

plant, scene_graph = AddMultibodyPlantSceneGraph(
    builder,
    time_step=0.001,
)
parser = Parser(builder)
directives = LoadModelDirectives("scenario/puckbot_scene.yaml")
ProcessModelDirectives(directives, parser)
AddFrameTriadIllustration(
    body=plant.GetBodyByName("table"),
    scene_graph=scene_graph,
    # length=0.15,
    # radius=0.006,
)

# parser.AddModels("assets/models/table_wide.sdf")
# parser.AddModels("assets/models/paddle.sdf")
# left_parser = Parser(plant, "left")
# left_paddle = left_parser.AddModels("assets/models/paddle.sdf")[0]

# right_parser = Parser(plant, "right")
# right_paddle = right_parser.AddModels("assets/models/paddle.sdf")[0]

# table_model = parser.AddModels("assets/models/table.xml")[0]

# plant.WeldFrames(
#     plant.world_frame(),
#     plant.GetFrameByName("paddle_body_link", left_paddle),
#     RigidTransform([0.3, 0.0, 0.15])   # left paddle placement
# )

# plant.WeldFrames(
#     plant.world_frame(),
#     plant.GetFrameByName("paddle_body_link", right_paddle),
#     RigidTransform([-0.3, 0.0, 0.15])  # right paddle placement
# )

# plant.WeldFrames(
#     plant.GetFrameByName("table_body"),
#     plant.GetFrameByName("paddle_body_link"),
#     RigidTransform([0, 0, 0.15])              # just above table
# )
# add_axes_for_all_bodies(plant, scene_graph)
plant.Finalize()


# plant_context = plant.CreateDefaultContext()
# # paddle = plant.GetBodyByName("paddle_body_link")
# # X_WorldPaddle = RigidTransform()
# # paddle_frame = plant.GetFrameByName("paddle_frame")
# # paddle_frame.SetTranslationInParent(plant_context, [0, 0, 0])
# # paddle_frame.SetRotationInParent(plant_context, RotationMatrix.Identity())

# AddDefaultVisualization(builder=builder, meshcat=meshcat)
# diagram = builder.Build()
# simulator = Simulator(diagram, plant_context)
# simulator.set_target_realtime_rate(1.0)
# simulator.set_publish_every_time_step(True)
# visualizer.Run()

AddDefaultVisualization(builder, meshcat)
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(True)

simulator.Initialize()
simulator.AdvanceTo(10.0)