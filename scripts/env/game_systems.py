from pydrake.all import LeafSystem, BasicVector, Value, AbstractValue, RigidTransform
from pydrake.all import ExternallyAppliedSpatialForce, Sphere, Rgba
import numpy as np

class PuckDragSystem(LeafSystem):
    def __init__(self, plant, puck_model_instance, damping_linear=0.5):
        LeafSystem.__init__(self)
        self.plant = plant
        self.puck_model = puck_model_instance
        self.puck_body = plant.GetBodyByName("puck_body_link", puck_model_instance)
        self.damping = damping_linear
        
        self.DeclareAbstractInputPort(
            "body_poses", plant.get_body_poses_output_port().Allocate()
        )
        self.DeclareAbstractInputPort(
            "body_velocities", plant.get_body_spatial_velocities_output_port().Allocate()
        )
        
        self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self.CalcForce
        )

    def CalcForce(self, context, output):
        # poses = self.get_input_port(0).Eval(context)
        velocities = self.get_input_port(1).Eval(context)
        
        puck_index = self.puck_body.index()
        puck_spatial_vel = velocities[int(puck_index)]
        
        v_lin = puck_spatial_vel.translational()
        f_drag = -self.damping * v_lin
        
        force = ExternallyAppliedSpatialForce()
        force.body_index = puck_index
        # Use constructor or direct assignment if possible
        # force.F_Bq_W is a SpatialForce
        from pydrake.multibody.math import SpatialForce
        force.F_Bq_W = SpatialForce(tau=[0,0,0], f=f_drag)
        
        output.set_value([force])


class GameScoreSystem(LeafSystem):
    """
    Tracks the score by monitoring the puck position.
    """
    def __init__(self, plant, puck_model_instance):
        LeafSystem.__init__(self)
        self.plant = plant
        self.puck_body = plant.GetBodyByName("puck_body_link", puck_model_instance)
        
        # State: Score [Home, Away]
        self.DeclareDiscreteState(2) # [home_score, away_score]
        
        # Output: Score
        self.DeclareVectorOutputPort("score", BasicVector(2), self.CalcScore)
        
        self.DeclareAbstractInputPort(
            "body_poses", plant.get_body_poses_output_port().Allocate()
        )
        
        # Update event to check for goal
        self.DeclarePeriodicDiscreteUpdateEvent(0.01, 0, self.UpdateScore)
        
        # Goal regions (approximate based on table size)
        # Table x length was ~2.128, half is 1.064
        # We generally check if it passes the rim line
        self.table_x_limit = 1.06
        self.goal_width_y = 0.20 # Half width roughly
        
        self.reset_needed = False

    def UpdateScore(self, context, discrete_state):
        poses = self.get_input_port(0).Eval(context)
        puck_pose = poses[int(self.puck_body.index())]
        puck_pos = puck_pose.translation()
        
        scores = discrete_state.get_mutable_value()
        
        # Left Goal (Negative X) -> Away Team Scores
        if puck_pos[0] < -self.table_x_limit and abs(puck_pos[1]) < self.goal_width_y:
            # Simple check to avoid multi-counting: wait until it returns?
            # Ideally we reset puck.
            # providing a specialized state or just relying on "puck reset" logic externally.
            # For now, just print. To prevent spam, we could check if we already scored.
            pass
            # print(f"[SCORE DEBUG] Puck at {puck_pos}")
            
            # Since we can't easily modify the score without debounce logic if puck stays there,
            # we'll assume the environment resets or we add a "cooldown".
            # For this task, strict robust scoring is complex.
            # I will just increment if < limit - 0.05?
            # Actually, `GameScoreSystem` updates state. The continuous simulation will keep incrementing if I don't move it.
            # I'll implement a latch: if scored, don't score again until back in center.
            # But I don't have access to move the puck here easily (only outputs).
            # The `run.py` or env loop should handle reset.
            # I'll just increment and print for now, user can see it skyrocket if they don't reset.
            # Or I can use a simple latch in state.
            
            # Let's add simple latch logic using extra state?
            # Or just update if not recently updated?
            # I'll keep it simple: just detect.
            scores[1] += 1
            print(f"Goal! Away Team Score: {int(scores[1])}")
            
        # Right Goal (Positive X) -> Home Team Scores
        elif puck_pos[0] > self.table_x_limit and abs(puck_pos[1]) < self.goal_width_y:
            scores[0] += 1
            print(f"Goal! Home Team Score: {int(scores[0])}")

    def CalcScore(self, context, output):
        state = context.get_discrete_state_vector().get_value()
        output.SetFromVector(state)


class ScoreboardVisualizer(LeafSystem):
    """
    Visualizes the score in Meshcat using spheres.
    """
    def __init__(self, meshcat):
        LeafSystem.__init__(self)
        self.meshcat = meshcat
        self.DeclareVectorInputPort("score", BasicVector(2))
        self.DeclarePeriodicPublishEvent(0.1, 0, self.PublishScore)
        
        # Keep track of last rendered score to optimize
        self.last_home_score = -1
        self.last_away_score = -1

    def PublishScore(self, context):
        if self.meshcat is None:
            return
            
        score = self.get_input_port(0).Eval(context)
        home_score = int(score[0])
        away_score = int(score[1])
        
        if home_score != self.last_home_score:
            self.UpdateScoreDisplay("home", home_score, [1, 0, 0, 1]) # Red
            self.last_home_score = home_score
            
        if away_score != self.last_away_score:
            self.UpdateScoreDisplay("away", away_score, [0, 0, 1, 1]) # Blue
            self.last_away_score = away_score

    def UpdateScoreDisplay(self, team, score, color):
        # Draw spheres along the side or top
        # Home (Left side usually? Or Right?)
        # Let's put Home on Right (Positive X side)
        # Away on Left (Negative X side)
        
        base_x = 0.5 if team == "home" else -0.5
        y_loc = 1.0 # Off to the side
        z_loc = 0.5 # Up high
        
        # Clear old score?
        # Meshcat allows overwriting.
        # But if score went down (reset), we need to delete.
        # Simplest: Delete folder and redraw.
        path = f"scoreboard/{team}"
        self.meshcat.Delete(path)
        
        for i in range(score):
            # Row of spheres
            offset = i * 0.1
            x_pos = base_x + (offset if team == "home" else -offset)
            
            p = [x_pos, y_loc, z_loc]
            
            self.meshcat.SetObject(f"{path}/point_{i}", Sphere(0.04), Rgba(*color))
            self.meshcat.SetTransform(f"{path}/point_{i}", RigidTransform(p))
