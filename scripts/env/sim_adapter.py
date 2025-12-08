import numpy as np
from pydrake.all import (
    MultibodyPlant,
    Simulator,
    RigidTransform,
    SpatialVelocity,
)

class SimAdapter:
    """
    Adapts the raw Drake simulation objects (Simulator, Plant) to the 
    interface expected by PaddleGrasper and MotionController.
    """
    def __init__(self, simulator: Simulator, plant: MultibodyPlant, diagram, meshcat=None):
        self.simulator = simulator
        self.plant = plant
        self.diagram = diagram
        self.meshcat = meshcat
        
        # Environment Properties
        self.table_height = 0.10 # As defined in scenario_builder
        self.paddle_grasped = False
        
        # Find Models
        # Scenario builder uses 'right_iiwa' and 'left_iiwa'
        # MotionController expects 'robot1' / 'robot2' mapping.
        # usually robot1 = left, robot2 = right? 
        # Check game_env.py: robot1 at -1.35 (Left), robot2 at 1.35 (Right).
        # Scenario builder: left_iiwa at -1.3, right_iiwa at 1.3.
        # So robot1 = left_iiwa, robot2 = right_iiwa.
        
        try:
            self.robot1_model = plant.GetModelInstanceByName("left_iiwa")
        except:
            self.robot1_model = None
            
        try:
            self.robot2_model = plant.GetModelInstanceByName("right_iiwa")
        except:
            self.robot2_model = None
            
        # PUCK
        try:
            self.puck_model = plant.GetModelInstanceByName("puck")
            self.puck_body = plant.GetBodyByName("puck_body_link", self.puck_model)
        except:
            self.puck_model = None
            self.puck_body = None
            
        # PADDLES
        # We need a primary paddle to grasp.
        # If we control Right Arm (robot2), we grasp Right Paddle.
        # If we control Left Arm (robot1), we grasp Left Paddle.
        # PaddleGrasper uses `env.paddle_body`.
        # This implies a single paddle focus for the "current" task.
        # We will set `self.paddle_body` dynamically or expose both.
        # But PaddleGrasper accesses `env.paddle_body`.
        # We'll default to the right paddle for single arm (Right Arm),
        # or left paddle if using left arm.
        # This might be tricky if we want to grasp BOTH.
        # Run.py decides which robot to control. 
        # But PaddleGrasper is instantiated for a specific robot.
        # Wait, PaddleGrasper takes `env` but uses `env.paddle_body`.
        # This suggests the Env was single-agent specific? 
        # No, game_env had `_add_paddle_object` creating a SINGLE paddle.
        # Our scenario creates TWO paddles.
        # WE NEED TO HANDLE THIS.
        # Best fix: Assign `self.paddle_body` based on which robot is asking?
        # No, `env.paddle_body` is a value.
        # I will expose helper methods to get specific paddles,
        # and setters to set the "active" paddle body if needed.
        # OR, I will try to find "right_paddle" by default as usually user plays as right?
        
        try:
            self.right_paddle = plant.GetBodyByName("paddle_body_link", plant.GetModelInstanceByName("right_paddle"))
        except:
            self.right_paddle = None
            
        try:
            self.left_paddle = plant.GetBodyByName("paddle_body_link", plant.GetModelInstanceByName("left_paddle"))
        except:
            self.left_paddle = None
            
        # Default to right paddle for now
        self.paddle_body = self.right_paddle if self.right_paddle else self.left_paddle

        # Grasping State (per paddle)
        self.grasped_paddles = {}  # paddle_body -> tool_frame, X_Tool_Paddle
        self._paddle_grasped = False  # For current active paddle
        self.grasped_tool_frame = None
        self.X_Tool_Paddle = None

    @property
    def context(self):
        return self.simulator.get_mutable_context()
    
    @property
    def plant_context(self):
        return self.plant.GetMyContextFromRoot(self.context)

    def set_active_paddle_for_robot(self, robot_id):
        if robot_id == 1: # Left
            self.paddle_body = self.left_paddle
            print(f"[SimAdapter] Active paddle set to LEFT for Robot {robot_id}")
        else: # Right
            self.paddle_body = self.right_paddle
            print(f"[SimAdapter] Active paddle set to RIGHT for Robot {robot_id}")
        
        # Reset grasped flag for this paddle
        self._paddle_grasped = self.paddle_body in self.grasped_paddles
        if self._paddle_grasped:
            data = self.grasped_paddles[self.paddle_body]
            self.grasped_tool_frame = data['tool_frame']
            self.X_Tool_Paddle = data['X_Tool_Paddle']
        else:
            self.grasped_tool_frame = None
            self.X_Tool_Paddle = None

    @property
    def paddle_grasped(self):
        return self._paddle_grasped
    
    @paddle_grasped.setter
    def paddle_grasped(self, value):
        self._paddle_grasped = value

    def grasp_paddle(self, tool_frame):
        """
        Kinematically attach the paddle to the tool frame.
        """
        self.grasped_tool_frame = tool_frame
        
        # Calculate relative pose at current state to preserve it
        context = self.plant_context
        X_W_Tool = self.plant.CalcRelativeTransform(context, self.plant.world_frame(), tool_frame)
        X_W_Paddle = self.plant.EvalBodyPoseInWorld(context, self.paddle_body)
        
        self.X_Tool_Paddle = X_W_Tool.inverse() @ X_W_Paddle
        self._paddle_grasped = True
        
        # Store per-paddle data
        self.grasped_paddles[self.paddle_body] = {
            'tool_frame': tool_frame,
            'X_Tool_Paddle': self.X_Tool_Paddle
        }
        
        print(f"[SimAdapter] Paddle grasped by {tool_frame.name()}")

    def step(self, duration=0.01):
        """Advances simulation by duration."""
        # Update grasped object BEFORE step (and before physics might move it wrong, though SetPositions overrides)
        if self.paddle_grasped and self.grasped_tool_frame and self.puck_body:
            self._update_grasped_paddle()
            
        current_time = self.context.get_time()
        self.simulator.AdvanceTo(current_time + duration)
        
        # Update again after step to match robot's new position
        if self.paddle_grasped and self.grasped_tool_frame and self.puck_body:
            self._update_grasped_paddle()

    def _update_grasped_paddle(self):
        # Compute new paddle pose
        X_W_Tool = self.plant.CalcRelativeTransform(self.plant_context, self.plant.world_frame(), self.grasped_tool_frame)
        # X_W_Paddle = X_W_Tool @ self.X_Tool_Paddle
        
        # self.plant.SetFreeBodyPose(self.plant_context, self.paddle_body, X_W_Paddle)
        pass
        
        # Set velocity to match tool? Ideally yes, for physics interactions with puck
        # V_W_Tool = self.plant.CalcSpatialVelocity(self.plant_context, self.grasped_tool_frame, self.plant.world_frame(), self.plant.world_frame())
        # self.plant.SetFreeBodySpatialVelocity(self.plant_context, self.paddle_body, V_W_Tool)
        
    def get_robot_joint_positions(self, robot_model):
        return self.plant.GetPositions(self.plant_context, robot_model)
        
    def set_robot_joint_positions(self, robot_model, q):
        self.plant.SetPositions(self.plant_context, robot_model, q)
        
    def get_paddle_pose(self):
        return self.plant.EvalBodyPoseInWorld(self.plant_context, self.paddle_body)

    def get_puck_state(self):
        # Return position (3,), velocity (3,)
        pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.puck_body)
        spatial_vel = self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, self.puck_body)
        
        pos = pose.translation()
        vel = spatial_vel.translational()
        
        return pos, vel
