"""
Paddle Grasping Module
Implements grasp sampling and execution for paddle manipulation
"""
import numpy as np
import trimesh
from pydrake.all import RigidTransform
from typing import Optional
from scripts.grasp.grasp_sampler import sample_grasp, compute_prepick_pose


class PaddleGrasper:
    """
    Manages the paddle grasping sequence using grasp sampling.
    """
    
    def __init__(
        self,
        env,
        robot_model,
        tool_frame_name: str,
        motion_controller: Optional = None,
        tool_model_instance = None
    ):
        """
        Args:
            env: SimAdapter instance
            robot_model: Robot model instance
            tool_frame_name: Name of tool/end-effector frame
            motion_controller: Optional motion controller
            tool_model_instance: Model instance for the tool
        """
        self.env = env
        self.robot_model = robot_model
        self.tool_frame_name = tool_frame_name
        self.tool_model_instance = tool_model_instance
        
        # Get tool frame
        if tool_model_instance:
            self.tool_frame = env.plant.GetFrameByName(tool_frame_name, tool_model_instance)
        else:
            self.tool_frame = env.plant.GetFrameByName(tool_frame_name, robot_model)
        
        # Load paddle mesh for grasp sampling
        self.paddle_mesh = self._load_paddle_mesh()
        
    def _load_paddle_mesh(self) -> trimesh.Trimesh:
        """Load the paddle mesh for grasp sampling"""
        try:
            mesh_path = "/Users/gdmen/MIT/F2025/6.4210/PuckBot/scripts/env/assets/models/paddle/paddle.obj"
            mesh = trimesh.load(mesh_path, force="mesh")
            # Apply scaling (from SDF: scale 2.0000E-02)
            mesh.apply_scale(1)
            return mesh
        except Exception as e:
            print(f"[PaddleGrasper] Warning: Could not load paddle mesh: {e}")
            print("[PaddleGrasper] Using default mesh")
            # Create simple cylinder as fallback
            return trimesh.creation.cylinder(radius=0.05, height=0.03)
    
    def _load_precomputed_grasp(self, grasp_index: int = 0) -> RigidTransform:
        """Load a precomputed grasp from the pickle file"""
        import pickle
        try:
            grasp_file = "/Users/gdmen/MIT/F2025/6.4210/PuckBot/scripts/grasp/precomputed_paddle_grasps.pkl"
            with open(grasp_file, 'rb') as f:
                grasps = pickle.load(f)
            
            if len(grasps) == 0:
                print("[PaddleGrasper] No precomputed grasps found")
                return None
            
            # Select grasp (use index modulo list length)
            idx = grasp_index % len(grasps)
            grasp_data = grasps[idx]
            
            # Convert matrix back to RigidTransform
            X_OG = RigidTransform(grasp_data['pose_matrix'])
            print(f"[PaddleGrasper] Loaded grasp {idx+1}/{len(grasps)} (quality={grasp_data['quality']:.3f})")
            return X_OG
            
        except Exception as e:
            print(f"[PaddleGrasper] Error loading precomputed grasp: {e}")
            return None
    
    def execute_grasp_sequence(self) -> bool:
        """
        Execute the full grasp sequence:
        1. Sample a grasp on the paddle
        2. Move to pre-pick
        3. Move to grasp
        4. Close gripper (kinematic attachment)
        5. Lift
        6. Move to ready position
        
        Returns:
            True if grasp succeeded
        """
        print("\n=== Starting Paddle Grasp Sequence ===")
        
        # Get current paddle pose
        X_WO = self.env.get_paddle_pose()
        if X_WO is None:
            print("[PaddleGrasper] Error: Could not get paddle pose")
            return False
        
        # Load precomputed grasp
        print("Step 1: Loading precomputed grasp...")
        X_OG = self._load_precomputed_grasp()
        if X_OG is None:
            print("[PaddleGrasper] Error: Could not load precomputed grasp")
            return False
        
        # Transform to world frame
        X_WG_grasp = X_WO @ X_OG
        
        # Compute pre-pick pose (offset along -y in gripper frame)
        from scripts.grasp.grasp_sampler import compute_prepick_pose
        X_WG_prepick = compute_prepick_pose(X_WG_grasp, offset=0.15)
        
        # Execute movement sequence using simple position control
        print("Step 2: Moving to pre-pick position...")
        success = self._move_to_pose(X_WG_prepick, duration=2.0)
        if not success:
            print("[PaddleGrasper] Warning: Failed to reach pre-pick, continuing...")
        
        print("Step 3: Moving to grasp position...")
        success = self._move_to_pose(X_WG_grasp, duration=1.5)
        if not success:
            print("[PaddleGrasper] Warning: Failed to reach grasp position")
        
        print("Step 4: Grasping paddle (kinematic attachment)...")
        self._grasp_paddle()
        
        print("Step 5: Lifting paddle...")
        X_WG_lift = RigidTransform(X_WG_grasp.rotation(), X_WG_grasp.translation() + np.array([0, 0, 0.15]))
        success = self._move_to_pose(X_WG_lift, duration=1.5)
        if not success:
            print("[PaddleGrasper] Warning: Failed to lift paddle")
        
        print("Step 6: Moving to ready position...")
        # Determine ready position based on robot side
        robot_x = 0.6 if self.robot_model == self.env.robot2_model else -0.6
        X_WG_ready = RigidTransform(X_WG_lift.rotation(), [robot_x, 0.0, 0.25])
        success = self._move_to_pose(X_WG_ready, duration=2.0)
        
        print("=== Paddle Grasp Sequence Complete ===\n")
        return True
    
    def _move_to_pose(self, X_WG_target: RigidTransform, duration: float = 2.0, dt: float = 0.01) -> bool:
        """
        Move robot to target pose using IK solver
        
        Args:
            X_WG_target: Target gripper pose in world frame
            duration: Duration of movement
            dt: Time step
            
        Returns:
            True if successful
        """
        from scripts.kinematics.motion_controller import MotionController
        
        # Create motion controller if needed
        if not hasattr(self, 'motion_controller'):
            robot_id = 1 if self.robot_model == self.env.robot1_model else 2
            self.motion_controller = MotionController(self.env, robot_id=robot_id)
        
        # Get target position from pose
        target_position = X_WG_target.translation()
        
        # Get current robot configuration
        q_current = self.env.get_robot_joint_positions(self.robot_model)
        
        # Plan trajectory to target position
        print(f"    Planning to position: {target_position}")
        success, trajectory = self.motion_controller.planner.plan_to_position(
            target_position,
            q_current,
            self.env.plant_context,
            duration=duration,
            dt=dt
        )
        
        if not success:
            print(f"    [IK FAILED] Could not reach target: {target_position}")
            return False
        
        # Execute trajectory
        print(f"    Executing trajectory ({trajectory.n_points} points)...")
        publish_interval = max(1, trajectory.n_points // 20)  # ~20 updates
        
        for i in range(trajectory.n_points):
            q = trajectory.positions[i]
            self.env.set_robot_joint_positions(self.robot_model, q)
            self.env.step(duration=dt)
            
            # Publish periodically for visualization
            if i % publish_interval == 0:
                self.env.diagram.ForcedPublish(self.env.context)
        
        print(f"    ✓ Reached target")
        return True
    
    def _grasp_paddle(self):
        """
        Kinematically attach paddle to gripper using SimAdapter
        """
        if hasattr(self.env, "grasp_paddle"):
            # Get current transforms
            context = self.env.plant_context
            X_WT = self.env.plant.CalcRelativeTransform(
                context, self.env.plant.world_frame(), self.tool_frame
            )
            X_WP = self.env.plant.EvalBodyPoseInWorld(context, self.env.paddle_body)
            
            # Compute desired relative pose (grasp from above/side)
            X_TP = X_WT.inverse() @ X_WP
            
            # Adjust paddle orientation if needed
            # Set it so paddle is aligned with gripper
            X_WP_adjusted = X_WT @ X_TP
            
            self.env.plant.SetFreeBodyPose(
                self.env.plant_context,
                self.env.paddle_body,
                X_WP_adjusted
            )
            
            # Attach via SimAdapter
            self.env.grasp_paddle(self.tool_frame)
            print(f"[PaddleGrasper] Paddle attached to {self.tool_frame.name()}")
        else:
            print("[PaddleGrasper] Error: SimAdapter does not support kinematic grasping")
