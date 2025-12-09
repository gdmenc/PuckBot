"""
Actuator-based robot controller for air hockey.
Uses PD control to track target positions via torque commands.
Now with STRIKE mode for aggressive shots toward goal!
"""
import numpy as np
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve
from pydrake.math import RollPitchYaw, RotationMatrix

from scripts.kinematics.strike_planner import StrikePlanner


class ActuatorBasedController:
    """
    Robot controller with strike/block mode switching.
    
    Modes:
    - STRIKE: Aggressive shots toward opponent goal
    - BLOCK: Defensive positioning to intercept puck
    """
    
    # Control modes
    STRIKE_MODE = "strike"
    BLOCK_MODE = "block"
    
    def __init__(self, plant, robot_model, gripper_model=None, enable_strike=True):  # Re-enabled with debug
        """
        Args:
            plant: MultibodyPlant
            robot_model: Robot model instance
            gripper_model: Gripper model instance (for frame lookup)
            enable_strike: Enable strike mode (requires shot planning)
        """
        self.plant = plant
        self.robot_model = robot_model
        
        # Get gripper frame
        frame_model = gripper_model if gripper_model is not None else robot_model
        self.gripper_frame = plant.GetFrameByName("body", frame_model)
        self.world_frame = plant.world_frame()
        
        self.table_height = 0.11
        
        # PD gains for joint control
        # These convert position error to torque
        self.kp = 100.0  # Proportional gain
        self.kd = 20.0   # Derivative gain
        
        # Simple joint limits (approximate for IIWA)
        self.q_min = np.array([-2.9, -2.0, -2.9, -2.0, -2.9, -2.0, -3.0])
        self.q_max = np.array([2.9, 2.0, 2.9, 2.0, 2.9, 2.0, 3.0])
        
        self.control_active = True
        self.target_q = None  # Current target joint positions
        self.target_qd = None  # Current target joint velocities (for strike mode)
        
        # Strike planner
        self.enable_strike = enable_strike
        if enable_strike:
            self.strike_planner = StrikePlanner(
                plant, robot_model, gripper_model,  # Pass gripper model!
                table_bounds={'x': (-1.0, 1.0), 'y': (-0.52, 0.52)},
                shot_speed=2.0,
                restitution=0.9
            )
            self.current_mode = self.STRIKE_MODE
            print("[CONTROLLER] Strike mode enabled! ✅")
        else:
            self.strike_planner = None
            self.current_mode = self.BLOCK_MODE
        
    def compute_torques(self, q_current, qd_current):
        """
        Compute joint torques using PD control.
        
        Args:
            q_current: Current joint positions
            qd_current: Current joint velocities
            
        Returns:
            Joint torques, or None if no target
        """
        if self.target_q is None:
            return None
        
        # PD control: tau = Kp * (q_target - q) - Kd * qd
        position_error = self.target_q - q_current
        torques = self.kp * position_error - self.kd * qd_current
        
        # Torque limits (safety)
        max_torque = 300.0  # N⋅m (conservative for IIWA)
        torques = np.clip(torques, -max_torque, max_torque)
        
        return torques
    
    def is_puck_threat(self, puck_pos, puck_vel, robot_x_side):
        """
        Check if puck requires reaction.
        
        ALWAYS react to pucks in reachable workspace for aggressive play.
        """
        # Check if puck is in reachable workspace
        if robot_x_side < 0:  # Left robot
            x_reachable = -0.9 <= puck_pos[0] <= 0.1
        else:  # Right robot  
            x_reachable = -0.1 <= puck_pos[0] <= 0.9
        
        y_reachable = -0.55 <= puck_pos[1] <= 0.55
        
        # React to ANY puck in workspace (including stationary)
        return x_reachable and y_reachable
    
    def solve_ik(self, target_pos, current_q):
        """
        Solve IK for target position with table-parallel constraint.
        
        CRITICAL: Uses ISOLATED context to avoid corrupting simulation state!
        """
        # Create FRESH isolated context for IK - DO NOT use simulation context!
        ik_context = self.plant.CreateDefaultContext()
        
        # Set ONLY robot joints in the isolated context
        self.plant.SetPositions(ik_context, self.robot_model, current_q)
        
        # Create IK solver with ISOLATED context
        ik = InverseKinematics(self.plant, ik_context)
        
        # Position constraint
        ik.AddPositionConstraint(
            frameB=self.gripper_frame,
            p_BQ=np.zeros(3),
            frameA=self.world_frame,
            p_AQ_lower=target_pos - 0.02,
            p_AQ_upper=target_pos + 0.02
        )
        
        # Orientation constraint (parallel to table)
        target_rotation = RotationMatrix(RollPitchYaw([0, 0, 0]))
        ik.AddOrientationConstraint(
            frameAbar=self.world_frame,
            R_AbarA=target_rotation,
            frameBbar=self.gripper_frame,
            R_BbarB=RotationMatrix.Identity(),
            theta_bound=0.2
        )
        
        # Solve
        prog = ik.prog()
        q_vars = ik.q()
        
        # Initial guess from isolated context
        full_q_guess = self.plant.GetPositions(ik_context)
        prog.SetInitialGuess(q_vars, full_q_guess)
        
        result = Solve(prog)
        
        if result.is_success():
            q_solution_full = result.GetSolution(q_vars)
            
            # Extract robot joints from solution
            context_temp = self.plant.CreateDefaultContext()
            self.plant.SetPositions(context_temp, q_solution_full)
            q_robot = self.plant.GetPositions(context_temp, self.robot_model)
            
            return True, np.clip(q_robot, self.q_min, self.q_max)
        
        return False, current_q
    
    def update(self, puck_pos, puck_vel, q_current, qd_current, context, robot_x_side):
        """
        Update controller - computes new target using strike or block mode.
        
        Returns:
            (has_torques, torques) - torque commands or None
        """
        if not self.control_active:
            return False, None
        
        # Check if we should react
        if not self.is_puck_threat(puck_pos, puck_vel, robot_x_side):
            # No threat - hold current position
            if self.target_q is None:
                self.target_q = q_current.copy()
            torques = self.compute_torques(q_current, qd_current)
            return True, torques
        
        # Adaptive prediction based on puck speed
        puck_speed = np.linalg.norm(puck_vel[:2])
        
        if puck_speed < 0.05:
            # Stationary puck (HIT mode) - CREATE A STRIKE PATH
            # Goal: Move THROUGH the puck at high speed toward opponent goal
            
            # 1. Determine goal to shoot at
            if robot_x_side > 0:  # Right robot
                goal_pos = np.array([-1.0, 0.0])  # Shoot toward left goal
            else:  # Left robot
                goal_pos = np.array([1.0, 0.0])  # Shoot toward right goal
            
            # 2. Calculate strike direction (puck → goal)
            puck_to_goal = goal_pos - puck_pos[:2]
            puck_to_goal_norm = np.linalg.norm(puck_to_goal)
            
            if puck_to_goal_norm > 0.01:
                strike_direction = puck_to_goal / puck_to_goal_norm
            else:
                # Fallback
                strike_direction = np.array([-1.0, 0.0]) if robot_x_side > 0 else np.array([1.0, 0.0])
            
            # 3. CRITICAL: Position target PAST the puck for follow-through
            # Robot will move THROUGH puck toward this point
            follow_through_distance = 0.3  # 30cm past puck
            intercept_pos_2d = puck_pos[:2] + strike_direction * follow_through_distance
            
            # 4. Set high target velocity for aggressive strike
            # This will be used by strike planner to set joint velocities
            # The robot should be moving at ~1.0 m/s when it hits the puck
            
        else:
            # Moving puck - predict intercept with adaptive lookahead
            # Faster pucks need even earlier intercept for better planning
            dt = np.clip(0.6 / max(puck_speed, 0.1), 0.15, 0.7)  # 0.15-0.7s lookahead (doubled from 0.1-0.5s)
            predicted_pos = puck_pos[:2] + puck_vel[:2] * dt
            
            # Position robot between puck and goal for defense
            if robot_x_side > 0:  # Right robot
                # Defend right goal, position between puck and (+1, 0)
                goal_pos = np.array([1.0, 0.0])
            else:  # Left robot
                # Defend left goal, position between puck and (-1, 0)
                goal_pos = np.array([-1.0, 0.0])
            
            # Intercept along line from puck to goal
            puck_to_goal = goal_pos - predicted_pos
            puck_to_goal_dist = np.linalg.norm(puck_to_goal)
            
            if puck_to_goal_dist > 0.1:
                # Position 10cm in front of predicted puck position toward goal (reduced from 20cm)
                intercept_offset = 0.1 * (puck_to_goal / puck_to_goal_dist)
                intercept_pos_2d = predicted_pos + intercept_offset
            else:
                intercept_pos_2d = predicted_pos
        
        # PADDLE GEOMETRY: Offset gripper position so paddle edge hits puck
        # Paddle radius ~0.095m
        paddle_radius = 0.095
        
        if puck_speed < 0.05:
            # Stationary puck: Target is already PAST the puck for follow-through
            # Paddle will hit puck on the way through - no additional offset needed
            pass
        else:
            # Moving puck: approach from direction opposite to velocity
            approach_dir = -puck_vel[:2] / max(puck_speed, 0.01)
            intercept_pos_2d += paddle_radius * approach_dir
        
        # Clamp to reachable workspace AFTER paddle offset
        # Robot base is at (±0.4, 0, 0), so keep robot on its side of table
        if robot_x_side > 0:  # Right robot at (+0.4, 0, 0)
            intercept_pos_2d[0] = np.clip(intercept_pos_2d[0], 0.1, 0.85)  # Stay on right side
        else:  # Left robot at (-0.4, 0, 0)
            intercept_pos_2d[0] = np.clip(intercept_pos_2d[0], -0.85, -0.1)  # Stay on left side
        
        intercept_pos_2d[1] = np.clip(intercept_pos_2d[1], -0.5, 0.5)  # Y limits
        
        intercept_pos_3d = np.array([intercept_pos_2d[0], intercept_pos_2d[1], self.table_height])
        
        # Use strike mode if enabled
        if self.enable_strike and self.strike_planner is not None:
            # STRIKE MODE: Plan aggressive shot toward goal
            success, q_target, qd_target, v_mallet = self.strike_planner.plan_strike(
                puck_pos, puck_vel,
                intercept_pos_3d,
                0.25,  # intercept_time (not used yet)
                q_current,
                robot_x_side
            )
            
            if success:
                self.target_q = q_target
                self.target_qd = qd_target
                self.current_mode = self.STRIKE_MODE
            else:
                # Fallback to block mode
                success_ik, q_target = self.solve_ik(intercept_pos_3d, q_current)
                if success_ik:
                    self.target_q = q_target
                    self.target_qd = np.zeros(7)
                self.current_mode = self.BLOCK_MODE
        else:
            # BLOCK MODE: Defensive positioning
            success, q_new = self.solve_ik(intercept_pos_3d, q_current)
            if success:
                self.target_q = q_new
                self.target_qd = np.zeros(7)
            self.current_mode = self.BLOCK_MODE
        
        # Compute torques to track target
        torques = self.compute_torques(q_current, qd_current)
        return True, torques
