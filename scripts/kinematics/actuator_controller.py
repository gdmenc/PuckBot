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
from scripts.kinematics.optimal_strike_planner import OptimalStrikePlanner
from scripts.kinematics.states import PuckState
from scripts.kinematics.workspace_safety import WorkspaceSafety


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
        
        # CRITICAL: Height must be high enough to prevent elbow-table collision
        # Puck center at 0.11, but we need clearance for arm geometry
        self.table_height = 0.120  # Raised from 0.105 to avoid elbow hitting table
        
        # PD gains for joint control
        # These convert position error to torque
        self.kp = 150.0  # Proportional gain
        self.kd = 20.0   # Derivative gain
        
        # Simple joint limits (approximate for IIWA)
        self.q_min = np.array([-2.9, -2.0, -2.9, -2.0, -2.9, -2.0, -3.0])
        self.q_max = np.array([2.9, 2.0, 2.9, 2.0, 2.9, 2.0, 3.0])
        
        self.control_active = True
        self.target_q = None  # Current target joint positions
        self.target_qd = None  # Current target joint velocities (for strike mode)
        self.previous_target_q = None  # For smoothing
        self.target_change_threshold = 0.15  # radians - reduced from 0.3 for faster updates
        
        # Push-through tracking
        self.last_contact_time = -10.0  # Time of last puck contact
        self.last_contact_pos = None  # Paddle position at contact
        self.push_duration = 0.3  # How long to push after contact (seconds)
        self.puck_radius = 0.04
        self.paddle_radius = 0.0475
        
        # Intercept tracking (prevent retreating target problem)
        self.last_intercept_pos = None  # Last planned intercept position
        self.last_intercept_time = -10.0  # When we last planned intercept
        
        # Determine robot side from model name
        model_name = plant.GetModelInstanceName(robot_model)
        if "right" in model_name.lower():
            robot_side_value = 1.0
        elif "left" in model_name.lower():
            robot_side_value = -1.0
        else:
            robot_side_value = 1.0  # Default to right
            print(f"[WARNING] Could not determine robot side from name '{model_name}', defaulting to right")
        
        # Workspace safety to prevent wall collisions
        self.workspace_safety = WorkspaceSafety(
            robot_side=robot_side_value,
            table_bounds={'x': (-1.064, 1.064), 'y': (-0.609, 0.609)}
        )
        print(f"[CONTROLLER] Workspace safety enabled: X={self.workspace_safety.x_min:.3f} to {self.workspace_safety.x_max:.3f}, Y={self.workspace_safety.y_min:.3f} to {self.workspace_safety.y_max:.3f}")
        
        # Strike planner
        self.enable_strike = enable_strike
        if enable_strike:
            self.strike_planner = StrikePlanner(
                plant, robot_model, gripper_model,  # Pass gripper model!
                table_bounds={'x': (-1.0, 1.0), 'y': (-0.52, 0.52)},
                shot_speed=2.0,
                restitution=0.9
            )
            # NEW: Optimal strike planner for predictive intercepts
            self.optimal_planner = OptimalStrikePlanner(
                table_bounds={'x': (-1.0, 1.0), 'y': (-0.52, 0.52)},
                puck_radius=0.04,
                paddle_radius=0.0475,
                max_robot_speed=1.5,
                desired_shot_speed=2.5
            )
            self.current_mode = self.STRIKE_MODE
            print("[CONTROLLER] Optimal strike mode enabled! ✅")
        else:
            self.strike_planner = None
            self.optimal_planner = None
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
        
        KEY CHANGE: React MUCH earlier for better anticipation!
        Don't wait until puck is in workspace - track it early!
        """
        # EXPANDED reaction zone - react to puck well before it's in reach
        # This allows robot to position itself with plenty of time
        if robot_x_side < 0:  # Left robot
            # React to anything on left half of table or approaching
            x_reachable = puck_pos[0] <= 0.2  # Expanded from -0.9 to 0.1
        else:  # Right robot  
            # React to anything on right half of table or approaching
            x_reachable = puck_pos[0] >= -0.2  # Expanded from -0.1 to 0.9
        
        y_reachable = -0.6 <= puck_pos[1] <= 0.6  # Slightly expanded
        
        # Also react if puck is APPROACHING (even if far away)
        # This enables early anticipatory positioning
        puck_speed = np.linalg.norm(puck_vel[:2])
        is_approaching = False
        if puck_speed > 0.1:  # Puck is moving
            # Check if velocity points toward robot
            if robot_x_side < 0:  # Left robot
                is_approaching = puck_vel[0] < -0.05  # Moving left
            else:  # Right robot
                is_approaching = puck_vel[0] > 0.05  # Moving right
        
        # React if in expanded zone OR approaching
        return (x_reachable and y_reachable) or is_approaching
    
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
        # CRITICAL: Keep paddle PERFECTLY FLAT with table
        # This ensures paddle makes proper contact with puck (not over/under)
        target_rotation = RotationMatrix(RollPitchYaw([0, 0, 0]))  # Perfectly level
        ik.AddOrientationConstraint(
            frameAbar=self.world_frame,
            R_AbarA=target_rotation,
            frameBbar=self.gripper_frame,
            R_BbarB=RotationMatrix.Identity(),
            theta_bound=0.01  # 0.01 rad = ~0.6 degrees - VERY flat!
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
    
    def _is_in_corner_zone(self, position_2d: np.ndarray, margin: float = 0.25) -> bool:
        """
        Check if position is in a dangerous corner zone where arm might collide.
        
        Args:
            position_2d: [x, y] position
            margin: Distance from corner to consider dangerous (meters)
            
        Returns:
            True if in corner zone
        """
        x, y = position_2d[0], position_2d[1]
        
        # Check distance from all 4 corners
        # Corners are at table bounds extremes
        workspace = self.workspace_safety.get_workspace_bounds()
        
        # Near X edge?
        near_x_min = abs(x - workspace['x'][0]) < margin
        near_x_max = abs(x - workspace['x'][1]) < margin
        near_x_edge = near_x_min or near_x_max
        
        # Near Y edge?
        near_y_min = abs(y - workspace['y'][0]) < margin
        near_y_max = abs(y - workspace['y'][1]) < margin
        near_y_edge = near_y_min or near_y_max
        
        # In corner if near both X and Y edges
        return near_x_edge and near_y_edge
    
    def _get_safe_corner_position(self, puck_pos_2d: np.ndarray, robot_x_side: float) -> np.ndarray:
        """
        Get a safe position away from corner when puck is stuck there.
        
        Args:
            puck_pos_2d: [x, y] puck position
            robot_x_side: Robot side (+1 for right, -1 for left)
            
        Returns:
            [x, y] safe position to wait at
        """
        # Pull back from corner toward center of robot workspace
        workspace = self.workspace_safety.get_workspace_bounds()
        
        # Position at 30cm from corner, toward workspace center
        if robot_x_side > 0:  # Right robot
            safe_x = workspace['x'][1] - 0.30  # Pull back from right edge
        else:  # Left robot
            safe_x = workspace['x'][0] + 0.30  # Pull back from left edge
        
        # Y: Move toward center (y=0)
        if puck_pos_2d[1] > 0:
            safe_y = workspace['y'][1] - 0.30  # Pull back from top
        else:
            safe_y = workspace['y'][0] + 0.30  # Pull back from bottom
        
        return np.array([safe_x, safe_y])
    
    def update(self, puck_pos, puck_vel, q_current, qd_current, context, robot_x_side):
        """
        Update controller - computes new target using OPTIMAL STRIKE planner.
        
        NEW: Uses predictive trajectory sampling and curved approach paths!
        PUSH-THROUGH: Continues pushing puck forward after contact!
        
        Returns:
            (has_torques, torques) - torque commands or None
        """
        if not self.control_active:
            return False, None
        
        # Get current gripper position
        gripper_pose = self.plant.CalcRelativeTransform(
            context,
            self.world_frame,
            self.gripper_frame
        )
        gripper_pos = gripper_pose.translation()
        
        # PUSH-THROUGH MODE: If we just made contact and puck is moving slowly,
        # continue pushing forward instead of stopping!
        current_time = context.get_time()
        time_since_contact = current_time - self.last_contact_time
        
        # Check if paddle is in contact with puck
        paddle_to_puck = np.linalg.norm(gripper_pos[:2] - puck_pos[:2])
        is_in_contact = paddle_to_puck < (self.paddle_radius + self.puck_radius + 0.02)  # 2cm tolerance
        
        if is_in_contact:
            self.last_contact_time = current_time
            self.last_contact_pos = gripper_pos.copy()
        
        # If recently contacted and puck is slow, PUSH THROUGH!
        puck_speed = np.linalg.norm(puck_vel[:2])
        if time_since_contact < self.push_duration and puck_speed < 0.8:  # Puck not moving fast enough
            # Keep pushing in the direction toward goal
            if robot_x_side > 0:
                goal_pos = np.array([-1.0, 0.0])  # Left goal
            else:
                goal_pos = np.array([1.0, 0.0])  # Right goal
            
            # Push target = puck position + direction toward goal
            push_dir = goal_pos - puck_pos[:2]
            push_dir_norm = np.linalg.norm(push_dir)
            if push_dir_norm > 0.01:
                push_dir = push_dir / push_dir_norm
                # Target position: push PAST puck toward goal
                # REDUCED: Keep closer to puck for better contact
                push_distance = 0.08  # 8cm past puck (was 15cm)
                push_target_2d = puck_pos[:2] + push_dir * push_distance
                push_target_2d = self.workspace_safety.clamp_to_workspace(push_target_2d)
                push_target_3d = np.array([push_target_2d[0], push_target_2d[1], self.table_height])
                
                success_ik, q_target = self.solve_ik(push_target_3d, q_current)
                if success_ik:
                    self.target_q = q_target
                    self.target_qd = np.zeros(7)
                    print(f"[PUSH-THROUGH] Pushing {push_distance:.2f}m past puck, Speed={puck_speed:.2f} m/s")
                    torques = self.compute_torques(q_current, qd_current)
                    return True, torques
        
        # Check if we should react
        if not self.is_puck_threat(puck_pos, puck_vel, robot_x_side):
            # No threat - hold current position
            # Debug: Show why we're not reacting (useful for troubleshooting)
            puck_speed = np.linalg.norm(puck_vel[:2])
            if puck_speed > 0.05:  # Only log for moving pucks
                print(f"[NO REACT] Puck at {puck_pos[:2]}, vel={puck_vel[:2]}, not a threat")
            
            if self.target_q is None:
                self.target_q = q_current.copy()
            torques = self.compute_torques(q_current, qd_current)
            return True, torques
        
        # Get current gripper position for optimal planning (already calculated above, but keep for clarity)
        gripper_pose = self.plant.CalcRelativeTransform(
            context,
            self.world_frame,
            self.gripper_frame
        )
        gripper_pos = gripper_pose.translation()
        
        # Check if gripper is too close to walls
        self.workspace_safety.warn_if_close_to_wall(gripper_pos, threshold=0.08)
        
        # Use OPTIMAL STRIKE PLANNER for intelligent intercept selection
        if self.enable_strike and self.optimal_planner is not None:
            # Create puck state for optimal planner
            current_time = context.get_time()
            puck_state = PuckState(
                position=puck_pos,
                velocity=puck_vel,
                t=current_time
            )
            
            # CRITICAL FIX FOR RETREATING: Check if we're already close to a good intercept position
            # AND actually in the puck's trajectory path
            # Only hold if BOTH conditions met!
            if hasattr(self, 'last_intercept_pos') and self.last_intercept_pos is not None:
                distance_to_intercept = np.linalg.norm(gripper_pos[:2] - self.last_intercept_pos[:2])
                
                # Also check if we're actually IN the puck's path (not just near some old position)
                puck_speed = np.linalg.norm(puck_vel[:2])
                in_puck_path = False
                
                if puck_speed > 0.1:  # Only check path for moving pucks
                    # Calculate perpendicular distance from robot to puck's trajectory line
                    # Trajectory line: puck_pos + t * puck_vel
                    puck_to_robot = gripper_pos[:2] - puck_pos[:2]
                    puck_vel_normalized = puck_vel[:2] / puck_speed
                    
                    # Project onto trajectory
                    projection_length = np.dot(puck_to_robot, puck_vel_normalized)
                    
                    # Only consider if robot is AHEAD of puck (positive projection)
                    if projection_length > 0:
                        # Perpendicular distance to trajectory line
                        perp_distance = np.linalg.norm(puck_to_robot - projection_length * puck_vel_normalized)
                        in_puck_path = perp_distance < 0.10  # Within 10cm of trajectory
                else:
                    # Stationary puck - just check distance
                    in_puck_path = np.linalg.norm(gripper_pos[:2] - puck_pos[:2]) < 0.20
                
                # HOLD only if VERY close to intercept AND in puck's path
                # Increased deadband to prevent micro-adjustments (was 0.10)
                if distance_to_intercept < 0.25 and in_puck_path:
                    print(f"[HOLD] In puck path, waiting (dist to intercept={distance_to_intercept:.3f}m)")
                    torques = self.compute_torques(q_current, qd_current)
                    return True, torques
            
            # Use workspace safety bounds
            workspace = self.workspace_safety.get_workspace_bounds()
            
            # ========================================================================
            # SIMPLE DIRECT PUCK TRACKING
            # ========================================================================
            # Use environment's EXACT physics for prediction
            # No complex modes - just follow the puck!
            
            from scripts.kinematics.direct_tracker import get_paddle_target,validate_target_reachable
            
            puck_speed = np.linalg.norm(puck_vel[:2])
            
            # INCREASED lookahead - predict MUCH further ahead so we arrive BEFORE puck
            # Robot was arriving too late, so we need more lead time
            if puck_speed > 1.0:
                lookahead = 1.2  # Was 0.6 - doubled for fast pucks
            elif puck_speed > 0.5:
                lookahead = 0.9  # Was 0.4 - more than doubled
            else:
                lookahead = 0.7  # Was 0.3 - more than doubled for slow pucks
            
            print(f"[TRACK] Puck at {puck_pos[:2]}, speed={puck_speed:.2f} m/s, lookahead={lookahead:.2f}s")
            
            # Get target using physics-accurate prediction
            target_pos = get_paddle_target(
                puck_pos, puck_vel,
                robot_x_side,
                lookahead_time=lookahead,
                table_bounds=workspace
            )
            
            # Validate reachability and safety
            target_pos = validate_target_reachable(
                target_pos,
                robot_x_side,
                max_reach=1.0,
                table_bounds=workspace
            )
            
            print(f"[TARGET] Moving to [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")
            
            # Create intercept candidate
            from scripts.kinematics.optimal_strike_planner import InterceptCandidate
            intercept = InterceptCandidate(
                position=target_pos,
                time=lookahead,
                puck_velocity=puck_vel,
                score=100.0,
                approach_angle=0.0
            )
            
            if intercept is not None:
                # Optimal intercept found!
                goal_pos = self.optimal_planner.get_goal_position(robot_x_side)
                
                # CRITICAL: Use FUTURE intercept position, not current puck position!
                # intercept.position is WHERE the puck WILL BE, not where it is NOW
                
                # ADAPTIVE FOLLOW-THROUGH: Adjust based on puck speed
                # Slower pucks (stationary) need less follow-through for precision
                # Faster pucks need more follow-through for momentum transfer
                puck_speed = np.linalg.norm(puck_vel[:2])
                if puck_speed < 0.1:
                    # Stationary - moderate follow-through for power
                    follow_through_dist = 0.18  # Increased from 0.12
                elif puck_speed < 0.5:
                    # Slow moving - strong follow-through
                    follow_through_dist = 0.25  # Increased from 0.18
                else:
                    # Fast moving - maximum follow-through for powerful deflection
                    follow_through_dist = 0.35  # Increased from 0.25
                
                # Get strike target (position PAST intercept for follow-through)
                strike_target = self.optimal_planner.compute_strike_target_position(
                    intercept,
                    goal_pos,
                    follow_through=follow_through_dist
                )
                
                # Get required velocity for strike
                required_vel = self.optimal_planner.compute_required_velocity(
                    intercept,
                    goal_pos
                )
                
                #CRITICAL FIX: Account for paddle geometry!
                # intercept.position is where PUCK CENTER will be
                # But gripper needs to target position OFFSET by combined radii
                # Otherwise robot stops (paddle_radius + puck_radius) short!
                
                # CORNER SAFETY: Check if puck is in a VERY tight corner zone
                # Only avoid extreme corners where arm will definitely collide
                # FURTHER REDUCED: 8cm margin for truly tight corners only
                puck_in_corner = self._is_in_corner_zone(intercept.position[:2], margin=0.08)
                
                if puck_in_corner:
                    # DON'T chase pucks into TIGHT corners - arm will collide!
                    # But only in truly dangerous spots
                    print(f"[CORNER AVOID] Puck in tight corner at {intercept.position[:2]}, pulling back")
                    
                    # Position at safe distance from corner
                    safe_pos = self._get_safe_corner_position(intercept.position[:2], robot_x_side)
                    safe_pos_3d = np.array([safe_pos[0], safe_pos[1], self.table_height])
                    
                    success_ik, q_target = self.solve_ik(safe_pos_3d, q_current)
                    if success_ik:
                        self.target_q = q_target
                        self.target_qd = np.zeros(7)
                        torques = self.compute_torques(q_current, qd_current)
                        return True, torques
                    else:
                        # If can't find safe position, try to hit anyway
                        print(f"[CORNER] Failed to find safe position, attempting strike anyway")
                
                # Calculate direction from puck to goal
                puck_to_goal = goal_pos - intercept.position[:2]
                puck_to_goal_norm = np.linalg.norm(puck_to_goal)
                
                if puck_to_goal_norm > 0.01:
                    # Direction we want to hit from (opposite of shot direction)
                    approach_dir = -puck_to_goal / puck_to_goal_norm
                    
                    # Offset by combined radii so paddle surface touches puck surface
                    # REDUCED: User reports growing distance - was too far!
                    # Experimenting with smaller offset
                    contact_offset = 0.01 # 5cm - simplified approach (was 9.7cm)
                    
                    # Gripper target = puck center + offset in approach direction
                    adjusted_intercept = intercept.position[:2] + approach_dir * contact_offset
                    adjusted_intercept_3d = np.array([adjusted_intercept[0], adjusted_intercept[1], self.table_height])
                    
                    # Debug: Show actual offset applied
                    actual_offset = np.linalg.norm(adjusted_intercept - intercept.position[:2])
                    print(f"  [OFFSET] Applied: {actual_offset:.3f}m, Puck at {intercept.position[:2]}, Target: {adjusted_intercept}")
                else:
                    # Fallback if can't determine direction
                    adjusted_intercept_3d = intercept.position
                
                # Use IK on ADJUSTED position (accounts for paddle size)
                success_ik, q_target = self.solve_ik(adjusted_intercept_3d, q_current)
                
                if success_ik:
                    # SMOOTHING: Blend with previous target for smooth motion
                    # SIMPLIFIED: Always blend, don't skip updates
                    if self.previous_target_q is not None:
                        # Blend 80% new, 20% old for responsiveness with smoothness
                        alpha = 0.80  # Increased from 0.7 for faster response
                        q_target = alpha * q_target + (1 - alpha) * self.previous_target_q
                    
                    self.target_q = q_target
                    self.previous_target_q = q_target.copy()
                    self.target_qd = np.zeros(7)
                    self.current_mode = self.STRIKE_MODE
                    
                    # Store intercept position to prevent retreating
                    self.last_intercept_pos = intercept.position.copy()
                    self.last_intercept_time = current_time
                    
                    # Debug output
                    puck_speed = np.linalg.norm(puck_vel[:2])
                    time_to_intercept = intercept.time - current_time
                    required_speed = np.linalg.norm(required_vel)
                    print(f"[INTERCEPT] Puck at {puck_pos[:2]}, will be at {intercept.position[:2]} in {time_to_intercept:.2f}s")
                    print(f"  Required paddle speed: {required_speed:.2f} m/s toward goal")
                    if puck_speed < 0.05:
                        print(f"  [STATIONARY] Approach: {intercept.position[:2]}, Score: {intercept.score:.1f}")
                    
                    torques = self.compute_torques(q_current, qd_current)
                    return True, torques
            
            # ANTICIPATORY POSITIONING: Even if no valid intercept in workspace,
            # move toward where puck is heading (get ready!)
            else:
                print(f"[NO INTERCEPT] Optimal planner returned None - using anticipatory positioning")
                # Predict where puck will be - REDUCED from 1.5s
                # Too far ahead causes robot to overshoot
                lookahead_time = 0.8  # 0.8s ahead (was 1.5s)
                predicted_pos_2d = puck_pos[:2] + puck_vel[:2] * lookahead_time
                
                # Move toward that direction, but stay in workspace
                predicted_pos_2d = self.workspace_safety.clamp_to_workspace(predicted_pos_2d)
                predicted_pos_3d = np.array([predicted_pos_2d[0], predicted_pos_2d[1], self.table_height])
                
                # Use IK to position robot in anticipation
                success_ik, q_target = self.solve_ik(predicted_pos_3d, q_current)
                if success_ik:
                    # Blend with previous target for very smooth anticipatory motion
                    if self.previous_target_q is not None:
                        alpha = 0.8  # 80% new, 20% old
                        q_target = alpha * q_target + (1 - alpha) * self.previous_target_q
                    
                    self.target_q = q_target
                    self.previous_target_q = q_target.copy()
                    self.target_qd = np.zeros(7)
                    self.current_mode = self.BLOCK_MODE
                    print(f"[ANTICIPATE] Moving toward predicted position {predicted_pos_2d} (0.8s ahead)")
                    torques = self.compute_torques(q_current, qd_current)
                    return True, torques
                else:
                    print(f"[WARN] IK failed for anticipatory position {predicted_pos_3d[:2]}, holding current")
                    # IK failed - just hold position
                    if self.target_q is None:
                        self.target_q = q_current.copy()
                    torques = self.compute_torques(q_current, qd_current)
                    return True, torques
        
        # FALLBACK: Original simple prediction if optimal planner fails
        puck_speed = np.linalg.norm(puck_vel[:2])
        
        if puck_speed < 0.05:
            # Stationary puck - simple follow-through
            if robot_x_side > 0:
                goal_pos = np.array([-1.0, 0.0])  # Right robot → Left goal ✓
            else:
                goal_pos = np.array([1.0, 0.0])  # Left robot → Right goal ✓
            
            puck_to_goal = goal_pos - puck_pos[:2]
            puck_to_goal_norm = np.linalg.norm(puck_to_goal)
            
            if puck_to_goal_norm > 0.01:
                strike_direction = puck_to_goal / puck_to_goal_norm
            else:
                strike_direction = np.array([-1.0, 0.0]) if robot_x_side > 0 else np.array([1.0, 0.0])
            
            follow_through_distance = 0.25
            intercept_pos_2d = puck_pos[:2] + strike_direction * follow_through_distance
        else:
            # Moving puck - simple prediction
            dt = np.clip(0.5 / max(puck_speed, 0.1), 0.15, 0.6)
            predicted_pos = puck_pos[:2] + puck_vel[:2] * dt
            
            # FIXED: Corrected goal direction (was backwards!)
            if robot_x_side > 0:
                goal_pos = np.array([-1.0, 0.0])  # Right robot → Left goal ✓
            else:
                goal_pos = np.array([1.0, 0.0])  # Left robot → Right goal ✓
            
            puck_to_goal = goal_pos - predicted_pos
            puck_to_goal_dist = np.linalg.norm(puck_to_goal)
            
            if puck_to_goal_dist > 0.1:
                intercept_offset = 0.1 * (puck_to_goal / puck_to_goal_dist)
                intercept_pos_2d = predicted_pos + intercept_offset
            else:
                intercept_pos_2d = predicted_pos
            
            # Paddle offset for moving puck - ENSURE we're on correct side
            approach_dir = -puck_vel[:2] / max(puck_speed, 0.01)
            
            # CRITICAL: Check if we'd be on wrong side of puck (behind it)
            # Robot should be between puck and its own goal, not behind puck
            potential_pos = intercept_pos_2d + 0.095 * approach_dir
            
            adjusted = False
            # Check if potential position is "behind" the puck (toward wrong goal)
            if robot_x_side > 0:  # Right robot
                # Should be on right side (positive X) of puck, not left
                if potential_pos[0] < puck_pos[0] - 0.1:  # Too far left = behind puck
                    print(f"[WARN] Would be behind puck! Adjusting approach...")
                    # Position on the right side of puck instead
                    intercept_pos_2d = puck_pos[:2] + np.array([0.1, 0.0])
                    adjusted = True
            else:  # Left robot
                # Should be on left side (negative X) of puck, not right
                if potential_pos[0] > puck_pos[0] + 0.1:  # Too far right = behind puck
                    print(f"[WARN] Would be behind puck! Adjusting approach...")
                    # Position on the left side of puck instead
                    intercept_pos_2d = puck_pos[:2] + np.array([-0.1, 0.0])
                    adjusted = True
            
            if not adjusted:
                intercept_pos_2d += 0.095 * approach_dir
        
        # Clamp to safe workspace using workspace safety
        intercept_pos_2d = self.workspace_safety.clamp_to_workspace(intercept_pos_2d)
        intercept_pos_3d = np.array([intercept_pos_2d[0], intercept_pos_2d[1], self.table_height])
        
        # Use IK as final fallback
        success, q_new = self.solve_ik(intercept_pos_3d, q_current)
        if success:
            self.target_q = q_new
            self.target_qd = np.zeros(7)
        self.current_mode = self.BLOCK_MODE
        
        torques = self.compute_torques(q_current, qd_current)
        return True, torques
