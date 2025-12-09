"""
Simple robot controller for air hockey simulation.
Runs in simulation loop to control robot paddles.
"""
import numpy as np
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve
from pydrake.math import RollPitchYaw


class SimpleAirHockeyController:
    """
    Reactive controller that:
    1. Observes puck position and velocity
    2. Predicts puck trajectory  
    3. Uses IK to position paddle at intercept point
    4. Keeps paddle parallel to table
    """
    
    def __init__(self, plant, robot_model, gripper_frame_name="body", gripper_model=None):
        """
        Args:
            plant: MultibodyPlant
            robot_model: Robot model instance
            gripper_frame_name: Name of end effector frame (gripper/paddle)
            gripper_model: Model instance containing the frame (if different from robot)
        """
        self.plant = plant
        self.robot_model = robot_model
        
        # Find frame in gripper model if provided, else in robot model
        frame_model = gripper_model if gripper_model is not None else robot_model
        self.gripper_frame = plant.GetFrameByName(gripper_frame_name, frame_model)
        self.world_frame = plant.world_frame()
        
        self.table_height = 0.11  # Paddle rests on table
        self.control_active = True
        
    def predict_puck_position(self, puck_pos, puck_vel, dt=0.5):
        """
        Simple linear prediction of puck position.
        
        Args:
            puck_pos: Current [x, y, z]
            puck_vel: Current [vx, vy, vz]
            dt: How far ahead to predict (seconds)
            
        Returns:
            Predicted [x, y, z]
        """
        # Simple ballistic prediction (ignoring walls for now)
        predicted = puck_pos[:2] + puck_vel[:2] * dt
        return np.array([predicted[0], predicted[1], self.table_height])
    
    def solve_ik_with_orientation(
        self, 
        target_pos, 
        current_q, 
        context,
        orientation_rpy=None
    ):
        """
        Solve IK with position and orientation constraints.
        
        Args:
            target_pos: Target [x, y, z] position
            current_q: Current joint positions (seed)
            context: Plant context
            orientation_rpy: Desired [roll, pitch, yaw] or None for table-parallel
            
        Returns:
            (success, joint_solution)
        """
        if orientation_rpy is None:
            # Default: parallel to table (no roll/pitch, any yaw)
            orientation_rpy = [0, 0, 0]  # Will add yaw tolerance
        
        ik = InverseKinematics(self.plant, context)
        
        # Position constraint
        position_tolerance = 0.02  # 2cm tolerance
        ik.AddPositionConstraint(
            frameB=self.gripper_frame,
            p_BQ=np.zeros(3),
            frameA=self.world_frame,
            p_AQ_lower=target_pos - position_tolerance,
            p_AQ_upper=target_pos + position_tolerance
        )
        
        # Orientation constraint (keep parallel to table)
        # Constrain roll and pitch to be near zero
        from pydrake.math import RotationMatrix
        target_rotation = RotationMatrix(RollPitchYaw(orientation_rpy))
        
        ik.AddOrientationConstraint(
            frameAbar=self.world_frame,
            R_AbarA=target_rotation,
            frameBbar=self.gripper_frame,
            R_BbarB=RotationMatrix.Identity(),
            theta_bound=0.2  # ~11 degrees tolerance
        )
        
        # Solve
        prog = ik.prog()
        q_vars = ik.q()
        
        # IMPORTANT: Create isolated context for IK that doesn't affect simulation
        # We pass the simulation context to IK for evaluation, but set initial guess
        # using a fresh context to avoid corrupting puck state
        
        # Create fresh context for initial guess
        context_guess = self.plant.CreateDefaultContext()
        self.plant.SetPositions(context_guess, self.robot_model, current_q)
        full_q_guess = self.plant.GetPositions(context_guess)
        prog.SetInitialGuess(q_vars, full_q_guess)
        
        result = Solve(prog)
        
        if result.is_success():
            # Extract ONLY robot joint positions from solution
            q_solution_full = result.GetSolution(q_vars)
            
            # Create temporary context to extract robot joints
            context_temp = self.plant.CreateDefaultContext()
            self.plant.SetPositions(context_temp, q_solution_full)
            q_robot = self.plant.GetPositions(context_temp, self.robot_model)
            
            return True, q_robot
        else:
            return False, current_q
    
    def is_puck_threat(self, puck_pos, puck_vel, robot_x_side):
        """
        Determine if puck is a threat that requires reaction.
        
        Args:
            puck_pos: Puck [x, y, z]
            puck_vel: Puck [vx, vy, vz]
            robot_x_side: Which side robot defends (negative or positive X)
            
        Returns:
            True if puck is approaching robot's zone and reachable
        """
        # Check 1: Is puck moving?
        puck_speed = np.linalg.norm(puck_vel[:2])
        if puck_speed < 0.05:  # Nearly stationary
            return False
        
        # Check 2: Is puck moving toward robot's side?
        if robot_x_side < 0:  # Left robot (defends negative X)
            if puck_vel[0] >= 0:  # Puck moving away (positive direction)
                return False
            if puck_pos[0] > 0.2:  # Puck on opponent's side
                return False
        else:  # Right robot (defends positive X)
            if puck_vel[0] <= 0:  # Puck moving away (negative direction)
                return False
            if puck_pos[0] < -0.2:  # Puck on opponent's side
                return False
        
        # Check 3: Is puck within reachable zone?
        # Robot can reach approximately x in [-0.8, 0] or [0, 0.8] and y in [-0.5, 0.5]
        if robot_x_side < 0:
            x_reachable = -0.9 <= puck_pos[0] <= 0.1
        else:
            x_reachable = -0.1 <= puck_pos[0] <= 0.9
        
        y_reachable = -0.55 <= puck_pos[1] <= 0.55
        
        if not (x_reachable and y_reachable):
            return False
        
        # Puck is a threat!
        return True
    
    def compute_target_position(self, puck_pos, puck_vel, robot_x_side):
        """
        Compute where robot should move paddle.
        
        Args:
            puck_pos: Puck [x, y, z]
            puck_vel: Puck [vx, vy, vz]
            robot_x_side: Which side robot is on (negative or positive X)
            
        Returns:
            Target [x, y, z] for paddle, or None if should not react
        """
        # Only compute target if puck is a threat
        if not self.is_puck_threat(puck_pos, puck_vel, robot_x_side):
            return None
        
        # Predict where puck will be in 0.3-0.5 seconds
        prediction_time = 0.4
        predicted_pos = self.predict_puck_position(puck_pos, puck_vel, prediction_time)
        
        # Move toward predicted position, staying on our side
        if robot_x_side < 0:  # Left robot
            target_x = np.clip(predicted_pos[0], -0.8, -0.1)
        else:  # Right robot
            target_x = np.clip(predicted_pos[0], 0.1, 0.8)
        
        target_y = np.clip(predicted_pos[1], -0.45, 0.45)
        
        return np.array([target_x, target_y, self.table_height])
    
    def update(self, puck_pos, puck_vel, current_q, context, robot_x_side):
        """
        Update control - called each timestep.
        
        Returns:
            (has_target, target_q) - target joint positions or None if no action needed
        """
        if not self.control_active:
            return False, current_q
        
        # Compute target position (returns None if puck not a threat)
        target_pos = self.compute_target_position(puck_pos, puck_vel, robot_x_side)
        
        if target_pos is None:
            # No threat - don't move
            return False, current_q
        
        # Solve IK
        success, target_q = self.solve_ik_with_orientation(
            target_pos, current_q, context
        )
        
        if not success:
            # IK failed - stay where we are
            return False, current_q
        
        return True, target_q
