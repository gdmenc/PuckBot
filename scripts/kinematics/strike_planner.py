"""
Strike planner for aggressive air hockey shots.

Combines:
- Impact physics (compute required mallet velocity)
- Shot planning (choose direction toward goal)
- Inverse kinematics (desired end-effector pose)
- Jacobian mapping (joint velocities for momentum transfer)
"""
import numpy as np
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve
from pydrake.math import RollPitchYaw, RotationMatrix, RigidTransform
from pydrake.multibody.tree import JacobianWrtVariable

try:
    from scripts.kinematics.impact_physics import ImpactPhysics
    from scripts.kinematics.shot_planner import ShotPlanner
except ModuleNotFoundError:
    # Standalone execution
    from impact_physics import ImpactPhysics
    from shot_planner import ShotPlanner


class StrikePlanner:
    """
    Plans aggressive strikes toward opponent goal.
    
    Workflow:
    1. Determine shot direction (toward goal)
    2. Compute required mallet velocity (impact physics)
    3. Solve IK for mallet position at intercept
    4. Map mallet velocity to joint velocity (Jacobian)
    5. Return target joint state (q, qd) for trajectory generation
    """
    
    def __init__(self, plant, robot_model, gripper_model=None,
                 table_bounds=None, shot_speed=2.0, restitution=0.9):
        """
        Args:
            plant: MultibodyPlant
            robot_model: Robot model instance
            gripper_model: Gripper model instance (optional)
            table_bounds: Table boundaries dict
            shot_speed: Desired puck speed after impact (m/s)
            restitution: Coefficient of restitution
        """
        self.plant = plant
        self.robot_model = robot_model
        
        # Get end-effector frame (gripper body)
        frame_model = gripper_model if gripper_model is not None else robot_model
        self.ee_frame = plant.GetFrameByName("body", frame_model)
        self.world_frame = plant.world_frame()
        
        # Physics and planning modules
        self.impact_physics = ImpactPhysics(restitution=restitution)
        
        if table_bounds is None:
            table_bounds = {'x': (-1.0, 1.0), 'y': (-0.52, 0.52)}
        self.shot_planner = ShotPlanner(table_bounds, goal_width=0.5)
        
        self.shot_speed = shot_speed
        self.table_height = 0.11
        
        # Joint limits (approximate for IIWA)
        self.q_min = np.array([-2.9, -2.0, -2.9, -2.0, -2.9, -2.0, -3.0])
        self.q_max = np.array([2.9, 2.0, 2.9, 2.0, 2.9, 2.0, 3.0])
        
    def get_opponent_goal(self, robot_side):
        """
        Get opponent goal center position.
        
        Args:
            robot_side: 1.0 for right robot, -1.0 for left robot
        
        Returns:
            goal_pos: [x, y] goal center
        """
        if robot_side > 0:
            # Right robot: opponent goal on left
            return np.array([-1.0, 0.0])
        else:
            # Left robot: opponent goal on right
            return np.array([1.0, 0.0])
    
    def solve_ik(self, target_pos_3d, current_q):
        """
        Solve IK for target end-effector position.
        
        Args:
            target_pos_3d: Desired position [x, y, z]
            current_q: Current joint configuration (for initial guess)
        
        Returns:
            (success, q_solution)
        """
        # Create isolated context for IK
        ik_context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(ik_context, self.robot_model, current_q)
        
        # Create IK solver
        ik = InverseKinematics(self.plant, ik_context)
        
        # Position constraint (very relaxed tolerance for fast moving scenarios)
        ik.AddPositionConstraint(
            frameB=self.ee_frame,
            p_BQ=np.zeros(3),
            frameA=self.world_frame,
            p_AQ_lower=target_pos_3d - 0.08,  # Relaxed from 0.05
            p_AQ_upper=target_pos_3d + 0.08
        )
        
        # Orientation constraint (very relaxed - allow significant tilt)
        target_rotation = RotationMatrix(RollPitchYaw([0, 0, 0]))
        ik.AddOrientationConstraint(
            frameAbar=self.world_frame,
            R_AbarA=target_rotation,
            frameBbar=self.ee_frame,
            R_BbarB=RotationMatrix.Identity(),
            theta_bound=0.8  # Relaxed from 0.5 (allow ~45deg tilt)
        )
        
        # Solve
        prog = ik.prog()
        q_vars = ik.q()
        
        full_q_guess = self.plant.GetPositions(ik_context)
        prog.SetInitialGuess(q_vars, full_q_guess)
        
        result = Solve(prog)
        
        if result.is_success():
            q_solution_full = result.GetSolution(q_vars)
            
            # Extract robot joints
            context_temp = self.plant.CreateDefaultContext()
            self.plant.SetPositions(context_temp, q_solution_full)
            q_robot = self.plant.GetPositions(context_temp, self.robot_model)
            
            return True, np.clip(q_robot, self.q_min, self.q_max)
        
        # Silently fail - don't spam console
        return False, current_q
    
    def compute_planar_jacobian(self, q):
        """
        Compute 2D (xy) Jacobian at current configuration.
        
        Args:
            q: Joint configuration (7-DOF robot only)
        
        Returns:
            J_xy: 2x7 Jacobian mapping joint velocities to end-effector xy velocity
        """
        # Create context with given configuration
        context = self.plant.CreateDefaultContext()
        self.plant.SetPositions(context, self.robot_model, q)
        
        # Compute spatial Jacobian (6 x num_velocities)
        # num_velocities includes ALL bodies in plant (puck, paddle, robot)
        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            context,
            JacobianWrtVariable.kV,
            self.ee_frame,
            np.zeros(3),  # Point on end-effector
            self.world_frame,
            self.world_frame
        )
        
        # Extract translational rows for x, y (rows 3, 4)
        # J_spatial format: [angular_velocity (3), translational_velocity (3)]
        J_spatial_xy = J_spatial[3:5, :]  # Get vx, vy rows
        
        # CRITICAL: Extract only robot velocity columns
        # The Jacobian has columns for ALL velocities in the plant
        # Plant structure: puck (6 DOF floating) + paddle (welded/0 DOF) + robot (7 DOF)
        # Total plant velocities = 6 (puck) + 0 (paddle welded) + 7 (robot) = 13 DOF
        # But we need to handle this generically
        
        total_velocities = J_spatial_xy.shape[1]
        robot_num_velocities = self.plant.num_velocities(self.robot_model)
        
        # Robot columns are the LAST robot_num_velocities columns
        # (robot added last to plant)
        robot_velocity_start = total_velocities - robot_num_velocities
        J_xy = J_spatial_xy[:, robot_velocity_start:]
        
        return J_xy
    
    def compute_nullspace_objective(self, q):
        """
        Compute desired joint velocity for nullspace optimization.
        
        Secondary objectives:
        - Avoid joint limits (move toward mid-range)
        - Maintain manipulability
        
        Args:
            q: Joint configuration
        
        Returns:
            qd_null: Desired joint velocity for secondary objectives
        """
        # Joint mid-range (optimal position away from limits)
        q_mid = (self.q_min + self.q_max) / 2.0
        
        # Gradient: move toward mid-range (away from limits)
        # This helps avoid singularities and joint limit violations
        qd_null = 0.5 * (q_mid - q)  # Proportional control toward center
        
        # Emphasize joints that are close to limits
        for i in range(len(q)):
            range_i = self.q_max[i] - self.q_min[i]
            dist_to_min = q[i] - self.q_min[i]
            dist_to_max = self.q_max[i] - q[i]
            
            # If within 20% of limit, push away more strongly
            if dist_to_min < 0.2 * range_i:
                qd_null[i] = 1.0  # Push away from lower limit
            elif dist_to_max < 0.2 * range_i:
                qd_null[i] = -1.0  # Push away from upper limit
        
        return qd_null
    
    def compute_joint_velocity_with_nullspace(self, J_xy, v_mallet_required, q_current):
        """
        Compute joint velocities using nullspace optimization.
        
        Primary task: Achieve desired end-effector velocity
        Secondary task: Avoid joint limits, maintain good configuration
        
        Args:
            J_xy: 2xN planar Jacobian
            v_mallet_required: Desired end-effector velocity [vx, vy]
            q_current: Current joint configuration
        
        Returns:
            qd: Joint velocities satisfying both tasks
        """
        # Primary task: End-effector velocity
        J_pinv = np.linalg.pinv(J_xy)
        qd_primary = J_pinv @ v_mallet_required
        
        # Secondary task: Nullspace objective (joint limit avoidance)
        qd_null = self.compute_nullspace_objective(q_current)
        
        # Nullspace projector: (I - J_pinv @ J)
        n_joints = len(q_current)
        I = np.eye(n_joints)
        N = I - J_pinv @ J_xy  # Projects onto nullspace of J
        
        # Combined: primary + nullspace component
        # The nullspace component doesn't affect end-effector velocity
        qd = qd_primary + N @ qd_null
        
        return qd
    
    def plan_strike(self, puck_pos, puck_vel, intercept_pos, intercept_time,
                    q_current, robot_side):
        """
        Plan aggressive strike toward opponent goal.
        
        Args:
            puck_pos: Current puck position [x, y, z]
            puck_vel: Current puck velocity [vx, vy, vz]
            intercept_pos: Predicted intercept position [x, y, z]
            intercept_time: Time to intercept
            q_current: Current joint configuration
            robot_side: 1.0 (right) or -1.0 (left)
        
        Returns:
            success: True if strike plan generated
            q_target: Target joint position at impact
            qd_target: Target joint velocity at impact (for momentum)
            v_mallet_desired: Required end-effector velocity [vx, vy]
        """
        # 1. Determine opponent goal
        opponent_goal = self.get_opponent_goal(robot_side)
        
        # 2. Choose shot direction
        shot_type, v_puck_post = self.shot_planner.choose_best_shot(
            intercept_pos[:2], opponent_goal, speed=self.shot_speed
        )
        
        # 3. Compute impact normal (mallet approaching from robot's side)
        # Approximate: normal points from mallet to puck
        # At intercept, assume mallet is slightly behind puck
        if robot_side > 0:
            # Right robot: mallet approaches from right
            mallet_approach = intercept_pos[:2] + np.array([0.05, 0.0])
        else:
            # Left robot: mallet approaches from left
            mallet_approach = intercept_pos[:2] - np.array([0.05, 0.0])
        
        normal = self.impact_physics.compute_impact_normal(
            intercept_pos[:2], mallet_approach
        )
        
        # 4. Compute required mallet velocity
        v_puck_pre = puck_vel[:2]
        v_mallet_required = self.impact_physics.compute_required_mallet_velocity(
            v_puck_pre, v_puck_post, normal
        )
        
        # 5. Solve IK for impact position
        success_ik, q_impact = self.solve_ik(intercept_pos, q_current)
        
        if not success_ik:
            print(f"[STRIKE] IK failed for intercept position {intercept_pos[:2]}")
            return False, q_current, np.zeros(7), np.zeros(2)
        
        # 6. Map desired mallet velocity to joint velocity WITH NULLSPACE OPTIMIZATION
        J_xy = self.compute_planar_jacobian(q_impact)
        qd_impact = self.compute_joint_velocity_with_nullspace(
            J_xy, v_mallet_required, q_impact
        )
        
        # Clip joint velocities
        qd_max = 1.5  # rad/s (conservative)
        qd_impact = np.clip(qd_impact, -qd_max, qd_max)
        
        print(f"[STRIKE] Shot: {shot_type}, Puck post: {v_puck_post}, Mallet req: {v_mallet_required}")
        
        return True, q_impact, qd_impact, v_mallet_required


# ============================================================================
# Unit Tests
# ============================================================================

def test_strike_planner_standalone():
    """
    Standalone test without Drake plant.
    Just tests the logic flow.
    """
    print("Testing Strike Planner (Standalone Logic)...")
    print("=" * 60)
    
    # Test physics and shot planning integration
    impact_physics = ImpactPhysics(restitution=0.9)
    shot_planner = ShotPlanner({'x': (-1.0, 1.0), 'y': (-0.52, 0.52)})
    
    print("\nTest: Strike planning workflow")
    
    # Scenario: Puck approaching from left, robot on right
    puck_pos = np.array([0.3, 0.1, 0.11])
    puck_vel = np.array([0.5, 0.0, 0.0])
    intercept_pos = np.array([0.6, 0.1, 0.11])
    opponent_goal = np.array([-1.0, 0.0])
    
    # Step 1: Choose shot
    shot_type, v_puck_post = shot_planner.choose_best_shot(
        intercept_pos[:2], opponent_goal, speed=2.0
    )
    print(f"  Shot type: {shot_type}")
    print(f"  Desired puck velocity: {v_puck_post}")
    
    # Step 2: Compute normal
    mallet_approach = intercept_pos[:2] + np.array([0.05, 0.0])
    normal = impact_physics.compute_impact_normal(intercept_pos[:2], mallet_approach)
    print(f"  Impact normal: {normal}")
    
    # Adjust desired velocity to preserve tangent (physics constraint)
    tangent = np.array([-normal[1], normal[0]])
    v_puck_pre_t = np.dot(puck_vel[:2], tangent)
    
    # Recompute desired shot with correct tangent
    v_puck_post_raw = v_puck_post.copy()
    v_puck_post_n = np.dot(v_puck_post_raw, normal)
    v_puck_post = v_puck_post_n * normal + v_puck_pre_t * tangent  # Physics-realizable
    print(f"  Adjusted puck velocity (physics): {v_puck_post}")
    
    # Step 3: Required mallet velocity
    v_mallet_req = impact_physics.compute_required_mallet_velocity(
        puck_vel[:2], v_puck_post, normal
    )
    print(f"  Required mallet velocity: {v_mallet_req}")
    print(f"  Required mallet speed: {np.linalg.norm(v_mallet_req):.3f} m/s")
    
    # Verify physics
    v_actual = impact_physics.compute_post_impact_velocity(
        puck_vel[:2], v_mallet_req, normal
    )
    error = np.linalg.norm(v_actual - v_puck_post)
    print(f"  Actual puck post-impact: {v_actual}")
    print(f"  Error: {error:.6f} m/s")
    
    assert error < 0.01, f"Physics validation failed with error {error}"
    print("  ✓ PASS")
    
    print("\n" + "=" * 60)
    print("✅ STANDALONE TESTS PASSED")


if __name__ == "__main__":
    test_strike_planner_standalone()
