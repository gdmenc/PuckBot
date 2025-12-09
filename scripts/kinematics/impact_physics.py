"""
Impact physics for air hockey puck-mallet collisions.

Based on research paper methodology:
- Forward model: Given mallet velocity, compute puck post-impact velocity
- Inverse model: Given desired puck trajectory, compute required mallet velocity

References:
    Impact law: v_p+ = v_p- - (1+α)·((v_p- - v_m)·n)·n
    where α is coefficient of restitution, n is surface normal
"""
import numpy as np


class ImpactPhysics:
    """
    Handles impulsive collision physics between mallet and puck.
    
    Assumes:
    - Planar motion (2D)
    - Point contact
    - No friction/spin effects
    - Constant coefficient of restitution
    """
    
    def __init__(self, restitution=0.9):
        """
        Args:
            restitution: Coefficient of restitution (0-1)
                        0 = perfectly inelastic, 1 = perfectly elastic
                        Typical air hockey value: 0.8-0.95
        """
        if not 0 <= restitution <= 1:
            raise ValueError(f"Restitution must be in [0,1], got {restitution}")
        
        self.alpha = restitution
        
    def compute_post_impact_velocity(self, v_puck_pre, v_mallet, normal):
        """
        Forward impact model: Compute puck velocity after collision.
        
        Args:
            v_puck_pre: Puck velocity before impact [vx, vy] (2D)
            v_mallet: Mallet velocity at impact [vx, vy] (2D)
            normal: Unit surface normal from mallet to puck [nx, ny] (2D)
        
        Returns:
            v_puck_post: Puck velocity after impact [vx, vy] (2D)
        
        Physics:
            v_p+ = v_p- - (1+α)·((v_p- - v_m)·n)·n
        """
        v_puck_pre = np.asarray(v_puck_pre, dtype=float)
        v_mallet = np.asarray(v_mallet, dtype=float)
        normal = np.asarray(normal, dtype=float)
        
        # Normalize normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Relative velocity
        v_rel = v_puck_pre - v_mallet
        
        # Normal component of relative velocity
        v_rel_normal = np.dot(v_rel, normal)
        
        # Post-impact velocity (impulse along normal)
        v_puck_post = v_puck_pre - (1 + self.alpha) * v_rel_normal * normal
        
        return v_puck_post
    
    def compute_required_mallet_velocity(self, v_puck_pre, v_puck_post_desired, normal):
        """
        Inverse impact model: Compute required mallet velocity for desired shot.
        
        Args:
            v_puck_pre: Puck velocity before impact [vx, vy] (2D)
            v_puck_post_desired: Desired puck velocity after impact [vx, vy] (2D)
            normal: Unit surface normal from mallet to puck [nx, ny] (2D)
        
        Returns:
            v_mallet_required: Required mallet velocity [vx, vy] (2D)
        
        Derivation:
            From impact law: v_p+ = v_p- - (1+α)·((v_p- - v_m)·n)·n
            
            Rearranging to solve for v_m:
            The impact only affects the normal component. The tangential component
            of the puck velocity is unchanged by the collision.
            
            For normal component:
            v_p+·n = v_p-·n - (1+α)·((v_p- - v_m)·n)
            v_p+·n = v_p-·n - (1+α)·v_p-·n + (1+α)·v_m·n
            v_p+·n = -α·v_p-·n + (1+α)·v_m·n
            v_m·n = (v_p+·n + α·v_p-·n) / (1+α)
            
            For tangent: Just match puck tangential velocity
        """
        v_puck_pre = np.asarray(v_puck_pre, dtype=float)
        v_puck_post_desired = np.asarray(v_puck_post_desired, dtype=float)
        normal = np.asarray(normal, dtype=float)
        
        # Normalize normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Tangent vector (perpendicular to normal)
        tangent = np.array([-normal[1], normal[0]])
        
         # Required normal component (derived formula)
        v_puck_pre_n = np.dot(v_puck_pre, normal)
        v_puck_post_n = np.dot(v_puck_post_desired, normal)
        v_mallet_n = (v_puck_post_n + self.alpha * v_puck_pre_n) / (1.0 + self.alpha)
        
        # For tangential: The collision does NOT change puck tangential velocity
        # v_puck_post·t should equal v_puck_pre·t (conservation)
        # But user may specify a different tangent in v_puck_post_desired
        # We have freedom to choose mallet tangent - let's match the incoming puck
        v_puck_pre_t = np.dot(v_puck_pre, tangent)
        v_mallet_t = v_puck_pre_t
        
        # Reconstruct full mallet velocity
        v_mallet_required = v_mallet_n * normal + v_mallet_t * tangent
        
        return v_mallet_required
    
    def compute_impact_normal(self, puck_pos, mallet_pos):
        """
        Compute surface normal at contact point.
        
        Args:
            puck_pos: Puck position [x, y] (2D)
            mallet_pos: Mallet position [x, y] (2D)
        
        Returns:
            normal: Unit normal vector from mallet to puck [nx, ny] (2D)
        """
        puck_pos = np.asarray(puck_pos, dtype=float)
        mallet_pos = np.asarray(mallet_pos, dtype=float)
        
        # Direction from mallet to puck
        direction = puck_pos - mallet_pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            # Degenerate case: assume head-on collision
            return np.array([1.0, 0.0])
        
        normal = direction / distance
        return normal
    
    def validate_impact(self, v_puck_pre, v_puck_post, v_mallet, normal, tolerance=0.1):
        """
        Validate that a given mallet velocity produces the desired post-impact velocity.
        
        Args:
            v_puck_pre: Puck velocity before impact
            v_puck_post: Desired puck velocity after impact
            v_mallet: Mallet velocity
            normal: Impact normal
            tolerance: Acceptable velocity error (m/s)
        
        Returns:
            (valid, error): (bool, velocity_error)
        """
        v_actual = self.compute_post_impact_velocity(v_puck_pre, v_mallet, normal)
        error = np.linalg.norm(v_actual - v_puck_post)
        return (error < tolerance, error)


# ============================================================================
# Unit Tests
# ============================================================================

def test_impact_physics():
    """Test suite for impact physics calculations."""
    physics = ImpactPhysics(restitution=0.9)
    
    print("Testing Impact Physics...")
    print("=" * 60)
    
    # Test 1: Stationary puck hit head-on
    print("\nTest 1: Stationary puck, moving mallet (head-on)")
    v_puck_pre = np.array([0.0, 0.0])
    v_mallet = np.array([1.0, 0.0])
    normal = np.array([1.0, 0.0])
    
    v_puck_post = physics.compute_post_impact_velocity(v_puck_pre, v_mallet, normal)
    print(f"  Puck pre:  {v_puck_pre}")
    print(f"  Mallet:    {v_mallet}")
    print(f"  Puck post: {v_puck_post}")
    print(f"  Expected:  ~[1.9, 0.0] (mallet velocity * (1+α))")
    assert v_puck_post[0] > 1.8 and v_puck_post[0] < 2.0
    assert abs(v_puck_post[1]) < 0.01
    print("  ✓ PASS")
    
    # Test 2: Inverse model - compute required mallet velocity
    print("\nTest 2: Inverse model (desired shot speed)")
    v_puck_pre = np.array([0.0, 0.0])
    v_puck_post_desired = np.array([2.0, 0.0])
    normal = np.array([1.0, 0.0])
    
    v_mallet_req = physics.compute_required_mallet_velocity(
        v_puck_pre, v_puck_post_desired, normal
    )
    print(f"  Desired puck post: {v_puck_post_desired}")
    print(f"  Required mallet:   {v_mallet_req}")
    
    # Validate by forward model
    v_actual = physics.compute_post_impact_velocity(v_puck_pre, v_mallet_req, normal)
    print(f"  Actual puck post:  {v_actual}")
    error = np.linalg.norm(v_actual - v_puck_post_desired)
    print(f"  Error: {error:.6f} m/s")
    assert error < 0.01
    print("  ✓ PASS")

    # Test 3: Moving puck, angled impact
    print("\nTest 3: Moving puck, angled mallet")
    v_puck_pre = np.array([0.5, 0.3])
    normal = np.array([0.707, 0.707])  # 45° angle
    tangent = np.array([-normal[1], normal[0]])

    # Desired post-impact: Change normal component, preserve tangent
    v_puck_pre_t = np.dot(v_puck_pre, tangent)  # Must be preserved
    v_puck_post_n_desired = 1.5  # Desired normal velocity
    v_puck_post_desired = v_puck_post_n_desired * normal + v_puck_pre_t * tangent

    v_mallet_req = physics.compute_required_mallet_velocity(
        v_puck_pre, v_puck_post_desired, normal
    )
    print(f"  Puck pre:          {v_puck_pre}")
    print(f"  Desired post:      {v_puck_post_desired}")
    print(f"  Required mallet:   {v_mallet_req}")
    
    v_actual = physics.compute_post_impact_velocity(v_puck_pre, v_mallet_req, normal)
    print(f"  Actual post:       {v_actual}")
    error = np.linalg.norm(v_actual - v_puck_post_desired)
    print(f"  Error: {error:.6f} m/s")
    assert error < 0.01
    print("  ✓ PASS")
    
    # Test 4: Validate roundtrip consistency
    print("\nTest 4: Roundtrip validation")
    for _ in range(5):
        # Random velocities and normal
        v_puck_pre = np.random.uniform(-1, 1, 2)
        angle = np.random.uniform(0, 2*np.pi)
        normal = np.array([np.cos(angle), np.sin(angle)])
        tangent = np.array([-normal[1], normal[0]])
        
        # Construct physically realizable desired velocity
        # (preserves tangent component)
        v_puck_pre_t = np.dot(v_puck_pre, tangent)
        v_puck_post_n_desired = np.random.uniform(-2, 2)
        v_puck_post_desired = v_puck_post_n_desired * normal + v_puck_pre_t * tangent
        
        v_mallet = physics.compute_required_mallet_velocity(
            v_puck_pre, v_puck_post_desired, normal
        )
        v_actual = physics.compute_post_impact_velocity(v_puck_pre, v_mallet, normal)
        
        valid, error = physics.validate_impact(
            v_puck_pre, v_puck_post_desired, v_mallet, normal
        )
        assert valid, f"Roundtrip failed with error {error:.6f}"
    
    print("  ✓ All 5 random roundtrips PASS")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")


if __name__ == "__main__":
    test_impact_physics()
