"""
Grasp Sampling using Antipodal Point Pairs
Based on MIT 6.4210 Manipulation Course methodology
"""
import numpy as np
import trimesh
from typing import List, Tuple
from pydrake.math import RigidTransform, RotationMatrix

AntipodeCandidateType = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def sample_colinear_points(
    mesh: trimesh.Trimesh, n_sample_points: int
) -> List[AntipodeCandidateType]:
    """
    Compute n_sample_points point pairs for the mesh that are colinear by
    ray casting along surface normals.
    
    Args:
        mesh: Trimesh object of the paddle
        n_sample_points: Number of surface points to sample
        
    Returns:
        List of (p1, p2, n1, n2) tuples representing antipodal candidates
    """
    candidates: List[AntipodeCandidateType] = []

    # 1. Sample points on the mesh surface
    points, face_indices = trimesh.sample.sample_surface(mesh, n_sample_points)

    # 2. Get normals at sampled points
    normals = mesh.face_normals[face_indices]

    # 3. Build a ray intersector
    try:
        rmi = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except BaseException:
        # fallback if pyembree not installed
        rmi = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    for i in range(n_sample_points):
        p1 = points[i]
        n1 = normals[i]

        # Cast ray in -normal direction
        ray_origin = p1 + 1e-6 * n1  # small offset to avoid self-hit
        ray_dir = -n1 / np.linalg.norm(n1)

        # 4. Find intersection with mesh
        locations, index_ray, index_tri = rmi.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_dir],
            multiple_hits=False
        )

        if len(locations) == 0:
            # No colinear point found
            continue

        # 5. Closest intersection
        p2 = locations[0]
        tri_idx = index_tri[0]
        n2 = mesh.face_normals[tri_idx]

        # 6. Add candidate
        candidates.append((p1, p2, n1, n2))

    return candidates


def compute_grasp_from_points(
    p1: np.ndarray,
    p2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    z_axis_thresh: float = 0.8,
) -> RigidTransform | None:
    """
    Given antipodal points and their normals on the object O,
    compute the grasp pose X_OG.
    
    Args:
        p1: First point on surface
        p2: Second point on surface  
        n1: Normal at p1
        n2: Normal at p2
        z_axis_thresh: Threshold for degeneracy check
        
    Returns:
        X_OG: Grasp pose in object frame, or None if degenerate
    """
    z_axis_O = np.array([0.0, 0.0, 1.0])

    # 1. x-axis = surface normal at first point
    x_axis = n1 / np.linalg.norm(n1)

    # 2. check degeneracy (parallel to z-axis of O)
    if np.abs(np.dot(x_axis, z_axis_O)) > z_axis_thresh:
        return None

    # 3. y-axis = projected -z_axis_O, orthogonalized against x_axis
    y_temp = -z_axis_O
    y_axis = y_temp - np.dot(y_temp, x_axis) * x_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        return None
    y_axis /= y_norm

    # 4. z-axis = orthogonal cross product
    z_axis = np.cross(x_axis, y_axis)

    # 5. rotation matrix
    R_OG = np.column_stack([x_axis, y_axis, z_axis])

    # 6. midpoint between the two colinear points
    p_OG_O = 0.5 * (p1 + p2)

    # 7. apply finger offset (-0.1m along gripper y-axis)
    p_OG_O = p_OG_O - 0.1 * y_axis

    # 8. return rigid transform
    return RigidTransform(RotationMatrix(R_OG), p_OG_O)


def check_collision_free(X_WG: RigidTransform, table_height: float = 0.0) -> bool:
    """
    Checks if the gripper collides with the table plane.
    
    Args:
        X_WG: Gripper pose in world frame
        table_height: Z-coordinate of table surface
        
    Returns:
        True if collision-free
    """
    # Approximate gripper geometry by bounding vertices (in gripper frame)
    gripper_vertices = np.array([
        [-0.073, -0.085383, -0.025],
        [ 0.073,  0.069   ,  0.025]
    ])
    verts_h = np.hstack((gripper_vertices, np.ones((gripper_vertices.shape[0], 1))))

    # Convert to 4x4 matrix
    X_WG_matrix = X_WG.GetAsMatrix4()

    # Map vertices into world frame
    verts_W = (X_WG_matrix @ verts_h.T).T[:, :3]

    # Collision-free if all vertices are above the table
    return np.all(verts_W[:, 2] >= table_height)


def get_filtered_grasps(
    candidate_list: List[AntipodeCandidateType],
    antipodal_thresh: float,
    z_axis_thresh: float,
    max_pt_dist: float,
    min_pt_dist: float,
    X_WO: RigidTransform,
    table_height: float = 0.0,
) -> List[RigidTransform]:
    """
    Filter grasp candidates on antipodality, distance, and collision.
    
    Args:
        candidate_list: List of (p1, p2, n1, n2) tuples
        antipodal_thresh: Dot product threshold for antipodality
        z_axis_thresh: Threshold for z-axis alignment check
        max_pt_dist: Maximum distance between grasp points
        min_pt_dist: Minimum distance between grasp points
        X_WO: Object pose in world frame
        table_height: Z-coordinate of table surface
        
    Returns:
        List of valid grasp poses in world frame
    """
    filtered_candidates = []

    for candidate in candidate_list:
        p1, p2, n1, n2 = candidate

        # === (1) Check antipodality ===
        dot = np.dot(n1, -n2)  # normals should point in opposite directions
        if dot < antipodal_thresh:
            continue

        # === (2) Check point distance ===
        dist = np.linalg.norm(p1 - p2)
        if dist < min_pt_dist or dist > max_pt_dist:
            continue

        # === (3) Construct the grasp pose (in object frame) ===
        X_OG = compute_grasp_from_points(p1, p2, n1, n2, z_axis_thresh)
        if X_OG is None:
            continue

        # === (4) Map grasp to world frame ===
        X_WG = X_WO @ X_OG

        # === (5) Collision check ===
        if not check_collision_free(X_WG, table_height):
            continue

        # If passed all checks, keep it
        filtered_candidates.append(X_WG)

    return filtered_candidates


def sample_grasp(
    mesh: trimesh.Trimesh,
    X_WO: RigidTransform,
    n_sample_pts: int = 1000,  # Increased samples
    table_height: float = 0.0,
) -> RigidTransform | None:
    """
    Sample a valid grasp pose for the given mesh.
    
    Args:
        mesh: Trimesh object
        X_WO: Object pose in world frame
        n_sample_pts: Number of points to sample
        table_height: Z-coordinate of table surface
        
    Returns:
        Valid grasp pose in world frame, or None if no valid grasp found
    """
    colinear_pts = sample_colinear_points(mesh, n_sample_points=n_sample_pts)
    
    if len(colinear_pts) == 0:
        print("[GraspSampler] No colinear points found")
        return None
    
    print(f"[GraspSampler] Found {len(colinear_pts)} colinear point pairs")
    
    candidate_grasps = get_filtered_grasps(
        colinear_pts,
        antipodal_thresh=-0.5,  # Relaxed from -0.95
        z_axis_thresh=0.95,  # Allow more alignment with z
        max_pt_dist=0.10,  # Increased from 0.04 - paddles are larger
        min_pt_dist=0.002,  # Decreased from 0.005
        X_WO=X_WO,
        table_height=table_height,
    )
    
    if len(candidate_grasps) == 0:
        print("[GraspSampler] No valid grasps after filtering")
        return None
    
    print(f"[GraspSampler] Found {len(candidate_grasps)} valid grasps")
    return candidate_grasps[0]


def compute_prepick_pose(X_WG: RigidTransform, offset: float = 0.17) -> RigidTransform:
    """
    Compute a pre-pick pose offset from the grasp pose.
    
    Args:
        X_WG: Grasp pose in world frame
        offset: Distance to offset along negative gripper y-axis
        
    Returns:
        Pre-pick pose in world frame
    """
    X_GGprepick = RigidTransform([0, -offset, 0.0])
    return X_WG @ X_GGprepick
