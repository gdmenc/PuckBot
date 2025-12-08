import numpy as np
import trimesh
from typing import List, Tuple
import os
import random
from pathlib import Path
from typing import List, Tuple

import mpld3
import numpy as np
import trimesh
from pydrake.all import (
    AddFrameTriadIllustration,
    BasicVector,
    Context,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    TrajectorySource,
)

AntipodeCandidateType = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def sample_colinear_points(
    mesh: trimesh.Trimesh, n_sample_points: int
) -> List[AntipodeCandidateType]:
    """
    Compute n_sample_points point pairs for the mesh that are colinear by
    ray casting along surface normals.
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

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix

def compute_grasp_from_points(
    antipodal_pt: AntipodeCandidateType,
) -> RigidTransform | None:
    """
    Given the tuple of antipodal points and their normals on the object O,
    compute the grasp X_OG.
    """
    z_axis_O = np.array([0.0, 0.0, 1.0])

    # 1. x-axis = surface normal at first point
    x_axis = antipodal_pt[2] / np.linalg.norm(antipodal_pt[2])

    # 2. check degeneracy (parallel to z-axis of O)
    if np.abs(np.dot(x_axis, z_axis_O)) > 0.99:
        return None

    # 3. y-axis = projected -z_axis_O, orthogonalized against x_axis
    y_temp = -z_axis_O
    y_axis = y_temp - np.dot(y_temp, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)

    # 4. z-axis = orthogonal cross product
    z_axis = np.cross(x_axis, y_axis)

    # 5. rotation matrix
    R_OG = np.column_stack([x_axis, y_axis, z_axis])

    # 6. midpoint between the two colinear points
    p_OG_O = 0.5 * (antipodal_pt[0] + antipodal_pt[1])

    # 7. apply finger offset (-0.1m along gripper y-axis)
    p_OG_O = p_OG_O - 0.1 * y_axis

    # 8. return rigid trasfnorm
    return RigidTransform(RotationMatrix(R_OG), p_OG_O)

import numpy as np
from pydrake.math import RigidTransform

def check_collision_free(X_WG: RigidTransform | np.ndarray) -> bool:
    """
    Checks if the gripper collides with the table (z=0 plane in world coordinates).
    """
    # Approximate gripper geometry by bounding vertices (in gripper frame)
    gripper_vertices = np.array([
        [-0.073, -0.085383, -0.025],
        [ 0.073,  0.069   ,  0.025]
    ])
    verts_h = np.hstack((gripper_vertices, np.ones((gripper_vertices.shape[0], 1))))

    # If a RigidTransform is passed, convert to 4x4 matrix
    if isinstance(X_WG, RigidTransform):
        X_WG = X_WG.GetAsMatrix4()

    # Map vertices into world frame
    verts_W = (X_WG @ verts_h.T).T[:, :3]

    # Collision-free if all vertices are above the table (z >= 0)
    return np.all(verts_W[:, 2] >= 0)

def get_filtered_grasps(
    candidate_list: List[AntipodeCandidateType],
    antipodal_thresh: float,
    z_axis_thresh: float,
    max_pt_dist: float,
    min_pt_dist: float,
    X_WO: RigidTransform,
) -> List[RigidTransform]:
    """
    Return a list of grasps filtered on:
    (1) Antipodality: antipodality is a good heuristic for finding grasps with a large total wrench cone
    (2) Point Distance: pairs of points too far apart won't fit inside the gripper.
        Points too close together are "false positives" that appear due to the numerics of the ray casting.
    (3) Collision: reject grasps that intersect the table plane.
    """
    filtered_candidates = []

    for candidate in candidate_list:
        p1, n1, p2, n2 = candidate  # unpack

        # === (1) Check antipodality ===
        dot = np.dot(n1, -n2)  # normals should point in opposite directions
        if dot > antipodal_thresh:
            continue

        # === (2) Check point distance ===
        dist = np.linalg.norm(p1 - p2)
        if dist < min_pt_dist or dist > max_pt_dist:
            continue

        # === (3) Construct the grasp pose (in object frame) ===
        X_OG = compute_grasp_from_candidate(p1, p2, n1, n2, z_axis_thresh)
        if X_OG is None:
            continue

        # === (4) Map grasp to world frame ===
        X_WG = X_WO @ X_OG

        # === (5) Collision check ===
        if not check_collision_free(X_WG):
            continue

        # If passed all checks, keep it
        filtered_candidates.append(X_WG)

    return filtered_candidates

def sample_grasp(
    mesh: trimesh.Trimesh, X_WO: RigidTransform, n_sample_pts: int = 500
) -> RigidTransform:
    colinear_pts = sample_colinear_points(mesh, n_sample_points=n_sample_pts)
    candidate_grasps = get_filtered_grasps(
        colinear_pts,
        antipodal_thresh=-0.95,
        z_axis_thresh=0.8,
        max_pt_dist=0.04,
        min_pt_dist=0.005,
        X_WO=X_WO,
    )
    return candidate_grasps[0]

def compute_prepick_pose(X_WG: RigidTransform) -> RigidTransform:
    X_GGprepick = RigidTransform([0, -0.17, 0.0])
    return X_WG @ X_GGprepick

class PseudoInverseController(LeafSystem):
    def __init__(self, plant: MultibodyPlant) -> None:
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context: Context, output: BasicVector) -> None:
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )
        J_G = J_G[:, self.iiwa_start : self.iiwa_end + 1]  # Only iiwa terms.
        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)