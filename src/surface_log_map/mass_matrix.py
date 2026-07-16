"""Compute and verify a vertex-centered mass matrix on a triangle mesh.

This file presents a reference solution for the exercise:

1. Compute the area of the vertex-centered dual cell around every mesh vertex.
2. Place these areas on the diagonal of a sparse mass matrix.
3. Verify the manual result against ``gpytoolbox.massmatrix(V, F)``.

The implementation follows the mixed Voronoi-area convention used by the
default GPyToolbox mass matrix.

Why the mass matrix is diagonal
-------------------------------
A scalar value stored at a mesh vertex represents a small region of surface
around that vertex. Let ``A_i`` denote the area assigned to vertex ``i``.
The lumped mass matrix is

    M = diag(A_0, A_1, ..., A_{n-1}).

The matrix therefore has one nonzero entry per vertex.

Vertex-centered dual area
-------------------------
For a non-obtuse triangle, the circumcenter lies inside the triangle. The
Voronoi region associated with each corner can be computed from the edge
lengths and cotangents of the opposite angles.

For an obtuse triangle, the circumcenter lies outside the triangle. The mixed
Voronoi convention assigns

- one half of the triangle area to the obtuse vertex;
- one quarter of the triangle area to each of the other two vertices.

Summing these per-face contributions over all incident faces gives the dual
cell area around each mesh vertex.

Mesh representation
-------------------
``V`` is a NumPy array with shape ``(n_vertices, 3)`` containing the vertex
coordinates.

``F`` is a NumPy array with shape ``(n_faces, 3)`` containing the three vertex
indices of every triangular face.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]


def validate_triangle_mesh(
    V: FloatArray,
    F: IntArray,
) -> tuple[FloatArray, IntArray]:
    """Check that the input arrays describe a valid triangular mesh."""
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(
            f"V must have shape (n_vertices, 3); received {V.shape}."
        )

    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(
            f"F must have shape (n_faces, 3); received {F.shape}."
        )

    if len(V) == 0:
        raise ValueError("The mesh contains no vertices.")

    if not np.all(np.isfinite(V)):
        raise ValueError("V contains NaN or infinite coordinates.")

    if len(F) > 0 and (F.min() < 0 or F.max() >= len(V)):
        raise ValueError("F contains a vertex index outside the valid range.")

    return V, F


def compute_face_areas(
    V: FloatArray,
    F: IntArray,
) -> FloatArray:
    """Compute the area of every triangular face.

    For a triangle with positions ``p0``, ``p1``, and ``p2``,

        area = 0.5 * ||(p1 - p0) × (p2 - p0)||.
    """
    V, F = validate_triangle_mesh(V, F)

    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]

    face_cross = np.cross(
        p1 - p0,
        p2 - p0,
    )

    return 0.5 * np.linalg.norm(
        face_cross,
        axis=1,
    )


def _cotangent(
    first_edge: FloatArray,
    second_edge: FloatArray,
    *,
    eps: float = 1e-12,
) -> FloatArray:
    """Compute the cotangent of the angle between vector pairs.

    For vectors ``a`` and ``b``,

        cot(theta) = (a · b) / ||a × b||.

    The function operates row by row. Degenerate angles receive cotangent zero
    to avoid division by a nearly zero cross-product magnitude.
    """
    numerator = np.sum(
        first_edge * second_edge,
        axis=1,
    )

    denominator = np.linalg.norm(
        np.cross(first_edge, second_edge),
        axis=1,
    )

    cotangents = np.zeros_like(numerator)

    valid = denominator > eps
    cotangents[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return cotangents


def compute_mixed_voronoi_vertex_areas(
    V: FloatArray,
    F: IntArray,
    *,
    eps: float = 1e-12,
) -> FloatArray:
    """Compute one mixed Voronoi dual-cell area at every vertex.

    For a non-obtuse triangle with corners ``i``, ``j``, and ``k``, the area
    contribution assigned to vertex ``i`` is

        A_i^f
        =
        (|v_j-v_i|^2 cot(theta_k)
        + |v_k-v_i|^2 cot(theta_j)) / 8,

    where ``theta_j`` and ``theta_k`` are the angles at the other two corners.

    For an obtuse triangle, the mixed-area correction assigns half of the face
    area to the obtuse corner and one quarter to each remaining corner.

    Returns
    -------
    vertex_areas
        Dual-cell areas with shape ``(n_vertices,)``.
    """
    V, F = validate_triangle_mesh(V, F)

    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]

    face_areas = compute_face_areas(V, F)
    valid_faces = face_areas > eps

    # Cotangent of the angle at each triangle corner.
    cot0 = _cotangent(
        p1 - p0,
        p2 - p0,
        eps=eps,
    )
    cot1 = _cotangent(
        p2 - p1,
        p0 - p1,
        eps=eps,
    )
    cot2 = _cotangent(
        p0 - p2,
        p1 - p2,
        eps=eps,
    )

    # A negative cotangent means that the corresponding angle is obtuse.
    obtuse0 = cot0 < 0.0
    obtuse1 = cot1 < 0.0
    obtuse2 = cot2 < 0.0

    any_obtuse = (
        obtuse0
        | obtuse1
        | obtuse2
    )

    # Squared edge lengths.
    length01_squared = np.sum(
        (p1 - p0) ** 2,
        axis=1,
    )
    length12_squared = np.sum(
        (p2 - p1) ** 2,
        axis=1,
    )
    length20_squared = np.sum(
        (p0 - p2) ** 2,
        axis=1,
    )

    # Store the contribution of each face to each of its three corners.
    contribution0 = np.zeros(len(F), dtype=float)
    contribution1 = np.zeros(len(F), dtype=float)
    contribution2 = np.zeros(len(F), dtype=float)

    # Voronoi formula for non-obtuse triangles.
    non_obtuse = valid_faces & ~any_obtuse

    contribution0[non_obtuse] = (
        length01_squared[non_obtuse] * cot2[non_obtuse]
        + length20_squared[non_obtuse] * cot1[non_obtuse]
    ) / 8.0

    contribution1[non_obtuse] = (
        length12_squared[non_obtuse] * cot0[non_obtuse]
        + length01_squared[non_obtuse] * cot2[non_obtuse]
    ) / 8.0

    contribution2[non_obtuse] = (
        length20_squared[non_obtuse] * cot1[non_obtuse]
        + length12_squared[non_obtuse] * cot0[non_obtuse]
    ) / 8.0

    # Mixed-area correction when corner 0 is obtuse.
    mask = valid_faces & obtuse0
    contribution0[mask] = face_areas[mask] / 2.0
    contribution1[mask] = face_areas[mask] / 4.0
    contribution2[mask] = face_areas[mask] / 4.0

    # Mixed-area correction when corner 1 is obtuse.
    mask = valid_faces & obtuse1
    contribution0[mask] = face_areas[mask] / 4.0
    contribution1[mask] = face_areas[mask] / 2.0
    contribution2[mask] = face_areas[mask] / 4.0

    # Mixed-area correction when corner 2 is obtuse.
    mask = valid_faces & obtuse2
    contribution0[mask] = face_areas[mask] / 4.0
    contribution1[mask] = face_areas[mask] / 4.0
    contribution2[mask] = face_areas[mask] / 2.0

    # Accumulate all incident-face contributions at their vertices.
    vertex_areas = np.zeros(len(V), dtype=float)

    np.add.at(
        vertex_areas,
        F[:, 0],
        contribution0,
    )
    np.add.at(
        vertex_areas,
        F[:, 1],
        contribution1,
    )
    np.add.at(
        vertex_areas,
        F[:, 2],
        contribution2,
    )

    return vertex_areas


def build_mass_matrix(
    V: FloatArray,
    F: IntArray,
    *,
    eps: float = 1e-12,
) -> sp.csr_matrix:
    """Build the diagonal sparse mass matrix.

    The diagonal entry ``M[i, i]`` is the mixed Voronoi dual-cell area around
    vertex ``i``.
    """
    vertex_areas = compute_mixed_voronoi_vertex_areas(
        V,
        F,
        eps=eps,
    )

    return sp.diags(
        vertex_areas,
        format="csr",
    )


def compare_mass_matrices(
    manual_matrix: sp.spmatrix,
    reference_matrix: sp.spmatrix,
) -> dict[str, float | int]:
    """Compare two sparse mass matrices entry by entry."""
    if manual_matrix.shape != reference_matrix.shape:
        raise ValueError(
            "The two matrices must have the same shape; "
            f"received {manual_matrix.shape} and {reference_matrix.shape}."
        )

    difference = (
        manual_matrix
        - reference_matrix
    ).tocsr()

    absolute_differences = np.abs(
        difference.data
    )

    if difference.nnz == 0:
        maximum_difference = 0.0
        mean_difference = 0.0
    else:
        maximum_difference = float(
            absolute_differences.max()
        )
        mean_difference = float(
            absolute_differences.mean()
        )

    return {
        "number_of_rows": int(manual_matrix.shape[0]),
        "number_of_columns": int(manual_matrix.shape[1]),
        "number_of_nonzero_differences": int(difference.nnz),
        "maximum_absolute_difference": maximum_difference,
        "mean_absolute_difference": mean_difference,
    }


def verify_against_gpytoolbox(
    V: FloatArray,
    F: IntArray,
    *,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Verify the manual matrix against ``gpytoolbox.massmatrix(V, F)``.

    GPyToolbox uses the Voronoi mass matrix by default, so this comparison uses

        gpytoolbox.massmatrix(V, F, type="voronoi").
    """
    try:
        import gpytoolbox as gpy
    except ImportError as exc:
        raise ImportError(
            "GPyToolbox is required for verification. "
            "Install it with: uv add gpytoolbox"
        ) from exc

    manual_matrix = build_mass_matrix(
        V,
        F,
        eps=eps,
    )

    reference_matrix = gpy.massmatrix(
        V,
        F,
        type="voronoi",
    )

    return compare_mass_matrices(
        manual_matrix,
        reference_matrix,
    )
