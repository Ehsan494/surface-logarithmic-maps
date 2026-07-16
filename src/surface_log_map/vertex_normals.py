"""Compute and verify area-weighted vertex normals on a triangle mesh.

This file presents a complete reference solution for the exercise:

1. Compute vertex normals manually from area-weighted face normals.
2. Verify the manual result against
   ``gpytoolbox.per_vertex_normals(V, F)``.

Mathematical idea
-----------------
Consider one triangular face with vertex positions ``p0``, ``p1``, and ``p2``.
Two edge vectors are

    e1 = p1 - p0
    e2 = p2 - p0.

Their cross product

    c_f = e1 × e2

is perpendicular to the triangle. Its magnitude equals twice the triangle
area:

    ||c_f|| = 2 A_f.

If ``n_f`` is the unit face normal, then

    c_f = 2 A_f n_f.

Therefore, adding the unnormalized cross products of the faces incident to a
vertex automatically forms an area-weighted sum of their normal directions.
The common factor 2 disappears when the accumulated vector is normalized.

Mesh representation
-------------------
``V`` is a NumPy array with shape ``(n_vertices, 3)``. Each row stores the
three-dimensional coordinates of one vertex.

``F`` is a NumPy array with shape ``(n_faces, 3)``. Each row stores the three
vertex indices of one triangular face.

The orientation of a normal depends on the ordering of the triangle vertices.
Reversing the order of a face reverses its normal.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]


def validate_triangle_mesh(
    V: FloatArray,
    F: IntArray,
) -> tuple[FloatArray, IntArray]:
    """Check that the input arrays describe a triangular mesh.

    Parameters
    ----------
    V
        Vertex coordinates with shape ``(n_vertices, 3)``.

    F
        Triangle vertex indices with shape ``(n_faces, 3)``.

    Returns
    -------
    V, F
        Standardized NumPy arrays with floating-point vertex coordinates and
        integer face indices.
    """
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


def compute_face_geometry(
    V: FloatArray,
    F: IntArray,
    *,
    eps: float = 1e-12,
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
    """Compute the normal and area of every triangular face.

    For each triangle, the function computes:

    - the unnormalized cross-product normal;
    - the triangle area;
    - the corresponding unit face normal;
    - whether the face is nondegenerate.

    A degenerate triangle has nearly zero area, so its unit normal cannot be
    defined reliably.

    Returns
    -------
    face_cross
        Unnormalized face normals with shape ``(n_faces, 3)``.

    face_areas
        Triangle areas with shape ``(n_faces,)``.

    face_normals
        Unit face normals with shape ``(n_faces, 3)``. Degenerate faces
        receive the zero vector.

    valid_faces
        Boolean array identifying nondegenerate faces.
    """
    V, F = validate_triangle_mesh(V, F)

    # Gather the three vertex positions of every triangle.
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]

    # Form two edges that start at p0.
    edge_1 = p1 - p0
    edge_2 = p2 - p0

    # The cross product is perpendicular to the face.
    face_cross = np.cross(edge_1, edge_2)

    # Its length equals twice the triangle area.
    double_areas = np.linalg.norm(face_cross, axis=1)
    face_areas = 0.5 * double_areas

    valid_faces = double_areas > eps

    # Normalize the cross products of the nondegenerate faces.
    face_normals = np.zeros_like(face_cross)
    face_normals[valid_faces] = (
        face_cross[valid_faces]
        / double_areas[valid_faces, None]
    )

    return face_cross, face_areas, face_normals, valid_faces


def compute_area_weighted_vertex_normals(
    V: FloatArray,
    F: IntArray,
    *,
    eps: float = 1e-12,
) -> tuple[FloatArray, BoolArray]:
    """Compute one area-weighted unit normal at every mesh vertex.

    Every face contributes its unnormalized cross-product normal to its three
    incident vertices. Since the magnitude of this vector is proportional to
    the face area, larger faces contribute more strongly.

    The accumulated vector at each vertex is normalized to obtain a unit
    vertex normal.

    Returns
    -------
    vertex_normals
        Unit vertex normals with shape ``(n_vertices, 3)``.

    valid_vertices
        Boolean array indicating which accumulated vectors could be normalized
        safely. Isolated or degenerate vertices receive the zero vector.
    """
    V, F = validate_triangle_mesh(V, F)

    face_cross, _, _, _ = compute_face_geometry(
        V,
        F,
        eps=eps,
    )

    # This array stores the sum of incident face contributions at each vertex.
    vertex_normal_sums = np.zeros((len(V), 3), dtype=float)

    # Each triangle contributes the same face_cross vector to all three of its
    # corners. np.add.at correctly accumulates values when an index appears
    # many times.
    for corner in range(3):
        np.add.at(
            vertex_normal_sums,
            F[:, corner],
            face_cross,
        )

    accumulated_lengths = np.linalg.norm(
        vertex_normal_sums,
        axis=1,
    )

    valid_vertices = accumulated_lengths > eps

    vertex_normals = np.zeros_like(vertex_normal_sums)
    vertex_normals[valid_vertices] = (
        vertex_normal_sums[valid_vertices]
        / accumulated_lengths[valid_vertices, None]
    )

    return vertex_normals, valid_vertices


def compare_vertex_normals(
    manual_normals: FloatArray,
    reference_normals: FloatArray,
    *,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Compare two corresponding fields of vertex normals.

    The comparison uses two measures:

    Euclidean difference
        ``||n_manual - n_reference||``

    Angular difference
        ``arccos(n_manual · n_reference)`` in degrees

    Both inputs are normalized again inside the function so that the
    comparison focuses on direction.

    Returns
    -------
    metrics
        A dictionary containing the number of compared vertices and the
        maximum and mean Euclidean and angular differences.
    """
    manual_normals = np.asarray(manual_normals, dtype=float)
    reference_normals = np.asarray(reference_normals, dtype=float)

    if (
        manual_normals.shape != reference_normals.shape
        or manual_normals.ndim != 2
        or manual_normals.shape[1] != 3
    ):
        raise ValueError(
            "Both normal arrays must have the same shape (n_vertices, 3)."
        )

    manual_lengths = np.linalg.norm(manual_normals, axis=1)
    reference_lengths = np.linalg.norm(reference_normals, axis=1)

    valid_pairs = (
        (manual_lengths > eps)
        & (reference_lengths > eps)
    )

    if not np.any(valid_pairs):
        raise ValueError("There are no valid normal pairs to compare.")

    manual_unit = (
        manual_normals[valid_pairs]
        / manual_lengths[valid_pairs, None]
    )
    reference_unit = (
        reference_normals[valid_pairs]
        / reference_lengths[valid_pairs, None]
    )

    euclidean_differences = np.linalg.norm(
        manual_unit - reference_unit,
        axis=1,
    )

    dot_products = np.sum(
        manual_unit * reference_unit,
        axis=1,
    )

    # Numerical rounding can produce values just outside [-1, 1].
    dot_products = np.clip(dot_products, -1.0, 1.0)

    angular_differences_degrees = np.degrees(
        np.arccos(dot_products)
    )

    return {
        "num_compared_vertices": int(
            np.count_nonzero(valid_pairs)
        ),
        "max_euclidean_difference": float(
            euclidean_differences.max()
        ),
        "mean_euclidean_difference": float(
            euclidean_differences.mean()
        ),
        "max_angular_difference_degrees": float(
            angular_differences_degrees.max()
        ),
        "mean_angular_difference_degrees": float(
            angular_differences_degrees.mean()
        ),
    }


def verify_against_gpytoolbox(
    V: FloatArray,
    F: IntArray,
    *,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Verify the manual normals against GPyToolbox.

    The reference field is computed with

        gpytoolbox.per_vertex_normals(V, F).

    The returned metrics should be close to zero when both implementations
    follow the same area-weighted convention.
    """
    try:
        import gpytoolbox as gpy
    except ImportError as exc:
        raise ImportError(
            "GPyToolbox is required for verification. "
            "Install it with: uv add gpytoolbox"
        ) from exc

    manual_normals, _ = compute_area_weighted_vertex_normals(
        V,
        F,
        eps=eps,
    )

    reference_normals = gpy.per_vertex_normals(V, F)

    return compare_vertex_normals(
        manual_normals,
        reference_normals,
        eps=eps,
    )