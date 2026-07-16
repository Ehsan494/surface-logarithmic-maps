"""Build a deterministic tangent frame at every mesh vertex.

This file presents a reference solution for the exercise:

    For every vertex v, deterministically compute a unit tangent vector t_v.

The construction uses the unit vertex normal ``n_v``. A tangent vector must
lie in the tangent plane, which means

    t_v · n_v = 0.

One tangent direction alone does not determine a full two-dimensional frame,
so after computing ``t1_v`` we define

    t2_v = n_v × t1_v.

The ordered triple

    (t1_v, t2_v, n_v)

forms a right-handed orthonormal frame.

Why a deterministic rule is needed
----------------------------------
There are infinitely many unit vectors perpendicular to a given normal.
Therefore, the normal alone does not select a unique tangent direction.

To make the result reproducible, this implementation follows a fixed rule:

1. Compare the normal with the three global coordinate axes.
2. Choose the axis least aligned with the normal.
3. Project that axis onto the tangent plane.
4. Normalize the projected vector.

Choosing the least-aligned axis avoids projecting an axis that is nearly
parallel to the normal, which would produce a very small and numerically
unstable vector.

The same input normals always produce the same tangent frames.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def validate_vertex_normals(
    vertex_normals: FloatArray,
) -> FloatArray:
    """Check and standardize an array of vertex normals.

    Parameters
    ----------
    vertex_normals
        Array with shape ``(n_vertices, 3)``.

    Returns
    -------
    vertex_normals
        Floating-point NumPy array with the same shape.
    """
    vertex_normals = np.asarray(
        vertex_normals,
        dtype=float,
    )

    if (
        vertex_normals.ndim != 2
        or vertex_normals.shape[1] != 3
    ):
        raise ValueError(
            "vertex_normals must have shape (n_vertices, 3); "
            f"received {vertex_normals.shape}."
        )

    if not np.all(np.isfinite(vertex_normals)):
        raise ValueError(
            "vertex_normals contains NaN or infinite values."
        )

    return vertex_normals


def normalize_vertex_normals(
    vertex_normals: FloatArray,
    *,
    eps: float = 1e-12,
) -> tuple[FloatArray, BoolArray]:
    """Normalize the supplied vertex normals safely.

    Returns
    -------
    unit_normals
        Unit normals. Invalid rows receive the zero vector.

    valid_vertices
        Boolean mask identifying rows whose lengths exceed ``eps``.
    """
    vertex_normals = validate_vertex_normals(
        vertex_normals
    )

    normal_lengths = np.linalg.norm(
        vertex_normals,
        axis=1,
    )

    valid_vertices = normal_lengths > eps

    unit_normals = np.zeros_like(
        vertex_normals
    )

    unit_normals[valid_vertices] = (
        vertex_normals[valid_vertices]
        / normal_lengths[valid_vertices, None]
    )

    return unit_normals, valid_vertices


def build_deterministic_tangent_frames(
    vertex_normals: FloatArray,
    *,
    eps: float = 1e-12,
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
    """Construct a deterministic orthonormal tangent frame at every vertex.

    Parameters
    ----------
    vertex_normals
        Vertex normals with shape ``(n_vertices, 3)``.

    eps
        Numerical threshold used to identify invalid normals or projected
        vectors that are too small to normalize safely.

    Returns
    -------
    tangent_1
        First unit tangent vector at every vertex, shape ``(n_vertices, 3)``.

    tangent_2
        Second unit tangent vector at every vertex, shape ``(n_vertices, 3)``.

    frames
        Orthonormal tangent frames with shape ``(n_vertices, 3, 2)``.

        ``frames[i, :, 0]`` is ``tangent_1[i]`` and
        ``frames[i, :, 1]`` is ``tangent_2[i]``.

    valid_vertices
        Boolean mask identifying vertices with valid tangent frames.
    """
    unit_normals, valid_normals = normalize_vertex_normals(
        vertex_normals,
        eps=eps,
    )

    number_of_vertices = len(unit_normals)

    # The three fixed coordinate axes are the only candidates used by the
    # deterministic rule.
    coordinate_axes = np.eye(3)

    # For each normal, compute its absolute alignment with x, y, and z.
    #
    # Since the coordinate axes are the standard basis vectors, these values
    # are simply the absolute normal components.
    absolute_alignments = np.abs(
        unit_normals
    )

    # Select the axis with the smallest absolute dot product with the normal.
    # This axis is the least parallel to the normal and therefore gives the
    # largest, most stable projection onto the tangent plane.
    chosen_axis_indices = np.argmin(
        absolute_alignments,
        axis=1,
    )

    chosen_axes = coordinate_axes[
        chosen_axis_indices
    ]

    # Orthogonally project each chosen axis a onto the tangent plane:
    #
    #     projected = a - (a · n) n.
    projection_coefficients = np.sum(
        chosen_axes * unit_normals,
        axis=1,
        keepdims=True,
    )

    projected_axes = (
        chosen_axes
        - projection_coefficients * unit_normals
    )

    projected_lengths = np.linalg.norm(
        projected_axes,
        axis=1,
    )

    valid_projections = projected_lengths > eps
    valid_vertices = (
        valid_normals
        & valid_projections
    )

    tangent_1 = np.zeros(
        (number_of_vertices, 3),
        dtype=float,
    )

    tangent_1[valid_vertices] = (
        projected_axes[valid_vertices]
        / projected_lengths[valid_vertices, None]
    )

    # The cross product produces a second tangent direction perpendicular to
    # both the normal and tangent_1. The ordering n × tangent_1 makes
    # (tangent_1, tangent_2, n) a right-handed frame.
    tangent_2 = np.cross(
        unit_normals,
        tangent_1,
    )

    tangent_2_lengths = np.linalg.norm(
        tangent_2,
        axis=1,
    )

    valid_tangent_2 = tangent_2_lengths > eps
    valid_vertices &= valid_tangent_2

    tangent_2[valid_vertices] = (
        tangent_2[valid_vertices]
        / tangent_2_lengths[valid_vertices, None]
    )

    # Keep invalid rows equal to zero.
    tangent_1[~valid_vertices] = 0.0
    tangent_2[~valid_vertices] = 0.0

    # Store both tangent basis vectors as the two columns of a 3-by-2 matrix.
    frames = np.stack(
        (tangent_1, tangent_2),
        axis=2,
    )

    return (
        tangent_1,
        tangent_2,
        frames,
        valid_vertices,
    )


def check_tangent_frames(
    vertex_normals: FloatArray,
    tangent_1: FloatArray,
    tangent_2: FloatArray,
    *,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Measure the orthonormality of a collection of tangent frames.

    A valid frame should satisfy

        ||t1|| = 1,
        ||t2|| = 1,
        t1 · n = 0,
        t2 · n = 0,
        t1 · t2 = 0.

    The returned values report the largest absolute violation of these
    conditions over all valid vertices.
    """
    unit_normals, valid_normals = normalize_vertex_normals(
        vertex_normals,
        eps=eps,
    )

    tangent_1 = np.asarray(
        tangent_1,
        dtype=float,
    )
    tangent_2 = np.asarray(
        tangent_2,
        dtype=float,
    )

    if (
        tangent_1.shape != unit_normals.shape
        or tangent_2.shape != unit_normals.shape
    ):
        raise ValueError(
            "vertex_normals, tangent_1, and tangent_2 must have "
            "the same shape (n_vertices, 3)."
        )

    tangent_1_lengths = np.linalg.norm(
        tangent_1,
        axis=1,
    )
    tangent_2_lengths = np.linalg.norm(
        tangent_2,
        axis=1,
    )

    valid = (
        valid_normals
        & (tangent_1_lengths > eps)
        & (tangent_2_lengths > eps)
    )

    if not np.any(valid):
        raise ValueError(
            "There are no valid tangent frames to check."
        )

    t1_dot_normal = np.sum(
        tangent_1[valid] * unit_normals[valid],
        axis=1,
    )

    t2_dot_normal = np.sum(
        tangent_2[valid] * unit_normals[valid],
        axis=1,
    )

    t1_dot_t2 = np.sum(
        tangent_1[valid] * tangent_2[valid],
        axis=1,
    )

    handedness = np.sum(
        np.cross(
            tangent_1[valid],
            tangent_2[valid],
        )
        * unit_normals[valid],
        axis=1,
    )

    return {
        "number_of_checked_frames": int(
            np.count_nonzero(valid)
        ),
        "maximum_t1_length_error": float(
            np.max(
                np.abs(
                    tangent_1_lengths[valid] - 1.0
                )
            )
        ),
        "maximum_t2_length_error": float(
            np.max(
                np.abs(
                    tangent_2_lengths[valid] - 1.0
                )
            )
        ),
        "maximum_abs_t1_dot_normal": float(
            np.max(np.abs(t1_dot_normal))
        ),
        "maximum_abs_t2_dot_normal": float(
            np.max(np.abs(t2_dot_normal))
        ),
        "maximum_abs_t1_dot_t2": float(
            np.max(np.abs(t1_dot_t2))
        ),
        "minimum_handedness": float(
            handedness.min()
        ),
        "maximum_handedness": float(
            handedness.max()
        ),
    }
