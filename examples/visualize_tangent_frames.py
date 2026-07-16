"""Visualize deterministic tangent frames in Polyscope.

This script loads a triangular mesh, computes area-weighted vertex normals,
constructs a deterministic tangent frame at every vertex, and displays a
readable subset of the frame vectors.

The numerical construction remains in:

    src/surface_log_map/tangent_frames.py

Run from the repository root:

    uv run python examples/visualize_tangent_frames.py data/bunny_small.obj
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polyscope as ps
import trimesh

from surface_log_map.vertex_normals import (
    compute_area_weighted_vertex_normals,
)
from surface_log_map.tangent_frames import (
    build_deterministic_tangent_frames,
    check_tangent_frames,
)


def load_triangle_mesh(
    mesh_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangular mesh and return its vertex and face arrays."""
    mesh = trimesh.load(mesh_path, process=True)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            f"Expected a triangle mesh; loaded {type(mesh).__name__}."
        )

    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()

    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(
            f"V must have shape (n_vertices, 3); received {V.shape}."
        )

    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(
            f"F must have shape (n_faces, 3); received {F.shape}."
        )

    return V, F


def build_display_mask(
    number_of_vertices: int,
    valid_vertices: np.ndarray,
    target_number_of_frames: int,
) -> np.ndarray:
    """Choose a readable subset of valid vertices for visualization.

    Tangent frames are computed at every valid mesh vertex. Only a subset is
    displayed because drawing all frames on a dense mesh can hide the surface.
    """
    target_number_of_frames = max(1, target_number_of_frames)
    sampling_step = max(
        1,
        number_of_vertices // target_number_of_frames,
    )

    display_mask = np.zeros(
        number_of_vertices,
        dtype=bool,
    )

    display_mask[::sampling_step] = True
    display_mask &= valid_vertices

    return display_mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize deterministic tangent frames in Polyscope."
    )

    parser.add_argument(
        "mesh",
        type=Path,
        help="Path to the triangular mesh file.",
    )

    parser.add_argument(
        "--target-frames",
        type=int,
        default=350,
        help="Approximate number of tangent frames to display.",
    )

    parser.add_argument(
        "--vector-length-fraction",
        type=float,
        default=0.012,
        help="Displayed vector length as a fraction of the mesh diagonal.",
    )

    parser.add_argument(
        "--show-normals",
        action="store_true",
        help="Also display the vertex normals.",
    )

    args = parser.parse_args()

    if not args.mesh.exists():
        raise FileNotFoundError(
            f"Mesh file not found: {args.mesh}"
        )

    V, F = load_triangle_mesh(args.mesh)

    # Step 1: compute one area-weighted unit normal at every mesh vertex.
    vertex_normals, valid_normals = (
        compute_area_weighted_vertex_normals(V, F)
    )

    # Step 2: construct the deterministic tangent basis (t1, t2).
    tangent_1, tangent_2, frames, valid_frames = (
        build_deterministic_tangent_frames(vertex_normals)
    )

    # Step 3: verify the orthonormality of the constructed frames.
    checks = check_tangent_frames(
        vertex_normals,
        tangent_1,
        tangent_2,
    )

    display_mask = build_display_mask(
        len(V),
        valid_frames,
        args.target_frames,
    )

    mesh_diagonal = np.linalg.norm(
        V.max(axis=0) - V.min(axis=0)
    )

    if mesh_diagonal <= 1e-12:
        raise ValueError(
            "The mesh has an approximately zero bounding-box size."
        )

    vector_length = (
        args.vector_length_fraction
        * mesh_diagonal
    )

    # The mathematical vectors have unit length. Scaling them before passing
    # them to Polyscope gives direct control over the displayed arrow length.
    display_tangent_1 = np.zeros_like(tangent_1)
    display_tangent_2 = np.zeros_like(tangent_2)
    display_normals = np.zeros_like(vertex_normals)

    display_tangent_1[display_mask] = (
        vector_length
        * tangent_1[display_mask]
    )

    display_tangent_2[display_mask] = (
        vector_length
        * tangent_2[display_mask]
    )

    display_normals[display_mask] = (
        vector_length
        * vertex_normals[display_mask]
    )

    print("Mesh information")
    print("----------------")
    print("Number of vertices:", len(V))
    print("Number of faces:", len(F))

    print("\nTangent-frame information")
    print("-------------------------")
    print("Valid normals:", np.count_nonzero(valid_normals))
    print("Valid tangent frames:", np.count_nonzero(valid_frames))
    print("Frames displayed:", np.count_nonzero(display_mask))
    print("Frame array shape:", frames.shape)

    print("\nOrthonormality checks")
    print("---------------------")
    for metric_name, metric_value in checks.items():
        print(f"{metric_name}: {metric_value}")

    ps.init()
    ps.set_program_name("Deterministic Tangent Frames")

    surface = ps.register_surface_mesh(
        "triangle mesh",
        V,
        F,
        smooth_shade=False,
        color=(0.18, 0.58, 0.92),
    )

    surface.add_vector_quantity(
        "tangent direction t1",
        display_tangent_1,
        defined_on="vertices",
        vectortype="ambient",
        enabled=True,
        color=(0.95, 0.20, 0.20),
        radius=0.00045 * mesh_diagonal,
    )

    surface.add_vector_quantity(
        "tangent direction t2",
        display_tangent_2,
        defined_on="vertices",
        vectortype="ambient",
        enabled=True,
        color=(0.15, 0.80, 0.30),
        radius=0.00045 * mesh_diagonal,
    )

    surface.add_vector_quantity(
        "vertex normals",
        display_normals,
        defined_on="vertices",
        vectortype="ambient",
        enabled=args.show_normals,
        color=(0.95, 0.15, 0.80),
        radius=0.00045 * mesh_diagonal,
    )

    ps.show()


if __name__ == "__main__":
    main()
