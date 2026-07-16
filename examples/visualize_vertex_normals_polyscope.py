"""Visualize manually computed vertex normals in Polyscope.

This script imports the numerical implementation from ``vertex_normals.py``.
It loads a mesh, computes the manual normals, prints the comparison with
GPyToolbox, and displays the manual normals in Polyscope.

Run from the repository root:

    uv run python examples/visualize_vertex_normals.py data/bunny_small.obj
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polyscope as ps
import trimesh

from surface_log_map.vertex_normals import (
    compute_area_weighted_vertex_normals,
    verify_against_gpytoolbox,
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

    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("The input mesh must contain triangular faces.")

    return V, F


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute, verify, and visualize area-weighted vertex normals."
        )
    )

    parser.add_argument(
        "mesh",
        type=Path,
        help="Path to the triangular mesh file.",
    )

    parser.add_argument(
        "--target-arrows",
        type=int,
        default=700,
        help="Approximate number of arrows to display.",
    )

    parser.add_argument(
        "--arrow-length-fraction",
        type=float,
        default=0.012,
        help="Arrow length as a fraction of the mesh diagonal.",
    )

    args = parser.parse_args()

    if not args.mesh.exists():
        raise FileNotFoundError(
            f"Mesh file not found: {args.mesh}"
        )

    V, F = load_triangle_mesh(args.mesh)

    manual_normals, valid_vertices = (
        compute_area_weighted_vertex_normals(V, F)
    )

    print("Mesh information")
    print("----------------")
    print("Number of vertices:", len(V))
    print("Number of faces:", len(F))
    print("Valid vertex normals:", np.count_nonzero(valid_vertices))
    print("Invalid vertex normals:", np.count_nonzero(~valid_vertices))

    comparison = verify_against_gpytoolbox(V, F)

    print("\nComparison with GPyToolbox")
    print("--------------------------")
    for metric_name, metric_value in comparison.items():
        print(f"{metric_name}: {metric_value}")

    # The computation produces a normal at every valid vertex. We display only
    # a subset to keep the figure readable on dense meshes.
    target_arrows = max(1, args.target_arrows)
    sampling_step = max(1, len(V) // target_arrows)

    display_mask = np.zeros(len(V), dtype=bool)
    display_mask[::sampling_step] = True
    display_mask &= valid_vertices

    mesh_diagonal = np.linalg.norm(
        V.max(axis=0) - V.min(axis=0)
    )

    if mesh_diagonal <= 1e-12:
        raise ValueError("The mesh has an approximately zero size.")

    arrow_length = (
        args.arrow_length_fraction
        * mesh_diagonal
    )

    # The normal field contains unit vectors. Scaling the vectors before
    # passing them to Polyscope gives direct control over their physical size.
    display_normals = np.zeros_like(manual_normals)
    display_normals[display_mask] = (
        arrow_length
        * manual_normals[display_mask]
    )

    ps.init()
    ps.set_program_name("Area-Weighted Vertex Normals")

    surface = ps.register_surface_mesh(
        "triangle mesh",
        V,
        F,
        smooth_shade=False,
        color=(0.18, 0.58, 0.92),
    )

    surface.add_vector_quantity(
        "manual vertex normals",
        display_normals,
        defined_on="vertices",
        vectortype="ambient",
        enabled=True,
        color=(1.0, 0.0, 0.72),
        radius=0.00045 * mesh_diagonal,
    )

    print("\nPolyscope visualization")
    print("-----------------------")
    print("Normals computed:", np.count_nonzero(valid_vertices))
    print("Normals displayed:", np.count_nonzero(display_mask))
    print("Arrow length:", arrow_length)

    ps.show()


if __name__ == "__main__":
    main()