"""Test the manual mixed-Voronoi mass matrix on a triangle mesh.

Run from the repository root:

    uv run python examples/test_mass_matrix.py data/bunny_small.obj
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from surface_log_map.mass_matrix import (
    build_mass_matrix,
    compute_face_areas,
    compute_mixed_voronoi_vertex_areas,
    verify_against_gpytoolbox,
)


def load_triangle_mesh(
    mesh_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangular mesh and return its vertices and faces."""
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
        description="Test the manually constructed mass matrix."
    )

    parser.add_argument(
        "mesh",
        type=Path,
        help="Path to a triangular mesh file.",
    )

    args = parser.parse_args()

    if not args.mesh.exists():
        raise FileNotFoundError(
            f"Mesh file not found: {args.mesh}"
        )

    V, F = load_triangle_mesh(args.mesh)

    # Compute the area assigned to every vertex.
    vertex_areas = compute_mixed_voronoi_vertex_areas(V, F)

    # Assemble the diagonal sparse mass matrix.
    mass_matrix = build_mass_matrix(V, F)

    # Compute the mesh surface area independently from the triangle areas.
    face_areas = compute_face_areas(V, F)
    total_mesh_area = float(face_areas.sum())

    # The sum of all vertex dual-cell areas should equal the surface area.
    total_vertex_area = float(vertex_areas.sum())

    print("Mesh information")
    print("----------------")
    print("Number of vertices:", len(V))
    print("Number of faces:", len(F))

    print("\nManual mass matrix")
    print("------------------")
    print("Shape:", mass_matrix.shape)
    print("Stored nonzero entries:", mass_matrix.nnz)
    print("Minimum diagonal entry:", mass_matrix.diagonal().min())
    print("Maximum diagonal entry:", mass_matrix.diagonal().max())

    print("\nArea conservation check")
    print("-----------------------")
    print("Total triangle area:", total_mesh_area)
    print("Sum of vertex dual-cell areas:", total_vertex_area)
    print(
        "Absolute difference:",
        abs(total_mesh_area - total_vertex_area),
    )

    comparison = verify_against_gpytoolbox(V, F)

    print("\nComparison with GPyToolbox")
    print("--------------------------")
    for metric_name, metric_value in comparison.items():
        print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    main()