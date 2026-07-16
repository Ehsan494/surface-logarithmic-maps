"""Test deterministic tangent frames on a triangle mesh.

Run from the repository root:

    uv run python examples/test_tangent_frames.py data/bunny_small.obj
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
        description="Test deterministic tangent frames."
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

    # Step 1: compute one unit normal at every mesh vertex.
    vertex_normals, valid_normals = (
        compute_area_weighted_vertex_normals(V, F)
    )

    # Step 2: use the normals to construct deterministic tangent frames.
    tangent_1, tangent_2, frames, valid_frames = (
        build_deterministic_tangent_frames(vertex_normals)
    )

    # Step 3: measure how closely the frames satisfy the orthonormality rules.
    checks = check_tangent_frames(
        vertex_normals,
        tangent_1,
        tangent_2,
    )

    print("Mesh information")
    print("----------------")
    print("Number of vertices:", len(V))
    print("Number of faces:", len(F))

    print("\nTangent-frame output")
    print("--------------------")
    print("tangent_1 shape:", tangent_1.shape)
    print("tangent_2 shape:", tangent_2.shape)
    print("frames shape:", frames.shape)
    print("Valid normals:", np.count_nonzero(valid_normals))
    print("Valid tangent frames:", np.count_nonzero(valid_frames))
    print("Invalid tangent frames:", np.count_nonzero(~valid_frames))

    print("\nOrthonormality checks")
    print("---------------------")
    for metric_name, metric_value in checks.items():
        print(f"{metric_name}: {metric_value}")

    # Determinism check: running the construction twice should give exactly
    # the same result for the same input normals.
    tangent_1_again, tangent_2_again, frames_again, valid_again = (
        build_deterministic_tangent_frames(vertex_normals)
    )

    print("\nDeterminism checks")
    print("------------------")
    print(
        "Maximum difference in tangent_1:",
        np.max(np.abs(tangent_1 - tangent_1_again)),
    )
    print(
        "Maximum difference in tangent_2:",
        np.max(np.abs(tangent_2 - tangent_2_again)),
    )
    print(
        "Maximum difference in frames:",
        np.max(np.abs(frames - frames_again)),
    )
    print(
        "Validity masks identical:",
        np.array_equal(valid_frames, valid_again),
    )


if __name__ == "__main__":
    main()