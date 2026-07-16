# Surface Logarithmic Maps

**MIT Summer Geometry Initiative (SGI) 2026**

## Project Team

### Mentors

- **Stephanie Wang** — Technische Universität Berlin (TU Berlin)
- **Yousuf Soliman** — Side Effects Software, Inc.  
 

### Volunteer

- **Ehsan Shams** — Alexandria University, Egypt

### SGI Fellows

- **Shannon Cudworth** — University of California, Berkeley
- **Anja Milutinović** — University of Belgrade

## Project Description

This project develops a discrete framework for computing surface logarithmic maps on triangle meshes. We construct vertex normals and tangent frames, define edgewise parallel transport using Rodrigues rotations, assemble a cotangent-weighted connection Laplacian, and use vector heat diffusion to propagate tangent directions across the surface. These tools provide the geometric information needed to estimate relative directions and logarithmic-map coordinates between selected surface points.

## Overview

A tangent vector at one point of a curved surface cannot be compared directly with a tangent vector at another point because the two vectors belong to differently oriented tangent planes. To account for this changing geometry, we construct a discrete connection on the edges of a triangle mesh.

For an edge \((i,j)\), the connection map

\[
\rho_{ij}\in\mathbb{R}^{3\times 3}
\]

transports a vector between the tangent spaces at vertices \(i\) and \(j\). These edgewise transport maps are combined with cotangent weights to define the connection-Laplacian energy

\[
E(z)
=
\frac{1}{2}
\sum_{(i,j)\in E}
w_{ij}
\left\|
z_j-\rho_{ij}z_i
\right\|^2,
\]

where \(z_i\) is the vector stored at vertex \(i\), \(w_{ij}\) is the cotangent weight of edge \((i,j)\), and \(\rho_{ij}\) is the discrete parallel-transport map.

The resulting connection Laplacian is then used in a vector heat diffusion process to propagate tangent directions smoothly over the surface.

## Current Implementation

The current workflow includes:

1. Computing area-weighted per-vertex normals.
2. Verifying the normals against `gpytoolbox.per_vertex_normals`.
3. Constructing deterministic tangent frames.
4. Computing Rodrigues rotations between neighboring vertex normals.
5. Computing cotangent edge weights.
6. Assembling a block connection Laplacian.
7. Verifying its quadratic form against a direct edge-based energy calculation.
8. Solving an implicit vector heat system.
9. Extracting relative directions and rotations between selected surface vertices.

## Repository Structure
```
surface-logarithmic-maps/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   └── surface_log_map/
│       ├── __init__.py
│       ├── vertex_normals.py
│       ├── mass_matrix.py
│       └── tangent_frames.py
│
├── examples/
│   ├── visualize_vertex_normals_polyscope.py
│   ├── test_mass_matrix.py
│   ├── test_tangent_frames.py
│   └── visualize_tangent_frames.py
│
├── tests/
│   ├── test_vertex_normals.py
│   ├── test_mass_matrix.py
│   └── test_tangent_frames.py
│
├── docs/
│   ├── deterministic_tangent_frames_note.tex
│   └── deterministic_tangent_frames_note.pdf
│
├── figures/
│   ├── vertex_normals_visualization.png
│   └── tangent_frames_visualization.png
│
└── data/
    └── sample_meshes/
        └── bunny_small.obj
```


## Dependencies

The current implementation uses:

```text
numpy
scipy
matplotlib
trimesh
gpytoolbox
```

Install the dependencies with:

```bash
pip install numpy scipy matplotlib trimesh gpytoolbox
```

## Status

This repository is under active development as part of SGI 2026.

- [x] Area-weighted vertex normals
- [x] Verification against GPyToolbox
- [ ] Deterministic tangent frames
- [ ] Rodrigues edge transports
- [ ] Cotangent edge weights
- [ ] Sparse connection-Laplacian assembly
- [ ] Direct energy verification
- [ ] Vector heat diffusion
- [ ] Reduced-graph transport extraction

## License

