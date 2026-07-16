# Surface Logarithmic Maps

**MIT Summer Geometry Initiative (SGI) 2026**

This repository contains my implementations, experiments, visualizations, and
notes produced while volunteering on the Surface Logarithmic Maps project
during SGI 2026.

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

The broader project studies discrete methods for computing and consistently
aligning surface logarithmic maps on triangle meshes.

A logarithmic map centered at one surface point gives a local flattening of the
surface into a tangent plane. Since a single logarithmic map becomes distorted
away from its center, the broader goal is to compute several local maps and
align them using tangent frames, parallel transport, connection operators, and
vector heat diffusion.

The current material in this repository focuses on the foundational geometric
quantities needed for that pipeline:

- area-weighted vertex normals;
- mixed Voronoi vertex masses;
- deterministic tangent frames;
- numerical verification and visualization.

## Geometric Overview

At every mesh vertex, the local surface orientation is represented by a unit
normal vector. The tangent plane is the two-dimensional plane perpendicular to
this normal.

A tangent frame consists of two perpendicular unit vectors lying inside the
tangent plane. It acts as a local two-dimensional coordinate system for tangent
vectors.

These local frames will later be used to represent discrete parallel transport,
connection Laplacians, vector heat diffusion, and the alignment of multiple
surface logarithmic-map patches.

## Area-Weighted Vertex Normals

For a triangular face \(f=(i,j,k)\), define its oriented area vector by

$$
c_f=(v_j-v_i)\times(v_k-v_i).
$$

The unnormalized normal at vertex \(i\) is obtained by summing the area vectors
of all incident faces:

$$
\widetilde n_i=\sum_{f\ni i}c_f.
$$

The unit vertex normal is

$$
n_i=
\frac{\widetilde n_i}
{\left\|\widetilde n_i\right\|}.
$$

The implementation is verified against
`gpytoolbox.per_vertex_normals`.

## Mixed Voronoi Mass Matrix

Each vertex is assigned the area of its mixed Voronoi dual cell. These vertex
areas form a diagonal mass matrix

$$
M=
\operatorname{diag}(m_1,\ldots,m_n).
$$

The implementation handles both non-obtuse and obtuse triangles and is verified
against `gpytoolbox.massmatrix`.

## Deterministic Tangent Frames

Let \(n_i\) be the unit normal at vertex \(i\).

First, choose the global coordinate axis \(a_i\) that is least aligned with
\(n_i\). Project this axis onto the tangent plane:

$$
\widetilde t_{1,i}
=
a_i-(a_i\cdot n_i)n_i.
$$

Normalize it:

$$
t_{1,i}
=
\frac{\widetilde t_{1,i}}
{\left\|\widetilde t_{1,i}\right\|}.
$$

The second tangent direction is

$$
t_{2,i}=n_i\times t_{1,i}.
$$

The resulting tangent frame is

$$
T_i=
\begin{bmatrix}
t_{1,i} & t_{2,i}
\end{bmatrix}.
$$

The implementation verifies unit length, tangency, orthogonality, right-handed
orientation, and determinism.

## Current Implementation

### Completed

1. Computing area-weighted per-vertex normals.
2. Verifying vertex normals against `gpytoolbox.per_vertex_normals`.
3. Constructing a mixed Voronoi vertex mass matrix.
4. Verifying the mass matrix against `gpytoolbox.massmatrix`.
5. Constructing deterministic orthonormal tangent frames.
6. Checking tangency, orthonormality, handedness, and determinism.
7. Visualizing vertex normals and tangent frames in Polyscope.

### Planned or in progress

1. Edgewise parallel transport.
2. Rodrigues rotations between neighboring tangent spaces.
3. Cotangent edge weights.
4. Sparse connection-Laplacian assembly.
5. Direct connection-energy verification.
6. Vector heat diffusion.
7. Seed-to-seed transport extraction.
8. Surface logarithmic-map patch alignment.

## Repository Structure

```text
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

Only list files in this section that are actually present in the repository.

## Dependencies

The current implementation uses:

```text
numpy
scipy
matplotlib
trimesh
gpytoolbox
polyscope
```

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

The corresponding `requirements.txt` file should contain:

```text
numpy
scipy
matplotlib
trimesh
gpytoolbox
polyscope
```

## Usage

Run the mass-matrix verification:

```bash
python examples/test_mass_matrix.py data/sample_meshes/bunny_small.obj
```

Run the tangent-frame verification:

```bash
python examples/test_tangent_frames.py data/sample_meshes/bunny_small.obj
```

Visualize the vertex normals:

```bash
python examples/visualize_vertex_normals_polyscope.py data/sample_meshes/bunny_small.obj
```

Visualize the tangent frames and normals:

```bash
python examples/visualize_tangent_frames.py data/sample_meshes/bunny_small.obj --show-normals
```

## Visualizations

### Vertex normals

The arrows represent area-weighted unit normals computed at the mesh vertices.

![Area-weighted vertex normals](figures/vertex_normals_visualization.png)

### Tangent frames

The red arrows represent \(t_1\), the green arrows represent \(t_2\), and the
magenta arrows represent the vertex normals.

![Deterministic tangent frames](figures/tangent_frames_visualization.png)

## Status

This repository is under active development as part of my SGI 2026 volunteer
work.

- [x] Area-weighted vertex normals
- [x] Verification against GPyToolbox
- [x] Mixed Voronoi vertex mass matrix
- [x] Mass-matrix verification against GPyToolbox
- [x] Deterministic tangent frames
- [x] Tangent-frame numerical validation
- [x] Polyscope visualizations
- [ ] Rodrigues edge transports
- [ ] Cotangent edge weights
- [ ] Sparse connection-Laplacian assembly
- [ ] Direct energy verification
- [ ] Vector heat diffusion
- [ ] Reduced-graph transport extraction
- [ ] Surface logarithmic-map patch alignment

## License

See the [LICENSE](LICENSE) file for details.
