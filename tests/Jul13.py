import gpytoolbox as gpy
import numpy as np
import polyscope as ps
from pathlib import Path


if __name__ == "__main__":
    vertices, faces = gpy.read_mesh(
        str(Path(__file__).parent.parent / "data" / "bunny_small.obj")
    )

    ps.init()
    ps.register_surface_mesh("mesh", vertices, faces, smooth_shade=True)
    ps.show()
