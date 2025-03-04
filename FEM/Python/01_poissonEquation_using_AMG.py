import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pyamg

from typing import List, Tuple
from numpy import ndarray, float64


def generate_square_mesh(nx: int, ny: int) -> Tuple[ndarray, ndarray]:
    """
    Divide a simple structural lattice into triangular indicesSet_of_primitives and returns node coordinates and indicesSet_of_primitives.
    nx, ny is the number of divisions in the x and y directions
    -> (nx+1)*(ny+1) nodes,
    Split each cell into two triangular indicesSet_of_primitives.

    generate_square_mesh(divisions_in_the_x_axis: int, divisions_in_the_y_axis: int) -> Tuple[2*nx_matrix_of_coordinate_sets: ndarray, A_list_of_node_indices_representing_a_triangle_mesh: ndarray]
    """
    x_coords: ndarray = np.linspace(0, 1, nx + 1)
    y_coords: ndarray = np.linspace(0, 1, ny + 1)
    X, Y = np.meshgrid(x_coords, y_coords)
    coords: ndarray = np.column_stack([X.ravel(), Y.ravel()])
    """
        coords = [     x0, y0     ]
                 [     x1, y1     ]
                 [     x2, y2     ]
                        ...
                 [x(nx+1), y(ny+1)]
        coords is the 2 x (nx+1) matrix of the ndarray type.
        coords stores coordinates of nodes in the mesh.
    """

    indicesSet_of_primitives: List[List[int]] = []
    for j in range(ny):
        for i in range(nx):
            p0: int = j * (nx + 1) + i
            p1: int = j * (nx + 1) + i + 1
            p2: int = (j + 1) * (nx + 1) + i
            p3: int = (j + 1) * (nx + 1) + i + 1
            """
                p0 is the linear index when traversing a mesh
                p1 is the linear index of the node adjacent to p0 in the x-direction
                p2 is the linear index of the node adjacent to p0 in the y-direction
                p3 is the linear index of the node diagonally above p0
            """
            indicesSet_of_primitives.append([p0, p1, p2])
            indicesSet_of_primitives.append([p1, p3, p2])
            """
               y
               ^
               |
               |   [i, j+1]        [i+1, j+1]
               |      p2               p3
         (j+1) |        +-------------+
               |        |\  triangle2 |
               |        |    \        |
               |        |        \    |
               |        | triangle1  \|
           (j) |        +-------------+
               |      p0               p1
               |    [i, j]          [i+1, j]
               |
               +-----------------------------------------> x
                       (i)          (i+1)
            """

    indicesSet_of_primitives: ndarray = np.array(
        indicesSet_of_primitives, dtype=np.int64
    )

    return coords, indicesSet_of_primitives


def local_stiffness_triangle(p0, p1, p2):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2

    detB: int = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)
    area: float = abs(detB) / 2.0

    b0: ndarray = np.array([y1 - y2, x2 - x1])
    b1: ndarray = np.array([y2 - y0, x0 - x2])
    b2: ndarray = np.array([y0 - y1, x1 - x0])

    K: ndarray = np.zeros((3, 3))
    K[0, 0]: float64 = np.dot(b0, b0)
    K[0, 1] = np.dot(b0, b1)
    K[0, 2] = np.dot(b0, b2)
    K[1, 0] = np.dot(b1, b0)
    K[1, 1] = np.dot(b1, b1)
    K[1, 2] = np.dot(b1, b2)
    K[2, 0] = np.dot(b2, b0)
    K[2, 1] = np.dot(b2, b1)
    K[2, 2] = np.dot(b2, b2)

    K *= 1.0 / (4.0 * area)
    return K


def assemble_system(coords, indicesSet_of_primitives):
    npoints: int = len(coords)

    row_ids = []
    col_ids = []
    vals = []
    b = np.zeros(npoints)

    for elem in indicesSet_of_primitives:
        p0 = coords[elem[0]]
        p1 = coords[elem[1]]
        p2 = coords[elem[2]]
        Ke = local_stiffness_triangle(p0, p1, p2)

        area = (
            abs(
                p0[0] * (p1[1] - p2[1])
                + p1[0] * (p2[1] - p0[1])
                + p2[0] * (p0[1] - p1[1])
            )
            / 2.0
        )

        fe = (area / 3.0) * np.ones(3)

        for i in range(3):
            gi = elem[i]
            b[gi] += fe[i]
            for j in range(3):
                gj = elem[j]
                row_ids.append(gi)
                col_ids.append(gj)
                vals.append(Ke[i, j])

    A = sp.coo_matrix((vals, (row_ids, col_ids)), shape=(npoints, npoints)).tocsr()
    return A, b


def apply_dirichlet_boundary(A, b, coords):
    A_lil = A.tolil()
    for i, (x, y) in enumerate(coords):
        if (
            abs(x - 0.0) < 1e-14
            or abs(x - 1.0) < 1e-14
            or abs(y - 0.0) < 1e-14
            or abs(y - 1.0) < 1e-14
        ):
            A_lil.rows[i] = [i]
            A_lil.data[i] = [1.0]
            b[i] = 0.0

    return A_lil.tocsr(), b


def visualize_solution(coords, indicesSet_of_primitives, x):
    triang = mtri.Triangulation(
        coords[:, 0], coords[:, 1], triangles=indicesSet_of_primitives
    )

    plt.figure(figsize=(6, 5))
    plt.tricontourf(triang, x, levels=20, cmap="rainbow")
    plt.colorbar(label="u")
    plt.title("Solution of Poisson (FEM + AMG)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()


def solve_poisson_with_amg(nx=8, ny=8):
    coords, indicesSet_of_primitives = generate_square_mesh(nx, ny)

    A, b = assemble_system(coords, indicesSet_of_primitives)

    A, b = apply_dirichlet_boundary(A, b, coords)

    ml = pyamg.ruge_stuben_solver(A)
    M = ml.aspreconditioner(cycle="V")

    x0 = np.zeros_like(b)
    x, info = spla.cg(A, b, x0=x0, maxiter=200, tol=6e-12, M=M)

    print(f"CG finished (info={info}, where 0 indicates successful convergence).")

    visualize_solution(coords, indicesSet_of_primitives, x)


if __name__ == "__main__":
    solve_poisson_with_amg(nx=16, ny=16)
