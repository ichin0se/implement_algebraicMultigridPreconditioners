import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def generate_square_mesh(nx, ny):
    x_coords = np.linspace(0, 1, nx+1)
    y_coords = np.linspace(0, 1, ny+1)
    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([X.ravel(), Y.ravel()])

    elements = []
    for j in range(ny):
        for i in range(nx):
            p0 = j*(nx+1) + i
            p1 = p0 + 1
            p2 = p0 + (nx+1)
            p3 = p2 + 1
            elements.append([p0, p1, p2])
            elements.append([p1, p3, p2])

    elements = np.array(elements, dtype=np.int64)
    return points, elements

def local_stiffness_triangle(p0, p1, p2):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2

    detB = x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1)
    area = abs(detB)/2.0

    b0 = np.array([y1 - y2, x2 - x1])
    b1 = np.array([y2 - y0, x0 - x2])
    b2 = np.array([y0 - y1, x1 - x0])

    K = np.zeros((3,3))
    K[0,0] = np.dot(b0, b0)
    K[0,1] = np.dot(b0, b1)
    K[0,2] = np.dot(b0, b2)
    K[1,0] = np.dot(b1, b0)
    K[1,1] = np.dot(b1, b1)
    K[1,2] = np.dot(b1, b2)
    K[2,0] = np.dot(b2, b0)
    K[2,1] = np.dot(b2, b1)
    K[2,2] = np.dot(b2, b2)

    K *= 1.0/(4.0*area)
    return K



def assemble_system(points, elements):
    npoints = len(points)

    row_ids = []
    col_ids = []
    vals = []
    b = np.zeros(npoints)

    for elem in elements:
        p0 = points[elem[0]]
        p1 = points[elem[1]]
        p2 = points[elem[2]]
        Ke = local_stiffness_triangle(p0, p1, p2)

        area = abs(
            p0[0]*(p1[1] - p2[1]) +
            p1[0]*(p2[1] - p0[1]) +
            p2[0]*(p0[1] - p1[1])
        ) / 2.0

        fe = (area/3.0)*np.ones(3)

        for i in range(3):
            gi = elem[i]
            b[gi] += fe[i]
            for j in range(3):
                gj = elem[j]
                row_ids.append(gi)
                col_ids.append(gj)
                vals.append(Ke[i,j])

    A = sp.coo_matrix((vals, (row_ids, col_ids)), shape=(npoints, npoints)).tocsr()
    return A, b



def apply_dirichlet_boundary(A, b, points):
    A_lil = A.tolil()
    for i, (x, y) in enumerate(points):
        if (abs(x - 0.0) < 1e-14 or abs(x - 1.0) < 1e-14 or
            abs(y - 0.0) < 1e-14 or abs(y - 1.0) < 1e-14):
            A_lil.rows[i] = [i]
            A_lil.data[i] = [1.0]
            b[i] = 0.0
    A = A_lil.tocsr()
    return A, b



def simple_amg_coarsen(A):
    n = A.shape[0]
    group_id = np.floor(np.arange(n)/2).astype(int)
    ngroups = group_id[-1] + 1

    row_idx = np.arange(n)
    col_idx = group_id
    data = np.ones(n)
    P = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, ngroups)).tocsr()

    R = P.T

    A_coarse = R @ A @ P
    return A_coarse, P, R



def gauss_seidel(A, b, x, iterations=1):
    n = A.shape[0]
    D = A.diagonal()
    for _ in range(iterations):
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i+1]
            Ai = A.indices[row_start:row_end]
            Av = A.data[row_start:row_end]
            s = b[i]
            for col, val in zip(Ai, Av):
                if col != i:
                    s -= val * x[col]
            x[i] = s / D[i]
    return x

def v_cycle(hierarchy, level, b, x):
    A, P, R = hierarchy[level]

    x = gauss_seidel(A, b, x, iterations=1)

    if level < len(hierarchy)-1:
        r = b - A@x
        bc = R @ r
        xc = np.zeros_like(bc)
        xc = v_cycle(hierarchy, level+1, bc, xc)
        x += P @ xc

    x = gauss_seidel(A, b, x, iterations=1)
    return x

def build_hierarchy(A, max_levels=5, min_size=10):

    hierarchy = []
    currentA = A
    for _ in range(max_levels):
        n = currentA.shape[0]
        if n < min_size:
            break
        A_c, P, R = simple_amg_coarsen(currentA)
        hierarchy.append((currentA, P, R))
        currentA = A_c
    hierarchy.append((currentA, None, None))
    return hierarchy

def solve_amg_vcycle(A, b, max_iter=20, tol=1e-10):
    hierarchy = build_hierarchy(A)
    x = np.zeros_like(b, dtype=float)
    for i in range(max_iter):
        x = v_cycle(hierarchy, 0, b, x)
        r = b - A@x
        res_norm = np.linalg.norm(r)
        if res_norm < tol:
            print(f"Iteration {i}: residual={res_norm:.2e}, converged.")
            break
    else:
        print(f"Reached max_iter={max_iter}, residual={res_norm:.2e}")
    return x



def visualize_solution(points, elements, x):
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles=elements)
    plt.figure(figsize=(6,5))
    plt.tricontourf(triang, x, levels=20, cmap='rainbow')
    plt.colorbar(label='u')
    plt.title("Solution (2D Poisson, Simple AMG)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()



def demo_amg_from_scratch_2D(nx=8, ny=8):
    points, elements = generate_square_mesh(nx, ny)

    A, b = assemble_system(points, elements)

    A, b = apply_dirichlet_boundary(A, b, points)

    x = solve_amg_vcycle(A, b, max_iter=200, tol=6e-12)

    visualize_solution(points, elements, x)

if __name__ == "__main__":
    demo_amg_from_scratch_2D(nx=16, ny=16)
