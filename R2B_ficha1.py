import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation

# ---------------------------
# Matrizes auxiliares
# ---------------------------
def translation_matrix(dx, dy, dz):
    M = np.eye(4)
    M[:3, 3] = [dx, dy, dz]
    return M

def rotation_matrix(axis, theta):
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    R3 = np.array([
        [c+(1-c)*x*x,   (1-c)*x*y - s*z, (1-c)*x*z + s*y],
        [(1-c)*y*x+s*z, c+(1-c)*y*y,     (1-c)*y*z - s*x],
        [(1-c)*z*x-s*y, (1-c)*z*y+s*x,   c+(1-c)*z*z]
    ])
    M = np.eye(4)
    M[:3,:3] = R3
    return M

def apply_affine(M, pts):
    pts_h = np.c_[pts, np.ones(len(pts))]
    return (M @ pts_h.T).T[:, :3]

# ---------------------------
# Dados da questão
# ---------------------------
P0 = np.array([-1, 1, 0], dtype=float)   # ponto da reta
v_axis = np.array([1, -1, 1], dtype=float)  # direção da reta
theta = np.deg2rad(30)

# Rotação em torno da reta
T_to = translation_matrix(*P0)
T_from = translation_matrix(*(-P0))
R = rotation_matrix(v_axis, theta)
M_rot = T_to @ R @ T_from

# Escala
M_scale = np.diag([3, -2, 0.5, 1])

# Translação
M_trans = translation_matrix(1, -2, -3)

# Operador total
M_total = M_trans @ M_scale @ M_rot

print("Matriz de transformação total:\n", M_total)

# ---------------------------
# Definição do cubo
# ---------------------------
# Vértices do cubo unitário
vertices = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1]
])

# Faces do cubo
faces = [
    [0,1,2,3],
    [4,5,6,7],
    [0,1,5,4],
    [2,3,7,6],
    [1,2,6,5],
    [0,3,7,4]
]

cores = ['red','green','blue','yellow','orange','purple']

# ---------------------------
# Plot 3D
# ---------------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-6,6); ax.set_ylim(-6,6); ax.set_zlim(-6,6)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("Transformação afim em cubo")

# Wireframe inicial (tracejado)
arestas = [[0,1],[1,2],[2,3],[3,0],
           [4,5],[5,6],[6,7],[7,4],
           [0,4],[1,5],[2,6],[3,7]]
linhas = [[vertices[a], vertices[b]] for a,b in arestas]
wire = Line3DCollection(linhas, colors='k', linestyles='dashed')
ax.add_collection3d(wire)

# Reta de rotação em vermelho
t_vals = np.linspace(-5, 5, 50)
reta = np.array([P0 + t*v_axis for t in t_vals])
ax.plot(reta[:,0], reta[:,1], reta[:,2], 'r-', linewidth=2, label="Eixo de rotação")

# Ponto inicial (centro do cubo)
centro_ini = vertices.mean(axis=0)
ax.scatter(*centro_ini, color='blue', s=80, label="Centro inicial")

# Ponto final (centro após transformação)
centro_fim = apply_affine(M_total, vertices).mean(axis=0)
ax.scatter(*centro_fim, color='red', s=80, label="Centro final")

ax.legend()

# Cubo transformado (faces coloridas)
poly = Poly3DCollection([], alpha=0.7, edgecolor='k')
ax.add_collection3d(poly)

# ---------------------------
# Animação
# ---------------------------
def interpolate_matrix(M, t):
    return (1-t)*np.eye(4) + t*M

frames = 60
def update(frame):
    t = frame/frames
    Mt = interpolate_matrix(M_total, t)
    v_transf = apply_affine(Mt, vertices)
    poly.set_verts([[v_transf[idx] for idx in face] for face in faces])
    poly.set_facecolor(cores)
    return poly,

ani = FuncAnimation(fig, update, frames=frames+1, interval=120, blit=False)
plt.show()
