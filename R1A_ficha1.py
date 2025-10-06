import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# ---------------------------
# Geometria do cubo
# ---------------------------
def cube_vertices(size=1.0, origin=(0.0, 0.0, 0.0)):
    ox, oy, oz = origin
    s = size
    V = np.array([
        [ox,   oy,   oz  ],
        [ox+s, oy,   oz  ],
        [ox+s, oy+s, oz  ],
        [ox,   oy+s, oz  ],
        [ox,   oy,   oz+s],
        [ox+s, oy,   oz+s],
        [ox+s, oy+s, oz+s],
        [ox,   oy+s, oz+s],
    ], dtype=float)
    return V

def cube_faces(V):
    return [
        [V[0], V[1], V[2], V[3]],
        [V[4], V[5], V[6], V[7]],
        [V[0], V[1], V[5], V[4]],
        [V[2], V[3], V[7], V[6]],
        [V[1], V[2], V[6], V[5]],
        [V[0], V[3], V[7], V[4]],
    ]

# --------------------------------
# Matrizes afins homogêneas em R^3
# --------------------------------
def translation_matrix(dx, dy, dz):
    M = np.eye(4)
    M[:3, 3] = [dx, dy, dz]
    return M

def rotation_z_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    M = np.eye(4)
    M[:3, :3] = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]])
    return M

def apply_affine(M, points):
    homog = np.c_[points, np.ones(len(points))]
    transformed = (M @ homog.T).T
    return transformed[:, :3]

# --------------------
# Configuração inicial
# --------------------
V0 = cube_vertices(size=1.0, origin=(0.0, 0.0, 0.0))

# Eixo de rotação s: x=2, y=1 (paralelo a z). Ponto base p=(2,1,0)
p = np.array([2.0, 1.0, 0.0])
T_minus_p = translation_matrix(-p[0], -p[1], -p[2])
T_plus_p  = translation_matrix(p[0],  p[1],  p[2])

# --- Aqui imprimimos a matriz uma vez ---
theta_exemplo = np.deg2rad(30)  # por exemplo, 30 graus
M_exemplo = T_plus_p @ rotation_z_matrix(theta_exemplo) @ T_minus_p
np.set_printoptions(precision=4, suppress=True)
print("Matriz homogênea do operador afim (rotação de 30° em torno da reta x=2, y=1):\n")
print(M_exemplo)

# -------------------------
# Plot 3D
# -------------------------
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Parte A: Rotação do cubo colorido em torno da reta x=2, y=1')

# posição original tracejada
for face in cube_faces(V0):
    xs, ys, zs = zip(*face)
    ax.plot(xs + (xs[0],), ys + (ys[0],), zs + (zs[0],),
            linestyle='--', color='gray', linewidth=1.0)

# Faces coloridas
colors = ['red', 'blue', 'green', 'yellow', 'orange', 'white']
faces_poly = Poly3DCollection(cube_faces(V0), facecolors=colors, edgecolor='k', linewidths=1.2, alpha=0.9)
ax.add_collection3d(faces_poly)

# Eixo de rotação
zmin, zmax = -2.0, 3.0
ax.plot([2, 2], [1, 1], [zmin, zmax], color='tab:red', linewidth=2.0, label='Eixo de rotação s')
ax.legend(loc='upper right')

# Limites
ax.set_xlim(-2, 4)
ax.set_ylim(-1, 4)
ax.set_zlim(-2, 3)
ax.set_box_aspect([1, 1, 1])

# ------------------------------------
# Função de animação
# ------------------------------------
frames = 240
def update(frame):
    theta = 2*np.pi * (frame / frames)
    Rz = rotation_z_matrix(theta)
    M = T_plus_p @ Rz @ T_minus_p
    Vt = apply_affine(M, V0)
    faces_poly.set_verts(cube_faces(Vt))
    return [faces_poly]

anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=False, repeat=True)
plt.show()
