import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# ---------------------------
# Cubo
# ---------------------------
def cube_vertices(size=1.0, origin=(0,0,0)):
    ox, oy, oz = origin
    s = size
    return np.array([
        [ox,oy,oz],[ox+s,oy,oz],[ox+s,oy+s,oz],[ox,oy+s,oz],
        [ox,oy,oz+s],[ox+s,oy,oz+s],[ox+s,oy+s,oz+s],[ox,oy+s,oz+s]
    ])

def cube_faces(V):
    return [[V[0],V[1],V[2],V[3]],
            [V[4],V[5],V[6],V[7]],
            [V[0],V[1],V[5],V[4]],
            [V[2],V[3],V[7],V[6]],
            [V[1],V[2],V[6],V[5]],
            [V[0],V[3],V[7],V[4]]]

# ---------------------------
# Matrizes
# ---------------------------
def translation_matrix(dx,dy,dz):
    M = np.eye(4); M[:3,3] = [dx,dy,dz]; return M

def rotation_matrix(axis,theta):
    axis = axis/np.linalg.norm(axis)
    x,y,z = axis; c,s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c+(1-c)*x*x,   (1-c)*x*y - s*z, (1-c)*x*z + s*y],
        [(1-c)*y*x+s*z, c+(1-c)*y*y,     (1-c)*y*z - s*x],
        [(1-c)*z*x-s*y, (1-c)*z*y+s*x,   c+(1-c)*z*z]
    ])
    M = np.eye(4); M[:3,:3] = R
    return M

def reflection_matrix(p0, n):
    n = n/np.linalg.norm(n)
    R = np.eye(3) - 2*np.outer(n,n)
    M = np.eye(4)
    M[:3,:3] = R
    M[:3,3] = (np.eye(3)-R)@p0
    return M

def apply_affine(M, points):
    homog = np.c_[points, np.ones(len(points))]
    return (M @ homog.T).T[:,:3]

# --------------------
# Configuração inicial
# --------------------
V0 = cube_vertices()

# Plano C
p0 = np.array([0,1,0])
u = np.array([-2,4,-2]); v = np.array([-1,-1,1])
n = np.cross(u,v)

# Eixo D
pD = np.array([0,1,0])
d = np.array([-1,-1,1]); d_unit = d/np.linalg.norm(d)

# -------------------------
# Plot 3D inicial
# -------------------------
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Parte E: Cubo executando A, B e C")

# Cubo colorido
colors = ['red','blue','green','yellow','orange','white']
faces_poly = Poly3DCollection(cube_faces(V0),
                              facecolors=colors, edgecolor='k',
                              alpha=0.9, zorder=2)
ax.add_collection3d(faces_poly)

# Plano C (ciano translúcido)
extent=4
q1 = p0 - u*extent - v*extent
q2 = p0 + u*extent - v*extent
q3 = p0 + u*extent + v*extent
q4 = p0 - u*extent + v*extent
plane_patch = Poly3DCollection([[q1,q2,q3,q4]],
                               alpha=0.25, facecolor='cyan',
                               edgecolor='k', linewidths=0.5, zorder=0)
ax.add_collection3d(plane_patch)

# Eixo D em vermelho tracejado
line_dir = d_unit * 6
axis_line, = ax.plot([pD[0]-line_dir[0], pD[0]+line_dir[0]],
                     [pD[1]-line_dir[1], pD[1]+line_dir[1]],
                     [pD[2]-line_dir[2], pD[2]+line_dir[2]],
                     color='red', linewidth=2.0, linestyle='--',
                     alpha=0.7, zorder=1, label='Eixo D')
ax.legend(loc='upper right')

# Limites
ax.set_xlim(-4,4); ax.set_ylim(-4,4); ax.set_zlim(-4,4)
ax.set_box_aspect([1,1,1])

# ---------------------------
# Animação em 3 fases
# ---------------------------
frames = 300
def update(frame):
    phase = frame // 100   # 0=A, 1=B, 2=C
    t = (frame % 100)/100

    if phase == 0:
        # A: rotação em torno do eixo Z
        theta = 2*np.pi*t
        R = rotation_matrix(np.array([0,0,1]), theta)
        M = R
    elif phase == 1:
        # B: reflexão no plano C
        if t < 0.5:
            M = np.eye(4)
        else:
            M = reflection_matrix(p0,n)
    else:
        # C: rotação em torno de D + translação
        theta = 2*np.pi*t
        R = rotation_matrix(d, theta)
        T_to = translation_matrix(*pD)
        T_from = translation_matrix(*-pD)
        T_along = translation_matrix(*( (2/np.pi)*d_unit*t ))
        M = T_along @ T_to @ R @ T_from

    Vt = apply_affine(M, V0)
    faces_poly.set_verts(cube_faces(Vt))
    return [faces_poly, plane_patch, axis_line]

anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False, repeat=True)
plt.show()
