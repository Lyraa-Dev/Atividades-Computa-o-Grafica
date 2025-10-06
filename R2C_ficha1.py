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
# Matrizes de transformação
# ---------------------------
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

def apply_affine(M, points):
    homog = np.c_[points, np.ones(len(points))]
    return (M @ homog.T).T[:,:3]

# ---------------------------
# Reflexão no plano x - y = 1
# ---------------------------
def reflection_plane_xy1():
    n = np.array([1,-1,0],dtype=float)/np.sqrt(2)
    p0 = np.array([1,0,0],dtype=float)
    R = np.eye(3) - 2*np.outer(n,n)
    t = (np.eye(3)-R)@p0
    M = np.eye(4)
    M[:3,:3] = R
    M[:3,3] = t
    return M

# ---------------------------
# Configuração inicial
# ---------------------------
V0 = cube_vertices()

# Reflexão
M_ref = reflection_plane_xy1()

# Rotação de 30° em torno da reta (t,0,-t), direção d=(1,0,-1)
theta_total = np.deg2rad(30)
d = np.array([1,0,-1],dtype=float)

# ---------------------------
# Mostrar matrizes no terminal
# ---------------------------
M_rot = rotation_matrix(d, theta_total)
M_total = M_rot @ M_ref

np.set_printoptions(precision=4, suppress=True)
print("Matriz homogênea da reflexão no plano x-y=1:\n", M_ref)
print("\nMatriz homogênea da rotação de 30° em torno da reta (t,0,-t):\n", M_rot)
print("\nMatriz homogênea composta M = M_rot @ M_ref:\n", M_total)

# ---------------------------
# Animação
# ---------------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Reflexão no plano x-y=1 seguida de rotação em torno da reta (t,0,-t)")

# Cubo original (tracejado)
for face in cube_faces(V0):
    xs, ys, zs = zip(*face)
    ax.plot(xs+(xs[0],), ys+(ys[0],), zs+(zs[0],),
            linestyle='--', color='gray', linewidth=1.0)

# Cubo colorido
colors = ['red','blue','green','yellow','orange','white']
faces_poly = Poly3DCollection(cube_faces(V0), facecolors=colors, edgecolor='k', alpha=0.9)
ax.add_collection3d(faces_poly)

# Eixo de rotação em vermelho
d_unit = d/np.linalg.norm(d)
L = 3
p0 = np.array([0,0,0])
p1 = p0 - L*d_unit
p2 = p0 + L*d_unit
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
        color='red', linewidth=2.0, label='Eixo de rotação')
ax.legend()

ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.set_zlim(-3,3)
ax.set_box_aspect([1,1,1])

frames = 120
def update(frame):
    half = frames//2
    if frame <= half:
        # Fase 1: interpolação até a reflexão
        t = frame/half
        M = (1-t)*np.eye(4) + t*M_ref
    else:
        # Fase 2: reflexão já feita, agora rotação incremental
        t = (frame-half)/(frames-half)
        theta = t*theta_total
        R = rotation_matrix(d, theta)
        M = R @ M_ref
    Vt = apply_affine(M, V0)
    faces_poly.set_verts(cube_faces(Vt))
    return [faces_poly]

anim = FuncAnimation(fig, update, frames=frames+1, interval=100, blit=False, repeat=True)
plt.show()
