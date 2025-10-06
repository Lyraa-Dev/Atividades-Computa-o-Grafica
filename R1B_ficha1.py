import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

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
# Reflexão em relação ao plano
# ---------------------------
def reflection_operator(p0, u, v):
    # normal unitária
    n = np.cross(u, v)
    n = n / np.linalg.norm(n)
    # parte linear
    A = np.eye(3) - 2*np.outer(n, n)
    # translação
    b = 2 * np.dot(p0, n) * n
    return A, b

def apply_affine(A, b, points):
    return (A @ points.T).T + b

# --------------------
# Configuração inicial
# --------------------
V0 = cube_vertices(size=1.0, origin=(0.0,0.0,0.0))

# Plano C
p0 = np.array([0,1,0])
u = np.array([-2,4,-2])
v = np.array([-1,-1,1])

# Operador afim da reflexão
A, b = reflection_operator(p0, u, v)
V_ref = apply_affine(A, b, V0)

# --- Imprimir no terminal ---
np.set_printoptions(precision=4, suppress=True)
print("Operador afim da reflexão em relação ao plano C:")
print("\nParte linear A =\n", A)
print("\nParte de translação b =\n", b)
print("\nTransformação: T(x) = A x + b")

# -------------------------
# Plot 3D
# -------------------------
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Parte B: Reflexão + Botões de Perspectiva")

# Cubo original (pontilhado)
for face in cube_faces(V0):
    xs, ys, zs = zip(*face)
    ax.plot(xs + (xs[0],), ys + (ys[0],), zs + (zs[0],),
            linestyle='--', color='gray', linewidth=1.0)

# Faces coloridas do cubo animado
colors = ['red','blue','green','yellow','orange','white']
faces_poly = Poly3DCollection(cube_faces(V0), facecolors=colors, edgecolor='k', alpha=0.9)
ax.add_collection3d(faces_poly)

# Plano C (patch grande)
extent=4
q1 = p0 - u*extent - v*extent
q2 = p0 + u*extent - v*extent
q3 = p0 + u*extent + v*extent
q4 = p0 - u*extent + v*extent
plane_patch = Poly3DCollection([[q1,q2,q3,q4]], alpha=0.3, facecolor='cyan')
ax.add_collection3d(plane_patch)

# Ajustes
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.set_zlim(-4,4)
ax.set_box_aspect([1,1,1])

# ---------------------------
# Reflexão animada (toggle)
# ---------------------------
def update(frame):
    if frame % 2 == 0:
        Vt = V0
    else:
        Vt = V_ref
    faces_poly.set_verts(cube_faces(Vt))
    return [faces_poly]

anim = FuncAnimation(fig, update, frames=100, interval=3000, blit=False, repeat=True)

# ---------------------------
# Botões de perspectiva
# ---------------------------
from matplotlib.widgets import Button
ax_button1 = plt.axes([0.1, 0.01, 0.15, 0.05])
ax_button2 = plt.axes([0.3, 0.01, 0.15, 0.05])
ax_button3 = plt.axes([0.5, 0.01, 0.15, 0.05])

btn_normal = Button(ax_button1, 'Normal')
btn_top = Button(ax_button2, 'Horizontal')
btn_side = Button(ax_button3, 'Lateral')

def set_normal(event):
    ax.view_init(elev=25, azim=45)
    plt.draw()

def set_top(event):
    ax.view_init(elev=90, azim=0)
    plt.draw()

def set_side(event):
    ax.view_init(elev=0, azim=0)
    plt.draw()

btn_normal.on_clicked(set_normal)
btn_top.on_clicked(set_top)
btn_side.on_clicked(set_side)

plt.show()
