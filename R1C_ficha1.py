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
# Matrizes de transformação
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

def apply_affine(M, points):
    homog = np.c_[points, np.ones(len(points))]
    return (M @ homog.T).T[:,:3]

# --------------------
# Configuração inicial
# --------------------
V0 = cube_vertices()

# Eixo D (reta D = (-t, 1-t, t) com direção d = (-1, -1, 1))
t = 0
p0 = np.array([-t, 1-t, t])
d = np.array([-1,-1,1])
d_unit = d/np.linalg.norm(d)

# --------------------
# Imprimir matriz homogênea uma vez
# --------------------
theta_exemplo = np.deg2rad(30)  # exemplo: rotação de 30 graus
R = rotation_matrix(d, theta_exemplo)
T_to = translation_matrix(*p0)
T_from = translation_matrix(*-p0)
T_along = translation_matrix(*( (2/np.pi) * d_unit ))  # fator de translação ao longo do eixo

M_exemplo = T_along @ T_to @ R @ T_from

np.set_printoptions(precision=4, suppress=True)
print("Matriz homogênea do operador afim (rotação em torno de D + translação ao longo do eixo):\n")
print(M_exemplo)

# -------------------------
# Plot 3D
# -------------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Parte C: Rotação em torno de D e Translação ao longo do eixo")

DEFAULT_ELEV, DEFAULT_AZIM = 25, 45
ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)

# Cubo original (pontilhado)
for face in cube_faces(V0):
    xs, ys, zs = zip(*face)
    ax.plot(xs+(xs[0],), ys+(ys[0],), zs+(zs[0],),
            linestyle='--', color='gray', linewidth=1.0, zorder=0.5)

# Cubo colorido
colors = ['red','blue','green','yellow','orange','white']
faces_poly = Poly3DCollection(cube_faces(V0),
                              facecolors=colors,
                              edgecolor='k',
                              linewidths=1.2,
                              alpha=0.92,
                              zorder=3)
ax.add_collection3d(faces_poly)

# Eixo D
line_dir = d_unit * 6
axis_line, = ax.plot([p0[0]-line_dir[0], p0[0]+line_dir[0]],
                     [p0[1]-line_dir[1], p0[1]+line_dir[1]],
                     [p0[2]-line_dir[2], p0[2]+line_dir[2]],
                     color='red', linewidth=2.0, linestyle='--',
                     alpha=0.6, zorder=1, label='Eixo D')
ax.legend(loc='upper right')

ax.set_xlim(-4,4); ax.set_ylim(-4,4); ax.set_zlim(-4,4)
ax.set_box_aspect([1,1,1])

# ---------------------------
# Animação
# ---------------------------
frames = 240
def update(frame):
    theta = 2*np.pi * (frame/frames)
    R = rotation_matrix(d, theta)
    T_to = translation_matrix(*p0)
    T_from = translation_matrix(*-p0)
    T_along = translation_matrix(*( (2/np.pi) * d_unit * (frame/frames) ))
    M = T_along @ T_to @ R @ T_from

    Vt = apply_affine(M, V0)
    faces_poly.set_verts(cube_faces(Vt))
    return [faces_poly, axis_line]

anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=False, repeat=True)

# Botões de perspectiva
ax_button_padrao = plt.axes([0.10, 0.01, 0.18, 0.06])
ax_button_topo   = plt.axes([0.32, 0.01, 0.18, 0.06])
ax_button_lateral= plt.axes([0.54, 0.01, 0.18, 0.06])

btn_padrao  = Button(ax_button_padrao, 'Padrão')
btn_topo    = Button(ax_button_topo,   'Topo')
btn_lateral = Button(ax_button_lateral,'Lateral')

def set_padrao(event):
    ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM); plt.draw()
def set_topo(event):
    ax.view_init(elev=90, azim=0); plt.draw()
def set_lateral(event):
    ax.view_init(elev=0, azim=0); plt.draw()

btn_padrao.on_clicked(set_padrao)
btn_topo.on_clicked(set_topo)
btn_lateral.on_clicked(set_lateral)

plt.show()
