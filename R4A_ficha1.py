import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =========================
# Utilidades de álgebra e transformações
# =========================
def hat(u):
    ux, uy, uz = u
    return np.array([[0, -uz,  uy],
                     [uz,  0, -ux],
                     [-uy,  ux, 0]], dtype=float)

def rodrigues(u_hat, theta):
    u_hat = u_hat / np.linalg.norm(u_hat)
    I = np.eye(3)
    uuT = np.outer(u_hat, u_hat)
    K = hat(u_hat)
    return np.cos(theta)*I + (1-np.cos(theta))*uuT + np.sin(theta)*K

def make_hom(R=None, t=None):
    M = np.eye(4)
    if R is not None:
        M[:3,:3] = R
    if t is not None:
        M[:3,3] = t
    return M

def reflect_plane_matrix(n, c):
    # Plano: n·x + c = 0
    n = np.array(n, dtype=float)
    nn = np.dot(n, n)
    A = np.eye(3) - 2.0 * (np.outer(n, n) / nn)
    t = -2.0 * c * (n / nn)
    return make_hom(R=A, t=t)

def apply_transform(M, P):
    P_h = np.c_[P, np.ones(len(P))]
    return (M @ P_h.T).T[:, :3]

def T_translate(t):
    return make_hom(R=None, t=np.array(t, dtype=float))

def T_rotate_about_axis(point_on_axis, axis_dir, theta):
    p0 = np.array(point_on_axis, dtype=float)
    u = np.array(axis_dir, dtype=float)
    R = rodrigues(u/np.linalg.norm(u), theta)
    return T_translate(p0) @ make_hom(R=R, t=None) @ T_translate(-p0)

# =========================
# Dados geométricos e planos
# =========================
def fA(P): return -2*P[:,0] + P[:,1] - P[:,2] - 1.0
def fB(P): return P[:,1] + P[:,2] - 1.0

# Plano C (reflexão)
nC = np.array([1,2,3]); cC = -2.0
M_refC = reflect_plane_matrix(nC, cC)

# Eixo D
p0 = np.array([0.0,1.0,0.0])          # ponto no eixo
u = np.array([-1.0,-1.0,1.0])         # direção do eixo
u_hat = u/np.linalg.norm(u)           # direção unitária

# =========================
# Hélice suave (pitch 2 por volta → deslocamento theta/pi ao longo de u_hat)
# =========================
radius = 0.8
turns = 8
theta_vals = np.linspace(0, 2*np.pi*turns, 6000)

def helix_points(theta_vals):
    base_dir = np.array([radius,0,0])
    pts = []
    for theta in theta_vals:
        R = rodrigues(u_hat, theta)
        pos = p0 + R @ base_dir + (theta/np.pi)*u_hat
        pts.append(pos)
    return np.array(pts)

helix = helix_points(theta_vals)

# =========================
# Cabeça orientada e colorida; retorna também o tip (topo)
# =========================
def build_head(center, direction, scale=2.4):
    d = direction/np.linalg.norm(direction)
    tmp = np.array([1,0,0]) if abs(d[0])<0.9 else np.array([0,1,0])
    v1 = np.cross(d, tmp); v1 /= np.linalg.norm(v1)
    v2 = np.cross(d, v1)
    base = [center + 0.11*scale*( v1+v2),
            center + 0.11*scale*( v1-v2),
            center + 0.11*scale*(-v1-v2),
            center + 0.11*scale*(-v1+v2)]
    tip = center + 0.33*scale*d
    faces = [
        [base[0], base[1], tip],
        [base[1], base[2], tip],
        [base[2], base[3], tip],
        [base[3], base[0], tip]
    ]
    colors = ['red','green','blue','yellow']
    return faces, colors, tip

# =========================
# Função para desenhar planos como quadriláteros
# =========================
def plane_mesh_from_eq(a, b, c, d, center, size=3.2):
    n = np.array([a,b,c], dtype=float)
    n_hat = n/np.linalg.norm(n)
    tmp = np.array([1,0,0]) if abs(n_hat[0])<0.9 else np.array([0,1,0])
    v1 = np.cross(n_hat, tmp); v1 /= np.linalg.norm(v1)
    v2 = np.cross(n_hat, v1)
    center = np.array(center, dtype=float)
    t0 = (d - np.dot(n, center)) / np.dot(n, n)
    p_on = center + t0*n
    corners = [
        p_on + size*( v1 + v2),
        p_on + size*( v1 - v2),
        p_on + size*(-v1 - v2),
        p_on + size*(-v1 + v2),
    ]
    return np.array(corners)

# =========================
# Cena: figura, textos e legenda
# =========================
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5,5); ax.set_ylim(-5,5); ax.set_zlim(-5,5)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title("Isneique: espiral em torno de D, com reflexões em C")
ax.text2D(0.03, 0.97,
          "Planos: A (vermelho), B (azul), C (verde)\n"
          "Eixo D em preto; cabeça colorida; trajetória pontilhada da cabeça",
          transform=ax.transAxes, fontsize=10)

# Eixo D
t_demo = np.linspace(-2, 2, 80)
D_line = np.stack([-t_demo, 1.0 - t_demo, t_demo], axis=1)
axis_line, = ax.plot(D_line[:,0], D_line[:,1], D_line[:,2], color='black', lw=1.8)

# Planos A, B, C (ax+by+cz=d)
A_quad = plane_mesh_from_eq(-2, 1, -1, 1, [0,0,0])
B_quad = plane_mesh_from_eq( 0, 1,  1, 1, [0,0,0])
C_quad = plane_mesh_from_eq( 1, 2,  3, 2, p0)
coll_A = Poly3DCollection([A_quad], alpha=0.22, facecolor='red')
coll_B = Poly3DCollection([B_quad], alpha=0.22, facecolor='blue')
coll_C = Poly3DCollection([C_quad], alpha=0.22, facecolor='green')
ax.add_collection3d(coll_A); ax.add_collection3d(coll_B); ax.add_collection3d(coll_C)

# Corpo da serpente (linha fina para suavidade)
body_line, = ax.plot([], [], [], color='purple', lw=1.2)

# Cabeça
head_poly = Poly3DCollection([], alpha=0.92)
ax.add_collection3d(head_poly)

# Trajetória da cabeça (pontilhada, usando o tip)
head_traj, = ax.plot([], [], [], 'o', color='orange', markersize=1.3, alpha=0.28)

# Legenda
legend_elems = [
    Line2D([0], [0], color='black', lw=1.8, label='Eixo D'),
    Patch(facecolor='red', alpha=0.22, label='Plano A'),
    Patch(facecolor='blue', alpha=0.22, label='Plano B'),
    Patch(facecolor='green', alpha=0.22, label='Plano C'),
    Line2D([0], [0], color='purple', lw=1.2, label='Corpo da serpente'),
    Line2D([0], [0], marker='o', color='orange', alpha=0.28, markersize=6,
           linestyle='None', label='Trajetória da cabeça (topo)'),
]
ax.legend(handles=legend_elems, loc='upper right', bbox_to_anchor=(1.12, 1.0))

# =========================
# Reflexão dinâmica e animação
# =========================
helix_current = helix.copy()
head_positions = []

def update(frame):
    global helix_current, head_positions
    window = 120  # trecho longo para corpo suave
    start = frame
    end = frame + window
    if end >= len(helix_current):
        return body_line, head_poly, head_traj

    body = helix_current[start:end]

    # Reflexão se cruzar A ou B
    if frame > 0:
        prev = helix_current[frame-1:frame]
        curr = helix_current[frame:frame+1]
        crossed_A = np.any(fA(prev) * fA(curr) < 0)
        crossed_B = np.any(fB(prev) * fB(curr) < 0)
        if crossed_A or crossed_B:
            helix_current[frame:] = apply_transform(M_refC, helix_current[frame:])
            body = helix_current[start:end]

    # Corpo
    body_line.set_data(body[:,0], body[:,1])
    body_line.set_3d_properties(body[:,2])

    # Cabeça orientada e colors; trilha pelo tip
    direction = body[-1] - body[-2]
    head_faces, colors, tip = build_head(body[-1], direction, scale=2.4)
    head_poly.set_verts(head_faces)
    head_poly.set_facecolor(colors)

    head_positions.append(tip)
    hp = np.array(head_positions)
    head_traj.set_data(hp[:,0], hp[:,1])
    head_traj.set_3d_properties(hp[:,2])

    return body_line, head_poly, head_traj

ani = FuncAnimation(fig, update, frames=len(helix)-120, interval=18, blit=False)

# =========================
# Impressão de matrizes no terminal (como solicitado)
# =========================
print("Matriz de reflexão no plano C (homogênea 4x4):\n", M_refC)

theta_example = np.pi / 6  # 30 graus
M_rot_example = T_rotate_about_axis(p0, u, theta_example)
print("\nMatriz de rotação em torno do eixo D para theta=π/6:\n", M_rot_example)

# Translação ao longo do eixo (pitch = 2 por volta → deslocamento theta/pi ao longo de u_hat)
t_axis_example = (theta_example / np.pi) * u_hat
M_trans_example = T_translate(t_axis_example)
print("\nMatriz de translação ao longo de D para theta=π/6:\n", M_trans_example)

M_helix_step_example = M_trans_example @ M_rot_example
print("\nComposição de um passo helicoidal (Translação ∘ Rotação) para theta=π/6:\n", M_helix_step_example)

plt.show()
