import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------
# Dados iniciais
# ---------------------------
A = np.array([2,-2,-3,1])  # homogêneo
B = np.array([2,1,0,1])
C = np.array([0,-1,-1])

# Vetores
rA = A[:3] - C
rB = B[:3] - C

# Eixo de rotação
n = np.cross(rA, rB)
n = n / np.linalg.norm(n)

# ---------------------------
# Matrizes auxiliares
# ---------------------------
def translation_matrix(dx,dy,dz):
    M = np.eye(4)
    M[:3,3] = [dx,dy,dz]
    return M

def rotation_matrix(axis,theta):
    x,y,z = axis
    c,s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c+(1-c)*x*x,   (1-c)*x*y - s*z, (1-c)*x*z + s*y],
        [(1-c)*y*x+s*z, c+(1-c)*y*y,     (1-c)*y*z - s*x],
        [(1-c)*z*x-s*y, (1-c)*z*y+s*x,   c+(1-c)*z*z]
    ])
    M = np.eye(4)
    M[:3,:3] = R
    return M

def apply_affine(M, p):
    return (M @ p)[:3]

# ---------------------------
# Matriz de rotação de 30°
# ---------------------------
theta = np.pi/6
T_to = translation_matrix(*C)
T_from = translation_matrix(*-C)
R = rotation_matrix(n, theta)
M30 = T_to @ R @ T_from

print("Matriz de um passo de 30°:")
print(M30)

# ---------------------------
# Trajetória (pontos discretos)
# ---------------------------
traj = [A[:3]]
p = A.copy()
for i in range(3):  # 3 passos de 30° = 90°
    p = M30 @ p
    traj.append(p[:3])

print("\nPontos da trajetória (A até B em 30°):")
for i,pt in enumerate(traj):
    print(f"Passo {i}: {pt}")

# ---------------------------
# Construir arco completo (guia da trajetoria)
# ---------------------------
arc_points = []
steps_arc = 90
for k in range(steps_arc+1):
    ang = (np.pi/2) * (k/steps_arc)  # de 0 a 90°
    Rk = rotation_matrix(n, ang)
    Mk = T_to @ Rk @ T_from
    arc_points.append(apply_affine(Mk, A))
arc_points = np.array(arc_points)

# ---------------------------
# Animação
# ---------------------------
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Partícula de A até B em arcos de 30°")

# Configurações do espaço
ax.set_xlim(-1,3); ax.set_ylim(-3,3); ax.set_zlim(-4,2)
ax.set_box_aspect([1,1,1])

# Plot de pontos A, B e C
ax.scatter(*A[:3], color='green', s=80, label='A (início)')
ax.scatter(*B[:3], color='red', s=80, label='B (fim)')
ax.scatter(*C, color='blue', s=80, label='C (centro)')

# Arco guia da trajetória
ax.plot(arc_points[:,0], arc_points[:,1], arc_points[:,2],
        color='gray', linestyle='--', alpha=0.7, label='Arco circular')

ax.legend()

# Partícula animada
particle, = ax.plot([],[],[],'o',color='purple',markersize=10)

# Linha da trajetória percorrida
path, = ax.plot([],[],[],color='purple',alpha=0.8)

# Frames: interpolando entre 3 p até chegar ao ponto B
frames = 90
def update(frame):
    step = frame // 30
    t = (frame % 30)/30
    if step < 3:
        # interpolação entre traj[step] e traj[step+1]
        p = (1-t)*traj[step] + t*traj[step+1]
    else:
        p = traj[-1]
    particle.set_data([p[0]],[p[1]])
    particle.set_3d_properties([p[2]])
    # atualizar linha
    path.set_data([pt[0] for pt in traj[:step+1]],
                  [pt[1] for pt in traj[:step+1]])
    path.set_3d_properties([pt[2] for pt in traj[:step+1]])
    return particle, path

ani = FuncAnimation(fig, update, frames=frames+1, interval=200, blit=False, repeat=True)
plt.show()
