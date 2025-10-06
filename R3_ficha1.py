import numpy as np
import matplotlib.pyplot as plt

def normalize(v):
    return v/np.linalg.norm(v)

def rodrigues(axis, theta):
    axis = normalize(axis)
    ux, uy, uz = axis
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c+ux*ux*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
        [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c), uy*uz*(1-c)-ux*s],
        [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)]
    ])

def rotate_point(p, P, axis, theta):
    R = rodrigues(axis, theta)
    return P + R@(p-P)

# --- Eixos ---
Pr = np.array([1,2,0])   # bico inicial
vr = np.array([1,-1,0])  # direção eixo r
Ps = np.array([2,0,1])   # ponto eixo s
vs = np.array([0,1,0])   # direção eixo s

ur = normalize(vr)
us = normalize(vs)

# --- Parâmetros ---
T = 10.0
spin_period = 2.5
omega_r = 2*np.pi/spin_period   # spin
omega_s = 2*np.pi/T             # precessão

# --- Cone simplificado ---
h = 2.0
r_base = 0.6

tip0 = Pr
head0 = Pr + h*ur

# base ortonormal
tmp = np.array([0,0,1]) if abs(ur[2])<0.9 else np.array([1,0,0])
e1 = normalize(np.cross(ur,tmp))
e2 = np.cross(ur,e1)

# círculo da base
theta = np.linspace(0,2*np.pi,30)
base_circle0 = np.array([head0 + r_base*np.cos(t)*e1 + r_base*np.sin(t)*e2 for t in theta])

# ponto lateral (frente)
front0 = head0 + r_base*e1

# --- Animação ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n_frames = 200
dt = T/n_frames

# listas para guardar trajetórias
trail_tip = []
trail_front = []
front_markers = []  # pontos discretos a cada spin completo

for k in range(n_frames+1):
    time = k*dt
    θs = omega_s*time
    θr = omega_r*time

    # precessão
    Rprec = rodrigues(us, θs)
    Pr_t = Ps + Rprec@(Pr-Ps)
    ur_t = Rprec@ur

    # aplica precessão
    tip_p   = Ps + Rprec@(tip0-Ps)
    head_p  = Ps + Rprec@(head0-Ps)
    base_p  = Ps + (Rprec@(base_circle0-Ps).T).T
    front_p = Ps + Rprec@(front0-Ps)

    # aplica spin
    tip   = rotate_point(tip_p, Pr_t, ur_t, θr)
    head  = rotate_point(head_p, Pr_t, ur_t, θr)
    base  = Pr_t + (rodrigues(ur_t, θr)@(base_p-Pr_t).T).T
    front = rotate_point(front_p, Pr_t, ur_t, θr)

    # guarda trajetórias
    trail_tip.append(tip)
    trail_front.append(front)

    # adiciona marcador discreto a cada volta completa (2.5s)
    if abs((time % spin_period) - 0) < dt/2:
        front_markers.append(front)

    # desenhar
    ax.clear()
    # cone: linhas do bico até a base
    for p in base:
        ax.plot([tip[0],p[0]],[tip[1],p[1]],[tip[2],p[2]],color='orange',alpha=0.7)
    # círculo da base
    ax.plot(base[:,0],base[:,1],base[:,2],color='orange')
    # marcadores
    ax.scatter(*tip,color='red',s=50,label="Bico (ponta)")
    ax.scatter(*head,color='blue',s=50,label="Cabeça (topo)")
    ax.scatter(*front,color='black',s=30,label="Frente (marca lateral)")
    # eixo r(t)
    ax.plot([tip[0],head[0]],[tip[1],head[1]],[tip[2],head[2]],color='purple',lw=2,label="Eixo r(t)")
    # eixo s
    sa, sb = Ps-2*us, Ps+2*us
    ax.plot([sa[0],sb[0]],[sa[1],sb[1]],[sa[2],sb[2]],color='green',lw=2,label="Eixo s")
    # trajetórias
    trail_tip_arr = np.array(trail_tip)
    trail_front_arr = np.array(trail_front)
    ax.plot(trail_tip_arr[:,0],trail_tip_arr[:,1],trail_tip_arr[:,2],'r--',label="Trajetória do bico")
    ax.plot(trail_front_arr[:,0],trail_front_arr[:,1],trail_front_arr[:,2],'k--',label="Trajetória da frente")
    # marcadores discretos da frente
    if front_markers:
        fm = np.array(front_markers)
        ax.scatter(fm[:,0],fm[:,1],fm[:,2],color='black',s=40,marker='o',label="Frente (voltas inteiras)")

    ax.set_xlim([-2,5]); ax.set_ylim([-3,5]); ax.set_zlim([-2,4])
    ax.view_init(25,40)
    ax.set_box_aspect([1,1,1])
    ax.set_title(f"t = {time:.2f}s")
    ax.legend(loc='upper left',fontsize=8)
    plt.pause(0.03)

plt.show()
