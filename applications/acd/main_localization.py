import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Funzione per calcolare direzione da vettore cartesiano
def compute_direction(vx, vy, vz):
    norm = math.sqrt(vx*vx + vy*vy + vz*vz)
    if norm == 0:
        raise ValueError("Vettore nullo, impossibile normalizzare")
    ux, uy, uz = vx/norm, vy/norm, vz/norm
    theta = math.acos(uz)                    # angolo da +Z
    phi = math.atan2(uy, ux) % (2*math.pi)   # azimutale [0, 2π)
    return {'unit_vector': (ux, uy, uz),
            'theta': theta,
            'phi': phi,
            'theta_deg': math.degrees(theta),
            'phi_deg': math.degrees(phi)}

# Dati di esempio: vettore direzionale nel sistema del satellite
# v_spacecraft = (0.02159, -0.005419, 0.002737)
v_spacecraft = (5.002954, 7.705379, 5.508827)
# v_spacecraft = (1, 0, 0)
direction = compute_direction(*v_spacecraft)
v_unit = direction['unit_vector']
theta = direction['theta']
phi = direction['phi']

# Coordinate catalogo per confronto (in gradi)
# theta_cat_deg, phi_cat_deg = 164.8130, 39.2921
theta_cat_deg, phi_cat_deg = 0.0474, 258.8200
# theta_cat_deg, phi_cat_deg = 82.4244, 359.7950
theta_cat = np.radians(theta_cat_deg)
phi_cat = np.radians(phi_cat_deg)

# Conversione in coordinate cartesiane unitari su sfera
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

x_cat = np.sin(theta_cat) * np.cos(phi_cat)
y_cat = np.sin(theta_cat) * np.sin(phi_cat)
z_cat = np.cos(theta_cat)

print(f"Direzione stimata: {direction}")
print(f"Direzione catalogo: theta={theta_cat_deg}°, phi={phi_cat_deg}°")

# Plotta la sfera
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Sfera
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_s = np.outer(np.cos(u), np.sin(v))
y_s = np.outer(np.sin(u), np.sin(v))
z_s = np.outer(np.ones_like(u), np.cos(v))
# ax.plot_surface(x_s, y_s, z_s, color='lightgray', alpha=0.3, linewidth=0)

# add x, y, z quivers
ax.quiver(0, 0, 0, 1, 0, 0, color='red')
ax.quiver(0, 0, 0, 0, 1, 0, color='green')
ax.quiver(0, 0, 0, 0, 0, 1, color='blue')

# add x, y, z labels
ax.text(1.1, 0, 0, 'X', color='red', fontsize=12)
ax.text(0, 1.1, 0, 'Y', color='green', fontsize=12)
ax.text(0, 0, 1.1, 'Z', color='blue', fontsize=12)

# Vettori
ax.quiver(0, 0, 0, x, y, z, color='black', label='Estimated ')
ax.quiver(0, 0, 0, x_cat, y_cat, z_cat, color='magenta', label='Catalog')

# quivers of the components
ax.quiver(0, 0, 0, v_unit[0], 0, 0, color='black')
ax.quiver(0, 0, 0, 0, v_unit[1], 0, color='black')
ax.quiver(0, 0, 0, 0, 0, v_unit[2], color='black')


ax.quiver(0, 0, 0, v_unit[0], v_unit[1], 0, color='black', ls='dotted')

# dotted lines to underline the components corresponding to the components in the xy plane and the z axis
ax.plot([v_unit[0], v_unit[0]], [0, v_unit[1]], [0, 0], color='black', linestyle='dotted', lw=3)
ax.plot([0, v_unit[0]], [v_unit[1], v_unit[1]], [0, 0], color='black', linestyle='dotted', lw=3)
ax.plot([0, v_unit[0]], [0, v_unit[1]], [v_unit[2], v_unit[2]], color='black', linestyle='dotted', lw=3)
ax.plot([v_unit[0], v_unit[0]], [v_unit[1], v_unit[1]], [0, v_unit[2]], color='black', linestyle='dotted', lw=3)

# add text of the theta angle between the component of z and the vector
ax.text(v_unit[0]*0.5, v_unit[1]*0.5, v_unit[2],
        f'θ = {direction["theta_deg"]:.2f}°',
        color='green', fontsize=12)


# add text of the phi angle between the component of z and the vector
ax.text(v_unit[0]*2, v_unit[1]*0.5, 0,
        f'φ = {direction["phi_deg"]:.2f}°',
        color='green', fontsize=12)


# Impostazioni grafiche
ax.set_box_aspect([1,1,1])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# hide axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(False)

plt.tight_layout()
plt.show()
