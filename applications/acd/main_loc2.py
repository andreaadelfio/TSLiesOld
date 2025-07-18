import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.colors import LogNorm

# Funzione per calcolare angolo tra due vettori unitari
def angle_between(u, v):
    dot = np.dot(u, v)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

# Definizione delle facce e dei segnali
faces = {
    'Z':  [0, 0, 1],
    'Yp': [0, 1, 0],
    'Xm': [1, 0, 0]
}

signals = {
    'Z': 5.508827,
    'Yp': 7.705379,
    'Xm': 5.002954
}

# Grid nel cielo (theta, phi)
theta_vals = np.radians(np.linspace(0.1, 179.9, 180))
phi_vals = np.radians(np.linspace(0, 360, 360))
THETA, PHI = np.meshgrid(theta_vals, phi_vals)

D = np.full_like(THETA, np.nan)

for i in range(THETA.shape[0]):
    for j in range(THETA.shape[1]):
        theta = THETA[i, j]
        phi = PHI[i, j]
        # vettore unitario per questa direzione
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        v = np.array([x, y, z])

        # correzione dei segnali
        S_corr = {}
        valid = True
        for key, face_vec in faces.items():
            angle = angle_between(v, face_vec)
            cos_angle = np.cos(angle)
            if np.isclose(cos_angle, 0):
                valid = False
                break
            S_corr[key] = signals[key] / cos_angle

        if valid:
            # funzione da minimizzare
            d = (S_corr['Z'] - S_corr['Yp'])**2 + \
                (S_corr['Xm'] - S_corr['Z'])**2 + \
                (S_corr['Yp'] - S_corr['Xm'])**2
            D[i, j] = d

# Trova minimo
min_idx = np.unravel_index(np.nanargmin(D), D.shape)
best_theta = THETA[min_idx]
best_phi = PHI[min_idx]

print(f"Miglior direzione stimata: theta = {np.degrees(best_theta):.2f}°, phi = {np.degrees(best_phi):.2f}°")

# Maschera valori troppo grandi
D_masked = np.ma.masked_where(D > 1e13, D)

# Plot della funzione D (con scala logaritmica)
plt.figure(figsize=(10, 6))
p = plt.pcolormesh(np.degrees(PHI), np.degrees(THETA), D_masked, shading='auto', cmap='viridis', norm=LogNorm(vmin=np.nanmin(D_masked), vmax=np.nanmax(D_masked)))
plt.colorbar(p, label='D')
plt.scatter(np.degrees(best_phi), np.degrees(best_theta), color='red', label='Minimo D')
plt.xlabel('phi (deg)')
plt.ylabel('theta (deg)')
plt.title('Funzione D nel cielo (theta-phi, LogNorm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
