import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simuliamo un intervallo temporale
np.random.seed(42)
T = 300  # Numero di istanti di tempo
time = np.arange(T)

# Simuliamo il background noto (mu_b) e la sua deviazione (sigma_b)
mu_b = 5 + np.sin(time / 30)  # Background atteso
sigma_b = 1 + 0.2 * np.cos(time / 30)  # Deviazione standard

# Simuliamo i dati osservati
X = np.random.normal(mu_b, sigma_b)

# Introduciamo un segnale anomalo tra t = 120 e t = 140
X[120:140] += 3  # Eccesso di segnale

# Standardizziamo i dati: X' = (X - mu_b) / sigma_b
X_std = (X - mu_b) / sigma_b

# Intervallo anomalo (individuato con soglia 5σ nel dataset normalizzato)
t_start, t_end = 120, 140
N = t_end - t_start

# Calcoliamo il segnale medio standardizzato e il suo Z-score
S_std = np.mean(X_std[t_start:t_end])
Z = S_std * np.sqrt(N)

# Visualizzazione
plt.figure(figsize=(10, 5))
plt.plot(time, X_std, label="Dati standardizzati", color="black", alpha=0.7)
plt.axhline(0, linestyle="dashed", color="red", label="Background atteso")
plt.axhline(5, linestyle="dotted", color="blue", label="Soglia 5σ")
plt.fill_between(time, -1, 1, color="red", alpha=0.2, label="±1σ background")
plt.axvspan(t_start, t_end, color="yellow", alpha=0.3, label="Intervallo di test")

plt.xlabel("Tempo")
plt.ylabel("Valore standardizzato (unità di sigma)")
plt.legend()
plt.title(f"Z-score: {Z:.2f}")
plt.show()

# Risultati
print(f"Intervallo analizzato: {t_start}-{t_end}")
print(f"Segnale medio standardizzato: {S_std:.2f}")
print(f"Z-score: {Z:.2f}")

# Test di significatività
p_value = 1 - norm.cdf(Z)
print(f"P-value: {p_value:.3e}")
if Z > 5:
    print("Segnale altamente significativo (>5σ)")
elif Z > 3:
    print("Segnale moderatamente significativo (3-5σ)")
else:
    print("Nessuna evidenza di segnale significativo (<3σ)")
