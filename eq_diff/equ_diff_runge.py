import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Chargement et préparation des données
# -----------------------------
df = pd.read_csv("data_cov19_ma_2.csv")
N = 36_580_000

# Conversion du format de date
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

# Compartiments
df['R'] = df['Cas rétablis']
df['D'] = df['Cas décédés']
df['I'] = df['Cas positifs'] - df['R'] - df['D']
df['S'] = N - df['I'] - df['R'] - df['D']

# Garder uniquement les jours où les cas positifs croissent
df_sorted = df[df['Cas positifs'].diff().fillna(1) > 0].copy().reset_index(drop=True)

# Filtrer à partir du 24 mars 2020
df_sorted = df_sorted[df_sorted['Date'] >= pd.to_datetime("2020-03-24")].copy().reset_index(drop=True)

# Prendre 70 premiers jours après le 24 mars
df_model = df_sorted.head(70).copy()
df_model['t'] = range(len(df_model))

# -----------------------------
# 2. Paramètres du modèle SIRD
# -----------------------------
a = 0.08221
b = 0.00272
r = 0.11023

T = len(df_model)
dt = 1.0
t = np.arange(T)

# Initialisation
S = np.zeros(T)
I = np.zeros(T)
R = np.zeros(T)
D = np.zeros(T)

S[0] = N - 170 - 6 - 5
I[0] = 170
R[0] = 6
D[0] = 5

# -----------------------------
# 3. Résolution par Runge-Kutta d'ordre 4
# -----------------------------
for k in range(T - 1):
    # k1
    dS1 = -r * S[k] * I[k] / N
    dI1 = r * S[k] * I[k] / N - (a + b) * I[k]
    dR1 = a * I[k]
    dD1 = b * I[k]

    # k2
    S2 = S[k] + dt * dS1 / 2
    I2 = I[k] + dt * dI1 / 2
    dS2 = -r * S2 * I2 / N
    dI2 = r * S2 * I2 / N - (a + b) * I2
    dR2 = a * I2
    dD2 = b * I2

    # k3
    S3 = S[k] + dt * dS2 / 2
    I3 = I[k] + dt * dI2 / 2
    dS3 = -r * S3 * I3 / N
    dI3 = r * S3 * I3 / N - (a + b) * I3
    dR3 = a * I3
    dD3 = b * I3

    # k4
    S4 = S[k] + dt * dS3
    I4 = I[k] + dt * dI3
    dS4 = -r * S4 * I4 / N
    dI4 = r * S4 * I4 / N - (a + b) * I4
    dR4 = a * I4
    dD4 = b * I4

    # Mise à jour
    S[k+1] = S[k] + (dt / 6) * (dS1 + 2*dS2 + 2*dS3 + dS4)
    I[k+1] = I[k] + (dt / 6) * (dI1 + 2*dI2 + 2*dI3 + dI4)
    R[k+1] = R[k] + (dt / 6) * (dR1 + 2*dR2 + 2*dR3 + dR4)
    D[k+1] = D[k] + (dt / 6) * (dD1 + 2*dD2 + 2*dD3 + dD4)

# -----------------------------
# 4. Export CSV
# -----------------------------
df_sird_rk = pd.DataFrame({
    'Jour': df_model['Date'].dt.strftime('%Y-%m-%d'),
    'S_simule': S,
    'I_simule': I,
    'R_simule': R,
    'D_simule': D
})
df_sird_rk.to_csv("sird_simulation_rk4.csv", index=False)

# -----------------------------
# 5. Graphiques comparatifs
# -----------------------------
I_reel = df_model['I'].values
R_reel = df_model['R'].values
D_reel = df_model['D'].values
S_reel = df_model['S'].values
dates = df_model['Date'].dt.strftime('%Y-%m-%d').values

plt.figure(figsize=(14, 8))

# Infectés
plt.subplot(2, 2, 1)
plt.plot(dates, I_reel, label='Infectés réels', color='red')
plt.plot(dates, I, label='Infectés simulés RK4', linestyle='--', color='orange')
plt.xticks(rotation=45)
plt.title("Infectés : réel vs RK4")
plt.legend()
plt.grid(True)

# Guéris
plt.subplot(2, 2, 2)
plt.plot(dates, R_reel, label='Guéris réels', color='green')
plt.plot(dates, R, label='Guéris simulés RK4', linestyle='--', color='lime')
plt.xticks(rotation=45)
plt.title("Guéris : réel vs RK4")
plt.legend()
plt.grid(True)

# Décès
plt.subplot(2, 2, 3)
plt.plot(dates, D_reel, label='Décès réels', color='black')
plt.plot(dates, D, label='Décès simulés RK4', linestyle='--', color='gray')
plt.xticks(rotation=45)
plt.title("Décès : réel vs RK4")
plt.legend()
plt.grid(True)

# Sains
plt.subplot(2, 2, 4)
plt.plot(dates, S_reel, label='Sains réels', color='blue')
plt.plot(dates, S, label='Sains simulés RK4', linestyle='--', color='skyblue')
plt.xticks(rotation=45)
plt.title("Sains : réel vs RK4")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
