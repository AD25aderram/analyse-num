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

# Calcul des compartiments SIRD
df['R'] = df['Cas rétablis']
df['D'] = df['Cas décédés']
df['I'] = df['Cas positifs'] - df['R'] - df['D']
df['S'] = N - df['I'] - df['R'] - df['D']

# Garder uniquement les jours où les cas positifs croissent
df_sorted = df[df['Cas positifs'].diff().fillna(1) > 0].copy().reset_index(drop=True)

# Filtrer à partir du 24 mars 2020
df_sorted = df_sorted[df_sorted['Date'] >= pd.to_datetime("2020-03-24")].copy().reset_index(drop=True)

# Prendre les 70 premiers jours après cette date
df_model = df_sorted.head(70).copy()
df_model['t'] = range(len(df_model))  # temps

# -----------------------------
# 2. Paramètres du modèle
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

# Conditions initiales réalistes
S[0] = N - 170 - 6 - 5
I[0] = 170
R[0] = 6
D[0] = 5

# -----------------------------
# 3. Résolution par la méthode d’Euler
# -----------------------------
for k in range(T - 1):
    dS = -r * S[k] * I[k] / N
    dI = r * S[k] * I[k] / N - (a + b) * I[k]
    dR = a * I[k]
    dD = b * I[k]

    S[k+1] = S[k] + dS * dt
    I[k+1] = I[k] + dI * dt
    R[k+1] = R[k] + dR * dt
    D[k+1] = D[k] + dD * dt

# -----------------------------
# 4. Export CSV
# -----------------------------
df_sird = pd.DataFrame({
    'Jour': df_model['Date'].dt.strftime('%Y-%m-%d'),
    'S_simule': S,
    'I_simule': I,
    'R_simule': R,
    'D_simule': D
})
df_sird.to_csv("sird_simulation_euler_explicite.csv", index=False)

# -----------------------------
# 5. Comparaison graphique
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
plt.plot(dates, I, label='Infectés simulés', linestyle='--', color='orange')
plt.xticks(rotation=45)
plt.title("Infectés : réel vs simulé")
plt.legend()
plt.grid(True)

# Rétablis
plt.subplot(2, 2, 2)
plt.plot(dates, R_reel, label='Guéris réels', color='green')
plt.plot(dates, R, label='Guéris simulés', linestyle='--', color='lime')
plt.xticks(rotation=45)
plt.title("Guéris : réel vs simulé")
plt.legend()
plt.grid(True)

# Décès
plt.subplot(2, 2, 3)
plt.plot(dates, D_reel, label='Décès réels', color='black')
plt.plot(dates, D, label='Décès simulés', linestyle='--', color='gray')
plt.xticks(rotation=45)
plt.title("Décès : réel vs simulé")
plt.legend()
plt.grid(True)

# Sains
plt.subplot(2, 2, 4)
plt.plot(dates, S_reel, label='Sains réels', color='blue')
plt.plot(dates, S, label='Sains simulés', linestyle='--', color='skyblue')
plt.xticks(rotation=45)
plt.title("Sains : réel vs simulé")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
