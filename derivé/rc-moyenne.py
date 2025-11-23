import pandas as pd

df = pd.read_excel("C:/Users/ikhla/Downloads/data-calcul.xlsx")

# Supprimer les valeurs NaN dans r(t)
r_values = df['rc_t'].dropna()

# Calcul de la moyenne
moyenne_r = r_values.mean()

# Affichage
print(f"La moyenne de rc_t est : {moyenne_r:.12f}")