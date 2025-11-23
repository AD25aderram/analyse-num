import pandas as pd
import matplotlib.pyplot as plt

# Load file
df = pd.read_excel("C:/Users/ikhla/Downloads/data-calcul.xlsx")
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values('Date')

early_phase = df.head(60)

plt.figure(figsize=(10, 5))
plt.plot(early_phase['Date'], early_phase['rc_t'], marker='o', color='darkred', label='rc(t) - Early Phase')
plt.title("Taux de croissance initial rc(t) — Début de la pandémie (2 premiers mois)")
plt.xlabel("Date")
plt.ylabel("rc(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the graph
plt.savefig("rc-premiers jours")
plt.show()
