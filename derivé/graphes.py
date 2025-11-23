import pandas as pd
import matplotlib.pyplot as plt

# Load Excel data
df = pd.read_excel("C:/Users/ikhla/Downloads/data-calcul.xlsx")
df['Date'] = pd.to_datetime(df['Date'])

# Columns and titles to plot
columns_to_plot = {
    'dI_dt': "Variation quotidienne de I(t) (dI/dt)",
    'r_t': "Taux de transmission r(t)",
    'rc_t': "Taux de croissance initial rc(t)",
    'dS_dt': "Variation quotidienne de S(t) (dS/dt)",
    'dR_dt': "Variation quotidienne de R(t) (dR/dt)"
}

# Generate plots
for col, title in columns_to_plot.items():
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df[col], label=title, color='blue')
    plt.xlabel('Date')
    plt.ylabel(title)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{col}_plot.png")
    plt.close()