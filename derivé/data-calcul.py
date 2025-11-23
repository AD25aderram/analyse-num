import pandas as pd

# data
df = pd.read_csv("data_cov19_ma_2.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

# Constantes
N = 36_580_000            # Population
a = 0.08221             # Recovery rate
b = 0.00272             # Death rate
dt = 1                    # Time step (daily)

df['I'] = df['Cas positifs'] - df['Cas décédés'] - df['Cas rétablis']
df['S'] = N - df['Cas positifs']
df['R'] = df['Cas rétablis']

df['dI_dt'] = df['I'].diff() / dt
df['dS_dt'] = df['S'].diff() / dt
df['dR_dt'] = df['R'].diff() / dt

df['I_frac'] = df['I'] / N
df['S_frac'] = df['S'] / N
df['dI_frac_dt'] = df['I_frac'].diff() / dt

df['r_t'] = (df['dI_frac_dt'] + (a + b) * df['I_frac']) / (df['S_frac'] * df['I_frac'])

df['rc_t'] = df['dI_dt'] / df['I']

final_df = df[['Date', 'Cas positifs', 'Cas décédés', 'Cas rétablis',
               'I', 'S', 'R', 'dI_dt', 'dS_dt', 'dR_dt', 'r_t', 'rc_t']].dropna()

# Export to Excel
final_df.to_excel('data-calcul.xlsx', index=False)