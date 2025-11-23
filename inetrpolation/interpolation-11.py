#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import (
    BarycentricInterpolator,
    CubicSpline,
    PchipInterpolator,
    CubicHermiteSpline
)
from numpy.polynomial import Polynomial
from datetime import timedelta

FILEPATH = 'data_cov19_ma_2.csv'
REPORT_FILE = 'interpolation_report.txt'
PLOT_DIR = 'plots'
CSV_DIR = 'comparisons'
GRAPH_PRECISION = 500

def read_data(path=FILEPATH):
    df = pd.read_csv(path)
    date_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    t = df.iloc[:, 0].to_numpy(dtype=float)
    dates = df[date_col]
    C = df.iloc[:, 2].to_numpy(dtype=float)
    D = df.iloc[:, 3].to_numpy(dtype=float)
    R = df.iloc[:, 4].to_numpy(dtype=float)
    I = C - (D + R)
    return t, dates, {'Infectés': I, 'Rétablis': R, 'Décès': D}

def train_test_split_strict(n, train_idx, n_test):
    available = np.setdiff1d(np.arange(n), train_idx)
    test_idx = np.random.choice(available, size=n_test, replace=False)
    return train_idx, test_idx

def chebyshev_nodes(x, m):
    a, b = x.min(), x.max()
    k = np.arange(m)
    return 0.5*(a+b) + 0.5*(b-a)*np.cos((2*k+1)/(2*m)*np.pi)

def prepare_methods(x, y, n_nodes, mode):
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    nodes = (np.linspace(x_sorted.min(), x_sorted.max(), n_nodes)
             if mode=='1'
             else chebyshev_nodes(x_sorted, n_nodes))
    nodes = np.sort(nodes)
    y_nodes = np.interp(nodes, x_sorted, y_sorted)
    lag_interp = BarycentricInterpolator(nodes, y_nodes)
    poly_coef = np.polyfit(nodes, y_nodes, deg=len(nodes)-1)
    lag_poly = Polynomial(poly_coef[::-1])
    spline = CubicSpline(nodes, y_nodes)
    pchip = PchipInterpolator(x_sorted, y_sorted)
    slopes = np.gradient(y_nodes, nodes)
    hermite = CubicHermiteSpline(nodes, y_nodes, slopes)
    methods = {
        'Lagrange': lag_interp,
        'Spline': spline,
        'PCHIP': pchip,
        'Hermite': hermite
    }
    eqs = {
        'Lagrange': lag_poly,
        'Spline': spline,
        'PCHIP': None,
        'Hermite': hermite
    }
    nodesets = {
        'Lagrange': nodes,
        'Spline': nodes,
        'Hermite': nodes,
        'PCHIP': x_sorted
    }
    ysets = {
        'Lagrange': y_nodes,
        'Spline': y_nodes,
        'Hermite': y_nodes,
        'PCHIP': y_sorted
    }
    return methods, eqs, nodesets, ysets

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    mode = ''
    while mode not in ('1','2'):
        mode = input('Choix nœuds (1=uniforme,2=chebyshev): ')
    label = 'uniforme' if mode=='1' else 'chebyshev'

    t, dates, series = read_data()

    with open(REPORT_FILE,'w') as rep:
        rep.write('Interpolation Report\n' + '='*80 + '\n')
        for lab, y in series.items():
            rep.write(f'Serie: {lab} (nœuds {label})\n' + '-'*40 + '\n')

            # Nombre de nœuds et points de test
            n_nodes = 0
            while n_nodes < 2:
                try:
                    n_nodes = int(input(f'Nombre de noeuds pour {lab}: '))
                except:
                    continue
            max_t = len(t) - n_nodes
            n_test = 0
            while not (1 <= n_test <= max_t):
                try:
                    n_test = int(input(f'Nombre de points test (1 à {max_t}): '))
                except:
                    continue

            methods, eqs, nodesets, ysets = prepare_methods(t, y, n_nodes, mode)

            sel = []
            for m in methods:
                c = ''
                while c not in ('1','0'):
                    c = input(f'Inclure {m}? (1=oui,0=non): ')
                if c == '1':
                    sel.append(m)
            rep.write('Méthodes sélectionnées: ' + ','.join(sel) + '\n')

            train_idx = np.unique([np.abs(t - n).argmin()
                                   for n in nodesets['Lagrange']])
            _, test_idx = train_test_split_strict(len(t), train_idx, n_test)
            x_test, y_test = t[test_idx], y[test_idx]
            rep.write(f'Points test demandés: {n_test}\n')

            # Équations & erreurs
            rep.write('Équations et erreurs (sans PCHIP):\n')
            for m in sel:
                if m == 'PCHIP': continue
                fn = methods[m]
                yp = fn(x_test)
                err = np.max(np.abs(yp - y_test))
                if m == 'Lagrange':
                    poly = eqs[m]
                    eqstr = ' + '.join(
                        f'({coef:.6g})*x^{i}'
                        if i > 0 else f'{coef:.6g}'
                        for i, coef in enumerate(poly.coef)
                    )
                else:
                    knots = fn.x
                    coeffs = fn.c.T
                    eqstr = ' | '.join(
                        f'[{knots[i]:.2f},{knots[i+1]:.2f}]:'
                        f'{a:.6g}+{b:.6g}(x-{knots[i]:.2f})'
                        f'+{c:.6g}(x-{knots[i]:.2f})^2'
                        f'+{d:.6g}(x-{knots[i]:.2f})^3'
                        for i, (a, b, c, d) in enumerate(coeffs)
                    )
                rep.write(f'{m}: {eqstr} | MaxErr={err:.2f}\n')

            rep.write('Nœuds par méthode:\n')
            for m in sel:
                rep.write(f'{m}: ' +
                          ','.join(f'{ni:.2f}' for ni in nodesets[m]) +
                          '\n')

            rep.write('Points test:\n')
            for xt, yt in zip(x_test, y_test):
                d = dates[np.abs(t - xt).argmin()].date()
                rep.write(f' t={xt:.2f},date={d},val={yt:.2f}\n')

            rep.write('Methode|RMS|MaxErr\n' + '-'*40 + '\n')
            for m in sel:
                yp = methods[m](x_test)
                rms = np.sqrt(np.mean((yp - y_test)**2))
                err = np.max(np.abs(yp - y_test))
                rep.write(f'{m}|{rms:.2f}|{err:.2f}\n')

            # Comparaison CSV
            comp = pd.DataFrame({
                't': x_test,
                'date': [dates[np.abs(t - xt).argmin()].date() for xt in x_test],
                'orig': y_test
            })
            for m in sel:
                comp[m] = methods[m](x_test)
            csv_path = os.path.join(CSV_DIR, f'cmp_{lab}.csv')
            comp.to_csv(csv_path, index=False)
            rep.write(f'CSV:{csv_path}\n')

            # Tracé interpolation
            fig, ax = plt.subplots(figsize=(10,4))
            ax.scatter(dates, y, marker='o', color='silver', alpha=0.5,
                       label='Données')
            ax.scatter([dates[np.abs(t - xt).argmin()]
                        for xt in x_test], y_test,
                       marker='D', color='black', label='Tests')
            colors = {'Lagrange':'red','Spline':'blue',
                      'PCHIP':'green','Hermite':'purple'}
            xg = np.linspace(t.min(), t.max(), GRAPH_PRECISION)
            dg = mdates.num2date(
                np.interp(xg, t, mdates.date2num(dates))
            )
            for m in sel:
                if m == 'PCHIP':
                    idx = np.argsort(t)
                    ax.plot(dates[idx], methods[m](t[idx]),
                            color=colors[m], label=m)
                else:
                    ax.plot(dg, methods[m](xg),
                            color=colors[m], label=m)
                    node_dates = [
                        dates[np.abs(t - xi).argmin()]
                        for xi in nodesets[m]
                    ]
                    node_values = ysets[m]
                    ax.scatter(node_dates, node_values,
                               edgecolors=colors[m],
                               facecolors='none', marker='o')
            ax.set_title(f'{lab} (tests: {n_test})')
            ax.set_ylim(y.min(), y.max())
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
            ax.legend()
            plot_path = os.path.join(PLOT_DIR, f'{lab}.png')
            fig.savefig(plot_path)
            plt.close(fig)
            rep.write(f'PNG:{plot_path}\n\n')

            # ————— AJOUT : Extrapolation hors intervalle CSV unifié avec alertes —————
            t_min, t_max = t.min(), t.max()
            horizons = {
                '1_sem_avant':  -7,
                '1_mois_avant':  -30,
                '3_mois_avant':  -90,
                '6_mois_avant': -180,
                '1_an_avant':   -365,
                '2_ans_avant':  -730,
                '1_sem_apres':    7,
                '1_mois_apres':   30,
                '3_mois_apres':   90,
                '6_mois_apres':  180,
                '1_an_apres':    365,
                '2_ans_apres':   730
            }
            rows = []
            for m in sel:
                fn = methods[m]
                for label_h, delta in horizons.items():
                    if 'avant' in label_h:
                        t_h = t_min + delta
                        ref_idx = np.abs(t - t_min).argmin()
                    else:
                        t_h = t_max + delta
                        ref_idx = np.abs(t - t_max).argmin()

                    y_h = fn(t_h)
                    alerte = ''
                    if np.isnan(y_h):
                        alerte = 'NaN'
                    elif np.isinf(y_h):
                        alerte = 'Infini'
                    elif y_h < 0:
                        alerte = 'Négatif'

                    date_ref = dates.iloc[ref_idx]
                    date_h = date_ref + timedelta(days=delta)

                    rows.append({
                        'Série':   lab,
                        'Méthode': m,
                        'Horizon': label_h,
                        't_ex':    t_h,
                        'date_ex': date_h.date(),
                        'y_ex':    float(y_h),
                        'Alerte':  alerte
                    })

            df_ex_all = pd.DataFrame(rows, columns=[
                'Série','Méthode','Horizon','t_ex','date_ex','y_ex','Alerte'
            ])
            csv_ex_all = os.path.join(CSV_DIR, f'extrapolation_tous_{lab}.csv')
            df_ex_all.to_csv(csv_ex_all, index=False)
            rep.write(f'CSV_extrapolation_unifie:{csv_ex_all}\n')
            # —————————————————————————————————————————————————————————————————

        print(f'Report:{REPORT_FILE}, Plots:{PLOT_DIR}, CSV:{CSV_DIR}')

if __name__=='__main__':
    main()
