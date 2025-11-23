import numpy as np



FILEPATH = 'data_cov19_ma_2.csv'



def read_csv_column(val_col):
    return np.loadtxt(FILEPATH, delimiter=',', skiprows=1, usecols=val_col)

def taux_SIRD():
    dI = np.diff(I)
    dR = np.diff(R)
    dD = np.diff(D)

    I_mid = (I[:-1] + I[1:]) / 2
    S_mid = (S[:-1] + S[1:]) / 2

    a_values = []
    b_values = []
    r_values = []

    for i in range(len(I_mid)):
        if I_mid[i] != 0:
            a_values.append(dR[i] / I_mid[i])
            b_values.append(dD[i] / I_mid[i])

    a = np.mean(np.array(a_values))
    b = np.mean(np.array(b_values))

    for i in range(len(I_mid)):
        if I_mid[i] != 0 and S_mid[i] != 0:
            r_values.append((dI[i] + (a+b) * I_mid[i]) / (S_mid[i] * I_mid[i]))

    r = np.mean(np.array(r_values))
    
    return a, b, r

def one_by_one(data, thresh, mode):
    match mode:
        case 0:
            for i in range(len(data)):
                if data[i] >= thresh:
                    return i
        case 1:
            for i in range(len(data)):
                if data[i] <= thresh:
                    return i
    return None



if __name__ == "__main__" :

            ###########################
            # PREPARATION DES DONNEES #
            ###########################

    N = 36580000
    t = read_csv_column(0)

    R = read_csv_column(4) / N
    D = read_csv_column(3) / N
    I = read_csv_column(2) / N - (D + R)
    S = 1 - (I + R + D)

    days = np.genfromtxt(FILEPATH, delimiter=',', skip_header=1, usecols=1, dtype=str)

    a, b, r = taux_SIRD()
    print(f"taux de guerison  : a = {a:.6f}")
    print(f"taux de mortalite : b = {b:.6f}")
    print(f"taux de contagion : r = {r:.6f}\n")



            ################################
            ### RESOLUTION DES EQUATIONS ###
            ################################

    # r*S(t)*I(t)-(a+b)*I(t) = 0
    Sc = (a + b) / r
    print(f"Le pic correspond a la valeur de S(t) = {100*Sc:.2f}%")
    t_pic = one_by_one(S, Sc, 1)
    if t_pic == None: print("Le pic n'as pas ete etteint pendant la duree d'etude.\n")
    else: print(f"Le pic a ete atteint le : {days[t_pic]}\n")

    # Pc = 1 - Sc
    print(f"La proportion minimale de la population qui doit etre immunisee : Pc = {100*(1 - Sc):.2f}%\n")

    # I(t) = Imax
    I_max = 40100/N
    t_sat = one_by_one(I, I_max, 0)
    if t_sat == None: print("Aucune intervention sanitaire necessaire pendant la duree d'etude.\n")
    else: print(f"Le temps critique pour une intervention sanitaire est : {days[t_sat]}\n")

    # R0 = r / (a + b)
    print(f"Le nombre de reproduction de base : R0 = {r / (a + b):.3f}\n")
