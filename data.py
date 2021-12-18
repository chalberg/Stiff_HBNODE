import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def rober(t, y):
    y1, y2, y3 = y
    xdot = -0.04 * y1 + 1.e4 * y2 * y3
    ydot = 0.04 * y1 - 1.e4 * y2 * y3 - 3.e7 * y2 ** 2
    zdot = 3.e7 * y2 ** 2
    return xdot, ydot, zdot


def pollu(t, y):
    k1 = .35e0
    k2 = .266e2
    k3 = .123e5
    k4 = .86e-3
    k5 = .82e-3
    k6 = .15e5
    k7 = .13e-3
    k8 = .24e5
    k9 = .165e5
    k10 = .9e4
    k11 = .22e-1
    k12 = .12e5
    k13 = .188e1
    k14 = .163e5
    k15 = .48e7
    k16 = .35e-3
    k17 = .175e-1
    k18 = .1e9
    k19 = .444e12
    k20 = .124e4
    k21 = .21e1
    k22 = .578e1
    k23 = .474e-1
    k24 = .178e4
    k25 = .312e1

    r1 = k1 * y[0]
    r2 = k2 * y[1] * y[3]
    r3 = k3 * y[4] * y[1]
    r4 = k4 * y[6]
    r5 = k5 * y[6]
    r6 = k6 * y[6] * y[5]
    r7 = k7 * y[8]
    r8 = k8 * y[8] * y[7]
    r9 = k9 * y[10] * y[1]
    r10 = k10 * y[10] * y[0]
    r11 = k11 * y[12]
    r12 = k12 * y[9] * y[1]
    r13 = k13 * y[13]
    r14 = k14 * y[0] * y[5]
    r15 = k15 * y[2]
    r16 = k16 * y[3]
    r17 = k17 * y[3]
    r18 = k18 * y[15]
    r19 = k19 * y[15]
    r20 = k20 * y[16] * y[5]
    r21 = k21 * y[18]
    r22 = k22 * y[18]
    r23 = k23 * y[0] * y[3]
    r24 = k24 * y[18] * y[0]
    r25 = k25 * y[19]

    dy = np.zeros(20)
    dy[0] = -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25
    dy[1] = -r2 - r3 - r9 - r12 + r1 + r21
    dy[2] = -r15 + r1 + r17 + r19 + r22
    dy[3] = -r2 - r16 - r17 - r23 + r15
    dy[4] = -r3 + r4 + r4 + r6 + r7 + r13 + r20
    dy[5] = -r6 - r8 - r14 - r20 + r3 + r18 + r18
    dy[6] = -r4 - r5 - r6 + r13
    dy[7] = r4 + r5 + r6 + r7
    dy[8] = -r7 - r8
    dy[9] = -r12 + r7 + r9
    dy[10] = -r9 - r10 + r8 + r11
    dy[11] = r9
    dy[12] = -r11 + r10
    dy[13] = -r13 + r12
    dy[14] = r14
    dy[15] = -r18 - r19 + r16
    dy[16] = -r20
    dy[17] = r20
    dy[18] = -r21 - r22 - r24 + r23 + r25
    dy[19] = -r25 + r24
    return dy


def get_data():
    rsteps = np.linspace(1e-5, 1e5, 50)
    rspan = (1e-5, 1e5)
    r0 = 1, 0, 0
    sol_rober = solve_ivp(rober, rspan, r0, method='Radau', t_eval=rsteps, atol=1e-7)
    #print(sol_rober.nfev, 'evaluations required.')

    psteps = np.linspace(0, 60, 50)
    pspan = (0, 60)
    p0 = np.zeros(20)
    p0[1] = 0.2
    p0[3] = 0.04
    p0[6] = 0.1
    p0[7] = 0.3
    p0[8] = 0.01
    p0[16] = 0.007

    # Note: unable to find OdeSolver in python that can forward solve the POLLU system
    # scipy gives following message due to stiffness: 'Required step size is less than spacing between numbers.'
    sol_pollu = solve_ivp(pollu, pspan, p0, method='Radau', t_eval=psteps, atol=1e-7)
    #print(sol_pollu.nfev, 'evaluations required.')
    return sol_rober, sol_pollu

def save_data():
    sol_rober, sol_pollu = get_data()
    r = {'sp1': sol_rober.y[0], 'sp2': sol_rober.y[1], 'sp3': sol_rober.y[2]}
    r_df = pd.DataFrame(r, index=sol_rober.t)
    p = {'sp1': sol_pollu.y[0], 'sp2': sol_pollu.y[1], 'sp3': sol_pollu.y[2], 'sp4': sol_pollu.y[3], 'sp5': sol_pollu.y[4],
        'sp6': sol_pollu.y[5], 'sp7': sol_pollu.y[6], 'sp8': sol_pollu.y[7], 'sp9': sol_pollu.y[8], 'sp10': sol_pollu.y[9],
        'sp11': sol_pollu.y[10], 'sp12': sol_pollu.y[11], 'sp13': sol_pollu.y[12], 'sp14': sol_pollu.y[13],
        'sp15': sol_pollu.y[14],
        'sp16': sol_pollu.y[15], 'sp17': sol_pollu.y[16], 'sp18': sol_pollu.y[17], 'sp19': sol_pollu.y[18],
        'sp20': sol_pollu.y[19]}
    p_df = pd.DataFrame(p, index=sol_pollu.t)
    p_df.to_csv("POLLU_data")
    r_df.to_csv("ROBER_data")

save_data()