"""Data for Fig. 1: the modal weight curves g(omega_r; omega) of the three damping models, written to data/g_curves.dat.
(The modal terms of Fig. 3 are written by code/export_paper.py.)
Procedural style. Writes .dat files readable by pgfplots into ./data/.
"""
import os
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

N = 11          # masses
IC = 5          # drive point (0-based), central mass
W0 = 300.0      # analysis frequency [rad/s]
ETA = 0.02      # hysteretic loss factor
ALPHA, BETA = 2.0, 1e-5   # Rayleigh
M = np.eye(N)
F = np.zeros(N, complex); F[IC] = 1.0


def build_K(k):
    """Fixed-fixed chain: N masses, N+1 springs k[0..N]."""
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = k[i] + k[i + 1]
        if i > 0:
            K[i, i - 1] = K[i - 1, i] = -k[i]
    return K


def dK_dkj(j):
    dK = np.zeros((N, N))
    if j < N: dK[j, j] += 1
    if j > 0: dK[j - 1, j - 1] += 1
    if 0 < j < N:
        dK[j, j - 1] -= 1; dK[j - 1, j] -= 1
    return dK


def Zmat(w, K, model):
    C = ALPHA * M + BETA * K
    if model == "visc":  return K - w**2 * M + 1j * w * C
    if model == "hyst":  return K * (1 + 1j * ETA) - w**2 * M
    if model == "both":  return K * (1 + 1j * ETA) - w**2 * M + 1j * w * C
    raise ValueError(model)


def input_power(w, K, model):
    u = np.linalg.solve(Zmat(w, K, model), F)
    return 0.5 * np.real(np.conj(F) @ (1j * w * u)), u


def modal_terms(w, K, Phi, model):
    _, u = input_power(w, K, model)
    q = Phi.T @ M @ u
    f = Phi.T @ F
    return 0.5 * np.real(1j * w * np.conj(f) * q)


def write(name, header, cols):
    arr = np.column_stack(cols)
    np.savetxt(os.path.join(OUT, name), arr, header=header, comments="")


# ---------------------------------------------------------------- g_r curves
def g_of(wr, model):
    if model == "visc": d = 2 * ((ALPHA / wr + BETA * wr) / 2) * wr * W0
    if model == "hyst": d = ETA * wr**2
    if model == "both": d = ETA * wr**2 + 2 * ((ALPHA / wr + BETA * wr) / 2) * wr * W0
    return 0.5 * W0 * d / ((wr**2 - W0**2)**2 + d**2)
wr = np.geomspace(10, 1e4, 2000)
# g of Eq. (Pr) in the manuscript carries no 1/2 factor; g_of includes it, hence the factor 2 here
write("g_curves.dat", "wr g_visc g_hyst g_both", [wr, 2 * g_of(wr, "visc"), 2 * g_of(wr, "hyst"), 2 * g_of(wr, "both")])
wstar = W0 * (1 + ETA**2)**-0.25
