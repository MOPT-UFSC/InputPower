"""Checks for Remark 'Classical damping' and for the Discussion on non-proportional damping (chain of Section 4.1).
(a) Caughey series C = M (a0 I + a1 M^-1 K + a2 (M^-1 K)^2) and prescribed modal ratios zeta_r: the modal sum
    1/2 sum_r p_r g_r with d_r = omega sum_j a_j omega_r^(2j) (or 2 zeta_r omega_r omega) equals the direct power.
(b) Hysteretic damping eta plus a dashpot c at the loaded mass: P(c) versus the Theorem 1 bound with the weights of
    the hysteretic minorant eta K / omega, for B = [omega/2, 2 omega]."""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from models import chain, W0, ETA

mdl = chain(); k = mdl["k0"]; m = mdl["m0"]; f = mdl["f"]; K = mdl["assemble_K"](k); M = mdl["assemble_M"](m); w = W0
lam, Phi = eigh(K, M); wr = np.sqrt(lam); p = (Phi.T @ f) ** 2


def power(Z):
    u = np.linalg.solve(Z, f); return 0.5 * np.real(np.conj(f) @ (1j * w * u))


def weight(d):
    return w * d / ((wr ** 2 - w ** 2) ** 2 + d ** 2)


a0, a1, a2 = 5.0, 1e-5, 1e-12
A = np.linalg.solve(M, K); C = M @ (a0 * np.eye(len(f)) + a1 * A + a2 * A @ A)
P = power(K - w ** 2 * M + 1j * w * C); Pm = 0.5 * np.sum(p * weight(w * (a0 + a1 * wr ** 2 + a2 * wr ** 4)))
print("Caughey (a0, a1, a2) = (%g, %g, %g): direct %.6e W, modal sum %.6e W, relative difference %.1e" % (a0, a1, a2, P, Pm, abs(P - Pm) / P))
zeta = 0.01 + 0.02 * np.random.default_rng(0).random(len(f)); Cm = M @ Phi @ np.diag(2 * zeta * wr) @ Phi.T @ M
P = power(K - w ** 2 * M + 1j * w * Cm); Pm = 0.5 * np.sum(p * weight(2 * zeta * wr * w))
print("prescribed modal ratios in [0.01, 0.03]: direct %.6e W, modal sum %.6e W, relative difference %.1e" % (P, Pm, abs(P - Pm) / P))

xs = np.linspace(w / 2, 2 * w, 200001); gB = np.min(w * ETA * xs ** 2 / ((xs ** 2 - w ** 2) ** 2 + (ETA * xs ** 2) ** 2))
B = (wr >= w / 2) & (wr <= 2 * w); share = p[B].sum(); bound = 0.5 * gB * share
print("hysteretic eta = %g, B = [w/2, 2w]: g_B = %.3e s, participation in B = %.3f, Theorem 1 lower bound on P = %.3e W" % (ETA, gB, share, bound))


def P_dashpot(c):
    C = np.zeros_like(K); C[5, 5] = c; return power(K * (1 + 1j * ETA) - w ** 2 * M + 1j * w * C)


for c in [0, 1e1, 1e2, 1e3, 3e3, 1e4, 1e5, 1e6]:
    print("  dashpot c = %.0e N s/m at the loaded mass: P = %.3e W, |F|^2/(2c) = %s, bound holds: %s" % (c, P_dashpot(c), "inf" if c == 0 else "%.2e" % (1 / (2 * c)), P_dashpot(c) >= bound))
cs = brentq(lambda c: P_dashpot(c) - bound, 1e3, 1e7); print("  the bound fails beyond c = %.2e N s/m" % cs)
cc = np.logspace(0, 5, 2001); PP = np.array([P_dashpot(c) for c in cc]); print("  maximum of P(c) at c = %.1e N s/m, P = %.3e W" % (cc[PP.argmax()], PP.max()))
