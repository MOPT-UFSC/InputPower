"""Displacement objectives on the chain (Section 4.2 check): minimise |u_6| (drive point) and |u_9| (non-collocated)
with the stiffness budget and box of Table 1, SQP on log k, initial design plus random starts. Reports the final
displacement, its real and imaginary parts, the power, the bound 2P/(omega |F|) and the excited gap.
Writes results/disp_<dof>.npz."""
import os, numpy as np
from scipy.optimize import minimize
from models import chain, power_and_grads, modes, excited_gap, W0, ETA
from sqp_driver import random_start

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results"); KB = (1e4, 5e5)
mdl = chain(); f = mdl["f"]; nk = mdl["n_k"]; K0 = mdl["k0"].sum(); dK = [np.asarray(D) for D in mdl["dK"]]


def disp_and_grad(k, j):
    Z = mdl["assemble_K"](k) * (1 + 1j * ETA) - W0**2 * mdl["assemble_M"](mdl["m0"]); u = np.linalg.solve(Z, f)
    lam = np.linalg.solve(Z.T, np.eye(len(f))[j])          # adjoint for u_j
    du = np.array([-(1 + 1j * ETA) * (lam @ (D @ u)) for D in dK])   # du_j/dk_i
    a = abs(u[j]); g = np.real(np.conj(u[j]) * du) / a
    return a, g, u


def solve(j, k_start):
    def fun(x):
        k = np.exp(x); a, g, _ = disp_and_grad(k, j); return np.log(a), g * k / a
    cons = [{"type": "eq", "fun": lambda x: np.exp(x).sum() / K0 - 1, "jac": lambda x: np.exp(x) / K0}]
    r = minimize(fun, np.log(k_start), jac=True, method="SLSQP", bounds=[(np.log(KB[0]), np.log(KB[1]))] * nk, constraints=cons, options=dict(maxiter=3000, ftol=1e-14))
    k = np.clip(np.exp(r.x) * K0 / np.exp(r.x).sum(), *KB); return k, disp_and_grad(k, j)[0]


rng = np.random.default_rng(3)
for j, lab in ((5, "u6"), (8, "u9")):
    runs = [("initial", *solve(j, mdl["k0"]))]
    for i in range(30):
        ks, _ = random_start(mdl, True, False, KB, (1, 1), rng); runs.append(("random %d" % i, *solve(j, ks)))
    runs.sort(key=lambda z: z[2]); name, k, a = runs[0]
    a0 = disp_and_grad(mdl["k0"], j)[0]; _, _, u = disp_and_grad(k, j); P = power_and_grads(mdl, k, mdl["m0"])[0]; P0 = power_and_grads(mdl, mdl["k0"], mdl["m0"])[0]
    wr, p, mu = modes(mdl, k, mdl["m0"]); lo, hi = excited_gap(wr, p, mu)
    print("min |%s|: start %s; |u| %.3e -> %.3e (x%.1e); Re u %.2e Im u %.2e; 2P/(w|F|) = %.3e; P %.3e -> %.3e; excited gap [%.0f, %.0f]; distinct minima %d/%d"
          % (lab, name, a0, a, a / a0, u[j].real, u[j].imag, 2 * P / W0, P0, P, lo, hi, len(set(np.round(np.log10([z[2] for z in runs]), 3))), len(runs)), flush=True)
    # statistics over all local minima found
    stats = []
    for nm, kk, aa in runs:
        _, _, uu = disp_and_grad(kk, j); PP = power_and_grads(mdl, kk, mdl["m0"])[0]; wr_, p_, mu_ = modes(mdl, kk, mdl["m0"]); lo_, hi_ = excited_gap(wr_, p_, mu_)
        near = wr_[(p_ / mu_ >= 1e-2) & (wr_ > 0.8 * W0) & (wr_ < 1.4 * W0)]
        stats.append((aa, PP, abs(uu[j].real) / aa, 2 * PP / W0 / aa, lo_, hi_, near))
    print("   over the %d minima: |u| from %.1e to %.1e m; P from %.2e to %.2e W (P0/P %.1f to %.1f); |Re u|/|u| max %.2f; (2P/(w|F|))/|u| min %.4f max %.4f; minima with an excited mode in [0.8w, 1.4w]: %d, those modes between %.0f and %.0f rad/s"
          % (len(stats), min(s[0] for s in stats), max(s[0] for s in stats), min(s[1] for s in stats), max(s[1] for s in stats), P0 / max(s[1] for s in stats), P0 / min(s[1] for s in stats), max(s[2] for s in stats), min(s[3] for s in stats), max(s[3] for s in stats),
             sum(1 for s in stats if len(s[6])), min([w for s in stats for w in s[6]] or [0]), max([w for s in stats for w in s[6]] or [0])), flush=True)
    np.savez(os.path.join(OUT, "disp_%s.npz" % lab), k=k, m=mdl["m0"], disp=a, disp0=a0, u=u[j], P=P, P0=P0, wr=wr, p=p, mu=mu, lo=lo, hi=hi, all_disp=np.array([z[2] for z in runs]), all_P=np.array([s[1] for s in stats]), all_lo=np.array([s[4] for s in stats]), all_hi=np.array([s[5] for s in stats]))
