"""Escape sequence of the truss (box [kbar/s, kbar s] and, for km, [mbar/s, mbar s]; no budgets) optimised with a
TRUNCATED modal basis: modes with normalised participation p_r/mu <= thr are dropped from the modal power sum and from
its gradient (exact eigen-derivatives for k and m). The basis is re-selected after each accepted step. Each final
design is evaluated with the full harmonic model. Usage: python escape_trunc.py [k|km] [thr] [iters per level]."""
import os, sys, numpy as np
from scipy.linalg import eigh
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from models import truss, power_and_grads, spectrum, W0, ETA

MODE = sys.argv[1] if len(sys.argv) > 1 else "k"; THR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 500; VK, VM = "k" in MODE, "m" in MODE
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results")
S_LEVELS = [1, 10, 100, 1e3, 1e4]
mdl = truss(theta_deg=90); f = mdl["f"]; dK = [np.asarray(D) for D in mdl["dK"]]; NB, NM, ND = mdl["n_k"], mdl["n_m"], mdl["n_dof"]
kbar, mbar = np.mean(mdl["k0"]), np.mean(mdl["m0"])
# dM_i as index lists (diagonal mass matrix)
mass_dofs = mdl["mass_dofs"]


def g_of(wr, w=W0, eta=ETA):
    d = eta * wr**2; return w * d / ((wr**2 - w**2)**2 + d**2)


def dg_dlam(wr, w=W0, eta=ETA):
    lam = wr**2; den = (lam - w**2)**2 + eta**2 * lam**2
    return w * eta * (den - lam * (2 * (lam - w**2) + 2 * eta**2 * lam)) / den**2


def trunc_power_grad(k, m, thr, S_fixed=None):
    lam, Phi = eigh(mdl["assemble_K"](k), mdl["assemble_M"](m)); q = Phi.T @ f; p = q**2; mu = p.sum()
    S = S_fixed if S_fixed is not None else (np.where(p / mu > thr)[0] if thr > 0 else np.arange(len(p)))
    wr = np.sqrt(lam); g = g_of(wr); dg = dg_dlam(wr); P = 0.5 * np.sum(p[S] * g[S])
    dl = lam[:, None] - lam[None, :]; inv = np.zeros_like(dl); ok = abs(dl) > 1e-9 * np.maximum(abs(lam[:, None]), 1e-30); inv[ok] = 1.0 / dl[ok]
    gk = np.zeros(NB); gm = np.zeros(NM)
    if VK:
        for j in range(NB):
            A = Phi.T @ dK[j] @ Phi; dlam = np.diag(A); dq = (A.T * inv) @ q
            gk[j] = 0.5 * np.sum(2 * q[S] * dq[S] * g[S] + p[S] * dg[S] * dlam[S])
    if VM:
        for i in range(NM):
            d = mass_dofs[i]
            if len(d) == 0: continue
            Bm = Phi[d, :].T @ Phi[d, :]                     # phi_s^T dM_i phi_r
            dlam = -lam * np.diag(Bm)                          # d lambda_r / dm_i
            A = -Bm * lam[None, :]                             # phi_s^T (dK - lambda_r dM_i) phi_r with dK = 0
            dq = (A.T * inv) @ q - 0.5 * np.diag(Bm) * q       # normalisation term
            gm[i] = 0.5 * np.sum(2 * q[S] * dq[S] * g[S] + p[S] * dg[S] * dlam[S])
    return P, gk, gm, S, wr, p / mu


def full_eval(k, m):
    P = power_and_grads(mdl, k, m)[0]; lam, Phi = eigh(mdl["assemble_K"](k), mdl["assemble_M"](m)); wr = np.sqrt(np.maximum(lam, 0)); p = (Phi.T @ f)**2; sh = p / p.sum()
    near = (wr > 0.5 * W0) & (wr < 2 * W0)
    if near.any():
        Pk = spectrum(mdl, k, m, wr[near]) / P; jj = np.argmax(Pk); return P, wr[near][jj], sh[near][jj], Pk[jj], wr, sh
    return P, np.nan, 0.0, 0.0, wr, sh


# gradient check
k = mdl["k0"] * (1 + 0.1 * np.random.default_rng(5).standard_normal(NB)); m = mdl["m0"] * (1 + 0.1 * np.random.default_rng(6).standard_normal(NM))
P, gk, gm, S, _, _ = trunc_power_grad(k, m, THR)
err = []
for j in (0, 25, 40):
    kp = k.copy(); kp[j] += 1.0; km_ = k.copy(); km_[j] -= 1.0; err.append(abs((trunc_power_grad(kp, m, THR, S)[0] - trunc_power_grad(km_, m, THR, S)[0]) / 2 - gk[j]) / abs(gk[j]))
if VM:
    for i in (3, 16, 20):
        mp = m.copy(); mp[i] += 1e-5; mm = m.copy(); mm[i] -= 1e-5; err.append(abs((trunc_power_grad(k, mp, THR, S)[0] - trunc_power_grad(k, mm, THR, S)[0]) / 2e-5 - gm[i]) / abs(gm[i]))
print("mode %s thr %g: |S|=%d, gradient check max rel err %.1e" % (MODE, THR, len(S), max(err)), flush=True)

# SQP (SLSQP) on the log10 variables with the truncated objective; the basis is re-selected at every evaluation; continuation in s
from scipy.optimize import minimize
k = mdl["k0"].copy(); m = mdl["m0"].copy(); seq = []
for s in S_LEVELS:
    lk = np.clip(np.log10(k), np.log10(kbar / s), np.log10(kbar * s)); lm = np.clip(np.log10(m), np.log10(mbar / s), np.log10(mbar * s))
    x0 = np.concatenate([lk if VK else [], lm if VM else []]); bounds = ([(np.log10(kbar / s), np.log10(kbar * s))] * NB if VK else []) + ([(np.log10(mbar / s), np.log10(mbar * s))] * NM if VM else [])
    def split(x):
        kk = 10**x[:NB] if VK else k; mm = 10**x[NB:] if (VK and VM) else (10**x if VM else m); return kk, mm
    hist = []
    def fun(x):
        kk, mm = split(x); P, gk, gm, S, _, _ = trunc_power_grad(kk, mm, THR)
        g = np.concatenate([gk * kk if VK else [], gm * mm if VM else []]) * np.log(10) / P   # d log P / d log10 x
        hist.append((len(hist), P, len(S))); return np.log(P), g
    r = minimize(fun, x0, jac=True, method="SLSQP", bounds=bounds, options=dict(maxiter=ITERS, ftol=1e-12))
    k, m = split(r.x); P, gk, gm, S, _, _ = trunc_power_grad(k, m, THR)
    Pf, wn, shn, pk, wr, sh = full_eval(k, m); hist = np.array(hist)
    seq.append(dict(s=s, k=k.copy(), m=m.copy(), P_model=P, P_full=Pf, w_peak=wn, share_peak=shn, peak=pk, wr=wr, sh=sh, hist=hist, nS=len(S)))
    print("s=%6g: P_model=%.3e P_full=%.3e (ratio %.3f) |S|=%2d; strongest peak in [w/2, 2w]: %.1f rad/s, share %.1e, P(wr)/P(w0)=%.1f; evals %d" % (s, P, Pf, P / Pf, len(S), wn, shn, pk, len(hist)), flush=True)
np.save(os.path.join(OUT, "escape_trunc_%s_thr%g.npy" % (MODE, THR)), seq, allow_pickle=True)

# figure: full spectra per level, truncated run vs full-model run (escape_seq_<mode>.npy)
R = np.load(os.path.join(OUT, "escape_seq_%s.npy" % MODE), allow_pickle=True).item()["truss"][0]
FULL_LEVELS = [1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5]
WS = np.geomspace(0.1, 1e6, 4000)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
cols = plt.cm.viridis(np.linspace(0.95, 0.0, len(S_LEVELS)))
for ax, (title, items) in zip(axes, [("full modal basis", [(a["k"], a["m"], a["P"]) for a in [R[FULL_LEVELS.index(s)] for s in S_LEVELS]]),
                                     ("truncated basis, thr=%g (evaluated with all modes)" % THR, [(a["k"], a["m"], a["P_full"]) for a in seq])]):
    for c, s, (k_, m_, P_) in zip(cols, S_LEVELS, items):
        ax.loglog(WS, spectrum(mdl, k_, m_, WS), color=c, lw=1.1, label="$s=%g$ ($P$=%.1e W)" % (s, P_))
    ax.axvline(W0, color="crimson", ls="--"); ax.axvspan(W0 / 2, 2 * W0, color="crimson", alpha=0.06)
    ax.set_xlim(0.1, 1e6); ax.set_ylim(1e-21 if VM else 1e-13, 1e2); ax.grid(True, alpha=0.3); ax.set_title("truss, variables %s: %s" % (MODE, title), fontsize=9); ax.legend(fontsize=7, loc="upper right"); ax.set_xlabel("$\\omega$ [rad/s]")
axes[0].set_ylabel("$P(\\omega)$ [W]")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "escape_trunc_%s_thr%g.png" % (MODE, THR)), dpi=150)
print("written escape_trunc_%s_thr%g.png" % (MODE, THR))
