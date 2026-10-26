"""Point 7 of the review: for representative designs of the escape sequences, eigenproblem residuals, mass-orthogonality
error, power by the direct, modal and dissipative forms in double precision, and the direct and modal power and the
lowest eigenvalue recomputed in 40-digit arithmetic (mpmath). Also the shares of the near-orthogonal modes between
300 and 330 rad/s of the truss stiffness sequence in double and in 40 digits. Writes verify_extreme.log and the LaTeX
rows of the table of Section 5.4 (verify_extreme_table.tex)."""
import numpy as np, mpmath as mp, time
from scipy.linalg import eigh
from models import chain, truss, W0, ETA
mp.mp.dps = 40
S = [1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5]
cases = [("k", "chain", 0, 10), ("k", "truss", 0, 10), ("k", "truss", 1, 10), ("m", "chain", 0, 10), ("m", "truss", 0, 10),
         ("km", "chain", 0, 6), ("km", "truss", 0, 6), ("km", "chain", 1, 6), ("km", "truss", 1, 6), ("km", "truss", 0, 8)]
rows = []
def mp_modes(K, M, f):
    n = len(f); Md = np.diag(M); dm = [1 / mp.sqrt(mp.mpf(x)) for x in Md]; A = mp.matrix(n, n)
    for i in range(n):
        for jj in range(n): A[i, jj] = mp.mpf(K[i, jj]) * dm[i] * dm[jj]
    E, Q = mp.eigsy(A); fm = [mp.mpf(x) for x in f]
    lam = np.array([float(E[i]) for i in range(n)]); p = np.array([float(sum(Q[i, r] * dm[i] * fm[i] for i in range(n))**2) for r in range(n)])
    return lam, p, E, fm
for mode, name, which, j in cases:
    mdl = chain() if name == "chain" else truss(); a = np.load("results/escape_seq_%s.npy" % mode, allow_pickle=True).item()[name][which][j]
    k, m = a["k"], a["m"]; K = mdl["assemble_K"](k); M = mdl["assemble_M"](m); f = mdl["f"]; n = len(f)
    lam, Phi = eigh(K, M); p = (Phi.T @ f)**2
    res = np.max(np.linalg.norm(K @ Phi - M @ Phi * lam, axis=0) / np.linalg.norm(K @ Phi, axis=0)); orth = np.max(abs(Phi.T @ M @ Phi - np.eye(n)))
    d = ETA * lam; g = W0 * d / ((lam - W0**2)**2 + d**2); Pm = 0.5 * np.sum(p * g)
    Z = K * (1 + 1j * ETA) - W0**2 * M; u = np.linalg.solve(Z, f); Pd = 0.5 * np.real(f @ (1j * W0 * u)); Pdiss = 0.5 * W0 * ETA * np.real(np.conj(u) @ K @ u)
    lam_mp, p_mp, E, fm = mp_modes(K, M, f)
    def g_mp(l):
        dd = mp.mpf(ETA) * l; return mp.mpf(W0) * dd / ((l - mp.mpf(W0)**2)**2 + dd**2)
    Pm_mp = float(mp.mpf(0.5) * sum(mp.mpf(p_mp[r]) * g_mp(E[r]) for r in range(n)))
    Zm = mp.matrix(n, n)
    for i in range(n):
        for jj in range(n): Zm[i, jj] = mp.mpc(K[i, jj]) * (1 + 1j * mp.mpf(ETA)) - (mp.mpf(W0)**2) * mp.mpf(M[i, jj])
    um = mp.lu_solve(Zm, mp.matrix(fm)); Pd_mp = float(mp.mpf(0.5) * mp.re(sum(fm[i] * (1j * mp.mpf(W0)) * um[i] for i in range(n))))
    l0, l0mp = lam.min(), lam_mp.min()
    row = (mode, name, "optimized" if which == 0 else "explicit", S[j], res, orth, abs(Pm - Pd) / Pd, abs(Pdiss - Pd) / Pd, abs(Pd_mp - Pd) / Pd, abs(Pm_mp - Pd_mp) / Pd_mp, abs(l0 - l0mp) / l0mp)
    rows.append(row); print("%-2s %-5s %-9s s=%-6g eig res %.1e  M-orth %.1e  |modal-direct| %.1e  |diss-direct| %.1e  |direct40-direct| %.1e  |modal40-direct40| %.1e  lowest eigenvalue double vs 40 %.1e" % row, flush=True)
# near-orthogonal modes of the truss stiffness sequence between 300 and 330 rad/s
print("\nTruss, stiffness only: shares of the modes between 300 and 330 rad/s, double vs 40 digits")
for j in (4, 8, 10):
    mdl = truss(); a = np.load("results/escape_seq_k.npy", allow_pickle=True).item()["truss"][0][j]
    K = mdl["assemble_K"](a["k"]); M = mdl["assemble_M"](a["m"]); f = mdl["f"]; lam, Phi = eigh(K, M); p = (Phi.T @ f)**2; mu = p.sum()
    lam_mp, p_mp, _, _ = mp_modes(K, M, f); mu_mp = p_mp.sum()
    for r in np.where((np.sqrt(lam) > 300) & (np.sqrt(lam) < 330))[0]:
        rm = np.argmin(abs(lam_mp - lam[r])); print("  s=%-6g wr=%.2f rad/s: share double %.1e, 40 digits %.1e" % (S[j], np.sqrt(lam[r]), p[r] / mu, p_mp[rm] / mu_mp))
d = np.load("results/truss_k.npz"); mdl = truss(); K = mdl["assemble_K"](d["k"]); M = mdl["assemble_M"](d["m"]); f = mdl["f"]; lam, Phi = eigh(K, M); p = (Phi.T @ f)**2; mu = p.sum()
lam_mp, p_mp, _, _ = mp_modes(K, M, f); mu_mp = p_mp.sum()
for r in np.where((np.sqrt(lam) > 270) & (np.sqrt(lam) < 330))[0]:
    rm = np.argmin(abs(lam_mp - lam[r])); print("  Table 1 truss bars only: wr=%.2f rad/s: share double %.1e, 40 digits %.1e" % (np.sqrt(lam[r]), p[r] / mu, p_mp[rm] / mu_mp))
with open("verify_extreme_table.tex", "w") as fh:
    for mode, name, fam, s, res, orth, e1, e2, e3, e4, e5 in rows:
        lab = {"k": "stiffness", "m": "masses", "km": "both"}[mode]
        fh.write("%s & %s & %s & $10^{%d}$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ \\\\\n" % (name, lab, fam, int(np.log10(s)), *["%.0e" % v for v in (res, orth, e1, e2, e3, e5)]))
print("table rows written")
