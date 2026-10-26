"""Multi-start search over designs symmetric about the load, stiffness only, with the budget (sum k = sum k0) and the
box [1e4, 5e5] of Table 1. SLSQP on the logarithms of the reduced variables (pair totals; chain: 6 pairs; truss: 25 pairs
plus the central vertical). Random log-uniform starts scaled to the budget. Saves the best design of each system in
results/sym_best_<system>.npz and prints all local minima found. Usage: python run_sym_multistart.py [n_starts]"""
import sys, os, numpy as np
from scipy.optimize import minimize
from models import chain, truss, power_and_grads, modes, excited_gap, W0, count_groups
from run_symmetric import red as truss_red, to_full as truss_full, groups as truss_groups, w as truss_w, t as truss_model
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rng = np.random.default_rng(7)
# chain reduction: springs j and 11-j are mirror images about mass 6 (index 5); 12 springs -> 6 pairs
c = chain(); nk = c["n_k"]; cg = [[j, nk - 1 - j] for j in range(nk // 2)]; cw = np.full(len(cg), 2.0)
def chain_full(y): k = np.zeros(nk); [k.__setitem__(g, y[i] / 2) for i, g in enumerate(cg)]; return k
chain_red = dict(c); chain_red["n_k"] = len(cg); chain_red["dK"] = [(np.asarray(c["dK"][g[0]]) + np.asarray(c["dK"][g[1]])) / 2 for g in cg]
chain_red["assemble_K"] = lambda y: c["assemble_K"](chain_full(y))
cases = [("chain", c, chain_red, chain_full, cw), ("truss", truss_model, truss_red, truss_full, truss_w)]
for name, full, red, to_full, w in cases:
    budget = full["k0"].sum(); lo, hi = np.log(1e4 * w), np.log(5e5 * w); m0 = full["m0"]
    def fobj(x):
        y = np.exp(x); P, gk, _ = power_and_grads(red, y, m0); return np.log(P), gk * y / P
    cons = {"type": "eq", "fun": lambda x: np.exp(x).sum() / budget - 1, "jac": lambda x: np.exp(x) / budget}
    found = []
    for it in range(N):
        y0 = np.exp(rng.uniform(lo, hi)); y0 *= budget / y0.sum()
        for _ in range(50):      # project the start into the box after scaling
            y0 = np.clip(y0, 1e4 * w, 5e5 * w); y0 *= budget / y0.sum()
        if np.any(y0 < 0.999 * 1e4 * w) or np.any(y0 > 1.001 * 5e5 * w): continue
        r = minimize(fobj, np.log(y0), jac=True, method="SLSQP", bounds=list(zip(lo, hi)), constraints=[cons], options=dict(maxiter=500, ftol=1e-12))
        y = np.exp(r.x); k = to_full(y)
        if abs(k.sum() / budget - 1) > 1e-6 or np.any(k < 0.999e4) or np.any(k > 0.5001e6): continue
        P = power_and_grads(full, k, m0)[0]; found.append((P, k))
    found.sort(key=lambda z: z[0]); P, k = found[0]; wr, p, mu = modes(full, k, m0)
    distinct = []
    for Pq, kq in found:
        if not any(abs(Pq - Pd) / Pd < 1e-3 for Pd, _ in distinct): distinct.append((Pq, kq))
    print("%s: %d successful starts, %d groups of final power values (0.2%% rule); best P=%.4e W, gap %s; the ten lowest: %s" % (name, len(found), count_groups([q for q, _ in found]), P, np.round(excited_gap(wr, p, mu), 1), ["%.3e" % q for q, _ in distinct[:10]]), flush=True)
    np.savez(os.path.join(OUT, "sym_best_%s.npz" % name), k=k, m=m0, P=P, all_P=np.array([q for q, _ in found]), wr=wr, p=p, mu=mu)
