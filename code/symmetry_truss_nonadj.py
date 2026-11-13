"""Non-adjacent antiphase pair on the truss: large search over mirror-symmetric designs in the reduced space of the 25
stiffness pairs and the central vertical (SLSQP, ftol 1e-12, 500 iterations), followed by a full-space polish of the
30 lowest (run_sqp, ftol 1e-14). Usage: python symmetry_truss_nonadj.py [n_starts]"""
import sys, os, time, numpy as np
from scipy.optimize import minimize
from models import truss, power_and_grads, mirror_maps, asymmetry, results_dir, results_file, count_groups
from run_symmetric import red, to_full, w
from sqp_driver import run_sqp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; KB = (1e4, 5e5)
base = truss(); f = np.load(results_file("pairs_truss_nonadj.npz"))["f"]; full = dict(base); full["f"] = f; rd = dict(red); rd["f"] = f
budget = base["k0"].sum(); lo, hi = np.log(KB[0] * w), np.log(KB[1] * w); m0 = base["m0"]; rng = np.random.default_rng(2026); mk, _ = mirror_maps(base)


def fobj(x):
    y = np.exp(x); P, gk, _ = power_and_grads(rd, y, m0); return np.log(P), gk * y / P


cons = {"type": "eq", "fun": lambda x: np.exp(x).sum() / budget - 1, "jac": lambda x: np.exp(x) / budget}
found = []; t0 = time.time()
for it in range(N):
    y0 = np.exp(rng.uniform(lo, hi)); y0 *= budget / y0.sum()
    for _ in range(50): y0 = np.clip(y0, KB[0] * w, KB[1] * w); y0 *= budget / y0.sum()
    r = minimize(fobj, np.log(y0), jac=True, method="SLSQP", bounds=list(zip(lo, hi)), constraints=[cons], options=dict(maxiter=500, ftol=1e-12))
    k = to_full(np.exp(r.x)); k = np.clip(k * budget / k.sum(), *KB); found.append((power_and_grads(full, k, m0)[0], k))
found.sort(key=lambda z: z[0]); Ps = np.array([q for q, _ in found])
print("reduced space: %d starts in %.0f s; lowest ten %s; runs within 0.2%% of the lowest: %d" % (N, time.time() - t0, ["%.4e" % q for q in Ps[:10]], int(np.sum(Ps <= Ps[0] * 1.002))), flush=True)
pol = []
for P0, k0 in found[:30]:
    k, m, P, hist, info = run_sqp(full, True, False, KB, (1, 1), k0, None); pol.append((P, k, P0))
pol.sort(key=lambda z: z[0]); P, k, P0 = pol[0]
print("full-space polish of the 30 lowest: best %.4e (from %.4e), asym %.1e; five lowest after polish %s" % (P, P0, asymmetry(k, mk), ["%.4e" % q[0] for q in pol[:5]]), flush=True)
np.savez(os.path.join(results_dir(), "symsearch_pairs_truss_nonadj.npz"), k=k, m=m0, P=P, f=f, all_P=Ps)
