"""Stiffness-and-mass problems of Table 1 (mirror-symmetric problems whose best designs are asymmetric): SQP in the full
space from 100 mirror-symmetric random starts and from the symmetrised best design, to see whether a symmetric design
of lower power exists and whether SQP stays on the symmetric subspace."""
import numpy as np
from models import chain, truss, power_and_grads, results_file
from sqp_driver import run_sqp, random_start
from symmetry_audit import mirrors, asym, NX

KB, MB = (1e4, 5e5), (0.1, 5.0)
for name in ("chain", "truss"):
    mdl = chain() if name == "chain" else truss(); mk, mm = mirrors(name, mdl)
    if name == "truss": mm = np.array([(i // NX) * NX + (NX - 1 - i % NX) for i in range(len(mdl["m0"]))])
    d = np.load(results_file("%s_km.npz" % name)); Pbest = float(d["P"]); rng = np.random.default_rng(21); runs = []
    def sym(k, m):
        k = 0.5 * (k + k[mk]); m = 0.5 * (m + m[mm]); return k * mdl["k0"].sum() / k.sum(), m * mdl["m0"].sum() / m.sum()
    starts = [("symmetrised best", *sym(d["k"], d["m"]))] + [("sym random %d" % i, *sym(*random_start(mdl, True, True, KB, MB, rng))) for i in range(100)]
    for nm, ks, ms in starts:
        k, m, P, hist, info = run_sqp(mdl, True, True, KB, MB, ks, ms); runs.append((P, asym(k, mk), asym(m, mm), nm))
    runs.sort(); stay = [r for r in runs if r[1] < 1e-3 and r[2] < 1e-3]
    print("%s km: best asymmetric (Table 1) %.4e | from 101 symmetric starts: lowest %.4e (asym k %.3f m %.3f, %s); %d runs stayed symmetric, lowest of those %.4e (ratio to Table 1 %.3f); runs below Table 1: %d"
          % (name, Pbest, runs[0][0], runs[0][1], runs[0][2], runs[0][3], len(stay), stay[0][0] if stay else np.nan, (stay[0][0] / Pbest) if stay else np.nan, sum(1 for r in runs if r[0] < Pbest * 0.999)), flush=True)
    print("   five lowest:", [("%.4e" % r[0], round(r[1], 3), round(r[2], 3)) for r in runs[:5]], flush=True)
