"""Convergence of the best power with the number of mirror-symmetric random starts for the four mirror-symmetric
problems whose designs changed (k+m of Table 1, non-adjacent pairs): 1000 symmetric starts each, seed 12011 (the
campaign uses 200, seed 1011). Prints the best value after 50, 100, 200, 500 and 1000 starts and saves the best design
to results/symconv_<case>.npz."""
import os, sys, numpy as np
from models import chain, truss, mirror_maps, asymmetry, results_dir, results_file
from sqp_driver import run_sqp, random_start

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000; KB = (1e4, 5e5)
for name in ("chain", "truss"):
    base = chain() if name == "chain" else truss(); mk, mm = mirror_maps(base)
    for case, vm, mb in (("%s_km" % name, True, (0.1, 5.0)), ("pairs_%s_nonadj" % name, False, (1, 1))):
        d = np.load(results_file(case + ".npz")); mdl = dict(base)
        if not vm: mdl["f"] = d["f"]
        rng = np.random.default_rng(12011); Ps = []; best = None
        for i in range(N):
            ks, ms = random_start(mdl, True, vm, KB, mb, rng); ks = 0.5 * (ks + ks[mk]); ks *= base["k0"].sum() / ks.sum()
            if vm: ms = 0.5 * (ms + ms[mm]); ms *= base["m0"].sum() / ms.sum()
            k, m, P, hist, info = run_sqp(mdl, True, vm, KB, mb, ks, ms if vm else None); Ps.append(P)
            if best is None or P < best[0]: best = (P, k, m, i)
        Ps = np.array(Ps); run = np.minimum.accumulate(Ps)
        print("%-20s campaign (200 starts) %.4e | best after 50/100/200/500/%d starts: %s | best %.4e at start %d (asym k %.1e) | runs within 0.2%% of best: %d"
              % (case, float(d["P"]), N, " ".join("%.4e" % run[min(q, N) - 1] for q in (50, 100, 200, 500, N)), best[0], best[3], asymmetry(best[1], mk), int(np.sum(Ps <= best[0] * 1.002))), flush=True)
        np.savez(os.path.join(results_dir(), "symconv_%s.npz" % case), k=best[1], m=best[2], P=best[0], all_P=np.sort(Ps))
