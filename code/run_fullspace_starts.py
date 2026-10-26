"""Stiffness-only problems of Table 1 from 300 random log-uniform starts in the full variable space (no symmetric start):
does any random start reach the mirror-symmetric optimum? Writes results/fullspace_<system>.npy (sorted final powers)."""
import numpy as np
from models import chain, truss
from sqp_driver import multistart
for name, mdl in (("chain", chain()), ("truss", truss())):
    k, m, P, hist, sname, allP = multistart(mdl, True, False, (1e4, 5e5), (1, 1), [], n_random=300, seed=5)
    best = np.load("results/%s_k.npz" % name)["P"]
    print("%s: 300 random full-space starts, lowest %.4e W (Table 1: %.4e, ratio %.4f), %d distinct values; ten lowest: %s" % (name, P, best, P / best, len(set(np.round(np.log10(allP), 3))), ["%.3e" % x for x in allP[:10]]), flush=True)
    np.save("results/fullspace_%s.npy" % name, allP)
