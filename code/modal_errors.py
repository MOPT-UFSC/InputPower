"""Relative difference between the modal sum and the direct power at every level of the escape sequences, optimized and
explicit designs (the numbers quoted in Section 4.4 for the modal accuracy)."""
import numpy as np
from scipy.linalg import eigh
from models import chain, truss, W0, ETA
S = [1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5]
for mode in ("k", "m", "km"):
    a = np.load("results/escape_seq_%s.npy" % mode, allow_pickle=True).item()
    for name, mdl in (("chain", chain()), ("truss", truss())):
        for which, lab in ((0, "optimized"), (1, "explicit")):
            errs = []
            for d in a[name][which]:
                K = mdl["assemble_K"](d["k"]); M = mdl["assemble_M"](d["m"]); f = mdl["f"]; lam, Phi = eigh(K, M); p = (Phi.T @ f)**2
                g = W0 * ETA * lam / ((lam - W0**2)**2 + (ETA * lam)**2); Pm = 0.5 * (p * g).sum(); u = np.linalg.solve(K * (1 + 1j * ETA) - W0**2 * M, f); Pd = 0.5 * np.real(f @ (1j * W0 * u)); errs.append(abs(Pm - Pd) / Pd)
            print("%-2s %-5s %-9s " % (mode, name, lab) + " ".join("s=%g:%.0e" % (s, e) for s, e in zip(S, errs)))
