"""Audit of the mirror symmetry of the reported constrained designs. A problem is mirror-symmetric when the load, the
bounds and the budgets are invariant under the mirror about the loaded point AND every non-design quantity is too
(with the masses only, the fixed stiffnesses are the random initial ones, so that problem is not symmetric).
For each design: asymmetry of k and m, power of the mirrored design, power of the symmetrised design (average of the
design and its mirror, which is admissible because the box and the budgets are convex)."""
import numpy as np
from models import chain, truss, power_and_grads, results_file

NX = 11


def mirrors(name, mdl):
    if name == "chain": return np.arange(mdl["n_k"])[::-1], np.arange(len(mdl["m0"]))[::-1]
    bars = [tuple(b) for b in mdl["bars"]]; mn = lambda nn: (nn // NX) * NX + (NX - 1 - nn % NX)
    key = {tuple(sorted(b)): i for i, b in enumerate(bars)}
    mk = np.array([key[tuple(sorted((mn(i), mn(j))))] for i, j in bars])
    nodes = mdl["mass_nodes"] if "mass_nodes" in mdl else None
    return mk, nodes


def asym(x, mx): return np.max(np.abs(x - x[mx])) / np.max(x)


if __name__ == "__main__":
    for name in ("chain", "truss"):
        mdl = chain() if name == "chain" else truss(); mk, mm = mirrors(name, mdl)
        print(name, "n_k", mdl["n_k"], "n_m", len(mdl["m0"]), "keys", [q for q in mdl.keys()][:14])
        print("  initial stiffness asymmetry %.2f" % asym(mdl["k0"], mk))

    print()
    KB, MB = (1e4, 5e5), (0.2, 5.0)
    for name in ("chain", "truss"):
        mdl = chain() if name == "chain" else truss(); mk, mm = mirrors(name, mdl)
        if name == "truss": mn = lambda nn: (nn // NX) * NX + (NX - 1 - nn % NX); mm = np.array([mn(i) for i in range(len(mdl["m0"]))])
        for vs in ("k", "m", "km"):
            d = np.load(results_file("%s_%s.npz" % (name, vs))); k, m = d["k"], d["m"]; P = float(d["P"])
            Pm = power_and_grads(mdl, k[mk], m[mm])[0]; Ps = power_and_grads(mdl, 0.5 * (k + k[mk]), 0.5 * (m + m[mm]))[0]
            print("Table 1 %-5s %-2s  P %.4e | asym k %.3f m %.3f | mirrored P %.4e (ratio %.4f) | symmetrised P %.4e (ratio %.3f) | problem symmetric: %s"
                  % (name, vs, P, asym(k, mk), asym(m, mm), Pm, Pm / P, Ps, Ps / P, "no (fixed k0 asymmetric)" if vs == "m" else "yes"))
        if name == "truss":
            fixed = [i for i in range(len(mdl["m0"])) if len(mdl["mass_dofs"][i]) == 0]
            for vs in ("m", "km"):
                m = np.load(results_file("truss_%s.npz" % vs))["m"]; print("   truss %s: masses at the fixed nodes %s = %s kg (sum %.2f of budget %.0f)" % (vs, fixed, np.round(m[fixed], 2), m[fixed].sum(), mdl["m0"].sum()))
