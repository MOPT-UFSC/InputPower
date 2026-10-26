"""Writes the supplementary material: the final stiffness and mass vectors of every design reported in the paper, as CSV
files (one row per spring/bar or mass, with its index), plus the load vectors of the pair examples. Reads the saved
results (code/results/) and writes to the supplement/ directory at the root of the repository, next to code/."""
import os, numpy as np
from models import chain, truss, results_file
SUPP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "supplement")); os.makedirs(SUPP, exist_ok=True)
S = [1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5]
def write(name, k, m, header):
    with open(os.path.join(SUPP, "%s.csv" % name), "w") as fh:
        fh.write("# %s\n# index,stiffness_N_per_m,mass_kg (masses listed up to their count; blank afterwards)\n" % header)
        for i in range(max(len(k), len(m))):
            fh.write("%d,%s,%s\n" % (i, "%.17g" % k[i] if i < len(k) else "", "%.17g" % m[i] if i < len(m) else ""))
# Table 1
for name in ("chain", "truss"):
    for c, lab in (("k", "stiffness only"), ("m", "masses only"), ("km", "stiffness and masses")):
        d = np.load(results_file("%s_%s.npz" % (name, c))); write("table1_%s_%s" % (name, c), d["k"], d["m"], "Table 1, %s, %s; P = %.6e W" % (name, lab, float(d["P"])))
for lab in ("u6", "u9"):
    d = np.load(results_file("disp_%s.npz" % lab)); write("disp_chain_%s" % lab, d["k"], d["m"], "Section 4.2, chain, minimum of |%s|; |u| = %.6e m, P = %.6e W" % (lab, float(d["disp"]), float(d["P"])))
# symmetric multistart best designs (stiffness only, masses at initial values)
for name, mdl in (("chain", chain()), ("truss", truss())):
    d = np.load(results_file("sym_best_%s.npz" % name)); write("best_symmetric_%s_k" % name, d["k"], d["m"], "Best design found by the symmetric multi-start search (run_sym_multistart.py, SLSQP, 200 starts), %s, stiffness only; P = %.6e W" % (name, float(d["P"])))
# pairs
for name in ("chain", "truss"):
    for pair in ("adj", "nonadj"):
        d = np.load(results_file("pairs_%s_%s.npz" % (name, pair))); write("pairs_%s_%s" % (name, pair), d["k"], d["m"], "Section 4.3, %s, %s pair; P = %.6e W" % (name, pair, float(d["P"])))
        np.savetxt(os.path.join(SUPP, "pairs_%s_%s_load.csv" % (name, pair)), d["f"], header="load vector on the free degrees of freedom (N)", comments="# ")
# escape sequences
for mode in ("k", "m", "km"):
    a = np.load(results_file("escape_seq_%s.npy" % mode), allow_pickle=True).item()
    for name in ("chain", "truss"):
        for which, fam in ((0, "optimized"), (1, "explicit")):
            for j, dd in enumerate(a[name][which]):
                if mode == "km" and S[j] > 1e3: continue
                write("escape_%s_%s_%s_s%g" % (mode, name, fam, S[j]), dd["k"], dd["m"], "Section 4.4, %s, variables %s, %s design, s = %g; P = %.6e W" % (name, mode, fam, S[j], float(dd["P"])))
print(len([f for f in os.listdir(SUPP) if f.endswith(".csv")]), "CSV files written to", SUPP)
