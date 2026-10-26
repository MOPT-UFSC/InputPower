"""Recomputes, from the CSV files of the supplement, the power (direct solution and modal sum), the total participation
and the excited gap (threshold 1e-2 on p_r/mu_f, participation summed over numerically repeated eigenvalues) of every
design. Usage: python supplement_check.py [pattern]"""
import sys, glob, os, numpy as np
from scipy.linalg import eigh
from models import chain, truss, excited_gap, ETA, W0
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
HERE = os.path.dirname(os.path.abspath(__file__))
SUPP = next((d for d in ("supplement", os.path.join(HERE, "..", "supplement"), HERE) if glob.glob(os.path.join(d, "*.csv"))), None)   # CSV directory
if SUPP is None:
    sys.exit("no CSV files found: the designs of the supplement are in supplement/data_.zip; run, from the repository root,\n    python -m zipfile -e supplement/data_.zip supplement")
for fn in sorted(glob.glob(os.path.join(SUPP, "%s.csv" % pat))):
    if fn.endswith("_load.csv"): continue
    rows = [l.split(",") for l in open(fn) if not l.startswith("#")]
    k = np.array([float(r[1]) for r in rows if r[1] != ""]); m = np.array([float(r[2].strip()) for r in rows if r[2].strip() != ""])
    name = "chain" if "chain" in fn else "truss"; mdl = chain() if name == "chain" else truss()
    f = mdl["f"]
    if "pairs" in fn: f = np.loadtxt(fn.replace(".csv", "_load.csv"))
    K = mdl["assemble_K"](k); M = mdl["assemble_M"](m); u = np.linalg.solve(K * (1 + 1j * ETA) - W0**2 * M, f)
    Pd = 0.5 * np.real(f @ (1j * W0 * u)); lam, Phi = eigh(K, M); p = (Phi.T @ f)**2; mu = p.sum(); g = W0 * ETA * lam / ((lam - W0**2)**2 + (ETA * lam)**2); Pm = 0.5 * (p * g).sum()
    lo, hi = excited_gap(np.sqrt(lam), p, mu)   # participation summed over clusters of repeated eigenvalues (models.py)
    print("%-45s P_direct %.4e  |modal-direct|/P %.1e  mu %.4f  gap [%.0f, %.0f]" % (fn.split("/")[-1], Pd, abs(Pm - Pd) / Pd, mu, lo, hi))
