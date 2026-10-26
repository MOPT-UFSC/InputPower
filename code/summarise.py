"""Tabulate results/*.npz: final P, P0/P, total participation, excited gap, iterations, number of increases of P
in the history (0 for the line-search driver), final design change. Usage: python summarise.py [prefix]"""
import glob, os, sys
import numpy as np
from models import results_dir

prefix = sys.argv[1] if len(sys.argv) > 1 else ""
files = sorted(glob.glob(os.path.join(results_dir(), prefix + "*.npz")))
print("%-26s %11s %8s %7s %18s %6s %5s %9s" % ("case", "P_final", "P0/P", "mu", "gap [rad/s]", "iters", "incr", "last chg"))
for fn in files:
    d = np.load(fn); h = d["hist"]; Ph = h[:, 1]
    inc = int(np.sum(np.diff(Ph[:-1]) > 0))
    print("%-26s %11.3e %8.1f %7.3f [%7.1f, %7.1f] %6d %5d %9.1e" % (os.path.basename(fn)[:-4], float(d["P"]), Ph[0] / float(d["P"]), float(d["mu"]), float(d["lo"]), float(d["hi"]), len(h) - 1, inc, h[-2, 2]))
