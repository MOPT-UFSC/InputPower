"""Recount of the groups of distinct final values of the power among the starts (column 'groups' of Table 1 and the
counts quoted in Section 4) from the saved all_P arrays, with the rule of models.count_groups: values sorted, the first
value of each group is its reference, the following values join it while P / P_ref - 1 <= 0.2%. The earlier rule used
in the original campaign logs (rounding log10 P to three decimals) is printed alongside for the record.
Usage: python recount_groups.py   (reads code/results/, writes nothing)"""
import numpy as np
from models import count_groups, results_file

def old_rule(allP): return len(set(np.round(np.log10(allP), 3)))

print("%-28s %6s %10s %10s" % ("design", "starts", "log10-round", "0.2% rule"))
for name in ("chain", "truss"):
    for vs, lab in (("k", "stiffness"), ("m", "masses"), ("km", "stiffness, masses")):
        a = np.load(results_file("%s_%s.npz" % (name, vs)))["all_P"]; print("%-28s %6d %10d %10d" % ("Table 1 %s %s" % (name, lab), len(a), old_rule(a), count_groups(a)))
for name in ("chain", "truss"):
    for pair in ("adj", "nonadj"):
        a = np.load(results_file("pairs_%s_%s.npz" % (name, pair)))["all_P"]; print("%-28s %6d %10d %10d" % ("pairs %s %s" % (name, pair), len(a), old_rule(a), count_groups(a)))
for lab in ("u6", "u9"):
    a = np.load(results_file("disp_%s.npz" % lab))["all_disp"]; print("%-28s %6d %10d %10d" % ("chain min |%s| (on |u|)" % lab, len(a), old_rule(a), count_groups(a)))
for name in ("chain", "truss"):
    a = np.load(results_file("sym_best_%s.npz" % name))["all_P"]; print("%-28s %6d %10d %10d" % ("symmetric search %s" % name, len(a), old_rule(a), count_groups(a)))
    a = np.load(results_file("fullspace_%s.npy" % name)); print("%-28s %6d %10d %10d" % ("300 full-space starts %s" % name, len(a), old_rule(a), count_groups(a)))
