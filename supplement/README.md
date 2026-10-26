# Supplementary material: final designs and verification scripts

Each `*.csv` file holds one design: one row per spring (chain, 12 springs, index 0 to 11 from the left support) or bar
(truss, 51 bars in the order stated in Section 4.1: lower chord, upper chord, verticals, then the two diagonals of each
panel) with its stiffness in N/m, and one row per mass (chain: 11 masses from the left; truss: 22 nodes, lower row then
upper row, from the left) with its value in kg. The first comment line names the figure or table the design belongs to
and gives its power at 300 rad/s.

- `table1_*.csv`: the six designs of Table 1 (stiffness only, masses only, both) and `disp_*.csv`, the designs of the displacement objectives.
- `best_symmetric_*_k.csv`: the lowest-power designs of the search over mirror-symmetric designs (Section 4.1), which are
  the stiffness-only designs of Table 1.
- `pairs_*.csv` and `pairs_*_load.csv`: the four designs of Section 4.3 and their load vectors.
- `escape_<variables>_<system>_<optimized|explicit>_s<level>.csv`: the designs of the escape sequences of Section 4.4.

Scripts (Python 3, NumPy, SciPy, mpmath): `models.py` builds the two systems from the description of Section 4.1;
`supplement_check.py` recomputes, from the CSV files, the direct power, the modal sum, the total participation and the
excited gap of every design (`python supplement_check.py 'table1*'`); `verify_extreme.py` performs the checks of
Table 2 in double precision and in 40-digit arithmetic (it reads the escape-sequence results of the full code package,
`results/escape_seq_*.npy`, whose designs are the `escape_*.csv` files here); `run_symmetric.py` (reduced symmetric model) and `sqp_driver.py` (the SLSQP driver, random starts and multi-start) are imported by the
multi-start script; `run_sym_multistart.py` is the symmetric multi-start search; `second_order.py` is the second-order check (projected Hessian of the
Lagrangian) of the reported designs; `run_disp.py` runs the displacement objectives of Section 4.2.

## Details of the search and of the checks (Section 4.1)

- Budgets: explicit equalities sum(k)/K0 - 1 = 0 and sum(m)/M0 - 1 = 0 in SLSQP (satisfied to 1e-12), then a final exact rescaling of each family.
- First-order residuals: computed for log P with the multipliers of the budgets fitted by least squares on the variables farther than 0.1% from a bound (`residuals_all.py`, log `residuals_all.log`); below 4e-7 of the gradient scale in every constrained design of Table 1, of the relaxation sweeps and of the pair examples; one relaxation level stopped at the iteration limit (3000) with a residual of 4e-9.
- Symmetric start for the stiffness-only point-force problems: `run_sym_multistart.py`, 200 log-uniform random starts in the reduced space of the 6 stiffness pairs of the chain and of the 25 pairs plus the central vertical of the truss, SLSQP with ftol 1e-12 and 500 iterations, best design polished in the full space (ftol 1e-14, 3000 iterations) and included among the starts of `run_cases.py`.
- Groups of distinct final powers: two values counted as distinct when they differ by more than 0.2%; different designs of nearly equal power are counted once.
- Second-order check (`second_order.py`, log `second_order.log`): multipliers of the active bounds read from the reduced gradient; Hessian of the Lagrangian in the logarithms of the variables (including the constraint-curvature term) by central differences of the adjoint gradient, projected on the subspace in which the budget and the active bounds are fixed; strict complementarity assessed by the ratio of the first-order residual to the smallest bound multiplier.
- Local stiffness of the mass-removal limit (Section 4.4): chain, k5 + k6 = 2.2502e5 N/m; truss, Schur complement K_yy - K_xy^2/K_xx = 2.2096e5 N/m of the 2x2 stiffness block of the loaded node (K_yy = 2.2105e5 N/m, the vertical bar plus half of each diagonal, differs by 0.04%).
- `classical_damping.py` (log `classical_damping.log`): checks of the remark on classical damping in Section 2.2 (Caughey series and prescribed modal ratios reproduce the direct power) and of the dashpot counterexample of Section 5 (a lower bound on the damping matrix does not restore the band inequality).
