"""Generate the pgfplots figure files fig_*.tex of Section 5 from the data files data/n_*.dat (written by
mma/export_paper.py). Pure templating, no computation."""
import os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data")
LV = ["yellow!75!black", "green!60!black", "teal!80!black", "cyan!70!black", "blue!70!black", "violet!80!black"]
K_LAB = ["$k_{\\min}=10^5$", "$3\\times10^4$", "$10^4$", "$3\\times10^3$", "$10^3$", "$3\\times10^2$"]
M_LAB = {"chain": ["$[0.9,1.1]$", "$[0.5,2]$", "$[0.25,4]$", "$[0.125,8]$", "$[0.1,10]$", "$[0.1,10]$"],
         "truss": ["$[0.9,1.1]$", "$[0.5,2]$", "$[0.25,4]$", "$[0.125,8]$", "$[0.1,10]$", "$[0.1,19.9]$"]}
KM_LAB = {s: ["%s, %s" % (a.replace("$k_{\\min}=", "$"), b) for a, b in zip(K_LAB, M_LAB[s])] for s in M_LAB}
VLINE = "  \\addplot[red!70!black, dashed, forget plot] coordinates {(300,%s) (300,%s)};\n"


def levels_available(name, vs):
    n = 0
    while os.path.exists(os.path.join(D, "n_relaxmodes_%s_%s_L%d.dat" % (name, vs, n))): n += 1
    if name == "chain" and vs == "m": n = min(n, 5)   # the last two mass levels coincide on the chain (budget binds)
    return n


def relax_panel(name, vs, title, legname=None):
    labs = K_LAB if vs == "k" else (M_LAB[name] if vs == "m" else KM_LAB[name]); n = levels_available(name, vs)
    s = "\\nextgroupplot[title={%s}%s]\n" % (title, (", legend to name=%s, legend columns=4" % legname) if legname else "")
    le = (lambda t: " \\addlegendentry{%s}" % t) if legname else (lambda t: "")
    s += "  \\addplot[gray, densely dotted] table[x=w, y=P] {\\datapath n_relax_%s_%s_P0.dat};%s\n" % (name, vs, le("initial"))
    for L in range(n):
        s += "  \\addplot[%s] table[x=w, y=P] {\\datapath n_relax_%s_%s_L%d.dat};%s\n" % (LV[L], name, vs, L, le(labs[L]))
    for L in range(n):
        s += "  \\addplot[%s, only marks, mark=*, mark size=1.1pt, forget plot] table[x=wr, y expr=10^(-10.7+0.3*%d)] {\\datapath n_relaxmodes_%s_%s_L%d.dat};\n" % (LV[L], L, name, vs, L)
    s += VLINE % ("1e-11", "1e-1")
    return s


def fig_relax():
    s = "\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=2 by 3, horizontal sep=1.6cm, vertical sep=2.4cm, xlabels at=edge bottom, ylabels at=edge left}, width=0.5\\textwidth, height=0.32\\textwidth,\n"
    s += "  xmode=log, ymode=log, xmin=5, xmax=3000, ymin=1e-11, ymax=1e-1, xlabel={$\\omega$ [rad/s]}, ylabel={$P(\\omega)$ [W]},\n"
    s += "  title style={font=\\footnotesize}, legend style={font=\\scriptsize, draw=none, fill=none}]\n"
    s += relax_panel("chain", "k", "(a) chain, springs only")
    s += relax_panel("truss", "k", "(b) truss, bars only", "legk")
    s += relax_panel("chain", "m", "(c) chain, masses only")
    s += relax_panel("truss", "m", "(d) truss, masses only", "legm")
    s += relax_panel("chain", "km", "(e) chain, springs and masses")
    s += relax_panel("truss", "km", "(f) truss, bars and masses", "legkm")
    s += "\\end{groupplot}\n"
    for r, ln in ((1, "legk"), (2, "legm"), (3, "legkm")):
        s += "\\node[anchor=north] at ($(group c1r%d.south west)!0.5!(group c2r%d.south east)$) [yshift=%s] {\\pgfplotslegendfromname{%s}};\n" % (r, r, "-0.4cm" if r < 3 else "-1.0cm", ln)
    s += "\\end{tikzpicture}\n"
    open(os.path.join(HERE, "fig_relax.tex"), "w").write(s)


MODE_LAB = {"k": "stiffness only", "m": "masses only", "km": "stiffness and masses"}
SYS_LAB = {"chain": "chain", "truss": "truss"}


def fig_escape_freq():
    s = "\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=2 by 3, horizontal sep=1.6cm, vertical sep=1.5cm, xlabels at=edge bottom, ylabels at=edge left}, width=0.5\\textwidth, height=0.36\\textwidth,\n"
    s += "  xmode=log, ymode=log, xmin=0.7, xmax=1.5e5, xlabel={$s$}, ylabel={excited $\\omega_r$ [rad/s]}, title style={font=\\footnotesize},\n"
    s += "  legend style={font=\\scriptsize, draw=none, fill=none}]\n"
    for row, mode in enumerate(["k", "m", "km"]):
        for col, name in enumerate(["chain", "truss"]):
            lab = "(%s) %s, %s" % ("abcdef"[2 * row + col], name, MODE_LAB[mode])
            ymin, ymax = ("0.3", "3e5") if mode == "k" else ("0.1", "3e3")
            first = (row == 0 and col == 0)
            s += "\\nextgroupplot[title={%s}, ymin=%s, ymax=%s%s]\n" % (lab, ymin, ymax, ", legend to name=legfreq, legend columns=-1" if first else "")
            le = (lambda t: " \\addlegendentry{%s}" % t) if first else (lambda t: "")
            s += "  \\addplot[blue!70!black, only marks, mark=*, mark size=1.3pt, opacity=0.7] table[x=s, y=wr] {\\datapath n_escapemodes_%s_%s_opt.dat};%s\n" % (mode, name, le("optimized designs"))
            s += "  \\addplot[orange!80!black, only marks, mark=square*, mark size=1.1pt, opacity=0.7] table[x=s, y=wr] {\\datapath n_escapemodes_%s_%s_exp.dat};%s\n" % (mode, name, le("explicit family"))
            s += "  \\addplot[blue!70!black, thin, dashed] table[x=s, y=lo] {\\datapath n_escape_%s_%s_opt.dat};%s\n" % (mode, name, le("excited gap, optimized"))
            s += "  \\addplot[blue!70!black, thin, dashed, forget plot, unbounded coords=discard] table[x=s, y=hi] {\\datapath n_escape_%s_%s_opt.dat};\n" % (mode, name)
            s += "  \\addplot[red!70!black, dashed] coordinates {(0.7,300) (1.5e5,300)};%s\n" % le("$\\omega$")
            s += "  \\addplot[red!70!black, fill=red!70!black, fill opacity=0.07, draw=none, forget plot] coordinates {(0.7,150) (1.5e5,150) (1.5e5,600) (0.7,600)} \\closedcycle;\n"
    s += "\\end{groupplot}\n\\node[anchor=north] at ($(group c1r3.south west)!0.5!(group c2r3.south east)$) [yshift=-1.05cm] {\\pgfplotslegendfromname{legfreq}};\n\\end{tikzpicture}\n"
    open(os.path.join(HERE, "fig_escape_freq.tex"), "w").write(s)


def fig_escape_P():
    s = "\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=2 by 3, horizontal sep=1.6cm, vertical sep=1.5cm, xlabels at=edge bottom, ylabels at=edge left}, width=0.5\\textwidth, height=0.36\\textwidth,\n"
    s += "  xmode=log, ymode=log, xmin=0.7, xmax=1.5e5, xlabel={$s$}, ylabel={$P/P_0$, $\\mu_k/\\mu_0$, share in $B$}, title style={font=\\footnotesize}, unbounded coords=jump,\n"
    s += "  legend style={font=\\scriptsize, draw=none, fill=none}]\n"
    for row, mode in enumerate(["k", "m", "km"]):
        for col, name in enumerate(["chain", "truss"]):
            lab = "(%s) %s, %s" % ("abcdef"[2 * row + col], name, MODE_LAB[mode])
            ymin = "1e-10" if mode == "k" else "1e-18"; first = (mode == "km" and col == 0)
            le = (lambda t: " \\addlegendentry{%s}" % t) if first else (lambda t: "")
            s += "\\nextgroupplot[title={%s}, ymin=%s, ymax=10%s]\n" % (lab, ymin, ", legend to name=legP, legend columns=3" if first else "")
            s += "  \\addplot[blue!70!black, thick, mark=*, mark size=1.2pt] table[x=s, y=Pn] {\\datapath n_escape_%s_%s_opt.dat};%s\n" % (mode, name, le("$P/P_0$, optimized"))
            s += "  \\addplot[orange!80!black, thick, mark=square*, mark size=1.1pt] table[x=s, y=Pn] {\\datapath n_escape_%s_%s_exp.dat};%s\n" % (mode, name, le("$P/P_0$, explicit family"))
            if mode != "k":
                s += "  \\addplot[blue!70!black, dashed, mark=triangle, mark size=1.4pt] table[x=s, y=mun] {\\datapath n_escape_%s_%s_opt.dat};%s\n" % (mode, name, le("$\\mu_k/\\mu_0$, optimized"))
                s += "  \\addplot[blue!70!black, dash dot, mark=diamond, mark size=1.4pt] table[x=s, y=Pmun] {\\datapath n_escape_%s_%s_opt.dat};%s\n" % (mode, name, le("$(P/\\mu_k)/(P_0/\\mu_0)$, optimized"))
            s += "  \\addplot[black, densely dotted, mark=o, mark size=1.2pt] table[x=s, y=shareB] {\\datapath n_escape_%s_%s_opt.dat};%s\n" % (mode, name, le("share in $B$, optimized"))
            s += "  \\addplot[black, thin] table[x=s, y=bound] {\\datapath n_escape_%s_%s_opt.dat};%s\n" % (mode, name, le("bound $2P/(g_B\\mu_k)$"))
    s += "\\end{groupplot}\n\\node[anchor=north] at ($(group c1r3.south west)!0.5!(group c2r3.south east)$) [yshift=-1.05cm] {\\pgfplotslegendfromname{legP}};\n\\end{tikzpicture}\n"
    open(os.path.join(HERE, "fig_escape_P.tex"), "w").write(s)


def fig_escape_spec():
    s = "\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=2 by 3, horizontal sep=1.6cm, vertical sep=1.5cm, xlabels at=edge bottom, ylabels at=edge left}, width=0.5\\textwidth, height=0.36\\textwidth,\n"
    s += "  xmode=log, ymode=log, xmin=0.1, xmax=1e6, xlabel={$\\omega$ [rad/s]}, ylabel={$P(\\omega)$ [W]}, title style={font=\\footnotesize},\n"
    s += "  legend style={font=\\scriptsize, draw=none, fill=none}]\n"
    for row, mode in enumerate(["k", "m", "km"]):
        for col, name in enumerate(["chain", "truss"]):
            lab = "(%s) %s, %s" % ("abcdef"[2 * row + col], name, MODE_LAB[mode])
            ymin = "1e-13" if mode == "k" else "1e-21"
            first = (row == 0 and col == 0)
            s += "\\nextgroupplot[title={%s}, ymin=%s, ymax=1e2%s]\n" % (lab, ymin, ", legend to name=legspec, legend columns=-1" if first else "")
            for L in range(6):
                if not os.path.exists(os.path.join(D, "n_escapespec_%s_%s_s%d.dat" % (mode, name, L))): continue
                s += "  \\addplot[%s] table[x=w, y=P] {\\datapath n_escapespec_%s_%s_s%d.dat};%s\n" % (LV[L], mode, name, L, (" \\addlegendentry{$s=10^{%d}$}" % L) if first else "")
            s += VLINE % (ymin, "1e2")
    s += "\\end{groupplot}\n\\node[anchor=north] at ($(group c1r3.south west)!0.5!(group c2r3.south east)$) [yshift=-1.05cm] {\\pgfplotslegendfromname{legspec}};\n\\end{tikzpicture}\n"
    open(os.path.join(HERE, "fig_escape_spec.tex"), "w").write(s)


def fig_pairs():
    s = "\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=2 by 2, horizontal sep=1.6cm, vertical sep=1.5cm, xlabels at=edge bottom, ylabels at=edge left}, width=0.5\\textwidth, height=0.34\\textwidth,\n"
    s += "  ymode=log, xmin=0, xmax=1500, ymin=1e-6, ymax=1.5, xlabel={$\\omega_r$ [rad/s]}, ylabel={$p_r/\\mu_{\\bm f}$}, title style={font=\\footnotesize}, legend style={font=\\tiny, at={(0.98,0.98)}, anchor=north east}]\n"
    for lab, name, tag in [("(a) chain, adjacent pair $\\bm e_6-\\bm e_7$", "chain", "adj"), ("(b) truss, adjacent pair, nodes 16 and 17", "truss", "adj"),
                           ("(c) chain, pair $\\bm e_4-\\bm e_8$", "chain", "nonadj"), ("(d) truss, pair, nodes 14 and 18", "truss", "nonadj")]:
        s += "\\nextgroupplot[title={%s}]\n" % lab
        s += "  \\addplot[ycomb, blue!60!black, thick, mark=*, mark size=1.3pt] table[x=wr, y=share] {\\datapath n_pairs_%s_%s_modes.dat}; \\addlegendentry{final}\n" % (name, tag)
        s += "  \\addplot[red!70!black, dashed] coordinates {(300,1e-6) (300,1.5)}; \\addlegendentry{$\\omega$}\n"
    s += "\\end{groupplot}\n\\end{tikzpicture}\n"
    open(os.path.join(HERE, "fig_pairs.tex"), "w").write(s)


def fig_designs_chain():
    s = "\\pgfplotsset{shortbar/.style={legend image code/.code={\\draw[#1] (0cm,-0.07cm) rectangle (0.22cm,0.07cm);}}}\n\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.7cm}, width=0.52\\textwidth, height=0.3\\textwidth, title style={font=\\footnotesize}, legend style={font=\\scriptsize, draw=none, fill=none}]\n"
    s += "\\nextgroupplot[ybar, bar width=2.6pt, title={(a) chain springs}, xlabel={spring $j$}, ylabel={$k_j/10^5$ [N/m]}, xmin=-0.7, xmax=11.7, ymin=0, ymax=5.4, xtick={0,...,11}, legend to name=legspr, legend columns=-1]\n"
    s += "  \\addplot[shortbar, fill=gray!50, draw=gray] table[x=j, y=k_init] {\\datapath n_design_chain_k.dat}; \\addlegendentry{initial}\n"
    s += "  \\addplot[shortbar, fill=blue!60!black, draw=blue!60!black] table[x=j, y=k_opt] {\\datapath n_design_chain_k.dat}; \\addlegendentry{springs only}\n"
    s += "  \\addplot[shortbar, fill=orange!80!black, draw=orange!80!black] table[x=j, y=k_opt] {\\datapath n_design_chain_km.dat}; \\addlegendentry{springs and masses}\n"
    s += "\\nextgroupplot[ybar, bar width=2.6pt, title={(b) chain masses}, xlabel={mass $i$}, ylabel={$m_i$ [kg]}, xmin=0.3, xmax=11.7, ymin=0, ymax=5.4, xtick={1,...,11}, legend to name=legmas, legend columns=-1]\n"
    s += "  \\addplot[shortbar, fill=gray!50, draw=gray] table[x=i, y=m_init] {\\datapath n_mass_chain_m.dat}; \\addlegendentry{initial}\n"
    s += "  \\addplot[shortbar, fill=blue!60!black, draw=blue!60!black] table[x=i, y=m_opt] {\\datapath n_mass_chain_m.dat}; \\addlegendentry{masses only}\n"
    s += "  \\addplot[shortbar, fill=orange!80!black, draw=orange!80!black] table[x=i, y=m_opt] {\\datapath n_mass_chain_km.dat}; \\addlegendentry{springs and masses}\n"
    s += "\\end{groupplot}\n"
    s += "\\node[anchor=north] at (group c1r1.below south) [yshift=-0.05cm] {\\pgfplotslegendfromname{legspr}};\n"
    s += "\\node[anchor=north] at (group c2r1.below south) [yshift=-0.05cm] {\\pgfplotslegendfromname{legmas}};\n"
    s += "\\end{tikzpicture}\n"
    open(os.path.join(HERE, "fig_designs_chain.tex"), "w").write(s)


def fig_decomp():
    s = "\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=2 by 2, horizontal sep=1.6cm, vertical sep=1.5cm, xlabels at=edge bottom, ylabels at=edge left}, width=0.5\\textwidth, height=0.34\\textwidth, xmin=20, xmax=700, xlabel={$\\omega$ [rad/s]}, title style={font=\\footnotesize}, legend style={font=\\tiny, draw=none, fill=none}]\n"
    s += "\\nextgroupplot[title={(a) chain, hysteretic damping}, ymode=log, ymin=1e-12, ymax=1e0, ylabel={$P$, $P_r$ [W]}, legend pos=north west, legend columns=3]\n"
    for r in range(1, 12): s += "  \\addplot[gray!60, thin, forget plot] table[x=w, y=P%d] {\\datapath ex1_power.dat};\n" % r
    s += "  \\addplot[black, thick] table[x=w, y=P] {\\datapath ex1_power.dat}; \\addlegendentry{$P$}\n"
    s += "  \\addplot[red!70!black, dashed, thick] coordinates {(300,1e-12) (300,1e-1)}; \\addlegendentry{$\\omega$}\n"
    s += "  \\addplot[gray!60, thin] coordinates {(0,1e-20) (0,1e-20)}; \\addlegendentry{$P_r$}\n"
    s += "\\nextgroupplot[title={(b) truss, hysteretic damping}, ymode=log, ymin=1e-12, ymax=1e0, legend pos=north west, legend columns=3]\n"
    for r in range(1, 37): s += "  \\addplot[gray!60, thin, forget plot] table[x=w, y=P%d] {\\datapath n_terms_truss.dat};\n" % r
    s += "  \\addplot[black, thick] table[x=w, y=P] {\\datapath n_terms_truss.dat}; \\addlegendentry{$P$}\n"
    s += "  \\addplot[red!70!black, dashed, thick] coordinates {(300,1e-12) (300,1e-1)}; \\addlegendentry{$\\omega$}\n"
    s += "  \\addplot[gray!60, thin] coordinates {(0,1e-20) (0,1e-20)}; \\addlegendentry{$P_r$}\n"
    s += "\\nextgroupplot[title={(c) chain, dashpot on $m_9$}, ymin=-400, ymax=400, ylabel={$P_r/P$}, legend pos=south east]\n"
    s += "  \\addplot[blue!70!black, thick] table[x=w, y expr=\\thisrow{Prmin}/\\thisrow{P}] {\\datapath n_nonprop_chain.dat}; \\addlegendentry{$\\min_r P_r/P$}\n"
    s += "  \\addplot[orange!80!black, thick] table[x=w, y expr=\\thisrow{Prmax}/\\thisrow{P}] {\\datapath n_nonprop_chain.dat}; \\addlegendentry{$\\max_r P_r/P$}\n"
    s += "  \\addplot[black, dashed, forget plot] coordinates {(20,0) (700,0)};\n"
    s += "\\nextgroupplot[title={(d) truss, dashpot at node 13}, ymin=-1600, ymax=4800, legend pos=north east]\n"
    s += "  \\addplot[blue!70!black, thick] table[x=w, y expr=\\thisrow{Prmin}/\\thisrow{P}] {\\datapath n_nonprop_truss.dat}; \\addlegendentry{$\\min_r P_r/P$}\n"
    s += "  \\addplot[orange!80!black, thick] table[x=w, y expr=\\thisrow{Prmax}/\\thisrow{P}] {\\datapath n_nonprop_truss.dat}; \\addlegendentry{$\\max_r P_r/P$}\n"
    s += "  \\addplot[black, dashed, forget plot] coordinates {(20,0) (700,0)};\n"
    s += "\\end{groupplot}\n\\end{tikzpicture}\n"
    open(os.path.join(HERE, "fig_decomp.tex"), "w").write(s)


if __name__ == "__main__":
    fig_relax(); fig_escape_freq(); fig_escape_P(); fig_escape_spec(); fig_decomp(); fig_designs_chain()
    if os.path.exists(os.path.join(D, "n_pairs_chain_adj_modes.dat")): fig_pairs()
    print("figures written")
