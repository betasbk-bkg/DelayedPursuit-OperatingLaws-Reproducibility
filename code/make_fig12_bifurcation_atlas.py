#!/usr/bin/env python3
"""
Fig. 12 (v23 plan) -- bifurcation atlas of the quantized loop in the (u, q) plane.

Data (all existing, DER = derivation):
  d119b (part B)  grazing death of W_n, n = 1-8, u = 0.05-0.95, from the exact death condition
                  (no q limit)
  d119c           birth of W_1-W_3, u = 0.20-0.95: at the border (short-excursion or end-window)
                  or at a smooth saddle-node just below it (then the fold q_F is drawn)
  d116b           birth of W_4, W_5 (its q scan stopped at q = 6, so these two curves end early;
                  said in the caption)
  d119d           codimension-two border-fold points on the short-excursion border (n = 1-4) and
                  the W_1 switch back on the end-window border
Closed forms: q_bc(u) = (1 + sqrt u)^2 (death of W_0), q_lo(u) (short-excursion border of W_1).

Usage:  python make_fig12_bifurcation_atlas_2026-09-14.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DER = os.path.join(HERE, "derivation")          # package layout
COL = {0: "#4C4C4C", 1: "#1f77b4", 2: "#d62728", 3: "#2ca02c", 4: "#9467bd", 5: "#8c564b"}


def J(name):
    return json.load(open(os.path.join(DER, name), encoding="utf-8"))


def main():
    deaths = J("d119b_birth_type_by_S_2026-09-14.json")["death"]
    bt = [r for r in J("d119c_birth_border_2d_2026-09-14.json") if r.get("status") == "ok"]
    bd = J("d116b_word_borders_2026-09-13.json")
    cd = J("d119d_birth_multiplier_closed_form_2026-09-14.json")

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    us = np.linspace(0.02, 0.98, 300)
    ax.plot(us, (1 + np.sqrt(us)) ** 2, color=COL[0], lw=1.8, label=r"$W_0=(+)$ grazing death, $q_{bc}=(1+\sqrt{u})^2$")
    ok = 1 - (2 * us - 1) ** 2 + us ** 2 / 4 >= 0
    ax.plot(us[ok], 1 + np.sqrt(1 - (2 * us[ok] - 1) ** 2 + us[ok] ** 2 / 4), color=COL[1], lw=0.9, ls=":",
            label=r"$W_1$ short-excursion border $q_{lo}(u)$ (closed form)")

    for n in range(1, 6):
        dd = sorted(((r["u"], r["q_death"]) for r in deaths if r["n"] == n), key=lambda x: x[0])
        if n <= 3:
            bb = sorted(((r["u"], r["q_F"] if (r["type"] == "fold" and "q_F" in r) else r["q_B"]) for r in bt if r["n"] == n),
                        key=lambda x: x[0])
        else:
            bb = sorted(((r["u"], r["birth"]) for r in bd if r["n"] == n and r["exists"] and r["birth"] is not None),
                        key=lambda x: x[0])
        if dd:
            ax.plot([x[0] for x in dd], [x[1] for x in dd], color=COL[n], lw=1.7,
                    label=r"$W_%d$ grazing death (solid), birth (dashed)" % n)
        if bb:
            ax.plot([x[0] for x in bb], [x[1] for x in bb], color=COL[n], lw=1.2, ls="--")
        if dd and bb:
            ug = np.linspace(max(dd[0][0], bb[0][0]), min(dd[-1][0], bb[-1][0]), 200)
            lo = np.interp(ug, [x[0] for x in bb], [x[1] for x in bb])
            hi = np.interp(ug, [x[0] for x in dd], [x[1] for x in dd])
            ax.fill_between(ug, lo, hi, where=hi > lo, color=COL[n], alpha=0.07, lw=0)

    for typ, mk, lab in (("border", "o", "birth at the border (border collision)"),
                         ("fold", "^", "birth at a saddle-node just below the border")):
        pts = [r for r in bt if r["type"] == typ]
        ax.scatter([r["u"] for r in pts], [r["q_F"] if (typ == "fold" and "q_F" in r) else r["q_B"] for r in pts],
                   s=13, marker=mk, facecolors="none", edgecolors="k", linewidths=0.6, label=lab, zorder=4)

    sw = [r for r in cd["c"] if r.get("status") == "ok"]
    ax.scatter([r["u_switch"] for r in sw], [r["q_B"] for r in sw], s=70, marker="*", color="orange",
               edgecolors="k", linewidths=0.6, zorder=5, label="border-fold points (codimension 2)")
    for r in sw:
        txt = "4/7" if r["n"] == 1 else "%.4f" % r["u_switch"]
        ax.annotate(r"$W_%d$: $u=$%s" % (r["n"], txt), (r["u_switch"], r["q_B"]), xytext=(6, -11),
                    textcoords="offset points", fontsize=7.5)
    d = cd.get("d", {})
    if d.get("status") == "ok":
        ax.scatter([d["u_switch"]], [d["q_B"]], s=48, marker="D", color="orange", edgecolors="k", linewidths=0.6,
                   zorder=5, label=r"$W_1$ switch back on the end-window border ($u=0.8345$)")

    ax.set_xlabel(r"delay margin $u = v\,\tau_{tot}/L$")
    ax.set_ylabel(r"$q = k\Delta/2$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 12.0)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    ax.set_title(r"Periodic words $W_n=(-,+)^n,+$ of the quantized loop: existence, birth type, death", fontsize=10)
    fig.tight_layout()
    out = os.path.join(HERE, "fig12_bifurcation_atlas.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
