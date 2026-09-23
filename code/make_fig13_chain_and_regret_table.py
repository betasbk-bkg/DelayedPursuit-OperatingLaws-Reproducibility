#!/usr/bin/env python3
"""
v23 plan, new figure and table (design doc Sec. 6, items 2 and 3).

Fig. 13  the analytic chain orbit -> sigma_theta -> operating speed (d127).
  (a) the two channels of the margin law in units of L^2, J/L^2 = (L/R)^2 (1/2 - u)^2 + c sigma(u)^2, with the
      closed-form sigma_theta of Eq. (13); argmin u* for c = 1 (Eq. 9 as written) and for the measured weight r(u);
      the five measured optimal margins marked on the u axis.
  (b) v*(tau_tot) = u* L / tau_tot against the five measured circle optima.
Table    Nav2 tuning regret of the law against ingredient-removed rules (d132, d128b), written as markdown for the
         stage-2 build, with the selection-bias caveat attached.

Data only from DER JSON files; nothing is recomputed from simulation.
Usage:  python make_fig13_chain_and_regret_table_2026-09-14.py
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DER = os.path.join(HERE, "derivation")          # package layout
L, R = 2.0, 10.0          # d127
WIN, T_S = 0.3, (1 / 60) / 0.2


def J(name):
    return json.load(open(os.path.join(DER, name), encoding="utf-8"))


def fig13():
    d = J("d127_closed_form_chain_margin_2026-09-14.json")
    u = np.array(d["sigma_closed_deg_on_grid"]["u"])
    sig = np.radians(np.array(d["sigma_closed_deg_on_grid"]["sigma_deg"]))
    v1 = d["variants"]["coefficient_1 (Eq. 9 as written)"]
    vr = d["variants"]["coefficient_r(u) (measured, d124)"]
    bias = (L / R) ** 2 * (0.5 - u) ** 2
    J1 = bias + sig ** 2
    i1 = int(np.nanargmin(J1))
    if abs(u[i1] - v1["u_star"]) > 1e-9:
        raise SystemExit("argmin does not reproduce d127 u*")

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    a.plot(u, 1e3 * bias, color="#1f77b4", lw=1.4, label=r"lag channel $(L/R)^2(1/2-u)^2$")
    a.plot(u, 1e3 * sig ** 2, color="#d62728", lw=1.4, label=r"quantization channel $\sigma_\theta(u)^2$, Eq. (13)")
    a.plot(u, 1e3 * J1, color="k", lw=2.0, label=r"sum $J/L^2$")
    a.axvline(v1["u_star"], color="k", ls="--", lw=0.9)
    a.axvline(vr["u_star"], color="gray", ls=":", lw=0.9)
    a.annotate(r"$u^*=%.4f$" % v1["u_star"], (v1["u_star"], 1e3 * np.nanmin(J1) * 0.55), xytext=(5, 0),
               textcoords="offset points", fontsize=8)
    uo = [r["u_obs"] for r in v1["rows"]]
    a.scatter(uo, np.interp(uo, u, 1e3 * J1), marker="o", s=22, facecolors="none", edgecolors="k", zorder=4,
              label="measured optimal margins (5 latencies)")
    a.set_xlabel(r"delay margin $u = v\,\tau_{tot}/L$")
    a.set_ylabel(r"cost $/\,L^2$  ($\times10^{-3}$)")
    a.set_xlim(0.35, 1.0)          # Eq. (13) is validated on u = 0.4-0.9 (Sec. VI-F)
    m = (u >= 0.35)
    a.set_ylim(-2, 1e3 * np.nanmax(J1[m]) * 1.25)
    a.grid(alpha=0.25, lw=0.5)
    a.legend(fontsize=7, loc="upper right")
    a.set_title("(a) one margin for every latency", fontsize=9)

    tt = np.linspace(0.18, 1.72, 200)
    b.plot(tt, v1["u_star"] * L / tt, color="k", lw=1.6, label=r"$v^* = u^* L/\tau_{tot}$, closed form ($c=1$)")
    b.plot(tt, vr["u_star"] * L / tt, color="gray", ls=":", lw=1.2, label=r"with measured weight $r(u)$")
    for r in v1["rows"]:
        t_tot = r["tau"] + WIN / 2 + T_S
        b.scatter([t_tot], [r["v_obs"]], color="#d62728", s=24, zorder=4)
        b.annotate(("%+.1f%%" % (100 * r["rel"])).replace("-", "−"),(t_tot, r["v_obs"]), xytext=(6, 3), textcoords="offset points",
                   fontsize=7.5)
    b.scatter([], [], color="#d62728", s=24, label="measured RMSE-optimal speed")
    b.set_xlabel(r"total delay $\tau_{tot}$ (s)")
    b.set_ylabel(r"optimal speed $v^*$ (m/s)")
    b.grid(alpha=0.25, lw=0.5)
    b.legend(fontsize=7, loc="upper right")
    b.set_title("(b) predicted vs measured, mean |error| %.1f%%" % (100 * v1["mean_abs_rel"]), fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, "fig13_analytic_chain.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


def regret_table():
    d = J("d132_nav2_tuning_regret_converged_2026-09-23.json")
    w = J("d128b_regret_window_2026-09-14.json")["nine"]
    nb = d["nine_blocks"]
    names = [("law", "law, v = 0.3078 L/τ_tot (curvature command)"),
             ("stab_0.5", "half the plant-class stability limit, 0.5·u_c"),
             ("stab_0.85", "85% of the stability limit"),
             ("transport", "law constant, transport delay only"),
             ("heading", "heading-servo constant 0.4984"),
             ("fixed", "fixed speed %.1f m/s" % d["fixed_v"])]
    pct = lambda x: "—" if x is None else "%.1f%%" % (100 * x)
    lines = ["| speed rule | inside sweep | median regret (interp.) | median regret (bowl fit) | p90 (bowl) | max (bowl) | curves where the law is better (interp. / bowl) |",
             "|---|---|---|---|---|---|---|"]
    for k, lab in names:
        s = nb["by_rule"][k]
        if k == "law":
            pv = "—"
        else:
            p = nb["paired_vs_law"][k]
            pv = "%d/%d / %d/%d" % (p["law_better_L"], p["both_inside"], p["law_better_Q"], p["n_Q"]) if p["both_inside"] else "rule outside every sweep"
        lines.append("| %s | %d/%d | %s | %s | %s | %s | %s |" % (lab, s["inside_grid"], s["n_defined"], pct(s["median_R_L"]),
                                                                pct(s["median_R_Q"]), pct(s["p90_R_Q"]), pct(s["max_R_Q"]), pv))
    note = ("Regret = RMSE(rule speed)/RMSE(optimum) − 1 over the %d Nav2 speed–RMSE curves of the nine blocks. "
            "A single u for every curve keeps the median bowl-fit regret below 3%% only on u = %.2f–%.2f and below 5%% on "
            "%.2f–%.2f; the best constant chosen from these data is u = %.2f. Selection bias: the sweeps were registered around "
            "the law's prediction, so the law's speed sits mid-sweep (position %.2f–%.2f) on every curve, while rules whose "
            "speed falls outside a sweep drop out of the regret columns and are flattered by it."
            % (nb["rows"], w["window3"][0], w["window3"][1], w["window5"][0], w["window5"][1], w["best_u"],
               w["law_position_range"][0], w["law_position_range"][1]))
    out = os.path.join(HERE, "table_nav2_regret_2026-09-14.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n\n" + note + "\n")
    print("wrote", out)
    print("\n".join(lines))
    print(note)


if __name__ == "__main__":
    fig13()
    regret_table()
