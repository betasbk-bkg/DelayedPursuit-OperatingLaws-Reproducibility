#!/usr/bin/env python3
"""
d124 -- where the +9% / +16% error of the analytic smooth margin (Eq. (9), d5) comes from.

d5 minimises J(v) = (L/R)^2 (L/2 - v tau_tot)^2 + L^2 sigma_th(v)^2 on v in [0.75, 4.5],
with sigma_th(v) measured in closed loop (4 seeds) and fitted by ONE global cubic in v.
Reading its code and output: at tau = 1.5 the minimum sits at the first grid point
(i = 0), where no refinement is applied, so v_pred = 0.75 is the grid edge; and the
measured sigma_th(v) is not smooth (e.g. tau = 1.0: 8.0 deg at v = 1.5, 10.8 at 2.25).

Tests, in order:
 (A) d5's stored sigma table: reproduce its predictions, flag edge minima, and show
     how the prediction moves with the sigma fit (global cubic as in d5, quadratic,
     piecewise-linear interpolation).
 (B) New closed-loop measurement in the same loop as d5 (engine Circle, gen_votes,
     smoothing, delay), speeds 0.30-4.50 step 0.15, the five taus, 16 seeds: mean and
     standard error of sigma_th, and in the same runs the tracking RMSE (distance to the
     circle, |pos| - R).
 (C) Eq. (9) margin from the new sigma on the extended grid (argmin of J with a local
     quadratic refinement; edge flagged), for the same three fits.
 (D) The additivity assumption itself: argmin of the directly measured RMSE(v) against
     the published optimum (does this reduced loop reproduce the engine's optimum?)
     and against argmin J(v); and the ratio (RMSE^2 - bias^2) / (L^2 sigma^2) along v.

Usage:  python d124_margin_error_diagnosis_2026-09-14.py
"""
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("d5", os.path.join(HERE, "d5_smooth_margin_2026-08-29.py"))
d5 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(d5)
sm = d5.sm
L, R, WIN, T_S = d5.L, d5.R, d5.WIN, d5.T_S
OBS = d5.OBS
TAUS = [0.1, 0.433, 0.6, 1.0, 1.5]


def run(speed, tau_s, seed):
    """one closed-loop run of the d5 loop; returns (std of aggregate-ideal angle, RMSE, RMSE after 5 s)."""
    delay_f = int(round(tau_s / sm.DT))
    traj = sm.Circle(R)
    rng = np.random.default_rng(4000 + seed)
    pos = traj.start()
    vel = np.zeros(2)
    hist = [pos.copy()]
    prev = 0.0
    cur = np.array([1.0, 0.0])
    errs, dist = [], []
    for f in range(d5.FRAMES):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            _, arc = traj.closest(dp)
            d = traj.at(arc + sm.LOOK) - dp
            nrm = np.linalg.norm(d)
            if nrm > 1e-10:
                d = d / nrm
            ia = math.degrees(math.atan2(d[1], d[0]))
            votes = sm.gen_votes(ia, prev, d5.TROLL, d5.N_AG, rng)
            prev = ia
            bl = sm.DIRS[votes].mean(axis=0)
            g = np.linalg.norm(bl)
            if g > 1e-10:
                cur = bl / g
            aa = math.degrees(math.atan2(cur[1], cur[0]))
            errs.append((aa - ia + 180) % 360 - 180)
        vel = vel + sm.SMOOTH * (cur * speed - vel)
        pos = pos + vel * sm.DT
        hist.append(pos.copy())
        dist.append(abs(np.linalg.norm(pos) - R))
    dist = np.array(dist)
    k5 = int(5.0 / sm.DT)
    return float(np.std(errs)), float(np.sqrt(np.mean(dist ** 2))), float(np.sqrt(np.mean(dist[k5:] ** 2)))


def fit(speeds, sig_rad, kind):
    if kind == "cubic":
        c = np.polyfit(speeds, sig_rad, 3)
        return lambda v: float(np.polyval(c, v))
    if kind == "quadratic":
        c = np.polyfit(speeds, sig_rad, 2)
        return lambda v: float(np.polyval(c, v))
    return lambda v: float(np.interp(v, speeds, sig_rad))


def argmin_J(tau, vgrid, sig):
    tau_tot = tau + WIN / 2 + T_S
    J = np.array([(L / R) ** 2 * (L / 2 - v * tau_tot) ** 2 + L ** 2 * sig(v) ** 2 for v in vgrid])
    i = int(np.argmin(J))
    v = float(vgrid[i])
    edge = i == 0 or i == len(vgrid) - 1
    if not edge:
        c = np.polyfit(vgrid[i - 1:i + 2], J[i - 1:i + 2], 2)
        if c[0] > 0:
            vv = -c[1] / (2 * c[0])
            if vgrid[i - 1] <= vv <= vgrid[i + 1]:
                v = vv
    return v, edge


def argmin_curve(v, y):
    i = int(np.argmin(y))
    edge = i == 0 or i == len(v) - 1
    vv = float(v[i])
    if not edge:
        c = np.polyfit(v[i - 1:i + 2], y[i - 1:i + 2], 2)
        if c[0] > 0:
            x = -c[1] / (2 * c[0])
            if v[i - 1] <= x <= v[i + 1]:
                vv = float(x)
    return vv, edge


def rel(v_obs, v_pred):
    return (v_obs - v_pred) / v_pred          # manuscript convention: (measured - predicted)/predicted


def main():
    t0 = time.time()
    out = {"A": [], "B": {}, "C": [], "D": []}
    # ---------------------------------------------------------------- (A)
    old = json.load(open(os.path.join(HERE, "d5_results_2026-08-29.json"), encoding="utf-8"))
    sp_old = np.array(old["speeds"])
    S_old = np.radians(np.array(old["sigma_deg"]))
    vg_old = np.linspace(sp_old[0], sp_old[-1], 400)
    print("(A) d5's sigma table, grid 0.75-4.5")
    for j, tau in enumerate(TAUS):
        row = {"tau": tau, "v_obs": OBS[tau]}
        for kind in ("cubic", "quadratic", "linear"):
            v, edge = argmin_J(tau, vg_old, fit(sp_old, S_old[:, j], kind))
            row[kind] = {"v_pred": v, "edge": edge, "rel": rel(OBS[tau], v)}
        out["A"].append(row)
        print("  tau=%.3f obs %.3f | cubic %.3f%s (%+.1f%%) | quad %.3f%s (%+.1f%%) | linear %.3f%s (%+.1f%%)"
              % (tau, OBS[tau], row["cubic"]["v_pred"], " EDGE" if row["cubic"]["edge"] else "", 100 * row["cubic"]["rel"],
                 row["quadratic"]["v_pred"], " EDGE" if row["quadratic"]["edge"] else "", 100 * row["quadratic"]["rel"],
                 row["linear"]["v_pred"], " EDGE" if row["linear"]["edge"] else "", 100 * row["linear"]["rel"]), flush=True)
    # ---------------------------------------------------------------- (B)
    speeds = np.round(np.arange(0.30, 4.5001, 0.15), 3)
    seeds = 16
    SIG = np.zeros((len(speeds), len(TAUS)))
    SIG_SE = np.zeros_like(SIG)
    RM = np.zeros_like(SIG)
    RM_SE = np.zeros_like(SIG)
    RM5 = np.zeros_like(SIG)
    print("(B) measuring: %d speeds x %d taus x %d seeds" % (len(speeds), len(TAUS), seeds), flush=True)
    for j, tau in enumerate(TAUS):
        for i, v in enumerate(speeds):
            r = np.array([run(float(v), tau, s) for s in range(seeds)])
            SIG[i, j], SIG_SE[i, j] = r[:, 0].mean(), r[:, 0].std(ddof=1) / math.sqrt(seeds)
            RM[i, j], RM_SE[i, j] = r[:, 1].mean(), r[:, 1].std(ddof=1) / math.sqrt(seeds)
            RM5[i, j] = r[:, 2].mean()
        print("  tau=%.3f done (%.0f s)" % (tau, time.time() - t0), flush=True)
        out["B"] = {"speeds": speeds.tolist(), "taus": TAUS, "seeds": seeds, "sigma_deg": SIG.tolist(),
                    "sigma_se_deg": SIG_SE.tolist(), "rmse": RM.tolist(), "rmse_se": RM_SE.tolist(), "rmse_after5s": RM5.tolist()}
        json.dump(out, open(os.path.join(HERE, "d124_margin_error_diagnosis_2026-09-14.json"), "w"), indent=1)
    # ---------------------------------------------------------------- (C) and (D)
    vg = np.linspace(speeds[0], speeds[-1], 800)
    print("(C) Eq. (9) margin from the new sigma, grid %.2f-%.2f; (D) direct RMSE argmin" % (speeds[0], speeds[-1]))
    for j, tau in enumerate(TAUS):
        tau_tot = tau + WIN / 2 + T_S
        sig_rad = np.radians(SIG[:, j])
        rowc = {"tau": tau, "v_obs": OBS[tau]}
        for kind in ("cubic", "quadratic", "linear"):
            v, edge = argmin_J(tau, vg, fit(speeds, sig_rad, kind))
            rowc[kind] = {"v_pred": v, "edge": edge, "rel": rel(OBS[tau], v)}
        out["C"].append(rowc)
        vd, ed = argmin_curve(speeds, RM[:, j])
        vd5, ed5 = argmin_curve(speeds, RM5[:, j])
        bias2 = (L / R) ** 2 * (L / 2 - speeds * tau_tot) ** 2
        Jm = bias2 + L ** 2 * sig_rad ** 2
        vJ, eJ = argmin_curve(speeds, Jm)
        ratio = (RM[:, j] ** 2 - bias2) / (L ** 2 * sig_rad ** 2)
        rowd = {"tau": tau, "v_obs": OBS[tau], "v_direct_rmse": vd, "edge_direct": ed, "v_direct_rmse_after5s": vd5,
                "v_argmin_J_measured_sigma": vJ, "edge_J": eJ,
                "direct_vs_obs_rel": (OBS[tau] - vd) / vd, "J_vs_direct_rel": (vd - vJ) / vJ,
                "ratio_rmse2_minus_bias2_over_L2sigma2": ratio.tolist()}
        out["D"].append(rowd)
        print("  tau=%.3f obs %.3f | Eq.(9) cubic %.3f%s (%+.1f%%), quad %.3f%s, linear %.3f%s | direct RMSE argmin %.3f%s "
              "(obs vs direct %+.1f%%) | argmin J(measured sigma) %.3f%s (direct vs J %+.1f%%) | ratio at v_direct %.2f"
              % (tau, OBS[tau], rowc["cubic"]["v_pred"], " EDGE" if rowc["cubic"]["edge"] else "", 100 * rowc["cubic"]["rel"],
                 rowc["quadratic"]["v_pred"], " EDGE" if rowc["quadratic"]["edge"] else "",
                 rowc["linear"]["v_pred"], " EDGE" if rowc["linear"]["edge"] else "",
                 vd, " EDGE" if ed else "", 100 * rowd["direct_vs_obs_rel"], vJ, " EDGE" if eJ else "",
                 100 * rowd["J_vs_direct_rel"], float(np.interp(vd, speeds, ratio))), flush=True)
    json.dump(out, open(os.path.join(HERE, "d124_margin_error_diagnosis_2026-09-14.json"), "w"), indent=1)
    print("%.0f s -> d124_margin_error_diagnosis_2026-09-14.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
