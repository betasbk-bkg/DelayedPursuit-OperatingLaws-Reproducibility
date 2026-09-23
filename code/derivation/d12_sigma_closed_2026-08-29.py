# -*- coding: utf-8 -*-
"""
Derivation note -- D12: sigma_theta(u) with no simulation of the crowd and
no simulation of the vehicle  (NEXT_c_and_mc plan, acceptance criterion 3).

Chain now available end to end:
    vote composition  --D9-->  b(m), s(m)         closed form
    b(m) + (k, u)     --D10--> rho(m)             delayed cylinder flow
    rho + b, s        --here-> sigma_theta(k, u)  the constant of Secs. IV-D/F

The residual spread s(m) is closed here as well.  Each agent contributes a unit
vector on the grid; with cell probabilities p_k (D9) the sample mean of N agents
has transverse variance Var_perp/N about its own direction, so

    s(m) = sqrt( Var_perp / N ) / |mu| ,
    Var_perp = sum_k p_k (e_k . n)^2 - (mu . n)^2 ,   n perpendicular to mu.

Test: the resulting attenuation sigma_theta(u)/sigma_open against the measured
table of D8 (0.908, 0.864, 0.800, 0.790, 0.754 at u = 0.40 ... 0.80).

Outputs: d12_results_2026-08-29.json
"""
import importlib.util, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


d9 = _load("d9", "d9_mc_law_2026-08-29.py")
d10 = _load("d10", "d10_delayed_flow_2026-08-29.py")
DELTA, N_AG = 45.0, 150


def moments(m, a_acc=3.0, n_dirs=8, N=N_AG, troll=0.05, a_other=30.0):
    """mean vector and transverse variance of one agent's grid vote."""
    delta = 360.0/n_dirs
    n_acc, n_slow, n_other, n_troll = d9.class_counts(N, troll)
    ks_all, ps_all = [], []
    for cnt, a in ((n_acc, a_acc), (n_other, a_other)):
        ks, p = d9.uniform_cell_mass(m, a, delta)
        ks_all.append(ks); ps_all.append(p*cnt/N)
    kk = round(m/delta)
    ks_all.append(np.array([kk])); ps_all.append(np.array([n_slow/N]))
    ks = np.arange(-n_dirs, n_dirs+1)
    ps = np.zeros(len(ks))
    for K_, P_ in zip(ks_all, ps_all):
        for kv, pv in zip(K_, P_):
            ps[np.searchsorted(ks, kv)] += pv
    # trolls: uniform over the grid
    ps += (n_troll/N)/n_dirs*np.isin(ks, np.arange(n_dirs)).astype(float)
    ang = np.radians(ks*delta)
    ex, ey = float(np.sum(ps*np.cos(ang))), float(np.sum(ps*np.sin(ang)))
    mu = math.hypot(ex, ey)
    if mu < 1e-9:
        return 0.0, 0.0, 0.0
    nx, ny = -ey/mu, ex/mu
    perp = ps*(np.cos(ang)*nx + np.sin(ang)*ny)**2
    var_perp = float(np.sum(perp))
    return math.degrees(math.atan2(ey, ex)) - m, mu, var_perp


def b_s_closed(m, **kw):
    b, mu, vp = moments(m, **kw)
    b = (b + 180.0) % 360.0 - 180.0
    s = math.degrees(math.sqrt(max(vp, 0.0)/N_AG)/mu) if mu > 0 else 0.0
    return b, s


if __name__ == "__main__":
    out = {}
    print("[A] closed-form residual spread s(m) vs the engine measurement")
    d3 = json.load(open(os.path.join(HERE, "d3_results_2026-08-29.json")))["transfer"]
    ms, sm = np.array(d3["m"]), np.array(d3["s"])
    sc = np.array([b_s_closed(m)[1] for m in ms])
    print(f"    max |diff| = {np.abs(sc-sm).max():.3f} deg,  rms = "
          f"{np.sqrt(np.mean((sc-sm)**2)):.3f} deg")
    for pr in [0.0, 10.0, 19.0, 22.4]:
        print(f"      m={pr:5.1f}   measured {np.interp(pr, ms, sm):5.2f}   "
              f"closed {b_s_closed(pr)[1]:5.2f}")
    out["s_validation"] = dict(max_diff=float(np.abs(sc-sm).max()),
                               rms=float(np.sqrt(np.mean((sc-sm)**2))))

    grid = d10.b_table()
    mg = np.linspace(-DELTA/2, DELTA/2, 361)
    BS = np.array([b_s_closed(m) for m in mg])
    B2S2 = BS[:, 0]**2 + BS[:, 1]**2
    sig_open = float(np.sqrt(np.mean(B2S2)))
    print(f"\n[B] open-loop uniform sweep from the closed form: {sig_open:.3f} deg "
          f"(engine MC 10.95, manuscript 11.06)")
    out["sigma_open_closed"] = sig_open

    print("\n[C] sigma_theta(u) from the derived density -- no agents, no vehicle sim")
    d8 = {r["u"]: r for r in json.load(
        open(os.path.join(HERE, "d8_results_2026-08-29.json")))["sigma_vs_u"]}
    print(f"{'u':>6} {'sigma_pred':>11} {'sigma_meas':>11} {'atten_pred':>11} "
          f"{'atten_meas':>11} {'err':>8}")
    rows = []
    for u in [0.40, 0.50, 0.60, 0.71, 0.80, 0.90, 1.00]:
        M, _ = d10.flow(u, mgrid=grid)
        ctr, rho = d10.density(M, nbin=180)
        val = np.interp(ctr, mg, B2S2)
        sig = float(np.sqrt(np.sum(rho*val)))
        meas = d8.get(u)
        sm_ = meas["sigma"] if meas else float("nan")
        rows.append(dict(u=u, sigma_pred=sig, sigma_meas=sm_,
                         atten_pred=sig/sig_open,
                         atten_meas=(sm_/10.9456) if meas else float("nan"),
                         rel=(sig-sm_)/sm_ if meas else float("nan")))
        print(f"{u:6.2f} {sig:11.3f} {sm_:11.3f} {sig/sig_open:11.3f} "
              f"{sm_/10.9456:11.3f} {100*(sig-sm_)/sm_:+7.1f}%")
    out["sigma_vs_u"] = rows
    e = [abs(r["rel"]) for r in rows if not math.isnan(r["rel"])]
    print(f"\n    mean |err| = {100*np.mean(e):.1f}%  (acceptance criterion: 5%)")
    out["mean_abs_err"] = float(np.mean(e))

    json.dump(out, open(os.path.join(HERE, "d12_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d12_results_2026-08-29.json")
