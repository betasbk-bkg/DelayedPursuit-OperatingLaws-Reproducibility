#!/usr/bin/env python3
"""
d36 -- Nav2 Regulated Pure Pursuit: plant-class audit and pre-registration.

WHY
  NAV2_VALIDATION_PLAN_2026-08-29.md proposes validating Theorem 1 on the
  deployed nav2_regulated_pure_pursuit_controller by sweeping v on a CIRCLE and
  locating the RMSE(v) minimum.  Before installing ROS2 this script asks whether
  that observable exists for RPP at all.

WHAT IS TRANSCRIBED (navigation2 @ jazzy, read 2026-08-29; no the reference engine code reused)
  path_handler.cpp        closest-pose search bounded by max_robot_pose_search_dist
                          (integrated distance), truncation at costmap extent,
                          destructive pruning
  getLookAheadPoint       first pose with |p| >= L, then circleSegmentIntersection
  circleSegmentIntersection   ported verbatim
  calculateCurvature      k = 2*y / (x^2 + y^2)
  computeVelocityCommands angular_vel = linear_vel * curvature
  applyConstraints        with regulation off, v = desired_linear_vel
                          (approach scaling inactive while the local plan is
                           longer than approach_velocity_scaling_dist = 0.6 m)
  nav2_loopback_sim       update_duration default 0.01 s, integrates cmd_vel

RESULT
  RPP is a CURVATURE-COMMAND controller (theta_dot = v*kappa), not the
  HEADING-SERVO plant (theta = kernel * theta_cmd) that Theorem 1 assumes.
  Consequences, all verified below:
    (1) a circle is an exact fixed point of RPP -> zero steady-state error at
        any speed and any lookahead -> the V-curve the plan wants DOES NOT EXIST
    (2) what exists on a circle is a stability cliff, closed form
            omega^4 - 4*omega^2 - 4 = 0,  u_c = arctan(omega)/omega = 0.520494
    (3) on a polygon an interior optimum DOES exist and v* ~ L/tau_tot holds,
        but the constant is ~0.31 and is INDEPENDENT of the corner angle, so
        Proposition 1's 0.5*cos(alpha/2) does not transfer to this plant class

Run:    python d36_rpp_model_audit_2026-08-29.py
Writes: d36_results_2026-08-29.json
"""
import json
import numpy as np

L_DEF = 0.6
T_CTRL = 0.05          # controller_frequency 20 Hz
DT_SIM = 0.01          # nav2_loopback_sim update_duration default
TAU_COMPOSITE = T_CTRL / 2 + DT_SIM / 2      # 0.030 s


# ----------------------------------------------------------------- RPP kernel
def circle_segment_intersection(p1, p2, r):
    """Verbatim port of RegulatedPurePursuitController::circleSegmentIntersection."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dr2 = dx * dx + dy * dy
    D = x1 * y2 - x2 * y1
    dd = (x2 * x2 + y2 * y2) - (x1 * x1 + y1 * y1)
    sq = np.sqrt(max(r * r * dr2 - D * D, 0.0))
    return ((D * dy + np.sign(dd) * dx * sq) / dr2,
            (-D * dx + np.sign(dd) * dy * sq) / dr2)


def circle_path(R, laps, spacing):
    total = 2 * np.pi * R * laps
    s = np.linspace(0.0, total, int(total / spacing))
    th = s / R
    return np.stack([R * np.sin(th), R - R * np.cos(th)], axis=1)


def ngon_path(n, side, laps, spacing):
    """Regular n-gon; exterior angle alpha = 2*pi/n."""
    pts, p, ang = [], np.array([0.0, 0.0]), 0.0
    for _ in range(int(laps * n)):
        d = np.array([np.cos(ang), np.sin(ang)])
        m = max(int(side / spacing), 2)
        for j in range(m):
            pts.append(p + d * side * j / m)
        p = p + d * side
        ang += 2 * np.pi / n
    pts.append(p)
    return np.array(pts)


def _xtrack(p, path, i0, w=45):
    lo, hi = max(0, i0 - w), min(len(path), i0 + w)
    a, b = path[lo:hi - 1], path[lo + 1:hi]
    ab, ap = b - a, p - a
    t = np.clip((ap * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return float(np.min(np.hypot(*(p - (a + ab * t[:, None])).T)))


def run(path, v_des, tau, L=L_DEF, dt=DT_SIM, ctrl_dt=T_CTRL,
        costmap_extent=2.5, search_dist=2.5, spacing=0.05, settle_frac=0.4):
    """One RPP run with an exact-timestamp delay of tau applied to cmd_vel."""
    npath, start = len(path), 0
    x, y = path[0]
    th = np.arctan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    plen = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
    nsteps = int(plen / v_des * 1.05 / dt)
    next_c, q, va, wa, t, errs = 0.0, [], 0.0, 0.0, 0.0, []
    for i in range(nsteps):
        if t >= next_c - 1e-12:
            next_c += ctrl_dt
            if start >= npath - 3:
                break
            hi = min(npath, start + int(search_dist / spacing) + 1)
            sub = path[start:hi]
            start += int(np.argmin(np.hypot(sub[:, 0] - x, sub[:, 1] - y)))
            sub = path[start:min(npath, start + int(costmap_extent / spacing) + 2)]
            if len(sub) < 2:
                break
            c, s_ = np.cos(-th), np.sin(-th)
            dx, dy = sub[:, 0] - x, sub[:, 1] - y
            bx, by = c * dx - s_ * dy, s_ * dx + c * dy
            rr = np.hypot(bx, by)
            if np.any(rr >= L):
                k = int(np.argmax(rr >= L))
                cx, cy = ((bx[-1], by[-1]) if k <= 0 else
                          circle_segment_intersection((bx[k - 1], by[k - 1]),
                                                      (bx[k], by[k]), L))
            else:
                cx, cy = bx[-1], by[-1]
            d2 = cx * cx + cy * cy
            q.append((t + tau, v_des, v_des * (2.0 * cy / d2 if d2 > 0.001 else 0.0)))
        while q and q[0][0] <= t + 1e-12:
            _, va, wa = q.pop(0)
        x += va * np.cos(th) * dt
        y += va * np.sin(th) * dt
        th += wa * dt
        t += dt
        if i % 5 == 0:
            errs.append(_xtrack(np.array([x, y]), path, start))
    e = np.array(errs)
    if len(e) < 60:
        return np.nan
    return float(np.sqrt(np.mean(e[int(len(e) * settle_frac):] ** 2)))


# ------------------------------------------------- closed-form stability limit
def stability_closed_form():
    """
    Linearise RPP about a path:  e_dot = v*psi,  psi_dot = -(2v/L^2)(e + L*psi),
    with the command delayed by tau_tot.  With s = v t / L and u = v*tau_tot/L:

        e''(s) + 2 e'(s-u) + 2 e(s-u) = 0
        characteristic:  lam^2 + 2 exp(-lam u)(lam + 1) = 0

    On lam = i*omega the imaginary part gives tan(omega u) = omega, and the real
    part then gives omega^4 - 4 omega^2 - 4 = 0, i.e. omega^2 = 2 + 2*sqrt(2).
    """
    w = np.sqrt(2 + 2 * np.sqrt(2))
    return float(np.arctan(w) / w), float(w)


def stability_dde(u, T=400.0, h=2e-4):
    """Independent check: method-of-steps integration of the same DDE."""
    n, nd = int(T / h), int(round(u / h))
    e, ep = np.zeros(n + 1), np.zeros(n + 1)
    e[0] = 1e-3
    for i in range(n):
        j = i - nd
        epp = -2 * (ep[j] if j >= 0 else 0.0) - 2 * (e[j] if j >= 0 else 0.0)
        ep[i + 1] = ep[i] + h * epp
        e[i + 1] = e[i] + h * ep[i + 1]
        if abs(e[i + 1]) > 1e6:
            return np.inf
    return float(np.max(np.abs(e[int(n * 0.6):])) / 1e-3)


def ustar(path, tau, L=L_DEF, u_guess=0.32, span=(0.45, 1.85), npts=24):
    """Locate the interior RMSE(v) minimum and return it as u* = v*tau_tot/L."""
    tt = tau + TAU_COMPOSITE
    vg = u_guess * L / tt
    vs = np.linspace(vg * span[0], vg * span[1], npts)
    rs = np.array([run(path, v, tau, L=L) for v in vs])
    ok = np.isfinite(rs)
    vo, ro = vs[ok], rs[ok]
    m = int(np.argmin(ro))
    if not (0 < m < len(vo) - 1):
        return float("nan"), False
    y0, y1, y2 = ro[m - 1], ro[m], ro[m + 1]
    dv = vo[1] - vo[0]
    v = vo[m] + 0.5 * dv * (y0 - y2) / (y0 - 2 * y1 + y2)
    return float(v * tt / L), True


def main():
    out = {"transcribed_from": "navigation2 @ jazzy, read 2026-08-29",
           "tau_composite_s": TAU_COMPOSITE}

    # -- 1. circle: is there any steady-state error at zero delay? ------------
    cp = circle_path(5.0, 4.0, 0.05)
    base = {f"v={v}": run(cp, v, 0.0) for v in (0.3, 0.6, 1.0, 1.5, 2.0)}
    out["circle_zero_delay_rmse"] = base
    out["circle_zero_delay_heading_servo_prediction_m"] = L_DEF ** 2 / (2 * 5.0)

    # -- 2. circle: the stability cliff ---------------------------------------
    uc, w = stability_closed_form()
    out["stability_closed_form"] = {"omega": w, "u_c": uc,
                                    "lookahead_time_over_tau_tot": 1.0 / uc}
    lo, hi = 0.30, 0.80
    for _ in range(30):
        mid = (lo + hi) / 2
        if stability_dde(mid) > 1.0:
            hi = mid
        else:
            lo = mid
    out["stability_dde_check_u_c"] = (lo + hi) / 2

    meas = {}
    thr = 10 * max(max(base.values()), 1e-4) + 0.005
    for tau in (0.2, 0.4, 0.6, 0.8, 1.0):
        tt = tau + TAU_COMPOSITE
        a, b = 0.05, 3.0
        for _ in range(16):
            m = (a + b) / 2
            r = run(cp, m, tau)
            if (not np.isfinite(r)) or r > thr:
                b = m
            else:
                a = m
        vc = (a + b) / 2
        meas[f"tau={tau}"] = {"tau_tot": tt, "v_crit": vc, "u_crit": vc * tt / L_DEF}
    out["stability_simulated"] = meas

    # -- 3. polygons: the interior optimum and its constant -------------------
    poly = {}
    for n, side in [(24, 5.0), (12, 5.0), (8, 5.0), (6, 5.0), (4, 5.0), (3, 6.0)]:
        alpha = 2 * np.pi / n
        laps = max(2, int(round(60 / n)))   # ~60 corner transits, independent of n
        path = ngon_path(n, side, laps, 0.05)
        us = [u for tau in (0.5, 0.9) for u, ok in [ustar(path, tau)] if ok]
        u = float(np.mean(us)) if us else float("nan")
        poly[f"n={n}"] = {"alpha_deg": float(np.degrees(alpha)),
                          "seg_over_L": side / L_DEF,
                          "laps": laps,
                          "u_star": u,
                          "half_cos_alpha_2": 0.5 * float(np.cos(alpha / 2)),
                          "u_over_cos": u / float(np.cos(alpha / 2))}
    out["polygon"] = poly

    with open("d36_results_2026-08-29.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
