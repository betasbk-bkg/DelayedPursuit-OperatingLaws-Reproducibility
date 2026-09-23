#!/usr/bin/env python3
"""
d99 -- the graceful prediction WITHOUT the small-angle linearisation.

WHY THIS EXISTS.  d97 linearises the Park-Kuipers law and predicts that the
shipped controller has no interior optimum, the mechanism being the tangent
feedforward B = k_delta k_phi.  That derivation has one weakness, and it is not
small.  The linearisation replaces

    atan(-k_phi phi)  ->  -k_phi phi        and      sin(delta) -> delta

but the test path is a SQUARE, whose corner is alpha = pi/2, and phi carries
that corner directly (phi = e - g + P, with P = alpha through the approach).
At k_phi = 3 and phi ~ 1.57 the argument k_phi phi is 4.7, where atan(4.7)=1.36
against a linear 4.7 -- the feedforward is a THIRD of its linearised strength.
The factor (1 + k_phi/(1 + (k_phi phi)^2)) collapses from 1 + k_phi to nearly 1
over the same range.

For RPP this never mattered: kappa = 2y/d^2 is exactly linear in y, so alpha
scales out and drops from the argmin, which is why d38's linear treatment was
sufficient.  For this controller it does not scale out, so the linear prediction
may be predicting the wrong thing.  This script therefore integrates the corner
transit with the law EXACTLY as the plugin computes it, at the actual corner
angle of the path that will be driven.

ACCEPTANCE TEST.  The same engine, with kappa = 2 y / d^2 substituted, must
return RPP's known optimum u* ~ 0.306 on the same square.  An engine that
cannot reproduce the answer we already have is not allowed to produce a new one.

WHAT IS FAITHFUL TO THE PLUGIN
  * target selection by INTEGRATED PATH distance <= L from the projection of the
    delayed pose (graceful_controller.cpp:200-222), while the control law then
    uses the EUCLIDEAN distance r to that point (ego_polar_coords.hpp:60-70).
    That mismatch is why the measured look-ahead was 0.514 against a configured
    0.600, and this engine reproduces it rather than assuming it away.
  * kappa = -(1/r)[k_delta(delta - atan(-k_phi phi))
                   + (1 + k_phi/(1+(k_phi phi)^2)) sin(delta)]
  * beta = 0 and no angular clamp, matching how the block is configured.

Usage:  python3 d99_graceful_nonlinear_2026-09-08.py
"""
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
L = 0.6
SIDE = 5.0


class Square:
    """Closed square path with EXACT projection, arclength and tangent.

    A sampled polyline needs a nearest-sample search per integration step, which
    dominates the cost and adds a discretisation error at exactly the place the
    experiment is about (the corner).  For a square all three operations are
    closed form, so they are written out.
    """

    def __init__(self, side=SIDE):
        self.side = side
        self.circ = 4.0 * side

    def at(self, s):
        s = s % self.circ
        k, t = int(s // self.side), s % self.side
        a = self.side
        return np.array([[t, 0.0], [a, t], [a - t, a], [0.0, a - t]][k])

    def tangent(self, s):
        k = int((s % self.circ) // self.side)
        return np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]][k])

    def project(self, p):
        """Nearest point on the square: (segment, arclength, distance)."""
        a = self.side
        best = None
        for k, (p0, d) in enumerate((((0.0, 0.0), (1.0, 0.0)),
                                     ((a, 0.0), (0.0, 1.0)),
                                     ((a, a), (-1.0, 0.0)),
                                     ((0.0, a), (0.0, -1.0)))):
            t = (p[0] - p0[0]) * d[0] + (p[1] - p0[1]) * d[1]
            t = min(max(t, 0.0), a)
            qx, qy = p0[0] + d[0] * t, p0[1] + d[1] * t
            e = math.hypot(p[0] - qx, p[1] - qy)
            if best is None or e < best[2]:
                best = (k, k * a + t, e)
        return best


def kappa_graceful(r, phi, delta, k_phi, k_delta):
    prop = k_delta * (delta - math.atan(-k_phi * phi))
    fb = (1.0 + (k_phi / (1.0 + (k_phi * phi) ** 2))) * math.sin(delta)
    return -(1.0 / max(r, 1e-6)) * (prop + fb)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def simulate(path, v, tau, law, k_phi=3.0, k_delta=2.0, look=L,
             dt=0.005, laps=3.0, settle=1.0):
    """Unicycle + pure transport delay + the plugin's own target selection."""
    s0 = 0.5 * path.side                     # start mid-side, away from a corner
    pos = path.at(s0).astype(float)
    th = math.atan2(*(path.tangent(s0)[::-1]))
    nd = max(int(round(tau / dt)), 0)
    hist = [(pos.copy(), th)]
    n = int(path.circ * laps / max(v, 1e-6) / dt)
    errs = []
    arcs = []
    dist = 0.0
    for _ in range(n):
        dp, dth = hist[max(0, len(hist) - 1 - nd)]
        _, s_near, _ = path.project(dp)
        tgt = path.at(s_near + look)          # selected by PATH distance
        dx, dy = tgt[0] - dp[0], tgt[1] - dp[1]
        r = math.hypot(dx, dy)                # law uses EUCLIDEAN distance
        los = math.atan2(-dy, dx)
        delta = wrap(dth + los)
        th_t = math.atan2(*(path.tangent(s_near + look)[::-1]))
        phi = wrap(th_t + los)
        if law == 'graceful':
            k = kappa_graceful(r, phi, delta, k_phi, k_delta)
        else:
            # RPP: kappa = 2 y / d^2, y the lateral offset of the target in the
            # robot frame.  This is the acceptance-test branch.
            y = -math.sin(dth) * dx + math.cos(dth) * dy
            k = 2.0 * y / max(r * r, 1e-9)
        th = th + k * v * dt
        pos = pos + np.array([math.cos(th), math.sin(th)]) * v * dt
        dist += v * dt
        hist.append((pos.copy(), th))
        _, sa, e = path.project(pos)
        errs.append(e)
        arcs.append(dist)
        if e > 3.0:                            # diverged
            return float('nan')
    errs = np.array(errs)
    arcs = np.array(arcs)
    m = arcs >= settle * path.circ
    if m.sum() < 100:
        return float('nan')
    return float(np.sqrt(np.mean(errs[m] ** 2)))


def locate(vs, rs):
    ok = [(v, r) for v, r in zip(vs, rs) if np.isfinite(r)]
    if len(ok) < 4:
        return None, 'too few', 0
    vs2 = [p[0] for p in ok]
    rs2 = [p[1] for p in ok]
    i = int(np.argmin(rs2))
    if i == 0:
        return vs2[0], 'EDGE_LOW', len(ok)
    if i == len(rs2) - 1:
        return vs2[-1], 'EDGE_HIGH', len(ok)
    sel = [k for k in range(len(rs2)) if rs2[k] <= min(rs2) * 1.35]
    if len(sel) >= 3:
        c = np.polyfit([vs2[k] for k in sel], [rs2[k] for k in sel], 2)
        if c[0] > 0:
            x = -c[1] / (2 * c[0])
            if vs2[0] <= x <= vs2[-1]:
                return float(x), 'bowl', len(sel)
    return vs2[i], 'grid', len(ok)


def main():
    path = Square()
    tau = 0.1325                      # tau_inject 0.10 + T_ctrl/2 + T_sim/2
    print('=' * 78)
    print('d99 -- graceful prediction with the EXACT law, no small-angle step')
    print('=' * 78)
    print('  square side %.1f m, look-ahead %.2f m, tau_tot %.4f s, corner 90 deg'
          % (path.side, L, tau))

    # ---- acceptance test: the same engine must give RPP's known optimum ------
    vs = np.linspace(0.55, 2.30, 15)
    rs = [simulate(path, v, tau, 'rpp') for v in vs]
    vr, how, npts = locate(vs, rs)
    ur = vr * tau / L if vr else float('nan')
    print()
    print('  ACCEPTANCE TEST -- RPP through this engine')
    print('    v* %.4f m/s -> u* %.4f  [%s, %d pts];  known u* 0.3060'
          % (vr, ur, how, npts))
    if not (abs(ur / 0.3060 - 1.0) < 0.12):
        print('    -> engine REJECTED: it does not reproduce the answer we have.')
        print('       No graceful prediction is issued from it.')
        json.dump({'accepted': False, 'rpp_u': ur},
                  open(HERE / 'd99_graceful_nonlinear_2026-09-08.json', 'w'),
                  indent=2)
        return
    print('    -> engine accepted (within 12%% of the known value).')

    out = {'accepted': True, 'rpp_u_star': ur, 'tau_tot': tau, 'rows': []}
    print()
    print('  %8s %6s %6s %10s %10s %9s %6s'
          % ('k_phi', 'A', 'B', 'v*', 'u*', 'how', 'npts'))
    for k_phi in (0.5, 1.0, 2.0, 3.0, 5.0):
        A, B = 2.0 + 1.0 + k_phi, 2.0 * k_phi
        vs = np.linspace(0.20, 1.80, 17)
        rs = [simulate(path, v, tau, 'graceful', k_phi=k_phi) for v in vs]
        v, how, npts = locate(vs, rs)
        u = (v * tau / L) if v else None
        out['rows'].append({'k_phi': k_phi, 'A': A, 'B': B, 'v_star': v,
                            'u_star': u, 'how': how, 'n': npts,
                            'v_grid': [float(x) for x in vs],
                            'rmse': [None if not np.isfinite(x) else float(x)
                                     for x in rs]})
        print('  %8.1f %6.1f %6.1f %10s %10s %9s %6d'
              % (k_phi, A, B, ('%.4f' % v) if v else '-',
                 ('%.4f' % u) if u else '-', how, npts))

    print()
    print('  linearised prediction (d97), for comparison:')
    print('    k_phi 0.5 -> 0.2270    1.0 -> 0.0995    2.0, 3.0, 5.0 -> none')
    print('  Where the two disagree, THIS one is the prediction of record: the')
    print('  square\'s corner is 90 deg and the plugin\'s atan/sin saturate there.')

    json.dump(out, open(HERE / 'd99_graceful_nonlinear_2026-09-08.json', 'w'),
              indent=2)
    print('\n-> d99_graceful_nonlinear_2026-09-08.json')


if __name__ == '__main__':
    main()
