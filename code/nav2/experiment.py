#!/usr/bin/env python3
"""
this study Nav2 experiment driver -- the FULL design, not just the delay sweep.

WHY THIS EXISTS

The first sweep varied only tau_inject.  That tests Theorem 2 only as a SUM, and
weakly: the composite term (T_ctrl/2 + T_sim/2 = 0.030 s) is 37% of tau_tot at
tau = 0.05 but 2.4% at tau = 1.2, so most points barely probe it at all.  The
this study engine's own Table I did better -- it varied EACH TERM of the delay chain
independently (WIN 6-fold, T_s 5-fold) and checked that the composition constant
tracked each one.  This driver reproduces that methodology on the deployed
controller, and promotes everything that could plausibly move the answer from
"fixed" to "variable".

PARAMETER CLASSIFICATION

  TERMS OF THE DELAY CHAIN -- sweeping these tests Theorem 2 term by term
    tau_inject          delay node        dynamic
    controller_frequency  -> T_ctrl/2     LAUNCH (read once in on_configure,
                                          controller_server.cpp:124; no dynamic cb)
    update_duration       -> T_sim/2      LAUNCH (builds the simulator's timer)

  TERMS OF THE GEOMETRY -- sweeping these tests Theorem 1's scaling
    lookahead_dist L    v* should be proportional to L      dynamic
    corner angle alpha  d36 predicts u* nearly alpha-free   run_point param
    side / L            validity needs seg/L >~ 5           run_point param

  SHOULD NOT MATTER -- swept as NULL CONTROLS.  If u* moves with any of these,
  the number is an artefact of the harness rather than a property of the loop.
    local costmap size          bounds the transformed plan   LAUNCH
    max_robot_pose_search_dist  bounds the closest-pose search LAUNCH
    sample_hz                   measurement only               run_point param
    transform_tolerance         TF lookup slack                dynamic

A block whose knob is marked LAUNCH restarts the stack per value; the rest reuse
one stack.  Every block writes its own JSON so blocks can be run independently:

    python3 experiment.py --block freq
    python3 experiment.py --block tsim,lookahead,alpha,nulls
"""
import argparse
import json
import math
import os
import subprocess
import time

L_DEFAULT = 0.6
U_STAR = 0.314
SIDE = 5.0
TSIM_DEFAULT = 0.01
FREQ_DEFAULT = 20.0
OUT_DIR = '/tmp/letg2/exp'
WS = os.path.expanduser('~/letg2_ws')
PKG_SRC = os.path.join(WS, 'src', 'letg2_nav2_validation')


def sh(cmd, timeout=900):
    # start_new_session + killpg: a timeout must take the whole process group with
    # it, otherwise the wrapper shell dies and an orphaned run_point keeps driving
    # the robot into the next point (audit finding).
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, executable='/bin/bash', start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), 9)
        out, err = p.communicate()
        print(f'   !! timeout after {timeout:.0f}s; process group killed')
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def composite(freq, tsim):
    return 1.0 / (2.0 * freq) + tsim / 2.0


def vstar(tau, freq=FREQ_DEFAULT, tsim=TSIM_DEFAULT, lookahead=L_DEFAULT):
    return U_STAR * lookahead / (tau + composite(freq, tsim))


# ------------------------------------------------------------------ stack mgmt
def cleanup():
    sh(f'{PKG_SRC}/scripts/cleanup.sh')
    # Drop the daemon's cached node graph too, so anything that does consult
    # `ros2 node list` afterwards sees reality rather than the previous stack.
    sh('ros2 daemon stop', timeout=60)
    time.sleep(1)
    sh('ros2 daemon start', timeout=60)
    time.sleep(2)


def launch(tau, freq=FREQ_DEFAULT, tsim=TSIM_DEFAULT, lookahead=L_DEFAULT,
           costmap=5.0, search=2.5, vel=1.0, timeout=60, sim_exe=None,
           act_tau=0.0, wheelbase=0.5, max_steer=0.0, controller='rpp'):
    cleanup()
    pf = f'{OUT_DIR}/params.yaml'
    r = sh(f'python3 {PKG_SRC}/letg2_nav2_validation/make_params.py '
           f'--vel {vel} --lookahead {lookahead} --freq {freq} '
           f'--costmap-size {costmap} --search-dist {search} '
           f'--controller {controller} --out {pf}')
    if r.returncode != 0:
        print(f'   !! make_params failed: {r.stderr[-300:]}')
        return False
    log = f'{OUT_DIR}/stack.log'
    # sim_exe selects the plant node.  Default is the untouched upstream
    # nav2_loopback_sim; 'loopback_midpoint' is our diagnostic build that uses the
    # midpoint heading instead of the pre-tick heading (see loopback_midpoint.py).
    simsel = ''
    if sim_exe:
        simsel = (f'sim_pkg:=letg2_nav2_validation sim_exe:={sim_exe} '
                  if sim_exe != 'loopback_simulator' else '')
    # Only loopback_actuator declares actuator_tau; passing it to any other build
    # would abort that node.  The launch file guards this too -- both, on purpose.
    if sim_exe == 'loopback_actuator':
        simsel += f'actuator_tau:={act_tau} '
    elif act_tau:
        raise ValueError('act_tau requires --sim-exe loopback_actuator')
    if sim_exe == 'loopback_bicycle':
        simsel += f'wheelbase:={wheelbase} max_steer:={max_steer} '
    elif max_steer:
        raise ValueError('max_steer requires --sim-exe loopback_bicycle')
    sh(f'setsid nohup ros2 launch letg2_nav2_validation pursuit_stack.launch.py '
       f'params_file:={pf} delay_sec:={tau} update_duration:={tsim} {simsel}'
       f'> {log} 2>&1 < /dev/null &')
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(3)
        if 'Managed nodes are active' in open(log, errors='ignore').read():
            time.sleep(2)
            if single_stack(sim_exe):
                return True
            # Leave nothing behind: a rejected stack that keeps running would be
            # cleaned by the NEXT launch, but if the block aborts here it stays up
            # and pollutes whatever runs next.
            print('   !! rejected this stack; cleaning up')
            cleanup()
            return False
    print('   !! stack did not become active')
    cleanup()
    return False


def single_stack(sim_exe=None):
    """
    Count RUNNING PROCESSES, not `ros2 node list`.

    The ROS 2 daemon caches the node graph: a node killed seconds ago keeps
    appearing in `ros2 node list` until the cache expires.  Using that list as the
    single-stack check made every relaunch after the first report
    "/controller_server: 2 instances" and abort the block -- the stale cache entry
    plus the freshly launched one.  pgrep is ground truth.
    """
    # The bracket is not decoration: `pgrep -f X` also matches the shell that is
    # running the pgrep, because X appears in that shell's own command line.  A
    # plain pattern therefore always counts one extra and single_stack() never
    # passes.  Writing [j]azzy makes the regex match "jazzy" while the shell's
    # cmdline contains the literal "[j]azzy", which the regex does not match.
    pats = {
        'controller_server': '[j]azzy/lib/nav2_controller/controller_server',
        # The plant node's process name depends on which implementation is
        # running.  Getting this wrong is not benign: the check counted 0 and
        # rejected every stack of the midpoint-diagnostic run, which is the
        # defence working, but only after wasting the launch.
        'loopback_simulator': ('nav2_loopback_sim/[l]oopback_simulator'
                               if not sim_exe or sim_exe == 'loopback_simulator'
                               else 'letg2_nav2_validation/[%s]%s'
                                    % (sim_exe[0], sim_exe[1:])),
        'delay_node': 'letg2_nav2_validation/[d]elay_node',
    }
    ok = True
    for name, pat in pats.items():
        r = sh(f"pgrep -f '{pat}' | wc -l", timeout=60)
        try:
            c = int(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            c = -1
        if c != 1:
            print(f'   !! {name}: {c} process(es), expected exactly 1')
            ok = False
    return ok


def set_param(node, name, value):
    r = sh(f'ros2 param set {node} {name} {value}', timeout=60)
    return 'Set parameter successful' in r.stdout


# --------------------------------------------------------------------- running
def run_point(tag, tau, v, laps, n_sides=4, side=SIDE, sample_hz=50.0,
              settle_laps=1.0, tail=3.5):
    out = f'{OUT_DIR}/{tag}.json'
    if os.path.exists(out):
        os.remove(out)
    lap_len = n_sides * side
    cmd = (f'ros2 run letg2_nav2_validation run_point --ros-args '
           f'-p path_kind:=ngon -p n_sides:={n_sides} -p side:={side} '
           f'-p laps:={laps} -p desired_vel:={v} -p tau:={tau} '
           f'-p sample_hz:={sample_hz} -p settle_laps:={settle_laps} '
           f'-p tail_exclude_m:={tail} -p out_file:={out}')
    budget = lap_len * laps / max(v, 1e-3) * 2.5 + tau + 120
    t0 = time.time()
    sh(cmd, timeout=budget)
    if not os.path.exists(out):
        return None
    d = json.load(open(out))
    d['wall_s'] = time.time() - t0
    return d


def get_param(node, name):
    r = sh(f'ros2 param get {node} {name}', timeout=60)
    # output looks like: "Double value is: 0.8695"
    for tok in r.stdout.replace(':', ' ').split():
        try:
            return float(tok)
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Corner isolation.  d45/d46 found that the design rule "seg/L >= 5" is not a
# constant: the arclength a corner transient needs to decay is a function of u,
# and u is the swept variable.  Solving the loop's own characteristic equation
#
#     lam^2 + 2 exp(-lam u) (lam + 1) = 0        (arclength units, s = dist / L)
#
# for the dominant root gives 3.95 L at u = 0.306 but 15.85 L at u = 0.44.  At
# seg/L = 8.3 (side 5.0, L = 0.6) everything above u = 0.389 has a live transient
# arriving at the next corner; 46 of 225 measured points were in that state.
#
# Worse, v_sweep spans +-40% of v*, and v* = u* L / tau_tot.  When tau_tot is
# swept (the freq block sweeps it 5x), the same relative span in v reaches a
# DIFFERENT u.  The 20 Hz row reached u = 0.5174, which is 99.4% of the stability
# limit u_c = 0.520494.  The sweep width has to be defined in u, not in v.
# ---------------------------------------------------------------------------
U_C = 0.520494


def _dominant_root(u):
    import cmath
    best = None
    for x0 in [-3.0, -2.0, -1.2, -0.6, -0.2, 0.2]:
        for y0 in [0.5, 1.5, 2.2, 3.0, 4.5, 6.0]:
            z = complex(x0, y0)
            good = True
            for _ in range(150):
                f = z * z + 2.0 * cmath.exp(-z * u) * (z + 1.0)
                h = 1e-7
                fp = ((z + h) ** 2 + 2.0 * cmath.exp(-(z + h) * u) * (z + h + 1.0)
                      - ((z - h) ** 2 + 2.0 * cmath.exp(-(z - h) * u) * (z - h + 1.0))) / (2 * h)
                if abs(fp) < 1e-16:
                    good = False
                    break
                zn = z - f / fp
                if abs(zn - z) < 1e-13:
                    z = zn
                    break
                z = zn
            if good and z.imag > 1e-6:
                r = z * z + 2.0 * cmath.exp(-z * u) * (z + 1.0)
                if abs(r) < 1e-8 and (best is None or z.real > best.real):
                    best = z
    return best


def settle_arclen(u):
    """Arclength, in units of L, for a corner transient to decay to 1%."""
    z = _dominant_root(u)
    if z is None or z.real >= 0:
        return float('inf')
    return math.log(100.0) / (-z.real)


def u_isolated(seg_over_L):
    """Largest u whose corner transient still fits inside one segment."""
    lo, hi = 0.05, U_C - 1e-4
    if settle_arclen(hi) <= seg_over_L:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if settle_arclen(mid) <= seg_over_L:
            lo = mid
        else:
            hi = mid
    return lo


def u_sweep(tag, tau, tau_tot, L, seg_over_L, npoints=9, u_lo=0.60, laps=2.75,
              u_cap=0.85, **kw):
    """
    Sweep in u, not in v.  u_hi is whatever the geometry can isolate, capped so
    the row never approaches u_c.  Returns the curve plus the limits it used, so
    the analysis can see that the isolation condition was enforced by design and
    not checked afterwards.
    """
    u_iso = u_isolated(seg_over_L)
    u_hi = min(u_iso, u_cap * U_C)
    us = [U_STAR * u_lo + (u_hi - U_STAR * u_lo) * i / (npoints - 1)
          for i in range(npoints)]
    print(f'      u-sweep: seg/L={seg_over_L:.1f}  u_iso={u_iso:.4f}  '
          f'u in [{us[0]:.4f}, {us[-1]:.4f}]  (u_c={U_C})')
    curve = []
    for u in us:
        v = u * L / tau_tot
        if not set_param('/controller_server', 'FollowPath.desired_linear_vel', v):
            print(f'      u={u:6.4f} v={v:7.4f}  !! could not set vel; aborting row')
            break
        if not set_param('/letg2_delay_node', 'delay_sec', tau):
            print(f'      u={u:6.4f}  !! could not set delay_sec; aborting row')
            break
        time.sleep(0.5)
        v_ctrl = get_param('/controller_server', 'FollowPath.desired_linear_vel')
        if v_ctrl is None or abs(v_ctrl - v) > 1e-6:
            print(f'      u={u:6.4f}  !! controller reports {v_ctrl}; aborting row')
            break
        d = run_point(f'{tag}_u{u:.4f}', tau, v, laps, **kw)
        if d is None:
            print(f'      u={u:6.4f} v={v:7.4f}  (no output)')
            continue
        d['v_controller'] = v_ctrl
        # the same two gates v_sweep applies.  The truncation gate is what caught
        # the block that silently ran every point at the launch speed, so it must
        # not be skipped here just because the sweep variable changed.
        if d.get('goal_status') != 4:
            print(f'      u={u:6.4f}  goal_status={d.get("goal_status")} -- excluded')
            continue
        if d.get('truncated'):
            print(f'      u={u:6.4f}  TRUNCATED {d["elapsed_fraction"]:.2f} -- excluded')
            continue
        # curve entries keep the SAME schema v_sweep produces (d40-d47 all read
        # 'v' and 'rmse'), with the u-space design values added alongside.
        curve.append({'v': v, 'rmse': d['rmse_m'],
                      'rmse_intlap': d.get('rmse_m_intlap'),
                      'f': v / (U_STAR * L / tau_tot),
                      'u_design': u, 'u_iso': u_iso,
                      'n_used': d['n_samples_used'], 'wall_s': d['wall_s'],
                      'tau_applied_mean': d.get('tau_applied_mean'),
                      'tau_applied_sd': d.get('tau_applied_sd')})
        print(f'      u={u:6.4f} v={v:7.4f}  RMSE {d["rmse_m"]:8.5f}  '
              f'{d["wall_s"]:5.0f}s')
    return curve


def v_sweep(tag, tau, vpred, npoints=9, span=0.40, laps=3.75, **kw):
    """Locate the RMSE(v) minimum around vpred.  Returns the curve."""
    curve = []
    for i in range(npoints):
        f = 1.0 - span + 2 * span * i / (npoints - 1)
        v = vpred * f
        # The controller's speed must be SET for every point and then READ BACK.
        # The first E2 attempt omitted the set: every point ran at the launch
        # speed while run_point believed v was different.  The truncation detector
        # caught it (elapsed/expected = 0.59), but only by luck of the ratio -- so
        # now the controller's own value is read back and must equal v, and it is
        # recorded in the curve so the analysis can check it too.
        if not set_param('/controller_server', 'FollowPath.desired_linear_vel', v):
            print(f'      v={v:7.4f}  !! could not set desired_linear_vel; aborting row')
            break
        if not set_param('/letg2_delay_node', 'delay_sec', tau):
            print(f'      v={v:7.4f}  !! could not set delay_sec; aborting row')
            break
        time.sleep(0.5)
        v_ctrl = get_param('/controller_server', 'FollowPath.desired_linear_vel')
        if v_ctrl is None or abs(v_ctrl - v) > 1e-6:
            print(f'      v={v:7.4f}  !! controller reports {v_ctrl}; aborting row')
            break
        d = run_point(f'{tag}_v{v:.4f}', tau, v, laps, **kw)
        if d is None:
            print(f'      v={v:7.4f}  (no output)')
            continue
        d['v_controller'] = v_ctrl
        if d.get('goal_status') != 4:
            print(f'      v={v:7.4f}  goal_status={d.get("goal_status")} -- excluded')
            continue
        if d.get('truncated'):
            print(f'      v={v:7.4f}  TRUNCATED {d["elapsed_fraction"]:.2f} -- excluded')
            continue
        curve.append({'v': v, 'rmse': d['rmse_m'], 'rmse_intlap': d.get('rmse_m_intlap'), 'f': f,
                      'n_used': d['n_samples_used'], 'wall_s': d['wall_s'],
                      'tau_applied_mean': d.get('tau_applied_mean'),
                      'tau_applied_sd': d.get('tau_applied_sd')})
        print(f'      v={v:7.4f} ({f:4.2f} v*)  RMSE {d["rmse_m"]:8.5f}  '
              f'{d["wall_s"]:5.0f}s')
    return curve


def locate(curve):
    pts = sorted([(c['v'], c['rmse']) for c in curve
                  if c['rmse'] == c['rmse']], key=lambda p: p[0])
    if len(pts) < 3:
        return None, 'too-few'
    vs = [p[0] for p in pts]
    rs = [p[1] for p in pts]
    i = min(range(len(rs)), key=lambda k: rs[k])
    if i in (0, len(rs) - 1):
        return vs[i], 'EDGE'
    (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(d) < 1e-15:
        return vs[i], 'grid'
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    if a <= 0:
        return vs[i], 'grid'
    v = -b / (2 * a)
    return (v, 'parabola') if x0 <= v <= x2 else (vs[i], 'grid')


def load_partial(out_name, key_fields):
    """
    Resume support.

    record() writes the block JSON after EVERY row, so a block interrupted by a
    shutdown leaves its completed rows on disk.  But each block function starts
    from rows = [] and record() opens with 'w', so simply re-running the block
    DESTROYS those rows.  E11b was 3.4 h; E1' is longer.

    This reads back whatever is already there and returns (rows, done) where
    `done` is the set of key tuples already measured, so a block can skip them.
    Nothing is overwritten until the first record() of the new run, which now
    carries the old rows forward.
    """
    path = f'{OUT_DIR}/{out_name}'
    if not os.path.exists(path):
        return [], set()
    try:
        blob = json.load(open(path))
    except Exception as e:
        print(f'   !! {out_name} unreadable ({e}); starting fresh')
        return [], set()
    rows = [r for r in blob.get('rows', []) if r.get('curve')]
    done = {tuple(r.get(k) for k in key_fields) for r in rows}
    if rows:
        print(f'   RESUME: {len(rows)} row(s) already in {out_name}; '
              f'skipping {sorted(done)}')
    return rows, done


def record(rows, out_name, meta):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f'{OUT_DIR}/{out_name}', 'w') as f:
        json.dump({'meta': meta, 'rows': rows}, f, indent=2)
    print(f'   -> {OUT_DIR}/{out_name}')


# ---------------------------------------------------------------------- blocks
def block_freq(a):
    """T_ctrl/2 term of Theorem 2.  Needs a relaunch per frequency."""
    tau = 0.10
    rows = []
    for freq in [20.0, 10.0, 5.0, 3.3333, 2.5, 2.0]:
        tt = tau + composite(freq, TSIM_DEFAULT)
        vp = U_STAR * L_DEFAULT / tt
        print(f'  freq {freq:7.3f} Hz  T_ctrl/2={1/(2*freq):.4f}  '
              f'tau_tot={tt:.4f}  v* predicted {vp:.4f}')
        if not launch(tau, freq=freq, vel=vp):
            continue
        curve = v_sweep(f'freq{freq:.2f}', tau, vp, npoints=a.npoints, span=a.span)
        v, how = locate(curve)
        rows.append({'freq': freq, 'T_ctrl_half': 1 / (2 * freq), 'tau_inject': tau,
                     'tau_tot': tt, 'v_pred': vp, 'v_obs': v, 'how': how,
                     'u_obs': (v * tt / L_DEFAULT) if v else None, 'curve': curve})
        print(f'   => v* obs {v}  ({how})  u*={(v*tt/L_DEFAULT) if v else float("nan"):.4f}')
        record(rows, 'block_freq.json', {'block': 'controller_frequency', 'tau': tau})
    cleanup()


def block_tsim(a):
    """T_sim/2 term of Theorem 2.  Needs a relaunch per update_duration."""
    tau = 0.20
    rows = []
    for tsim in [0.005, 0.01, 0.02, 0.05, 0.10]:
        tt = tau + composite(FREQ_DEFAULT, tsim)
        vp = U_STAR * L_DEFAULT / tt
        print(f'  update_duration {tsim:.3f}  T_sim/2={tsim/2:.4f}  '
              f'tau_tot={tt:.4f}  v* predicted {vp:.4f}')
        if not launch(tau, tsim=tsim, vel=vp):
            continue
        curve = v_sweep(f'tsim{tsim:.3f}', tau, vp, npoints=a.npoints, span=a.span)
        v, how = locate(curve)
        rows.append({'update_duration': tsim, 'T_sim_half': tsim / 2, 'tau_inject': tau,
                     'tau_tot': tt, 'v_pred': vp, 'v_obs': v, 'how': how,
                     'u_obs': (v * tt / L_DEFAULT) if v else None, 'curve': curve})
        record(rows, 'block_tsim.json', {'block': 'update_duration', 'tau': tau})
    cleanup()


def block_lookahead(a):
    """Theorem 1's proportionality: v* should scale with L at fixed tau_tot."""
    tau = 0.20
    tt = tau + composite(FREQ_DEFAULT, TSIM_DEFAULT)
    rows = []
    if not launch(tau, vel=1.0):
        return
    for Lk in [0.3, 0.45, 0.6, 0.9]:
        if SIDE / Lk < 5.0:
            print(f'   skip L={Lk}: seg/L={SIDE/Lk:.1f} violates the >~5 condition')
            continue
        vp = U_STAR * Lk / tt
        ok = all(set_param('/controller_server', f'FollowPath.{k}', Lk)
                 for k in ('lookahead_dist', 'min_lookahead_dist',
                           'max_lookahead_dist', 'curvature_lookahead_dist'))
        if not ok:
            print(f'   !! could not set lookahead {Lk}')
            continue
        print(f'  L {Lk:.2f} m  seg/L={SIDE/Lk:.1f}  v* predicted {vp:.4f}')
        curve = v_sweep(f'look{Lk:.2f}', tau, vp, npoints=a.npoints, span=a.span)
        v, how = locate(curve)
        rows.append({'lookahead': Lk, 'seg_over_L': SIDE / Lk, 'tau_tot': tt,
                     'v_pred': vp, 'v_obs': v, 'how': how,
                     'u_obs': (v * tt / Lk) if v else None, 'curve': curve})
        record(rows, 'block_lookahead.json', {'block': 'lookahead_dist', 'tau': tau})
    cleanup()


def block_alpha(a):
    """d36 predicts u* is nearly independent of corner angle for this plant."""
    tau = 0.20
    tt = tau + composite(FREQ_DEFAULT, TSIM_DEFAULT)
    vp = U_STAR * L_DEFAULT / tt
    rows = []
    if not launch(tau, vel=vp):
        return
    for n in [3, 4, 6, 8, 12]:
        alpha = 360.0 / n
        print(f'  n={n}  alpha={alpha:.0f} deg  seg/L={SIDE/L_DEFAULT:.1f}')
        curve = v_sweep(f'ngon{n}', tau, vp, npoints=a.npoints, span=a.span,
                        n_sides=n, laps=2.75 if n >= 8 else 3.75)
        v, how = locate(curve)
        rows.append({'n_sides': n, 'alpha_deg': alpha, 'tau_tot': tt,
                     'v_pred': vp, 'v_obs': v, 'how': how,
                     'u_obs': (v * tt / L_DEFAULT) if v else None,
                     'half_cos': 0.5 * math.cos(math.radians(alpha) / 2), 'curve': curve})
        record(rows, 'block_alpha.json', {'block': 'corner_angle', 'tau': tau})
    cleanup()


def block_replicate(a):
    """
    E7 -- the calibration measurement E6 needed and did not have.

    E6 judged each null knob by the spread of u* across its rows, against a noise
    model built from ONE repeatability triple taken at a single operating point.
    That triple measures the reproducibility of a single RMSE number; it does not
    measure the reproducibility of a whole ROW, which is what a null knob is
    compared across, and it does not contain a stack relaunch, which every
    cross-row comparison does contain.

    So: run the identical row N times, each with its own launch(), and read off
    sigma_row(u*) and sigma_row(tau_tot) directly.  That is the denominator the
    null test actually needs.  Nothing is swept -- the spread IS the measurement.
    """
    tau = 0.20
    tt = tau + composite(FREQ_DEFAULT, TSIM_DEFAULT)
    vp = U_STAR * L_DEFAULT / tt
    rows = []
    for rep in range(a.reps):
        print(f'  REPLICATE {rep + 1}/{a.reps}  (fresh stack, identical settings)')
        if not launch(tau, vel=vp):
            continue
        curve = v_sweep(f'rep{rep}', tau, vp, npoints=5, span=0.25)
        v, how = locate(curve)
        rows.append({'knob': 'replicate', 'rep': rep, 'v_obs': v, 'how': how,
                     'u_obs': (v * tt / L_DEFAULT) if v else None, 'curve': curve})
        record(rows, 'block_replicate.json',
               {'block': 'row replicates', 'tau': tau, 'reps': a.reps})
        cleanup()


def block_freq_iso(a):
    """
    E8 -- the freq block re-measured on a geometry that actually isolates corners.

    d46 showed the published T_ctrl/2 coefficient (1.0921) does not survive
    dropping the points that break corner isolation: it moves to 1.2378, and the
    fit residual IMPROVES from 7.47% to 4.65%, which says the dropped points were
    not noise but physics the model does not contain.  Neither number is clean --
    one includes contaminated points, the other throws away the high-u shape
    information the fit needs.  The only way out is a geometry where the whole
    sweep is isolated, so nothing has to be dropped.

    Two changes from block_freq: side 10 m instead of 5 (seg/L 8.3 -> 16.7), and
    the sweep defined in u with its upper end set by u_isolated(), so every point
    is isolated BY CONSTRUCTION.  Laps drop 3.75 -> 2.75 to pay for the longer
    lap (lap-invariance was measured at 0.42%).
    """
    tau = a.tau_inject
    L = L_DEFAULT
    seg_over_L = a.side / L
    # RESUME.  This block used to start from rows = [] and so threw away every
    # completed frequency if it was interrupted -- and it is the longest block
    # here, four hours at the 17-point grid.  record() already writes the JSON
    # after every row, so the completed rows were on disk the whole time and only
    # the restart discarded them.
    out_name = f'block_freq_iso_tau{tau:.2f}.json'
    rows, done = load_partial(out_name, ('freq',))
    print(f'  E8: side {a.side} m, seg/L = {seg_over_L:.1f}, '
          f'u_iso = {u_isolated(seg_over_L):.4f}  (block_freq ran at seg/L 8.3, u_iso 0.389)')
    for freq in [20.0, 10.0, 5.0, 3.3333, 2.5, 2.0]:
        if (freq,) in done:
            print(f'  freq {freq:7.3f} Hz  already recorded, skipping')
            continue
        tt = tau + composite(freq, TSIM_DEFAULT)
        vp = U_STAR * L / tt
        print(f'  freq {freq:7.3f} Hz  T_ctrl/2={1/(2*freq):.4f}  tau_tot={tt:.4f}')
        if not launch(tau, freq=freq, vel=vp):
            continue
        # side is a run_point kwarg (path generation), not a launch kwarg
        curve = u_sweep(f'iso{freq:.2f}', tau, tt, L, seg_over_L,
                        npoints=a.npoints, laps=a.laps, side=a.side)
        v, how = locate(curve)
        rows.append({'freq': freq, 'T_ctrl_half': 1 / (2 * freq), 'tau_inject': tau,
                     'tau_tot': tt, 'v_pred': vp, 'v_obs': v, 'how': how,
                     'side': a.side, 'seg_over_L': seg_over_L,
                     'u_iso': u_isolated(seg_over_L),
                     'u_obs': (v * tt / L) if v else None, 'curve': curve})
        print(f'   => v* obs {v}  ({how})  u*={(v*tt/L) if v else float("nan"):.4f}')
        record(rows, out_name,
               {'block': 'controller_frequency, isolated geometry',
                'tau': tau, 'side': a.side, 'laps': a.laps})
    cleanup()


def block_alpha_iso(a):
    """
    E9 -- corner angle re-measured on a geometry that isolates corners, with the
    sweep defined in u and pushed to the top of the isolated range.

    Why this is not cleanup.  E5 found u* flat to 4.5% over alpha = 30-90 deg
    while the 0.5 cos(alpha/2) law would need 29.9%, and alpha = 120 deg had no
    interior minimum inside the swept range at all -- against d36's prediction of
    a DIP to 0.278 there.  Reading the reference engine sharpened the question: its crowd
    engine, a heading-servo plant, shows u* varying 2.43x over alpha = 0-120 deg,
    while Nav2, a curvature-command plant, shows 1.05x over 30-90 deg.  d38
    linearises the corner, so alpha cancels out of argmin and it predicts
    INVARIANCE for both classes.  One of those two measurements disagrees with
    the theory, and E5 never reached the angle where they would differ most.

    So: side 10 m (seg/L = 16.7 at L = 0.6), u-space sweep with the upper end at
    u_isolated, and the triangle included with enough range to bracket a minimum
    if one exists.
    """
    tau = a.tau_inject
    L = L_DEFAULT
    tt = tau + composite(FREQ_DEFAULT, TSIM_DEFAULT)
    rows, done = load_partial('block_alpha_iso.json', ('n_sides',))
    for n in [3, 4, 6, 8, 12]:
        if (n,) in done:
            continue
        alpha = 360.0 / n
        seg_over_L = a.side / L
        print(f'  n={n}  alpha={alpha:.0f} deg  side={a.side}  '
              f'seg/L={seg_over_L:.1f}  u_iso={u_isolated(seg_over_L):.4f}')
        if not launch(tau, vel=U_STAR * L / tt):
            continue
        curve = u_sweep(f'aiso{n}', tau, tt, L, seg_over_L,
                        npoints=a.npoints, laps=a.laps, u_cap=a.u_cap,
                        n_sides=n, side=a.side)
        v, how = locate(curve)
        rows.append({'n_sides': n, 'alpha_deg': alpha, 'tau_inject': tau,
                     'tau_tot': tt, 'v_obs': v, 'how': how,
                     'side': a.side, 'seg_over_L': seg_over_L,
                     'u_iso': u_isolated(seg_over_L),
                     'u_obs': (v * tt / L) if v else None,
                     'half_cos': 0.5 * math.cos(math.radians(alpha) / 2),
                     'curve': curve})
        print(f'   => v* obs {v}  ({how})  u*={(v*tt/L) if v else float("nan"):.4f}')
        record(rows, 'block_alpha_iso.json',
               {'block': 'corner_angle, isolated geometry', 'tau': tau,
                'side': a.side, 'laps': a.laps})
    cleanup()


def block_tsim_iso(a):
    """
    E10 -- the T_sim/2 term with a lever arm long enough to measure it.

    d48 fitted the published tsim block and got s = 1.0424, but the profile spans
    [0.840, 1.320] and leave-one-row-out has sd 0.309: dropping the single
    update_duration = 0.1 row moves s to 0.39.  One row carries the coefficient.
    The term itself is tiny (T_sim/2 = 0.0025..0.05 s against tau_tot ~ 0.23), so
    the 20x sweep in the ratio is only a 0.05 s sweep in the quantity.

    Here update_duration goes to 0.40 s (T_sim/2 = 0.2 s, comparable to tau
    itself), on the isolating geometry with the u-space sweep, so the term is
    large enough to stand on its own.
    """
    tau = a.tau_inject
    L = L_DEFAULT
    seg_over_L = a.side / L
    out_name = ('block_tsim_iso.json' if not a.sim_exe
                else 'block_tsim_iso_%s.json' % a.sim_exe)
    rows, done = load_partial(out_name, ('update_duration',))
    for ud in [0.01, 0.025, 0.05, 0.10, 0.20, 0.40]:
        if (ud,) in done:
            continue
        tt = tau + composite(FREQ_DEFAULT, ud)
        print(f'  update_duration {ud:.3f}  T_sim/2={ud/2:.4f}  tau_tot={tt:.4f}  '
              f'seg/L={seg_over_L:.1f}')
        if not launch(tau, tsim=ud, vel=U_STAR * L / tt, sim_exe=a.sim_exe):
            continue
        curve = u_sweep(f'tiso{ud:.3f}', tau, tt, L, seg_over_L,
                        npoints=a.npoints, laps=a.laps, u_cap=a.u_cap,
                        side=a.side)
        v, how = locate(curve)
        rows.append({'update_duration': ud, 'T_sim_half': ud / 2,
                     'tau_inject': tau, 'tau_tot': tt, 'v_obs': v, 'how': how,
                     'side': a.side, 'seg_over_L': seg_over_L,
                     'u_obs': (v * tt / L) if v else None, 'curve': curve})
        print(f'   => v* obs {v}  ({how})  u*={(v*tt/L) if v else float("nan"):.4f}')
        record(rows, out_name,
               {'block': 'update_duration, isolated geometry', 'tau': tau,
                'side': a.side, 'laps': a.laps})
    cleanup()


def block_actuator(a):
    """
    E12 -- a fourth term for Theorem 2, and the answer to "the plant has no
    dynamics".

    Every term of tau_tot measured so far is a DISCRETISATION: the injected
    transport delay, the controller's ZOH, the plant's ZOH.  A reviewer can
    grant all three and still object that the plant is an ideal integrator --
    real vehicles do not reach a commanded velocity instantly.  The usual
    answer is to re-run in a physics engine, which adds realism but adds it
    UNCONTROLLED: contact, friction, mass and a solver all arrive at once and
    none of them is separately measurable.

    Instead, add the omitted term alone and sweep it.  A first-order lag
        vdot = (u - v) / T_act
    has impulse response (1/T)exp(-t/T) whose first moment is exactly T, so
    Theorem 2's composition rule predicts, with NO free parameter,

        tau_tot = tau + T_ctrl/2 + T_sim/2 + T_act        <- coefficient 1

    PRE-REGISTERED before the first launch:
      * predicted coefficient s_act = 1.000; the null s_act = 0 is "actuator
        dynamics do not enter tau_tot at all";
      * u* stays at 0.306 across the sweep once T_act is booked, exactly as
        u* stayed put across the tau sweep (E1') -- the sweep is in u-space, so
        a term missing from the bookkeeping shows up as a DRIFT in u*, which is
        how E10 caught the Euler-integration excess;
      * the lever arm: T_act 0 -> 0.30 s against tau_tot ~ 0.23 s at T_act = 0,
        i.e. the term reaches ~1.3x the entire rest of the budget.  That is a
        larger lever than E10 had.

    This also puts the experiment inside the family Heredia & Ollero (2007)
    analyse, whose stability results are nondimensionalised by exactly such a
    steering time constant -- so a confirmation here connects the this study
    optimum to published stability limits rather than only to our own.

    NOTE: this build is a DIAGNOSTIC, like loopback_midpoint.  Headline results
    stay on the shipped plant; this block reports what happens when a known
    dynamic term is added to it.
    """
    tau = a.tau_inject
    L = L_DEFAULT
    seg_over_L = a.side / L
    rows, done = load_partial('block_actuator.json', ('actuator_tau',))
    for at in [0.0, 0.025, 0.05, 0.10, 0.20, 0.30]:
        if (at,) in done:
            continue
        # tau_tot includes the actuator term with the coefficient Theorem 2
        # predicts (1.0).  If that is wrong, u* drifts with T_act and the fit
        # returns a coefficient away from 1 -- the drift is the measurement.
        tt = tau + composite(FREQ_DEFAULT, TSIM_DEFAULT) + at
        print(f'  actuator_tau {at:.3f}  tau_tot={tt:.4f}  seg/L={seg_over_L:.1f}')
        if not launch(tau, vel=U_STAR * L / tt, sim_exe='loopback_actuator',
                      act_tau=at):
            continue
        curve = u_sweep(f'act{at:.3f}', tau, tt, L, seg_over_L,
                        npoints=a.npoints, laps=a.laps, u_cap=a.u_cap,
                        side=a.side)
        v, how = locate(curve)
        rows.append({'actuator_tau': at, 'tau_inject': tau, 'tau_tot': tt,
                     'v_obs': v, 'how': how, 'side': a.side,
                     'seg_over_L': seg_over_L,
                     'u_obs': (v * tt / L) if v else None, 'curve': curve})
        print(f'   => v* obs {v}  ({how})  u*={(v*tt/L) if v else float("nan"):.4f}')
        record(rows, 'block_actuator.json',
               {'block': 'actuator lag, isolated geometry', 'tau': tau,
                'side': a.side, 'laps': a.laps, 'plant': 'loopback_actuator'})
    cleanup()


def block_bicycle(a):
    """
    E13 -- Ackermann curvature saturation.

    A kinematic bicycle with instantaneous, unbounded steering is algebraically
    the SAME plant as a unicycle: delta = atan(omega W / v) fed through
    omega = v tan(delta)/W returns omega.  So swapping the plant for a bicycle
    tests nothing on its own.  The kinematic difference lives in the steering
    ANGLE limit, which bounds the curvature at kappa_max = tan(max_steer)/W --
    a saturation, and the one thing no block has touched.  (The steering RATE
    limit is a first-order lag, and E12 already measured its coefficient.)

    PRE-REGISTERED, from the controller's own curvature law.  RPP commands
    kappa = 2y/d^2 with d = lookahead; the worst case at a corner is y = d, so

        kappa_corner = 2/L = 3.333 1/m   (minimum radius 0.30 m)

    At W = 0.5 m that is max_steer = atan(kappa_corner * W) = 1.030 rad.  So:

      * max_steer = 0 (bound off) must reproduce the shipped plant exactly --
        this row is the block's own control;
      * max_steer >= 1.03 rad should be indistinguishable from it;
      * below that the corner cannot be taken at the commanded curvature and
        u* must move -- the question is whether it drifts smoothly or the law
        fails outright.

    Everything else is E1'/E8-E12's geometry: side 10 m, u-space sweep capped at
    u_isolated, tau_inject 0.10 s.
    """
    tau = a.tau_inject
    L = L_DEFAULT
    seg_over_L = a.side / L
    rows, done = load_partial('block_bicycle.json', ('max_steer',))
    for ms in [0.0, 1.20, 1.00, 0.85, 0.70, 0.55]:
        if (ms,) in done:
            continue
        tt = tau + composite(FREQ_DEFAULT, TSIM_DEFAULT)
        kmax = (math.tan(ms) / a.wheelbase) if ms > 0 else float('inf')
        print(f'  max_steer {ms:.2f} rad  kappa_max {kmax:.3f} 1/m  '
              f'(corner needs {2.0 / L:.3f})  tau_tot={tt:.4f}')
        if not launch(tau, vel=U_STAR * L / tt, sim_exe='loopback_bicycle',
                      wheelbase=a.wheelbase, max_steer=ms):
            continue
        curve = u_sweep(f'bike{ms:.2f}', tau, tt, L, seg_over_L,
                        npoints=a.npoints, laps=a.laps, u_cap=a.u_cap,
                        side=a.side)
        v, how = locate(curve)
        rows.append({'max_steer': ms, 'wheelbase': a.wheelbase,
                     'kappa_max': (kmax if ms > 0 else None),
                     'kappa_corner': 2.0 / L, 'saturates': bool(ms > 0 and kmax < 2.0 / L),
                     'tau_inject': tau, 'tau_tot': tt, 'v_obs': v, 'how': how,
                     'side': a.side, 'seg_over_L': seg_over_L,
                     'u_obs': (v * tt / L) if v else None, 'curve': curve})
        print(f'   => v* obs {v}  ({how})  u*={(v*tt/L) if v else float("nan"):.4f}')
        record(rows, 'block_bicycle.json',
               {'block': 'Ackermann curvature saturation', 'tau': tau,
                'side': a.side, 'laps': a.laps, 'plant': 'loopback_bicycle',
                'wheelbase': a.wheelbase})
    cleanup()


def block_tau_iso(a):
    """
    E1' -- the tau sweep, re-run on the fixed harness.

    The original tau sweep is the only block still marked PRELIMINARY: its
    measurement window contained 1 m of the exit stub, the lap count changed
    inside the tau = 0.6 row, and 5 of its 36 points break corner isolation.
    d50 fitted a coefficient of 0.9713 on tau_inject from it -- below 1, where
    the two ZOH terms come out above 1 -- and that asymmetry is the argument
    that rejected the "J(u) is off by a global factor" hypothesis.  It should
    not rest on the preliminary dataset.

    Same treatment as E8/E9/E10: side 10 m, u-space sweep capped at u_isolated.
    """
    tau = 0.0
    L = L_DEFAULT
    seg_over_L = a.side / L
    rows, done = load_partial('block_tau_iso.json', ('tau_inject',))
    for tau in [0.05, 0.10, 0.20, 0.40, 0.80]:
        if (tau,) in done:
            continue
        tt = tau + composite(FREQ_DEFAULT, TSIM_DEFAULT)
        print(f'  tau_inject {tau:.3f}  tau_tot={tt:.4f}  seg/L={seg_over_L:.1f}')
        if not launch(tau, vel=U_STAR * L / tt):
            continue
        curve = u_sweep(f'tauiso{tau:.2f}', tau, tt, L, seg_over_L,
                        npoints=a.npoints, laps=a.laps, u_cap=a.u_cap,
                        side=a.side)
        v, how = locate(curve)
        rows.append({'tau_inject': tau, 'tau_tot': tt, 'v_obs': v, 'how': how,
                     'side': a.side, 'seg_over_L': seg_over_L,
                     'u_obs': (v * tt / L) if v else None, 'curve': curve})
        print(f'   => v* obs {v}  ({how})  u*={(v*tt/L) if v else float("nan"):.4f}')
        record(rows, 'block_tau_iso.json',
               {'block': 'tau sweep, isolated geometry', 'side': a.side,
                'laps': a.laps})
    cleanup()


def block_replicate_iso(a):
    """
    E7'' -- the reproducibility measurement, repeated on the isolating geometry.

    d63 ruled out the partial-corner window as the source of E7's clustered
    scatter: scoring over whole laps changed the sd by 1%.  But the per-speed
    breakdown of E7' points somewhere else.  The scatter across five identical
    runs is 0.34% at the optimum and 13.2% at the top of the sweep, and the top
    of that sweep (u = 0.3925) is the only point ABOVE the isolation limit
    u_iso = 0.389 for seg/L = 8.3.  The same pattern appeared in d56's
    cross-block comparison: agreement is excellent at the optimum and collapses
    above it, exactly where corners stop being isolated.

    So the hypothesis is now that the reproducibility floor itself is corner
    interaction, not measurement noise.  Test: run the identical row five times
    on side 10 m (seg/L 16.7, u_iso 0.443) with the u-space sweep, where no point
    violates isolation, and see whether the sd falls below the 1.1-1.5% the
    seg/L = 8.3 geometry gives.

    This matters out of proportion to its cost: 1.52% is the denominator every
    null control and every coefficient interval in this study is measured against.
    """
    tau = a.tau_inject
    L = L_DEFAULT
    seg_over_L = a.side / L
    tt = tau + composite(FREQ_DEFAULT, TSIM_DEFAULT)
    rows, done = load_partial('block_replicate_iso.json', ('rep',))
    for rep in range(a.reps):
        if (rep,) in done:
            continue
        print(f'  REPLICATE {rep + 1}/{a.reps}  (fresh stack, seg/L={seg_over_L:.1f})')
        if not launch(tau, vel=U_STAR * L / tt):
            continue
        curve = u_sweep(f'repiso{rep}', tau, tt, L, seg_over_L,
                        npoints=a.npoints, laps=a.laps, u_cap=a.u_cap,
                        side=a.side)
        v, how = locate(curve)
        rows.append({'rep': rep, 'tau_inject': tau, 'tau_tot': tt,
                     'v_obs': v, 'how': how, 'side': a.side,
                     'seg_over_L': seg_over_L, 'u_iso': u_isolated(seg_over_L),
                     'u_obs': (v * tt / L) if v else None, 'curve': curve})
        record(rows, 'block_replicate_iso.json',
               {'block': 'row replicates, isolated geometry', 'tau': tau,
                'side': a.side, 'laps': a.laps, 'reps': a.reps})
        cleanup()


def block_nulls(a):
    """
    Knobs that MUST NOT move u*.  If any of them does, the measured optimum is a
    property of the harness, not of the control loop.
    """
    tau = 0.20
    tt = tau + composite(FREQ_DEFAULT, TSIM_DEFAULT)
    vp = U_STAR * L_DEFAULT / tt
    rows = []

    for costmap, search in [(3.0, 1.5), (5.0, 2.5), (8.0, 4.0)]:
        print(f'  NULL costmap={costmap} m (extent {costmap/2}), search={search} m')
        if not launch(tau, costmap=costmap, search=search, vel=vp):
            continue
        curve = v_sweep(f'null_cm{costmap:.0f}', tau, vp, npoints=5, span=0.25)
        v, how = locate(curve)
        rows.append({'knob': 'costmap_size', 'costmap': costmap, 'search_dist': search,
                     'v_obs': v, 'how': how,
                     'u_obs': (v * tt / L_DEFAULT) if v else None, 'curve': curve})
        record(rows, 'block_nulls.json', {'block': 'null controls', 'tau': tau})

    if launch(tau, vel=vp):
        for hz in [25.0, 50.0, 100.0]:
            print(f'  NULL sample_hz={hz}')
            curve = v_sweep(f'null_hz{hz:.0f}', tau, vp, npoints=5, span=0.25,
                            sample_hz=hz)
            v, how = locate(curve)
            rows.append({'knob': 'sample_hz', 'sample_hz': hz, 'v_obs': v, 'how': how,
                         'u_obs': (v * tt / L_DEFAULT) if v else None, 'curve': curve})
            record(rows, 'block_nulls.json', {'block': 'null controls', 'tau': tau})
        for tol in [0.05, 0.1, 0.3]:
            print(f'  NULL transform_tolerance={tol}')
            if not set_param('/controller_server', 'FollowPath.transform_tolerance', tol):
                continue
            curve = v_sweep(f'null_tol{tol:.2f}', tau, vp, npoints=5, span=0.25)
            v, how = locate(curve)
            rows.append({'knob': 'transform_tolerance', 'tol': tol, 'v_obs': v,
                         'how': how, 'u_obs': (v * tt / L_DEFAULT) if v else None,
                         'curve': curve})
            record(rows, 'block_nulls.json', {'block': 'null controls', 'tau': tau})
    cleanup()


GRACEFUL_PROBE = r"""
import json, math, os, rclpy, time as _t
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
class P(Node):
    def __init__(self):
        super().__init__('letg2_graceful_probe')
        self.sp=[]; self.tg=[]; self.ct=[]
        self.create_subscription(Odometry,'/odom',self.o,10)
        self.create_subscription(PoseStamped,'/motion_target',self.t,10)
        self.create_subscription(Twist,'/cmd_vel_nav',self.c,10)
    def o(self,m):
        self.sp.append(math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y))
    def t(self,m):
        self.tg.append(math.hypot(m.pose.position.x, m.pose.position.y))
    def c(self,m):
        self.ct.append(_t.time())
rclpy.init(); n=P()
t0=_t.time()
# DURATION is only a backstop.  The parent drops STOPFILE the moment run_point
# returns; without that the probe's own timer -- always set generously above the
# run -- would set the pace of the whole block and waste more wall clock than
# the measurements take.
while _t.time()-t0 < DURATION and not os.path.exists('STOPFILE'):
    rclpy.spin_once(n, timeout_sec=0.1)
sp=[x for x in n.sp if x>1e-3]
d=sorted(b-a for a,b in zip(n.ct, n.ct[1:]) if 1e-4 < (b-a) < 2.0)
def pct(x,p):
    return x[min(len(x)-1, int(p*len(x)))] if x else 0.0
print('PROBE '+json.dumps({
  'n_moving':len(sp), 'v_mean':(sum(sp)/len(sp) if sp else 0.0),
  'n_target':len(n.tg), 'L_eff':(sum(n.tg)/len(n.tg) if n.tg else 0.0),
  'n_cycles':len(d), 'T_ctrl':(sum(d)/len(d) if d else 0.0),
  'T_ctrl_p95':pct(d,0.95)}))
"""


def graceful_probe(dur):
    """Measure achieved speed, effective look-ahead and control period.

    None of the three is what the configuration says on this controller, and
    each enters u = v tau_tot / L directly.  See PREREG_GRACEFUL_2026-09-08.md.
    """
    src = f'{OUT_DIR}/graceful_probe.py'
    stop = f'{OUT_DIR}/graceful_probe.stop'
    if os.path.exists(stop):
        os.remove(stop)
    open(src, 'w').write(GRACEFUL_PROBE.replace('DURATION', '%.1f' % dur)
                         .replace('STOPFILE', stop))
    return subprocess.Popen(['python3', src], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)


def graceful_probe_read(pr, dur):
    open(f'{OUT_DIR}/graceful_probe.stop', 'w').write('stop')
    try:
        out = pr.communicate(timeout=dur + 60)[0]
    except Exception:
        pr.kill()
        return {}
    for line in out.splitlines():
        if line.startswith('PROBE '):
            return json.loads(line[6:])
    return {}


def gset(name, value):
    return set_param('/controller_server', f'FollowPath.{name}', value)


def graceful_row(tag, tau, vs, laps=2.75):
    """One k_phi row: run the speed list, measuring the denominators each time."""
    curve = []
    for v in vs:
        if not gset('v_linear_max', v):
            print(f'      v={v:7.4f}  !! could not set v_linear_max; aborting row')
            break
        time.sleep(0.4)
        vb = get_param('/controller_server', 'FollowPath.v_linear_max')
        if vb is None or abs(vb - v) > 1e-6:
            print(f'      v={v:7.4f}  !! controller reports {vb}; aborting row')
            break
        dur = 20.0 * laps / max(v, 0.05) * 1.8 + 60
        pr = graceful_probe(dur)
        time.sleep(1.0)
        d = run_point(f'{tag}_v{v:.4f}', tau, v, laps)
        pb = graceful_probe_read(pr, dur)
        if d is None:
            print(f'      v={v:7.4f}  (no output)')
            continue
        if d.get('goal_status') != 4:
            print(f'      v={v:7.4f}  goal_status={d.get("goal_status")} -- excluded')
            continue
        if d.get('truncated'):
            print(f'      v={v:7.4f}  TRUNCATED -- excluded')
            continue
        ratio = (pb.get('v_mean', 0.0) / v) if v else 0.0
        curve.append({'v_set': v, 'v_achieved': pb.get('v_mean'),
                      'speed_ratio': ratio, 'L_eff': pb.get('L_eff'),
                      'T_ctrl': pb.get('T_ctrl'), 'T_ctrl_p95': pb.get('T_ctrl_p95'),
                      'v': pb.get('v_mean') or v,
                      'rmse': d['rmse_m'], 'rmse_intlap': d.get('rmse_m_intlap'),
                      'n_used': d['n_samples_used'], 'wall_s': d['wall_s'],
                      'tau_applied_mean': d.get('tau_applied_mean')})
        print(f'      v={v:7.4f}  achieved {pb.get("v_mean", 0):6.4f} '
              f'({ratio:5.3f})  L_eff {pb.get("L_eff", 0):6.4f}  '
              f'T_ctrl {pb.get("T_ctrl", 0):6.4f}  RMSE {d["rmse_m"]:8.5f}  '
              f'{d["wall_s"]:5.0f}s')
    return curve


def block_graceful(a):
    """
    E14 -- nav2_graceful_controller: a predicted TRANSITION, not a number.

    d97 linearises the Park-Kuipers law straight out of the plugin source and
    finds that this controller and RPP are the SAME delay equation at different
    points of one family:

        e'' = -(A+B) e(x-u) - A e'(x-u) + (A+B) g(x-u) - B P(x-u)
        A = k_delta + 1 + k_phi,  B = k_delta k_phi,  and RPP is (2, 0)

    (the A=2, B=0 reduction reproduces d38's curvature-command integrator to
    4.4e-16, and the RPP anchor comes back at u* 0.3100 / u_c 0.5202 against the
    published 0.3078 / 0.520494; d131).

    PRE-REGISTERED (PREREG_GRACEFUL_2026-09-08.md):

        k_phi 0.5  (A 3.5, B 1)   u_c 0.3336   u* 0.2270
        k_phi 1.0  (A 4,   B 2)   u_c 0.2900   u* 0.0995
        k_phi 3.0  (A 6,   B 6)   u_c 0.2005   NO interior optimum   <- shipped
        k_phi 8.0  (A 11,  B 16)  u_c 0.1204   NO interior optimum

    so the claim under test is that the optimum DISAPPEARS between k_phi 1 and 2,
    and the mechanism is named: the decomposition row (A, B) = (6, 0) keeps an
    optimum at 0.0980, so it is the tangent feedforward B, not the gain A, that
    removes it.

    Three denominators are measured per point, because none is what the config
    says: the achieved speed (beta modulates it; this block sets beta = 0 and
    requires the ratio to be 1), the effective look-ahead (target chosen by path
    distance, law driven by Euclidean distance -- 0.514 measured against a
    configured 0.600) and the control period (this controller simulates a
    trajectory per candidate per cycle).
    """
    tau = a.tau_inject
    L_nom = L_DEFAULT
    rows, done = load_partial('block_graceful.json', ('k_phi', 'beta'))
    if not launch(tau, vel=0.5, controller='graceful', timeout=90):
        print('  !! launch failed')
        return
    # beta = 0 makes the swept parameter the speed; the wide angular bound keeps
    # the omega clamp (which would silently rescale v) out of the experiment;
    # prefer_final_rotation stops the terminal sweeping turn that aborted the
    # first drive.  All three are recorded in every row.
    for k, v in (('beta', 0.0), ('lambda', 2.0), ('v_angular_max', 50.0),
                 ('prefer_final_rotation', 'true'), ('k_delta', 2.0)):
        if not gset(k, v):
            print(f'  !! could not set {k}')
            return

    # tau_tot uses the MEASURED control period once it exists; until the first
    # row runs, the nominal value seeds the speed grid.
    tau_tot = tau + composite(FREQ_DEFAULT, TSIM_DEFAULT)
    L_use = a.graceful_L if a.graceful_L > 0 else L_nom

    # (k_phi, u_c, u_lo, u_hi, note).  The window is NOT +-40% of one prediction:
    # the linearised derivation (d97) and the exact-law engine (d99) disagree by
    # a factor of two on where the optimum sits (0.2270 vs 0.1118 at k_phi 0.5),
    # because the square's 90 deg corner saturates the plugin's atan and sin.  A
    # window bracketing both lets the data choose between them.  Upper end stays
    # under 0.85 u_c.
    plan = [
        (0.5, 0.3336, 0.05, 0.28, 'optimum predicted: 0.2270 lin / 0.1118 exact'),
        (1.0, 0.2900, 0.04, 0.24, 'optimum predicted: 0.0995 lin / 0.0778 exact'),
        (2.0, 0.2350, 0.04, 0.19, 'NO optimum predicted (both derivations)'),
        (3.0, 0.2005, 0.04, 0.16, 'SHIPPED -- no optimum predicted (both)'),
    ]
    for k_phi, u_c, u_lo, u_hi, note in plan:
        if (k_phi, 0.0) in done:
            print(f'  k_phi {k_phi}: already done, skipping')
            continue
        if not gset('k_phi', k_phi):
            print(f'  !! could not set k_phi {k_phi}')
            break
        time.sleep(0.4)
        kb = get_param('/controller_server', 'FollowPath.k_phi')
        if kb is None or abs(kb - k_phi) > 1e-6:
            print(f'  !! k_phi reads {kb}; aborting block')
            break
        A = 2.0 + 1.0 + k_phi
        B = 2.0 * k_phi
        print(f'\n  k_phi {k_phi:.1f}  (A {A:.1f}, B {B:.1f})  u_c {u_c:.4f}  -- {note}')
        assert u_hi < 0.85 * u_c, 'sweep window reaches the stability boundary'
        vs = [(u_lo + (u_hi - u_lo) * i / (a.npoints - 1)) * L_use / tau_tot
              for i in range(a.npoints)]
        print(f'      sweeping u {u_lo:.3f}..{u_hi:.3f} -> v {vs[0]:.4f}..'
              f'{vs[-1]:.4f} m/s  (L {L_use:.4f}, tau_tot {tau_tot:.4f})')
        curve = graceful_row(f'grace_kphi{int(k_phi * 10):03d}', tau, vs)
        v, how, npts = bowl_locate_curve(curve)
        rows.append({'k_phi': k_phi, 'k_delta': 2.0, 'beta': 0.0,
                     'A': A, 'B': B, 'u_c_pred': u_c, 'u_window': [u_lo, u_hi],
                     'tau_tot_nominal': tau_tot, 'L_nominal': L_use,
                     'v_obs': v, 'how': how, 'n_bowl': npts, 'curve': curve})
        record(rows, 'block_graceful.json',
               {'block': 'graceful controller (A,B) family',
                'tau': tau, 'prereg': 'PREREG_GRACEFUL_2026-09-08.md'})

    # control row: the controller exactly as shipped, beta included, so the
    # paper can say what the out-of-the-box plugin does and at what speed.
    if (3.0, 0.4) not in done:
        for k, v in (('beta', 0.4), ('k_phi', 3.0), ('v_angular_max', 5.0)):
            gset(k, v)
        time.sleep(0.4)
        print('\n  SHIPPED control row: beta 0.4, k_phi 3.0, v_angular_max 5.0')
        vs = [(0.04 + (0.16 - 0.04) * i / 4) * L_use / tau_tot for i in range(5)]
        curve = graceful_row('grace_shipped', tau, vs)
        rows.append({'k_phi': 3.0, 'k_delta': 2.0, 'beta': 0.4,
                     'A': 6.0, 'B': 6.0, 'u_c_pred': 0.2005, 'u_star_pred': None,
                     'tau_tot_nominal': tau_tot, 'L_nominal': L_use,
                     'v_obs': None, 'how': 'control', 'curve': curve})
        record(rows, 'block_graceful.json',
               {'block': 'graceful controller (A,B) family',
                'tau': tau, 'prereg': 'PREREG_GRACEFUL_2026-09-08.md'})
    cleanup()


def bowl_locate_curve(curve):
    """argmin of RMSE over the MEASURED speed, with the edge test kept.

    An edge verdict is the prediction for three of the four rows, so it must be
    reported as such and not smoothed into an interior number.
    """
    pts = sorted((c['v'], c['rmse']) for c in curve
                 if c.get('rmse') is not None and c['rmse'] == c['rmse'])
    if len(pts) < 4:
        return None, 'too few points', 0
    vs = [p[0] for p in pts]
    rs = [p[1] for p in pts]
    i = min(range(len(rs)), key=lambda k: rs[k])
    if i == 0:
        return vs[0], 'EDGE_LOW', len(pts)
    if i == len(rs) - 1:
        return vs[-1], 'EDGE_HIGH', len(pts)
    lo = min(rs) * 1.35
    sel = [k for k in range(len(rs)) if rs[k] <= lo]
    if len(sel) >= 3:
        import numpy as _np
        c = _np.polyfit([vs[k] for k in sel], [rs[k] for k in sel], 2)
        if c[0] > 0:
            v = -c[1] / (2 * c[0])
            if vs[0] <= v <= vs[-1]:
                return float(v), 'bowl', len(sel)
    return vs[i], 'grid', len(pts)


BLOCKS = {'freq': block_freq, 'tsim': block_tsim, 'lookahead': block_lookahead,
          'alpha': block_alpha, 'nulls': block_nulls,
          'replicate': block_replicate,
          'alpha_iso': block_alpha_iso,
          'tsim_iso': block_tsim_iso,
          'tau_iso': block_tau_iso,
          'replicate_iso': block_replicate_iso,
          'actuator': block_actuator,
          'bicycle': block_bicycle,
          'graceful': block_graceful,
          'freq_iso': block_freq_iso}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', default='freq,tsim,lookahead,alpha,nulls')
    ap.add_argument('--sim-exe', default=None, dest='sim_exe',
                    help="plant node; 'loopback_midpoint' for the midpoint-heading "
                         'diagnostic build (default: upstream nav2_loopback_sim)')
    ap.add_argument('--wheelbase', type=float, default=0.5,
                    help='bicycle wheelbase; kappa_max = tan(max_steer)/W')
    ap.add_argument('--u-cap', type=float, default=0.85, dest='u_cap',
                    help='u-sweep upper end as a fraction of u_c = 0.520494')
    ap.add_argument('--tau-inject', type=float, default=0.10, dest='tau_inject',
                    help='injected delay for freq_iso; E11 compares 0.05 / 0.10 / 0.20')
    ap.add_argument('--side', type=float, default=SIDE,
                    help='polygon side length in metres (E8 uses 10.0)')
    ap.add_argument('--laps', type=float, default=2.75,
                    help='laps per point for u-space blocks; never an integer')
    ap.add_argument('--reps', type=int, default=4,
                    help='block replicate: how many identical rows')
    ap.add_argument('--npoints', type=int, default=9)
    ap.add_argument('--graceful-L', type=float, default=0.0, dest='graceful_L',
                    help='measured effective look-ahead for the graceful block; '
                         '0 uses the nominal L, but the self-test measures it '
                         '(0.514 against a configured 0.600) and that value is '
                         'what u must be formed with')
    ap.add_argument('--span', type=float, default=0.40)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in a.block.split(','):
        name = name.strip()
        if name not in BLOCKS:
            print(f'unknown block: {name}')
            continue
        print(f'\n{"="*72}\nBLOCK: {name}\n{"="*72}')
        t0 = time.time()
        BLOCKS[name](a)
        print(f'block {name} finished in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
