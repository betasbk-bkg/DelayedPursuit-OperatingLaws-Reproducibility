#!/usr/bin/env python3
"""
probe_track_unknown -- did adding `track_unknown_space: False` to the local
costmap change what RPP does, or is the freq_iso re-run difference session noise?

Runs ONLY after the freq_iso block has exited (it launches its own stack).

Two configurations, same 10 Hz stack, same three fast-side speeds where the
re-run differed from the original by -17.6%, -28.8% and -28.2%:

  STOCK_KEY   the key removed, so Costmap2D's own default applies (what the
              original 2026-09-06 run had)
  FALSE_KEY   the key set False (what the 2026-09-13 re-run had)

For each, the parameter the running node actually holds is read back first --
if it reports False under STOCK_KEY, the harness change was a no-op and no
driving is needed to settle the caveat; the points are run anyway so the
answer does not rest on the read-back alone.

Usage (WSL):  python3 -u probe_track_unknown.py RERUN.json ORIGINAL.json OUT.json
"""
import json
import os
import sys
import time

import yaml

WS_PKG = '/home/betas/letg2_ws/src/letg2_nav2_validation/letg2_nav2_validation'
sys.path.insert(0, WS_PKG)
import experiment as E  # noqa: E402

FREQ, TAU, SIDE, LAPS, L = 10.0, 0.10, 10.0, 2.75, 0.6
U_TARGETS = (0.3789, 0.4107, 0.4424)


def launch_variant(mode, vel):
    E.cleanup()
    pf = f'{E.OUT_DIR}/params_probe.yaml'
    r = E.sh(f'python3 {E.PKG_SRC}/letg2_nav2_validation/make_params.py '
             f'--vel {vel} --lookahead {L} --freq {FREQ} --costmap-size 5.0 '
             f'--search-dist 2.5 --controller rpp --out {pf}')
    if r.returncode != 0:
        print('   !! make_params failed', r.stderr[-300:])
        return False
    p = yaml.safe_load(open(pf))
    lc = p['local_costmap']['local_costmap']['ros__parameters']
    if mode == 'STOCK_KEY':
        lc.pop('track_unknown_space', None)
    else:
        lc['track_unknown_space'] = False
    yaml.safe_dump(p, open(pf, 'w'), sort_keys=False)
    log = f'{E.OUT_DIR}/stack_probe.log'
    E.sh(f'setsid nohup ros2 launch letg2_nav2_validation pursuit_stack.launch.py '
         f'params_file:={pf} delay_sec:={TAU} update_duration:={E.TSIM_DEFAULT} '
         f'> {log} 2>&1 < /dev/null &')
    t0 = time.time()
    while time.time() - t0 < 60:
        time.sleep(3)
        if 'Managed nodes are active' in open(log, errors='ignore').read():
            time.sleep(2)
            return E.single_stack()
    print('   !! stack did not become active')
    return False


def main():
    rerun, orig, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    rows_n = {round(r['freq'], 3): r for r in json.load(open(rerun))['rows']}
    rows_o = {round(r['freq'], 3): r for r in json.load(open(orig))['rows']}
    rn, ro = rows_n[FREQ], rows_o[FREQ]
    k = rn['u_obs'] / rn['v_obs']
    by_u_n = {round(c['v'] * k, 4): c for c in rn['curve']}
    by_u_o = {round(c['v'] * k, 4): c for c in ro['curve']}
    tt = rn['tau_tot']
    out = {'freq': FREQ, 'tau': TAU, 'runs': []}
    for mode in ('STOCK_KEY', 'FALSE_KEY'):
        print(f'\n== {mode}')
        if not launch_variant(mode, vel=E.U_STAR * L / tt):
            out['runs'].append({'mode': mode, 'error': 'launch failed'})
            continue
        r = E.sh('ros2 param get /local_costmap/local_costmap track_unknown_space',
                 timeout=60)
        held = r.stdout.strip()
        print(f'   node holds: {held}')
        for u in U_TARGETS:
            c_n, c_o = by_u_n[u], by_u_o[u]
            v = c_n['v']
            if abs(c_o['v'] / v - 1) > 1e-9:
                raise SystemExit('original and re-run speeds differ at u=%s' % u)
            E.set_param('/controller_server', 'FollowPath.desired_linear_vel', v)
            E.set_param('/letg2_delay_node', 'delay_sec', TAU)
            time.sleep(0.5)
            d = E.run_point(f'probe_{mode}_u{u:.4f}', TAU, v, LAPS, side=SIDE)
            rm = d.get('rmse_m') if d else None
            print(f'   u={u:.4f}  rmse {rm}   original {c_o["rmse"]:.5f}   '
                  f're-run {c_n["rmse"]:.5f}   goal_status {d.get("goal_status") if d else None}')
            out['runs'].append({'mode': mode, 'node_holds': held, 'u': u, 'v': v,
                                'rmse': rm, 'rmse_original': c_o['rmse'],
                                'rmse_rerun': c_n['rmse'],
                                'goal_status': d.get('goal_status') if d else None,
                                'truncated': d.get('truncated') if d else None})
            json.dump(out, open(out_path, 'w'), indent=2)
    E.cleanup()
    print('-> %s' % out_path)


if __name__ == '__main__':
    main()
