#!/usr/bin/env python3
"""
d45 -- Audit of the project's own JUDGMENT CRITERIA.

The E6 post-mortem (d44) found that its PASS/FAIL threshold was a round number we
chose, defended by a quantity of the wrong kind.  That is a class of defect, not a
one-off, so this script sweeps every numeric criterion the project uses and asks of
each: WHERE DOES THIS NUMBER COME FROM?

Two mechanised parts.

PART A -- the 93 tolerances in audit_2026-08-29.py.
    That script is the report <-> JSON gate: it asserts abs(got - want) <= tol.
    For a pure transcription check the ONLY defensible tolerance is round-off:
    half a unit in the last place of `want` AS THE REPORT PRINTS IT.  Anything
    looser is not checking transcription, it is silently granting scientific
    slack -- and a check that passes with 30% slack is not a check.
    So: parse every call, recover the printed precision of `want`, and classify

      ROUNDOFF   tol <= 0.5 ulp            -- transcription check, defensible
      TIGHT      tol <= 5 ulp              -- round-off plus a little; acceptable
      LOOSE      tol >  5 ulp              -- a scientific tolerance in disguise;
                                             must be justified individually
    and rank the LOOSE ones by how much slack they grant relative to |want|.

PART B -- the experiment-side design constants and pass criteria.
    These cannot be parsed out of code (they live in prose), so they are listed
    here explicitly with their basis, and the ones that are DERIVED get recomputed
    from first principles right here so the claim "derived" is not just a label.

Basis classes used throughout:
    DERIVED    follows from the theory or from arithmetic; no freedom
    MEASURED   fixed by a measurement in this experiment; reproducible
    STRUCTURAL forced by the software or the geometry (violate it and it breaks)
    CHOSEN     we picked it.  Must be flagged, and either grounded or removed.

Usage:  python3 d45_criteria_audit_2026-09-06.py
"""
import ast
import json
import math
import pathlib
import re

import numpy as np

HERE = pathlib.Path(__file__).parent
AUDIT = HERE / 'audit_2026-08-29.py'


# --------------------------------------------------------------------------
# PART A -- tolerances in the report <-> JSON gate
# --------------------------------------------------------------------------
def ulp_of_literal(src):
    """Half a unit in the last printed place of a numeric literal AS WRITTEN."""
    s = src.strip()
    if 'e' in s.lower():
        mant, _, exp = s.lower().partition('e')
        dec = len(mant.split('.')[1]) if '.' in mant else 0
        return 0.5 * 10 ** (-dec + int(exp))
    if '.' in s:
        return 0.5 * 10 ** (-len(s.split('.')[1]))
    return 0.5


def part_a():
    tree = ast.parse(AUDIT.read_text(encoding='utf-8'))
    src = AUDIT.read_text(encoding='utf-8').splitlines()
    rows = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'check' and len(node.args) >= 4):
            continue
        want_n, tol_n = node.args[2], node.args[3]
        try:
            tol = float(ast.literal_eval(tol_n))
        except Exception:
            continue
        want_src = ast.get_source_segment('\n'.join(src), want_n)
        try:
            want = float(ast.literal_eval(want_n))
            literal_want = True
        except Exception:
            want, literal_want = None, False
        # the printed precision belongs to `want` only when it is a literal;
        # otherwise fall back to the tolerance's own written precision
        ulp = ulp_of_literal(want_src) if (literal_want and want_src) else None
        lab = node.args[0]
        try:
            label = ast.literal_eval(lab)
        except Exception:
            label = ast.get_source_segment('\n'.join(src), lab) or '<f-string>'
        label = re.sub(r'\s+', ' ', str(label))[:60]
        rows.append({'line': node.lineno, 'label': label, 'want': want,
                     'want_src': want_src, 'tol': tol, 'ulp': ulp,
                     'ratio': (tol / ulp) if ulp else None,
                     'rel_slack_pct': (100 * tol / abs(want))
                     if (want not in (None, 0)) else None})
    for r in rows:
        # want in {0,1} with tol 0.5 is a BOOLEAN assertion encoded numerically,
        # not a loose tolerance: it can only pass if the flag has the right value.
        if r['want'] in (0.0, 1.0) and abs(r['tol'] - 0.5) < 1e-12:
            r['class'] = 'BOOLEAN'
        elif r['ratio'] is None:
            r['class'] = 'NON-LITERAL'
        elif r['ratio'] <= 1.0:
            r['class'] = 'ROUNDOFF'
        elif r['ratio'] <= 5.0:
            r['class'] = 'TIGHT'
        else:
            r['class'] = 'LOOSE'

    print('=' * 78)
    print('PART A -- the %d tolerances in audit_2026-08-29.py' % len(rows))
    print('=' * 78)
    counts = {}
    for r in rows:
        counts[r['class']] = counts.get(r['class'], 0) + 1
    for c in ('ROUNDOFF', 'TIGHT', 'LOOSE', 'BOOLEAN', 'NON-LITERAL'):
        print('  %-12s %3d' % (c, counts.get(c, 0)))

    loose = sorted([r for r in rows if r['class'] == 'LOOSE'],
                   key=lambda r: -(r['rel_slack_pct'] or 0))
    print('\n  LOOSE checks ranked by the slack they grant, relative to the value')
    print('  (a check whose tolerance is a large fraction of the value cannot fail)')
    print('  %-60s %10s %10s %8s' % ('label', 'want', 'tol', 'slack%'))
    for r in loose[:18]:
        print('  %-60s %10.4g %10.4g %7.1f%%'
              % (r['label'], r['want'], r['tol'], r['rel_slack_pct'] or float('nan')))

    # the sharpest question: which checks could not fail even if the value were wrong
    toothless = [r for r in loose if (r['rel_slack_pct'] or 0) >= 20.0]
    print('\n  >=20%% slack: %d check(s) -- these assert almost nothing'
          % len(toothless))
    for r in toothless:
        print('    line %-4d %-58s slack %.0f%%'
              % (r['line'], r['label'], r['rel_slack_pct']))
    return rows


# --------------------------------------------------------------------------
# PART B -- experiment-side constants, with the DERIVED ones recomputed
# --------------------------------------------------------------------------
def part_b():
    print()
    print('=' * 78)
    print('PART B -- experiment design constants: recomputing the DERIVED ones')
    print('=' * 78)
    out = {}

    # B1. sweep span +-40%: claimed to stay clear of the stability limit u_c.
    U_STAR, U_C, SPAN = 0.306, 0.520494, 0.40
    u_hi = U_STAR * (1 + SPAN)
    out['span'] = {'u_star': U_STAR, 'u_max_at_span': u_hi, 'u_c': U_C,
                   'margin_pct': 100 * (U_C - u_hi) / U_C,
                   'max_span_before_u_c': U_C / U_STAR - 1}
    print('\n  [B1] sweep span +-%.0f%%   -> u_max = %.4f, u_c = %.4f'
          % (100 * SPAN, u_hi, U_C))
    print('       margin to the stability limit: %.1f%%   '
          '(span could go to +%.0f%% before touching u_c)'
          % (out['span']['margin_pct'], 100 * out['span']['max_span_before_u_c']))
    print('       BASIS: DERIVED from u_c (d36 closed form). Not a chosen number.')

    # B2. seg/L >= 5: the corner-isolation condition. Recompute what it buys.
    SIDE, LS = 5.0, [0.3, 0.45, 0.6, 0.9]
    out['seg_over_L'] = {'side': SIDE, 'L': LS, 'ratio': [SIDE / x for x in LS]}
    print('\n  [B2] side %.1f m, lookahead %s  -> seg/L = %s'
          % (SIDE, LS, ['%.1f' % (SIDE / x) for x in LS]))
    print('       BASIS: STRUCTURAL. The transient must die before the next corner.')
    print('       WEAKNESS: the number 5 is itself CHOSEN. It is testable -- see the')
    print('                 settling-length measurement proposed below.')

    # B3. settling length, from the linear loop itself: how many L does a corner
    #     transient need?  This is the measurement that would GROUND seg/L >= 5.
    #     Linearised delayed pure pursuit, curvature-command plant.  d36 line 145
    #     gives the characteristic equation in arclength units (s = distance / L):
    #
    #         lam^2 + 2 * exp(-lam * u) * (lam + 1) = 0
    #
    #     Its dominant root must cross the imaginary axis exactly at
    #     u_c = 0.520494; that crossing is the check that this transcription is
    #     right, and it is printed before the table.
    def char(lam, u):
        return lam * lam + 2.0 * np.exp(-lam * u) * (lam + 1.0)

    def dominant_root(u):
        best = None
        for x0 in np.linspace(-3.0, 1.0, 25):
            for y0 in np.linspace(0.05, 8.0, 40):
                z = complex(x0, y0)
                good = True
                for _ in range(150):
                    f = char(z, u)
                    h = 1e-7
                    fp = (char(z + h, u) - char(z - h, u)) / (2 * h)
                    if abs(fp) < 1e-16:
                        good = False
                        break
                    zn = z - f / fp
                    if not (np.isfinite(zn.real) and np.isfinite(zn.imag)):
                        good = False
                        break
                    if abs(zn - z) < 1e-13:
                        z = zn
                        break
                    z = zn
                if good and abs(char(z, u)) < 1e-8 and z.imag > 1e-6:
                    if best is None or z.real > best.real:
                        best = z
        return best

    zc = dominant_root(0.520494)
    print('\n  [B3] transcription check: dominant root at u = u_c is %s'
          % ('%+.6f %+.6fj  (Re must be 0)' % (zc.real, zc.imag) if zc else 'NOT FOUND'))

    print('\n  [B3] settling length implied by the loop (grounds seg/L):')
    print('       %6s %22s %14s %14s' % ('u', 'dominant root', 'decay/arclen',
                                         'L to 1% (in L)'))
    b3 = []
    for u in (0.19, 0.306, 0.44):
        z = dominant_root(u)
        if z is None:
            continue
        dec = -z.real
        n99 = math.log(100) / dec if dec > 0 else float('inf')
        b3.append({'u': u, 'root_re': z.real, 'root_im': z.imag,
                   'decay_per_arclength': dec, 'arclen_to_1pct_in_L': n99})
        print('       %6.3f  %9.4f %+8.4fj %14.4f %14.2f' % (u, z.real, z.imag, dec, n99))
    out['settling'] = b3
    if b3:
        worst = max(x['arclen_to_1pct_in_L'] for x in b3)
        print('       worst case over the swept band: %.2f L to decay to 1%%.' % worst)
        print('       seg/L = %.1f (E1-E4) leaves %.1fx that.  seg/L >= 5 is therefore'
              % (SIDE / 0.6, (SIDE / 0.6) / worst))
        print('       NOT arbitrary -- but it was never checked before now.')

    # B6. THE DEFECT B3 EXPOSES.
    #     seg/L was justified as "the transient must die before the next corner",
    #     but the required length depends on u, and u is the swept variable.  So
    #     the condition is not a constant: it tightens as the sweep goes up.
    #     Solve, for each geometry, the largest u whose transient still fits.
    def settle_L(u):
        z = dominant_root(u)
        if z is None or z.real >= 0:
            return float('inf')
        return math.log(100.0) / (-z.real)

    def u_isolated(seg_over_L):
        lo, hi = 0.05, 0.5204
        if settle_L(hi) <= seg_over_L:
            return hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if settle_L(mid) <= seg_over_L:
                lo = mid
            else:
                hi = mid
        return lo

    print()
    print('  [B6] the condition seg/L >= 5 is NOT a constant -- it depends on u,')
    print('       and u is the swept variable.  Largest u that stays isolated:')
    print('       %10s %10s %12s' % ('seg/L', 'u_isolated', 'settle @0.44'))
    geoms = [16.7, 11.1, 8.3, 5.6]
    b6 = []
    for g in geoms:
        ui = u_isolated(g)
        b6.append({'seg_over_L': g, 'u_isolated': ui})
        print('       %10.1f %10.4f %12.2f' % (g, ui, settle_L(0.44)))
    out['u_isolated'] = b6

    # audit every measured point in every block against its own geometry
    import glob as _glob
    print()
    print('       point-by-point audit of the blocks actually run:')
    print('       %-26s %7s %7s %7s %7s %8s' %
          ('block / row', 'seg/L', 'u_max', 'u_iso', 'n_pts', 'n_over'))
    audit = []
    for f in sorted(_glob.glob(str(HERE / 'nav2_block_*.json'))):
        blk = json.load(open(f))
        tau0 = blk.get('meta', {}).get('tau')
        for r in blk.get('rows', []):
            cur = r.get('curve')
            if not cur:
                continue
            Lr = r.get('lookahead', 0.6)
            side = 5.0
            seg = side / Lr
            ui = u_isolated(seg)
            # the swept delay TERM belongs in tau_tot too -- omitting it (as the
            # first version of this audit did) undercounts the freq and tsim
            # blocks badly, because a 2 Hz row carries T_ctrl/2 = 0.25 s.
            term = r.get('T_ctrl_half', 0.0) or r.get('T_sim_half', 0.0) or 0.0
            tt = (r.get('tau_inject', tau0) or 0.2) + term + 0.028
            uu = [c['v'] * tt / Lr for c in cur]
            over = sum(1 for x in uu if x > ui)
            name = pathlib.Path(f).stem.replace('nav2_block_', '')[:14]
            key = next((str(r[k]) for k in
                        ('freq', 'update_duration', 'lookahead', 'alpha', 'knob', 'rep')
                        if k in r), '')
            audit.append({'block': name, 'row': key, 'seg_over_L': seg,
                          'u_max': max(uu), 'u_isolated': ui,
                          'n_points': len(uu), 'n_over': over})
            print('       %-26s %7.1f %7.4f %7.4f %7d %8d'
                  % (name + ' ' + key, seg, max(uu), ui, len(uu), over))
    out['point_audit'] = audit
    tot = sum(a['n_points'] for a in audit)
    bad = sum(a['n_over'] for a in audit)
    print()
    print('       %d of %d measured points (%.0f%%) sit above the isolation limit'
          % (bad, tot, 100.0 * bad / max(tot, 1)))
    print('       -> the MINIMUM is unaffected (u* ~ 0.31 needs ~4 L), but the')
    print('          HIGH-u TAIL is contaminated by corner-to-corner interaction,')
    print('          and d43 -- the primary estimator -- fits the WHOLE curve.')
    out['points_over_isolation'] = {'bad': bad, 'total': tot,
                                    'pct': 100.0 * bad / max(tot, 1)}

    # B4. laps .75: structural, and the invariance was measured.
    print('\n  [B4] laps = 3.75 / 4.75, settle_laps = 1.0')
    print('       BASIS: STRUCTURAL (integer laps re-trigger the goal checker at t=0)')
    print('       + MEASURED (2.75 vs 4.75 laps differ by 0.42%% in RMSE).')

    # B5. the numbers that remain CHOSEN.
    print('\n  [B5] still CHOSEN, i.e. no derivation and no measurement behind them:')
    chosen = [
        ('sweep points = 9 (5 in blocks)',
         'a 3-point parabola needs neighbours; 5 is the cheapest that has them. '
         'The cost of 5 is that the estimator sees only 3 of them.'),
        ('seg/L >= 5 (the number 5)',
         'now bounded by B3 above; adopt "seg/L >= 3x settling length" instead'),
        ('tail_exclude_m = 3.5',
         'must exceed EXIT_M = 3.0; the extra 0.5 m is padding, not derived'),
        ('side = 5.0 m',
         'inherited from the pre-test; couples to seg/L and to costmap extent'),
        ('alpha set {120,90,60,45,30}',
         'spacing is convenience; the falsification only needs two well-separated'),
    ]
    for name, why in chosen:
        print('       - %-32s %s' % (name, why))
    out['chosen'] = [c[0] for c in chosen]
    return out


def main():
    rows = part_a()
    out = part_b()
    res = {'part_a': rows, 'part_b': out}
    json.dump(res, open(HERE / 'd45_criteria_audit_2026-09-06.json', 'w'),
              indent=2, default=float)
    print('\n-> d45_criteria_audit_2026-09-06.json')


if __name__ == '__main__':
    main()
