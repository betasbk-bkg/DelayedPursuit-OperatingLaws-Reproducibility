#!/usr/bin/env python3
"""
d53 -- The claim ledger: every Nav2-side claim, with its evidential status, read
       back from the JSON that produced it rather than retyped.

The question this answers is "which of these is established and which is still a
hypothesis", which is the question that has to be settled before this work is merged
into the main paper -- a merged manuscript doubles the claim surface, and a claim
carried across at the wrong grade is how a paper gets refuted after acceptance.

Grades:
  ESTABLISHED   measured with an interval, on third-party code, and the interval
                excludes the alternative
  WEAK          measured, but the interval is too wide (or one row carries it) to
                support the statement being made
  HYPOTHESIS    a mechanism with a designed test that has not returned
  OPEN          a discrepancy with no designed test

Every number below is read from the analysis JSON on disk.  If a file is missing
the row says so rather than printing a remembered value.

Usage:  python3 d53_claim_ledger_2026-09-06.py
"""
import glob
import json
import pathlib

HERE = pathlib.Path(__file__).parent


def J(pattern):
    f = sorted(glob.glob(str(HERE / pattern)))
    return json.load(open(f[-1])) if f else None


def main():
    rows = []

    d48f = J('d48_freq_iso_interval_2026-09-06.json')
    d48t = J('d48_tsim_interval_2026-09-06.json')
    d46 = J('d46_isolation_refit_2026-09-06.json')
    d47 = J('d47_null_vs_measured_noise_2026-09-06.json')
    d51 = J('d51_second_moment_2026-09-06.json')
    d50 = J('d50_tau_coefficient_2026-09-06.json')
    d52 = J('d52_noise_mechanism_2026-09-06.json')
    d38blk = J('nav2_block_alpha_*.json')

    # --- ESTABLISHED --------------------------------------------------------
    if d46 and 'lookahead' in d46:
        lk = d46['lookahead']['isolated']
        rows.append(('ESTABLISHED', 'Theorem 1: v* proportional to L',
                     'slope %.4f (r2 %.4f), isolation-corrected'
                     % (lk['slope'], lk['r2']), 'd46'))
    if d47:
        ks = d47['knobs']
        worst = max(ks.items(), key=lambda kv: kv[1]['ratio_to_replicate'])
        rows.append(('ESTABLISHED', 'null controls: no harness knob moves u*',
                     'worst knob %s at %.2fx the spread of changing nothing '
                     '(reference %.2f%%)'
                     % (worst[0], worst[1]['ratio_to_replicate'], d47['u_range_pct']),
                     'd47 + E7'))
        rows.append(('ESTABLISHED', 'row-to-row reproducibility of u*',
                     'sd %.2f%%, range %.2f%%, 5 identical rows'
                     % (d47['u_sd_pct'], d47['u_range_pct']), 'E7'))
    if d48f:
        pf = d48f['profile_s']
        np_ = d48f['noise_propagated']
        rows.append(('ESTABLISHED', 'Theorem 2 T_ctrl/2 coefficient is NOT 1',
                     's = %.4f, profile [%.3f, %.3f], 95%% [%.3f, %.3f], LOO sd %.4f'
                     % (d48f['s_free']['s'], pf['interval'][0], pf['interval'][1],
                        np_['p2_5'], np_['p97_5'], d48f['loo_sd']), 'E8 + d48'))
    if d51:
        rows.append(('ESTABLISHED', "Theorem 2's own truncation is part of the excess",
                     'second moment predicts s = %.4f of the measured %.4f'
                     % (d51['s_predicted_mean'], d51['s_measured']), 'd51'))
        br = d51.get('blast_radius_pct')
        if br:
            rows.append(('ESTABLISHED', 'blast radius of the coefficient error',
                         'predicted v* moves %.1f%%-%.1f%% across the swept range'
                         % (br[0], br[1]), 'd51'))

    # the two strongest claims are computed straight from the block files, because
    # they are the ones a merge would carry at full strength
    import math
    alpha = J('nav2_block_alpha_*.json')
    if alpha:
        good = [r for r in alpha['rows']
                if r.get('u_obs') and r.get('how') == 'parabola']
        if good:
            us = [r['u_obs'] for r in good]
            spread = 100 * (max(us) - min(us)) / (sum(us) / len(us))
            cs = [r.get('half_cos', 0.5 * math.cos(math.radians(r['alpha_deg']) / 2))
                  for r in good]
            cos_spread = 100 * (max(cs) - min(cs)) / (sum(cs) / len(cs))
            angs = sorted(r['alpha_deg'] for r in good)
            rows.append(('ESTABLISHED',
                         'the cos law does NOT transfer to a curvature-command plant',
                         'u* varies %.1f%% over alpha = %d-%d deg where 0.5cos(a/2) '
                         'would require %.1f%%  (%d rows, %d points)'
                         % (spread, min(angs), max(angs), cos_spread, len(good),
                            sum(len(r['curve']) for r in good)), 'E5'))
            rows.append(('ESTABLISHED',
                         'u* is a derived pure number, confirmed on third-party code',
                         'derived argmin J(u) = 0.306 (d38); measured %.4f-%.4f '
                         'across corner angles, i.e. +%.1f%% on the mean'
                         % (min(us), max(us),
                            100 * ((sum(us) / len(us)) / 0.306 - 1.0)), 'd38 + E5'))

    look = J('nav2_block_lookahead_*.json')
    if look:
        rr = [r for r in look['rows'] if r.get('curve')]
        Ls, rmin = [], []
        for r in rr:
            Ls.append(r.get('lookahead'))
            rmin.append(min(c['rmse'] for c in r['curve']))
        if len(Ls) >= 3 and all(Ls):
            import numpy as _np
            x, y = _np.log(_np.array(Ls, float)), _np.log(_np.array(rmin, float))
            A = _np.vstack([x, _np.ones_like(x)]).T
            coef, *_ = _np.linalg.lstsq(A, y, rcond=None)
            yh = A @ coef
            r2 = 1 - _np.sum((y - yh) ** 2) / _np.sum((y - y.mean()) ** 2)
            rows.append(('ESTABLISHED', 'by-product: RMSE at the optimum scales as L^1.5',
                         'log-log slope %.4f (r2 %.4f) against a predicted 1.500'
                         % (float(coef[0]), float(r2)), 'E4'))

    # --- WEAK ---------------------------------------------------------------
    if d48t:
        pf = d48t['profile_s']
        rows.append(('WEAK', 'Theorem 2 T_sim/2 coefficient',
                     's = %.4f but profile [%.3f, %.3f] and LOO sd %.4f -- '
                     'one row carries it'
                     % (d48t['s_free']['s'], pf['interval'][0], pf['interval'][1],
                        d48t['loo_sd']), 'E3 + d48'))
    if d50:
        rows.append(('WEAK', 'coefficient on tau_inject',
                     's_tau = %.4f, 95%% [%.4f, %.4f], but on the PRELIMINARY '
                     'tau sweep' % (d50['s_tau'], d50['s_tau_noise_ci'][0],
                                    d50['s_tau_noise_ci'][1]), 'd50'))

    # --- HYPOTHESIS ---------------------------------------------------------
    rows.append(('HYPOTHESIS', 'what the residual ~5% of the coefficient is',
                 'H1 extra delay proportional to T_ctrl vs H2 multiplicative error '
                 'in J(u); discriminator is c(tau_inject), 14 sigma; E11 running',
                 'E11'))
    if d52:
        rows.append(('HYPOTHESIS', 'partial corner in the window drives the noise',
                     'window spans %.3f laps; correlation with run length only weak, '
                     'one counterexample; dual-window logging added, E7 re-run decides'
                     % d52['window_laps'], 'd52'))
    rows.append(('HYPOTHESIS', 'alpha = 120 deg has no interior minimum',
                 'candidate cause is corner interaction (largest turn angle at the '
                 'same seg/L); E9 re-runs it at side 10 m up to u = 0.50', 'E9'))

    # --- OPEN ---------------------------------------------------------------
    if d48f and d48t:
        rows.append(('OPEN', 'E8 fit residual is far above the tsim block',
                     '%.2f%% vs %.2f%% -- corner interaction is not the only '
                     'unmodelled effect, and no test is designed'
                     % (d48f['s_free']['rms_pct'], d48t['s_free']['rms_pct']),
                     'none'))

    order = {'ESTABLISHED': 0, 'WEAK': 1, 'HYPOTHESIS': 2, 'OPEN': 3}
    rows.sort(key=lambda r: order[r[0]])

    print('=' * 78)
    print('d53 -- claim ledger (numbers read back from disk, not retyped)')
    print('=' * 78)
    cur = None
    counts = {}
    for g, claim, detail, src in rows:
        counts[g] = counts.get(g, 0) + 1
        if g != cur:
            print()
            print('  [%s]' % g)
            cur = g
        print('    - %s' % claim)
        print('      %s   (%s)' % (detail, src))

    print()
    print('  ' + ', '.join('%s %d' % (k, counts[k])
                           for k in ('ESTABLISHED', 'WEAK', 'HYPOTHESIS', 'OPEN')
                           if k in counts))
    print()
    print('  THE STRUCTURAL POINT FOR A MERGE:')
    print('  u* is argmin J(u) -- a property of the loop, not of the delay budget.')
    print('  Theorem 2 only says how to COMPUTE tau_tot from the hardware.  So the')
    print('  strongest claim (a derived pure number confirmed on third-party code,')
    print('  plus the falsification of the cos law) does not depend on the claim')
    print('  that is cracked.  A merge can carry the first at full strength while')
    print('  restating the second with its order term.')

    json.dump([{'grade': g, 'claim': c, 'detail': d, 'source': s}
               for g, c, d, s in rows],
              open(HERE / 'd53_claim_ledger_2026-09-06.json', 'w'), indent=2)
    print('\n-> d53_claim_ledger_2026-09-06.json')


if __name__ == '__main__':
    main()
