#!/usr/bin/env python3
"""
d68 -- why the crowd's direction error sits BELOW the quantisation RMS, and what
       number the reference engine's sentence (ii) should carry.

d67 established the arithmetic slip: a uniform distribution on [-22.5, 22.5] deg
has RMS 22.5/sqrt(3) = 12.99 deg, not 22.5/sqrt(6) = 9.19 deg (that is the
triangular value).  Correcting it alone would leave the manuscript worse off --
the measured 8.60 deg would then sit 34% BELOW the model instead of 6% above it,
and the sentence's conclusion ("quantisation is the dominant source") would look
unsupported.

But the manuscript's OTHER result already tells us the conclusion is right:
result (i) reports sigma_theta = 8.60 +- 0.15 deg with a power-law exponent of
N^-0.008 over N = 10..200.  Independent vote noise would fall as N^-0.5.  A floor
that does not fall with N is, as the manuscript says, structural.  The mechanism
is that every agent quantises the SAME ideal direction, so the accurate majority
lands in the SAME bin and averaging cannot remove a common-mode error.

So the correct model is not "average of independent uniform errors".  It is:

    blend ~= the modal bin direction, whose error is uniform on [-22.5, 22.5]
             -> 12.99 deg,
    REDUCED because the +-3 deg voter noise splits the accurate group across two
    bins whenever the ideal direction lies within ~3 deg of a bin boundary, and a
    two-bin blend interpolates toward the true direction.

That predicts a floor BELOW 12.99 deg, N-independent, and controlled by the ratio
of the voter noise to the bin width.  This script tests all three, by turning the
voter noise from 0 up through the engine's 3 deg and out to 20 deg.

Prediction stated before running:
  * sigma = 0     -> exactly the quantisation RMS, 12.99 deg
  * sigma = 3 deg -> below it, and N-independent
  * large sigma   -> the split spreads over many bins and the error falls further,
                     approaching the voter noise itself

Usage:  python3 d68_quantisation_floor_mechanism_2026-09-08.py
"""
import json
import math
import pathlib

import sys

import numpy as np

REPO = pathlib.Path(r'C:\Users\betas\Downloads\_letg_main_read\crowd_control_engine')
sys.path.insert(0, str(REPO))
import simulation_main as sm            # noqa: E402

HERE = pathlib.Path(__file__).parent
NBINS = 8
STEP = 360.0 / NBINS
HALF = STEP / 2
BINS = np.radians(np.arange(NBINS) * STEP)
DIRS = np.stack([np.cos(BINS), np.sin(BINS)], axis=1)


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def blend_rms(n_agents, sigma_deg, draws, rng, troll=0.0, slow_frac=0.0,
              slow_lag_deg=0.0):
    """RMS error of the normalised mean vote, for a crowd that all sees the same
    ideal direction.  troll/slow are optional so the engine's composition can be
    switched on one piece at a time."""
    ideal = rng.uniform(-180, 180, draws)
    err = np.empty(draws)
    n_troll = int(round(n_agents * troll))
    n_slow = int(round((n_agents - n_troll) * slow_frac))
    n_acc = n_agents - n_troll - n_slow
    for i, th in enumerate(ideal):
        v = np.zeros(2)
        if n_acc:
            noisy = th + rng.normal(0, sigma_deg, n_acc) if sigma_deg > 0 else \
                np.full(n_acc, th)
            idx = np.round(np.asarray(noisy) / STEP).astype(int) % NBINS
            v += DIRS[idx].sum(axis=0)
        if n_slow:
            idx = np.round((th - slow_lag_deg) / STEP).astype(int) % NBINS
            v += DIRS[idx] * n_slow
        if n_troll:
            v += DIRS[rng.integers(0, NBINS, n_troll)].sum(axis=0)
        g = float(np.linalg.norm(v))
        err[i] = wrap(math.degrees(math.atan2(v[1], v[0])) - th) if g > 1e-12 else 0.0
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    rng = np.random.default_rng(11)
    draws = 4000
    print('=' * 78)
    print('d68 -- the quantisation floor: where 8.60 deg comes from')
    print('=' * 78)
    print('  %d bins, step %.1f deg;  uniform RMS on a bin = %.3f deg'
          % (NBINS, STEP, HALF / math.sqrt(3)))
    print()

    # ---- 1. the floor does not average away ---------------------------------
    print('  (1) N-dependence at the engine\'s voter noise (sigma = 3 deg):')
    ns = [10, 25, 50, 100, 150, 200]
    r_n = [blend_rms(n, 3.0, draws, rng) for n in ns]
    for n, r in zip(ns, r_n):
        print('        N = %4d   RMS = %6.3f deg' % (n, r))
    ex = float(np.polyfit(np.log(ns), np.log(r_n), 1)[0])
    print('        power-law exponent %+0.4f   (independent noise would give -0.5;'
          % ex)
    print('         the manuscript reports -0.008 on its own engine)')
    print()

    # ---- 2. the floor is set by voter noise / bin width ----------------------
    print('  (2) voter noise sets the floor (N = 150):')
    print('      %8s %12s' % ('sigma', 'blend RMS'))
    sig = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    r_s = [blend_rms(150, s, draws, rng) for s in sig]
    for s, r in zip(sig, r_s):
        tag = '   <- the engine' if abs(s - 3.0) < 1e-9 else ''
        print('      %8.1f %12.3f%s' % (s, r, tag))
    print('      sigma = 0 gives %.3f deg against the analytic %.3f deg'
          % (r_s[0], HALF / math.sqrt(3)))
    print()

    # ---- 3. the engine's actual composition ---------------------------------
    print('  (3) adding the engine\'s other voter groups (N = 150, sigma = 3):')
    combos = [('accurate only', dict()),
              ('+ 5% trolls', dict(troll=0.05)),
              ('+ 20% slow (lag 10 deg)', dict(slow_frac=0.2105, slow_lag_deg=10.0)),
              ('+ both', dict(troll=0.05, slow_frac=0.2105, slow_lag_deg=10.0))]
    r_c = {}
    for name, kw in combos:
        r_c[name] = blend_rms(150, 3.0, draws, rng, **kw)
        print('      %-26s RMS = %6.3f deg' % (name, r_c[name]))
    print()

    # ---- 4. the slow group's lag is NOT a free parameter --------------------
    # The engine's "slow" agents vote the PREVIOUS aggregation step's direction.
    # On the trajectory result (iii) names -- Circle R = 10 m at v = 2.0 m/s --
    # the ideal direction turns by v*WIN/R per step, so the lag is fixed by the
    # operating point rather than chosen.
    R_E4, V_E4, WIN = 10.0, 2.0, sm.WIN
    lag_derived = math.degrees(V_E4 * WIN / R_E4)
    print("  (4) the engine's OWN gen_votes, at the E4 operating point.")
    print('      Sections 1-3 use a reduced model, and it is wrong in three ways')
    print('      that matter: the engine draws accurate agents UNIFORM +-3 deg')
    print('      (sd 1.73, not 3), puts slow agents only 20-50%% of the way back,')
    print('      and gives 10%% of voters +-30 deg.  So the number to quote has to')
    print('      come from calling gen_votes itself, not from my transcription.')
    print('      Circle R = %.0f m at v = %.1f m/s turns %.2f deg per WIN = %.1f s'
          % (R_E4, V_E4, lag_derived, WIN))
    print('      step, so prev = ideal - %.2f deg.  Derived, not fitted.'
          % lag_derived)
    print()
    print('      %10s %12s' % ('turn/step', 'blend RMS'))
    r_l = []
    for lg in sorted([0.0, 1.0, lag_derived, 5.0, 10.0, 20.0]):
        errs = []
        for _ in range(draws):
            th = float(rng.uniform(-180, 180))
            votes = sm.gen_votes(th, th - lg, 0.05, 150, rng)
            b = sm.DIRS[votes].mean(axis=0)
            if float(np.linalg.norm(b)) > 1e-12:
                errs.append(wrap(math.degrees(math.atan2(b[1], b[0])) - th))
        v = float(np.sqrt(np.mean(np.array(errs) ** 2)))
        r_l.append((lg, v))
        tag = '   <- derived for E4' if abs(lg - lag_derived) < 1e-9 else ''
        print('      %10.2f %12.3f%s' % (lg, v, tag))
    at_derived = [v for lg, v in r_l if abs(lg - lag_derived) < 1e-9][0]
    print()
    print('      engine composition at the DERIVED turn rate: %.3f deg' % at_derived)
    print('      manuscript measurement:                      8.60 +- 0.15 deg'
          '  (%+.1f%%)' % (100 * (at_derived / 8.60 - 1)))
    print('      the printed 22.5/sqrt(6):                    9.19 deg')
    print('      corrected uniform-on-a-bin RMS:             %5.2f deg'
          % (HALF / math.sqrt(3)))
    print()
    # ---- 5. which theta_actual? --------------------------------------------
    # Section 4 measures the error of the BLEND direction and lands 28% above the
    # manuscript's 8.60.  But "theta_err = theta_actual - theta_ideal" does not
    # say whether theta_actual is the commanded blend or the direction the agent
    # is actually travelling, and those differ: the WIN hold and the first-order
    # lag low-pass the blend, and a low-pass of a zero-mean error is smaller.
    # If the travelling direction lands near 8.60 while the blend lands near 11,
    # the manuscript is measuring the filtered quantity, and the model to compare
    # it against is the filtered one too.
    print('  (5) blend direction vs travelling direction, on the E4 circle:')
    frames = int(240.0 / sm.DT)
    delay_f = int(round(0.433 / sm.DT))
    pos = np.array([R_E4, 0.0])
    vel = np.zeros(2)
    hist = [pos.copy()]
    cur = np.array([0.0, 1.0])
    prev = 90.0
    e_blend, e_travel = [], []
    for f in range(frames):
        if f % sm.VOTE_INT == 0:
            dp = hist[max(0, len(hist) - 1 - delay_f)]
            ang = math.atan2(dp[1], dp[0])
            look = ang + V_E4 * 0.0 + sm.LOOK / R_E4      # arclength lookahead
            tgt = np.array([R_E4 * math.cos(look), R_E4 * math.sin(look)])
            ideal = tgt - dp
            g = float(np.linalg.norm(ideal))
            if g > 1e-10:
                ideal = ideal / g
            ia = math.degrees(math.atan2(ideal[1], ideal[0]))
            votes = sm.gen_votes(ia, prev, 0.05, 150, rng)
            b = sm.DIRS[votes].mean(axis=0)
            if float(np.linalg.norm(b)) > 1e-12:
                cur = b / float(np.linalg.norm(b))
                e_blend.append(wrap(math.degrees(math.atan2(cur[1], cur[0])) - ia))
                sp = float(np.linalg.norm(vel))
                if sp > 1e-6:
                    e_travel.append(wrap(math.degrees(math.atan2(vel[1], vel[0]))
                                         - ia))
            prev = ia
        vel += sm.SMOOTH * (cur * V_E4 - vel)
        pos = pos + vel * sm.DT
        hist.append(pos.copy())
    rb = float(np.sqrt(np.mean(np.array(e_blend) ** 2)))
    rt = float(np.sqrt(np.mean(np.array(e_travel[len(e_travel) // 4:]) ** 2)))
    print('      blend direction       RMS = %6.3f deg' % rb)
    print('      travelling direction  RMS = %6.3f deg   (first quarter discarded)'
          % rt)
    print('      manuscript                  8.60 +- 0.15 deg')
    print('      -> blend %+.1f%%, travelling %+.1f%%'
          % (100 * (rb / 8.60 - 1), 100 * (rt / 8.60 - 1)))
    print()

    print('  READING: the floor is real and N-independent, exactly as the')
    print('  manuscript concludes -- but its size is set by the voter noise')
    print('  relative to the bin, not by the bin alone.  The comparison the')
    print('  sentence should make is against %.2f deg (uniform on the bin),' %
          (HALF / math.sqrt(3)))
    print('  with the measured value sitting BELOW it for a stated reason.')

    json.dump({'n_bins': NBINS, 'step_deg': STEP,
               'uniform_rms_deg': HALF / math.sqrt(3),
               'triangular_rms_deg': HALF / math.sqrt(6),
               'N': ns, 'rms_vs_N': r_n, 'N_exponent': ex,
               'sigma_deg': sig, 'rms_vs_sigma': r_s,
               'composition': r_c, 'draws': draws,
               'slow_lag_sweep': r_l, 'lag_derived_deg': lag_derived,
               'rms_at_derived_lag': at_derived,
               'e4_circle_blend_rms_deg': rb,
               'e4_circle_travel_rms_deg': rt,
               'paper_measured_deg': 8.60},
              open(HERE / 'd68_quantisation_floor_2026-09-08.json', 'w'), indent=2)
    print('\n-> d68_quantisation_floor_2026-09-08.json')


if __name__ == '__main__':
    main()
