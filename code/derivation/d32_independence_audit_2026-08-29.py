# -*- coding: utf-8 -*-
"""
Derivation note -- D32: contamination audit.

Question: the branch-selection thread contained real bugs (a toy integrator that
did not reproduce the flow's attractor, a cell-frame error that invented events,
non-converged roots accepted as fixed points, a word filter that discarded the
correct branch).  Could those have corrupted the THEORY -- Eq. (12), m_c, sigma?

Two parts.

PART 1 -- dependency trace.  Which load-bearing claim used which code path.
Printed as a table so the reader can check the separation directly.

PART 2 -- the one verification gap that was actually open.  The law
m_c = Delta/2 - a_acc was verified by sweeping a_acc and n_dirs THROUGH MY OWN
CLOSED FORM for b(m).  That closed form was validated against the engine only at
the engine's own setting (a_acc = 3 deg, n_dirs = 8).  If the closed form were
wrong in how it handles a different a_acc or a different grid, the sweep would
inherit the error and the m_c law would be circular.

Closing it: build a parameterised Monte-Carlo vote generator that mirrors
gen_votes exactly, validate IT against the engine at the engine's setting, then
use it -- not the closed form -- to measure b(m) and m_c away from the defaults.

Outputs: d32_results_2026-08-29.json
"""
import importlib.util, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.environ.get("ENGINE_DIR") or next(
    p for p in [os.path.join(HERE, "..", "..", "engine", "crowd_control_engine"),
                os.path.join(HERE, "..", "submission", "repo",
                             "engine", "crowd_control_engine")]
    if os.path.isdir(p))
sys.path.insert(0, ENGINE)
import simulation_main as sm

def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
d9 = _load("d9", "d9_mc_law_2026-08-29.py")

N_AG, TROLL = 150, 0.05


# ------------------------------------------------------- parameterised MC votes
def gen_votes_param(ideal, prev, rng, a_acc=3.0, a_other=30.0, n_dirs=8,
                    N=N_AG, troll=TROLL):
    """mirrors simulation_main.gen_votes, with a_acc / a_other / n_dirs exposed."""
    n_troll = round(N*troll)
    n_troll = min(n_troll, N)
    rem = N - n_troll
    n_acc = round(rem*0.7368)
    n_slow = round(rem*0.2105)
    n_other = rem - n_acc - n_slow
    if n_other < 0:
        n_acc += n_other; n_other = 0
    ang = np.empty(n_acc + n_slow + n_other)
    i = 0
    ang[i:i+n_acc] = ideal + rng.uniform(-a_acc, a_acc, n_acc); i += n_acc
    diff = ideal - prev
    if diff > 180: diff -= 360
    if diff < -180: diff += 360
    if n_slow > 0:
        lag = rng.uniform(0.2, 0.5, n_slow)
        ang[i:i+n_slow] = prev + diff*(1-lag); i += n_slow
    if n_other > 0:
        ang[i:i+n_other] = ideal + rng.uniform(-a_other, a_other, n_other); i += n_other
    step = 360.0/n_dirs
    dirs = np.arange(n_dirs)*step
    a = ang[:i] % 360
    d = np.abs(dirs[None, :] - a[:, None])
    d = np.minimum(d, 360-d)
    votes = np.argmin(d, axis=1)
    trolls = rng.integers(0, n_dirs, n_troll) if n_troll > 0 else np.array([], int)
    allv = np.concatenate([votes, trolls])
    vec = np.stack([np.cos(np.radians(dirs[allv])), np.sin(np.radians(dirs[allv]))], 1)
    b = vec.mean(axis=0)
    return (math.degrees(math.atan2(b[1], b[0])) - ideal + 180) % 360 - 180


def measure_b(ms, a_acc, n_dirs, draws=3000, seed=7):
    rng = np.random.default_rng(seed)
    return np.array([np.mean([gen_votes_param(m, m, rng, a_acc=a_acc, n_dirs=n_dirs)
                              for _ in range(draws)]) for m in ms])


def m_c_from(ms, b):
    return float(ms[int(np.argmax(np.abs(b)))])


if __name__ == "__main__":
    out = {}
    print("PART 1 -- dependency trace (which claim used which code)")
    trace = [
        ("b(m), s(m) closed form", "d9, d12", "engine Monte Carlo (rms 0.025 / 0.051 deg)", "none"),
        ("m_c = Delta/2 - a_acc", "d9", "closed form sweep -> CHECKED HERE with MC", "none"),
        ("cylinder flow, similarity", "d6, d7", "1-DOF surrogate vs 150-agent engine (0.9-3.6%)", "none"),
        ("fold caustic rho ~ 1/sqrt", "d7, d10", "rho^-2 linear, R2 = 0.9997", "none"),
        ("Eq. (12) edge law / c", "d19 vs d10", "flow, and independently d28 return map", "none"),
        ("sigma_theta (13)", "d20, d21", "engine sigma (7.0-7.8%)", "none"),
        ("symbolic word +, +-+", "d29", "crossing counts in the faithful flow", "d25 discarded"),
        ("branch selection, basins", "d28, d29", "flow's own z located in the basin", "d27 superseded"),
        ("transition curve", "d30, d31", "NOT closed", "d26, d30[C] discarded"),
    ]
    print(f"    {'claim':28s} {'code':12s} {'verified against':44s} {'buggy input':16s}")
    for c, k, v, b in trace:
        print(f"    {c:28s} {k:12s} {v:44s} {b:16s}")
    out["trace"] = [dict(claim=c, code=k, verified=v, buggy=b) for c, k, v, b in trace]

    print("\nPART 2 -- closing the open verification gap")
    print("  (a) does the parameterised MC generator reproduce the engine at its own setting?")
    ms = np.linspace(0, 22.5, 31)
    rng = np.random.default_rng(3)
    eng = []
    for m in ms:
        errs = []
        for _ in range(3000):
            v = sm.gen_votes(m, m, TROLL, N_AG, rng)
            bl = sm.DIRS[v].mean(axis=0)
            errs.append((math.degrees(math.atan2(bl[1], bl[0])) - m + 180) % 360 - 180)
        eng.append(np.mean(errs))
    eng = np.array(eng)
    mine = measure_b(ms, 3.0, 8)
    print(f"      max |diff| = {np.abs(mine-eng).max():.3f} deg, "
          f"rms = {np.sqrt(np.mean((mine-eng)**2)):.3f} deg")
    print(f"      m_c: engine MC {m_c_from(ms, eng):.2f}, parameterised MC {m_c_from(ms, mine):.2f}")
    out["generator_validation"] = dict(max_diff=float(np.abs(mine-eng).max()),
                                       rms=float(np.sqrt(np.mean((mine-eng)**2))),
                                       m_c_engine=m_c_from(ms, eng),
                                       m_c_param=m_c_from(ms, mine))

    print("\n  (b) m_c away from the defaults, measured by MC (not by the closed form)")
    print(f"    {'a_acc':>6} {'n_dirs':>7} {'m_c MC':>8} {'m_c closed':>11} "
          f"{'Delta/2 - a':>12} {'MC - law':>9}")
    rows = []
    for a_acc, n_dirs in ((1.0, 8), (6.0, 8), (10.0, 8), (3.0, 4), (3.0, 16)):
        half = 180.0/n_dirs
        grid = np.linspace(0, half, 61)
        bmc = measure_b(grid, a_acc, n_dirs, draws=2500, seed=11)
        bcf = np.array([d9.b_closed(m, a_acc, n_dirs) for m in grid])
        mc_mc, mc_cf = m_c_from(grid, bmc), m_c_from(grid, bcf)
        law = half - a_acc
        rows.append(dict(a_acc=a_acc, n_dirs=n_dirs, m_c_mc=mc_mc, m_c_closed=mc_cf,
                         law=law, err=mc_mc-law))
        print(f"    {a_acc:6.1f} {n_dirs:7d} {mc_mc:8.2f} {mc_cf:11.2f} {law:12.2f} "
              f"{mc_mc-law:+9.2f}")
    out["m_c_mc"] = rows
    e = [abs(r["err"]) for r in rows]
    print(f"\n    max |MC - (Delta/2 - a_acc)| = {max(e):.2f} deg "
          f"(grid spacing {grid[1]-grid[0]:.2f} deg)")
    json.dump(out, open(os.path.join(HERE, "d32_results_2026-08-29.json"), "w"),
              indent=2, default=float)
    print("\n-> d32_results_2026-08-29.json")
