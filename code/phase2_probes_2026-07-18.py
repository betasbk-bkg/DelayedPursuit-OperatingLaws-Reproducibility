"""
this study Phase 2 probes -- 2026-07-18
Probe A (quantization ablation): circle, n_dirs in {8, 360}, tau in {100,433,1000},
  full 13-speed sweep, MC=15, RAW RMSE CURVES SAVED. Tests whether the low-v
  branch (plateau ~0.42 for circle at 8 dirs) vanishes when quantization is
  removed -- the decisive test of the "quantization scallop vs dither" theory
  for the descending branch of the U.
  n_dirs=8 is a sanity check: must reproduce published unified_vstar curves.

Probe B (start-phase randomization):
  B1: Stadium R in {10,5,2.5,1.5}, L=15, tau=433ms, random start arc per MC run.
      Fixes the finite-horizon geometry-sampling artifact discovered in the v1
      probe (fixed start on a grid-aligned straight -> v*=0.25 artifact).
  B2: Square (published geometry, h=10), tau=1000ms, random start arc --
      quantifies how much the published fixed-start protocol biased the
      low-speed end (square v*=0.615 at tau=1000 covers only 40m of the 80m
      perimeter in 65s).

Engine: crowd_control_engine/simulation_main.py, unmodified except documented
monkey-patches (DIRS/gen_votes for probe A -- same pattern as the published
expa_quantization.py -- and per-run start phase for probe B).
"""
import os
import sys, json, time, numpy as np
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine")))
import simulation_main as sm
from simulation_main import Circle, Square

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "phase2_probes_results_2026-07-18.json")

SPEEDS = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
MC, N = 15, 150
TROLL = 0.05

# ---------- n_dirs patching (identical pattern to expa_quantization.py) ----------
def make_dirs(n):
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])

def angle_to_dir_n(angles_deg, dirs):
    dir_angles = np.degrees(np.arctan2(dirs[:,1], dirs[:,0])) % 360
    a = np.asarray(angles_deg) % 360
    diffs = np.abs(dir_angles[None,:] - a[:,None])
    diffs = np.minimum(diffs, 360-diffs)
    return np.argmin(diffs, axis=1)

def patched_gen_votes_n(ideal_angle, prev_angle, troll_ratio, N_agents, rng, n_dirs=8):
    dirs = make_dirs(n_dirs)
    n_troll = round(N_agents*troll_ratio); n_troll = min(n_troll, N_agents)
    remaining = N_agents - n_troll
    n_accurate = round(remaining*0.7368); n_slow = round(remaining*0.2105)
    n_other = remaining - n_accurate - n_slow
    if n_other < 0: n_accurate += n_other; n_other = 0
    angles = np.empty(n_accurate+n_slow+n_other); idx = 0
    angles[idx:idx+n_accurate] = ideal_angle + rng.uniform(-3,3,n_accurate); idx += n_accurate
    diff = ideal_angle - prev_angle
    if diff > 180: diff -= 360
    if diff < -180: diff += 360
    if n_slow > 0:
        lag = rng.uniform(0.2,0.5,n_slow)
        angles[idx:idx+n_slow] = prev_angle + diff*(1-lag); idx += n_slow
    if n_other > 0:
        angles[idx:idx+n_other] = ideal_angle + rng.uniform(-30,30,n_other); idx += n_other
    non_troll = angle_to_dir_n(angles[:idx], dirs)
    troll_votes = rng.integers(0, n_dirs, n_troll) if n_troll>0 else np.array([],dtype=int)
    return np.concatenate([non_troll, troll_votes])

def sweep_ndirs(traj, tau_f, n_dirs):
    dirs = make_dirs(n_dirs)
    orig_dirs, orig_gv, orig_delay = sm.DIRS, sm.gen_votes, sm.DELAY_F
    sm.DIRS = dirs
    sm.gen_votes = lambda ia,pa,tr,na,rng: patched_gen_votes_n(ia,pa,tr,na,rng,n_dirs)
    sm.DELAY_F = tau_f
    try:
        rmses = []
        for v in SPEEDS:
            r = sm.run_condition(traj, N, TROLL, MC, method='fixed',
                                  speed_override=v, seed_base=31, seed_offset=N)
            rmses.append(r['rmse_mean'])
    finally:
        sm.DIRS, sm.gen_votes, sm.DELAY_F = orig_dirs, orig_gv, orig_delay
    return refine_vstar(rmses), rmses

def refine_vstar(rmses):
    idx = int(np.argmin(rmses)); vstar = float(SPEEDS[idx])
    if 1 <= idx <= len(SPEEDS)-2:
        c = np.polyfit(SPEEDS[idx-1:idx+2], rmses[idx-1:idx+2], 2)
        if c[0] > 0:
            vf = -c[1]/(2*c[0])
            if SPEEDS[idx-1] <= vf <= SPEEDS[idx+1]:
                vstar = float(vf)
    return vstar

# ---------- Stadium (analytic, from phase1 v2) ----------
class Stadium:
    def __init__(self, L, R, start_arc=0.0):
        self.name = f"stadium_L{L}_R{R}"
        self.L, self.R = L, R
        self.circ = 2*L + 2*np.pi*R
        self.s0, self.s1 = 0.0, L
        self.s2 = L + np.pi*R
        self.s3 = 2*L + np.pi*R
        self.start_arc = start_arc
    def _bottom(self, p):
        x = np.clip(p[0], -self.L/2, self.L/2)
        return np.array([x,-self.R]), self.s0 + (x+self.L/2)
    def _top(self, p):
        x = np.clip(p[0], -self.L/2, self.L/2)
        return np.array([x,self.R]), self.s2 + (self.L/2 - x)
    def _rcap(self, p):
        c = np.array([self.L/2,0.]); v = p-c
        ang = np.clip(np.arctan2(v[1],v[0]), -np.pi/2, np.pi/2)
        return c+self.R*np.array([np.cos(ang),np.sin(ang)]), self.s1+(ang+np.pi/2)*self.R
    def _lcap(self, p):
        c = np.array([-self.L/2,0.]); v = p-c
        ang = np.arctan2(v[1],v[0])
        if ang < np.pi/2: ang += 2*np.pi
        ang = np.clip(ang, np.pi/2, 3*np.pi/2)
        return c+self.R*np.array([np.cos(ang),np.sin(ang)]), self.s3+(ang-np.pi/2)*self.R
    def closest(self, p):
        cands = [self._bottom(p), self._top(p), self._rcap(p), self._lcap(p)]
        best = min(cands, key=lambda c: np.linalg.norm(p-c[0]))
        return best[0], best[1]
    def at(self, arc):
        arc = arc % self.circ
        if arc <= self.s1:
            return np.array([-self.L/2+arc, -self.R])
        elif arc <= self.s2:
            ang = -np.pi/2 + (arc-self.s1)/self.R
            return np.array([self.L/2,0.]) + self.R*np.array([np.cos(ang),np.sin(ang)])
        elif arc <= self.s3:
            return np.array([self.L/2-(arc-self.s2), self.R])
        else:
            ang = np.pi/2 + (arc-self.s3)/self.R
            return np.array([-self.L/2,0.]) + self.R*np.array([np.cos(ang),np.sin(ang)])
    def start(self):
        return self.at(self.start_arc)

class RandomStartWrapper:
    """Wraps any trajectory; start() returns at(start_arc). Set start_arc per run."""
    def __init__(self, base):
        self.base = base
        self.circ = base.circ
        self.name = base.name + "_randstart"
        self.start_arc = 0.0
    def closest(self, p): return self.base.closest(p)
    def at(self, arc): return self.base.at(arc)
    def start(self): return self.base.at(self.start_arc)

def sweep_random_start(base_traj, tau_f, phase_seed=12345):
    """13-speed sweep, MC=15, random start arc per MC run (same phases across speeds
    so the speed comparison is paired)."""
    wrapper = RandomStartWrapper(base_traj)
    phase_rng = np.random.default_rng(phase_seed)
    phases = phase_rng.uniform(0, base_traj.circ, MC)
    orig_delay = sm.DELAY_F
    sm.DELAY_F = tau_f
    try:
        rmses = []
        for v in SPEEDS:
            runs = []
            for i in range(MC):
                wrapper.start_arc = phases[i]
                r = sm.simulate(wrapper, N, TROLL, seed=i*31+N,
                                method='fixed', speed_override=v)
                runs.append(r['rmse'])
            rmses.append(float(np.mean(runs)))
    finally:
        sm.DELAY_F = orig_delay
    return refine_vstar(rmses), rmses

if __name__ == "__main__":
    t0 = time.time()
    out = {"speeds": SPEEDS, "probeA": {}, "probeB_stadium": {}, "probeB_square": {}}

    print("=== Probe A: quantization ablation (circle) ===", flush=True)
    circle = Circle()
    for n_dirs in [8, 360]:
        out["probeA"][str(n_dirs)] = {}
        for tau_ms, tau_f in [(100,6),(433,26),(1000,60)]:
            vstar, rmses = sweep_ndirs(circle, tau_f, n_dirs)
            out["probeA"][str(n_dirs)][str(tau_ms)] = {"vstar": vstar, "rmses": rmses}
            print(f"  n={n_dirs:>3} tau={tau_ms:>4}: v*={vstar:.3f}  "
                  f"rmses={np.round(rmses,3).tolist()}  [{time.time()-t0:.0f}s]", flush=True)

    print("=== Probe B1: Stadium with random start phase (tau=433) ===", flush=True)
    for R in [10.0, 5.0, 2.5, 1.5]:
        st = Stadium(15.0, R)
        vstar, rmses = sweep_random_start(st, 26)
        out["probeB_stadium"][str(R)] = {"circ": st.circ, "vstar": vstar,
                                          "rmses": rmses, "rmse_min": float(min(rmses))}
        print(f"  R={R:>5.1f} circ={st.circ:6.1f}m: v*={vstar:.3f} "
              f"RMSE_min={min(rmses):.4f}  [{time.time()-t0:.0f}s]", flush=True)

    print("=== Probe B2: Square tau=1000, random start (protocol-bias check) ===", flush=True)
    sq = Square()
    vstar, rmses = sweep_random_start(sq, 60)
    out["probeB_square"] = {"tau_ms": 1000, "vstar": vstar, "rmses": rmses,
                             "published_fixed_start_vstar": 0.615}
    print(f"  square tau=1000 random-start: v*={vstar:.3f} (published fixed-start 0.615)  "
          f"rmses={np.round(rmses,3).tolist()}  [{time.time()-t0:.0f}s]", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}  total {time.time()-t0:.0f}s", flush=True)
