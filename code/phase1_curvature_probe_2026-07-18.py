"""
this study Phase 1 feasibility probe -- 2026-07-18 (v2: analytic Stadium, fast)
NEW simulation (not a re-fit of existing data): tests whether v* depends on
continuous curvature magnitude even when the discrete corner angle is held at
alpha=0 (no corner), i.e. whether "corner_freq / curvature" is a genuine axis
independent of alpha. This is the arc-plus-corner design sketched in the
companion documentation (Section 3, item 4) but not previously executed.

Trajectory: "Stadium" (racetrack) = two straight segments of length L joined by
two semicircular arcs of radius R, closest()/at() computed ANALYTICALLY
(4 candidate pieces per frame) instead of via a fine polyline -- the first
version used a 400-segment polyline and was too slow (O(3900 frames x 400
segments) per simulate() call); this version is O(1) per frame like the
Circle class.

Engine: crowd_control_engine/simulation_main.py (identical engine used for the
main manuscript, unmodified). Reduced grid vs. the manuscript's campaigns
(feasibility probe, not a publication-grade sweep): MC=15, speeds=13
(unchanged), tau=433ms only, troll=5% only, R in {10,5,2.5,1.5,1.0}.
"""
import os
import sys, json, time, numpy as np
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine")))
import simulation_main as sm
from scipy import stats


class Stadium:
    """Racetrack: straights of length L at y=+-R, semicircular caps of radius R
    centered at (+-L/2, 0). Fully analytic closest()/at() (4 pieces, O(1))."""
    def __init__(self, L, R):
        self.name = f"stadium_L{L}_R{R}"
        self.L, self.R = L, R
        self.circ = 2*L + 2*np.pi*R
        # arc-length breakpoints for: bottom line, right cap, top line, left cap
        self.s0 = 0.0
        self.s1 = L                 # end of bottom line
        self.s2 = L + np.pi*R       # end of right cap
        self.s3 = 2*L + np.pi*R     # end of top line
        self.s4 = self.circ         # end of left cap == circ

    def _bottom(self, p):
        x = np.clip(p[0], -self.L/2, self.L/2)
        pt = np.array([x, -self.R])
        arc = self.s0 + (x + self.L/2)
        return pt, arc

    def _top(self, p):
        x = np.clip(p[0], -self.L/2, self.L/2)
        pt = np.array([x, self.R])
        arc = self.s2 + (self.L/2 - x)
        return pt, arc

    def _right_cap(self, p):
        c = np.array([self.L/2, 0.0])
        v = p - c
        ang = np.arctan2(v[1], v[0])
        ang = np.clip(ang, -np.pi/2, np.pi/2)
        pt = c + self.R*np.array([np.cos(ang), np.sin(ang)])
        arc = self.s1 + (ang + np.pi/2)*self.R
        return pt, arc

    def _left_cap(self, p):
        c = np.array([-self.L/2, 0.0])
        v = p - c
        ang = np.arctan2(v[1], v[0])
        # left cap spans pi/2 .. 3pi/2 ; normalize ang into that range
        if ang < np.pi/2:
            ang += 2*np.pi
        ang = np.clip(ang, np.pi/2, 3*np.pi/2)
        pt = c + self.R*np.array([np.cos(ang), np.sin(ang)])
        arc = self.s3 + (ang - np.pi/2)*self.R
        return pt, arc

    def closest(self, p):
        cands = [self._bottom(p), self._top(p), self._right_cap(p), self._left_cap(p)]
        best = min(cands, key=lambda c: np.linalg.norm(p - c[0]))
        return best[0], best[1]

    def at(self, arc):
        arc = arc % self.circ
        if arc <= self.s1:
            x = -self.L/2 + arc
            return np.array([x, -self.R])
        elif arc <= self.s2:
            ang = -np.pi/2 + (arc - self.s1)/self.R
            return np.array([self.L/2, 0.0]) + self.R*np.array([np.cos(ang), np.sin(ang)])
        elif arc <= self.s3:
            x = self.L/2 - (arc - self.s2)
            return np.array([x, self.R])
        else:
            ang = np.pi/2 + (arc - self.s3)/self.R
            return np.array([-self.L/2, 0.0]) + self.R*np.array([np.cos(ang), np.sin(ang)])

    def start(self):
        return np.array([-self.L/2, -self.R])


SPEEDS = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
MC, N, TROLL, TAU_MS, TAU_F = 15, 150, 0.05, 433, 26
L_FIXED = 15.0
R_VALUES = [10.0, 5.0, 2.5, 1.5, 1.0]

def measure_vstar(traj, tau_f):
    orig = sm.DELAY_F
    sm.DELAY_F = tau_f
    try:
        rmses = []
        for v in SPEEDS:
            r = sm.run_condition(traj, N, TROLL, MC, method='fixed',
                                  speed_override=v, seed_base=31, seed_offset=N)
            rmses.append(r['rmse_mean'])
    finally:
        sm.DELAY_F = orig
    idx = int(np.argmin(rmses)); vstar = float(SPEEDS[idx])
    if 1 <= idx <= len(SPEEDS)-2:
        c = np.polyfit(SPEEDS[idx-1:idx+2], rmses[idx-1:idx+2], 2)
        if c[0] > 0:
            vf = -c[1]/(2*c[0])
            if SPEEDS[idx-1] <= vf <= SPEEDS[idx+1]:
                vstar = float(vf)
    return vstar, rmses

if __name__ == "__main__":
    print("Sanity check: Stadium(L=15,R=10) circ vs analytic:", Stadium(15,10).circ,
          "expected", 2*15+2*np.pi*10)

    results = {}
    t0 = time.time()
    for R in R_VALUES:
        st = Stadium(L_FIXED, R)
        vstar, rmses = measure_vstar(st, TAU_F)
        curvature = 1.0/R
        results[R] = dict(circ=st.circ, vstar=vstar, rmse_min=min(rmses), curvature=curvature)
        print(f"  R={R:>5.1f}  circ={st.circ:>7.2f}m  curvature={curvature:.3f}/m  "
              f"v*={vstar:.4f}  RMSE_min={min(rmses):.4f}   [{time.time()-t0:.0f}s elapsed]")

    Rs = np.array(R_VALUES)
    curvs = 1.0/Rs
    vstars = np.array([results[R]['vstar'] for R in R_VALUES])
    r, p = stats.pearsonr(curvs, vstars)
    print(f"\nPearson r(curvature=1/R, v*) at alpha=0 fixed, tau=433ms: r={r:.3f}, p={p:.4f}, n={len(R_VALUES)}")

    circle_v = 2.1336206896552032
    print(f"\nReference: Circle (R=10, curvature=0.1/m everywhere) v*={circle_v:.4f}")
    print(f"Stadium R=10 (same curvature magnitude, but only on 2 of 4 sides, "
          f"rest is straight/zero curvature) v*={results[10.0]['vstar']:.4f}")

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "phase1_stadium_results_2026-07-18.json")
    with open(outpath, "w") as f:
        json.dump({str(R): v for R, v in results.items()}, f, indent=2)
    print(f"\nSaved: {outpath}")
