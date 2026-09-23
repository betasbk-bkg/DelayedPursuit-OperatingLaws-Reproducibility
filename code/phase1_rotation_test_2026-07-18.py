"""
this study Phase 1 -- THE decisive descriptor experiment: grid-alignment rotation test.
2026-07-18

Hypothesis under test (grid-commensurability): the Hexagon anomaly in the
C(alpha) = A cos^p(alpha/2) model is caused not by alpha but by the alignment
of segment directions with the 8-direction vote quantizer grid.

  square  (h=10):        4/4 sides exactly on the grid (0/90/180/270 deg)
  hexagon (circumr=10):  2/6 sides on grid, 4/6 sides 15 deg off
  triangle(circumr=10):  1/3 sides on grid, 2/3 sides 15 deg off
  circle:                tangent sweeps uniformly (mean |qe| = Delta/4)

Test: rotate the square by 22.5 deg -> alpha unchanged (90), corner frequency
unchanged, segment lengths unchanged; ALL 4 sides now maximally off-grid
(22.5 deg). Also rotate hexagon by 7.5 deg -> all 6 sides off-grid by
7.5..22.5. And rotate the square by 45 deg (sides on diagonal grid dirs ->
fully aligned again; control that should match the unrotated square).

  alpha-only model prediction:  C unchanged under any rotation
  grid-alignment prediction:    C(sq22.5) < C(sq0) = C(sq45);
                                hexagon shifts toward its "aligned" value

Protocol: tau in {100,433,1000}ms, 13 speeds, MC=15, troll=5%, N=150,
START PHASE RANDOMIZED per MC run (fix for the finite-horizon sampling bias
found in the stadium probe). Baselines (sq0, hex0) re-run under the SAME
random-start protocol so the comparison is internally consistent.
"""
import os
import sys, json, time, numpy as np
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine")))
import simulation_main as sm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "phase1_rotation_results_2026-07-18.json")

SPEEDS = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
MC, N, TROLL = 15, 150, 0.05
TAUS = [(100,6),(433,26),(1000,60)]

class RotatedPolygon:
    """Regular n-gon, circumradius r, rotated by rot_deg. start() at arc start_arc."""
    def __init__(self, n_sides, radius=10.0, rot_deg=0.0):
        self.name = f"poly{n_sides}_rot{rot_deg}"
        rot = np.radians(rot_deg)
        ang = np.linspace(0, 2*np.pi, n_sides, endpoint=False) + rot
        verts = np.array([[radius*np.cos(a), radius*np.sin(a)] for a in ang])
        verts = np.vstack([verts, verts[0]])
        self.segs = [(verts[i], verts[i+1]) for i in range(n_sides)]
        self.lens = [np.linalg.norm(b-a) for a,b in self.segs]
        self.circ = sum(self.lens)
        self.cum = np.array([0]+list(np.cumsum(self.lens)))
        self.start_arc = 0.0
        # segment direction angles mod 45 (grid misalignment record)
        self.seg_angles = [float(np.degrees(np.arctan2(*(b-a)[::-1])) % 360)
                           for a,b in self.segs]
    def closest(self, p):
        bd,bp,ba = 1e10, self.segs[0][0], 0.
        for i,(a,b) in enumerate(self.segs):
            v=b-a; l2=v@v
            if l2<1e-10: continue
            t=np.clip((p-a)@v/l2,0,1)
            pt=a+t*v; d=np.linalg.norm(p-pt)
            if d<bd: bd,bp,ba=d,pt,self.cum[i]+t*self.lens[i]
        return bp,ba
    def at(self, arc):
        arc=arc%self.circ
        for i,(a,b) in enumerate(self.segs):
            if arc<=self.cum[i+1]+1e-9:
                t=(arc-self.cum[i])/self.lens[i]
                return a+np.clip(t,0,1)*(b-a)
        return self.segs[-1][1]
    def start(self):
        return self.at(self.start_arc)

# square via RotatedPolygon(4, radius) -- side = r*sqrt(2). To match the published
# Square(h=10) side length 20 -> circumradius = 20/sqrt(2) = 14.142.
# hexagon/triangle: circumradius 10, matching unified_vstar_experiment.py.
CASES = {
    "sq_rot0":    RotatedPolygon(4, 20/np.sqrt(2), rot_deg=45.0),   # sides axis-aligned (matches published Square orientation)
    "sq_rot22.5": RotatedPolygon(4, 20/np.sqrt(2), rot_deg=67.5),   # sides 22.5 off-grid (max misalignment)
    "hex_rot0":   RotatedPolygon(6, 10.0, rot_deg=0.0),             # published hexagon: 2/6 aligned
    "hex_rot7.5": RotatedPolygon(6, 10.0, rot_deg=7.5),             # all sides off-grid
    "tri_rot0":   RotatedPolygon(3, 10.0, rot_deg=0.0),             # published triangle
    "tri_rot30":  RotatedPolygon(3, 10.0, rot_deg=30.0),            # sides at 0/120/240 -> 1 aligned, shifted set
}

def qe_of(angles):
    g = np.array(angles) % 45.0
    return np.minimum(g, 45.0-g)

def sweep_random_start(traj, tau_f, phase_seed=2222):
    rng = np.random.default_rng(phase_seed)
    phases = rng.uniform(0, traj.circ, MC)
    orig = sm.DELAY_F; sm.DELAY_F = tau_f
    try:
        rmses = []
        for v in SPEEDS:
            runs = []
            for i in range(MC):
                traj.start_arc = phases[i]
                r = sm.simulate(traj, N, TROLL, seed=i*31+N,
                                method='fixed', speed_override=v)
                runs.append(r['rmse'])
            rmses.append(float(np.mean(runs)))
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
    t0 = time.time()
    out = {"speeds": SPEEDS, "cases": {}}
    for cname, traj in CASES.items():
        mis = qe_of(traj.seg_angles)
        out["cases"][cname] = {
            "circ": traj.circ,
            "seg_angles": traj.seg_angles,
            "grid_misalign_deg": mis.tolist(),
            "rms_misalign": float(np.sqrt(np.mean(mis**2))),
            "taus": {}
        }
        print(f"{cname}: circ={traj.circ:.2f} seg_angles={np.round(traj.seg_angles,1).tolist()} "
              f"misalign={np.round(mis,1).tolist()} RMS={np.sqrt(np.mean(mis**2)):.2f}deg", flush=True)
        for tau_ms, tau_f in TAUS:
            vstar, rmses = sweep_random_start(traj, tau_f)
            xstar = vstar*np.sqrt(tau_ms/1000.0)
            out["cases"][cname]["taus"][str(tau_ms)] = {
                "vstar": vstar, "x_star": xstar, "rmses": rmses,
                "rmse_min": float(min(rmses))}
            print(f"  tau={tau_ms:>4}: v*={vstar:.3f} x*={xstar:.3f} "
                  f"RMSE_min={min(rmses):.4f}  [{time.time()-t0:.0f}s]", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}  total {time.time()-t0:.0f}s", flush=True)
