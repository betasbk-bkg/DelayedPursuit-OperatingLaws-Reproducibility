"""
this study -- alpha sweep: discriminate cos(alpha/2) functional form and the
1/2 vs 1/e amplitude degeneracy. 2026-07-19

New polygons (8-dir votes, random start, MC=15, troll=5%, N=150, L=2):
  n=12 (alpha=30), side 12 m  (R = 6/sin(15deg)  = 23.18)
  n=8  (alpha=45), side 12 m  (R = 6/sin(22.5)   = 15.68), rotated so ALL sides
       are grid-aligned (octagon is the fully commensurate polygon: 45deg = Delta)
  n=5  (alpha=72), side 12 m  (R = 6/sin(36)     = 10.21)
Existing anchors (same protocol, from rotation test): alpha=60 (0.428),
90 (0.362), 120 (0.279/0.280).

PRE-REGISTERED predictions for u* = v**(tau+0.2333)/L:
  H1  u* = (1/2)cos(alpha/2):        30:0.483  45:0.462  72:0.404
  H1' u* = (1/e)/cos(45)*cos(a/2):   30:0.503  45:0.481  72:0.420  (+4.2% shape-identical)
  H2  u* = 1/e (alpha-independent):  0.368 everywhere (already rejected by hex/tri)
Side lengths (12 m) keep seg/L = 6 >= 5 (validity domain).
Possible discovery mode: upturn of u* at alpha=30 toward the 8-dir circle value
(0.65-0.75) would mark the corner->dither crossover.
"""
import os
import sys, json, time, numpy as np
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine")))
import simulation_main as sm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "phase3_alpha_sweep_results_2026-07-19.json")

SPEEDS = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
MC, N, TROLL = 15, 150, 0.05
TAUS = [(100,6),(433,26),(1000,60)]
C_LAG = 0.2333; L = 2.0

class RegularPolygonR:
    def __init__(self, n_sides, side_len, rot_deg=0.0):
        self.name = f"poly{n_sides}_rot{rot_deg}"
        self.n = n_sides
        R = side_len/(2*np.sin(np.pi/n_sides))
        rot = np.radians(rot_deg)
        ang = np.linspace(0, 2*np.pi, n_sides, endpoint=False) + rot
        verts = np.array([[R*np.cos(a), R*np.sin(a)] for a in ang])
        verts = np.vstack([verts, verts[0]])
        self.segs = [(verts[i], verts[i+1]) for i in range(n_sides)]
        self.lens = [np.linalg.norm(b-a) for a,b in self.segs]
        self.circ = sum(self.lens)
        self.cum = np.array([0]+list(np.cumsum(self.lens)))
        self.start_arc = 0.0
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

def qe(angles):
    g = np.array(angles) % 45.0
    return np.minimum(g, 45.0-g)

def refine_vstar(speeds, rmses):
    idx = int(np.argmin(rmses)); vstar = float(speeds[idx])
    if 1 <= idx <= len(speeds)-2:
        c = np.polyfit(speeds[idx-1:idx+2], rmses[idx-1:idx+2], 2)
        if c[0] > 0:
            vf = -c[1]/(2*c[0])
            if speeds[idx-1] <= vf <= speeds[idx+1]:
                vstar = float(vf)
    return vstar

def sweep_random_start(traj, tau_f, phase_seed=555):
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
    return refine_vstar(SPEEDS, rmses), rmses

# rotations chosen to avoid cell-boundary (22.5+-3deg) sides where possible
CASES = {
    # n=12: rot=0 -> side dirs step 30deg: mod45 pattern {15,0,30->15...}; check printout
    "dodecagon_a30": (RegularPolygonR(12, 12.0, rot_deg=0.0), 30.0),
    # n=8: rot chosen so sides land on grid multiples: vertices every 45deg,
    # side dir = vertex angle + 90 + 22.5 -> rot=22.5 puts sides ON grid
    "octagon_a45":   (RegularPolygonR(8, 12.0, rot_deg=22.5), 45.0),
    "pentagon_a72":  (RegularPolygonR(5, 12.0, rot_deg=0.0), 72.0),
}

if __name__ == "__main__":
    t0 = time.time()
    out = {"speeds": SPEEDS, "cases": {}}
    for cname, (traj, alpha) in CASES.items():
        mis = qe(traj.seg_angles)
        boundary_frac = float(np.mean(np.abs(mis-22.5) < 3.0))
        out["cases"][cname] = dict(alpha=alpha, circ=traj.circ,
            seg_angles=traj.seg_angles, misalign=mis.tolist(),
            boundary_frac=boundary_frac, taus={})
        print(f"{cname}: alpha={alpha} circ={traj.circ:.1f} "
              f"misalign={np.round(mis,1).tolist()} boundary_frac={boundary_frac:.2f}",
              flush=True)
        for tau_ms, tau_f in TAUS:
            vstar, rmses = sweep_random_start(traj, tau_f)
            tau_s = tau_ms/1000
            u = vstar*(tau_s+C_LAG)/L
            out["cases"][cname]["taus"][str(tau_ms)] = dict(
                vstar=vstar, u_tot=u, rmses=rmses, rmse_min=float(min(rmses)))
            h1 = 0.5*np.cos(np.radians(alpha/2))
            print(f"  tau={tau_ms:>4}: v*={vstar:.3f} u_tot={u:.3f} "
                  f"(H1 pred {h1:.3f}, err {100*(u-h1)/h1:+.1f}%)  "
                  f"RMSE_min={min(rmses):.4f}  [{time.time()-t0:.0f}s]", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}  total {time.time()-t0:.0f}s", flush=True)
