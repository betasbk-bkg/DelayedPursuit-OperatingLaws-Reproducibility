"""
this study -- L (lookahead) variation sweep: the decisive test of L-proportionality.
2026-07-18

Every headline law contains L, but ALL prior data used L=2 m only. The laws
predict v* proportional to L with NO other change:

  PRE-REGISTERED zero-parameter predictions (tau=433 ms, tau_tot=0.6663):
    continuum circle  : v* = (L/2)/tau_tot          -> L=1: 0.750 | L=4: 3.002
    square (8-dir)    : v* = (L/2)cos45/tau_tot     -> L=1: 0.531 | L=4: 2.123
    hexagon (8-dir)   : v* = (L/2)cos30/tau_tot     -> L=1: 0.650 | L=4: 2.600
    circle 8-dir plateau = L*Delta/sqrt(12)         -> L=1: 0.227 | L=4: 0.907 m
    circle 8-dir optimum x* prop. L (if x_q prop. L)-> L=1: ~0.66 | L=4: ~2.65
  Failure of any row falsifies the corresponding law's L-structure.

Also continuum at tau=200 and tau=1000 for the full lag-law shape at each L.

Protocol: N=150, troll=5%, MC=15; polygons + stadium-free; random start phase
for polygons (protocol standard from this session), fixed start for circle
(rotation-symmetric). LOOK monkey-patched (single use site, line 288).
"""
import os
import sys, json, time, numpy as np
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine")))
import simulation_main as sm
from simulation_main import Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "phase2_L_sweep_results_2026-07-18.json")

MC, N, TROLL = 15, 150, 0.05
C_LAG = 0.2333

SPEED_GRIDS = {
    1.0: [0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5,2.0,2.5],
    4.0: [0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.5],
}

class RotatedPolygon:
    def __init__(self, n_sides, radius=10.0, rot_deg=0.0, name=None):
        self.name = name or f"poly{n_sides}"
        rot = np.radians(rot_deg)
        ang = np.linspace(0, 2*np.pi, n_sides, endpoint=False) + rot
        verts = np.array([[radius*np.cos(a), radius*np.sin(a)] for a in ang])
        verts = np.vstack([verts, verts[0]])
        self.segs = [(verts[i], verts[i+1]) for i in range(n_sides)]
        self.lens = [np.linalg.norm(b-a) for a,b in self.segs]
        self.circ = sum(self.lens)
        self.cum = np.array([0]+list(np.cumsum(self.lens)))
        self.start_arc = 0.0
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

def make_dirs(n):
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])

def angle_to_dir_n(angles_deg, dirs):
    da = np.degrees(np.arctan2(dirs[:,1], dirs[:,0])) % 360
    a = np.asarray(angles_deg) % 360
    d = np.abs(da[None,:] - a[:,None]); d = np.minimum(d, 360-d)
    return np.argmin(d, axis=1)

def patched_gv(ideal_angle, prev_angle, troll_ratio, N_agents, rng, n_dirs):
    dirs = make_dirs(n_dirs)
    n_troll = min(round(N_agents*troll_ratio), N_agents)
    remaining = N_agents - n_troll
    n_acc = round(remaining*0.7368); n_slow = round(remaining*0.2105)
    n_oth = remaining - n_acc - n_slow
    if n_oth < 0: n_acc += n_oth; n_oth = 0
    ang = np.empty(n_acc+n_slow+n_oth); i = 0
    ang[i:i+n_acc] = ideal_angle + rng.uniform(-3,3,n_acc); i += n_acc
    diff = ideal_angle - prev_angle
    if diff > 180: diff -= 360
    if diff < -180: diff += 360
    if n_slow > 0:
        lag = rng.uniform(0.2,0.5,n_slow)
        ang[i:i+n_slow] = prev_angle + diff*(1-lag); i += n_slow
    if n_oth > 0:
        ang[i:i+n_oth] = ideal_angle + rng.uniform(-30,30,n_oth); i += n_oth
    nt = angle_to_dir_n(ang[:i], dirs)
    tv = rng.integers(0, n_dirs, n_troll) if n_troll>0 else np.array([],dtype=int)
    return np.concatenate([nt, tv])

def refine_vstar(speeds, rmses):
    idx = int(np.argmin(rmses)); vstar = float(speeds[idx])
    if 1 <= idx <= len(speeds)-2:
        c = np.polyfit(speeds[idx-1:idx+2], rmses[idx-1:idx+2], 2)
        if c[0] > 0:
            vf = -c[1]/(2*c[0])
            if speeds[idx-1] <= vf <= speeds[idx+1]:
                vstar = float(vf)
    return vstar

def sweep(traj, tau_f, n_dirs, L, speeds, random_start=False, phase_seed=333):
    orig = (sm.DIRS, sm.gen_votes, sm.DELAY_F, sm.LOOK)
    if n_dirs != 8:
        sm.DIRS = make_dirs(n_dirs)
        sm.gen_votes = lambda ia,pa,tr,na,rng: patched_gv(ia,pa,tr,na,rng,n_dirs)
    sm.DELAY_F = tau_f
    sm.LOOK = L
    try:
        phases = (np.random.default_rng(phase_seed).uniform(0, traj.circ, MC)
                  if random_start else None)
        rmses = []
        for v in speeds:
            runs = []
            for i in range(MC):
                if phases is not None:
                    traj.start_arc = phases[i]
                r = sm.simulate(traj, N, TROLL, seed=i*31+N,
                                method='fixed', speed_override=v)
                runs.append(r['rmse'])
            rmses.append(float(np.mean(runs)))
    finally:
        sm.DIRS, sm.gen_votes, sm.DELAY_F, sm.LOOK = orig
    return refine_vstar(speeds, rmses), rmses

if __name__ == "__main__":
    t0 = time.time()
    out = {"speed_grids": {str(k): v for k,v in SPEED_GRIDS.items()}, "runs": []}
    circle = Circle()
    square  = RotatedPolygon(4, 20/np.sqrt(2), rot_deg=45.0, name="square")
    hexagon = RotatedPolygon(6, 10.0, rot_deg=0.0, name="hexagon")

    def pred(L, alpha_deg, tau_s):
        return 0.5*L*np.cos(np.radians(alpha_deg/2))/(tau_s + C_LAG)

    JOBS = []
    for L in [1.0, 4.0]:
        # continuum circle, 3 tau
        for tau_ms, tau_f in [(200,12),(433,26),(1000,60)]:
            JOBS.append(dict(name="circle_n360", traj=circle, n_dirs=360, L=L,
                             tau_ms=tau_ms, tau_f=tau_f, rs=False,
                             pred=pred(L, 0, tau_ms/1000)))
        # 8-dir circle (plateau + dither x*), tau=433
        JOBS.append(dict(name="circle_n8", traj=circle, n_dirs=8, L=L,
                         tau_ms=433, tau_f=26, rs=False, pred=None))
        # polygons 8-dir, tau=433, random start
        JOBS.append(dict(name="square_n8", traj=square, n_dirs=8, L=L,
                         tau_ms=433, tau_f=26, rs=True, pred=pred(L, 90, 0.433)))
        JOBS.append(dict(name="hexagon_n8", traj=hexagon, n_dirs=8, L=L,
                         tau_ms=433, tau_f=26, rs=True, pred=pred(L, 60, 0.433)))

    for job in JOBS:
        speeds = SPEED_GRIDS[job["L"]]
        vstar, rmses = sweep(job["traj"], job["tau_f"], job["n_dirs"], job["L"],
                             speeds, random_start=job["rs"])
        rec = dict(name=job["name"], L=job["L"], tau_ms=job["tau_ms"],
                   vstar=vstar, rmses=rmses, prediction=job["pred"])
        out["runs"].append(rec)
        ptxt = f" pred={job['pred']:.3f} err={100*(vstar-job['pred'])/job['pred']:+.1f}%" if job["pred"] else ""
        print(f"{job['name']:<12} L={job['L']} tau={job['tau_ms']:>4}: v*={vstar:.3f}{ptxt}"
              f"  low-v rmse[0]={rmses[0]:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}  total {time.time()-t0:.0f}s", flush=True)
