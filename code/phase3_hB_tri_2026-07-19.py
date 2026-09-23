"""
this study -- remaining-items experiments. 2026-07-19
(1) h(m) sweep: square rotated by theta in {5.625, 11.25, 16.875, 19.6875} deg
    (all four sides share misalignment m = theta). With known m=0 (u*=0.362)
    and m=22.5 (u*=0.170), this maps the margin-damage function h(m).
    Relay-hunting theory predicts h ~ 1 until the vote-split zone
    (m > 22.5 - ~3deg accurate-noise half-width), then a sharp drop.
(2) triangle tau-extension: tau in {1500, 2000} ms, random start -- does the
    +10% anomaly persist in the deep lag regime?
Protocol: 8-dir, N=150, troll 5%, MC=15, random start phase.
"""
import os
import sys, json, time, numpy as np
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine")))
import simulation_main as sm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "phase3_hB_tri_results_2026-07-19.json")

MC, N, TROLL = 15, 150, 0.05
C_LAG, L = 0.2333, 2.0
SPEEDS = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
SPEEDS_LOWT = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,1.0]

class RotatedPolygon:
    def __init__(self, n_sides, radius, rot_deg=0.0, name=None):
        self.name = name or f"poly{n_sides}_rot{rot_deg}"
        rot = np.radians(rot_deg)
        ang = np.linspace(0, 2*np.pi, n_sides, endpoint=False) + rot
        verts = np.array([[radius*np.cos(a), radius*np.sin(a)] for a in ang])
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

def refine_vstar(speeds, rmses):
    idx = int(np.argmin(rmses)); vstar = float(speeds[idx])
    if 1 <= idx <= len(speeds)-2:
        c = np.polyfit(speeds[idx-1:idx+2], rmses[idx-1:idx+2], 2)
        if c[0] > 0:
            vf = -c[1]/(2*c[0])
            if speeds[idx-1] <= vf <= speeds[idx+1]:
                vstar = float(vf)
    return vstar

def sweep(traj, tau_f, speeds, phase_seed=777):
    rng = np.random.default_rng(phase_seed)
    phases = rng.uniform(0, traj.circ, MC)
    orig = sm.DELAY_F; sm.DELAY_F = tau_f
    try:
        rmses = []
        for v in speeds:
            runs = []
            for i in range(MC):
                traj.start_arc = phases[i]
                r = sm.simulate(traj, N, TROLL, seed=i*31+N,
                                method='fixed', speed_override=v)
                runs.append(r['rmse'])
            rmses.append(float(np.mean(runs)))
    finally:
        sm.DELAY_F = orig
    return refine_vstar(speeds, rmses), rmses

if __name__ == "__main__":
    t0 = time.time()
    out = {"h_sweep": {}, "tri_ext": {}}

    print("=== (1) h(m): square rotations ===", flush=True)
    R_sq = 20/np.sqrt(2)
    for theta in [5.625, 11.25, 16.875, 19.6875]:
        # base square (rot=45) has sides on grid; extra rotation theta -> m = theta
        traj = RotatedPolygon(4, R_sq, rot_deg=45.0+theta, name=f"sq_m{theta}")
        mis = np.array(traj.seg_angles) % 45.0
        mis = np.minimum(mis, 45-mis)
        out["h_sweep"][str(theta)] = {"misalign": mis.tolist(), "taus": {}}
        print(f"sq m={theta}: misalign={np.round(mis,2).tolist()}", flush=True)
        for tau_ms, tau_f in [(433,26),(1000,60)]:
            vstar, rmses = sweep(traj, tau_f, SPEEDS)
            u = vstar*(tau_ms/1000+C_LAG)/L
            out["h_sweep"][str(theta)]["taus"][str(tau_ms)] = dict(
                vstar=vstar, u_tot=u, rmses=rmses)
            print(f"  tau={tau_ms}: v*={vstar:.3f} u_tot={u:.3f} "
                  f"h=u/0.362={u/0.362:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    print("=== (2) triangle tau-extension ===", flush=True)
    tri = RotatedPolygon(3, 10.0, rot_deg=0.0, name="triangle")
    for tau_ms, tau_f in [(1500,90),(2000,120)]:
        vstar, rmses = sweep(tri, tau_f, SPEEDS_LOWT)
        tau_s = tau_ms/1000
        u = vstar*(tau_s+C_LAG)/L
        pred = 0.25  # (1/2)cos(60deg)
        out["tri_ext"][str(tau_ms)] = dict(vstar=vstar, u_tot=u, rmses=rmses)
        print(f"  tau={tau_ms}: v*={vstar:.3f} u_tot={u:.3f} "
              f"(law 0.250, dev {100*(u-pred)/pred:+.1f}%)  [{time.time()-t0:.0f}s]", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT}  total {time.time()-t0:.0f}s", flush=True)
