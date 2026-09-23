# -*- coding: utf-8 -*-
"""MC=50 precision confirmation of four headline conditions. 2026-07-19
Pre-registered MC=15 values / law predictions:
  A continuum circle n=360, tau=433: MC15 v*=1.489, law 1.501
  B hexagon 8-dir random-start, tau=433: MC15 v*=1.276, law 1.300
  C circle 8-dir, tau=433 (dither regime): MC15 v*=2.134
  D circle 8-dir, tau=1500 (lag-capped): MC15 v*=0.869, law(u=0.75) 0.866"""
import os
import sys, json, time, numpy as np
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine")))
import simulation_main as sm
from simulation_main import Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "phase6_mc50_results_2026-07-19.json")
MC, N, TROLL = 50, 150, 0.05

def make_dirs(n):
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(a), np.sin(a)])

def a2d(deg, dirs):
    da = np.degrees(np.arctan2(dirs[:,1], dirs[:,0])) % 360
    d = np.abs(da[None,:] - (np.asarray(deg) % 360)[:,None]); d = np.minimum(d, 360-d)
    return np.argmin(d, axis=1)

def gv360(ia, pa, tr, na, rng):
    dirs = make_dirs(360)
    nt = min(round(na*tr), na); rem = na - nt
    nacc = round(rem*0.7368); nslow = round(rem*0.2105); noth = rem-nacc-nslow
    if noth < 0: nacc += noth; noth = 0
    ang = np.empty(nacc+nslow+noth); i = 0
    ang[i:i+nacc] = ia + rng.uniform(-3,3,nacc); i += nacc
    diff = (ia - pa + 180) % 360 - 180
    if nslow: ang[i:i+nslow] = pa + diff*(1-rng.uniform(0.2,0.5,nslow)); i += nslow
    if noth: ang[i:i+noth] = ia + rng.uniform(-30,30,noth); i += noth
    return np.concatenate([a2d(ang[:i], dirs),
                           rng.integers(0,360,nt) if nt else np.array([],dtype=int)])

class Hexagon:
    def __init__(self, radius=10.0):
        a = np.linspace(0, 2*np.pi, 6, endpoint=False)
        v = np.array([[radius*np.cos(x), radius*np.sin(x)] for x in a])
        v = np.vstack([v, v[0]])
        self.segs = [(v[i], v[i+1]) for i in range(6)]
        self.lens = [np.linalg.norm(b-a_) for a_,b in self.segs]
        self.circ = sum(self.lens)
        self.cum = np.array([0]+list(np.cumsum(self.lens)))
        self.start_arc = 0.0
    def closest(self, p):
        bd,bp,ba = 1e10, self.segs[0][0], 0.
        for i,(a_,b) in enumerate(self.segs):
            w=b-a_; l2=w@w
            if l2<1e-10: continue
            t=np.clip((p-a_)@w/l2,0,1); pt=a_+t*w; d=np.linalg.norm(p-pt)
            if d<bd: bd,bp,ba=d,pt,self.cum[i]+t*self.lens[i]
        return bp,ba
    def at(self, arc):
        arc=arc%self.circ
        for i,(a_,b) in enumerate(self.segs):
            if arc<=self.cum[i+1]+1e-9:
                return a_+np.clip((arc-self.cum[i])/self.lens[i],0,1)*(b-a_)
        return self.segs[-1][1]
    def start(self): return self.at(self.start_arc)

def refine(sp, rm):
    i = int(np.argmin(rm)); v = float(sp[i])
    if 1 <= i <= len(sp)-2:
        c = np.polyfit(sp[i-1:i+2], rm[i-1:i+2], 2)
        if c[0] > 0:
            vf = -c[1]/(2*c[0])
            if sp[i-1] <= vf <= sp[i+1]: v = float(vf)
    return v

def sweep(traj, tau_f, speeds, n_dirs=8, random_start=False, phase_seed=888):
    orig = (sm.DIRS, sm.gen_votes, sm.DELAY_F)
    if n_dirs == 360:
        sm.DIRS = make_dirs(360); sm.gen_votes = gv360
    sm.DELAY_F = tau_f
    phases = (np.random.default_rng(phase_seed).uniform(0, traj.circ, MC)
              if random_start else None)
    try:
        rm = []
        for v in speeds:
            runs = []
            for i in range(MC):
                if phases is not None: traj.start_arc = phases[i]
                runs.append(sm.simulate(traj, N, TROLL, seed=i*31+N,
                                        method='fixed', speed_override=v)['rmse'])
            rm.append(float(np.mean(runs)))
    finally:
        sm.DIRS, sm.gen_votes, sm.DELAY_F = orig
    return refine(speeds, rm), rm

if __name__ == "__main__":
    t0 = time.time(); out = {}
    SP = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
    SPF = [0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,2.0,2.2]
    SPL = [0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.1,1.25,1.5]

    v,_ = sweep(Circle(), 26, SPF, n_dirs=360)
    out["A_continuum_433"] = v
    print(f"A continuum tau=433: MC50 v*={v:.3f} (MC15 1.489, law 1.501) [{time.time()-t0:.0f}s]", flush=True)

    v,_ = sweep(Hexagon(), 26, SP, n_dirs=8, random_start=True)
    out["B_hexagon_433"] = v
    print(f"B hexagon tau=433: MC50 v*={v:.3f} (MC15 1.276, law 1.300) [{time.time()-t0:.0f}s]", flush=True)

    v,_ = sweep(Circle(), 26, SP, n_dirs=8)
    out["C_circle8_433"] = v
    print(f"C circle 8-dir tau=433: MC50 v*={v:.3f} (MC15 2.134) [{time.time()-t0:.0f}s]", flush=True)

    v,_ = sweep(Circle(), 90, SPL, n_dirs=8)
    out["D_circle8_1500"] = v
    print(f"D circle 8-dir tau=1500: MC50 v*={v:.3f} (MC15 0.869, lag-cap 0.866) [{time.time()-t0:.0f}s]", flush=True)

    json.dump(out, open(OUT, "w"), indent=2)
    print("Saved:", OUT, flush=True)
