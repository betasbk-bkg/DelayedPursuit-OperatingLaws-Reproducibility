"""
this study Phase 2 -- tau-extension discriminating experiment. 2026-07-18

Within tau in [0.1, 1.0] s the two candidate laws for the 8-direction system,
    (P) power/collapse law    v* = 1.347 / sqrt(tau)
    (H) lag-compensation law  v* = K / (tau + 0.2333)
are numerically near-degenerate (both fit published v* within ~7%).
They diverge for tau >= 2 s:
    tau=2.0 s: (P) 0.953   (H, K=1.4 8-dir empirical) 0.627
    tau=3.0 s: (P) 0.778   (H) 0.433
This experiment measures v* at tau = 1500, 2000, 3000 ms for the 8-direction
circle (published protocol: fixed start, N=150, troll=5%, MC=15) and at
n_dirs=360 (lag-law out-of-sample check, prediction v* = 1.0/(tau+0.2333),
zero free parameters).

Speed grid refined at the low end because predicted v* < 1.
"""
import os
import sys, json, time, numpy as np
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine")))
import simulation_main as sm
from simulation_main import Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "phase2_tau_ext_results_2026-07-18.json")

SPEEDS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.25,1.5,1.75,2.0]
MC, N, TROLL = 15, 150, 0.05
TAUS = [(1500,90),(2000,120),(3000,180)]

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

def sweep(traj, tau_f, n_dirs):
    orig_dirs, orig_gv, orig_delay = sm.DIRS, sm.gen_votes, sm.DELAY_F
    if n_dirs != 8:
        sm.DIRS = make_dirs(n_dirs)
        sm.gen_votes = lambda ia,pa,tr,na,rng: patched_gv(ia,pa,tr,na,rng,n_dirs)
    sm.DELAY_F = tau_f
    try:
        rmses = []
        for v in SPEEDS:
            r = sm.run_condition(traj, N, TROLL, MC, method='fixed',
                                  speed_override=v, seed_base=31, seed_offset=N)
            rmses.append(r['rmse_mean'])
    finally:
        sm.DIRS, sm.gen_votes, sm.DELAY_F = orig_dirs, orig_gv, orig_delay
    idx = int(np.argmin(rmses)); vstar = float(SPEEDS[idx])
    if 1 <= idx <= len(SPEEDS)-2:
        c = np.polyfit(SPEEDS[idx-1:idx+2], rmses[idx-1:idx+2], 2)
        if c[0] > 0:
            vf = -c[1]/(2*c[0])
            if SPEEDS[idx-1] <= vf <= SPEEDS[idx+1]:
                vstar = float(vf)
    return vstar, rmses

if __name__ == "__main__":
    circle = Circle()
    t0 = time.time()
    out = {"speeds": SPEEDS, "results": {}}
    for n_dirs in [8, 360]:
        out["results"][str(n_dirs)] = {}
        for tau_ms, tau_f in TAUS:
            tau_s = tau_ms/1000
            vstar, rmses = sweep(circle, tau_f, n_dirs)
            pP = 1.347/np.sqrt(tau_s)
            pH8 = 1.4/(tau_s+0.2333)
            pH360 = 1.0/(tau_s+0.2333)
            out["results"][str(n_dirs)][str(tau_ms)] = {"vstar": vstar, "rmses": rmses}
            print(f"n={n_dirs:>3} tau={tau_ms}: v*={vstar:.3f} | power-law pred {pP:.3f} | "
                  f"lag-law pred {'%.3f'%(pH8 if n_dirs==8 else pH360)} | "
                  f"rmses={np.round(rmses,3).tolist()} [{time.time()-t0:.0f}s]", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT}", flush=True)
