"""
this study -- c-decomposition test: verify each term of tau_tot = tau + WIN/2 + DT/SMOOTH
independently by varying WIN and SMOOTH. 2026-07-18

Until now only the SUM c = 0.2333 was verified (fit 0.231+-0.005). The claimed
decomposition (ZOH average hold WIN/2 + smoothing time constant DT/SMOOTH) makes
distinct predictions when WIN or SMOOTH change:

  PRE-REGISTERED (continuum circle n=360, tau=433ms, L=2, v* = 1/(0.433+c)):
    baseline WIN=0.3 SMOOTH=0.2 : c=0.2333 -> v*=1.501   (obs 1.489, known)
    WIN=0.6  (VOTE_INT=36)      : c=0.3833 -> v*=1.225
    WIN=0.1  (VOTE_INT=6)       : c=0.1333 -> v*=1.766
    SMOOTH=0.1 (T_s=1/6 s)      : c=0.3167 -> v*=1.334
    SMOOTH=0.5 (T_s=1/30 s)     : c=0.1833 -> v*=1.623

Note: gen_votes noise statistics change slightly with WIN (slow-agent lag is
per-vote-window), a second-order confound acknowledged up front.
"""
import os
import sys, json, time, numpy as np
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine")))
import simulation_main as sm
from simulation_main import Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "phase2_c_decomp_results_2026-07-18.json")

MC, N, TROLL, TAU_F, TAU_S = 15, 150, 0.05, 26, 0.433
SPEEDS = [0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2]
DT = 1/60

def make_dirs(n):
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])

def angle_to_dir_n(angles_deg, dirs):
    da = np.degrees(np.arctan2(dirs[:,1], dirs[:,0])) % 360
    a = np.asarray(angles_deg) % 360
    d = np.abs(da[None,:] - a[:,None]); d = np.minimum(d, 360-d)
    return np.argmin(d, axis=1)

def patched_gv(ideal_angle, prev_angle, troll_ratio, N_agents, rng, n_dirs=360):
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

def sweep(vote_int, smooth):
    orig = (sm.DIRS, sm.gen_votes, sm.DELAY_F, sm.VOTE_INT, sm.SMOOTH)
    sm.DIRS = make_dirs(360)
    sm.gen_votes = lambda ia,pa,tr,na,rng: patched_gv(ia,pa,tr,na,rng,360)
    sm.DELAY_F, sm.VOTE_INT, sm.SMOOTH = TAU_F, vote_int, smooth
    try:
        rmses = []
        for v in SPEEDS:
            r = sm.run_condition(Circle(), N, TROLL, MC, method='fixed',
                                  speed_override=v, seed_base=31, seed_offset=N)
            rmses.append(r['rmse_mean'])
    finally:
        sm.DIRS, sm.gen_votes, sm.DELAY_F, sm.VOTE_INT, sm.SMOOTH = orig
    return refine_vstar(SPEEDS, rmses), rmses

CASES = [
    ("baseline",   18, 0.2),
    ("WIN=0.6",    36, 0.2),
    ("WIN=0.1",     6, 0.2),
    ("SMOOTH=0.1", 18, 0.1),
    ("SMOOTH=0.5", 18, 0.5),
]

if __name__ == "__main__":
    t0 = time.time()
    out = {"speeds": SPEEDS, "cases": {}}
    for name, vi, smo in CASES:
        win = vi*DT
        c_pred = win/2 + DT/smo
        v_pred = 1.0/(TAU_S + c_pred)
        vstar, rmses = sweep(vi, smo)
        c_obs = 1.0/vstar - TAU_S
        out["cases"][name] = dict(vote_int=vi, smooth=smo, c_pred=c_pred,
                                   v_pred=v_pred, vstar=vstar, c_obs=c_obs,
                                   rmses=rmses)
        print(f"{name:<11} WIN={win:.2f} SMOOTH={smo}: c_pred={c_pred:.4f} "
              f"v*_pred={v_pred:.3f} | v*_obs={vstar:.3f} c_obs={c_obs:.4f} "
              f"err={100*(vstar-v_pred)/v_pred:+.1f}%  [{time.time()-t0:.0f}s]", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT}", flush=True)
