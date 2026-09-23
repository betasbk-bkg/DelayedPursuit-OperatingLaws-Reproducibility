"""
this study -- transfer ladder: where does the lag law hold, where does it break?
2026-07-19. All systems are independent implementations (no the reference engine engine code).

Base law under test:  v* = (L/2) / (tau + T_ctrl/2 + T_act_mean)
  T_act_mean = mean delay (first moment) of the actuator filter.

PRE-REGISTERED predictions per system (L=3, R=15 unless noted, tau=0.3):
  S2 bicycle (wheelbase b=2, steering lag T_s=0.25): c=0.05+0.25=0.30 -> v*=2.50
     (uncertain whether the heading-integrator adds lag -- deviation informative)
  S3 heading-rate saturation w_max in {1.0, 0.3, 0.12} rad/s:
     steady turn rate at optimum = v*/R = 0.167 -> predict OK / marginal / broken
  S4 second-order heading (zeta=1, wn=8 -> 2*zeta/wn=0.25): c=0.30 -> v*=2.50
  S5 ellipse a=20,b=10 (curvature 0.05..0.2): v* SAME 2.50 (curvature-independent
     optimum -- sharp corollary of Theorem 1)
  S6 continuum square (side 24, alpha=90, Gaussian noise, NO quantization):
     u* = 0.5*cos(45) -> v* = 0.354*3/0.6 = 1.77 (chord law without quantizer)
  S7 noise x5 (sigma=10 deg): small downward shift of v* expected (<10%)
  S8 T_ctrl=0.5 s (> tau): c=0.25+0.25=0.50 -> v*=1.875
"""
import os
import numpy as np, json, time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "phase4_transfer_ladder_results_2026-07-19.json")

DT, DUR = 0.01, 120.0
L, R = 3.0, 15.0
MC = 8

# ---------- paths ----------
class CirclePath:
    def __init__(self, R): self.R = R
    def project_aim(self, p, L):
        th = np.arctan2(p[1], p[0])
        aim_th = th + L/self.R
        return self.R*np.array([np.cos(aim_th), np.sin(aim_th)])
    def err(self, p): return abs(np.hypot(*p) - self.R)
    def start(self): return np.array([self.R, 0.0]), np.pi/2

class TablePath:
    """arc-length table path (ellipse, polygon...)"""
    def __init__(self, pts):
        self.pts = pts
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(d)])
        self.circ = self.cum[-1]
    def project_aim(self, p, L):
        i = np.argmin(np.sum((self.pts - p)**2, axis=1))
        s_aim = (self.cum[i] + L) % self.circ
        j = np.searchsorted(self.cum, s_aim) % len(self.pts)
        return self.pts[j]
    def err(self, p):
        return float(np.sqrt(np.min(np.sum((self.pts - p)**2, axis=1))))
    def start(self):
        d = self.pts[1]-self.pts[0]
        return self.pts[0].copy(), np.arctan2(d[1], d[0])

def ellipse_path(a=20, b=10, n=3000):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return TablePath(np.column_stack([a*np.cos(t), b*np.sin(t)]))

def square_path(side=24.0, n_per=800):
    h = side/2
    corners = np.array([[h,-h],[h,h],[-h,h],[-h,-h],[h,-h]])
    pts = []
    for i in range(4):
        for f in np.linspace(0, 1, n_per, endpoint=False):
            pts.append(corners[i]*(1-f) + corners[i+1]*f)
    return TablePath(np.array(pts))

# ---------- generic simulator ----------
def simulate(path, v, tau, seed, T_ctrl=0.10, sigma_deg=2.0,
             actuator="first_order", T_f=0.25, w_max=None,
             zeta=1.0, wn=8.0, wheelbase=2.0):
    rng = np.random.default_rng(seed)
    n = int(DUR/DT); ce = max(1, int(T_ctrl/DT)); dl = int(tau/DT)
    pos, heading = path.start()
    cmd = heading
    hist = [pos.copy()]
    errs = np.empty(n)
    hrate = 0.0        # for second_order
    steer = 0.0        # for bicycle
    for k in range(n):
        if k % ce == 0:
            dp = hist[max(0, len(hist)-1-dl)]
            aim = path.project_aim(dp, L)
            d = aim - dp
            cmd = np.arctan2(d[1], d[0]) + rng.normal(0, np.radians(sigma_deg))
        dh = (cmd - heading + np.pi) % (2*np.pi) - np.pi
        if actuator == "first_order":
            rate = dh/T_f
            if w_max is not None:
                rate = np.clip(rate, -w_max, w_max)
            heading += DT*rate
        elif actuator == "second_order":
            acc = wn*wn*dh - 2*zeta*wn*hrate
            hrate += DT*acc
            heading += DT*hrate
        elif actuator == "bicycle":
            # steering angle lags desired; desired steer from heading error
            # (simple mapping: delta_des = atan(2*b*sin(dh)/L))
            delta_des = np.arctan2(2*wheelbase*np.sin(dh), L)
            steer += (DT/T_f)*(delta_des - steer)
            heading += DT*v*np.tan(steer)/wheelbase
        pos = pos + v*DT*np.array([np.cos(heading), np.sin(heading)])
        hist.append(pos.copy())
        errs[k] = path.err(pos)
    return float(np.sqrt(np.mean(errs**2)))

def vstar(path, tau, pred, span=(0.4,1.8), npts=13, **kw):
    speeds = np.round(np.linspace(span[0]*pred, span[1]*pred, npts), 3)
    rmses = []
    for v in speeds:
        rmses.append(float(np.mean([simulate(path, v, tau, 1000+7*i, **kw)
                                     for i in range(MC)])))
    i = int(np.argmin(rmses)); vs = float(speeds[i])
    if 1 <= i <= npts-2:
        c = np.polyfit(speeds[i-1:i+2], rmses[i-1:i+2], 2)
        if c[0] > 0:
            vf = -c[1]/(2*c[0])
            if speeds[i-1] <= vf <= speeds[i+1]: vs = float(vf)
    return vs, speeds.tolist(), rmses

if __name__ == "__main__":
    t0 = time.time(); out = {}
    circle = CirclePath(R)
    TAU = 0.3

    def report(name, pred, vs, extra=""):
        err = 100*(vs-pred)/pred
        verdict = "PASS" if abs(err) <= 7 else ("MARGINAL" if abs(err) <= 15 else "BREAK")
        print(f"{name:<28} pred={pred:.3f} obs={vs:.3f} err={err:+6.1f}%  [{verdict}] {extra}"
              f"  [{time.time()-t0:.0f}s]", flush=True)
        return dict(pred=pred, obs=vs, err_pct=err, verdict=verdict)

    # S2 bicycle
    pred = (L/2)/(TAU + 0.05 + 0.25)
    vs,_,_ = vstar(circle, TAU, pred, actuator="bicycle", T_f=0.25)
    out["S2_bicycle"] = report("S2 bicycle b=2", pred, vs)

    # S3 saturation
    for wm in [1.0, 0.3, 0.12]:
        pred = (L/2)/(TAU + 0.30)
        vs,_,_ = vstar(circle, TAU, pred, w_max=wm)
        out[f"S3_sat_{wm}"] = report(f"S3 saturation w_max={wm}", pred, vs,
                                      extra=f"(turn-rate demand v*/R={pred/R:.3f})")

    # S4 second order
    pred = (L/2)/(TAU + 0.05 + 2*1.0/8.0)
    vs,_,_ = vstar(circle, TAU, pred, actuator="second_order", zeta=1.0, wn=8.0)
    out["S4_second_order"] = report("S4 2nd-order zeta=1 wn=8", pred, vs)

    # S5 ellipse
    ell = ellipse_path()
    pred = (L/2)/(TAU + 0.30)
    vs,_,_ = vstar(ell, TAU, pred)
    out["S5_ellipse"] = report("S5 ellipse a20 b10", pred, vs,
                                extra="(curvature-independent optimum corollary)")

    # S6 continuum square (chord law without quantization)
    sq = square_path()
    pred = 0.5*np.cos(np.radians(45))*L/(TAU + 0.30)
    vs, sp, rm = vstar(sq, TAU, pred, span=(0.4, 2.2), npts=15)
    out["S6_square_cont"] = report("S6 square cont. alpha=90", pred, vs,
                                    extra="(chord law w/o quantizer)")
    out["S6_square_cont"]["curve"] = [sp, rm]

    # S7 noise x5
    pred = (L/2)/(TAU + 0.30)
    vs,_,_ = vstar(circle, TAU, pred, sigma_deg=10.0)
    out["S7_noise_x5"] = report("S7 noise sigma=10deg", pred, vs)

    # S8 T_ctrl = 0.5
    pred = (L/2)/(TAU + 0.25 + 0.25)
    vs,_,_ = vstar(circle, TAU, pred, T_ctrl=0.5)
    out["S8_Tctrl_0.5"] = report("S8 T_ctrl=0.5s", pred, vs)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}  total {time.time()-t0:.0f}s", flush=True)
