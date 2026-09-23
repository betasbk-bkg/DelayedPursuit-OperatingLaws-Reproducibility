"""One-parameter regime fits for non-regular trajectories (manuscript Sec. VI-D).
Input: collapse_nonregular_results.json from the the reference engine's repository
(place in the engine directory or set its path below).
Output: data/phase5_nonregular_fits_2026-07-19.json"""
import os, json
import numpy as np

ENGINE = os.environ.get("ENGINE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine"))
SRC = os.path.join(ENGINE, "collapse_nonregular_results.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                   "phase5_nonregular_fits_2026-07-19.json")

d = json.load(open(SRC, encoding="utf-8"))
taus = np.array([100, 200, 300, 433, 600, 1000]) / 1000
C_LAG, L = 0.2333, 2.0

out = {}
for name, rd in d["results"].items():
    vs = np.array(rd["vstar_by_tau"])
    u_g = float(np.mean(vs * (taus + C_LAG) / L))
    pred = u_g * L / (taus + C_LAG)
    r2_lag = 1 - np.sum((vs - pred) ** 2) / np.sum((vs - vs.mean()) ** 2)
    x_g = float(np.mean(vs * np.sqrt(taus)))
    pred_d = x_g / np.sqrt(taus)
    r2_dit = 1 - np.sum((vs - pred_d) ** 2) / np.sum((vs - vs.mean()) ** 2)
    out[name] = dict(u_g=round(u_g, 3), r2_lag=round(float(r2_lag), 4),
                     x_g=round(x_g, 3), r2_dither=round(float(r2_dit), 4))
    print(f"{name:<14} lag: u={u_g:.3f} R2={r2_lag:.4f} | "
          f"dither: x={x_g:.3f} R2={r2_dit:.4f}")
json.dump(out, open(OUT, "w"), indent=2)
print("Saved:", OUT)
