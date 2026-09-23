# engine/ — the crowd-steering engine (shipped)

`engine/crowd_control_engine/` is the reference system of Secs. 3 and 7 of the paper (and Supplement S1):
`simulation_main.py` (150 voting agents, 8-direction commands, look-ahead
pursuit) and the result JSONs the Sec. 7 figures and tables are read from
(`unified_vstar_results.json`, `expa_results.json`, `corner_freq_results.json`,
`theta_autocorr_results.json`, `collapse_nonregular_results.json`).

It is part of this paper's own work and is shipped in full (21 files). Scripts
find it through the environment variable `ENGINE_DIR` if set, otherwise at
this location. Older derivation scripts hard-code an absolute path to it; see
`code/derivation/README.md` for the one-command retarget.
