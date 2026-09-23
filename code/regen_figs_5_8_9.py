# -*- coding: utf-8 -*-
"""
Regenerate Figs. 5, 8 and 9 with the text that is drawn INTO the images corrected.

WHAT WAS WRONG.  The three figures were drawn for an earlier manuscript and carry
its numbering and its framing in their own pixels, where no text gate looks:

  Fig. 8(a)  "analytic edge m_s, Eq. (11.3)"      -> the edge law is Eq. (12)
  Fig. 9(a)  "closed form, Eq. (12.1)"            -> the closed form is Eq. (13)
             "published optima"                   -> the band is this paper's
                                                     measured circle optima
             "the constant that was not a constant" (title) -> a description
  Fig. 9(b)  "Eq. (11.3)", "(Sec. 11.4)"          -> Eq. (12); no such section
             the dashed curve is (12) shifted by its mean offset from this flow;
             it is labelled as exactly that
  Fig. 5     "published range"                    -> "swept range"
             "lag cap u_max L sqrt(tau)/tau_tot"   -> the cap is drawn at u = 0.75,
                                                     and the legend now says so

Found by opening the image files after an independent reviewer flagged the
captions; no manuscript check reads figure pixels.

HOW.  Figs. 8 and 9: d22 is executed from its own file with the label strings
substituted in memory -- the derivation file itself is not edited, and every
substitution must hit exactly once or the script stops.  Fig. 5: its block in
make_figures has no data inputs (the observed points are literals), so the block
is executed alone; running the whole generator would overwrite the corrected
Fig. 6, which regen_fig6 produced.

Usage:  python regen_figs_5_8_9_2026-09-13.py
"""
import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.abspath(os.path.join(HERE, "..", "derivation"))
D22 = os.path.join(DERIV, "d22_figures_2026-08-29.py")
GEN = os.path.join(HERE, "repo", "code", "make_figures_2026-07-19.py")
FIG = os.path.join(HERE, "figures")


def sub_once(src, old, new, what):
    n = src.count(old)
    if n != 1:
        raise SystemExit("%s: expected exactly one %r, found %d" % (what, old, n))
    return src.replace(old, new)


# ------------------------------------------------------------ Figs. 8 and 9
s = io.open(D22, encoding="utf-8").read()
for old, new in [
    ('label=r"analytic edge $m_s$, Eq. (11.3)")', 'label=r"analytic edge $m_s$, Eq. (12)")'),
    ('label="closed form, Eq. (12.1)")', 'label="closed form, Eq. (13)")'),
    ('ax[0].text(0.704, ax[0].get_ylim()[0]+0.12, "published optima"',
     'ax[0].text(0.704, 10.9, "measured\\ncircle optima"'),
    ('ax[1].legend(frameon=False, fontsize=6.5, loc="lower right")',
     'ax[1].legend(frameon=False, fontsize=6.5, loc="upper left")'),
    ('"(a)  the constant that was not a constant"', r'"(a)  $\\sigma_\\theta$ versus delay margin"'),
    ('label=r"Eq. (11.3),  $c=k\\Delta/2+(1-u)$")', 'label=r"Eq. (12),  $c=k\\Delta/2+(1-u)$")'),
    ('label=r"Eq. (11.3) $+$ offset %.1f$^\\circ$ (Sec. 11.4)" % off)',
     'label=r"Eq. (12) $+$ mean offset from this flow, %.1f$^\\circ$" % off)'),
    ('r"(b)  edge law, valid for $1<k\\Delta/2<2.55$"',
     'r"(b)  edge law, predicted validity $1<k\\Delta/2<2.55$"'),
]:
    s = sub_once(s, old, new, "d22")
left = [x for x in ("11.3", "12.1", "Sec. 11", "published") if x in s.split('"""', 2)[2]]
if left:
    raise SystemExit("d22 labels still carry: %s" % left)
g = {"__file__": D22, "__name__": "__main__"}
exec(compile(s, D22, "exec"), g)
print("  Figs. 8 and 9 regenerated from d22 with corrected labels")

# ------------------------------------------------------------------ Fig. 5
gen = io.open(GEN, encoding="utf-8").read()
m = re.search(r"# -+ Fig 5:.*?(?=# -+ Fig 6:)", gen, re.S)
if not m:
    raise SystemExit("Fig. 5 block not found in make_figures")
blk = m.group(0)
blk = sub_once(blk, 'ax.annotate("published range"', 'ax.annotate("swept range"', "fig5")
blk = sub_once(blk, r'label=r"lag cap $u_{max}L\sqrt{\tau}/\tau_{tot}$")',
               r'label=r"lag cap at $u=0.75$: $0.75\,L\sqrt{\tau}/\tau_{tot}$")', "fig5")
if "0.75*L0" not in blk:
    raise SystemExit("Fig. 5: the cap is no longer drawn at 0.75; the legend would be false")
import numpy as np                                                  # noqa: E402
import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
env = {"np": np, "plt": plt, "os": os, "OUT": FIG, "C_LAG": 0.2333, "L0": 2.0,
       "BLUE": "#0072B2", "ORANGE": "#E69F00", "GREEN": "#009E73",
       "VERM": "#D55E00", "GRAY": "#666666"}
exec(compile(blk, GEN + " [Fig 5 block]", "exec"), env)
print("  Fig. 5 regenerated with corrected labels")
