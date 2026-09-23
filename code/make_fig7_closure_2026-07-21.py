# -*- coding: utf-8 -*-
"""Fig. 7: computational closure of the reduced model.
Predicted vs measured for the two smooth-geometry constants, comparing the
correct reduction (measured vote transfer function) against the incorrect one
(noisy angle then grid quantization). Zero fitted constants in either case."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(HERE,"..","data")
OUT=os.path.join(HERE,"..","..","figures"); os.makedirs(OUT,exist_ok=True)
BLUE,ORANGE,GREEN,VERM,GRAY="#0072B2","#E69F00","#009E73","#D55E00","#666666"
plt.rcParams.update({"font.size":9,"axes.linewidth":0.8,"figure.dpi":300,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "font.family":"DejaVu Sans"})

wrong=json.load(open(os.path.join(DATA,"phase12_unified_closure_results_2026-07-21.json")))
r1=json.load(open(os.path.join(DATA,"phase14_correct_reduction_results_2026-07-21.json")))
r2=json.load(open(os.path.join(DATA,"phase15_verify_closure_results_2026-07-21.json")))

targets={"dither invariant\n$x^*$":1.347,"smooth margin\n$u^*$ (ellipse)":0.562}
correct={"dither invariant\n$x^*$":[r1["circle_xstar"],r2["circle_xstar"]["pred"]],
         "smooth margin\n$u^*$ (ellipse)":[r1["ellipse"],r2["ellipse_u"]["pred"]]}
incorrect={"dither invariant\n$x^*$":[wrong["circle_xstar"]],
           "smooth margin\n$u^*$ (ellipse)":[wrong["ellipse_u"]]}

fig,axs=plt.subplots(1,2,figsize=(6.6,2.6))
for ax,(name,tgt) in zip(axs,targets.items()):
    ax.axhline(tgt,color=GRAY,lw=1.2,ls="--",label="measured (full 150-agent system)")
    ax.axhspan(tgt*0.95,tgt*1.05,color=GRAY,alpha=0.12,lw=0)
    xs=[1,2]; ys=correct[name]
    ax.plot(xs,ys,"o",color=BLUE,ms=7,mfc="white",mew=1.6,
            label="reduced model, measured transfer fn.")
    ax.plot([0],incorrect[name],"s",color=VERM,ms=7,
            label="reduced model, naive quantization")
    ax.set_xlim(-0.6,2.6); ax.set_xticks([0,1,2])
    ax.set_xticklabels(["naive","run 1","run 2"],fontsize=8)
    ax.set_title(name,fontsize=8.5)
    lo=min(min(ys),incorrect[name][0],tgt); hi=max(max(ys),tgt)
    ax.set_ylim(lo*0.88,hi*1.12)
axs[0].set_ylabel("predicted value")
axs[0].annotate("±5%",xy=(2.05,1.347*1.05),fontsize=7,color=GRAY)
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=3,frameon=False,fontsize=7,bbox_to_anchor=(0.5,-0.13))
fig.suptitle("Computational closure: constants predicted with zero fitted parameters",
             fontsize=9,y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT,"fig7_closure.png"),bbox_inches="tight")
print("wrote fig7_closure.png")
