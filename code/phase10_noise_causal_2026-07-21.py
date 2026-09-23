# -*- coding: utf-8 -*-
"""CAUSAL test of the noise hypothesis (per reviewer critique 2026-07-21).
Same system, ONLY the injected direction-noise sigma varies:
sigma/sigma0 in {0, 0.25, 0.5, 1, 2}, sigma0 = 8.60 deg (measured aggregate).
Everything else identical: same corner geometry, same window, same speed grid,
same optimum extraction, same run length, continuous (unquantized) commands so
quantization cannot confound.
Prediction under the noise hypothesis: u* decreases monotonically with sigma,
from the deterministic value toward ~0.50 at sigma0.
"""
import os, json, time
import numpy as np
DT=1/60; W=0.3; DELAY_F=26; SMOOTH=0.2; L=2.0
tau_tot=DELAY_F*DT+W/2+DT/SMOOTH; VI=int(W/DT)
SIG0=np.radians(8.60)

def path(a,seg):
    ar=np.radians(a)
    return (np.array([-seg,0.]),np.array([0.,0.]),np.array([seg*np.cos(ar),seg*np.sin(ar)]))
def aim(dp,p0,pc,p1,Lv):
    def sp(a,b,pt):
        v=b-a;t=np.clip((pt-a)@v/(v@v),0,1);return np.linalg.norm(pt-(a+t*v)),t,np.linalg.norm(v)
    d0,t0,l0=sp(p0,pc,dp); d1,t1,l1=sp(pc,p1,dp)
    s=t0*l0 if d0<=d1 else l0+t1*l1; sa=s+Lv
    if sa<=l0: return p0+(sa/l0)*(pc-p0)
    r=min(sa-l0,l1); return pc+(r/l1)*(p1-pc)
def dpath(pos,p0,pc,p1):
    def d(a,b,pt):
        v=b-a;t=np.clip((pt-a)@v/(v@v),0,1);return np.linalg.norm(pt-(a+t*v))
    return min(d(p0,pc,pos),d(pc,p1,pos))

def J(a_deg,v,sigma,seed,seg=60.,win=40.):
    rng=np.random.default_rng(seed)
    p0,pc,p1=path(a_deg,seg); pos=p0.copy(); vel=np.array([v,0.]); cur=np.array([1.,0.])
    hist=[pos.copy()]; errs=[]
    for f in range(int(1.9*seg/(v*DT))):
        if f%VI==0:
            dpp=hist[max(0,len(hist)-1-DELAY_F)]
            d=aim(dpp,p0,pc,p1,L)-dpp; n=np.linalg.norm(d)
            if n>1e-9:
                th=np.arctan2(d[1],d[0])
                if sigma>0: th+=rng.normal(0,sigma)
                cur=np.array([np.cos(th),np.sin(th)])
        vel+=SMOOTH*(cur*v-vel); pos=pos+vel*DT; hist.append(pos.copy())
        if np.linalg.norm(pos-pc)<win: errs.append(dpath(pos,p0,pc,p1))
    return np.sqrt(np.mean(np.array(errs)**2))

def ustar(a_deg,sigma,mc):
    ca=np.cos(np.radians(a_deg/2))
    us=np.arange(0.38,0.80,0.02)
    js=[]
    for u in us:
        v=u*ca*L/tau_tot
        js.append(np.mean([J(a_deg,v,sigma,1000+7*k) for k in range(mc)]))
    js=np.array(js); i=int(np.argmin(js)); u=us[i]
    if 1<=i<=len(us)-2:
        c=np.polyfit(us[i-1:i+2],js[i-1:i+2],2)
        if c[0]>0 and us[i-1]<=-c[1]/(2*c[0])<=us[i+1]: u=-c[1]/(2*c[0])
    return float(u)

if __name__=="__main__":
    t0=time.time(); out={}
    ratios=[0.0,0.25,0.5,1.0,2.0]
    print("CAUSAL noise test: u*/cos(a/2) vs sigma (only sigma varies)",flush=True)
    print(f"{'sig/sig0':>9} {'a=60':>8} {'a=90':>8} {'a=120':>8}",flush=True)
    for r in ratios:
        mc = 1 if r==0 else (12 if r<=0.5 else 20)
        row={}
        for a in [60,90,120]:
            row[a]=ustar(a,r*SIG0,mc)
        out[str(r)]=row
        print(f"{r:>9.2f} {row[60]:>8.3f} {row[90]:>8.3f} {row[120]:>8.3f}  [{time.time()-t0:.0f}s]",flush=True)
    json.dump(out,open("phase10_noise_causal_results_2026-07-21.json","w"),indent=2)
    print("saved",flush=True)
