# -*- coding: utf-8 -*-
"""Corner theorem verification (finite-angle, noise-free): does u*/cos(a/2) -> 0.5
as accumulation window D grows? Confirms the permanent-offset-cancellation
derivation for finite corner angles. Light grid for speed."""
import os, numpy as np
DT=1/60; W=0.3; DELAY_F=26; SMOOTH=0.2; L=2.0
tau_tot=DELAY_F*DT + W/2 + DT/SMOOTH
VI=int(W/DT)

def path(a_deg, seglen):
    a=np.radians(a_deg)
    return (np.array([-seglen,0.]), np.array([0.,0.]),
            np.array([seglen*np.cos(a), seglen*np.sin(a)]))

def aim(dp,p0,pc,p1,Lv):
    def sp(a,b,pt):
        v=b-a; t=np.clip((pt-a)@v/(v@v),0,1); return np.linalg.norm(pt-(a+t*v)),t,np.linalg.norm(v)
    d0,t0,l0=sp(p0,pc,dp); d1,t1,l1=sp(pc,p1,dp)
    s=t0*l0 if d0<=d1 else l0+t1*l1
    sa=s+Lv
    if sa<=l0: return p0+(sa/l0)*(pc-p0)
    r=min(sa-l0,l1); return pc+(r/l1)*(p1-pc)

def dpath(pos,p0,pc,p1):
    def d(a,b,pt):
        v=b-a; t=np.clip((pt-a)@v/(v@v),0,1); return np.linalg.norm(pt-(a+t*v))
    return min(d(p0,pc,pos),d(pc,p1,pos))

def run(a_deg,v,seglen,window):
    p0,pc,p1=path(a_deg,seglen)
    pos=p0.copy(); vel=np.array([v,0.]); cur=np.array([1.,0.]); hist=[pos.copy()]; errs=[]
    nf=int((1.9*seglen)/(v*DT))
    for f in range(nf):
        if f%VI==0:
            dp=hist[max(0,len(hist)-1-DELAY_F)]
            dd=aim(dp,p0,pc,p1,L)-dp; n=np.linalg.norm(dd)
            if n>1e-9: cur=dd/n
        vel+=SMOOTH*(cur*v-vel); pos=pos+vel*DT; hist.append(pos.copy())
        if np.linalg.norm(pos-pc)<window: errs.append(dpath(pos,p0,pc,p1))
    return np.sqrt(np.mean(np.array(errs)**2)) if len(errs)>5 else np.nan

def ustar(a_deg,window):
    vs=np.linspace(0.4,2.6,23); rms=[run(a_deg,v,max(2*window,50),window) for v in vs]
    i=int(np.nanargmin(rms)); vstar=vs[i]
    if 1<=i<=len(vs)-2:
        c=np.polyfit(vs[i-1:i+2],rms[i-1:i+2],2)
        if c[0]>0 and vs[i-1]<=-c[1]/(2*c[0])<=vs[i+1]: vstar=-c[1]/(2*c[0])
    return vstar*tau_tot/L

if __name__=="__main__":
    import time,json; t0=time.time()
    out={}
    print("u*/cos(a/2) vs window D (finite-angle noise-free):", flush=True)
    print(f"{'D':>5} {'a=60':>8} {'a=90':>8} {'a=120':>8}", flush=True)
    for D in [10,20,40,80]:
        row={}
        for a in [60,90,120]:
            u=ustar(a,D); row[a]=u/np.cos(np.radians(a/2))
        out[D]=row
        print(f"{D:>5} {row[60]:>8.3f} {row[90]:>8.3f} {row[120]:>8.3f}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump({str(k):v for k,v in out.items()}, open("phase9_corner_theorem_results_2026-07-21.json","w"), indent=2)
    print("theorem: -> 0.500 as D grows", flush=True)
