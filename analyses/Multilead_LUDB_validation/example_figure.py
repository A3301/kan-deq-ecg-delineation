"""Prespecified illustrative case: record 1, complete reference beat nearest 5 s."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
if hasattr(pd.options.future,"infer_string"):pd.options.future.infer_string=False
import wfdb
from score_events import LEADS, fuse

w=Path(__file__).resolve().parent;out=w/"outputs"
case=json.loads((out/"reference.json").read_text())[0]
raw,h=wfdb.rdsamp(str(w/"raw"/"1")); leadnames=[lead.lower() for lead in h["sig_name"]];raw=raw[:,[leadnames.index(x) for x in LEADS]]
rg=fuse([case["leads"][k]["qrs"] for k in LEADS]);g=min([z for z in rg if len(z["leads"])==12],key=lambda z:abs(np.median(z["centers"])-5000))
on,off=min(g["on"]),max(g["off"]);origin=(on+off)/2
fig,(a,b)=plt.subplots(1,2,figsize=(8.0,4.5),gridspec_kw={"width_ratios":[1.25,1]},sharey=True,layout="constrained")
for j,l in enumerate(LEADS):
    idx=g["leads"].index(j);lo,hi=g["on"][idx],g["off"][idx];y=11-j
    time=np.arange(raw.shape[0])*2;use=(time>=origin-185)&(time<=origin+185)
    xx=raw[use,j];xx=xx-np.median(xx);xx=xx/(np.ptp(xx)+1e-12)*.47
    a.plot(time[use]-origin,xx+y,lw=.85,color="#607488")
    wave=(time[use]>=lo)&(time[use]<=hi);a.plot(time[use][wave]-origin,xx[wave]+y,lw=1.35,color="#005D64")
    b.plot([lo-origin,hi-origin],[y,y],color="#005D64",lw=2.0)
    b.scatter([lo-origin,hi-origin],[y,y],s=13,c=["#034E55","#034E55"],zorder=3)
for ax in (a,b):
    ax.set_yticks(np.arange(12),[{"avr":"aVR","avl":"aVL","avf":"aVF"}.get(x,x.upper()) for x in reversed(LEADS)])
    ax.axvline(on-origin,ls="--",lw=.8,color="#A84C26");ax.axvline(off-origin,ls="--",lw=.8,color="#A84C26")
    ax.spines[["top","right","left"]].set_visible(False);ax.grid(axis="y",color="#dce5e5",lw=.45);ax.tick_params(axis="y",length=0)
    ax.set_xlabel("Time relative to interval centre (ms)")
a.set_title("A   Simultaneous lead waveforms",loc="left",fontsize=11,fontweight="bold")
b.set_title("B   Lead-specific reference intervals",loc="left",fontsize=11,fontweight="bold")
a.set_xlim(-185,185);b.set_xlim(on-origin-10,off-origin+10);b.tick_params(labelleft=True)
fig.savefig(out/"QRS_multilead_example.png",dpi=330,bbox_inches="tight",facecolor="white")
(out/"figure_caption.txt").write_text(f"Derivation of a multilead QRS interval from cardiologist-marked lead-specific boundaries. Example selected by record ID (LUDB 1) and temporal proximity to the middle of the recording among all-12-lead reference complexes. A, simultaneous lead waveforms scaled independently for display; teal marks the annotated QRS region. B, native individual-lead onset and offset times. Orange dashed lines delimit the earliest onset at {on} ms and latest offset at {off} ms (elapsed width {off-on} ms). Time zero is the midpoint of the resulting multilead interval. This panel illustrates the measurement rule and contains no model predictions. Source: Kalyakulina et al., Lobachevsky University Electrocardiography Database v1.0.1, PhysioNet, doi:10.13026/eegm-h675.",encoding="utf8")
