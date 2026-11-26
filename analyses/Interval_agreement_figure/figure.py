"""Descriptive participant-level interval agreement figure for frozen RDB-trained models."""
from pathlib import Path
from hashlib import sha256
import argparse
import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
SEEDS=[1,2,3]
HEADS=["KAN","GELU"]
PALETTE={"KAN":"#006d72","GELU":"#714b85"}


def read_qrs(parent):
    raw=pd.read_csv(parent/"outputs"/"paired_events.csv.gz")
    references=raw[raw.scope.eq("global12") & raw.reference_index.notna()].copy()
    assert len(references)==1786*6 and references.record.nunique()==197
    pairs=references[references.pred_index.notna()].copy()
    keys=["record","reference_index"]
    assert not pairs.duplicated(keys+["arm","seed"]).any()
    assert (pairs.groupby(keys)[["ref_on_ms","ref_off_ms"]].nunique()==1).all().all()
    pairs=pairs[pairs.groupby(keys).seed.transform("size")==6].copy()
    pairs["person"]=pairs.record.astype(str);pairs["event"]=pairs.reference_index.astype(int).astype(str)
    pairs["prediction"]=pairs.pred_off_ms-pairs.pred_on_ms
    pairs["reference"]=pairs.ref_off_ms-pairs.ref_on_ms
    pairs=pairs.rename(columns={"arm":"head"})
    return pairs


def read_qt(parent):
    df=pd.read_csv(parent/"results"/"per_record.csv")
    assert df.id.nunique()==548 and df.patient.nunique()==289
    assert not df.duplicated(["id","head","seed"]).any()
    assert (df.groupby("id")[["reference_q_ms","reference_t_ms","patient"]].nunique()==1).all().all()
    pairs=df[df.status.eq("returned")].copy()
    pairs=pairs[pairs.groupby("id").seed.transform("size")==6].copy()
    common=pd.read_csv(parent/"results"/"common_set.csv")
    assert set(pairs.id.unique())==set(common.loc[common.all_six_returned,"id"])
    pairs["prediction"]=pairs.predicted_t_ms-pairs.predicted_q_ms
    pairs["reference"]=pairs.reference_t_ms-pairs.reference_q_ms
    return pairs.rename(columns={"patient":"person","id":"event"})


def aggregate(unit,df):
    assert set(df.seed.unique())==set(SEEDS) and set(df["head"].unique())==set(HEADS)
    keys=["person","event","head"]
    assert (df.groupby(keys).seed.nunique()==3).all()
    # Average the three fixed predictions for each common event before calculating
    # its signed error and arithmetic mean relative to the invariant reference.
    event=df.groupby(keys).agg(predicted_ms=("prediction","mean"),reference_ms=("reference","first")).reset_index()
    event["pair_mean_ms"]=(event.predicted_ms+event.reference_ms)/2
    event["error_ms"]=event.predicted_ms-event.reference_ms
    subject=event.groupby(["person","head"]).agg(common_events=("event","nunique"),mean_reference_ms=("reference_ms","mean"),
              mean_prediction_ms=("predicted_ms","mean"),pair_mean_ms=("pair_mean_ms","mean"),error_ms=("error_ms","mean")).reset_index()
    subject.insert(0,"interval",unit)
    assert subject.groupby("person").common_events.nunique().eq(1).all()
    return subject


def verified_points(qrs,qt,out):
    g=aggregate("QRS",read_qrs(qrs));t=aggregate("QT",read_qt(qt));points=pd.concat([g,t],ignore_index=True)
    points.to_csv(out/"participant_points.csv",index=False)
    for df,n,e in [(g,196,1742),(t,289,547)]:
        assert all(df.groupby("head").person.nunique()==n)
        assert all(df.groupby("head").common_events.sum()==e)
    summary=points.groupby(["interval","head"]).agg(participants=("person","size"),common_events=("common_events","sum"),
            min_events_per_person=("common_events","min"),max_events_per_person=("common_events","max"),
            equal_person_mean_error_ms=("error_ms","mean"),min_person_error_ms=("error_ms","min"),
            max_person_error_ms=("error_ms","max"),min_pairmean_ms=("pair_mean_ms","min"),max_pairmean_ms=("pair_mean_ms","max")).reset_index()
    summary.to_csv(out/"panel_summary.csv",index=False)
    return points,summary


def draw(points,summary,out):
    mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.right":False,"axes.spines.top":False,
                         "axes.labelcolor":"#202c31","axes.edgecolor":"#a1aaad","axes.linewidth":.65,
                         "pdf.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})
    fig,axes=plt.subplots(2,2,figsize=(7.25,5.66),sharex="row",sharey="row")
    fig.subplots_adjust(left=.101,right=.985,top=.925,bottom=.145,hspace=.61,wspace=.12)
    frame={"QRS":dict(x=(85,215),y=(-63,55),xtick=25,ytick=25,name="LUDB global QRS",sample="196 participants, 1,742 beats"),
           "QT":dict(x=(255,565),y=(-104,96),xtick=50,ytick=40,name="Berlin QT",sample="289 participants, 547 records")}
    for row,unit in enumerate(("QRS","QT")):
        f=frame[unit]
        for col,head in enumerate(HEADS):
            ax=axes[row,col];p=points[(points.interval==unit)&(points["head"]==head)]
            mean=summary[(summary.interval==unit)&(summary["head"]==head)].iloc[0].equal_person_mean_error_ms
            x=p.pair_mean_ms.values;y=p.error_ms.values
            assert min(x)>f["x"][0] and max(x)<f["x"][1] and min(y)>f["y"][0] and max(y)<f["y"][1]
            ax.grid(axis="y",lw=.5,color="#e0e7e6");ax.set_axisbelow(True)
            ax.axhline(0,color="#657277",ls=(0,(3,3)),lw=.85,zorder=1)
            ax.scatter(x,y,s=13,c=PALETTE[head],alpha=.4,linewidths=0,zorder=2,clip_on=True)
            ax.axhline(mean,color=PALETTE[head],lw=1.25,zorder=3)
            ax.set_xlim(*f["x"]);ax.set_ylim(*f["y"])
            ax.xaxis.set_major_locator(MultipleLocator(f["xtick"]));ax.yaxis.set_major_locator(MultipleLocator(f["ytick"]))
            ax.tick_params(axis="both",length=3,width=.55,labelsize=8,color="#93a0a1",pad=3)
            ax.text(.025,.945,"Mean "+f"{mean:.1f}".replace("-","−")+" ms",transform=ax.transAxes,
                    va="top",ha="left",weight="semibold",color=PALETTE[head],size=8.1,
                    bbox={"boxstyle":"round,pad=.18","facecolor":"white","edgecolor":"none","alpha":.88})
            ax.set_title(chr(65+2*row+col)+"   "+f["name"]+" · "+head,loc="left",size=10.2,pad=6,weight="semibold",color="#203a42")
            if col==0:ax.set_ylabel("Mean signed error (ms)",fontsize=8.3)
    fig.supxlabel("Participant mean of model and reference duration (ms)",size=9,y=.073)
    handles=[Line2D([0],[0],marker="o",linestyle="none",markersize=4,color="#49666b",alpha=.65,label="One participant"),
            Line2D([0,1],[0,0],lw=1.3,color="#49666b",label="Mean across participants"),
            Line2D([0,1],[0,0],lw=.85,ls=(0,(3,3)),color="#657277",label="Zero error")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.004),ncol=3,frameon=False,fontsize=8,
               columnspacing=1.8,handlelength=2.3)
    for ext in ["png","pdf","svg"]:
        fig.savefig(out/f"Participant_interval_agreement.{ext}",dpi=400,bbox_inches="tight",pad_inches=.08)
    plt.close(fig)


if __name__=="__main__":
    args=argparse.ArgumentParser();args.add_argument("--qrs",type=Path);args.add_argument("--qt",type=Path)
    args.add_argument("--from-points",action="store_true");args.add_argument("--out",type=Path,default=HERE);a=args.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    if a.from_points:
        points=pd.read_csv(a.out/"participant_points.csv",dtype={"person":str});summary=pd.read_csv(a.out/"panel_summary.csv")
    else:
        points,summary=verified_points(a.qrs,a.qt,a.out)
        sources=[a.qrs/"outputs"/"paired_events.csv.gz",a.qt/"results"/"per_record.csv",a.qt/"results"/"common_set.csv"]
        (a.out/"input_hashes.json").write_text(json.dumps({p.name:sha256(p.read_bytes()).hexdigest() for p in sources},indent=2),encoding="utf8")
    draw(points,summary,a.out)
