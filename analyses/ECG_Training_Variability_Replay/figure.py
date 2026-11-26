"""Render all three observed fits per architecture and tolerance."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent

def draw(tag):
    result=HERE/"results"
    with (result/f"{tag}_scores.csv").open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    names=[r["label"] for r in rows[:8]]
    colors=["#15527A","#BB591A","#14705B"]
    symbols=["o","s","^"]
    fig,axes=plt.subplots(1,2,figsize=(7.7,3.55),sharey=True)
    for ax,tol in zip(axes,[150,40]):
        panel=[row for row in rows if int(row["tolerance_ms"])==tol]
        for j,r in enumerate(panel):
            y=7-j
            scores=[float(r[f"seed_{s}"]) for s in (1,2,3)]
            ax.hlines(y,min(scores),max(scores),color="#68727A",lw=1)
            for k,v in enumerate(scores):
                ax.scatter(v,y+(.115,0,-.115)[k],marker=symbols[k],color=colors[k],s=22,zorder=3,
                           linewidths=.25,edgecolors="white")
        ax.set_title(f"±{tol} ms",fontsize=11,weight="bold",loc="left",pad=8)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel("Mean boundary F1 (%)",fontsize=9)
        ax.set_ylim(-.48,7.5)
        ax.grid(axis="x",color="#DFE5E7",lw=.65)
        ax.tick_params(length=0,labelsize=8,pad=4)
        for s in ("left","right","top"):ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#A6B0B6")
        ax.margins(x=.07)
    axes[0].set_yticks(range(8),names[::-1],fontsize=8.5)
    handles=[Line2D([0],[0],marker=m,color=c,markeredgecolor="none",linestyle="",label=f"Seed {s+1}",markersize=5.5)
             for s,(m,c) in enumerate(zip(symbols,colors))]
    fig.legend(handles=handles,ncol=3,loc="upper center",bbox_to_anchor=(.61,.997),
               frameon=False,handletextpad=.35,columnspacing=1.6,fontsize=8)
    fig.subplots_adjust(left=.22,right=.985,bottom=.16,top=.87,wspace=.16)
    for ext in ("png","pdf"):
        fig.savefig(result/f"{tag}_training_variation.{ext}",dpi=320,bbox_inches="tight",facecolor="white")
    plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("tag",choices=["fold0","ensemble","homogeneous","harmonized_ensemble"]);args=ap.parse_args();draw(args.tag)
