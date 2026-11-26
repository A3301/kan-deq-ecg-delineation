"""Refit archive-trained members of fold zero and measure seeds under a shared implementation."""
from pathlib import Path
import shutil
import train_panel as panel
from prepare import batches

ROOT=Path(__file__).resolve().parent
orig_path=panel.output_path
archived={Path(x.replace("\\","/")).name for x in (ROOT/"inventory.txt").read_text().splitlines() if x.strip()}

def weight_path(model,fold,seed):
    assert fold==0
    p=orig_path(model,fold,seed)
    return ROOT/"homogeneous_weights"/p.name if p.name in archived else p

def main():
    batches(0,1)
    (ROOT/"homogeneous_weights").mkdir(exist_ok=True)
    panel.LOGS=ROOT/"homogeneous_logs"; panel.output_path=weight_path
    x,rows=panel.metadata()
    for seed in (1,2,3):
        for m in panel.FAMILIES:
            p=weight_path(m,0,seed)
            if p.parent.name=="weights" and not p.exists():
                raise RuntimeError("Run primary fold 0 before the homogeneous panel")
            panel.train(m,0,seed,x,rows)
    ix=[i for i,r in enumerate(rows) if r["subject"] in panel.SPLITS["test"]]
    for m in panel.FAMILIES:
        for seed in (1,2,3):panel.score(m,seed,[0],x[ix],[rows[i] for i in ix],"homogeneous")

if __name__=="__main__":main()
