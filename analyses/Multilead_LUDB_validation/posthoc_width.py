"""Descriptive post-hoc global QRS width strata in records with documented adult age."""
from pathlib import Path
import re

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main(work):
    out = work / "outputs"
    ages = {}
    for file in (work / "raw").glob("*.hea"):
        age = re.search(r"^#<age>:\s*([0-9]+)", file.read_text(), flags=re.M)
        ages[int(file.stem)] = int(age.group(1)) if age else None
    assert len(ages) == 200
    adults = {k for k,v in ages.items() if v is not None and v >= 18}
    ev = pd.read_csv(out/"paired_events.csv.gz")
    ev = ev[ev.scope.eq("global12") & ev.reference_index.notna() & ev.record.isin(adults)].copy()
    ev["ref_width"] = ev.ref_off_ms - ev.ref_on_ms
    ev["pred_width"] = ev.pred_off_ms - ev.pred_on_ms
    ev["error"] = ev.pred_width - ev.ref_width
    ev["group"] = np.where(ev.ref_width>=120, "at_least_120_ms", "under_120_ms")
    rows=[]
    for (group,arm,seed),g in ev.groupby(["group","arm","seed"]):
        rows.append(dict(stratum=group,arm=arm,seed=seed,adult_records=len(adults),all_eligible_adult_records=ev.record.nunique(),
                         contributors=g.record.nunique(),ref_beats=len(g),matched=g.error.notna().sum(),bias=g.error.mean(),
                         mae=g.error.abs().mean(),allref_20ms_pct=100*g.error.abs().le(20).mean(),
                         detected_and_at_least_120_pct=100*g.pred_width.ge(120).mean()))
    pd.DataFrame(rows).to_csv(out/"posthoc_width_strata.csv",index=False)


if __name__=="__main__":
    main(HERE)
