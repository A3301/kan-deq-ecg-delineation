"""P-peak matching on either the archived grid or exact native reference times."""
import numpy as np

def counts(raw_ref,peaks,native_fs,n250,tolerance,coordinate):
    raw_ref=np.asarray(raw_ref,dtype=np.int64)
    peaks=np.asarray(peaks,dtype=np.int64)
    if coordinate=="historical_grid":
        ref=np.rint(raw_ref*250./native_fs).astype(int)
        det=peaks
        radius=int(round(tolerance*.25))
        start=max(ref[0]-125,0) if len(ref) else 0
        end=min(ref[-1]+125,n250) if len(ref) else n250
    elif coordinate=="native_ms":
        # Integer products implement exact milliseconds without round-off near a cutoff.
        ref=raw_ref*1000
        det=peaks*4*native_fs
        radius=tolerance*native_fs
        start=max(ref[0]-500*native_fs,0) if len(ref) else 0
        end=min(ref[-1]+500*native_fs,n250*4*native_fs) if len(ref) else n250*4*native_fs
    else:
        raise ValueError(coordinate)
    assert len(set(det))==len(det)
    distance=np.abs(ref[:,None]-det[None,:])
    rr,dd=np.where(distance<=radius)
    chosen_r,chosen_d=set(),set()
    for k in np.argsort(distance[rr,dd],kind="stable"):
        i,j=rr[k],dd[k]
        if i not in chosen_r and j not in chosen_d:
            chosen_r.add(i)
            chosen_d.add(j)
    tp=len(chosen_r)
    fp=sum(j not in chosen_d and start<=v<end for j,v in enumerate(det))
    return tp,len(ref)-tp,int(fp)
