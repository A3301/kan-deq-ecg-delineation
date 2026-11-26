"""Extract the first 10 s of digital lead II from each complete hash-verified release file."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main():
    published = dict((path.removeprefix("./"), sha) for sha, path in
                     (line.split(maxsplit=1) for line in (SOURCE/"SHA256SUMS.txt").read_text().splitlines() if line))
    ids = (SOURCE/"RECORDS").read_text().splitlines()
    assert set(ids) == set(pd.read_csv(SOURCE/"manual_reference.csv").id)
    rows, clips = [], []
    for rid in ids:
        header = SOURCE / "raw" / (rid + ".hea")
        raw = header.with_suffix(".dat")
        for path in [header, raw]:
            assert digest(path) == published[rid + path.suffix], path
        lines = [s.strip().split() for s in header.read_text(encoding="latin1").splitlines()
                 if s.strip() and not s.lstrip().startswith("#")]
        filename, n_channels, fs, n_samples = lines[0][:4]
        assert filename == raw.stem and int(n_channels) == 15 and int(fs) == 1000
        channels = [line for line in lines[1:] if line[0] == raw.name]
        assert len(channels) == 12 and all(line[1] == "16" for line in channels)
        idx = [line[-1].lower() for line in channels].index("ii")
        channel = channels[idx]
        gain, zero = float(channel[2]), int(channel[4])
        assert raw.stat().st_size == int(n_samples)*2*len(channels)
        digital = np.fromfile(raw, dtype="<i2", count=10000*len(channels)).reshape(10000,len(channels))[:,idx]
        assert digital[0] == int(channel[5]) and gain > 0 and not np.any(digital == -32768), rid
        clips.append(digital.copy())
        rows.append(dict(id=rid,patient=rid.split("/")[0],gain=gain,baseline=zero,source_hz=1000,
                         header_sha256=published[rid+".hea"],signal_sha256=published[rid+".dat"],
                         clip_int16_sha256=hashlib.sha256(digital.tobytes()).hexdigest()))
    assert len(set(r["clip_int16_sha256"] for r in rows)) == len(ids)
    np.savez_compressed(SOURCE/"ptb_lead_ii_10s.npz", id=ids, digital=np.stack(clips))
    pd.DataFrame(rows).to_csv(SOURCE/"cohort.csv",index=False)
    print(f"Extracted {len(ids)} distinct fixed clips")

if __name__ == "__main__":
    main()
