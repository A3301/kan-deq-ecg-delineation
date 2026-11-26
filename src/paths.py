"""Central dataset paths (data/raw/<ds>/extracted/<physionet folder>)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

LUDB = RAW / "ludb_1.0.1" / "extracted" / "lobachevsky-university-electrocardiography-database-1.0.1"
LUDB_DATA = LUDB / "data"
QTDB = RAW / "qtdb_1.0.0" / "extracted" / "qt-database-1.0.0"
BUTPDB = (RAW / "but-pdb_1.0.0" / "extracted"
          / "brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0")
PWAVE = RAW / "pwave_1.0.0" / "extracted" / "mit-bih-arrhythmia-database-p-wave-annotations-1.0.0"

LUDB_LEADS = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]


def records(db_dir: Path) -> list[str]:
    return [r.strip() for r in (db_dir / "RECORDS").read_text().splitlines() if r.strip()]
