"""Parsing of WFDB delineation annotations into wave triplets (onset, peak, offset).

WFDB delineation convention (LUDB, QTDB .q1c/.q2c/.qt1/.pu*):
    '(' onset   <peak symbol>   ')' offset
peak symbols: 'p' (P wave), 'N' (QRS; QTDB may also use other beat labels), 't' (T wave), 'u' (U wave).
An onset '(' or offset ')' belongs to the peak symbol immediately preceding/following it.
Missing boundaries are returned as None.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import wfdb

WAVE_OF = {"p": "P", "t": "T", "u": "U"}  # every other non-bracket symbol is treated as QRS


@dataclass
class Wave:
    kind: str           # 'P', 'QRS', 'T', 'U'
    on: int | None
    peak: int | None
    off: int | None
    symbol: str


def symbol_to_kind(sym: str) -> str:
    return WAVE_OF.get(sym, "QRS")


def parse_waves(samples: np.ndarray, symbols: list[str]) -> list[Wave]:
    waves: list[Wave] = []
    n = len(symbols)
    for i, s in enumerate(symbols):
        if s in ("(", ")"):
            continue
        on = int(samples[i - 1]) if i > 0 and symbols[i - 1] == "(" else None
        off = int(samples[i + 1]) if i + 1 < n and symbols[i + 1] == ")" else None
        waves.append(Wave(symbol_to_kind(s), on, int(samples[i]), off, s))
    return waves


def read_waves(record_path: str, ext: str) -> tuple[list[Wave], float]:
    a = wfdb.rdann(record_path, ext)
    return parse_waves(a.sample, list(a.symbol)), float(a.fs) if a.fs else float("nan")


def orphan_brackets(symbols: list[str]) -> int:
    """Count '(' / ')' that are not adjacent to a wave symbol (consistency check)."""
    bad = 0
    n = len(symbols)
    for i, s in enumerate(symbols):
        if s == "(" and (i + 1 >= n or symbols[i + 1] in ("(", ")")):
            bad += 1
        if s == ")" and (i == 0 or symbols[i - 1] in ("(", ")")):
            bad += 1
    return bad
