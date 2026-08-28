"""Single source of truth for paths, constants, and logging.

Repository copy: the path table below lists the phase folders included in this
replication repository. The full research project defines additional phases; none of
them is needed to reproduce the reported results.

Every analysis script imports from here. No script hard-codes an absolute path.
PROJECT_ROOT is resolved from this file's own location, so the tree can be moved.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

P = {
    "mgmt":         PROJECT_ROOT / "01_Project_Management",
    "screening":    PROJECT_ROOT / "06_Screening",
    "external":     PROJECT_ROOT / "10_External_Data",
    "integration":  PROJECT_ROOT / "12_Data_Integration",
    "indices":      PROJECT_ROOT / "13_Indices",
    "weights":      PROJECT_ROOT / "14_Spatial_Weights",
    "descriptive":  PROJECT_ROOT / "15_Descriptive_Analysis",
    "econ":         PROJECT_ROOT / "16_Econometrics",
    "spatialecon":  PROJECT_ROOT / "17_Spatial_Econometrics",
    "robustness":   PROJECT_ROOT / "20_Robustness",
    "figures":      PROJECT_ROOT / "21_Figures",
    "tables":       PROJECT_ROOT / "22_Tables",
    "repro":        PROJECT_ROOT / "28_Reproducibility",
    "logs":         PROJECT_ROOT / "29_Logs",
}
for _p in P.values():
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- constants
SEARCH_FREEZE_DATE = "2026-07-30"   # decision D5, approved
SEARCH_START_DATE  = "2000-01-01"
CONTACT_EMAIL      = "ksaugat506@gmail.com"
USER_AGENT         = f"GlobalAIYieldResearchGaps/1.0 (mailto:{CONTACT_EMAIL})"
RANDOM_SEED        = 20260730

OPENALEX_BASE = "https://api.openalex.org"
CROSSREF_BASE = "https://api.crossref.org"

# Equal Earth for global thematic maps; WGS84 for storage/exchange only.
CRS_STORAGE = "EPSG:4326"
CRS_DISPLAY = "EPSG:8857"

PERIODS = [("2000-2009", 2000, 2009), ("2010-2017", 2010, 2017),
           ("2018-2022", 2018, 2022), ("2023-2026", 2023, 2026)]


# --------------------------------------------------------------------------- helpers
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# disk safety thresholds
# ---------------------------------------------------------------------------
#
# All in binary GiB (2**30 bytes), matching what `df -g` reports on macOS and what the
# shell wrapper tests. Before this was made explicit, `free_gb` divided by 1e9 while
# `run_phase1_daily.sh` used `df -g`, so the same literal 20 meant 20.0 decimal GB in
# Python and 20 GiB in the shell — a 7% gap. At 18.63 GiB free the Python check read
# 20.0 and passed while the shell check read 18 and refused, from the same disk.
#
# The thresholds differ by *what the operation writes*, not by how important it is.
#
# Phase 1 metadata retrieval writes compressed JSON: the whole 59-query corpus on disk is
# 423 MB, and a full daily allowance adds 100-200 MB. Holding it to the same floor as
# bulk PDF acquisition blocked a job needing a fifth of a gigabyte.
#
# This is a refinement of the workflow, not a relaxation of the full-text rule. The
# full-text floor is unchanged at 20 GiB, with 25 GiB preferred before bulk acquisition,
# because a few thousand PDFs is several gigabytes and is the case the original floor was
# written for.
METADATA_MIN_FREE_GIB = 16.0        # Phase 1 / metadata-only retrieval
FULLTEXT_MIN_FREE_GIB = 20.0        # any full-text acquisition: unchanged
FULLTEXT_PREFERRED_FREE_GIB = 25.0  # preferred before *bulk* PDF acquisition
ANALYTICAL_MIN_FREE_GIB = 20.0      # large analytical temporary outputs: unchanged

GIB = 2 ** 30


def free_gib(path=PROJECT_ROOT) -> float:
    """Free space in binary GiB — the same unit `df -g` reports."""
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / GIB


def free_gb(path=PROJECT_ROOT) -> float:
    """Free space in decimal GB. Retained for display only; thresholds use `free_gib`."""
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def check_disk(min_gib: float, path=PROJECT_ROOT) -> tuple[bool, float]:
    """Return (ok, free_gib) without raising. For periodic checks inside a loop."""
    gib = free_gib(path)
    return gib >= min_gib, gib


def require_disk(min_gib: float = ANALYTICAL_MIN_FREE_GIB, note: str = "") -> None:
    """Halt before a large operation if free space is below the given floor.

    The floor is a parameter because it depends on what is about to be written. Callers
    should pass one of the named thresholds above rather than a literal.
    """
    ok, gib = check_disk(min_gib)
    if not ok:
        raise SystemExit(
            f"HALT: free disk {gib:.2f} GiB is below the {min_gib:g} GiB floor. {note}\n"
            "Per CLAUDE.md: preserve checkpoints, do not delete user files or caches."
        )


def require_fulltext_disk(bulk: bool = False, note: str = "") -> None:
    """Guard full-text acquisition. Never call this with the metadata threshold.

    `bulk=True` applies the higher preferred floor, for acquiring many documents at once
    rather than resolving or fetching a single one.
    """
    floor = FULLTEXT_PREFERRED_FREE_GIB if bulk else FULLTEXT_MIN_FREE_GIB
    require_disk(floor, note or ("Bulk full-text acquisition halted."
                                 if bulk else "Full-text acquisition halted."))


class RunLogger:
    """Per-script log meeting the CLAUDE.md requirement: times, inputs, outputs,
    row counts, errors, warnings, package versions, seed."""

    def __init__(self, name: str):
        self.name = name
        self.t0 = time.time()
        self.start = utcnow()
        self.inputs, self.outputs, self.counts = [], [], {}
        self.errors, self.warnings, self.notes = [], [], []
        self.path = P["logs"] / f"{name}.log"
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
            handlers=[logging.FileHandler(self.path, mode="a"), logging.StreamHandler(sys.stdout)],
            force=True,
        )
        self.log = logging.getLogger(name)
        self.log.info("=" * 78)
        self.log.info("START %s  %s", name, self.start)

    def add_input(self, path):
        path = Path(path)
        rec = {"path": str(path.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in path.parents else str(path),
               "exists": path.exists(),
               "sha256": sha256(path) if path.exists() and path.is_file() else None,
               "bytes": path.stat().st_size if path.exists() and path.is_file() else None}
        self.inputs.append(rec)
        return rec

    def add_output(self, path, rows=None):
        path = Path(path)
        rec = {"path": str(path.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in path.parents else str(path),
               "exists": path.exists(),
               "sha256": sha256(path) if path.exists() and path.is_file() else None,
               "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
               "rows": rows}
        self.outputs.append(rec)
        self.log.info("OUTPUT %s  rows=%s  bytes=%s", rec["path"], rows, rec["bytes"])
        return rec

    def count(self, key, value):
        self.counts[key] = value
        self.log.info("COUNT %s = %s", key, value)

    def warn(self, msg):
        self.warnings.append(msg)
        self.log.warning(msg)

    def error(self, msg):
        self.errors.append(msg)
        self.log.error(msg)

    def note(self, msg):
        self.notes.append(msg)
        self.log.info(msg)

    def finish(self):
        import importlib
        pkgs = {}
        for m in ("numpy", "pandas", "scipy", "statsmodels", "geopandas", "libpysal",
                  "esda", "spreg", "networkx", "pyarrow", "pymc", "arviz", "sklearn"):
            try:
                pkgs[m] = getattr(importlib.import_module(m), "__version__", "?")
            except Exception:
                pkgs[m] = "unavailable"
        rec = {"script": self.name, "start": self.start, "end": utcnow(),
               "elapsed_s": round(time.time() - self.t0, 1),
               "python": platform.python_version(), "platform": platform.platform(),
               "random_seed": RANDOM_SEED, "packages": pkgs,
               "inputs": self.inputs, "outputs": self.outputs, "counts": self.counts,
               "errors": self.errors, "warnings": self.warnings, "notes": self.notes,
               "free_disk_gb_end": round(free_gb(), 1)}
        jp = P["logs"] / f"{self.name}.json"
        jp.write_text(json.dumps(rec, indent=2))
        self.log.info("END %s  elapsed=%ss  errors=%d  warnings=%d",
                      self.name, rec["elapsed_s"], len(self.errors), len(self.warnings))
        return rec


def append_decision(decision_id: str, phase: str, decision: str, rationale: str,
                    alternatives: str = "", effect: str = "") -> None:
    """Append to decision_log.md. Every autonomous choice is recorded."""
    f = P["mgmt"] / "decision_log.md"
    if not f.exists():
        f.write_text(
            "# Decision Log\n\nEvery choice made without the author present, with its rationale.\n"
            "Minor methodological choices follow the rule: identify defensible options, take the most\n"
            "transparent and conservative, run sensitivity analysis where feasible, document, continue.\n\n"
            "---\n\n"
        )
    with open(f, "a") as fh:
        fh.write(f"## {decision_id} — {decision}\n\n"
                 f"**Date:** {utcnow()[:10]} · **Phase:** {phase}\n\n"
                 f"**Rationale:** {rationale}\n\n")
        if alternatives:
            fh.write(f"**Alternatives considered:** {alternatives}\n\n")
        if effect:
            fh.write(f"**Effect on interpretation:** {effect}\n\n")
        fh.write("---\n\n")


if __name__ == "__main__":
    print(f"PROJECT_ROOT      {PROJECT_ROOT}")
    print(f"free disk         {free_gb():.1f} GB")
    print(f"search window     {SEARCH_START_DATE} .. {SEARCH_FREEZE_DATE}")
    print(f"random seed       {RANDOM_SEED}")
    print(f"directories       {len(P)} registered, all present")
