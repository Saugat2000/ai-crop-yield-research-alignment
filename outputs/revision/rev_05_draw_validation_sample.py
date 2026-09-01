"""Step 5: draw a reproducible stratified random validation sample for study-location
and crop coding.

The published corpus carries loc_verification = 'unverified' and loc_coded_by =
'model_assisted' for all 7,045 records; no location or crop validation exists. This
draws a stratified probability sample so that coding accuracy can be estimated.

Adjudication is model-assisted and is labelled as such everywhere. It is a first pass
that a human must confirm; it is never described as human review.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

OUT = HERE
SEED = 20260831
TARGET = {  # stratum -> records to draw
    "single_country|study_area": 45,
    "single_country|locative_only": 45,
    "multi_country": 20,
    "global|allocated": 30,
    "region": 20,
    "unresolved|allocated": 30,
    "unresolved|not_allocated": 20,
}


def main():
    lg = RunLogger("rev_05_draw_validation_sample")
    src = ROOT / "12_Data_Integration" / "study_level_dataset.parquet"
    lg.add_input(src)
    d = pd.read_parquet(src)
    scc = pd.read_parquet(ROOT / "12_Data_Integration" / "study_country_crop_dataset.parquet")
    lg.add_input(ROOT / "12_Data_Integration" / "study_country_crop_dataset.parquet")
    allocated = set(scc.openalex_id.unique())
    d["is_allocated"] = d.openalex_id.isin(allocated)

    def stratum(r):
        s = r.loc_study_scope
        if s == "single_country":
            return f"single_country|{r.loc_location_cue_type or 'none'}"
        if s in ("multi_country", "region"):
            return s
        return f"{s}|{'allocated' if r.is_allocated else 'not_allocated'}"

    d["stratum"] = d.apply(stratum, axis=1)
    sizes = d.stratum.value_counts().to_dict()
    print("STRATUM SIZES IN THE CORPUS")
    for k, v in sorted(sizes.items()):
        print(f"  {k:36s} {v:6d}")

    rng = np.random.default_rng(SEED)
    picks = []
    for st, n in TARGET.items():
        pool = d[d.stratum == st]
        if len(pool) == 0:
            print(f"  WARNING stratum absent: {st}")
            continue
        take = min(n, len(pool))
        sel = pool.iloc[rng.choice(len(pool), size=take, replace=False)].copy()
        sel["stratum_size"] = len(pool)
        sel["stratum_sampled"] = take
        sel["design_weight"] = len(pool) / take
        picks.append(sel)
    s = pd.concat(picks, ignore_index=True)

    cols = ["openalex_id", "stratum", "stratum_size", "stratum_sampled", "design_weight",
            "title", "abstract", "loc_study_scope", "loc_country_iso3", "loc_country_name",
            "loc_countries_all_iso3", "loc_location_cue_type", "loc_confidence",
            "loc_evidence_text", "loc_rejected_mentions", "crop_standardized_crops",
            "crop_crop_scope", "crop_confidence", "crop_evidence_text", "is_allocated"]
    s = s[[c for c in cols if c in s.columns]]
    s["adjudicated_location_correct"] = ""
    s["adjudicated_country_should_be"] = ""
    s["adjudicated_crop_correct"] = ""
    s["adjudicated_note"] = ""
    s["decision_method"] = "MODEL_ASSISTED_VALIDATION - not human review"
    s["is_human_review"] = False

    s.to_csv(OUT / "loc_crop_validation_sample.csv", index=False)
    print(f"\ndrew {len(s)} records across {s.stratum.nunique()} strata, seed {SEED}")
    print(s.groupby("stratum").agg(sampled=("openalex_id", "size"),
                                   size=("stratum_size", "first"),
                                   weight=("design_weight", "first")).round(1).to_string())
    print(f"\nabstract available for {int(s.abstract.notna().sum())} of {len(s)} sampled records")
    lg.add_output(OUT / "loc_crop_validation_sample.csv", rows=len(s))
    lg.finish()


if __name__ == "__main__":
    main()
