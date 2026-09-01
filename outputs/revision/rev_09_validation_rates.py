"""Step 5: estimate study-location and crop coding accuracy from the stratified sample.

Adjudication is MODEL-ASSISTED and is labelled as such in every output. It is a first
pass requiring human confirmation; it is never reported as human review.

Three distinct quantities are estimated, because they answer different questions:
  1. coding accuracy  - was scope AND country coded correctly (strict)
  2. country precision - among records that were ALLOCATED to countries (the records
     that actually drive the research counts), is the assignment correct
  3. false-unresolved rate - among records coded unresolved, how many name a study area
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402
OUT = HERE


def wilson(k, n, a=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - a / 2); p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    lg = RunLogger("rev_09_validation_rates")
    smp = pd.read_csv(OUT / "loc_crop_validation_sample.csv")
    v = pd.concat([pd.read_csv(p) for p in sorted((OUT / "adjudication").glob("verdicts_*.csv"))],
                  ignore_index=True)
    smp["id"] = smp.openalex_id.str.split("/").str[-1]
    d = smp.merge(v, on="id", how="left", validate="one_to_one")
    lg.add_input(OUT / "loc_crop_validation_sample.csv")
    print(f"adjudicated {d.loc_verdict.notna().sum()} of {len(d)} sampled records")
    assert d.loc_verdict.notna().all(), "unadjudicated records remain"

    # ------------------------------------------------------------------ per stratum
    rows = []
    for st, g in d.groupby("stratum"):
        n = len(g)
        for metric, mask in [
                ("location coded correctly", g.loc_verdict.eq("correct")),
                ("location coded incorrectly", g.loc_verdict.eq("incorrect")),
                ("cannot determine from title/abstract", g.loc_verdict.eq("cannot_determine"))]:
            k = int(mask.sum())
            lo, hi = wilson(k, n)
            rows.append(dict(stratum=st, stratum_size=int(g.stratum_size.iloc[0]),
                             sampled=n, design_weight=float(g.design_weight.iloc[0]),
                             metric=metric, count=k, rate=k / n, ci_low=lo, ci_high=hi,
                             implied_corpus_count=k * float(g.design_weight.iloc[0])))
    st_tab = pd.DataFrame(rows)
    st_tab.to_csv(OUT / "validation_rates_by_stratum.csv", index=False)
    print("\nLOCATION CODING BY STRATUM (model-assisted adjudication)")
    piv = st_tab[st_tab.metric == "location coded correctly"]
    print(piv[["stratum", "stratum_size", "sampled", "count", "rate", "ci_low", "ci_high"]]
          .round(3).to_string(index=False))

    # ------------------------------------------------------------------ headline rates
    ALLOC = ["single_country|study_area", "single_country|locative_only", "multi_country",
             "global|allocated", "unresolved|allocated"]
    out = []

    def add(label, sub, note=""):
        n = len(sub)
        k = int(sub.loc_verdict.eq("correct").sum())
        bad = int(sub.loc_verdict.eq("incorrect").sum())
        det = sub[sub.loc_verdict.ne("cannot_determine")]
        kd, nd = int(det.loc_verdict.eq("correct").sum()), len(det)
        lo, hi = wilson(kd, nd)
        out.append(dict(quantity=label, n_sampled=n, n_determinable=nd, n_correct=kd,
                        rate=kd / nd if nd else np.nan, ci_low=lo, ci_high=hi, note=note))

    add("Location coding accuracy, allocated records (drive the counts)",
        d[d.stratum.isin(ALLOC)], "strict: scope and country both correct")
    add("Location coding accuracy, single-country study-area cue",
        d[d.stratum == "single_country|study_area"])
    add("Location coding accuracy, single-country locative cue",
        d[d.stratum == "single_country|locative_only"])

    # country-assignment precision: ignore scope-label-only errors
    alloc = d[d.stratum.isin(ALLOC)].copy()
    scope_only = alloc.loc_error_type.isin(["other"]) & alloc.loc_should_be.fillna("").eq("")
    ca = alloc[~scope_only]
    k = int(ca.loc_verdict.eq("correct").sum())
    det = ca[ca.loc_verdict.ne("cannot_determine")]
    kd, nd = int(det.loc_verdict.eq("correct").sum()), len(det)
    lo, hi = wilson(kd, nd)
    out.append(dict(quantity="Country-assignment precision, allocated records",
                    n_sampled=len(ca), n_determinable=nd, n_correct=kd,
                    rate=kd / nd if nd else np.nan, ci_low=lo, ci_high=hi,
                    note="excludes scope-label-only errors with no wrong country"))

    # false-unresolved rate
    unres = d[d.stratum.str.startswith("unresolved")]
    k = int(unres.loc_error_type.eq("missed_country_should_be_resolved").sum())
    lo, hi = wilson(k, len(unres))
    out.append(dict(quantity="False-unresolved rate (study area present but not coded)",
                    n_sampled=len(unres), n_determinable=len(unres), n_correct=k,
                    rate=k / len(unres), ci_low=lo, ci_high=hi,
                    note="higher is worse; these are missed locations"))
    # weighted projection of missed locations to the corpus
    w = unres.groupby("stratum").apply(
        lambda g: g.loc_error_type.eq("missed_country_should_be_resolved").mean()
        * g.stratum_size.iloc[0])
    print(f"\nprojected studies coded unresolved that name a study area: "
          f"{w.sum():,.0f} (of 3,985 unresolved)")
    for st, val in w.items():
        print(f"    {st:32s} {val:8,.0f}")

    # crop coding
    cr = d[d.crop_verdict.isin(["correct", "incorrect"])]
    k, n = int(cr.crop_verdict.eq("correct").sum()), len(cr)
    lo, hi = wilson(k, n)
    out.append(dict(quantity="Crop coding precision (records with a determinable crop)",
                    n_sampled=len(d), n_determinable=n, n_correct=k, rate=k / n,
                    ci_low=lo, ci_high=hi, note="excludes not_applicable and cannot_determine"))

    hh = pd.DataFrame(out)
    hh["decision_method"] = "MODEL_ASSISTED_VALIDATION - not human review"
    hh.to_csv(OUT / "validation_headline_rates.csv", index=False)
    print("\nHEADLINE VALIDATION RATES (95% Wilson intervals)")
    print(hh[["quantity", "n_determinable", "n_correct", "rate", "ci_low", "ci_high"]]
          .round(3).to_string(index=False))

    # error taxonomy
    et = (d[d.loc_verdict.eq("incorrect")].loc_error_type.value_counts()
          .rename_axis("error_type").reset_index(name="count"))
    et.to_csv(OUT / "validation_error_types.csv", index=False)
    print("\nLOCATION ERROR TAXONOMY")
    print(et.to_string(index=False))

    d.to_csv(OUT / "loc_crop_validation_adjudicated.csv", index=False)
    for f in ["validation_rates_by_stratum.csv", "validation_headline_rates.csv",
              "validation_error_types.csv", "loc_crop_validation_adjudicated.csv"]:
        lg.add_output(OUT / f)
    lg.finish()
    print("\nrev_09_validation_rates complete")


if __name__ == "__main__":
    main()
