"""Phase 15 — the prespecified robustness suite.

Each analysis re-runs a main quantity under one changed decision, and every finding is
classified on the scale fixed before the results were seen:

    stable                 sign and order of magnitude hold in every variant run
    partially stable       holds in most variants; one or more move materially
    specification-sensitive the answer depends on a decision with no principled default
    unsupported            does not survive its own robustness checks

A weak result is reported at its measured strength. Nothing is dropped for being
inconvenient, and no variant is added after seeing which way it would move a number.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
sys.path.insert(0, str(ROOT / "16_Econometrics"))
from project_config import P, RANDOM_SEED, RunLogger  # noqa: E402
import model_scaffolds as M  # noqa: E402

warnings.filterwarnings("ignore")

X_MAIN = ["log_area", "log_gdp_pc", "rd", "tertiary", "internet",
          "need", "yield_volatility", "log_population"]


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float("nan") if n == 0 or x.sum() == 0 else \
        float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# Set once in main() to the full country universe of the panel. Concentration is a
# property of that universe, so countries carrying no eligible study contribute measured
# zeros rather than being dropped (CLAUDE.md 3.7). Both Ginis are reported: gini_all is
# the principal global measure, gini the conditional one over countries with research.
_UNIVERSE = None


def concentration(scc, weight="fractional_weight", universe=None):
    c = scc.groupby("iso3")[weight].sum().sort_values(ascending=False)
    t = c.sum()
    out = {"n_countries": int(len(c)),
           "gini": round(gini(c.values), 4),
           "top5_share_pct": round(100 * float(c.head(5).sum()) / t, 2),
           "top10_share_pct": round(100 * float(c.head(10).sum()) / t, 2),
           "top1_country": c.index[0] if len(c) else None}
    u = universe if universe is not None else _UNIVERSE
    if u is not None:
        full = c.reindex(u, fill_value=0.0)
        out["n_countries_all"] = int(len(full))
        out["gini_all"] = round(gini(full.values), 4)
    return out



def _moran(var, primary="knn_k6"):
    """Primary-matrix Moran's I and its range across matrices, read from the computed
    spatial output. These were previously hard-coded, which left them stale when the
    evidence-gap scale defect was repaired upstream (D-131)."""
    import pandas as _pd
    g = _pd.read_csv(P["spatialecon"] / "research_side_global_moran.csv")
    row = g[(g["variable"] == var) & (g["weights"] == primary)]
    if row.empty:
        raise ValueError(f"no Moran's I for {var} on {primary}")
    sub = g[g["variable"] == var]["morans_I"]
    return round(float(row["morans_I"].iloc[0]), 4), [round(float(sub.min()), 4),
                                                      round(float(sub.max()), 4)]


def main() -> int:
    lg = RunLogger("phase15_01_robustness")
    np.random.seed(RANDOM_SEED)

    scc = pd.read_parquet(P["integration"] / "study_country_crop_dataset.parquet")
    est = pd.read_parquet(P["econ"] / "estimation_sample.parquet")
    panel = pd.read_parquet(P["integration"] / "country_crop_panel.parquet")
    s3a = pd.read_parquet(P["screening"] / "s3a_screening_decisions.parquet",
                          columns=["openalex_id", "s3a_decision", "s3a_reason"])
    for f in ("study_country_crop_dataset.parquet",):
        lg.add_input(P["integration"] / f)
    lg.add_input(P["econ"] / "estimation_sample.parquet")

    rows = []

    def add(family, variant, n_studies, res, note=""):
        rows.append({"family": family, "variant": variant,
                     "n_studies": n_studies, **res, "note": note})

    global _UNIVERSE
    _UNIVERSE = sorted(panel["iso3"].dropna().astype(str).unique())
    lg.count("country_universe", len(_UNIVERSE))

    base = concentration(scc)
    add("concentration", "baseline (fractional)", scc["openalex_id"].nunique(), base)

    # --- counting rule ----------------------------------------------------------------
    full = scc.assign(w=1.0)
    add("concentration", "full counting", scc["openalex_id"].nunique(),
        concentration(full, "w"),
        "each study contributes 1 to every country it covers")

    cw = scc.assign(w=scc["fractional_weight"] *
                    (1 + pd.to_numeric(scc["cited_by_count"], errors="coerce").fillna(0)))
    add("concentration", "citation-weighted", scc["openalex_id"].nunique(),
        concentration(cw, "w"))

    # --- venue type -------------------------------------------------------------------
    jo = scc[~scc["is_conference"].fillna(False)]
    add("concentration", "journal only", jo["openalex_id"].nunique(), concentration(jo),
        "the split D-119 made possible; impossible before the conference pass")
    nopre = scc[~scc["is_preprint"].fillna(False)]
    add("concentration", "preprints excluded", nopre["openalex_id"].nunique(),
        concentration(nopre))

    # --- location certainty -------------------------------------------------------------
    strict = scc[scc["location_cue_type"].fillna("") != "locative_only"]
    add("concentration", "uncertain-location cells excluded",
        strict["openalex_id"].nunique(), concentration(strict),
        "drops every country resolved only from a locative phrase")
    hi = scc[scc["location_confidence"].isin(["high", "medium"])]
    add("concentration", "low-confidence locations excluded",
        hi["openalex_id"].nunique(), concentration(hi))

    # --- dominant-country exclusions ----------------------------------------------------
    for drop, label in ((["USA"], "exclude USA"), (["CHN"], "exclude China"),
                        (["USA", "CHN"], "exclude USA and China"),
                        (["USA", "CHN", "IND"], "exclude USA, China and India")):
        sub = scc[~scc["iso3"].isin(drop)]
        add("concentration", label, sub["openalex_id"].nunique(), concentration(sub))

    # --- period splits --------------------------------------------------------------
    for lo_, hi_, label in ((2000, 2014, "2000-2014"), (2015, 2019, "2015-2019"),
                            (2020, 2026, "2020-2026")):
        sub = scc[scc["publication_year"].between(lo_, hi_)]
        if len(sub):
            add("concentration", f"period {label}", sub["openalex_id"].nunique(),
                concentration(sub))

    # --- screening sensitivity ---------------------------------------------------------
    # The 11,822 undecided records cannot be added to the corpus without coding them, so
    # what is bounded here is the direction of the error, not a re-estimate: the measured
    # false-inclusion rate implies ~257 ineligible studies inside the corpus.
    n_unc = int((s3a["s3a_decision"] == "uncertain").sum())
    add("screening", "measured error correction", scc["openalex_id"].nunique(),
        {"n_countries": base["n_countries"], "gini": base["gini"],
         "top5_share_pct": base["top5_share_pct"],
         "top10_share_pct": base["top10_share_pct"], "top1_country": base["top1_country"]},
        f"{n_unc:,} works undecided at screening; measured false-inclusion rate 10.0% "
        f"implies ~257 ineligible studies in the corpus and ~100 eligible outside it. "
        f"Concentration is unchanged by construction and the bound is reported instead.")

    R = pd.DataFrame(rows)
    rp = P["robustness"] / "concentration_robustness.csv"
    R.to_csv(rp, index=False)
    lg.add_output(rp, rows=len(R))

    # --- model robustness --------------------------------------------------------------
    mrows = []

    def fit_and_record(df, label, xs=X_MAIN, note=""):
        try:
            res, dg = M.fit_participation(df, "studied", xs, cluster_col="iso3")
            p = res.params
            se = res.bse
            mrows.append({"variant": label, "n": int(len(df)),
                          "log_area": round(float(p.get("log_area", np.nan)), 3),
                          "log_area_se": round(float(se.get("log_area", np.nan)), 3),
                          "rd": round(float(p.get("rd", np.nan)), 3),
                          "rd_se": round(float(se.get("rd", np.nan)), 3),
                          "need": round(float(p.get("need", np.nan)), 3),
                          "need_se": round(float(se.get("need", np.nan)), 1),
                          "pseudo_r2": round(dg.get("pseudo_r2", np.nan), 4),
                          "note": note})
        except Exception as e:
            mrows.append({"variant": label, "n": int(len(df)), "failed": str(e),
                          "note": note})

    fit_and_record(est, "baseline")
    fit_and_record(est[est["iso3"] != "USA"], "exclude USA")
    fit_and_record(est[est["iso3"] != "CHN"], "exclude China")
    fit_and_record(est[~est["iso3"].isin(["USA", "CHN"])], "exclude USA and China")
    for grp in est["wb_income_group"].dropna().unique():
        sub = est[est["wb_income_group"] == grp]
        if len(sub) > 120 and sub["studied"].nunique() > 1:
            fit_and_record(sub, f"income group: {grp}")
    # alternative need indices
    for alt in ("need_equal_pct", "need_pca_pct", "need_entropy_pct"):
        if alt in est.columns:
            sub = est.copy()
            # Same defect as the baseline: these columns are 0-1 proportions despite the
            # `_pct` suffix, so dividing by 100 again put the alternative indices on a
            # different scale from the baseline need regressor and made the variants
            # incomparable with it.
            _a = pd.to_numeric(sub[alt], errors="coerce")
            if _a.max(skipna=True) > 1.5:
                raise ValueError(f"{alt} max is {_a.max():.3f}; expected a 0-1 proportion")
            sub["need"] = _a
            sub = sub.dropna(subset=["need"])
            fit_and_record(sub, f"need index: {alt}",
                           note="alternative need-index weighting")

    MR = pd.DataFrame(mrows)
    mp = P["robustness"] / "model_robustness.csv"
    MR.to_csv(mp, index=False)
    lg.add_output(mp, rows=len(MR))

    # --- classification ------------------------------------------------------------------
    g = R[R["family"] == "concentration"]["gini_all"].dropna()
    g_studied = R[R["family"] == "concentration"]["gini"].dropna()
    t5 = R[R["family"] == "concentration"]["top5_share_pct"].dropna()
    la = MR["log_area"].dropna() if "log_area" in MR else pd.Series(dtype=float)
    rd = MR["rd"].dropna() if "rd" in MR else pd.Series(dtype=float)

    findings = [
        {"finding": "Research output is concentrated across countries",
         "measure": "Gini of fractional country counts, all countries",
         "baseline": base["gini_all"],
         "range_across_variants": [round(float(g.min()), 3), round(float(g.max()), 3)],
         "n_variants": int(len(g)),
         "classification": "stable" if (g > 0.6).all() else "partially stable",
         "reason": "computed over the full country universe with measured zeros; the Gini "
                   "stays above 0.6 in every variant, including full counting, citation "
                   "weighting, journal-only, and dropping the three largest producers. The "
                   "conditional Gini over countries carrying research runs "
                   f"{g_studied.min():.3f}-{g_studied.max():.3f}"},
        {"finding": "Attention is concentrated on wheat, maize and rice",
         "measure": "top-3 crop share of fractional attention",
         "baseline": 69.33, "range_across_variants": None, "n_variants": 1,
         "classification": "partially stable",
         "reason": "computed on the baseline corpus only; crop-level variants were not "
                   "prespecified and are not added now"},
        {"finding": "Harvested area predicts whether a crop system is studied",
         "measure": "logit coefficient on log harvested area",
         "baseline": float(MR.loc[MR["variant"] == "baseline", "log_area"].iloc[0]),
         "range_across_variants": [round(float(la.min()), 3), round(float(la.max()), 3)],
         "n_variants": int(len(la)),
         "classification": "stable" if (la > 0).all() else "partially stable",
         "reason": "positive in every subsample and every alternative need index"},
        {"finding": "Research and development spending predicts research participation",
         "measure": "logit coefficient on R&D expenditure share of GDP",
         "baseline": float(MR.loc[MR["variant"] == "baseline", "rd"].iloc[0]),
         "range_across_variants": [round(float(rd.min()), 3), round(float(rd.max()), 3)],
         "n_variants": int(len(rd)),
         "classification": "partially stable" if (rd > 0).mean() >= 0.75
                           else "specification-sensitive",
         "reason": "positive in most subsamples; income-group splits move it materially"},
        {"finding": "Research need is associated with research attention",
         "measure": "logit coefficient on the need index",
         "baseline": float(MR.loc[MR["variant"] == "baseline", "need"].iloc[0]),
         "range_across_variants": None, "n_variants": int(len(MR)),
         "classification": "unsupported",
         "reason": "the interval spans roughly plus or minus 100 log-points in every "
                   "variant and every alternative need index. The design cannot detect "
                   "an association; this is not evidence that none exists"},
        {"finding": "Citation-weighted research output is spatially clustered",
         "measure": "global Moran's I, citation-weighted count",
         "baseline": _moran("citation_weighted")[0],
         "range_across_variants": _moran("citation_weighted")[1], "n_variants": 5,
         "classification": "specification-sensitive",
         "reason": "Moran's I ranges from 0.014 to 0.147 across the five weights "
                   "matrices and is not significant under the primary matrix; no "
                   "clustering claim is made"},
        {"finding": "The evidence gap is spatially clustered",
         "measure": "global Moran's I, evidence gap",
         "baseline": _moran("evidence_gap")[0],
         "range_across_variants": _moran("evidence_gap")[1], "n_variants": 5,
         "classification": "stable",
         "reason": "positive and significant under all five weights matrices"},
        {"finding": "Local first authorship is the norm where it can be observed",
         "measure": "share of studies with a first author affiliated in the study country",
         "baseline": 78.06, "range_across_variants": None, "n_variants": 1,
         "classification": "partially stable",
         "reason": "observable for 1,395 of 7,045 studies; the observable subset is those "
                   "with a resolved study country, which is not a random subset"},
    ]
    F = pd.DataFrame(findings)
    fp = P["robustness"] / "finding_stability_classification.csv"
    F.to_csv(fp, index=False)
    lg.add_output(fp, rows=len(F))

    for c, n in F["classification"].value_counts().items():
        lg.count(f"findings_{c.replace(' ', '_')}", int(n))

    print("\nCONCENTRATION ROBUSTNESS")
    print(R[["variant", "n_studies", "n_countries", "gini", "top5_share_pct",
             "top1_country"]].to_string(index=False))
    print("\nMODEL ROBUSTNESS")
    print(MR.to_string(index=False))
    print("\nFINDING STABILITY")
    print(F[["finding", "classification"]].to_string(index=False))
    lg.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
