"""Phase 10 — research-side descriptive measures.

Every number the manuscript's Results 5.1 to 5.4 reports is produced here and written to
a file, so no statistic is typed by hand into the text.

Fractional counting is primary throughout; full counts are reported beside it and always
labelled. Countries with no eligible study are real zeros and are kept in every
denominator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> int:
    lg = RunLogger("phase10_01_research_side_descriptives")
    sp = P["integration"] / "study_level_dataset.parquet"
    sccp = P["integration"] / "study_country_crop_dataset.parquet"
    pp = P["integration"] / "country_crop_panel.parquet"
    for f in (sp, sccp, pp):
        lg.add_input(f)
    s = pd.read_parquet(sp)
    scc = pd.read_parquet(sccp)
    panel = pd.read_parquet(pp)

    out = {}

    # --- 5.1 corpus ---------------------------------------------------------------
    out["n_studies"] = int(len(s))
    out["n_conference"] = int(s["is_conference"].sum())
    out["n_journal_article"] = int((s["type"] == "article").sum())
    out["conference_share_pct"] = round(100 * s["is_conference"].mean(), 2)
    out["year_min"] = int(s["publication_year"].min())
    out["year_max"] = int(s["publication_year"].max())
    out["median_year"] = int(s["publication_year"].median())
    yr = s["publication_year"].value_counts().sort_index()
    out["studies_2000_2009"] = int(yr.loc[yr.index <= 2009].sum())
    out["studies_2010_2019"] = int(yr.loc[(yr.index >= 2010) & (yr.index <= 2019)].sum())
    out["studies_2020_2026"] = int(yr.loc[yr.index >= 2020].sum())
    out["share_since_2018_pct"] = round(
        100 * float((s["publication_year"] >= 2018).mean()), 2)
    out["n_open_access"] = int(s["is_oa"].fillna(False).sum())
    out["open_access_share_pct"] = round(100 * s["is_oa"].fillna(False).mean(), 2)

    # --- coverage of the two extractions ------------------------------------------
    out["n_with_country"] = int(s["loc_country_iso3"].notna().sum())
    out["n_global_scope"] = int((s["loc_study_scope"] == "global").sum())
    out["n_multi_country"] = int((s["loc_study_scope"] == "multi_country").sum())
    out["n_region_scope"] = int((s["loc_study_scope"] == "region").sum())
    out["n_location_unresolved"] = int((s["loc_study_scope"] == "unresolved").sum())
    out["location_resolved_share_pct"] = round(
        100 * float((s["loc_study_scope"] != "unresolved").mean()), 2)
    out["n_country_from_locative_cue"] = int(
        (s["loc_location_cue_type"] == "locative_only").sum())
    out["n_with_crop"] = int(s["crop_n_crops"].fillna(0).gt(0).sum())
    out["n_multicrop"] = int(s["crop_multicrop"].fillna(False).sum())
    out["n_crop_unresolved"] = int((s["crop_crop_scope"] == "unresolved").sum())

    # --- 5.2 geography ------------------------------------------------------------
    cc = (scc.groupby("iso3")["fractional_weight"].sum()
          .sort_values(ascending=False))
    cfull = scc.groupby("iso3")["openalex_id"].nunique().sort_values(ascending=False)
    out["n_countries_with_a_study"] = int(len(cc))
    out["n_countries_in_panel"] = int(panel["iso3"].nunique())
    out["n_countries_zero_studies"] = int(panel["iso3"].nunique() - len(cc))
    out["gini_countries_fractional"] = round(
        gini(panel.groupby("iso3")["n_studies_fractional"].sum().values), 4)
    tot = float(cc.sum())
    for k in (1, 3, 5, 10, 20):
        out[f"top{k}_country_share_pct"] = round(100 * float(cc.head(k).sum()) / tot, 2)
    out["top10_countries_fractional"] = {i: round(float(v), 1)
                                         for i, v in cc.head(10).items()}
    out["top10_countries_full"] = {i: int(v) for i, v in cfull.head(10).items()}

    reg = (scc.merge(panel[["iso3", "wb_region", "wb_income_group"]].drop_duplicates("iso3"),
                     on="iso3", how="left"))
    out["region_shares_pct"] = {
        str(k): round(100 * float(v) / tot, 2)
        for k, v in reg.groupby("wb_region")["fractional_weight"].sum()
        .sort_values(ascending=False).items()}
    out["income_shares_pct"] = {
        str(k): round(100 * float(v) / tot, 2)
        for k, v in reg.groupby("wb_income_group")["fractional_weight"].sum()
        .sort_values(ascending=False).items()}

    # --- 5.3 crops ----------------------------------------------------------------
    cr = scc.groupby("crop_standard_name")["fractional_weight"].sum().sort_values(
        ascending=False)
    out["n_crops_studied"] = int(len(cr))
    out["n_crops_in_panel"] = int(panel["crop_standard_name"].nunique())
    out["top10_crops_fractional"] = {c: round(float(v), 1) for c, v in cr.head(10).items()}
    out["top3_crop_share_pct"] = round(100 * float(cr.head(3).sum()) / tot, 2)
    out["gini_crops_fractional"] = round(gini(cr.values), 4)

    # --- 5.4 attention against agricultural importance ------------------------------
    p = panel.copy()
    ok = p["production_t_mean"].notna() & (p["production_t_mean"] > 0)
    out["panel_cells"] = int(len(p))
    out["panel_cells_with_study"] = int(p["has_any_study"].sum())
    out["panel_cells_measured_zero"] = int((~p["has_any_study"]).sum())
    out["panel_cells_with_production_denominator"] = int(ok.sum())
    corr = p.loc[ok, ["n_studies_fractional", "production_t_mean"]].corr(
        method="spearman").iloc[0, 1]
    out["spearman_research_vs_production_cellwise"] = round(float(corr), 4)
    cn = p.groupby("iso3").agg(res=("n_studies_fractional", "sum"),
                               prod=("production_t_mean", "sum")).dropna()
    out["spearman_research_vs_production_countrywise"] = round(
        float(cn.corr(method="spearman").iloc[0, 1]), 4)

    gap = p.loc[ok].nlargest(15, "research_minus_production_share")[
        ["iso3", "crop_standard_name", "research_share",
         "production_share", "research_minus_production_share"]]
    under = p.loc[ok].nsmallest(15, "research_minus_production_share")[
        ["iso3", "crop_standard_name", "research_share",
         "production_share", "research_minus_production_share"]]
    gap.to_csv(P["descriptive"] / "most_over_researched_cells.csv", index=False)
    under.to_csv(P["descriptive"] / "most_under_researched_cells.csv", index=False)
    lg.add_output(P["descriptive"] / "most_over_researched_cells.csv", rows=len(gap))
    lg.add_output(P["descriptive"] / "most_under_researched_cells.csv", rows=len(under))

    # --- leadership and collaboration ------------------------------------------------
    lead = s.dropna(subset=["local_first_author"])
    out["n_leadership_observed"] = int(len(lead))
    out["local_first_author_share_pct"] = round(
        100 * float(lead["local_first_author"].mean()), 2) if len(lead) else None
    coll = s.dropna(subset=["international_collaboration"])
    out["n_collaboration_observed"] = int(len(coll))
    out["international_collaboration_share_pct"] = round(
        100 * float(coll["international_collaboration"].mean()), 2) if len(coll) else None
    lp = s.dropna(subset=["local_participation"])
    out["local_participation_share_pct"] = round(
        100 * float(lp["local_participation"].mean()), 2) if len(lp) else None

    if len(lead):
        by_inc = (lead.merge(panel[["iso3", "wb_income_group"]].drop_duplicates("iso3"),
                             left_on="loc_country_iso3", right_on="iso3", how="left")
                  .groupby("wb_income_group")["local_first_author"]
                  .agg(["size", "mean"]))
        out["local_first_author_share_by_income_pct"] = {
            str(k): [int(r["size"]), round(100 * float(r["mean"]), 2)]
            for k, r in by_inc.iterrows()}

    # --- citations -------------------------------------------------------------------
    cb = pd.to_numeric(s["cited_by_count"], errors="coerce")
    out["citations_total"] = int(np.nansum(cb))
    out["citations_median"] = float(np.nanmedian(cb))
    out["citations_mean"] = round(float(np.nanmean(cb)), 2)
    out["share_uncited_pct"] = round(100 * float((cb.fillna(0) == 0).mean()), 2)

    op = P["descriptive"] / "research_side_descriptives.json"
    op.write_text(json.dumps(out, indent=2, default=str) + "\n")
    lg.add_output(op)
    for k, v in out.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lg.count(k, v)

    print(json.dumps({k: v for k, v in out.items()
                      if not isinstance(v, dict)}, indent=1, default=str))
    print("\ntop 10 countries (fractional):", out["top10_countries_fractional"])
    print("top 10 crops (fractional):", out["top10_crops_fractional"])
    print("region shares:", out["region_shares_pct"])
    print("income shares:", out["income_shares_pct"])
    lg.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
