"""Pre-submission audit M1/Mo1/XVI: scale-excluded research-need index.

Rebuilds the original 11-component rank-aggregation index (guard: must match the stored
values exactly), then constructs the scale-excluded index dropping the two agricultural-
scale components (share of global harvested area, share of global production). Yield
volatility stays in the index; the regression audit drops the standalone volatility
regressor when this index is used, so the quantity appears in exactly one place.

The scale-excluded index is computed for the same 193 countries as the original so that
every downstream comparison is like-for-like; coverage sensitivity is a separate exercise.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

COMP = {  # component -> direction (+1 higher = more need)
    "undernourishment_pct": +1, "food_insecurity_mod_sev_pct": +1,
    "cereal_import_dependency_pct": +1, "dietary_energy_adequacy_pct": -1,
    "employment_agriculture_share_pct": +1, "agri_value_added_share_gdp_pct": +1,
    "agri_value_added_per_worker_usd": -1, "temp_warming_since_baseline_c": +1,
    "area_weighted_yield_volatility": +1,
    "crop_area_share_global": +1, "crop_production_share_global": +1,
}
SCALE = ["crop_area_share_global", "crop_production_share_global"]


def rank_agg(d, comps, keep_mask):
    R = pd.DataFrame({c: (pd.to_numeric(d[c], errors="coerce") * s).rank(pct=True)
                      for c, s in comps.items()})
    mean_rank = R.mean(axis=1, skipna=True)
    return mean_rank.where(keep_mask).rank(pct=True)


def main() -> int:
    lg = RunLogger("audit_01_need_index")
    d = pd.read_parquet(P["indices"] / "country_need_indices.parquet")
    lg.add_input(P["indices"] / "country_need_indices.parquet")
    stored = pd.to_numeric(d["need_rank_pct"], errors="coerce")
    indexed = stored.notna()

    # guard: exact replication of the original construction
    recon = rank_agg(d, COMP, indexed)
    dev = float((recon[indexed] - stored[indexed]).abs().max())
    lg.count("replication_max_abs_dev", dev)
    if dev > 1e-9:
        raise ValueError(f"cannot replicate stored index (max dev {dev}); do not proceed")

    # scale-excluded index on the SAME 193 countries
    comps9 = {c: s for c, s in COMP.items() if c not in SCALE}
    d["need9_rank_pct"] = rank_agg(d, comps9, indexed)
    d["n_components9"] = d[list(comps9)].notna().sum(axis=1).where(indexed)

    sp = d.loc[indexed, ["need_rank_pct", "need9_rank_pct"]].corr(method="spearman").iloc[0, 1]
    lg.count("spearman_11_vs_9", round(float(sp), 4))
    move = (d["need9_rank_pct"] - stored).abs() * 100
    movers = d.loc[indexed].assign(move_pp=move).nlargest(10, "move_pp")[
        ["iso3", "wb_income_group", "need_rank_pct", "need9_rank_pct", "move_pp"]]

    # research percentile identical to the spatial pipeline: country fractional totals
    panel = pd.read_parquet(P["integration"] / "country_crop_panel.parquet")
    lg.add_input(P["integration"] / "country_crop_panel.parquet")
    cc = panel.groupby("iso3")["n_studies_fractional"].sum()
    d = d.merge(cc.rename("n_studies_fractional"), left_on="iso3", right_index=True, how="left")
    d["n_studies_fractional"] = d["n_studies_fractional"].fillna(0.0)
    d["research_pct"] = d["n_studies_fractional"].rank(pct=True) * 100

    d["gap11"] = d["research_pct"] - stored * 100
    d["gap9"] = d["research_pct"] - d["need9_rank_pct"] * 100
    gcorr = d[["gap11", "gap9"]].corr(method="spearman").iloc[0, 1]
    lg.count("spearman_gap11_vs_gap9", round(float(gcorr), 4))

    # high-need / low-research quadrant (Fig 3 definition: >= median need, <= median research)
    for tag, col in (("11", stored * 100), ("9", d["need9_rank_pct"] * 100)):
        m = col.notna()
        q = ((col >= col[m].median()) & (d["research_pct"] <= d.loc[m, "research_pct"].median()) & m).sum()
        lg.count(f"quadrant_highneed_lowresearch_{tag}", int(q))

    # tie handling (XVI): min-rank research percentile and z-score formulation
    d["research_pct_min"] = d["n_studies_fractional"].rank(pct=True, method="min") * 100
    gap_min = d["research_pct_min"] - stored * 100
    lg.count("spearman_gap11_vs_gapminrank",
             round(float(pd.concat([d["gap11"], gap_min], axis=1).corr(method="spearman").iloc[0, 1]), 4))
    z = lambda s: (s - s.mean()) / s.std()
    gap_z = z(np.log1p(d["n_studies_fractional"])) - z(stored)
    lg.count("spearman_gap11_vs_gapz",
             round(float(pd.concat([d["gap11"], gap_z.rename("gz")], axis=1).corr(method="spearman").iloc[0, 1]), 4))

    # coverage sensitivity (Mo1): rank stability under minimum-component thresholds
    rows = []
    for thr in (5, 7, 9, 11):
        keep = indexed & (d["n_components_observed"] >= thr)
        r = rank_agg(d, COMP, keep)
        both = keep & stored.notna()
        rows.append({"threshold": f">={thr}", "countries": int(keep.sum()),
                     "spearman_vs_baseline": round(float(
                         pd.concat([r[both], stored[both]], axis=1).corr(method="spearman").iloc[0, 1]), 4)})
    cov = pd.DataFrame(rows)

    out = HERE / "need_index_scale_excluded.parquet"
    d.to_parquet(out, index=False); lg.add_output(out, rows=len(d))
    movers.to_csv(HERE / "need_index_top_movers.csv", index=False)
    cov.to_csv(HERE / "need_coverage_sensitivity.csv", index=False)
    lg.add_output(HERE / "need_coverage_sensitivity.csv", rows=len(cov))
    print(f"spearman 11-comp vs 9-comp need: {sp:.4f}")
    print(f"spearman mismatch11 vs mismatch9: {gcorr:.4f}")
    print(movers.to_string(index=False))
    print(cov.to_string(index=False))
    lg.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
