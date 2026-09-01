"""Revision analyses: research-need index diagnostics and the corrected coverage floor.

  Step 12 / item 3 - component availability, correlations, reliability, index-variant
                     correlations, PCA loadings, coverage-threshold sensitivity
  R1-M3            - the published coverage floor counts ELEVEN recorded indicators,
                     including the two agricultural-scale shares that are never missing.
                     Five countries therefore receive a nine-component index on fewer
                     than five observed need components. This rebuilds the index with
                     the floor applied to the nine primary components and reruns the
                     downstream models on the corrected index.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

OUT = HERE
NINE = ["undernourishment_pct", "food_insecurity_mod_sev_pct", "cereal_import_dependency_pct",
        "dietary_energy_adequacy_pct", "employment_agriculture_share_pct",
        "agri_value_added_share_gdp_pct", "agri_value_added_per_worker_usd",
        "temp_warming_since_baseline_c", "area_weighted_yield_volatility"]
# Direction: True where a HIGHER raw value means GREATER need.
DIRECTION = {"undernourishment_pct": True, "food_insecurity_mod_sev_pct": True,
             "cereal_import_dependency_pct": True, "dietary_energy_adequacy_pct": False,
             "employment_agriculture_share_pct": True, "agri_value_added_share_gdp_pct": True,
             "agri_value_added_per_worker_usd": False, "temp_warming_since_baseline_c": True,
             "area_weighted_yield_volatility": True}
SHORT = {"undernourishment_pct": "Undernourishment", "food_insecurity_mod_sev_pct": "Food insecurity",
         "cereal_import_dependency_pct": "Cereal import dep.", "dietary_energy_adequacy_pct": "Dietary adequacy",
         "employment_agriculture_share_pct": "Agri. employment", "agri_value_added_share_gdp_pct": "Agri. value added",
         "agri_value_added_per_worker_usd": "Value added/worker", "temp_warming_since_baseline_c": "Temp. change",
         "area_weighted_yield_volatility": "Yield volatility"}
X_PRIMARY = ["log_area", "rd", "need9", "log_gdp_pc", "tertiary", "internet", "log_population"]


def harmonised_ranks(df):
    """Direction-harmonised percentile ranks, exactly as the published index builds them."""
    r = pd.DataFrame(index=df.index)
    for c in NINE:
        pr = df[c].rank(pct=True)
        r[c] = pr if DIRECTION[c] else 1.0 - pr
    return r


def build_index(df, floor, cols=NINE):
    """Mean of observed harmonised ranks where coverage meets the floor, re-percentiled."""
    r = harmonised_ranks(df)
    nobs = r[cols].notna().sum(axis=1)
    raw = r[cols].mean(axis=1, skipna=True).where(nobs >= floor)
    return raw.rank(pct=True), nobs


def cronbach(x: pd.DataFrame) -> float:
    x = x.dropna()
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1))


def main():
    lg = RunLogger("rev_03_needindex")
    src = ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet"
    lg.add_input(src)
    n = pd.read_parquet(src)

    # ------------------------------------------------------- coverage
    r = harmonised_ranks(n)
    n["obs9"] = r[NINE].notna().sum(axis=1)
    cov = (n.obs9.value_counts().sort_index().rename_axis("components_observed")
           .reset_index(name="countries"))
    cov["cumulative_at_or_above"] = cov.countries[::-1].cumsum()[::-1]
    cov.to_csv(OUT / "need_component_coverage.csv", index=False)
    print("COMPONENT COVERAGE (nine primary need components)")
    print(cov.to_string(index=False))
    print(f"\npublished floor uses n_components_observed (ELEVEN recorded indicators): "
          f"{(n.n_components_observed >= 5).sum()} countries indexed")
    print(f"floor applied to the NINE primary components:            "
          f"{(n.obs9 >= 5).sum()} countries")
    mism = n[(n.n_components_observed >= 5) & (n.obs9 < 5)]
    print(f"countries indexed on fewer than five NEED components:    {len(mism)}")
    print(mism[["iso3", "n_components_observed", "obs9", "need9_rank_pct"]].to_string(index=False))
    mism.to_csv(OUT / "need_floor_affected_countries.csv", index=False)

    # ------------------------------------------------------- correlations + reliability
    rr = r[NINE].rename(columns=SHORT)
    pear = rr.corr(method="pearson"); spear = rr.corr(method="spearman")
    pear.to_csv(OUT / "need_component_corr_pearson.csv")
    spear.to_csv(OUT / "need_component_corr_spearman.csv")
    alpha = cronbach(rr)
    print(f"\nCronbach's alpha over the nine harmonised component ranks: {alpha:.3f}")
    off = spear.where(~np.eye(len(spear), dtype=bool))
    print(f"Spearman off-diagonal: mean {off.stack().mean():.3f}, "
          f"min {off.stack().min():.3f}, max {off.stack().max():.3f}")

    # ------------------------------------------------------- index-variant correlations
    variants = {"rank (primary, 9-comp)": n.need9_rank_pct,
                "rank (11-comp)": n.need_rank_pct, "equal weight (11)": n.need_equal_pct,
                "entropy (11)": n.need_entropy_pct, "PCA (11)": n.need_pca_pct}
    V = pd.DataFrame(variants)
    vs = V.corr(method="spearman")
    vs.to_csv(OUT / "need_index_variant_correlations.csv")
    print("\nINDEX-VARIANT SPEARMAN CORRELATIONS")
    print(vs.round(3).to_string())

    # ------------------------------------------------------- PCA loadings
    z = rr.dropna()
    zs = (z - z.mean()) / z.std(ddof=0)
    u, s, vt = np.linalg.svd(zs.values, full_matrices=False)
    load = pd.DataFrame({"component": rr.columns, "PC1_loading": vt[0],
                         "PC2_loading": vt[1]})
    load["PC1_var_explained"] = (s[0] ** 2) / (s ** 2).sum()
    load.to_csv(OUT / "need_pca_loadings.csv", index=False)
    print(f"\nPCA on complete cases (n={len(z)}): PC1 explains "
          f"{load.PC1_var_explained.iloc[0]:.3f} of variance")
    print(load[["component", "PC1_loading", "PC2_loading"]].round(3).to_string(index=False))

    # ------------------------------------------------------- corrected floor + reruns
    idx5, nobs = build_index(n, floor=5)
    print(f"\nreplication guard: corrected-floor index reproduces published ranking on the "
          f"shared sample, Spearman = "
          f"{pd.concat([idx5, n.need9_rank_pct], axis=1).dropna().corr(method='spearman').iloc[0,1]:.4f}")
    n["need9_floor9"] = idx5

    thr_rows = []
    e = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    for floor in [5, 7, 9]:
        idx, _ = build_index(n, floor=floor)
        tmp = n[["iso3"]].copy(); tmp["need_v"] = idx
        ee = e.merge(tmp, on="iso3", how="left").dropna(subset=["need_v"])
        X = sm.add_constant(ee[["log_area", "rd", "need_v", "log_gdp_pc", "tertiary",
                                "internet", "log_population"]].astype(float), has_constant="add")
        m = sm.Logit(ee.studied.astype(float), X).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": ee.iso3.to_numpy()})
        ci = m.conf_int()
        thr_rows.append(dict(floor=floor, countries_indexed=int(idx.notna().sum()),
                             cells=int(m.nobs), area=m.params["log_area"],
                             need=m.params["need_v"], need_lo=ci.loc["need_v", 0],
                             need_hi=ci.loc["need_v", 1], rd=m.params["rd"]))
    th = pd.DataFrame(thr_rows)
    th.to_csv(OUT / "need_floor_sensitivity.csv", index=False)
    print("\nCOVERAGE-FLOOR SENSITIVITY (floor applied to the nine primary components)")
    print(th.round(3).to_string(index=False))

    n[["iso3", "need9_rank_pct", "need9_floor9", "obs9", "n_components_observed"]].to_parquet(
        OUT / "need_index_corrected_floor.parquet", index=False)
    for f in ["need_component_coverage.csv", "need_component_corr_pearson.csv",
              "need_component_corr_spearman.csv", "need_index_variant_correlations.csv",
              "need_pca_loadings.csv", "need_floor_sensitivity.csv",
              "need_floor_affected_countries.csv", "need_index_corrected_floor.parquet"]:
        lg.add_output(OUT / f)
    lg.finish()
    print("\nrev_03_needindex complete")


if __name__ == "__main__":
    main()
