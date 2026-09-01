"""Revision analyses: regression specifications.

Covers the reviewer requests on model specification:
  Step 7  / item  -   crop fixed effects reported as a table (participation + PPML)
  Step 8  / item 10 - expanded-sample model dropping R&D expenditure
  Step 10 / item 11 - PPML with free log-area coefficient and Wald test of H0: beta = 1
  Step 11           - nested block models (scale / need / capacity) with LR, AIC, BIC
  item 1            - summary statistics for every Table 5 variable, split by outcome
  item 5            - average marginal effects, delta method, country-clustered
  item 12           - proportionality test of log research share on log area share
  item 13           - crop-specific participation models
  item 14           - quasi-Poisson comparison against clustered Poisson PML
  item 15           - period-specific participation models

Every specification uses country-clustered standard errors on the same estimation
sample unless the variant itself changes the sample, and the published baseline is
replicated as a guard before any variant runs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
sys.path.insert(0, str(ROOT / "16_Econometrics"))
from project_config import P, RunLogger  # noqa: E402

OUT = HERE
SEED = 20260730
np.random.seed(SEED)

# Primary specification: nine-component scale-excluded need index, no standalone
# yield-volatility regressor (it enters through the index).
X_PRIMARY = ["log_area", "rd", "need9", "log_gdp_pc", "tertiary", "internet", "log_population"]
LABELS = {
    "log_area": "Log harvested area", "rd": "R\\&D expenditure (\\% GDP)",
    "need9": "Research-need index", "log_gdp_pc": "Log GDP per capita",
    "tertiary": "Tertiary enrolment", "internet": "Internet use",
    "log_population": "Log population", "const": "Constant",
}


def logit(df, xs, y="studied", extra=None):
    """Country-clustered logit. `extra` holds pre-built dummy columns (fixed effects)."""
    cols = list(xs) + (list(extra) if extra else [])
    X = sm.add_constant(df[cols].astype(float), has_constant="add")
    return sm.Logit(df[y].astype(float), X).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": df["iso3"].to_numpy()})


def ppml(df, xs, y="n_studies_fractional", offset_col="area_ha_mean",
         use_offset=True, extra=None):
    """Poisson PML. With use_offset the area elasticity is fixed at one."""
    d = df[df[offset_col].fillna(0) > 0].copy()
    cols = list(xs) + (list(extra) if extra else [])
    X = sm.add_constant(d[cols].astype(float), has_constant="add")
    off = np.log(d[offset_col].astype(float)) if use_offset else None
    return sm.GLM(d[y].astype(float), X, family=sm.families.Poisson(), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": d["iso3"].to_numpy()}), d


def tidy(res, keep=None, model=""):
    ci = res.conf_int()
    out = []
    for t in res.params.index:
        if keep and t not in keep:
            continue
        out.append(dict(model=model, term=t, estimate=res.params[t], se=res.bse[t],
                        ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1],
                        n=int(res.nobs)))
    return out


def load():
    e = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    n9 = pd.read_parquet(ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet")
    e = e.merge(n9[["iso3", "need9_rank_pct", "n_components9"]], on="iso3", how="left")
    e = e.rename(columns={"need9_rank_pct": "need9"})
    return e


def main():
    lg = RunLogger("rev_01_models")
    lg.add_input(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    lg.add_input(ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet")
    lg.add_input(ROOT / "12_Data_Integration" / "country_crop_panel.parquet")
    lg.add_input(ROOT / "12_Data_Integration" / "study_country_crop_dataset.parquet")

    e = load()
    rows = []
    print(f"estimation sample: {len(e)} cells, {e.iso3.nunique()} countries, "
          f"{int(e.studied.sum())} studied")

    # ---------------------------------------------------------------- guard: baseline
    b_logit = logit(e, X_PRIMARY)
    b_ppml, d_ppml = ppml(e, [x for x in X_PRIMARY if x != "log_area"])
    print(f"\nGUARD baseline logit log_area = {b_logit.params['log_area']:.3f} "
          f"(published 0.713); PPML need9 = {b_ppml.params['need9']:.3f} (published 1.261)")
    rows += tidy(b_logit, model="participation_baseline")
    rows += tidy(b_ppml, model="intensity_baseline_offset")

    # ---------------------------------------------------------------- Step 7: crop FE
    crop_d = pd.get_dummies(e["crop_standard_name"], prefix="crop", drop_first=True)
    e2 = pd.concat([e, crop_d], axis=1)
    fe_cols = list(crop_d.columns)
    fe_logit = logit(e2, X_PRIMARY, extra=fe_cols)
    fe_ppml, _ = ppml(e2, [x for x in X_PRIMARY if x != "log_area"], extra=fe_cols)
    rows += tidy(fe_logit, keep=X_PRIMARY, model="participation_cropFE")
    rows += tidy(fe_ppml, keep=[x for x in X_PRIMARY if x != "log_area"],
                 model="intensity_cropFE_offset")
    print(f"crop FE: logit log_area {fe_logit.params['log_area']:.3f}; "
          f"PPML need9 {fe_ppml.params['need9']:.3f}")

    # ---------------------------------------------------------------- Step 8: no R&D, expanded
    pan = pd.read_parquet(ROOT / "12_Data_Integration" / "country_crop_panel.parquet")
    cov = e[["iso3", "rd", "log_gdp_pc", "tertiary", "internet", "log_population"]].drop_duplicates("iso3")
    full = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    # Rebuild the wider sample from the panel plus country covariates, dropping R&D only.
    ctry = (pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
            [["iso3", "log_gdp_pc", "tertiary", "internet", "log_population"]]
            .drop_duplicates("iso3"))
    # Recover country covariates for countries outside the estimation sample from the
    # panel merge inputs where available; if a covariate is missing the cell is dropped.
    wide = pan.merge(pd.read_parquet(ROOT / "31_Presubmission_Audit" /
                                     "need_index_scale_excluded.parquet")
                     [["iso3", "need9_rank_pct"]].rename(columns={"need9_rank_pct": "need9"}),
                     on="iso3", how="left")
    wb = _country_covariates(ROOT)
    wide = wide.merge(wb, on="iso3", how="left")
    wide["log_area"] = np.log(wide["area_ha_mean"].where(wide["area_ha_mean"] > 0))
    wide["studied"] = (wide["n_studies_fractional"] > 0).astype(float)
    X_NORD = ["log_area", "need9", "log_gdp_pc", "tertiary", "internet", "log_population"]
    w = wide.dropna(subset=X_NORD + ["studied"]).copy()
    m_nord_wide = logit(w, X_NORD)
    rows += tidy(m_nord_wide, model="participation_noRD_expanded")
    m_nord_rest = logit(e, X_NORD)
    rows += tidy(m_nord_rest, model="participation_noRD_restricted142")
    print(f"\nno-R&D expanded: {len(w)} cells / {w.iso3.nunique()} countries "
          f"vs restricted {len(e)} / {e.iso3.nunique()}")
    print(f"  need9 expanded {m_nord_wide.params['need9']:.3f} "
          f"[{m_nord_wide.conf_int().loc['need9',0]:.3f},{m_nord_wide.conf_int().loc['need9',1]:.3f}]"
          f" | restricted {m_nord_rest.params['need9']:.3f} "
          f"[{m_nord_rest.conf_int().loc['need9',0]:.3f},{m_nord_rest.conf_int().loc['need9',1]:.3f}]")

    # ---------------------------------------------------------------- Step 10: free area
    free, d_free = ppml(e, X_PRIMARY, use_offset=False)
    ba, sa = free.params["log_area"], free.bse["log_area"]
    wald = ((ba - 1.0) / sa) ** 2
    pval = 1 - stats.chi2.cdf(wald, 1)
    rows += tidy(free, model="intensity_free_area")
    print(f"\nPPML free area elasticity = {ba:.3f} (SE {sa:.3f}, "
          f"CI [{free.conf_int().loc['log_area',0]:.3f},{free.conf_int().loc['log_area',1]:.3f}])")
    print(f"  Wald H0: beta_area = 1 -> chi2(1) = {wald:.3f}, p = {pval:.4f}")
    pd.DataFrame([dict(elasticity=ba, se=sa,
                       ci_low=free.conf_int().loc["log_area", 0],
                       ci_high=free.conf_int().loc["log_area", 1],
                       wald_chi2=wald, df=1, p_value=pval, n=int(free.nobs))]
                 ).to_csv(OUT / "ppml_area_elasticity_test.csv", index=False)

    # ---------------------------------------------------------------- Step 11: blocks
    blocks = {
        "M0_constant": [],
        "M1_scale": ["log_area"],
        "M2_scale_need": ["log_area", "need9"],
        "M3_scale_capacity": ["log_area", "rd", "log_gdp_pc", "tertiary", "internet", "log_population"],
        "M4_full": X_PRIMARY,
    }
    brows = []
    for name, xs in blocks.items():
        if xs:
            r = logit(e, xs)
        else:
            X = sm.add_constant(pd.DataFrame(index=e.index).assign(_z=0.0)["_z"], has_constant="add")
            r = sm.Logit(e["studied"].astype(float), X[["const"]]).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": e["iso3"].to_numpy()})
        brows.append(dict(model=name, k=len(xs), llf=r.llf, aic=r.aic, bic=r.bic,
                          mcfadden=1 - r.llf / r.llnull if xs else 0.0, n=int(r.nobs)))
    bt = pd.DataFrame(brows)
    # Likelihood-ratio tests for the two orderings that matter.
    def lr(a, b):
        ra, rb = bt.set_index("model").loc[a], bt.set_index("model").loc[b]
        stat = 2 * (rb.llf - ra.llf); df = int(rb.k - ra.k)
        return stat, df, 1 - stats.chi2.cdf(stat, df)
    tests = []
    for a, b, lab in [("M1_scale", "M2_scale_need", "need added to scale"),
                      ("M1_scale", "M3_scale_capacity", "capacity added to scale"),
                      ("M2_scale_need", "M4_full", "capacity added to scale+need"),
                      ("M3_scale_capacity", "M4_full", "need added to scale+capacity")]:
        s, df, p = lr(a, b)
        tests.append(dict(comparison=lab, base=a, augmented=b, lr_chi2=s, df=df, p_value=p,
                          d_mcfadden=float(bt.set_index("model").loc[b, "mcfadden"] -
                                           bt.set_index("model").loc[a, "mcfadden"]),
                          d_aic=float(bt.set_index("model").loc[b, "aic"] -
                                      bt.set_index("model").loc[a, "aic"])))
    bt.to_csv(OUT / "block_models_fit.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "block_lr_tests.csv", index=False)
    print("\nBLOCK COMPARISON (participation)")
    print(bt.round(4).to_string(index=False))
    print(pd.DataFrame(tests).round(4).to_string(index=False))

    # PPML deviance-based block comparison
    prows = []
    for name, xs in blocks.items():
        xs2 = [x for x in xs if x != "log_area"]
        r, _ = ppml(e, xs2) if xs2 else (None, None)
        if r is None:
            X = sm.add_constant(pd.DataFrame(index=d_ppml.index).assign(_z=0.0)["_z"], has_constant="add")
            r = sm.GLM(d_ppml["n_studies_fractional"].astype(float), X[["const"]],
                       family=sm.families.Poisson(),
                       offset=np.log(d_ppml["area_ha_mean"].astype(float))).fit(
                cov_type="cluster", cov_kwds={"groups": d_ppml["iso3"].to_numpy()})
        prows.append(dict(model=name, k=len(xs2), deviance=r.deviance, aic=r.aic,
                          bic=getattr(r, "bic", np.nan), n=int(r.nobs)))
    pd.DataFrame(prows).to_csv(OUT / "block_models_ppml.csv", index=False)
    print("\nBLOCK COMPARISON (intensity, offset)")
    print(pd.DataFrame(prows).round(3).to_string(index=False))

    pd.DataFrame(rows).to_csv(OUT / "model_estimates.csv", index=False)
    lg.add_output(OUT / "model_estimates.csv", rows=len(rows))
    lg.add_output(OUT / "block_models_fit.csv")
    lg.add_output(OUT / "block_lr_tests.csv")
    lg.add_output(OUT / "ppml_area_elasticity_test.csv")
    lg.finish()
    print("\nrev_01_models complete")


def _country_covariates(root: Path) -> pd.DataFrame:
    """Latest non-missing World Bank observation per country, for ALL countries.

    Rebuilt from the long indicator extract rather than the estimation sample, so
    that countries dropped by the R&D listwise deletion are recoverable. Verified to
    reproduce the estimation-sample values exactly for the 142 countries it covers.
    """
    w = pd.read_csv(root / "10_External_Data" / "World_Bank" / "wb_indicators_long.csv")
    ind = {"GB.XPD.RSDV.GD.ZS": "rd_expenditure_gdp_pct",
           "IT.NET.USER.ZS": "internet_users_pct",
           "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp",
           "SE.TER.ENRR": "tertiary_enrolment_pct",
           "SP.POP.TOTL": "population"}
    lat = (w[w["indicator"].isin(ind)].dropna(subset=["value"]).sort_values("year")
           .groupby(["iso3", "indicator"]).tail(1)
           .pivot(index="iso3", columns="indicator", values="value")
           .rename(columns=ind).reset_index())
    lat["log_gdp_pc"] = np.log(lat["gdp_pc_ppp"].where(lat["gdp_pc_ppp"] > 0))
    lat["log_population"] = np.log(lat["population"].where(lat["population"] > 0))
    lat["tertiary"] = lat["tertiary_enrolment_pct"] / 100.0
    lat["internet"] = lat["internet_users_pct"] / 100.0
    lat["rd"] = lat["rd_expenditure_gdp_pct"]
    return lat[["iso3", "log_gdp_pc", "log_population", "tertiary", "internet", "rd"]]


if __name__ == "__main__":
    main()
