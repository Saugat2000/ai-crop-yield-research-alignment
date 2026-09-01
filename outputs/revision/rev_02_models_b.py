"""Revision analyses, part 2: descriptive, marginal-effect, and heterogeneity models.

  item  1 - summary statistics for Table 5 variables, split by outcome
  item  5 - average marginal effects (delta method, country-clustered)
  item 12 - proportionality test: log research share on log area share
  item 13 - crop-specific participation models
  item 14 - quasi-Poisson comparison, and dispersion-scaled block tests for PPML
  item 15 - period-specific participation models
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
from project_config import P, RunLogger  # noqa: E402

OUT = HERE
np.random.seed(20260730)
X_PRIMARY = ["log_area", "rd", "need9", "log_gdp_pc", "tertiary", "internet", "log_population"]
PRETTY = {"log_area": "Log harvested area", "rd": "R&D expenditure (% GDP)",
          "need9": "Research-need index (0-1)", "log_gdp_pc": "Log GDP per capita",
          "tertiary": "Tertiary enrolment (share)", "internet": "Internet use (share)",
          "log_population": "Log population", "studied": "Cell carries a study",
          "n_studies_fractional": "Fractional study count"}


def load():
    e = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    n9 = pd.read_parquet(ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet")
    return e.merge(n9[["iso3", "need9_rank_pct"]].rename(columns={"need9_rank_pct": "need9"}),
                   on="iso3", how="left")


def logit(df, xs, y="studied", extra=None):
    cols = list(xs) + (list(extra) if extra else [])
    X = sm.add_constant(df[cols].astype(float), has_constant="add")
    return sm.Logit(df[y].astype(float), X).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": df["iso3"].to_numpy()})


def main():
    lg = RunLogger("rev_02_models_b")
    lg.add_input(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    e = load()

    # ------------------------------------------------------- item 1: summary statistics
    srows = []
    for v in X_PRIMARY + ["n_studies_fractional"]:
        for lab, sub in [("All cells", e), ("Cells with research (y=1)", e[e.studied == 1]),
                         ("Cells without research (y=0)", e[e.studied == 0])]:
            x = sub[v].astype(float)
            srows.append(dict(variable=PRETTY.get(v, v), sample=lab, n=int(x.notna().sum()),
                              mean=x.mean(), sd=x.std(), median=x.median(),
                              minimum=x.min(), maximum=x.max()))
    summ = pd.DataFrame(srows)
    summ.to_csv(OUT / "summary_statistics.csv", index=False)
    print("SUMMARY STATISTICS (all cells)")
    print(summ[summ["sample"] == "All cells"].round(3).to_string(index=False))

    # ------------------------------------------------------- item 5: average marginal effects
    m = logit(e, X_PRIMARY)
    me = m.get_margeff(at="overall", method="dydx")
    ame = pd.DataFrame({"term": [t for t in m.params.index if t != "const"],
                        "ame": me.margeff, "se": me.margeff_se})
    ame["ci_low"] = ame.ame - 1.959964 * ame.se
    ame["ci_high"] = ame.ame + 1.959964 * ame.se
    ame["z"] = ame.ame / ame.se
    ame["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(ame.z)))
    # Interquartile-range effect on the same scale, for cross-covariate comparison.
    iqr = []
    for t in X_PRIMARY:
        q1, q3 = e[t].quantile(.25), e[t].quantile(.75)
        d = e.copy()
        base = m.predict(sm.add_constant(d[X_PRIMARY].astype(float), has_constant="add")).mean()
        d1 = d.copy(); d1[t] = q1
        d3 = d.copy(); d3[t] = q3
        p1 = m.predict(sm.add_constant(d1[X_PRIMARY].astype(float), has_constant="add")).mean()
        p3 = m.predict(sm.add_constant(d3[X_PRIMARY].astype(float), has_constant="add")).mean()
        iqr.append(dict(term=t, label=PRETTY[t], p25=q1, p75=q3, iqr_dprob=p3 - p1))
    ame = ame.merge(pd.DataFrame(iqr), on="term")
    ame.to_csv(OUT / "average_marginal_effects.csv", index=False)
    print("\nAVERAGE MARGINAL EFFECTS (delta method, clustered by country)")
    print(ame[["label", "ame", "se", "ci_low", "ci_high", "p_value", "iqr_dprob"]]
          .round(4).to_string(index=False))

    # ------------------------------------------------------- item 12: proportionality
    pan = pd.read_parquet(ROOT / "12_Data_Integration" / "country_crop_panel.parquet")
    pos = pan[(pan.n_studies_fractional > 0) & (pan.area_ha_mean > 0)].copy()
    pos["rshare"] = pos.n_studies_fractional / pan.n_studies_fractional.sum()
    pos["ashare"] = pos.area_ha_mean / pan.area_ha_mean.sum()
    X = sm.add_constant(np.log(pos.ashare))
    pr = sm.OLS(np.log(pos.rshare), X).fit(cov_type="cluster",
                                           cov_kwds={"groups": pos.iso3.to_numpy()})
    b, se = pr.params.iloc[1], pr.bse.iloc[1]
    w = ((b - 1) / se) ** 2
    prop = dict(slope=b, se=se, ci_low=pr.conf_int().iloc[1, 0], ci_high=pr.conf_int().iloc[1, 1],
                wald_chi2=w, p_value=1 - stats.chi2.cdf(w, 1), n=int(pr.nobs), r2=pr.rsquared)
    pd.DataFrame([prop]).to_csv(OUT / "proportionality_test.csv", index=False)
    print(f"\nPROPORTIONALITY: slope = {b:.3f} (SE {se:.3f}, CI [{prop['ci_low']:.3f},"
          f"{prop['ci_high']:.3f}]); H0 slope=1 -> chi2 {w:.2f}, p = {prop['p_value']:.4f}, n={prop['n']}")

    # ------------------------------------------------------- item 13: crop-specific
    crows = []
    for crop in ["wheat", "maize", "rice", "soybean", "sugarcane"]:
        s = e[e.crop_standard_name == crop]
        if s.studied.sum() < 12 or s.studied.nunique() < 2:
            print(f"  {crop}: skipped (events={int(s.studied.sum())})")
            continue
        try:
            r = logit(s, X_PRIMARY)
            ci = r.conf_int()
            for t in ["log_area", "need9"]:
                crows.append(dict(crop=crop, term=t, estimate=r.params[t], se=r.bse[t],
                                  ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1],
                                  n=int(r.nobs), events=int(s.studied.sum())))
        except Exception as ex:
            print(f"  {crop}: failed ({type(ex).__name__})")
    cs = pd.DataFrame(crows)
    cs.to_csv(OUT / "crop_specific_models.csv", index=False)
    print("\nCROP-SPECIFIC PARTICIPATION MODELS")
    print(cs.round(3).to_string(index=False))

    # ------------------------------------------------------- item 14: quasi-Poisson
    d = e[e.area_ha_mean.fillna(0) > 0].copy()
    Xp = sm.add_constant(d[[x for x in X_PRIMARY if x != "log_area"]].astype(float),
                         has_constant="add")
    off = np.log(d.area_ha_mean.astype(float))
    base = sm.GLM(d.n_studies_fractional.astype(float), Xp,
                  family=sm.families.Poisson(), offset=off)
    clust = base.fit(cov_type="cluster", cov_kwds={"groups": d.iso3.to_numpy()})
    naive = base.fit()
    phi = float(naive.pearson_chi2 / naive.df_resid)
    qse = naive.bse * np.sqrt(phi)
    qp = pd.DataFrame({"term": clust.params.index, "estimate": clust.params.values,
                       "se_clustered": clust.bse.values, "se_naive_poisson": naive.bse.values,
                       "se_quasipoisson": qse.values})
    qp["ratio_clustered_to_quasi"] = qp.se_clustered / qp.se_quasipoisson
    qp.to_csv(OUT / "quasipoisson_comparison.csv", index=False)
    print(f"\nQUASI-POISSON (dispersion phi = {phi:.2f})")
    print(qp.round(4).to_string(index=False))

    # Dispersion-scaled block tests for the intensity model (deviance LR is invalid
    # under this much overdispersion, so it is scaled by phi and read as an F test).
    blocks = {"M0": [], "M2_need": ["need9"],
              "M3_capacity": ["rd", "log_gdp_pc", "tertiary", "internet", "log_population"],
              "M4_full": [x for x in X_PRIMARY if x != "log_area"]}
    fit = {}
    for nm, xs in blocks.items():
        Xb = sm.add_constant(d[xs].astype(float), has_constant="add") if xs else \
            pd.DataFrame({"const": np.ones(len(d))}, index=d.index)
        r = sm.GLM(d.n_studies_fractional.astype(float), Xb,
                   family=sm.families.Poisson(), offset=off).fit()
        fit[nm] = (r.deviance, r.df_resid, len(xs))
    frows = []
    for a, b_, lab in [("M3_capacity", "M4_full", "need added to capacity"),
                       ("M2_need", "M4_full", "capacity added to need")]:
        dd = fit[a][0] - fit[b_][0]; ddf = fit[b_][2] - fit[a][2]
        F = (dd / ddf) / phi
        frows.append(dict(comparison=lab, d_deviance=dd, df=ddf, phi=phi,
                          F=F, p_value=1 - stats.f.cdf(F, ddf, fit[b_][1]),
                          naive_lr_p=1 - stats.chi2.cdf(dd, ddf)))
    fb = pd.DataFrame(frows)
    fb.to_csv(OUT / "block_ppml_scaled_tests.csv", index=False)
    print("\nPPML BLOCK TESTS, SCALED BY DISPERSION")
    print(fb.round(4).to_string(index=False))

    # ------------------------------------------------------- item 15: period-specific
    scc = pd.read_parquet(ROOT / "12_Data_Integration" / "study_country_crop_dataset.parquet")
    prows = []
    for lab, lo, hi in [("2000-2019", 2000, 2019), ("2020-2026", 2020, 2026)]:
        sub = scc[(scc.publication_year >= lo) & (scc.publication_year <= hi)]
        agg = (sub.groupby(["iso3", "crop_standard_name"], as_index=False)
               .fractional_weight.sum().rename(columns={"fractional_weight": "r_period"}))
        ep = e.merge(agg, on=["iso3", "crop_standard_name"], how="left")
        ep["studied_p"] = (ep.r_period.fillna(0) > 0).astype(float)
        r = logit(ep, X_PRIMARY, y="studied_p")
        ci = r.conf_int()
        for t in X_PRIMARY:
            prows.append(dict(period=lab, term=t, estimate=r.params[t], se=r.bse[t],
                              ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1],
                              n=int(r.nobs), events=int(ep.studied_p.sum())))
    ps = pd.DataFrame(prows)
    ps.to_csv(OUT / "period_specific_models.csv", index=False)
    print("\nPERIOD-SPECIFIC PARTICIPATION MODELS")
    print(ps[ps.term.isin(["log_area", "need9", "rd"])].round(3).to_string(index=False))

    for f in ["summary_statistics.csv", "average_marginal_effects.csv", "proportionality_test.csv",
              "crop_specific_models.csv", "quasipoisson_comparison.csv",
              "block_ppml_scaled_tests.csv", "period_specific_models.csv"]:
        lg.add_output(OUT / f)
    lg.finish()
    print("\nrev_02_models_b complete")


if __name__ == "__main__":
    main()
