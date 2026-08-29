"""Pre-submission audit M2/M4/M5/Mo1/Mo2/XXIV: model specifications.

Every specification is estimated with country-clustered standard errors on the same
estimation sample unless the variant itself changes the sample; nothing is dropped
silently and the baseline is replicated as a guard before any variant runs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_Project_Management"))
sys.path.insert(0, str(HERE.parent / "16_Econometrics"))
from project_config import P, RunLogger  # noqa: E402

X_MAIN = ["log_area", "log_gdp_pc", "rd", "tertiary", "internet",
          "need", "yield_volatility", "log_population"]


def logit(df, xs, y="studied"):
    X = sm.add_constant(df[xs].astype(float))
    m = sm.Logit(df[y].astype(float), X)
    return m.fit(disp=0, cov_type="cluster", cov_kwds={"groups": df["iso3"].to_numpy()})


def ppml(df, xs, y="n_studies_fractional", offset_col="area_ha_mean"):
    d = df[df[offset_col].fillna(0) > 0]
    X = sm.add_constant(d[xs].astype(float))
    m = sm.GLM(d[y].astype(float), X, family=sm.families.Poisson(),
               offset=np.log(d[offset_col].astype(float)))
    return m.fit(cov_type="cluster", cov_kwds={"groups": d["iso3"].to_numpy()}), d


def row(res, term, model, n):
    b = res.params.get(term, np.nan); se = res.bse.get(term, np.nan)
    return {"model": model, "term": term, "estimate": round(float(b), 3),
            "se": round(float(se), 3),
            "ci_low": round(float(b - 1.96 * se), 3), "ci_high": round(float(b + 1.96 * se), 3),
            "n": int(n)}


def main() -> int:
    lg = RunLogger("audit_02_models")
    est = pd.read_parquet(P["econ"] / "estimation_sample.parquet")
    lg.add_input(P["econ"] / "estimation_sample.parquet")
    need9 = pd.read_parquet(HERE / "need_index_scale_excluded.parquet")[
        ["iso3", "need9_rank_pct", "n_components_observed"]]
    est = est.merge(need9, on="iso3", how="left")
    est["need9"] = pd.to_numeric(est["need9_rank_pct"], errors="coerce")

    out = []

    # --- guard: replicate the reported baseline ------------------------------------
    b = logit(est, X_MAIN)
    ref = pd.read_csv(P["econ"] / "model_coefficients.csv")
    ref_area = float(ref[(ref.model == "participation_logit") & (ref.term == "log_area")]["estimate"].iloc[0])
    if abs(float(b.params["log_area"]) - ref_area) > 5e-4:
        raise ValueError("baseline logit does not replicate reported coefficients")
    lg.count("baseline_replicated", 1)
    for t in X_MAIN: out.append(row(b, t, "baseline_logit", b.nobs))

    # --- M4: crop fixed effects ----------------------------------------------------
    cd = pd.get_dummies(est["crop_standard_name"], prefix="crop", drop_first=True).astype(float)
    estF = pd.concat([est, cd], axis=1)
    xs_fe = X_MAIN + list(cd.columns)
    bfe = logit(estF, xs_fe)
    for t in X_MAIN: out.append(row(bfe, t, "logit_crop_FE", bfe.nobs))
    pfe, dfe = ppml(estF, [x for x in xs_fe if x != "log_area"])
    for t in ["rd", "need", "internet", "tertiary", "log_population", "log_gdp_pc", "yield_volatility"]:
        out.append(row(pfe, t, "ppml_crop_FE", len(dfe)))

    # region FE (secondary)
    rd_ = pd.get_dummies(est["wb_region"], prefix="reg", drop_first=True).astype(float)
    estR = pd.concat([est, rd_], axis=1)
    brf = logit(estR, X_MAIN + list(rd_.columns))
    for t in X_MAIN: out.append(row(brf, t, "logit_region_FE", brf.nobs))

    # --- M1: scale-excluded need index ---------------------------------------------
    xs9 = ["log_area", "log_gdp_pc", "rd", "tertiary", "internet", "need9", "log_population"]
    e9 = est.dropna(subset=["need9"])
    b9 = logit(e9, xs9)
    for t in xs9: out.append(row(b9, t, "logit_need9_novol", b9.nobs))
    p9, d9 = ppml(e9, [x for x in xs9 if x != "log_area"])
    for t in ["rd", "need9", "internet", "tertiary", "log_population"]:
        out.append(row(p9, t, "ppml_need9_novol", len(d9)))
    b9v = logit(e9, xs9 + ["yield_volatility"])           # volatility kept, for comparison
    out.append(row(b9v, "need9", "logit_need9_withvol", b9v.nobs))
    bfe9 = logit(pd.concat([e9, cd.loc[e9.index]], axis=1), xs9 + list(cd.columns))
    for t in ["log_area", "rd", "need9"]:
        out.append(row(bfe9, t, "logit_need9_crop_FE", bfe9.nobs))

    # --- Mo1: coverage thresholds on the need coefficient ---------------------------
    for thr in (7, 9):
        sub = est[est["n_components_observed"] >= thr]
        bt = logit(sub, X_MAIN)
        out.append(row(bt, "need", f"logit_needcov_ge{thr}", bt.nobs))

    # --- M2: allocation-scope sensitivity ------------------------------------------
    sl = pd.read_parquet(P["integration"] / "study_level_dataset.parquet")[
        ["openalex_id", "loc_study_scope"]]
    scc = pd.read_parquet(P["integration"] / "study_country_crop_dataset.parquet").merge(
        sl, on="openalex_id", how="left")
    for tag, keep in (("strict", ["single_country", "multi_country"]),
                      ("noglobal", ["single_country", "multi_country", "unresolved"])):
        s2 = scc[scc.loc_study_scope.isin(keep)]
        cnt = s2.groupby(["iso3", "crop_standard_name"])["fractional_weight"].sum()
        e2 = est.copy()
        e2["nf2"] = e2.set_index(["iso3", "crop_standard_name"]).index.map(cnt).fillna(0.0)
        e2["studied2"] = (e2["nf2"] > 0).astype(float)
        b2 = logit(e2, X_MAIN, y="studied2")
        for t in ["log_area", "rd", "need"]:
            out.append(row(b2, t, f"logit_alloc_{tag}", b2.nobs))
        lg.count(f"alloc_{tag}_studies", int(s2.openalex_id.nunique()))

    # --- Mo2: country-level complementary models ------------------------------------
    cn = est.groupby("iso3").agg(
        nf=("n_studies_fractional", "sum"), area=("area_ha_mean", "sum"),
        **{c: (c, "first") for c in ["log_gdp_pc", "rd", "tertiary", "internet",
                                     "need", "log_population"]}).reset_index()
    cn["studied"] = (cn["nf"] > 0).astype(float)
    cn["log_area"] = np.log(cn["area"])
    xs_c = ["log_area", "log_gdp_pc", "rd", "tertiary", "internet", "need", "log_population"]
    Xc = sm.add_constant(cn[xs_c].astype(float))
    bc = sm.Logit(cn["studied"], Xc).fit(disp=0, cov_type="HC1")
    for t in xs_c: out.append(row(bc, t, "country_logit", bc.nobs))
    mc = sm.GLM(cn["nf"], Xc.drop(columns="log_area"), family=sm.families.Poisson(),
                offset=np.log(cn["area"])).fit(cov_type="HC1")
    for t in ["rd", "need", "internet", "tertiary", "log_population"]:
        out.append(row(mc, t, "country_ppml", mc.nobs))
    lg.count("country_level_share_studied", round(float(cn["studied"].mean()), 3))

    # --- M5: interquartile-range average effects (baseline logit and PPML) ----------
    iqr_rows = []
    pB, dB = ppml(est, [x for x in X_MAIN if x != "log_area"])
    for t in X_MAIN:
        q1, q3 = est[t].quantile([0.25, 0.75])
        X0 = sm.add_constant(est[X_MAIN].astype(float))
        lo, hi = X0.copy(), X0.copy(); lo[t] = q1; hi[t] = q3
        dp = float((b.predict(hi) - b.predict(lo)).mean())
        mult = np.nan
        if t != "log_area":
            mult = float(np.exp(pB.params[t] * (q3 - q1)))
        iqr_rows.append({"term": t, "p25": round(float(q1), 3), "p75": round(float(q3), 3),
                         "iqr_dprob_participation": round(dp, 4),
                         "iqr_multiplier_intensity": round(mult, 3) if mult == mult else None})
    iqr = pd.DataFrame(iqr_rows).sort_values("iqr_dprob_participation", key=abs, ascending=False)

    # --- XXIV: covariate reference years --------------------------------------------
    yr_rows = []
    for c in ["rd_expenditure_gdp_pct_year", "gdp_pc_ppp_year", "tertiary_enrolment_pct_year",
              "internet_users_pct_year", "population_year"]:
        y = pd.to_numeric(est.groupby("iso3")[c].first(), errors="coerce")
        inc = est.groupby("iso3")["wb_income_group"].first()
        stale = y[inc.isin(["Low income", "Lower middle income"])]
        yr_rows.append({"covariate": c.replace("_year", ""), "median": int(y.median()),
                        "min": int(y.min()), "max": int(y.max()),
                        "median_low_lmic": int(stale.median()) if len(stale) else None})
    yrs = pd.DataFrame(yr_rows)

    R = pd.DataFrame(out)
    R.to_csv(HERE / "model_audit_results.csv", index=False); lg.add_output(HERE / "model_audit_results.csv", rows=len(R))
    iqr.to_csv(HERE / "iqr_effect_comparison.csv", index=False); lg.add_output(HERE / "iqr_effect_comparison.csv", rows=len(iqr))
    yrs.to_csv(HERE / "covariate_reference_years.csv", index=False); lg.add_output(HERE / "covariate_reference_years.csv", rows=len(yrs))
    pd.set_option("display.width", 160)
    print("=== key coefficients across specifications (log-odds / log-rate) ===")
    key = R[R.term.isin(["log_area", "rd", "need", "need9"])]
    print(key.pivot_table(index="model", columns="term", values="estimate", aggfunc="first").to_string())
    print("\n=== IQR effect comparison (participation prob change; intensity multiplier) ===")
    print(iqr.to_string(index=False))
    print("\n=== covariate reference years ===")
    print(yrs.to_string(index=False))
    lg.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
