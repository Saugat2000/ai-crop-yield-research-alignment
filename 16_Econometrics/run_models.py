"""Phase 11 — the econometric model sequence on the country-crop panel.

What is estimated, and in what order:

  1  participation      logit, and probit as a link check: is this crop system studied at all
  2  intensity          PPML with harvested area as exposure; negative binomial for comparison
  3  hurdle             participation and intensity fitted as one two-part model
  4  leadership         fractional logit on the local first-author share, where observed
  5  citation uptake    PPML on citations, study level

Design commitments, all made before the estimates were seen:

* **Fractional counts are the outcome.** They are non-integer, which is why PPML is the
  primary intensity specification: it needs a correct conditional mean, not a count
  likelihood. The negative binomial is fitted on rounded counts for comparison only, and
  is labelled as such wherever it appears.
* **Zeros are data.** 2,131 of 2,616 country-crop cells have no eligible study. They carry
  the model, and dropping them would answer a different question.
* **Cluster-robust standard errors by country.** Rows within a country share unobserved
  research capacity, so independent errors would be a fiction.
* **Observational language throughout.** These are associations. Nothing here identifies a
  causal effect and the manuscript says so.
* **No specification is chosen for its sign or its significance.** Every model that was
  fitted is reported, including the ones that fail their diagnostics.

Residual spatial autocorrelation is tested after each country-level model, because a
significant residual Moran's I means the model is missing spatial structure and its
standard errors are optimistic.
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

WB = {
    "GB.XPD.RSDV.GD.ZS": "rd_expenditure_gdp_pct",
    "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp",
    "SE.TER.ENRR": "tertiary_enrolment_pct",
    "IT.NET.USER.ZS": "internet_users_pct",
    "SP.POP.TOTL": "population",
    "SL.AGR.EMPL.ZS": "agri_employment_pct",
}


def wb_latest(lg) -> pd.DataFrame:
    """Most recent non-missing value per country per indicator, with its year kept so the
    staleness of a covariate is visible rather than hidden."""
    src = ROOT / "10_External_Data" / "World_Bank" / "wb_indicators_long.csv"
    lg.add_input(src)
    d = pd.read_csv(src)
    d = d[d["indicator"].isin(WB) & d["value"].notna()]
    d = d.sort_values("year").groupby(["iso3", "indicator"], as_index=False).last()
    w = d.pivot(index="iso3", columns="indicator", values="value").rename(columns=WB)
    yrs = d.pivot(index="iso3", columns="indicator", values="year").rename(
        columns={k: v + "_year" for k, v in WB.items()})
    return w.join(yrs).reset_index()


def main() -> int:
    lg = RunLogger("phase11_01_econometric_models")
    np.random.seed(RANDOM_SEED)
    lg.count("random_seed", RANDOM_SEED)

    panel_p = P["integration"] / "country_crop_panel.parquet"
    lg.add_input(panel_p)
    d = pd.read_parquet(panel_p)
    w = wb_latest(lg)
    d = d.merge(w, on="iso3", how="left")

    # --- covariates ------------------------------------------------------------------
    d["log_area"] = np.log1p(pd.to_numeric(d["area_ha_mean"], errors="coerce"))
    d["log_production"] = np.log1p(pd.to_numeric(d["production_t_mean"], errors="coerce"))
    d["log_gdp_pc"] = np.log(pd.to_numeric(d["gdp_pc_ppp"], errors="coerce"))
    d["log_population"] = np.log(pd.to_numeric(d["population"], errors="coerce"))
    # `need_rank_pct` is a 0-1 proportion despite the `_pct` suffix. Dividing by 100 again
    # left the regressor spanning 0-0.01, which inflated the printed coefficient and its
    # interval a hundredfold. Inference is invariant to the rescale, but the reported
    # numbers were not interpretable. The regressor is the 0-1 percentile rank, so the
    # coefficient reads as the change in log-odds from the lowest to the highest need rank.
    _need = pd.to_numeric(d["need_rank_pct"], errors="coerce")
    if _need.max(skipna=True) > 1.5:
        raise ValueError(f"need_rank_pct max is {_need.max():.3f}; expected a 0-1 proportion")
    d["need"] = _need
    d["yield_volatility"] = pd.to_numeric(d["yield_volatility_cv"], errors="coerce")
    d["rd"] = pd.to_numeric(d["rd_expenditure_gdp_pct"], errors="coerce")
    d["tertiary"] = pd.to_numeric(d["tertiary_enrolment_pct"], errors="coerce") / 100.0
    d["internet"] = pd.to_numeric(d["internet_users_pct"], errors="coerce") / 100.0
    d["studied"] = d["has_any_study"].astype(float)

    X_MAIN = ["log_area", "log_gdp_pc", "rd", "tertiary", "internet",
              "need", "yield_volatility", "log_population"]

    # Listwise deletion is reported, never silent.
    est = d.dropna(subset=X_MAIN + ["studied", "n_studies_fractional"]).copy()
    lg.count("panel_rows", len(d))
    lg.count("estimation_rows", len(est))
    lg.count("rows_dropped_missing_covariates", len(d) - len(est))
    lg.count("estimation_countries", int(est["iso3"].nunique()))
    lg.count("estimation_cells_studied", int(est["studied"].sum()))
    miss = {c: int(d[c].isna().sum()) for c in X_MAIN}
    lg.note(f"missingness by covariate before deletion: {miss}")
    pd.Series(miss).to_csv(P["econ"] / "covariate_missingness.csv",
                           header=["n_missing"])
    lg.add_output(P["econ"] / "covariate_missingness.csv", rows=len(miss))

    chk = M.check_estimability(est, "studied", X_MAIN)
    lg.note(f"estimability: {chk}")

    fitted, diags = {}, {}

    # --- 1 participation -------------------------------------------------------------
    for link in ("logit", "probit"):
        try:
            res, dg = M.fit_participation(est, "studied", X_MAIN, link=link,
                                          cluster_col="iso3")
            fitted[f"participation_{link}"] = res
            diags[f"participation_{link}"] = dg
        except Exception as e:               # a refusal is a result and is recorded
            diags[f"participation_{link}"] = {"failed": str(e)}
            lg.warn(f"participation {link} failed: {e}")

    # --- 2 intensity -----------------------------------------------------------------
    est_exp = est[est["area_ha_mean"].fillna(0) > 0].copy()
    lg.count("intensity_rows_with_positive_exposure", len(est_exp))
    X_INT = [c for c in X_MAIN if c != "log_area"]
    for fam, key, xs, exp in (("poisson", "ppml_area_exposure", X_INT, "area_ha_mean"),
                              ("poisson", "ppml_no_exposure", X_MAIN, None),
                              ("negbin", "negbin_area_exposure", X_INT, "area_ha_mean")):
        try:
            res, dg = M.fit_count(est_exp if exp else est, "n_studies_fractional", xs,
                                  family=fam, exposure_col=exp, cluster_col="iso3")
            fitted[key] = res
            diags[key] = dg
        except Exception as e:
            diags[key] = {"failed": str(e)}
            lg.warn(f"{key} failed: {e}")

    if "ppml_area_exposure" in fitted:
        try:
            od = M.overdispersion_test(fitted["ppml_area_exposure"],
                                       est_exp["n_studies_fractional"])
            diags["overdispersion"] = od
            lg.note(f"overdispersion: {od}")
        except Exception as e:
            lg.warn(f"overdispersion test failed: {e}")

    # --- 3 hurdle --------------------------------------------------------------------
    try:
        h = M.fit_hurdle(est, "n_studies_fractional", X_MAIN, X_MAIN, cluster_col="iso3")
        diags["hurdle"] = {k: v for k, v in h.items() if not hasattr(v, "params")} \
            if isinstance(h, dict) else {"fitted": True}
        if isinstance(h, dict):
            for k, v in h.items():
                if hasattr(v, "params"):
                    fitted[f"hurdle_{k}"] = v
    except Exception as e:
        diags["hurdle"] = {"failed": str(e)}
        lg.warn(f"hurdle failed: {e}")

    # --- 4 local leadership ----------------------------------------------------------
    lead = est.dropna(subset=["local_first_author_share"])
    lead = lead[lead["n_leadership_observed"].fillna(0) > 0]
    lg.count("leadership_rows", len(lead))
    if len(lead) >= 40:
        try:
            res, dg = M.fit_fractional(lead, "local_first_author_share", X_MAIN,
                                       cluster_col="iso3")
            fitted["leadership_fractional_logit"] = res
            diags["leadership_fractional_logit"] = dg
        except Exception as e:
            diags["leadership_fractional_logit"] = {"failed": str(e)}
            lg.warn(f"leadership model failed: {e}")
    else:
        diags["leadership_fractional_logit"] = {
            "not_estimated": f"only {len(lead)} country-crop cells have an observed local "
                             f"first-author share; below the 40-cell floor"}
        lg.warn(diags["leadership_fractional_logit"]["not_estimated"])

    # --- 5 citation uptake, study level ----------------------------------------------
    sl = pd.read_parquet(P["integration"] / "study_level_dataset.parquet")
    lg.add_input(P["integration"] / "study_level_dataset.parquet")
    sl = sl.merge(w, left_on="loc_country_iso3", right_on="iso3", how="left")
    sl["age"] = 2026 - pd.to_numeric(sl["publication_year"], errors="coerce")
    sl["log_gdp_pc"] = np.log(pd.to_numeric(sl["gdp_pc_ppp"], errors="coerce"))
    sl["is_oa_f"] = sl["is_oa"].fillna(False).astype(float)
    sl["is_conf_f"] = sl["is_conference"].astype(float)
    sl["intl"] = pd.to_numeric(sl["international_collaboration"], errors="coerce")
    sl["n_auth"] = pd.to_numeric(sl["n_authors"], errors="coerce")
    sl["cites"] = pd.to_numeric(sl["cited_by_count"], errors="coerce").fillna(0)
    X_CIT = ["age", "is_oa_f", "is_conf_f", "intl", "n_auth", "log_gdp_pc"]
    cit = sl.dropna(subset=X_CIT + ["cites"]).copy()
    lg.count("citation_model_rows", len(cit))
    if len(cit) >= 100:
        try:
            res, dg = M.fit_count(cit, "cites", X_CIT, family="poisson",
                                  cluster_col="loc_country_iso3")
            fitted["citation_ppml"] = res
            diags["citation_ppml"] = dg
        except Exception as e:
            diags["citation_ppml"] = {"failed": str(e)}
            lg.warn(f"citation model failed: {e}")

    # --- residual spatial autocorrelation --------------------------------------------
    with open(P["weights"] / "spatial_weights.pkl", "rb") as fh:
        W = pickle.load(fh)
    w6 = W["knn_k6"]
    # `residual_moran` takes a dense, row-standardised matrix, not a libpysal object.
    # Passing the object raised "setting an array element with a sequence" and the test was
    # skipped with only a warning — which is exactly how a missing spatial diagnostic goes
    # unnoticed, so the conversion is explicit here.
    w6_dense = w6.full()[0]
    w6_dense = w6_dense / np.where(w6_dense.sum(1, keepdims=True) == 0, 1,
                                   w6_dense.sum(1, keepdims=True))
    res_moran = {}
    for key in ("participation_logit", "ppml_area_exposure"):
        if key not in fitted:
            continue
        base = est_exp if key.startswith("ppml") else est
        r = pd.Series(np.asarray(fitted[key].resid_response), index=base.index)
        agg = base.assign(_r=r).groupby("iso3")["_r"].mean()
        ids = list(w6.id_order)
        v = agg.reindex(ids)
        n_missing = int(v.isna().sum())
        try:
            rm = M.residual_moran(v.fillna(v.mean()).values, w6_dense, seed=RANDOM_SEED)
            rm["n_countries_missing_filled"] = n_missing
            res_moran[key] = rm
            lg.note(f"residual Moran {key}: {rm}")
        except Exception as e:
            lg.warn(f"residual Moran for {key} failed: {e}")

    # --- outputs ----------------------------------------------------------------------
    try:
        tbl = M.publication_table({k: v for k, v in fitted.items()})
        tp = P["econ"] / "model_coefficients.csv"
        tbl.to_csv(tp)
        lg.add_output(tp, rows=len(tbl))
        print("\nCOEFFICIENTS\n", tbl.to_string())
    except Exception as e:
        lg.warn(f"publication table failed: {e}")

    try:
        cmp_ = M.compare_models(fitted)
        cp = P["econ"] / "model_comparison.csv"
        cmp_.to_csv(cp, index=False)
        lg.add_output(cp, rows=len(cmp_))
        print("\nMODEL COMPARISON\n", cmp_.to_string(index=False))
    except Exception as e:
        lg.warn(f"model comparison failed: {e}")

    dp = P["econ"] / "model_diagnostics.json"
    dp.write_text(json.dumps({"diagnostics": diags, "residual_moran": res_moran,
                              "estimability": chk}, indent=2, default=str) + "\n")
    lg.add_output(dp)

    ap = P["econ"] / "estimation_sample.parquet"
    est.to_parquet(ap, index=False)
    lg.add_output(ap, rows=len(est))

    print(f"\nfitted {len(fitted)} models on {len(est):,} country-crop cells "
          f"({est['iso3'].nunique()} countries); "
          f"{len(d) - len(est):,} cells dropped for missing covariates")
    for k, v in res_moran.items():
        print(f"residual Moran {k}: I={v.get('I'):.4f} p={v.get('p_value')} "
              f"clustered={v.get('residual_clustering')}")
    lg.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
