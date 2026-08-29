"""Pre-submission audit M2/M3/M7/Mo4/Mo5: corpus accounting, allocation-scope
sensitivity of concentration, resolution-bias diagnostics, within-crop production
benchmarking, and a rule-based method-family keyword audit."""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> int:
    lg = RunLogger("audit_04_corpus")
    sl = pd.read_parquet(P["integration"] / "study_level_dataset.parquet")
    scc = pd.read_parquet(P["integration"] / "study_country_crop_dataset.parquet")
    panel = pd.read_parquet(P["integration"] / "country_crop_panel.parquet")
    for f in ("study_level_dataset.parquet", "study_country_crop_dataset.parquet"):
        lg.add_input(P["integration"] / f)

    # ---- M7/XI: mutually exclusive accounting -------------------------------------
    acc = sl["loc_study_scope"].value_counts(dropna=False).rename_axis("location_status")
    acc.to_csv(HERE / "location_accounting.csv")
    assert int(acc.sum()) == 7045
    crop_acc = sl["crop_crop_scope"].value_counts(dropna=False)
    crop_acc.to_csv(HERE / "crop_accounting.csv")
    al = set(scc.openalex_id)
    flow = {
        "corpus": len(sl),
        "with_any_accepted_country_evidence": int((sl.loc_country_iso3.notna()
            | sl.loc_countries_all_iso3.fillna("").astype(str).str.len().gt(0)).sum()),
        "with_faostat_crop": int(sl.crop_faostat_items.fillna("").astype(str).str.len().gt(0).sum()),
        "allocated_studies": int(scc.openalex_id.nunique()),
        "allocation_records": len(scc),
        "fractional_weight_total": round(float(scc.fractional_weight.sum()), 2),
        "weight_inside_2616_panel": round(float(
            scc.merge(panel[["iso3", "crop_standard_name"]], on=["iso3", "crop_standard_name"],
                      how="inner").fractional_weight.sum()), 2),
    }
    pd.Series(flow).to_csv(HERE / "sample_flow.csv")
    for k, v in flow.items(): lg.count(f"flow_{k}", v)
    # which allocated pairs fall outside the panel
    out_pairs = scc.merge(panel[["iso3", "crop_standard_name"]].assign(inp=1),
                          on=["iso3", "crop_standard_name"], how="left")
    outside = out_pairs[out_pairs.inp.isna()]
    lg.count("weight_outside_panel", round(float(outside.fractional_weight.sum()), 2))
    outside.groupby(["iso3", "crop_standard_name"])["fractional_weight"].sum().sort_values(
        ascending=False).head(20).to_csv(HERE / "allocation_outside_panel.csv")

    # ---- M2: concentration under allocation-scope restrictions ---------------------
    sl2 = sl[["openalex_id", "loc_study_scope"]]
    s = scc.merge(sl2, on="openalex_id", how="left")
    uni = sorted(panel.iso3.dropna().unique())
    rows = []
    for tag, keep in (("baseline_all", None),
                      ("no_global", ["single_country", "multi_country", "unresolved"]),
                      ("strict_specific_only", ["single_country", "multi_country"])):
        d = s if keep is None else s[s.loc_study_scope.isin(keep)]
        c = d.groupby("iso3")["fractional_weight"].sum()
        rows.append({"allocation": tag, "studies": int(d.openalex_id.nunique()),
                     "gini_all_countries": round(gini(c.reindex(uni, fill_value=0).values), 3),
                     "gini_studied": round(gini(c[c > 0].values), 3),
                     "top1_share_pct": round(100 * c.max() / c.sum(), 1),
                     "countries_with_any": int((c > 0).sum())})
    conc = pd.DataFrame(rows); conc.to_csv(HERE / "allocation_scope_concentration.csv", index=False)

    # ---- Mo4: resolution-bias diagnostics ------------------------------------------
    sl["resolved"] = sl.loc_study_scope.isin(["single_country", "multi_country"])
    sl["has_abstract"] = sl.abstract.fillna("").astype(str).str.len().gt(50)
    cmp_rows = []
    for name, col in [("publication_year", "publication_year"),
                      ("cited_by_count", "cited_by_count"), ("n_authors", "n_authors")]:
        a = pd.to_numeric(sl.loc[sl.resolved, col], errors="coerce")
        b = pd.to_numeric(sl.loc[~sl.resolved, col], errors="coerce")
        cmp_rows.append({"characteristic": name, "resolved_median": float(a.median()),
                         "unresolved_median": float(b.median())})
    for name, col in [("conference", "is_conference"), ("preprint", "is_preprint"),
                      ("abstract_available", "has_abstract")]:
        cmp_rows.append({"characteristic": name,
                         "resolved_median": round(float(sl.loc[sl.resolved, col].mean()), 3),
                         "unresolved_median": round(float(sl.loc[~sl.resolved, col].mean()), 3)})
    en = (sl.language.fillna("") == "en")
    cmp_rows.append({"characteristic": "english",
                     "resolved_median": round(float(en[sl.resolved].mean()), 3),
                     "unresolved_median": round(float(en[~sl.resolved].mean()), 3)})
    cmp = pd.DataFrame(cmp_rows); cmp.to_csv(HERE / "resolution_bias_comparison.csv", index=False)
    import statsmodels.api as sm
    m = sl.dropna(subset=["publication_year"])
    X = pd.DataFrame({
        "year_c": pd.to_numeric(m.publication_year) - 2018,
        "conference": m.is_conference.astype(float),
        "preprint": m.is_preprint.astype(float),
        "has_abstract": m.has_abstract.astype(float),
        "log_citations": np.log1p(pd.to_numeric(m.cited_by_count, errors="coerce").fillna(0)),
    })
    lr = sm.Logit(m.resolved.astype(float), sm.add_constant(X)).fit(disp=0, cov_type="HC1")
    lr_out = pd.DataFrame({"term": lr.params.index, "estimate": lr.params.round(3).values,
                           "se": lr.bse.round(3).values})
    lr_out.to_csv(HERE / "resolution_logit.csv", index=False)

    # ---- M3: within-crop production benchmarking ------------------------------------
    p = panel.copy()
    p["res_share_in_crop"] = p.groupby("crop_standard_name")["n_studies_fractional"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else np.nan)
    p["prod_share_in_crop"] = p.groupby("crop_standard_name")["production_t_mean"].transform(
        lambda x: x / x.sum() if pd.notna(x.sum()) and x.sum() > 0 else np.nan)
    p["area_share_in_crop"] = p.groupby("crop_standard_name")["area_ha_mean"].transform(
        lambda x: x / x.sum() if pd.notna(x.sum()) and x.sum() > 0 else np.nan)
    p["within_gap_pp"] = 100 * (p["res_share_in_crop"].fillna(0) - p["prod_share_in_crop"])
    keep = p[["iso3", "crop_standard_name", "n_studies_fractional", "res_share_in_crop",
              "prod_share_in_crop", "area_share_in_crop", "within_gap_pp"]]
    keep.to_csv(HERE / "within_crop_benchmarks.csv", index=False)
    ex = keep[(keep.iso3.isin(["BRA", "IND", "IDN", "CHN", "USA"])) &
              (keep.crop_standard_name.isin(["sugarcane", "oil_palm", "wheat", "maize", "rice"]))]
    ex.to_csv(HERE / "within_crop_examples.csv", index=False)
    sp = keep.dropna(subset=["res_share_in_crop", "prod_share_in_crop"])
    lg.count("spearman_res_vs_prod_withincrop", round(float(
        sp[["res_share_in_crop", "prod_share_in_crop"]].corr(method="spearman").iloc[0, 1]), 3))
    lg.count("spearman_res_vs_area_withincrop", round(float(
        keep.dropna(subset=["res_share_in_crop", "area_share_in_crop"])[
            ["res_share_in_crop", "area_share_in_crop"]].corr(method="spearman").iloc[0, 1]), 3))

    # ---- Mo5: rule-based method-family keyword audit --------------------------------
    txt = (sl.title.fillna("") + " " + sl.abstract.fillna("")).str.lower()
    fam = {
        "neural_deep": r"neural network|deep learning|cnn|convolution|lstm|transformer|recurrent|multilayer perceptron|\bmlp\b|\bdnn\b|\bann\b|autoencoder",
        "tree_boosting": r"random forest|gradient boost|xgboost|lightgbm|catboost|decision tree|extra trees|bagging|adaboost",
        "svm_kernel": r"support vector|\bsvm\b|\bsvr\b|kernel (?:ridge|regression)",
        "other_ml": r"machine learning|\bknn\b|k-nearest|gaussian process|elastic net|\blasso\b|ridge regression|ensemble|genetic (?:algorithm|programming)|artificial intelligence|\bfuzzy\b|reinforcement learning",
        "classical_regression": r"linear regression|multiple regression|multiple linear|stepwise regression|logistic regression|regression (?:model|analysis)|\bpls\b|partial least squares|principal component regression",
        "process_model": r"\bdssat\b|\bapsim\b|\bceres\b|aquacrop|\bwofost\b|\bstics\b|cropsyst|\bepic model\b|crop simulation model",
    }
    F = pd.DataFrame({k: txt.str.contains(v, regex=True) for k, v in fam.items()})
    F["any_ml"] = F[["neural_deep", "tree_boosting", "svm_kernel", "other_ml"]].any(axis=1)
    F["classical_only"] = F.classical_regression & ~F.any_ml & ~F.process_model
    F["process_only"] = F.process_model & ~F.any_ml
    F["no_signal"] = ~F[list(fam)].any(axis=1)
    summ = F.mean().round(4) * 100
    summ.to_csv(HERE / "method_family_keyword_audit.csv")
    for k in ("any_ml", "classical_only", "process_only", "no_signal"):
        lg.count(f"method_{k}_pct", round(float(F[k].mean() * 100), 2))
    print("=== allocation-scope concentration ==="); print(conc.to_string(index=False))
    print("\n=== resolution-bias comparison ==="); print(cmp.to_string(index=False))
    print("\n=== resolution logit ==="); print(lr_out.to_string(index=False))
    print("\n=== within-crop examples ==="); print(ex.to_string(index=False))
    print("\n=== method families (% of corpus, title+abstract keywords) ===")
    print(summ.to_string())
    lg.finish(); return 0


if __name__ == "__main__":
    sys.exit(main())
