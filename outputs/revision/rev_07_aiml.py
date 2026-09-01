"""Step 6: AI/ML eligibility audit and a full rerun on the ML-confirmed subset.

The eligibility rule admits any "data-driven predictive method", which is broader than
the "AI-based" label in the title. This builds a deterministic model-family taxonomy
from titles and abstracts and reruns the MAJOR analyses on the subset whose method is
explicitly machine-learning, so the label can be defended or revised on evidence.
"""
from __future__ import annotations
import re, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
warnings.filterwarnings("ignore")
from libpysal.weights import KNN
from esda.moran import Moran

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

OUT = HERE
PERM = 9999

FAMILIES = {
    "deep_learning": r"\b(deep learning|neural network|CNN|convolutional|LSTM|RNN|recurrent|"
                     r"transformer|autoencoder|MLP|multilayer perceptron|ANN\b|deep neural)",
    "tree_ensemble": r"\b(random forest|decision tree|gradient boost|XGBoost|LightGBM|CatBoost|"
                     r"bagging|extra trees|boosted regression)",
    "svm_kernel": r"\b(support vector|SVM|SVR|kernel ridge|gaussian process)",
    "other_ml": r"\b(machine learning|k-nearest|KNN\b|naive bayes|ensemble learning|"
                r"reinforcement learning|artificial intelligence|LASSO|elastic net|ridge regression|"
                r"random subspace|adaboost)",
    "classical_stat": r"\b(linear regression|multiple regression|ordinary least squares|OLS\b|"
                      r"stepwise regression|ARIMA|logistic regression|generalized linear|"
                      r"principal component regression|partial least squares)",
    "process_model": r"\b(APSIM|DSSAT|CERES|WOFOST|AquaCrop|EPIC model|STICS|crop simulation|"
                     r"process-based model|crop growth model)",
}
ML_FAMILIES = ["deep_learning", "tree_ensemble", "svm_kernel", "other_ml"]


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() else np.nan


def main():
    lg = RunLogger("rev_07_aiml")
    sld = pd.read_parquet(ROOT / "12_Data_Integration" / "study_level_dataset.parquet")
    scc = pd.read_parquet(ROOT / "12_Data_Integration" / "study_country_crop_dataset.parquet")
    pan = pd.read_parquet(ROOT / "12_Data_Integration" / "country_crop_panel.parquet")
    layer = pd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    n9 = pd.read_parquet(ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet")
    lg.add_input(ROOT / "12_Data_Integration" / "study_level_dataset.parquet")

    txt = (sld.title.fillna("") + " " + sld.abstract.fillna("")).str.lower()
    for fam, pat in FAMILIES.items():
        sld[fam] = txt.str.contains(pat, regex=True, case=False, na=False)
    sld["any_ml"] = sld[ML_FAMILIES].any(axis=1)
    sld["any_method_kw"] = sld[list(FAMILIES)].any(axis=1)
    sld["has_abstract"] = sld.abstract.notna()

    tax = []
    for fam in list(FAMILIES) + ["any_ml", "any_method_kw"]:
        tax.append(dict(family=fam, studies=int(sld[fam].sum()),
                        share_pct=100 * sld[fam].mean(),
                        share_of_abstracted_pct=100 * sld.loc[sld.has_abstract, fam].mean()))
    tax.append(dict(family="no_method_keyword", studies=int((~sld.any_method_kw).sum()),
                    share_pct=100 * (~sld.any_method_kw).mean(),
                    share_of_abstracted_pct=100 * (~sld.loc[sld.has_abstract, "any_method_kw"]).mean()))
    tx = pd.DataFrame(tax)
    tx.to_csv(OUT / "aiml_taxonomy.csv", index=False)
    print("MODEL-FAMILY TAXONOMY (deterministic keyword screen over title + abstract)")
    print(tx.round(2).to_string(index=False))
    print(f"\ncorpus {len(sld)}; abstracts present {int(sld.has_abstract.sum())}; "
          f"ML-confirmed {int(sld.any_ml.sum())} "
          f"({100*sld.any_ml.mean():.1f}% of corpus, "
          f"{100*sld.loc[sld.has_abstract,'any_ml'].mean():.1f}% of abstracted)")

    # ------------------------------------------------------------------ ML-confirmed rerun
    ml_ids = set(sld.loc[sld.any_ml, "openalex_id"])
    sub = scc[scc.openalex_id.isin(ml_ids)].copy()
    print(f"\nML-confirmed allocation: {sub.openalex_id.nunique()} studies, "
          f"{sub.fractional_weight.sum():.1f} fractional weight "
          f"(full corpus 2,031 / 2,031.0)")

    res = []
    # 1-2. country and crop concentration
    for lab, full_df, sub_df, key in [
            ("country", scc, sub, "iso3"), ("crop", scc, sub, "crop_standard_name")]:
        f = full_df.groupby(key, as_index=False).fractional_weight.sum()
        s = sub_df.groupby(key, as_index=False).fractional_weight.sum()
        if key == "iso3":
            f = layer[["iso3"]].merge(f, on="iso3", how="left").fillna({"fractional_weight": 0})
            s = layer[["iso3"]].merge(s, on="iso3", how="left").fillna({"fractional_weight": 0})
        res.append(dict(analysis=f"{lab}_gini", full_corpus=gini(f.fractional_weight),
                        ml_subset=gini(s.fractional_weight)))
    # 3. top-three crop share
    f3 = scc.groupby("crop_standard_name").fractional_weight.sum().sort_values(ascending=False)
    s3 = sub.groupby("crop_standard_name").fractional_weight.sum().sort_values(ascending=False)
    res.append(dict(analysis="top3_crop_share_pct", full_corpus=100 * f3.head(3).sum() / f3.sum(),
                    ml_subset=100 * s3.head(3).sum() / s3.sum()))
    # 4. scale alignment: Spearman of cell research on harvested area
    def align(df):
        g = df.groupby(["iso3", "crop_standard_name"], as_index=False).fractional_weight.sum()
        m = pan[["iso3", "crop_standard_name", "area_ha_mean"]].merge(g, how="left",
             on=["iso3", "crop_standard_name"]).fillna({"fractional_weight": 0})
        m = m[m.area_ha_mean > 0]
        return m.fractional_weight.corr(m.area_ha_mean, method="spearman")
    res.append(dict(analysis="spearman_research_vs_area_cell",
                    full_corpus=align(scc), ml_subset=align(sub)))
    # 5-6. participation and intensity models on ML-subset outcomes
    e = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet").merge(
        n9[["iso3", "need9_rank_pct"]].rename(columns={"need9_rank_pct": "need9"}),
        on="iso3", how="left")
    sml = sub.groupby(["iso3", "crop_standard_name"], as_index=False).fractional_weight.sum() \
             .rename(columns={"fractional_weight": "r_ml"})
    e = e.merge(sml, on=["iso3", "crop_standard_name"], how="left")
    e["r_ml"] = e.r_ml.fillna(0.0)
    e["studied_ml"] = (e.r_ml > 0).astype(float)
    X = ["log_area", "rd", "need9", "log_gdp_pc", "tertiary", "internet", "log_population"]
    def lg_(y):
        M = sm.add_constant(e[X].astype(float), has_constant="add")
        return sm.Logit(e[y].astype(float), M).fit(disp=0, cov_type="cluster",
                                                   cov_kwds={"groups": e.iso3.to_numpy()})
    m_full, m_ml = lg_("studied"), lg_("studied_ml")
    for t in ["log_area", "need9", "rd"]:
        res.append(dict(analysis=f"logit_{t}", full_corpus=m_full.params[t],
                        ml_subset=m_ml.params[t]))
    d = e[e.area_ha_mean.fillna(0) > 0]
    def pp(y):
        M = sm.add_constant(d[[x for x in X if x != "log_area"]].astype(float), has_constant="add")
        return sm.GLM(d[y].astype(float), M, family=sm.families.Poisson(),
                      offset=np.log(d.area_ha_mean.astype(float))).fit(
            cov_type="cluster", cov_kwds={"groups": d.iso3.to_numpy()})
    p_full, p_ml = pp("n_studies_fractional"), pp("r_ml")
    for t in ["need9", "rd"]:
        res.append(dict(analysis=f"ppml_{t}", full_corpus=p_full.params[t], ml_subset=p_ml.params[t]))
    # 7. Moran's I on the mismatch measure
    geom = gpd.GeoSeries.from_wkb(layer.geometry) if layer.geometry.dtype == object else layer.geometry
    cent = gpd.GeoDataFrame(layer.copy(), geometry=geom, crs="EPSG:4326") \
        .to_crs("+proj=eqearth").geometry.centroid
    def moran_gap(alloc):
        g = alloc.groupby("iso3", as_index=False).fractional_weight.sum()
        dd = layer[["iso3"]].merge(g, on="iso3", how="left").fillna({"fractional_weight": 0})
        dd = dd.merge(n9[["iso3", "need9_rank_pct"]], on="iso3", how="left")
        dd = dd[dd.need9_rank_pct.notna()].reset_index(drop=True)
        gap = dd.fractional_weight.rank(pct=True) * 100 - dd.need9_rank_pct * 100
        pos = [layer.iso3.tolist().index(i) for i in dd.iso3]
        gg = gpd.GeoDataFrame(dd[["iso3"]], geometry=gpd.points_from_xy(
            cent.x.values[pos], cent.y.values[pos]))
        w = KNN.from_dataframe(gg, k=6, ids=dd.iso3.tolist())
        assert list(w.id_order) == dd.iso3.tolist()
        w.transform = "r"
        m = Moran(gap.values, w, permutations=PERM)
        return m.I, m.p_sim
    If, pf = moran_gap(scc); Im, pm = moran_gap(sub)
    res.append(dict(analysis="morans_I_gap9", full_corpus=If, ml_subset=Im))
    res.append(dict(analysis="morans_I_gap9_p", full_corpus=pf, ml_subset=pm))

    r = pd.DataFrame(res)
    r["abs_change"] = (r.ml_subset - r.full_corpus).abs()
    r.to_csv(OUT / "aiml_subset_rerun.csv", index=False)
    print("\nMAJOR ANALYSES: FULL CORPUS vs ML-CONFIRMED SUBSET")
    print(r.round(4).to_string(index=False))

    lg.add_output(OUT / "aiml_taxonomy.csv")
    lg.add_output(OUT / "aiml_subset_rerun.csv")
    lg.finish()
    print("\nrev_07_aiml complete")


if __name__ == "__main__":
    main()
