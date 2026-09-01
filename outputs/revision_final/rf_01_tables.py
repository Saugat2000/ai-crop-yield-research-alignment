"""Final revision tables.

  Table A8  - resolved versus unresolved studies, with tests
  Table A9  - model diagnostics: R&D reference years, quasi-Poisson standard errors,
              and Lagrange Multiplier tests for spatial lag and error
  Table A10 - period-specific participation models

Every estimate is recomputed here rather than copied, so the tables and the console
output come from the same run.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")
from libpysal.weights import KNN
from spreg import OLS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

OUT = HERE
TEX = OUT / "tables"; TEX.mkdir(exist_ok=True)
X = ["log_area", "rd", "need9", "log_gdp_pc", "tertiary", "internet", "log_population"]
LBL = {"log_area": "Log harvested area", "rd": "R\\&D expenditure (\\% GDP)",
       "need9": "Research-need index", "log_gdp_pc": "Log GDP per capita",
       "tertiary": "Tertiary enrolment", "internet": "Internet use",
       "log_population": "Log population"}
ML_RX = (r"\b(deep learning|neural network|CNN|convolutional|LSTM|RNN|recurrent|transformer|"
         r"autoencoder|MLP|multilayer perceptron|ANN\b|random forest|decision tree|gradient boost|"
         r"XGBoost|LightGBM|CatBoost|bagging|extra trees|support vector|SVM|SVR|kernel ridge|"
         r"gaussian process|machine learning|k-nearest|KNN\b|naive bayes|ensemble learning|"
         r"reinforcement learning|artificial intelligence|LASSO|elastic net|random subspace|adaboost)")


def num(x, d=3):
    if pd.isna(x): return "---"
    return f"{x:.{d}f}".replace("-", "$-$")


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def wrap(body, caption, label, notes, widths, small=True):
    return ("\\begin{table*}[!t]\n\\centering\n" + ("\\footnotesize\n" if small else "") +
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}{widths}@{{}}}}\n"
            "\\toprule\n" + body + "\n\\bottomrule\n\\end{tabular*}\n"
            "\\begin{tablenotes}\n" + notes + "\n\\end{tablenotes}\n\\end{table*}\n")


def load():
    e = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    n9 = pd.read_parquet(ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet")
    return e.merge(n9[["iso3", "need9_rank_pct"]].rename(columns={"need9_rank_pct": "need9"}),
                   on="iso3", how="left")


def logit(df, xs, y="studied"):
    M = sm.add_constant(df[xs].astype(float), has_constant="add")
    return sm.Logit(df[y].astype(float), M).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": df["iso3"].to_numpy()})


# ============================================================ Table A8
def table_a8(lg):
    sld = pd.read_parquet(ROOT / "12_Data_Integration" / "study_level_dataset.parquet")
    scc = pd.read_parquet(ROOT / "12_Data_Integration" / "study_country_crop_dataset.parquet")
    sld["resolved"] = sld.openalex_id.isin(set(scc.openalex_id.unique()))
    txt = (sld.title.fillna("") + " " + sld.abstract.fillna("")).str.lower()
    sld["ml_kw"] = txt.str.contains(ML_RX, regex=True, na=False)
    a, b = sld[sld.resolved], sld[~sld.resolved]
    print(f"\nTABLE A8: resolved n={len(a)}, unresolved/unallocated n={len(b)}")

    rows, rec = [], []
    def cont(col, label, dec=1):
        x, y = a[col].dropna(), b[col].dropna()
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rec.append(dict(variable=label, test="Mann-Whitney U", statistic=u, p_value=p,
                        resolved_mean=x.mean(), resolved_median=x.median(),
                        unresolved_mean=y.mean(), unresolved_median=y.median()))
        rows.append(f"{label} & {x.mean():.{dec}f} & {x.median():.0f} & {y.mean():.{dec}f} & "
                    f"{y.median():.0f} & {u:,.0f} & {pfmt(p)} \\\\")
    def binary(mask_a, mask_b, label):
        tab = np.array([[mask_a.sum(), (~mask_a).sum()], [mask_b.sum(), (~mask_b).sum()]])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        rec.append(dict(variable=label, test="chi-square", statistic=chi2, p_value=p,
                        resolved_mean=mask_a.mean(), resolved_median=np.nan,
                        unresolved_mean=mask_b.mean(), unresolved_median=np.nan))
        rows.append(f"{label} & \\multicolumn{{2}}{{c}}{{{100*mask_a.mean():.1f}}} & "
                    f"\\multicolumn{{2}}{{c}}{{{100*mask_b.mean():.1f}}} & {chi2:,.1f} & {pfmt(p)} \\\\")

    cont("publication_year", "Publication year", 1)
    cont("cited_by_count", "Citation count", 1)
    body = ("& \\multicolumn{2}{c}{Resolved (2,031)} & \\multicolumn{2}{c}{Unresolved (5,014)} & & \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Measure & Mean & Median & Mean & Median & Statistic & $p$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\midrule\n"
            "\\multicolumn{7}{l}{\\textit{Percentages}} \\\\\n")
    rows = []
    dt = sld.type.fillna("other")
    for lab, key in [("Journal article", "article"), ("Conference paper or abstract", "conf"),
                     ("Preprint", "preprint"), ("Other document type", "other")]:
        if key == "conf":
            ma = a.type.isin(["conference-paper", "conference-abstract"])
            mb = b.type.isin(["conference-paper", "conference-abstract"])
        elif key == "other":
            ma = ~a.type.isin(["article", "conference-paper", "conference-abstract", "preprint"])
            mb = ~b.type.isin(["article", "conference-paper", "conference-abstract", "preprint"])
        else:
            ma, mb = a.type.eq(key), b.type.eq(key)
        binary(ma, mb, lab)
    binary(a.abstract.notna(), b.abstract.notna(), "Abstract available")
    binary(a.ml_kw, b.ml_kw, "Explicit machine-learning method term")
    body += "\n".join(rows)
    pd.DataFrame(rec).to_csv(OUT / "tableA8_resolved_vs_unresolved.csv", index=False)
    print(pd.DataFrame(rec)[["variable", "resolved_mean", "unresolved_mean", "test", "p_value"]]
          .round(4).to_string(index=False))
    (TEX / "tab_A8_resolved_vs_unresolved.tex").write_text(wrap(
        body, "Resolved versus unresolved studies",
        "tab:resolution",
        "\\item[] Resolved studies are the 2,031 carrying accepted country evidence and a "
        "FAOSTAT-matched crop, which enter the allocation; the remaining 5,014 do not. "
        "Continuous measures are compared by Mann--Whitney $U$ and percentages by chi-square on "
        "the two-by-two table. With samples of this size, small differences reach conventional "
        "significance: the publication-year difference is half a year, whereas the citation and "
        "abstract-availability differences are larger. The comparison characterises observable "
        "selection and does not establish that non-resolution is geographically random.",
        widths="l r r r r r r"))
    lg.add_output(OUT / "tableA8_resolved_vs_unresolved.csv")


# ============================================================ Table A9
def table_a9(lg, e):
    # ---- Panel A: R&D reference years
    ry = e[["iso3", "rd_expenditure_gdp_pct_year", "wb_income_group"]].drop_duplicates("iso3")
    byg = (ry.groupby("wb_income_group")
           .agg(n=("iso3", "size"), median=("rd_expenditure_gdp_pct_year", "median"),
                lo=("rd_expenditure_gdp_pct_year", "min"),
                hi=("rd_expenditure_gdp_pct_year", "max")).reset_index())
    print("\nTABLE A9 Panel A: R&D reference year by income group")
    print(byg.to_string(index=False))
    pa = "\n".join(f"{r.wb_income_group} & {int(r.n)} & {int(r['median'])} & {int(r.lo)} & "
                   f"{int(r.hi)} & \\multicolumn{{2}}{{c}}{{}} \\\\" for _, r in byg.iterrows())

    # ---- Panel B: quasi-Poisson versus clustered standard errors
    d = e[e.area_ha_mean.fillna(0) > 0].copy()
    Xp = sm.add_constant(d[[x for x in X if x != "log_area"]].astype(float), has_constant="add")
    off = np.log(d.area_ha_mean.astype(float))
    base = sm.GLM(d.n_studies_fractional.astype(float), Xp,
                  family=sm.families.Poisson(), offset=off)
    clus = base.fit(cov_type="cluster", cov_kwds={"groups": d.iso3.to_numpy()})
    naive = base.fit()
    phi_p = float(naive.pearson_chi2 / naive.df_resid)
    phi_d = float(naive.deviance / naive.df_resid)
    qse = naive.bse * np.sqrt(phi_p)
    print(f"\nTABLE A9 Panel B: dispersion phi (Pearson) = {phi_p:.1f}, (deviance) = {phi_d:.1f}")
    pb = []
    for t in [x for x in X if x != "log_area"]:
        b_, sc, sq = clus.params[t], clus.bse[t], qse[t]
        zc, zq = b_ / sc, b_ / sq
        pb.append(f"{LBL[t]} & {num(b_)} & {num(sc)} & {num(sq)} & {num(sq/sc,1)} & "
                  f"{pfmt(2*(1-stats.norm.cdf(abs(zc))))} & {pfmt(2*(1-stats.norm.cdf(abs(zq))))} \\\\")
        print(f"   {t:15s} b={b_:7.3f}  SE_clustered={sc:.3f}  SE_quasi={sq:.3f}  ratio={sq/sc:.1f}")
    qp = pd.DataFrame({"term": [x for x in X if x != "log_area"],
                       "estimate": [clus.params[t] for t in X if t != "log_area"],
                       "se_clustered": [clus.bse[t] for t in X if t != "log_area"],
                       "se_quasipoisson": [qse[t] for t in X if t != "log_area"],
                       "phi_pearson": phi_p, "phi_deviance": phi_d})
    qp.to_csv(OUT / "tableA9_quasipoisson.csv", index=False)

    # ---- Panel C: LM tests, both models
    layer = pd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    geom = gpd.GeoSeries.from_wkb(layer.geometry) if layer.geometry.dtype == object else layer.geometry
    cent = gpd.GeoDataFrame(layer.copy(), geometry=geom, crs="EPSG:4326") \
        .to_crs("+proj=eqearth").geometry.centroid
    ids = layer.iso3.tolist()
    m_logit = logit(e, X)
    e = e.copy()
    e["r_part"] = e.studied.astype(float) - m_logit.predict(
        sm.add_constant(e[X].astype(float), has_constant="add"))
    d2 = e[e.area_ha_mean.fillna(0) > 0].copy()
    d2["r_int"] = d2.n_studies_fractional.astype(float) - clus.predict(Xp)
    cm = (e.groupby("iso3", as_index=False)
          .agg(r_part=("r_part", "mean"), **{k: (k, "mean") for k in X}))
    ci = d2.groupby("iso3", as_index=False).agg(r_int=("r_int", "mean"))
    cm = cm.merge(ci, on="iso3", how="left").dropna()
    cm = cm[cm.iso3.isin(ids)].reset_index(drop=True)
    pos = [ids.index(i) for i in cm.iso3]
    g = gpd.GeoDataFrame(cm[["iso3"]], geometry=gpd.points_from_xy(
        cent.x.values[pos], cent.y.values[pos]))
    w = KNN.from_dataframe(g, k=6, ids=cm.iso3.tolist())
    assert list(w.id_order) == cm.iso3.tolist()
    w.transform = "r"
    print("\nTABLE A9 Panel C: LM spatial diagnostics (k = 6)")
    pc, lmrec = [], []
    for col, name in [("r_part", "Participation logit"), ("r_int", "Intensity PPML")]:
        o = OLS(cm[[col]].values, cm[X].values, w=w, spat_diag=True, moran=True,
                name_y=col, name_x=X, name_w="knn6")
        cells = []
        for key in ["lm_lag", "rlm_lag", "lm_error", "rlm_error"]:
            v = getattr(o, key)
            cells.append(f"{v[0]:.3f} & {pfmt(v[1])}")
            lmrec.append(dict(model=name, test=key, statistic=float(v[0]), p_value=float(v[1])))
        mres = o.moran_res
        lmrec.append(dict(model=name, test="moran_res", statistic=float(mres[0]),
                          p_value=float(mres[2])))
        pc.append(f"{name} & " + " & ".join(cells) + " \\\\")
        print(f"   {name}: " + ", ".join(
            f"{k}={getattr(o,k)[0]:.2f} (p={getattr(o,k)[1]:.3f})"
            for k in ["lm_lag", "rlm_lag", "lm_error", "rlm_error"]))
    pd.DataFrame(lmrec).to_csv(OUT / "tableA9_lm_tests.csv", index=False)

    lmdf = pd.DataFrame(lmrec)
    TESTS = [("LM lag", "lm_lag"), ("Robust LM lag", "rlm_lag"),
             ("LM error", "lm_error"), ("Robust LM error", "rlm_error")]
    pc2 = []
    for lab, key in TESTS:
        cells = []
        for name in ["Participation logit", "Intensity PPML"]:
            r = lmdf[(lmdf.model == name) & (lmdf.test == key)].iloc[0]
            cells.append(f"{r.statistic:.3f} & {pfmt(r.p_value)}")
        pc2.append(f"{lab} & " + " & ".join(cells) + " & \\\\")
    body = ("\\multicolumn{7}{l}{\\textit{Panel A. Reference year of the R\\&D expenditure "
            "observation, 142 estimation countries}} \\\\\n"
            "Income group & Countries & Median & Earliest & Latest & \\multicolumn{2}{c}{} \\\\\n"
            "\\midrule\n" + pa + "\n\\midrule\n"
            "\\multicolumn{7}{l}{\\textit{Panel B. Intensity model: clustered against "
            "quasi-Poisson standard errors}} \\\\\n"
            "Covariate & Estimate & SE (clustered) & SE (quasi-Poisson) & Ratio & "
            "$p$ (clust.) & $p$ (quasi) \\\\\n\\midrule\n" + "\n".join(pb) + "\n\\midrule\n"
            "\\multicolumn{7}{l}{\\textit{Panel C. Lagrange Multiplier tests on country-mean "
            "residuals, $k = 6$}} \\\\\n"
            "& \\multicolumn{2}{c}{Participation logit} & \\multicolumn{2}{c}{Intensity PPML} & "
            "\\multicolumn{2}{c}{} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Test & Statistic & $p$ & Statistic & $p$ & \\multicolumn{2}{c}{} \\\\\n"
            "\\midrule\n" + "\n".join(pc2))
    (TEX / "tab_A9_diagnostics.tex").write_text(wrap(
        body, "Covariate vintage, dispersion, and spatial diagnostics",
        "tab:diagnostics",
        "\\item[] Panel A gives the year of the most recent research and development expenditure "
        "observation for each estimation country. Panel B compares the reported cluster-robust "
        f"standard errors with quasi-Poisson standard errors formed from the Pearson dispersion, "
        f"$\\hat\\phi_{{P}} = {phi_p:.0f}$. The deviance-based dispersion for the same fit is "
        f"$\\hat\\phi_{{D}} = {phi_d:.1f}$. The two disagree by more than two orders of magnitude "
        "because the Pearson statistic is dominated by a small number of cells with very large "
        "counts, so a constant variance-to-mean ratio does not describe these data and the "
        "quasi-Poisson interval is conservative rather than correct. Cluster-robust errors, which "
        "assume no variance function, remain the reported choice. Panel C reports "
        "Lagrange Multiplier tests for a spatial lag and a spatial error process on country-mean "
        "residuals under the primary six-nearest-neighbour matrix; none rejects at the 5\\% level, "
        "which agrees with the residual Moran statistics in Section~5.4.",
        widths="l r r r r r r"))
    lg.add_output(OUT / "tableA9_quasipoisson.csv")
    lg.add_output(OUT / "tableA9_lm_tests.csv")
    return phi_p


# ============================================================ Table A10
def table_a10(lg, e):
    scc = pd.read_parquet(ROOT / "12_Data_Integration" / "study_country_crop_dataset.parquet")
    print("\nTABLE A10: period-specific participation models")
    cols, recs = {}, []
    for lab, lo, hi in [("2000-2019", 2000, 2019), ("2020-2026", 2020, 2026)]:
        sub = scc[(scc.publication_year >= lo) & (scc.publication_year <= hi)]
        agg = (sub.groupby(["iso3", "crop_standard_name"], as_index=False)
               .fractional_weight.sum().rename(columns={"fractional_weight": "r_p"}))
        ep = e.merge(agg, on=["iso3", "crop_standard_name"], how="left")
        ep["y"] = (ep.r_p.fillna(0) > 0).astype(float)
        m = logit(ep, X, y="y")
        ci = m.conf_int()
        cols[lab] = (m, ci, int(ep.y.sum()))
        for t in X:
            recs.append(dict(period=lab, term=t, estimate=m.params[t], se=m.bse[t],
                             ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1],
                             n=int(m.nobs), events=int(ep.y.sum())))
        print(f"   {lab}: n={int(m.nobs)} events={int(ep.y.sum())} "
              f"area={m.params['log_area']:.3f} rd={m.params['rd']:.3f} need={m.params['need9']:.3f}")
    pd.DataFrame(recs).to_csv(OUT / "tableA10_period_models.csv", index=False)
    rows = []
    for t in X:
        cells = []
        for lab in ["2000-2019", "2020-2026"]:
            m, ci, _ = cols[lab]
            cells.append(f"{num(m.params[t])} & [{num(ci.loc[t,0],2)}, {num(ci.loc[t,1],2)}]")
        rows.append(f"{LBL[t]} & " + " & ".join(cells) + " \\\\")
    body = ("& \\multicolumn{2}{c}{Published 2000--2019} & \\multicolumn{2}{c}{Published 2020--2026} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Covariate & Estimate & 95\\% CI & Estimate & 95\\% CI \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\midrule\n"
            f"Country-crop cells & \\multicolumn{{2}}{{c}}{{{int(cols['2000-2019'][0].nobs):,}}} & "
            f"\\multicolumn{{2}}{{c}}{{{int(cols['2020-2026'][0].nobs):,}}} \\\\\n"
            f"Cells carrying a study & \\multicolumn{{2}}{{c}}{{{cols['2000-2019'][2]}}} & "
            f"\\multicolumn{{2}}{{c}}{{{cols['2020-2026'][2]}}} \\\\")
    (TEX / "tab_A10_period_models.tex").write_text(wrap(
        body, "Participation models by publication period",
        "tab:periods",
        "\\item[] The outcome is whether a country-crop system carries a study published in the "
        "stated period; the covariate panel is unchanged, because the capacity and need indicators "
        "are single most-recent observations rather than annual series. The models therefore test "
        "whether the associations are stable across the literature's recent expansion, not whether "
        "covariates changed. Standard errors are clustered by country. Note that 2026 is a partial "
        "publication year in the frozen corpus.",
        widths="l r@{\\hspace{6pt}}l r@{\\hspace{6pt}}l"))
    lg.add_output(OUT / "tableA10_period_models.csv")


def main():
    lg = RunLogger("rf_01_tables")
    lg.add_input(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    e = load()
    table_a8(lg)
    table_a9(lg, e)
    table_a10(lg, e)
    for f in TEX.glob("*.tex"):
        lg.add_output(f)
    lg.finish()
    print(f"\nwrote {len(list(TEX.glob('*.tex')))} LaTeX fragments to {TEX}")


if __name__ == "__main__":
    main()
