"""Regenerate every appendix table that depends on the research-need index, on the
corrected nine-component coverage floor."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")
from libpysal.weights import KNN
from spreg import OLS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402
TEX = HERE / "tables"; TEX.mkdir(exist_ok=True)
X = ["log_area", "rd", "need", "log_gdp_pc", "tertiary", "internet", "log_population"]
LBL = {"log_area": "Log harvested area", "rd": "R\\&D expenditure (\\% GDP)",
       "need": "Research-need index", "log_gdp_pc": "Log GDP per capita",
       "tertiary": "Tertiary enrolment", "internet": "Internet use",
       "log_population": "Log population"}


def n(x, d=3):
    return "---" if pd.isna(x) else f"{x:.{d}f}".replace("-", "$-$")
def pf(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"
def wrap(body, cap, lab, notes, w, small=True):
    return ("\\begin{table*}[!t]\n\\centering\n"+("\\footnotesize\n" if small else "")+
            f"\\caption{{{cap}}}\n\\label{{{lab}}}\n"
            f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}{w}@{{}}}}\n\\toprule\n"
            +body+"\n\\bottomrule\n\\end{tabular*}\n\\begin{tablenotes}\n"+notes+
            "\n\\end{tablenotes}\n\\end{table*}\n")
def logit(df, xs, y="studied"):
    M = sm.add_constant(df[xs].astype(float), has_constant="add")
    return sm.Logit(df[y].astype(float), M).fit(disp=0, cov_type="cluster",
                                                cov_kwds={"groups": df.iso3.to_numpy()})
def ppml(df, xs, offset=True, y="n_studies_fractional"):
    d = df[df.area_ha_mean.fillna(0) > 0].copy()
    M = sm.add_constant(d[xs].astype(float), has_constant="add")
    off = np.log(d.area_ha_mean.astype(float)) if offset else None
    return sm.GLM(d[y].astype(float), M, family=sm.families.Poisson(), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": d.iso3.to_numpy()}), d


def main():
    lg = RunLogger("nf_03_appendix_tables")
    corr = pd.read_parquet(ROOT / "outputs" / "revision" / "need_index_corrected_floor.parquet")
    e0 = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    e = e0.drop(columns=["need"]).merge(
        corr[["iso3", "need9_floor9"]].rename(columns={"need9_floor9": "need"}),
        on="iso3", how="left").dropna(subset=["need"])
    lg.add_input(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    Xo = [x for x in X if x != "log_area"]

    # ---------------- A3 specification checks ----------------
    base = logit(e, X)
    cd = pd.get_dummies(e.crop_standard_name, prefix="crop", drop_first=True)
    e2 = pd.concat([e, cd], axis=1)
    fe = logit(e2, X + list(cd.columns))
    wb = pd.read_csv(ROOT / "10_External_Data" / "World_Bank" / "wb_indicators_long.csv")
    ind = {"IT.NET.USER.ZS": "internet_users_pct", "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp",
           "SE.TER.ENRR": "tertiary_enrolment_pct", "SP.POP.TOTL": "population"}
    lat = (wb[wb.indicator.isin(ind)].dropna(subset=["value"]).sort_values("year")
           .groupby(["iso3", "indicator"]).tail(1)
           .pivot(index="iso3", columns="indicator", values="value").rename(columns=ind).reset_index())
    lat["log_gdp_pc"] = np.log(lat.gdp_pc_ppp.where(lat.gdp_pc_ppp > 0))
    lat["log_population"] = np.log(lat.population.where(lat.population > 0))
    lat["tertiary"] = lat.tertiary_enrolment_pct / 100
    lat["internet"] = lat.internet_users_pct / 100
    pan = pd.read_parquet(ROOT / "12_Data_Integration" / "country_crop_panel.parquet")
    wide = pan.merge(corr[["iso3", "need9_floor9"]].rename(columns={"need9_floor9": "need"}),
                     on="iso3", how="left").merge(
        lat[["iso3", "log_gdp_pc", "log_population", "tertiary", "internet"]], on="iso3", how="left")
    wide["log_area"] = np.log(wide.area_ha_mean.where(wide.area_ha_mean > 0))
    wide["studied"] = (wide.n_studies_fractional > 0).astype(float)
    XN = ["log_area", "need", "log_gdp_pc", "tertiary", "internet", "log_population"]
    w_ = wide.dropna(subset=XN + ["studied"])
    exp_ = logit(w_, XN)
    off_, _ = ppml(e, Xo)
    free_, _ = ppml(e, X, offset=False)
    rows = []
    for t in X:
        cells = []
        for m_, ks in [(base, X), (fe, X), (exp_, XN), (off_, Xo), (free_, X)]:
            cells.append(n(m_.params[t]) if t in ks and t in m_.params.index else "\\multicolumn{1}{c}{---}")
        rows.append(f"{LBL[t]} & " + " & ".join(cells) + " \\\\")
    ba, sa = free_.params["log_area"], free_.bse["log_area"]
    wald = ((ba - 1) / sa) ** 2; pw = 1 - stats.chi2.cdf(wald, 1)
    ns = [f"{int(m_.nobs):,}" for m_ in [base, fe, exp_, off_, free_]]
    body = ("& Baseline & Crop FE & No R\\&D, & Intensity & Intensity, \\\\\n"
            "Covariate & participation & participation & expanded & (offset) & free area \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\midrule\nCells & " + " & ".join(ns) + " \\\\")
    (TEX / "tab_A3_specchecks.tex").write_text(wrap(
        body, "Specification checks on the participation and intensity models", "tab:specchecks",
        "\\item[] Coefficients only; intervals are in the replication outputs. Crop fixed effects "
        "identify the harvested-area association from variation across countries within a crop. The "
        f"expanded column drops research and development expenditure, which restores "
        f"{w_.iso3.nunique()-e.iso3.nunique()} countries and {len(w_)-len(e)} systems. The final column "
        f"estimates the harvested-area elasticity rather than fixing it at one: the estimate is "
        f"{ba:.3f} (95\\% CI [{free_.conf_int().loc['log_area',0]:.3f}, "
        f"{free_.conf_int().loc['log_area',1]:.3f}]), and a Wald test rejects unit elasticity "
        f"($\\chi^2_1 = {wald:.2f}$, $p = {pw:.4f}$). Standard errors are clustered by country.",
        "l r r r r r"))
    print(f"A3: expanded {len(w_)} cells / {w_.iso3.nunique()} countries; elasticity {ba:.3f} p={pw:.4f}")

    # ---------------- A4 nested block models ----------------
    blocks = {"M0_constant": [], "M1_scale": ["log_area"], "M2_scale_need": ["log_area", "need"],
              "M3_scale_capacity": ["log_area", "rd", "log_gdp_pc", "tertiary", "internet", "log_population"],
              "M4_full": X}
    NM = {"M0_constant": "Constant only", "M1_scale": "Agricultural scale",
          "M2_scale_need": "Scale $+$ research need", "M3_scale_capacity": "Scale $+$ capacity indicators",
          "M4_full": "Scale $+$ need $+$ capacity"}
    fit = {}
    for k, xs in blocks.items():
        if xs: r = logit(e, xs)
        else:
            M = pd.DataFrame({"const": np.ones(len(e))}, index=e.index)
            r = sm.Logit(e.studied.astype(float), M).fit(disp=0, cov_type="cluster",
                                                         cov_kwds={"groups": e.iso3.to_numpy()})
        fit[k] = (r, len(xs))
    brows = [f"{NM[k]} & {v[1]} & {v[0].llf:.1f} & {n(1-v[0].llf/v[0].llnull if v[1] else 0.0)} & "
             f"{v[0].aic:.1f} & {v[0].bic:.1f} \\\\" for k, v in fit.items()]
    trows = []
    for a, b, lab in [("M1_scale", "M2_scale_need", "Need added to scale"),
                      ("M1_scale", "M3_scale_capacity", "Capacity added to scale"),
                      ("M2_scale_need", "M4_full", "Capacity added to scale and need"),
                      ("M3_scale_capacity", "M4_full", "Need added to scale and capacity")]:
        ra, rb = fit[a][0], fit[b][0]; df = fit[b][1] - fit[a][1]
        st = 2 * (rb.llf - ra.llf); p = 1 - stats.chi2.cdf(st, df)
        dm = (1 - rb.llf / rb.llnull) - (1 - ra.llf / ra.llnull)
        trows.append(f"{lab} & {st:.2f} & {df} & {pf(p)} & {n(dm,4)} & \\\\")
    body = ("Model & Covariates & Log-likelihood & McFadden $R^2$ & AIC & BIC \\\\\n\\midrule\n"
            + "\n".join(brows) + "\n\\midrule\n"
            "\\multicolumn{6}{l}{\\textit{Likelihood-ratio tests}} \\\\\n"
            "Comparison & $\\chi^2$ & df & $p$ & $\\Delta$ McFadden $R^2$ & \\\\\n" + "\n".join(trows))
    (TEX / "tab_A4_blocks.tex").write_text(wrap(
        body, "Nested block models of research participation", "tab:blocks",
        "\\item[] Participation logit on the estimation sample, standard errors clustered by country. "
        "Capacity indicators are research and development expenditure, log GDP per capita, tertiary "
        "enrolment, internet use, and log population. Differences in fit describe how much each block "
        "adds to in-sample explanation and are not evidence of causal importance. The corresponding "
        "comparison for the intensity model, scaled by the estimated dispersion, distinguishes neither "
        "block.", "l r r r r r"))

    # ---------------- A5 country-level ----------------
    c = (e.groupby("iso3").agg(studies=("n_studies_fractional", "sum"), area=("area_ha_mean", "sum"),
                               rd=("rd", "first"), need=("need", "first"),
                               log_gdp_pc=("log_gdp_pc", "first"), tertiary=("tertiary", "first"),
                               internet=("internet", "first"),
                               log_population=("log_population", "first")).reset_index())
    c["log_area"] = np.log(c.area.where(c.area > 0)); c = c.dropna(subset=["log_area"])
    M = sm.add_constant(c[X].astype(float), has_constant="add")
    m_lin = sm.OLS(np.log1p(c.studies), M).fit(cov_type="HC1")
    m_pp = sm.GLM(c.studies, M, family=sm.families.Poisson(), offset=np.log(c.area)).fit(cov_type="HC1")
    rows = []
    for t in X:
        a_, b_ = m_lin, m_pp
        ca, cb = a_.conf_int(), b_.conf_int()
        rows.append(f"{LBL[t]} & {n(a_.params[t])} & [{n(ca.loc[t,0],2)}, {n(ca.loc[t,1],2)}] & "
                    f"{n(b_.params[t])} & [{n(cb.loc[t,0],2)}, {n(cb.loc[t,1],2)}] \\\\")
    body = ("& \\multicolumn{2}{c}{Log study count} & \\multicolumn{2}{c}{Studies per hectare (PPML)} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nCovariate & Estimate & 95\\% CI & Estimate & 95\\% CI \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\midrule\n"
            f"Countries & \\multicolumn{{2}}{{c}}{{{len(c)}}} & \\multicolumn{{2}}{{c}}{{{len(c)}}} \\\\")
    (TEX / "tab_A5_country.tex").write_text(wrap(
        body, "Country-level models of research output", "tab:countrylevel",
        "\\item[] Countries are the unit. The left column regresses the log of one plus the country's "
        "fractional study count on the same covariates as Table~\\ref{tab:reg}; the right column is "
        "Poisson pseudo-maximum likelihood with total harvested area as an exposure offset. "
        "Heteroskedasticity-robust standard errors. Estimates describe association, not effect.",
        "l r@{\\hspace{6pt}}l r@{\\hspace{6pt}}l"))
    print(f"A5: country-level need OLS {m_lin.params['need']:.3f}, PPML {m_pp.params['need']:.3f}")

    # ---------------- A7 diagnostics ----------------
    ry = e[["iso3", "rd_expenditure_gdp_pct_year", "wb_income_group"]].drop_duplicates("iso3")
    byg = (ry.groupby("wb_income_group").agg(n=("iso3", "size"),
           med=("rd_expenditure_gdp_pct_year", "median"), lo=("rd_expenditure_gdp_pct_year", "min"),
           hi=("rd_expenditure_gdp_pct_year", "max")).reset_index())
    pa = "\n".join(f"{r.wb_income_group} & {int(r.n)} & {int(r.med)} & {int(r.lo)} & {int(r.hi)} & "
                   f"\\multicolumn{{2}}{{c}}{{}} \\\\" for _, r in byg.iterrows())
    d = e[e.area_ha_mean.fillna(0) > 0].copy()
    Xp = sm.add_constant(d[Xo].astype(float), has_constant="add")
    off = np.log(d.area_ha_mean.astype(float))
    g = sm.GLM(d.n_studies_fractional.astype(float), Xp, family=sm.families.Poisson(), offset=off)
    cl = g.fit(cov_type="cluster", cov_kwds={"groups": d.iso3.to_numpy()}); nv = g.fit()
    php = float(nv.pearson_chi2 / nv.df_resid); phd = float(nv.deviance / nv.df_resid)
    qse = nv.bse * np.sqrt(php)
    pb = []
    for t in Xo:
        b_, sc, sq = cl.params[t], cl.bse[t], qse[t]
        pb.append(f"{LBL[t]} & {n(b_)} & {n(sc)} & {n(sq)} & {n(sq/sc,1)} & "
                  f"{pf(2*(1-stats.norm.cdf(abs(b_/sc))))} & {pf(2*(1-stats.norm.cdf(abs(b_/sq))))} \\\\")
    lay = pd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    geom = gpd.GeoSeries.from_wkb(lay.geometry) if lay.geometry.dtype == object else lay.geometry
    cent = gpd.GeoDataFrame(lay[["iso3"]], geometry=gpd.points_from_xy(lay.centroid_lon, lay.centroid_lat),
                            crs="EPSG:4326").to_crs("+proj=eqearth")
    ids = lay.iso3.tolist()
    e = e.copy(); e["r_part"] = e.studied.astype(float) - base.predict(
        sm.add_constant(e[X].astype(float), has_constant="add"))
    d2 = d.copy(); d2["r_int"] = d2.n_studies_fractional.astype(float) - cl.predict(Xp)
    cm = e.groupby("iso3", as_index=False).agg(r_part=("r_part", "mean"), **{k: (k, "mean") for k in X})
    cm = cm.merge(d2.groupby("iso3", as_index=False).agg(r_int=("r_int", "mean")), on="iso3").dropna()
    cm = cm[cm.iso3.isin(ids)].reset_index(drop=True)
    pos = [ids.index(i) for i in cm.iso3]
    gg = gpd.GeoDataFrame(cm[["iso3"]], geometry=gpd.points_from_xy(
        cent.geometry.x.values[pos], cent.geometry.y.values[pos]))
    w6 = KNN.from_dataframe(gg, k=6, ids=cm.iso3.tolist()); w6.transform = "r"
    lmrec = {}
    for col, nm in [("r_part", "Participation logit"), ("r_int", "Intensity PPML")]:
        o = OLS(cm[[col]].values, cm[X].values, w=w6, spat_diag=True, moran=True,
                name_y=col, name_x=X, name_w="knn6")
        lmrec[nm] = {k: getattr(o, k) for k in ["lm_lag", "rlm_lag", "lm_error", "rlm_error"]}
    pc = []
    for lab, key in [("LM lag", "lm_lag"), ("Robust LM lag", "rlm_lag"),
                     ("LM error", "lm_error"), ("Robust LM error", "rlm_error")]:
        cells = [f"{lmrec[nm][key][0]:.3f} & {pf(lmrec[nm][key][1])}" for nm in
                 ["Participation logit", "Intensity PPML"]]
        pc.append(f"{lab} & " + " & ".join(cells) + " & \\\\")
    body = ("\\multicolumn{7}{l}{\\textit{Panel A. Reference year of the R\\&D expenditure observation}} \\\\\n"
            "Income group & Countries & Median & Earliest & Latest & \\multicolumn{2}{c}{} \\\\\n\\midrule\n"
            + pa + "\n\\midrule\n"
            "\\multicolumn{7}{l}{\\textit{Panel B. Intensity model: clustered against quasi-Poisson standard errors}} \\\\\n"
            "Covariate & Estimate & SE (clustered) & SE (quasi-Poisson) & Ratio & $p$ (clust.) & $p$ (quasi) \\\\\n"
            "\\midrule\n" + "\n".join(pb) + "\n\\midrule\n"
            "\\multicolumn{7}{l}{\\textit{Panel C. Lagrange Multiplier tests on country-mean residuals, $k=6$}} \\\\\n"
            "& \\multicolumn{2}{c}{Participation logit} & \\multicolumn{2}{c}{Intensity PPML} & \\multicolumn{2}{c}{} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Test & Statistic & $p$ & Statistic & $p$ & \\multicolumn{2}{c}{} \\\\\n\\midrule\n" + "\n".join(pc))
    (TEX / "tab_A7_diagnostics.tex").write_text(wrap(
        body, "Covariate vintage, dispersion, and spatial diagnostics", "tab:diagnostics",
        "\\item[] Panel A gives the year of the most recent research and development expenditure "
        "observation for each estimation country. Panel B compares the reported cluster-robust standard "
        f"errors with quasi-Poisson standard errors formed from the Pearson dispersion, "
        f"$\\hat\\phi_{{P}} = {php:.0f}$. The deviance-based dispersion for the same fit is "
        f"$\\hat\\phi_{{D}} = {phd:.1f}$. The two disagree by more than two orders of magnitude because "
        "the Pearson statistic is dominated by a small number of systems with very large counts, so a "
        "constant variance-to-mean ratio does not describe these data and the quasi-Poisson interval is "
        "conservative rather than correct. Cluster-robust errors, which assume no variance function, "
        "remain the reported choice. Panel C reports Lagrange Multiplier tests for a spatial lag and a "
        "spatial error process; none rejects at the 5\\% level.", "l r r r r r r"))
    print(f"A7: phi_P={php:.1f} phi_D={phd:.2f}")

    # ---------------- A8 period models ----------------
    scc = pd.read_parquet(ROOT / "12_Data_Integration" / "study_country_crop_dataset.parquet")
    cols = {}
    for lab, lo, hi in [("2000-2019", 2000, 2019), ("2020-2026", 2020, 2026)]:
        sub = scc[(scc.publication_year >= lo) & (scc.publication_year <= hi)]
        agg = (sub.groupby(["iso3", "crop_standard_name"], as_index=False).fractional_weight.sum()
               .rename(columns={"fractional_weight": "rp"}))
        ep = e.merge(agg, on=["iso3", "crop_standard_name"], how="left")
        ep["y"] = (ep.rp.fillna(0) > 0).astype(float)
        cols[lab] = (logit(ep, X, y="y"), int(ep.y.sum()))
    rows = []
    for t in X:
        cells = []
        for lab in ["2000-2019", "2020-2026"]:
            m_, _ = cols[lab]; ci = m_.conf_int()
            cells.append(f"{n(m_.params[t])} & [{n(ci.loc[t,0],2)}, {n(ci.loc[t,1],2)}]")
        rows.append(f"{LBL[t]} & " + " & ".join(cells) + " \\\\")
    body = ("& \\multicolumn{2}{c}{Published 2000--2019} & \\multicolumn{2}{c}{Published 2020--2026} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nCovariate & Estimate & 95\\% CI & Estimate & 95\\% CI \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\midrule\n"
            f"Country-crop systems & \\multicolumn{{2}}{{c}}{{{int(cols['2000-2019'][0].nobs):,}}} & "
            f"\\multicolumn{{2}}{{c}}{{{int(cols['2020-2026'][0].nobs):,}}} \\\\\n"
            f"Systems carrying a study & \\multicolumn{{2}}{{c}}{{{cols['2000-2019'][1]}}} & "
            f"\\multicolumn{{2}}{{c}}{{{cols['2020-2026'][1]}}} \\\\")
    (TEX / "tab_A8_periods.tex").write_text(wrap(
        body, "Participation models by publication period", "tab:periods",
        "\\item[] The outcome is whether a country-crop system carries a study published in the stated "
        "period; the covariate panel is unchanged, because the capacity and need indicators are single "
        "most-recent observations rather than annual series. The models test whether the associations are "
        "stable across the literature's recent expansion, not whether covariates changed. Standard errors "
        "are clustered by country. Note that 2026 is a partial publication year in the corpus.",
        "l r@{\\hspace{6pt}}l r@{\\hspace{6pt}}l"))
    print(f"A8: area {cols['2000-2019'][0].params['log_area']:.3f} / "
          f"{cols['2020-2026'][0].params['log_area']:.3f}")
    for f in TEX.glob("*.tex"): lg.add_output(f)
    lg.finish()
    print(f"\nwrote {len(list(TEX.glob('*.tex')))} appendix tables")


if __name__ == "__main__":
    main()
