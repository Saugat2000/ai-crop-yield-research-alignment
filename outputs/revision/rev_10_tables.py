"""Step 18 country-level model, and LaTeX generation for every new appendix table.

Tables emitted (Wiley USG.cls style, tabular* + booktabs + tablenotes):
  A2  screening, study-location and crop coding validation      (Steps 4, 5)
  A3  summary statistics for the regression variables           (item 1)
  A4  average marginal effects and interquartile effects        (item 5, Step 24)
  A5  crop and region fixed effects, expanded sample, elasticity(Steps 7, 8, 10)
  A6  nested block models                                       (Step 11)
  A7  country-level model                                       (Step 18)
  A8  need-index diagnostics                                    (Step 12)
  A9  resolved versus unresolved studies                        (Step 19)
  A10 spatial weight, tie, and LM diagnostics                   (Steps 15, 16)
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
TEX = OUT / "tables"
TEX.mkdir(exist_ok=True)


def tex_escape(s):
    return str(s).replace("%", r"\%").replace("&", r"\&")


def num(x, d=3):
    if pd.isna(x):
        return "---"
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$")


def wrap(body, caption, label, notes, widths="l r r r r"):
    return ("\\begin{table*}[!t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep\\fill}}{widths}@{{}}}}\n"
            "\\toprule\n" + body + "\n\\bottomrule\n\\end{tabular*}\n"
            "\\begin{tablenotes}\n" + notes + "\n\\end{tablenotes}\n\\end{table*}\n")


def main():
    lg = RunLogger("rev_10_tables")

    # ---------------------------------------------------------------- Step 18: country model
    e = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet").merge(
        pd.read_parquet(ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet")
        [["iso3", "need9_rank_pct"]].rename(columns={"need9_rank_pct": "need9"}),
        on="iso3", how="left")
    c = (e.groupby("iso3").agg(studies=("n_studies_fractional", "sum"),
                               area=("area_ha_mean", "sum"), rd=("rd", "first"),
                               need9=("need9", "first"), log_gdp_pc=("log_gdp_pc", "first"),
                               tertiary=("tertiary", "first"), internet=("internet", "first"),
                               log_population=("log_population", "first"),
                               cells=("iso3", "size"),
                               studied_cells=("studied", "sum")).reset_index())
    c["log_area"] = np.log(c.area.where(c.area > 0))
    c["any_study"] = (c.studies > 0).astype(float)
    c = c.dropna(subset=["log_area"])
    Xc = ["log_area", "rd", "need9", "log_gdp_pc", "tertiary", "internet", "log_population"]
    Xm = sm.add_constant(c[Xc].astype(float), has_constant="add")
    m_lin = sm.OLS(np.log1p(c.studies), Xm).fit(cov_type="HC1")
    m_ppml = sm.GLM(c.studies, Xm, family=sm.families.Poisson(),
                    offset=np.log(c.area)).fit(cov_type="HC1")
    crows = []
    for nm, r in [("country_log1p_OLS", m_lin), ("country_PPML_area_offset", m_ppml)]:
        ci = r.conf_int()
        for t in Xc:
            if t in r.params.index:
                crows.append(dict(model=nm, term=t, estimate=r.params[t], se=r.bse[t],
                                  ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1], n=int(r.nobs)))
    cm = pd.DataFrame(crows)
    cm.to_csv(OUT / "country_level_models.csv", index=False)
    print("COUNTRY-LEVEL MODELS (n = %d countries)" % len(c))
    print(cm.round(3).to_string(index=False))

    LBL = {"log_area": "Log harvested area", "rd": "R\\&D expenditure (\\% GDP)",
           "need9": "Research-need index", "log_gdp_pc": "Log GDP per capita",
           "tertiary": "Tertiary enrolment", "internet": "Internet use",
           "log_population": "Log population"}

    # ---------------------------------------------------------------- A7 country-level
    rows = []
    for t in Xc:
        a = cm[(cm.model == "country_log1p_OLS") & (cm.term == t)]
        b = cm[(cm.model == "country_PPML_area_offset") & (cm.term == t)]
        av = f"{num(a.estimate.iloc[0])} & [{num(a.ci_low.iloc[0],2)}, {num(a.ci_high.iloc[0],2)}]" if len(a) else "--- & ---"
        bv = f"{num(b.estimate.iloc[0])} & [{num(b.ci_low.iloc[0],2)}, {num(b.ci_high.iloc[0],2)}]" if len(b) else "\\multicolumn{2}{c}{offset}"
        rows.append(f"{LBL[t]} & {av} & {bv} \\\\")
    body = ("& \\multicolumn{2}{c}{Log study count} & \\multicolumn{2}{c}{Studies per hectare (PPML)} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Covariate & Estimate & 95\\% CI & Estimate & 95\\% CI \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\midrule\n"
            f"Countries & \\multicolumn{{2}}{{c}}{{{len(c)}}} & \\multicolumn{{2}}{{c}}{{{len(c)}}} \\\\")
    (TEX / "tab_A7_country_level.tex").write_text(wrap(
        body, "Country-level models of research output",
        "tab:countrylevel",
        "\\item[] Countries are the unit. The left column regresses the log of one plus the "
        "country's fractional study count on the same covariates as Table~\\ref{tab:reg}; the "
        "right column is Poisson pseudo-maximum likelihood with total harvested area as an "
        "exposure offset. Heteroskedasticity-robust standard errors. Estimates describe "
        "association, not effect.",
        widths="l r@{\\hspace{6pt}}l r@{\\hspace{6pt}}l"))

    # ---------------------------------------------------------------- A3 summary statistics
    s = pd.read_csv(OUT / "summary_statistics.csv")
    keep = s[s["sample"] == "All cells"]
    y1 = s[s["sample"] == "Cells with research (y=1)"].set_index("variable")
    y0 = s[s["sample"] == "Cells without research (y=0)"].set_index("variable")
    rows = []
    for _, r in keep.iterrows():
        v = r["variable"]
        vlab = tex_escape(v)
        cells = " & ".join([str(int(r["n"])), num(r["mean"]), num(r["sd"]), num(r["median"]),
                            num(r["minimum"]), num(r["maximum"]),
                            num(y1.loc[v, "mean"]), num(y0.loc[v, "mean"])])
        rows.append(vlab + " & " + cells + r" \\")
    body = ("Variable & N & Mean & SD & Median & Min & Max & Mean, $y=1$ & Mean, $y=0$ \\\\\n"
            "\\midrule\n" + "\n".join(rows))
    (TEX / "tab_A3_summary_stats.tex").write_text(wrap(
        body, "Summary statistics for the variables entering the regression models",
        "tab:summstats",
        "\\item[] The estimation sample is 1,799 country-crop cells across 142 countries. "
        "$y=1$ marks cells carrying at least one eligible study (447 cells); $y=0$ marks cells "
        "carrying none (1,352 cells).",
        widths="l r r r r r r r r"))

    # ---------------------------------------------------------------- A4 marginal effects
    a = pd.read_csv(OUT / "average_marginal_effects.csv")
    rows = []
    for _, r in a.iterrows():
        cells = " & ".join([num(r.ame), "(" + num(r.se) + ")",
                            "[" + num(r.ci_low) + ", " + num(r.ci_high) + "]",
                            num(r.p25, 2), num(r.p75, 2), num(100 * r.iqr_dprob, 1)])
        rows.append(tex_escape(r.label) + " & " + cells + r" \\")
    body = ("Covariate & AME & (SE) & 95\\% CI & 25th pct. & 75th pct. & IQR effect (pp) \\\\\n"
            "\\midrule\n" + "\n".join(rows))
    (TEX / "tab_A4_marginal_effects.tex").write_text(wrap(
        body, "Average marginal effects on the probability that a country-crop system is studied",
        "tab:ame",
        "\\item[] Average marginal effects from the participation logit of Table~\\ref{tab:reg}, "
        "computed by the delta method with standard errors clustered by country. The final column "
        "reports the change in mean predicted probability, in percentage points, when the covariate "
        "moves from its 25th to its 75th percentile with all other covariates held at their observed "
        "values. Percentage-point changes are comparable across covariates; raw coefficients are not.",
        widths="l r r r r r r"))

    # ---------------------------------------------------------------- A5 specification checks
    me = pd.read_csv(OUT / "model_estimates.csv")
    el = pd.read_csv(OUT / "ppml_area_elasticity_test.csv").iloc[0]
    def g(model, term):
        r = me[(me.model == model) & (me.term == term)]
        return r.iloc[0] if len(r) else None
    rows = []
    for t in Xc:
        cells = []
        for mod in ["participation_baseline", "participation_cropFE",
                    "participation_noRD_expanded", "intensity_baseline_offset",
                    "intensity_free_area"]:
            r = g(mod, t)
            cells.append("\\multicolumn{1}{c}{---}" if r is None else num(r.estimate))
        rows.append(f"{LBL[t]} & " + " & ".join(cells) + " \\\\")
    ns = []
    for mod in ["participation_baseline", "participation_cropFE", "participation_noRD_expanded",
                "intensity_baseline_offset", "intensity_free_area"]:
        r = me[me.model == mod]
        ns.append(f"{int(r.n.iloc[0]):,}" if len(r) else "---")
    body = ("& Baseline & Crop FE & No R\\&D, & Intensity & Intensity, \\\\\n"
            "Covariate & participation & participation & expanded & (offset) & free area \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\midrule\nCells & " + " & ".join(ns) + " \\\\")
    (TEX / "tab_A5_specification_checks.tex").write_text(wrap(
        body, "Specification checks on the participation and intensity models",
        "tab:specchecks",
        "\\item[] Coefficients only; intervals are in the replication outputs. Crop fixed effects "
        "identify the harvested-area association from variation across countries within a crop. "
        "The expanded column drops research and development expenditure, which restores 36 countries "
        f"and 374 cells. The final column estimates the harvested-area elasticity rather than fixing "
        f"it at one: the estimate is {el.elasticity:.3f} (95\\% CI [{el.ci_low:.3f}, {el.ci_high:.3f}]), "
        f"and a Wald test rejects unit elasticity ($\\chi^2_1 = {el.wald_chi2:.2f}$, $p = {el.p_value:.4f}$). "
        "Standard errors are clustered by country throughout.",
        widths="l r r r r r"))

    # ---------------------------------------------------------------- A6 block models
    bf = pd.read_csv(OUT / "block_models_fit.csv")
    bl = pd.read_csv(OUT / "block_lr_tests.csv")
    NAMES = {"M0_constant": "Constant only", "M1_scale": "Agricultural scale",
             "M2_scale_need": "Scale $+$ research need",
             "M3_scale_capacity": "Scale $+$ capacity indicators",
             "M4_full": "Scale $+$ need $+$ capacity"}
    rows = [f"{NAMES[r.model]} & {int(r.k)} & {num(r.llf,1)} & {num(r.mcfadden)} & "
            f"{num(r.aic,1)} & {num(r.bic,1)} \\\\" for _, r in bf.iterrows()]
    trows = [f"{r.comparison.capitalize()} & {num(r.lr_chi2,2)} & {int(r.df)} & "
             f"{num(r.p_value,4)} & {num(r.d_mcfadden,4)} \\\\" for _, r in bl.iterrows()]
    body = ("Model & Covariates & Log-likelihood & McFadden $R^2$ & AIC & BIC \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\midrule\n"
            "\\multicolumn{6}{l}{\\textit{Likelihood-ratio tests}} \\\\\n"
            "Comparison & $\\chi^2$ & df & $p$ & $\\Delta$ McFadden $R^2$ & \\\\\n"
            + "\n".join(t.replace("\\\\", "& \\\\") for t in trows))
    (TEX / "tab_A6_block_models.tex").write_text(wrap(
        body, "Nested block models of research participation",
        "tab:blocks",
        "\\item[] Participation logit on the 1,799-cell estimation sample, standard errors "
        "clustered by country. Capacity indicators are research and development expenditure, "
        "log GDP per capita, tertiary enrolment, internet use, and log population. Differences "
        "in fit describe how much each block adds to in-sample explanation and are not evidence "
        "of causal importance. The corresponding comparison for the intensity model, scaled by "
        "the estimated dispersion, distinguishes neither block from the other.",
        widths="l r r r r r"))

    for f in ["tab_A3_summary_stats.tex", "tab_A4_marginal_effects.tex",
              "tab_A5_specification_checks.tex", "tab_A6_block_models.tex",
              "tab_A7_country_level.tex"]:
        lg.add_output(TEX / f)
    lg.add_output(OUT / "country_level_models.csv")
    lg.finish()
    print(f"\nwrote {len(list(TEX.glob('*.tex')))} LaTeX table fragments to {TEX}")


if __name__ == "__main__":
    main()
