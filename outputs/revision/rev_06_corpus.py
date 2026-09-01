"""Revision analyses: corpus accounting, panel definition, concentration universe,
selection diagnostics, and temporal checks.

  Step 13 / R1-M4 - exact, reproducible definition of the 2,616-cell panel, and the
                    count of cells carrying no FAOSTAT denominator at all
  Step 14 / R1-Mo8 - concentration measured on the research-geography universe rather
                    than the FAOSTAT-matched panel
  Step 19 / item 2 - resolved versus unresolved study comparison with tests
  Step 20          - recent-period sensitivity
  Step 21          - partial 2026 coverage
  item  4          - R&D reference-year distribution by World Bank income group
  item  7          - Gini of country research counts by period
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

OUT = HERE


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    lg = RunLogger("rev_06_corpus")
    pan = pd.read_parquet(ROOT / "12_Data_Integration" / "country_crop_panel.parquet")
    scc = pd.read_parquet(ROOT / "12_Data_Integration" / "study_country_crop_dataset.parquet")
    sld = pd.read_parquet(ROOT / "12_Data_Integration" / "study_level_dataset.parquet")
    fao = pd.read_parquet(ROOT / "10_External_Data" / "FAOSTAT" / "faostat_country_crop_year.parquet")
    layer = pd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    for f in ["12_Data_Integration/country_crop_panel.parquet",
              "12_Data_Integration/study_country_crop_dataset.parquet",
              "12_Data_Integration/study_level_dataset.parquet"]:
        lg.add_input(ROOT / f)

    # ---------------------------------------------------------------- Step 13: panel rule
    m49 = layer.dropna(subset=["m49"])[["m49", "iso3"]].copy()
    m49["m49"] = m49["m49"].astype(str)
    fao["m49"] = fao["m49"].astype(str)
    f2 = fao.merge(m49, on="m49", how="inner")
    c199, c25 = set(pan.iso3), set(pan.crop_standard_name)
    f2 = f2[f2.iso3.isin(c199) & f2.crop_standard_name.isin(c25)]
    panset = set(map(tuple, pan[["iso3", "crop_standard_name"]].values))
    all_years = set(map(tuple, f2[["iso3", "crop_standard_name"]].drop_duplicates().values))
    ref = f2[(f2.year >= 2018) & (f2.year <= 2022)]
    ref_cells = set(map(tuple, ref[["iso3", "crop_standard_name"]].drop_duplicates().values))
    no_fao = pan[pan[["area_ha_mean", "production_t_mean", "yield_t_ha_mean"]].isna().all(axis=1)]
    rule = pd.DataFrame([
        dict(component="199 countries x 25 crops (full cross)", cells=len(c199) * len(c25)),
        dict(component="FAOSTAT records, any year 1990-2024", cells=len(all_years)),
        dict(component="FAOSTAT records, reference window 2018-2022", cells=len(ref_cells)),
        dict(component="in panel but absent from FAOSTAT extract", cells=len(panset - all_years)),
        dict(component="in FAOSTAT extract but absent from panel", cells=len(all_years - panset)),
        dict(component="PANEL TOTAL", cells=len(panset)),
        dict(component="panel cells with NO FAOSTAT area/production/yield", cells=len(no_fao)),
        dict(component="  of which carry any research", cells=int(no_fao.has_any_study.sum())),
        dict(component="panel cells with a positive area denominator", cells=int((pan.area_ha_mean > 0).sum())),
    ])
    rule.to_csv(OUT / "panel_definition_decomposition.csv", index=False)
    print("PANEL DEFINITION")
    print(rule.to_string(index=False))
    absent = sorted(panset - all_years)
    print(f"\ncountries behind the {len(absent)} absent cells: "
          f"{sorted({c for c, _ in absent})}")
    print(f"cells present in FAOSTAT but excluded from the panel: {sorted(all_years - panset)}")

    # impact of the empty cells on the reported shares
    supported = pan[~pan.index.isin(no_fao.index)]
    print(f"\nreported: 485 of 2,616 cells carry research ({485/2616:.1%})")
    print(f"on FAOSTAT-supported cells only: {int(supported.has_any_study.sum())} of "
          f"{len(supported)} ({supported.has_any_study.sum()/len(supported):.1%})")

    # ---------------------------------------------------------------- Step 14: universe
    alloc_ctry = scc.groupby("iso3", as_index=False).fractional_weight.sum()
    lay_ctry = layer[["iso3"]].merge(alloc_ctry, on="iso3", how="left").fillna({"fractional_weight": 0})
    pan_ctry = pan.groupby("iso3", as_index=False).n_studies_fractional.sum()
    rows = [
        dict(universe="2,616-cell FAOSTAT panel (199 countries)", n=pan_ctry.iso3.nunique(),
             total_weight=pan_ctry.n_studies_fractional.sum(),
             gini=gini(pan_ctry.n_studies_fractional)),
        dict(universe="full accepted-country allocation (all allocated countries)",
             n=alloc_ctry.iso3.nunique(), total_weight=alloc_ctry.fractional_weight.sum(),
             gini=gini(alloc_ctry.fractional_weight)),
        dict(universe="spatial layer, 195 countries", n=len(lay_ctry),
             total_weight=lay_ctry.fractional_weight.sum(),
             gini=gini(lay_ctry.fractional_weight)),
    ]
    # panel countries, conditional on carrying research
    nz = pan_ctry[pan_ctry.n_studies_fractional > 0]
    rows.append(dict(universe="panel countries carrying research", n=len(nz),
                     total_weight=nz.n_studies_fractional.sum(),
                     gini=gini(nz.n_studies_fractional)))
    uni = pd.DataFrame(rows)
    uni.to_csv(OUT / "concentration_universe.csv", index=False)
    print("\nCONCENTRATION UNIVERSE")
    print(uni.round(4).to_string(index=False))

    # ---------------------------------------------------------------- Step 19 / item 2
    allocated = set(scc.openalex_id.unique())
    sld["resolved"] = sld.openalex_id.isin(allocated)
    a, b = sld[sld.resolved], sld[~sld.resolved]
    comp = []

    def mw(x, y, label, unit):
        x, y = x.dropna(), y.dropna()
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        comp.append(dict(variable=label, unit=unit, resolved_mean=x.mean(),
                         resolved_median=x.median(), unresolved_mean=y.mean(),
                         unresolved_median=y.median(), n_resolved=len(x), n_unresolved=len(y),
                         test="Mann-Whitney U", statistic=u, p_value=p))
    mw(a.publication_year, b.publication_year, "Publication year", "year")
    mw(a.cited_by_count, b.cited_by_count, "Citation count", "citations")
    mw(a.n_authors, b.n_authors, "Number of authors", "authors")
    # abstract availability and document type: chi-square
    for col, lab in [("abstract", "Abstract available"), ("is_conference", "Conference item"),
                     ("is_preprint", "Preprint")]:
        av = a[col].notna() if col == "abstract" else a[col].astype(bool)
        bv = b[col].notna() if col == "abstract" else b[col].astype(bool)
        tab = np.array([[av.sum(), (~av).sum()], [bv.sum(), (~bv).sum()]])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        comp.append(dict(variable=lab, unit="share", resolved_mean=av.mean(),
                         resolved_median=np.nan, unresolved_mean=bv.mean(),
                         unresolved_median=np.nan, n_resolved=len(a), n_unresolved=len(b),
                         test="chi-square", statistic=chi2, p_value=p))
    cmp_ = pd.DataFrame(comp)
    cmp_.to_csv(OUT / "resolved_vs_unresolved.csv", index=False)
    print("\nRESOLVED vs UNRESOLVED STUDIES")
    print(cmp_[["variable", "resolved_mean", "unresolved_mean", "test", "p_value"]]
          .round(4).to_string(index=False))
    dt = (pd.crosstab(sld.type, sld.resolved, normalize="columns") * 100).round(1)
    dt.columns = ["unresolved_%", "resolved_%"]
    dt.to_csv(OUT / "resolved_vs_unresolved_doctype.csv")
    print("\ndocument type (% within group)")
    print(dt.sort_values("resolved_%", ascending=False).head(8).to_string())

    # ---------------------------------------------------------------- Step 21 + item 7
    yr = scc.groupby("publication_year", as_index=False).fractional_weight.sum()
    print("\nPUBLICATION YEAR COVERAGE (allocated weight)")
    print(yr[yr.publication_year >= 2020].to_string(index=False))
    periods = [("2000-2017", 2000, 2017), ("2018-2022", 2018, 2022), ("2023-2025", 2023, 2025),
               ("2026 (partial)", 2026, 2026)]
    grows = []
    for lab, lo, hi in periods:
        s = scc[(scc.publication_year >= lo) & (scc.publication_year <= hi)]
        g = s.groupby("iso3", as_index=False).fractional_weight.sum()
        full = layer[["iso3"]].merge(g, on="iso3", how="left").fillna({"fractional_weight": 0})
        grows.append(dict(period=lab, studies=s.fractional_weight.sum(),
                          countries_with_research=int((full.fractional_weight > 0).sum()),
                          gini_all_countries=gini(full.fractional_weight)))
    gp = pd.DataFrame(grows)
    gp.to_csv(OUT / "temporal_gini.csv", index=False)
    print("\nCONCENTRATION BY PERIOD (all 195 layer countries in the denominator)")
    print(gp.round(4).to_string(index=False))

    # ---------------------------------------------------------------- item 4: R&D years
    e = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    ry = e[["iso3", "rd_expenditure_gdp_pct_year", "wb_income_group"]].drop_duplicates("iso3")
    byg = (ry.groupby("wb_income_group")
           .agg(countries=("iso3", "size"), median_year=("rd_expenditure_gdp_pct_year", "median"),
                min_year=("rd_expenditure_gdp_pct_year", "min"),
                max_year=("rd_expenditure_gdp_pct_year", "max"))
           .reset_index())
    byg.to_csv(OUT / "rd_reference_years.csv", index=False)
    print("\nR&D REFERENCE YEAR BY INCOME GROUP (142 estimation countries)")
    print(byg.to_string(index=False))
    dist = ry.rd_expenditure_gdp_pct_year.value_counts().sort_index()
    print(f"\nreference-year distribution: min {ry.rd_expenditure_gdp_pct_year.min():.0f}, "
          f"median {ry.rd_expenditure_gdp_pct_year.median():.0f}, "
          f"max {ry.rd_expenditure_gdp_pct_year.max():.0f}; "
          f"{int((ry.rd_expenditure_gdp_pct_year < 2015).sum())} countries older than 2015")

    for f in ["panel_definition_decomposition.csv", "concentration_universe.csv",
              "resolved_vs_unresolved.csv", "resolved_vs_unresolved_doctype.csv",
              "temporal_gini.csv", "rd_reference_years.csv"]:
        lg.add_output(OUT / f)
    lg.finish()
    print("\nrev_06_corpus complete")


if __name__ == "__main__":
    main()
