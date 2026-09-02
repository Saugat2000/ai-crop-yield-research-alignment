"""Export the plotted data behind each figure in the submitted manuscript.

The submitted figures were drawn by the correction cascade (`outputs/geofix/gf_03_figures.py`,
`outputs/needfix/nf_02_figures.py`, `outputs/revision/rev_08_figures.py`,
`outputs/revision_final/rf_02_figures.py`). Those scripts draw from the corrected panel but do
not write per-figure data sidecars. This script recomputes, with the same source files and the
same transformations, the series each figure plots, and writes one CSV per figure next to it.

It asserts the recomputed values against `outputs/geofix/headline_numbers.json` and exits
non-zero on disagreement, so a sidecar can never drift from the reported results.

Run:  python 21_Figures/export_plotted_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import RunLogger  # noqa: E402

HERE = ROOT / "21_Figures"
GEOFIX = ROOT / "outputs" / "geofix"
REVISION = ROOT / "outputs" / "revision"

BREAKS = [0, 1, 5, 20, 100, np.inf]
CLASS_LABELS = ["0 (no resolved study)", "0 < n <= 1", "1 < n <= 5",
                "5 < n <= 20", "20 < n <= 100", "n > 100"]


def intensity_class(v: float) -> int:
    if v == 0:
        return 0
    for i, (a, b) in enumerate(zip(BREAKS[:-1], BREAKS[1:]), start=1):
        if a < v <= b:
            return i
    return len(BREAKS) - 1


def check(name: str, got, want, tol=5e-3) -> None:
    ok = abs(float(got) - float(want)) <= tol
    print(f"    {'ok  ' if ok else 'FAIL'} {name}: {got} (headline {want})")
    if not ok:
        raise SystemExit(f"plotted data disagrees with headline_numbers.json: {name}")


def main() -> None:
    lg = RunLogger("export_plotted_data")
    head = json.loads((GEOFIX / "headline_numbers.json").read_text())[0]
    lg.add_input(GEOFIX / "headline_numbers.json")

    panel = pd.read_parquet(GEOFIX / "country_crop_panel_corrected.parquet")
    lg.add_input(GEOFIX / "country_crop_panel_corrected.parquet")

    # ---- Figure 1: fractional studies per country, mapped in six classes -------------
    country = (panel.groupby("iso3", as_index=False)
                    .n_studies_fractional.sum()
                    .sort_values("n_studies_fractional", ascending=False))
    country["intensity_class"] = country.n_studies_fractional.map(intensity_class)
    country["intensity_label"] = country.intensity_class.map(dict(enumerate(CLASS_LABELS)))
    country["share_of_world_pct"] = 100 * country.n_studies_fractional / country.n_studies_fractional.sum()
    country.to_csv(HERE / "fig_01_research_intensity_map_data.csv", index=False)

    print("  Figure 1")
    check("countries with research", int((country.n_studies_fractional > 0).sum()),
          head["countries_with_research"], tol=0)
    top3 = country.head(3)
    check("top-1 share %", top3.share_of_world_pct.iloc[0], head["top1_share"], tol=1e-2)
    check("top-3 share %", top3.share_of_world_pct.sum(), head["top3_share"], tol=1e-2)
    for iso, val in zip(head["top3_names"], head["top3_vals"]):
        check(f"{iso} fractional count",
              round(float(country.loc[country.iso3 == iso, "n_studies_fractional"].iloc[0]), 1), val, tol=0.05)

    # ---- Figure 2: research attention against harvested area, by crop ---------------
    crop = (panel.groupby("crop_standard_name", as_index=False)
                 .agg(attention=("n_studies_fractional", "sum"), area_ha=("area_ha_mean", "sum")))
    crop["attention_share_pct"] = 100 * crop.attention / crop.attention.sum()
    crop["area_share_pct"] = 100 * crop.area_ha / crop.area_ha.sum()
    crop = crop.sort_values("attention_share_pct", ascending=False)
    crop["plotted_in_figure"] = np.arange(len(crop)) < 15
    crop.to_csv(HERE / "fig_02_v2_crop_attention_area_data.csv", index=False)

    print("  Figure 2")
    check("top-3 crop attention share %", crop.attention_share_pct.head(3).sum(),
          head["top3crop_share"], tol=1e-2)
    for name, val in head["top3crop"]:
        check(f"{name} fractional attention",
              round(float(crop.loc[crop.crop_standard_name == name, "attention"].iloc[0]), 1), val, tol=0.05)

    # ---- Figure 3: research share against harvested-area share ----------------------
    tot_r = panel.n_studies_fractional.sum()
    tot_a = panel.area_ha_mean.sum()
    cells = panel.dropna(subset=["area_ha_mean"])
    cells = cells[cells.area_ha_mean > 0].copy()
    cells["research_share_pct"] = 100 * cells.n_studies_fractional / tot_r
    cells["area_share_pct"] = 100 * cells.area_ha_mean / tot_a
    cells["panel"] = "country-crop system"
    ct = (panel.groupby("iso3", as_index=False)
               .agg(n_studies_fractional=("n_studies_fractional", "sum"),
                    area_ha_mean=("area_ha_mean", "sum")))
    ct = ct[ct.area_ha_mean > 0].copy()
    ct["research_share_pct"] = 100 * ct.n_studies_fractional / tot_r
    ct["area_share_pct"] = 100 * ct.area_ha_mean / tot_a
    ct["panel"] = "country"
    cols = ["panel", "iso3", "crop_standard_name", "n_studies_fractional",
            "area_ha_mean", "research_share_pct", "area_share_pct"]
    fig3 = pd.concat([cells.reindex(columns=cols), ct.reindex(columns=cols)], ignore_index=True)
    fig3.to_csv(HERE / "fig_03_v2_scale_alignment_data.csv", index=False)

    print("  Figure 3")
    check("cells with positive harvested area", int(len(cells)), head["pos_area_cells"], tol=0)
    check("zero-research cells with harvested area",
          int((cells.n_studies_fractional == 0).sum()), head["zero_cells_posarea"], tol=0)
    check("their share of panel harvested area %",
          100 * cells.loc[cells.n_studies_fractional == 0, "area_ha_mean"].sum() / tot_a,
          head["zero_area_share"], tol=1e-2)

    # ---- Figures 4 and 5: need against research, and the LISA classification --------
    lisa = pd.read_csv(GEOFIX / "gap_corrected_lisa.csv")
    lg.add_input(GEOFIX / "gap_corrected_lisa.csv")
    med_need, med_res = lisa.need_pct.median(), lisa.research_pct.median()
    lisa["high_need_low_research"] = (lisa.need_pct >= med_need) & (lisa.research_pct <= med_res)
    lisa.to_csv(HERE / "fig_04_v2_need_vs_research_data.csv", index=False)
    lisa[["iso3", "need_pct", "research_pct", "gap", "lisa_cat_gap9"]].to_csv(
        HERE / "fig_05_v2_mismatch_lisa_data.csv", index=False)

    print("  Figures 4 and 5")
    check("countries with a need index", int(len(lisa)), head["n_gap"], tol=0)
    check("high need, low research", int(lisa.high_need_low_research.sum()), head["quadrant"], tol=0)
    counts = lisa.lisa_cat_gap9.value_counts()
    for cat, key in (("High-High", "HH"), ("Low-Low", "LL"),
                     ("High-Low", "HL"), ("Low-High", "LH")):
        check(f"LISA {cat}", int(counts.get(cat, 0)), head[key], tol=0)

    # ---- Figure 6: coefficient plot -------------------------------------------------
    est = pd.read_csv(GEOFIX / "model_estimates_geofix.csv")
    lg.add_input(GEOFIX / "model_estimates_geofix.csv")
    est.to_csv(HERE / "fig_06_v2_coefficients_data.csv", index=False)
    print(f"  Figure 6: {len(est)} estimates copied from geofix/model_estimates_geofix.csv")

    # ---- Appendix figures already carry their plotted data --------------------------
    for src, dst in ((REVISION / "temporal_gini.csv", "fig_A4_temporal_concentration_data.csv"),
                     (REVISION / "need_component_corr_spearman.csv",
                      "fig_A5_need_component_correlations_data.csv")):
        pd.read_csv(src).to_csv(HERE / dst, index=False)
        lg.add_input(src)
    print("  Figures A4, A5: plotted data copied from outputs/revision/")

    for f in sorted(HERE.glob("fig_*_data.csv")):
        lg.add_output(f)
    lg.finish()
    print("\nall figure data exported and checked against headline_numbers.json")


if __name__ == "__main__":
    main()
