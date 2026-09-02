"""Build the Wiley graphical abstract for Manuscript 1.

One landscape panel carrying the paper's single take-home finding: AI-based crop-yield
evidence accumulates where cropland is extensive and research is funded, not where the
measured need is highest.

Three sub-panels, left to right:
  A  where the evidence is      -- country choropleth of fractional study counts
  B  what predicts it           -- interquartile-range marginal effects on participation
  C  where it is missing        -- research percentile against need percentile

Every plotted value is read from a saved output; nothing is retyped. Panel styling
(palette, class breaks, projection) matches Figures 1, 4 and 6 so the graphical abstract
and the article read as one system.

Outputs: vector PDF, 600 dpi PNG and TIFF, the plotted data for each panel, a metadata
record, and the <=50-word summary text Wiley asks for alongside the image.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

GEO = ROOT / "outputs" / "geofix"
# The submission folder lives outside this replication repository. When it is absent
# (the normal case for anyone running this repo) the deliverables are written here only.
_SUB = ROOT / "Final Manuscript" / "Manuscript 1" / "09_Wiley_Submission"
SUB = _SUB if _SUB.is_dir() else None

# Palette and class breaks are copied from gf_03_figures.py so the graphical abstract
# uses the same colours as Figure 1. Okabe-Ito; colourblind-safe; no rainbow, no red-green.
OKB, OKV, OKG = "#0072B2", "#D55E00", "#009E73"
SEQ = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
COL_ZERO = "#F0F0F0"
BREAKS = [0, 1, 5, 20, 100, np.inf]
INK, MUTE = "#1A1A1A", "#5A5A5A"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.6, "axes.labelsize": 6.6,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "figure.dpi": 600, "savefig.facecolor": "white",
})

# Covariate labels for panel B. Order is set by the estimated effect, not hard-coded.
TERM_LABEL = {
    "log_area":       "Harvested area",
    "log_gdp_pc":     "GDP per capita",
    "internet":       "Internet use",
    "need":           "Research need",
    "rd":             "R&D expenditure",
    "log_population": "Population",
    "tertiary":       "Tertiary enrolment",
}

SUMMARY_50W = (
    "AI-based crop-yield studies cluster where cropland is extensive and research is "
    "funded. Across 2,616 country-crop systems, 483 carry any study; harvested area "
    "raises the chance of being studied by 35 percentage points, five times any other "
    "predictor. Fifty-four of 188 countries pair high need with low research."
)


def cls(v: float) -> int:
    """Figure 1 class assignment: 0 is a true zero, then five upper-bounded classes."""
    if v == 0:
        return 0
    for i, (a, b) in enumerate(zip(BREAKS[:-1], BREAKS[1:]), start=1):
        if a < v <= b:
            return i
    return len(BREAKS) - 1


def main() -> None:
    lg = RunLogger("ga_01_build")

    panel_fp = GEO / "country_crop_panel_corrected.parquet"
    ame_fp = GEO / "ame_geofix.csv"
    gap_fp = GEO / "gap_corrected_lisa.csv"
    head_fp = GEO / "headline_numbers.json"
    lay_fp = ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet"
    for fp in (panel_fp, ame_fp, gap_fp, head_fp, lay_fp):
        lg.add_input(fp)

    pan = pd.read_parquet(panel_fp)
    ame = pd.read_csv(ame_fp)
    gap = pd.read_csv(gap_fp)
    head = json.load(open(head_fp))[0]
    lay = gpd.read_parquet(lay_fp)

    lg.count("panel_cells_in", len(pan))
    lg.count("ame_rows_in", len(ame))
    lg.count("gap_countries_in", len(gap))

    # ---- assertions: the figure must agree with the saved headline numbers ----
    cty = pan.groupby("iso3", as_index=False).n_studies_fractional.sum()
    n_ctry_res = int((cty.n_studies_fractional > 0).sum())
    n_cells_res = int(pan.has_any_study.astype(bool).sum())
    assert n_ctry_res == head["countries_with_research"], (n_ctry_res, head["countries_with_research"])
    assert n_cells_res == head["cells_with_research"], (n_cells_res, head["cells_with_research"])
    assert len(gap) == head["n_gap"], (len(gap), head["n_gap"])
    quad = gap[(gap.need_pct >= 50) & (gap.research_pct <= 50)]
    assert len(quad) == head["quadrant"], (len(quad), head["quadrant"])
    lg.note(f"verified against headline_numbers.json: {n_ctry_res} countries, "
            f"{n_cells_res} cells, {len(gap)} indexed, {len(quad)} in quadrant")

    # 180 x 90 mm. Two panels and a statistic ribbon: a graphical abstract has to
    # carry one idea at thumbnail size, so the composition stays deliberately sparse.
    fig = plt.figure(figsize=(180 / 25.4, 90 / 25.4))

    # ---------------- headline ----------------
    fig.text(0.5, 0.945, "AI-based crop-yield research follows cropland and funding, not need",
             ha="center", va="center", fontsize=11.2, fontweight="bold", color=INK)
    fig.text(0.5, 0.876,
             "7,045 eligible studies, 2000\u20132026  \u00b7  study location read from the "
             "reported study area, not author affiliation",
             ha="center", va="center", fontsize=6.4, color=MUTE)

    # ---------------- Panel A: where the evidence is ----------------
    axA = fig.add_axes([0.005, 0.212, 0.535, 0.598])
    g = lay.merge(cty, on="iso3", how="left")
    g["n"] = g.n_studies_fractional.fillna(0.0)
    g["cls"] = g.n.map(cls)
    gp = g.to_crs("+proj=eqearth")
    cols = [COL_ZERO] + SEQ
    for i, c in enumerate(cols):
        s_ = gp[gp.cls == i]
        if len(s_):
            s_.plot(ax=axA, color=c, edgecolor="white", linewidth=0.18)
    # Clip Antarctica so the map fills the panel width instead of floating in it.
    ymin, ymax = axA.get_ylim()
    axA.set_ylim(ymin + 0.155 * (ymax - ymin), ymax)
    axA.set_axis_off()
    fig.text(0.015, 0.822, "Where the evidence is", fontsize=8.0, fontweight="bold",
             color=INK, ha="left", va="center")
    fig.text(0.500, 0.822, f"Gini {head['gini_panel199']:.3f}",
             fontsize=6.4, color=MUTE, ha="right", va="center")
    hs = [Patch(facecolor=cols[0], edgecolor="0.62", linewidth=0.3, label="no study"),
          Patch(facecolor=SEQ[0], edgecolor="none", label="\u22641"),
          Patch(facecolor=SEQ[1], edgecolor="none", label="\u22645"),
          Patch(facecolor=SEQ[2], edgecolor="none", label="\u226420"),
          Patch(facecolor=SEQ[3], edgecolor="none", label="\u2264100"),
          Patch(facecolor=SEQ[4], edgecolor="none", label=">100")]
    # Legend sits inside the empty South Pacific so the map can use the full panel.
    leg = axA.legend(handles=hs, loc="lower left", frameon=False, ncol=3, fontsize=6.0,
                     bbox_to_anchor=(0.005, 0.005), handlelength=0.9, handleheight=0.8,
                     columnspacing=0.5, handletextpad=0.3, labelspacing=0.35,
                     title="eligible studies per country\n(fractional count)",
                     title_fontsize=6.0, alignment="left")
    leg.get_title().set_color(MUTE)

    # ---------------- Panel B: what predicts it ----------------
    axB = fig.add_axes([0.685, 0.315, 0.290, 0.470])
    b = ame.copy()
    b["pp"] = b.iqr_dprob * 100
    b["label"] = b.term.map(TERM_LABEL)
    assert b.label.notna().all(), b.loc[b.label.isna(), "term"].tolist()
    b = b.sort_values("pp")
    bar_c = [OKV if t == "log_area" else "#B9C3CB" for t in b.term]
    axB.barh(b.label, b.pp, color=bar_c, height=0.66, edgecolor="none")
    for lab, v in zip(b.label, b.pp):
        axB.text(v + 0.9, lab, f"{v:.1f}", va="center", ha="left", fontsize=6.4,
                 color=OKV if v > 30 else MUTE,
                 fontweight="bold" if v > 30 else "normal")
    axB.set_xlim(0, b.pp.max() * 1.20)
    axB.set_xlabel("change in probability a system is studied (percentage points)",
                   labelpad=2, fontsize=6.2, color=MUTE)
    axB.spines[["top", "right"]].set_visible(False)
    axB.tick_params(axis="y", length=0, pad=1.5)
    axB.set_xticks([0, 10, 20, 30])
    fig.text(0.545, 0.822, "What predicts it", fontsize=8.0, fontweight="bold",
             color=INK, ha="left", va="center")
    fig.text(0.985, 0.822,
             "interquartile-range increase  \u00b7  1,799 systems",
             fontsize=6.4, color=MUTE, ha="right", va="center")

    # ---------------- statistic ribbon ----------------
    rib = fig.add_axes([0.015, 0.030, 0.970, 0.155]); rib.set_axis_off()
    rib.add_patch(plt.Rectangle((0, 0), 1, 1, transform=rib.transAxes,
                                facecolor="#F4F6F8", edgecolor="none", zorder=0))
    n_unstud_area = head["zero_cells_posarea"]
    cells_ = [
        (f"{head['countries_with_research']} of 199", "countries carry any study"),
        (f"{head['cells_with_research']} of 2,616", "country\u2013crop systems studied"),
        (f"{head['zero_area_share']:.1f}%", f"of cropland in the {n_unstud_area:,} unstudied systems"),
        (f"{head['quadrant']} of {head['n_gap']}", "countries: high need, low research"),
    ]
    for i, (big, small) in enumerate(cells_):
        x = (i + 0.5) / len(cells_)
        rib.text(x, 0.66, big, ha="center", va="center", fontsize=9.4,
                 fontweight="bold", color=OKB if i < 2 else OKV, transform=rib.transAxes)
        rib.text(x, 0.24, small, ha="center", va="center", fontsize=6.3,
                 color=INK, transform=rib.transAxes)
    for i in range(1, len(cells_)):
        rib.plot([i / len(cells_)] * 2, [0.14, 0.86], color="#D6DBE0", lw=0.6,
                 transform=rib.transAxes, clip_on=False)

    # ---------------- write ----------------
    stem = "graphical_abstract"
    pdf = HERE / f"{stem}.pdf"
    fig.savefig(pdf, format="pdf")                       # vector master
    png = HERE / f"{stem}_600dpi.png"
    fig.savefig(png, dpi=600)
    plt.close(fig)

    # TIFF (LZW) for journals that require a bitmap deposit.
    from PIL import Image
    tif = HERE / f"{stem}_600dpi.tif"
    im = Image.open(png).convert("RGB")
    im.save(tif, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
    w, h = im.size

    # ---- plotted data, one file per panel (CLAUDE.md 6.5) ----
    dA = g[["iso3", "n", "cls"]].rename(columns={"n": "n_studies_fractional",
                                                 "cls": "figure1_class"})
    dA.to_csv(HERE / f"{stem}_panelA_data.csv", index=False)
    dB = b[["term", "label", "ame", "se", "p_value", "p25", "p75", "iqr_dprob", "pp"]]
    dB.to_csv(HERE / f"{stem}_panelB_data.csv", index=False)
    dC = gap.assign(high_need_low_research=(gap.need_pct >= 50) & (gap.research_pct <= 50))
    dC.to_csv(HERE / f"{stem}_ribbon_gap_data.csv", index=False)

    (HERE / f"{stem}_summary_text.md").write_text(
        "# Graphical abstract summary text (Wiley: 50 words or fewer)\n\n"
        f"{SUMMARY_50W}\n\n"
        f"Word count: {len(SUMMARY_50W.split())}\n"
    )

    meta = {
        "figure": "Graphical abstract, Manuscript 1",
        "built_by": "outputs/graphical_abstract/ga_01_build.py",
        "size_mm": [180, 90],
        "raster_px": [w, h],
        "raster_dpi": 600,
        "files": {"vector": pdf.name, "png": png.name, "tiff": tif.name},
        "colour_mode": "RGB",
        "palette": "Okabe-Ito + sequential blues; identical to Figure 1",
        "summary_text_words": len(SUMMARY_50W.split()),
        "summary_text": SUMMARY_50W,
        "caption": (
            "Graphical abstract. Left, fractional count of eligible AI-based crop-yield "
            "studies by country, 2000-2026, equal-earth projection, Antarctica omitted; "
            "grey marks a true zero rather than missing data. Right, average marginal "
            "effects on the probability that a country-crop system carries research, in "
            "percentage points, for an interquartile-range increase in each covariate, from "
            "the participation logit on 1,799 systems across 142 countries. The ribbon "
            "reports the four headline counts: countries carrying any study; country-crop "
            "systems carrying any study; the share of the panel's harvested area lying in "
            "systems with no study; and countries at or above median research need and at "
            "or below median research, of the 188 with a computable nine-component need index."
        ),
        "values_shown": {
            "countries_with_research": head["countries_with_research"],
            "cells_with_research": head["cells_with_research"],
            "gini_panel199": round(head["gini_panel199"], 3),
            "n_gap": head["n_gap"],
            "quadrant": head["quadrant"],
            "moran_k6": round(head["moran_k6"], 3),
            "wald_p": round(head["wald_p"], 3),
            "iqr_dprob_log_area_pp": round(float(b.loc[b.term == "log_area", "pp"].iloc[0]), 1),
        },
        "source_outputs": [str(fp.relative_to(ROOT)) for fp in
                           (panel_fp, ame_fp, gap_fp, head_fp, lay_fp)],
    }
    (HERE / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2))

    # mirror the deliverables into the submission folder when it is present
    if SUB is not None:
        for f in (pdf, png, tif, HERE / f"{stem}_summary_text.md"):
            (SUB / f.name).write_bytes(f.read_bytes())

    for f in (pdf, png, tif):
        lg.add_output(f)
    lg.add_output(HERE / f"{stem}_panelA_data.csv", rows=len(dA))
    lg.add_output(HERE / f"{stem}_panelB_data.csv", rows=len(dB))
    lg.add_output(HERE / f"{stem}_ribbon_gap_data.csv", rows=len(dC))
    lg.add_output(HERE / f"{stem}_metadata.json")
    lg.count("raster_px_w", w); lg.count("raster_px_h", h)
    lg.count("summary_text_words", len(SUMMARY_50W.split()))
    lg.finish()
    print(f"  graphical abstract: {w} x {h} px at 600 dpi (180x90 mm)")
    print(f"  summary text: {len(SUMMARY_50W.split())} words")


if __name__ == "__main__":
    main()
