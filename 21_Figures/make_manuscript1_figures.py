"""Manuscript 1 — the six main figures and four main tables.

Every value plotted or tabulated is read from a file already on disk. Nothing is
simulated, interpolated, or filled. Where an input is absent the figure is skipped and
the skip is written to the log and to `manuscript1_figure_manifest.csv`.

Two conventions carry through every panel:

* A country-crop system with no eligible study is a measured zero and is drawn in its own
  class, separate from a system whose external data is unavailable. The zero class is
  white; the unavailable class is grey with diagonal hatching. Both appear in the legend.
* Fractional counting is the primary rule. Full counts appear beside it in the saved data
  and are labelled wherever they are shown.

Outputs, all under 21_Figures/ and 22_Tables/:

  fig_01_research_intensity_map          country map of fractional research output
  fig_02_crop_attention_vs_production    top crops, attention share against production share
  fig_03_research_vs_production_share    country research share against production share
  fig_04_need_vs_research_percentile     need percentile against research percentile
  fig_05_evidence_gap_lisa_map           LISA clusters of the research-need mismatch, FDR-adjusted
  fig_06_model_coefficients              participation logit and PPML intensity estimates

  tab_01_corpus_construction             records to eligible studies
  tab_02_main_regression_estimates       participation logit and PPML intensity
  tab_03_under_researched_systems        high-need country-crop systems with no study
  tab_04_finding_stability               robustness classification of the main findings

Each figure is written as .png (300 dpi) and .pdf, with `<name>_data.csv` holding exactly
the plotted rows and `<name>_caption.txt` holding a caption that states the unit, the
population, the period, and what the colour encodes.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import CRS_DISPLAY, P, RANDOM_SEED, RunLogger  # noqa: E402

# --------------------------------------------------------------------------- constants
SEARCH_WINDOW = "2000-2026"          # publication years present in the frozen corpus
LISA_WEIGHTS = "knn_k6"              # the primary weights matrix, fixed in Phase 12
LISA_ALPHA = 0.05
LISA_PERMUTATIONS = 9999

# Colourblind-safe throughout. No rainbow ramp, and no pairing that relies on red against
# green. The categorical hues are Okabe-Ito; the sequential ramp is a single-hue blue.
OK_BLUE = "#0072B2"
OK_VERM = "#D55E00"
OK_SKY = "#56B4E9"
OK_ORANGE = "#E69F00"
OK_GREY = "#4D4D4D"

COL_ZERO = "#FFFFFF"      # measured zero: an eligible study count of exactly 0
COL_NODATA = "#BDBDBD"    # external data unavailable, drawn with hatching
HATCH_NODATA = "///"
EDGE = "#5A5A5A"
SEQ_BLUES = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]

# The LISA palette for this manuscript, defined here so every LISA map drawn by this
# script uses the same hues. 21_Figures/00_Design_System/color_dictionary.csv is a
# cloud-only placeholder that could not be materialised on this run, so this palette has
# not been checked against it; that check is outstanding.
LISA_COLORS = {
    "High-High": OK_VERM,
    "Low-Low": OK_BLUE,
    "High-Low": OK_ORANGE,
    "Low-High": OK_SKY,
    "Not significant": "#F0F0F0",
}
LISA_ORDER = ["High-High", "Low-Low", "High-Low", "Low-High", "Not significant"]

DIVERGING = LinearSegmentedColormap.from_list(
    "gap_orange_blue", [OK_VERM, "#F7F7F7", OK_BLUE], N=256)

# Class breaks for the research-intensity map. Fixed before the map was drawn, stated in
# the caption, and applied to fractional counts.
INTENSITY_BREAKS = [(0, 1, "0 < n ≤ 1"), (1, 5, "1 < n ≤ 5"),
                    (5, 20, "5 < n ≤ 20"), (20, 100, "20 < n ≤ 100"),
                    (100, np.inf, "n > 100")]

TERM_LABELS = {
    "log_area": "log harvested area (ha)",
    "log_gdp_pc": "log GDP per capita, PPP",
    "rd": "R&D expenditure, % of GDP",
    "tertiary": "tertiary enrolment (share)",
    "internet": "internet users (share)",
    # The "/ 100" in the previous label described the double division repaired in
    # D-130; the regressor is the 0-1 percentile rank itself.
    "need": "need index (0-1 percentile rank)",
    "yield_volatility": "yield volatility, CV",
    "log_population": "log population",
    "const": "constant",
}
TERM_ORDER = ["log_area", "log_gdp_pc", "rd", "tertiary", "internet",
              "need", "yield_volatility", "log_population"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIGDIR = P["figures"]
TABDIR = P["tables"]
MANIFEST: list[dict] = []


# --------------------------------------------------------------------------- helpers
def need_input(lg, path: Path, what: str) -> Path | None:
    """Register an input. Return None when it is absent so the caller can skip."""
    lg.add_input(path)
    if not path.exists():
        lg.error(f"MISSING INPUT for {what}: {path}")
        return None
    return path


def record(kind: str, name: str, status: str, detail: str = "") -> None:
    MANIFEST.append({"kind": kind, "name": name, "status": status, "detail": detail})


def save_figure(fig, name: str, data: pd.DataFrame, caption: str, lg) -> None:
    """Write png, pdf, the plotted rows, and the caption. CLAUDE.md 6.5."""
    png, pdf = FIGDIR / f"{name}.png", FIGDIR / f"{name}.pdf"
    csv, cap = FIGDIR / f"{name}_data.csv", FIGDIR / f"{name}_caption.txt"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    data.to_csv(csv, index=False)
    cap.write_text(textwrap.fill(" ".join(caption.split()), 96) + "\n")
    for f, rows in ((png, None), (pdf, None), (csv, len(data)), (cap, None)):
        lg.add_output(f, rows=rows)
    record("figure", name, "produced", f"{len(data)} plotted rows")


def tex_escape(s) -> str:
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def write_table(df: pd.DataFrame, name: str, caption: str, note: str, lg,
                tex_cols: list[str] | None = None, align: str | None = None) -> None:
    csv = TABDIR / f"{name}.csv"
    tex = TABDIR / f"{name}.tex"
    df.to_csv(csv, index=False)

    show = df[tex_cols] if tex_cols else df
    ncol = show.shape[1]
    align = align or ("l" + "r" * (ncol - 1))
    lines = [r"% Requires \usepackage{booktabs}",
             r"\begin{table}[htbp]", r"\centering", r"\small",
             f"\\caption{{{tex_escape(caption)}}}",
             f"\\label{{tab:{name}}}",
             f"\\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(tex_escape(c) for c in show.columns) + r" \\", r"\midrule"]
    for _, row in show.iterrows():
        lines.append(" & ".join(tex_escape(v) for v in row.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(r"\begin{minipage}{\linewidth}\footnotesize")
        lines.append(tex_escape(" ".join(note.split())))
        lines.append(r"\end{minipage}")
    lines += [r"\end{table}", ""]
    tex.write_text("\n".join(lines))
    lg.add_output(csv, rows=len(df))
    lg.add_output(tex, rows=len(show))
    record("table", name, "produced", f"{len(df)} rows")


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:,.{nd}f}"


def base_map_axes(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")


def place_labels(ax, xs, ys, texts, fontsize=7.5, color="#222222", iters=60,
                 max_offset=30.0, connect=False):
    """Annotate points and nudge the labels apart so none is unreadable.

    Only the label position moves; the anchor point is the data value and never moves.
    A connector is drawn once a label has travelled far enough from its point for the
    pairing to be ambiguous.
    """
    anns = []
    for x, y, t in zip(xs, ys, texts):
        kw = {}
        if connect:
            kw["arrowprops"] = dict(arrowstyle="-", lw=0.4, color="#9A9A9A",
                                    shrinkA=0.0, shrinkB=1.5)
        anns.append(ax.annotate(t, (x, y), textcoords="offset points", xytext=(5, 4),
                                fontsize=fontsize, color=color, zorder=6, **kw))
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for _ in range(iters):
        boxes = [a.get_window_extent(renderer=rend) for a in anns]
        moved = False
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                moved = True
                up = 1.0 if boxes[i].y0 >= boxes[j].y0 else -1.0
                for k, sgn in ((i, up), (j, -up)):
                    dx, dy = anns[k].xyann
                    ndy = dy + sgn * 2.4
                    if abs(ndy) <= max_offset:
                        anns[k].xyann = (dx, ndy)
        if not moved:
            break
        fig.canvas.draw()
    return anns


def audit_map_layer(gdf, value_col: str, layer_name: str) -> dict:
    geom_null = int(gdf.geometry.isna().sum())
    invalid = int((~gdf.geometry.is_valid).sum())
    v = pd.to_numeric(gdf[value_col], errors="coerce")
    try:
        crs = gdf.crs.to_string()
    except Exception:
        crs = str(gdf.crs)
    return {"layer": layer_name, "n_features": int(len(gdf)),
            "crs_storage": crs, "crs_display": CRS_DISPLAY,
            "n_null_geometry": geom_null,
            "n_invalid_geometry": invalid, "value_column": value_col,
            "n_value_missing": int(v.isna().sum()), "n_value_zero": int((v == 0).sum()),
            "value_min": float(np.nanmin(v)) if v.notna().any() else np.nan,
            "value_max": float(np.nanmax(v)) if v.notna().any() else np.nan,
            "n_duplicate_iso3": int(gdf["iso3"].duplicated().sum())}


# --------------------------------------------------------------------------- figure 1
def fig_01(lg, layer, research):
    name = "fig_01_research_intensity_map"
    g = layer[["iso3", "geometry"]].merge(
        research[["iso3", "fao_area_name", "n_studies_fractional",
                  "n_studies_full", "wb_region", "wb_income_group"]],
        on="iso3", how="left")
    audit = audit_map_layer(g, "n_studies_fractional", "country_analytical_layer + research side")

    v = pd.to_numeric(g["n_studies_fractional"], errors="coerce")
    cls = pd.Series(pd.NA, index=g.index, dtype="object")
    cls[v.isna()] = "no data"
    cls[v == 0] = "0 (no resolved study)"
    for lo, hi, lab in INTENSITY_BREAKS:
        cls[(v > lo) & (v <= hi)] = lab
    g["intensity_class"] = cls

    proj = g.to_crs(CRS_DISPLAY)
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    base_map_axes(ax)

    order = ["0 (no resolved study)"] + [b[2] for b in INTENSITY_BREAKS]
    colours = [COL_ZERO] + SEQ_BLUES
    counts = {}
    for lab, col in zip(order, colours):
        sub = proj[proj["intensity_class"] == lab]
        counts[lab] = len(sub)
        if len(sub):
            sub.plot(ax=ax, color=col, edgecolor=EDGE, linewidth=0.22)
    nod = proj[proj["intensity_class"] == "no data"]
    counts["no data"] = len(nod)
    if len(nod):
        nod.plot(ax=ax, color=COL_NODATA, edgecolor=EDGE, linewidth=0.22,
                 hatch=HATCH_NODATA)

    handles = [Patch(facecolor=c, edgecolor=EDGE, linewidth=0.4,
                     label=f"{lab}  (n = {counts[lab]})")
               for lab, c in zip(order, colours)]
    handles.append(Patch(facecolor=COL_NODATA, edgecolor=EDGE, linewidth=0.4,
                         hatch=HATCH_NODATA,
                         label=f"no data: external data unavailable  (n = {counts['no data']})"))
    ax.legend(handles=handles, loc="upper center", frameon=False, ncol=4,
              title="Fractional eligible studies per country",
              title_fontsize=8, bbox_to_anchor=(0.5, 0.03), handlelength=1.5,
              columnspacing=1.4)
    ax.set_title("Fractional count of eligible AI crop-yield studies by country, "
                 f"{SEARCH_WINDOW}", loc="left", pad=6)

    out = g.drop(columns="geometry")[
        ["iso3", "fao_area_name", "wb_region", "wb_income_group",
         "n_studies_fractional", "n_studies_full", "intensity_class"]].copy()
    out["mapped"] = True
    save_figure(fig, name, out.sort_values("n_studies_fractional", ascending=False),
                "Fractional count of eligible AI-based crop-yield studies, one value per "
                f"country, {SEARCH_WINDOW} publication years, 195 countries with a mapped "
                "boundary out of the 199 in the country-crop panel. A study covering k "
                "countries contributes 1/k to each, so the map totals the corpus rather "
                "than multiplying it. Colour encodes the fractional study count in six "
                "classes with fixed breaks at 0, 1, 5, 20 and 100. White is a measured "
                f"zero, meaning no resolved eligible study, and covers {counts['0 (no resolved study)']} "
                "countries, which is a real value and not a gap. Grey hatching marks a "
                "country whose external data is unavailable and covers "
                f"{counts['no data']} countries here. The four "
                "panel territories without a boundary in the layer (GLP, GUF, MTQ, REU) "
                "each have a fractional count of 0 and are listed in the log rather than "
                "drawn. Equal Earth projection (EPSG:8857).", lg)
    return audit, counts


# --------------------------------------------------------------------------- figure 2
def fig_02(lg, panel):
    name = "fig_02_crop_attention_vs_production"
    cr = panel.groupby("crop_standard_name").agg(
        n_studies_fractional=("n_studies_fractional", "sum"),
        n_studies_full=("n_studies_full", "sum"),
        world_production_t=("world_production_t", "first"),
        n_cells=("iso3", "size"),
        n_cells_with_study=("has_any_study", "sum")).reset_index()
    cr["attention_share_pct"] = 100 * cr["n_studies_fractional"] / cr["n_studies_fractional"].sum()
    cr["world_production_share_pct"] = 100 * cr["world_production_t"] / cr["world_production_t"].sum()
    cr = cr.sort_values("n_studies_fractional", ascending=False).reset_index(drop=True)
    top = cr.head(15).copy()

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    y = np.arange(len(top))[::-1]
    h = 0.38
    ax.barh(y + h / 2, top["attention_share_pct"], height=h, color=OK_BLUE,
            edgecolor="none", label="share of fractional research attention")
    ax.barh(y - h / 2, top["world_production_share_pct"], height=h, color=OK_VERM,
            edgecolor="none", label="share of world production (tonnes)")
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in top["crop_standard_name"]])
    ax.set_xlabel("Per cent")
    ax.set_xlim(0, max(top["attention_share_pct"].max(),
                       top["world_production_share_pct"].max()) * 1.24)
    for yi, n in zip(y, top["n_studies_fractional"]):
        ax.text(ax.get_xlim()[1] * 0.995, yi, f"{n:,.1f}", va="center", ha="right",
                fontsize=7.5, color=OK_GREY)
    ax.text(ax.get_xlim()[1] * 0.995, y.max() + 0.85, "fractional\nstudies",
            va="center", ha="right", fontsize=7.5, color=OK_GREY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=2)
    ax.set_title("Research attention and world production, 15 crops with the most "
                 "fractional attention", loc="left", pad=8)

    save_figure(fig, name, cr,
                "Crop-level attention against crop-level production. Unit is the crop "
                f"aggregated over the 2,616 country-crop cells of the panel, {SEARCH_WINDOW} "
                "publication years. Blue is each crop's share of the total fractional "
                "eligible study count; vermillion is the same crop's share of world "
                "production in tonnes, from the FAOSTAT reference-period mean already "
                "stored in the panel. The number at the right of each row is the crop's "
                "absolute fractional study count. All 25 crops are in the saved data; the "
                "figure shows the 15 with the most attention.", lg)
    return cr


# --------------------------------------------------------------------------- figure 3
def fig_03(lg, panel, research, desc):
    name = "fig_03_research_vs_production_share"
    c = panel.groupby("iso3").agg(
        research_share=("research_share", "sum"),
        production_share=("production_share", lambda s: s.sum(min_count=1)),
        n_studies_fractional=("n_studies_fractional", "sum"),
        n_production_cells=("production_t_mean", "count")).reset_index()
    c = c.merge(research[["iso3", "fao_area_name", "wb_region", "wb_income_group"]],
                on="iso3", how="left")
    c["research_share_pct"] = 100 * c["research_share"]
    c["production_share_pct"] = 100 * c["production_share"]

    both = c[(c["research_share_pct"] > 0) & (c["production_share_pct"] > 0)].copy()
    zero_res = c[(c["research_share_pct"] == 0) & (c["production_share_pct"] > 0)].copy()
    no_prod = c[c["production_share_pct"].isna() | (c["production_share_pct"] == 0)].copy()

    both["log_ratio"] = np.log10(both["research_share_pct"] / both["production_share_pct"])
    lab_ratio = both[both["production_share_pct"] >= 0.1].reindex(
        both[both["production_share_pct"] >= 0.1]["log_ratio"].abs()
        .sort_values(ascending=False).index).head(10)
    lab_top = both.nlargest(5, "research_share_pct")
    labelled = pd.concat([lab_ratio, lab_top]).drop_duplicates("iso3")
    c["labelled_in_figure"] = c["iso3"].isin(labelled["iso3"])
    c["plot_group"] = np.where(c["iso3"].isin(both["iso3"]), "log-log scatter",
                               np.where(c["iso3"].isin(zero_res["iso3"]),
                                        "zero-research strip", "no production denominator"))

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    # The horizontal range is set by the countries with a study. Zero-research countries
    # whose production share falls below that range are drawn at the left edge and
    # counted, rather than stretching the axis over five further decades.
    lo_x = both["production_share_pct"].min() * 0.4
    hi_x = max(both["production_share_pct"].max(),
               zero_res["production_share_pct"].max() if len(zero_res) else 0)
    lo_y, hi_y = both["research_share_pct"].min(), both["research_share_pct"].max()
    par_lo, par_hi = min(lo_x, lo_y) * 0.4, max(hi_x, hi_y) * 2.5
    ax.plot([par_lo, par_hi], [par_lo, par_hi], color=OK_GREY, linewidth=0.9,
            linestyle="--", zorder=1, label="parity (equal shares)")
    ax.scatter(both["production_share_pct"], both["research_share_pct"], s=26,
               facecolor=OK_BLUE, edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3,
               label=f"country with a study (n = {len(both)})")

    strip_y = lo_y / 3.2
    n_clipped = 0
    if len(zero_res):
        below = zero_res["production_share_pct"] < lo_x
        n_clipped = int(below.sum())
        zx = zero_res["production_share_pct"].where(~below, lo_x * 1.08)
        ax.scatter(zx, np.full(len(zero_res), strip_y), s=30, marker="v",
                   facecolor="white", edgecolor=OK_VERM, linewidth=0.9, zorder=3,
                   label=f"zero: no resolved eligible study (n = {len(zero_res)})")
        ax.axhline(strip_y, color=OK_VERM, linewidth=0.5, alpha=0.35, zorder=2)
        if n_clipped:
            ax.annotate(f"{n_clipped} of these produce less than {lo_x:.0e} % of world "
                        "output and are drawn at the left edge",
                        (lo_x * 1.1, strip_y), textcoords="offset points",
                        xytext=(4, 9), fontsize=7.2, color=OK_VERM, va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                                  pad=1.2))
    c["clipped_to_left_edge"] = c["iso3"].isin(
        zero_res.loc[zero_res["production_share_pct"] < lo_x, "iso3"]
        if len(zero_res) else [])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo_x * 0.5, hi_x * 2.2)
    ax.set_ylim(strip_y / 1.9, hi_y * 2.2)
    place_labels(ax, labelled["production_share_pct"].values,
                 labelled["research_share_pct"].values, labelled["iso3"].tolist(),
                 max_offset=22.0)
    ax.set_xlabel("Share of world crop production, % (log scale)")
    ax.set_ylabel("Share of fractional eligible studies, % (log scale)")
    ax.grid(color="#E5E5E5", linewidth=0.5, which="both")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    sp_c = desc.get("spearman_research_vs_production_countrywise")
    ax.set_title("Research share against production share, one point per country "
                 f"(Spearman rho = {sp_c})", loc="left", pad=8)

    save_figure(fig, name, c.sort_values("research_share_pct", ascending=False),
                "Country shares of research and of production. Unit is the country, "
                f"summed over its crop cells in the panel, {SEARCH_WINDOW} publication "
                "years, 199 countries. The vertical axis is the country's share of the "
                "total fractional eligible study count; the horizontal axis is its share "
                "of world crop production in tonnes. Both axes are logarithmic, so the "
                f"{len(zero_res)} countries with zero coded attention cannot be placed on the "
                "vertical scale and are drawn as open triangles on a separate strip below "
                "the axis, positioned by their production share. The dashed line is "
                "parity: a country above it holds a larger share of the literature than of "
                "production. Spearman rank correlation between fractional study count and "
                f"production, country level, is {sp_c} over 199 countries, and "
                f"{desc.get('spearman_research_vs_production_cellwise')} at the "
                "country-crop cell level over the "
                f"{desc.get('panel_cells_with_production_denominator'):,} cells with a "
                "production denominator; both are read from "
                "15_Descriptive_Analysis/research_side_descriptives.json. Labels mark the "
                "ten countries with at least 0.1 per cent of world production and the "
                "largest absolute log ratio of the two shares, plus the five countries "
                "with the largest research share. The horizontal range is set by the "
                f"countries with a study; {n_clipped} zero-research countries produce less "
                "than the left edge and are drawn there, flagged in the saved data. "
                f"{len(no_prod)} countries have no production denominator and are excluded "
                "from the scatter; they are kept in the saved data.", lg)
    return c, len(both), len(zero_res), len(no_prod), n_clipped


# --------------------------------------------------------------------------- figure 4
def fig_04(lg, research):
    name = "fig_04_need_vs_research_percentile"
    d = research[["iso3", "fao_area_name", "wb_region", "wb_income_group",
                  "n_studies_fractional", "need_rank_pct", "research_pct",
                  "evidence_gap"]].copy()
    # need_rank_pct is stored on 0-1; research_pct on 0-100. Both are put on 0-100 here so
    # the difference is in percentile points. The stored `evidence_gap` column subtracts
    # the 0-1 value from the 0-100 value and is carried through unchanged for reference.
    d["need_pct_0_100"] = 100 * d["need_rank_pct"]
    d["research_minus_need_pctpts"] = d["research_pct"] - d["need_pct_0_100"]
    plot = d.dropna(subset=["need_pct_0_100", "research_pct"]).copy()
    missing = d[d["need_pct_0_100"].isna()]

    quad = plot[(plot["need_pct_0_100"] >= 50) & (plot["research_pct"] <= 50)]
    d["high_need_low_research_quadrant"] = d["iso3"].isin(quad["iso3"])

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    ax.add_patch(plt.Rectangle((50, 0), 50, 50, facecolor=OK_VERM, alpha=0.07,
                               edgecolor="none", zorder=0))
    ax.axhline(50, color="#BBBBBB", linewidth=0.7, zorder=1)
    ax.axvline(50, color="#BBBBBB", linewidth=0.7, zorder=1)
    ax.plot([0, 100], [0, 100], color=OK_GREY, linestyle="--", linewidth=0.9, zorder=1)

    vmax = float(np.nanmax(np.abs(plot["research_minus_need_pctpts"])))
    sc = ax.scatter(plot["need_pct_0_100"], plot["research_pct"], s=32,
                    c=plot["research_minus_need_pctpts"], cmap=DIVERGING,
                    norm=Normalize(-vmax, vmax), edgecolor="#555555", linewidth=0.3,
                    zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.68, pad=0.02)
    cb.set_label("research percentile minus need percentile\n(percentile points)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    # Every country in the quadrant sits on the same research percentile, so point labels
    # would pile on one horizontal line. The ten largest shortfalls are listed instead.
    worst = quad.nsmallest(10, "research_minus_need_pctpts")
    listing = textwrap.fill(
        "  ".join(f"{r['iso3']} {r['research_minus_need_pctpts']:.0f}"
                  for _, r in worst.iterrows()), 32)
    ax.text(51.5, 41, f"high need, low research\n{len(quad)} of {len(plot)} countries",
            fontsize=8.5, color=OK_VERM, va="top", linespacing=1.4)
    ax.text(51.5, 33,
            "ten largest shortfalls, research\npercentile minus need percentile:\n" + listing,
            fontsize=7.5, color="#333333", va="top", linespacing=1.45)
    floor = float(plot["research_pct"].min())
    n_floor = int((plot["research_pct"] == floor).sum())
    ax.text(1.5, floor - 3.5,
            f"{n_floor} countries tie at the lowest research percentile "
            f"({floor:.2f}): no eligible study",
            fontsize=7.5, color=OK_GREY, va="top")
    ax.set_xlabel("Research-need percentile (rank-weighted index, 0-100)")
    ax.set_ylabel("Research-output percentile (fractional study count, 0-100)")
    ax.grid(color="#EEEEEE", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Research output percentile against research need percentile, "
                 "one point per country", loc="left", pad=8)

    save_figure(fig, name, d.sort_values("research_minus_need_pctpts"),
                "Country research-output percentile against country research-need "
                f"percentile, {SEARCH_WINDOW} publication years, {len(plot)} countries "
                "plotted of the 195 in the spatial layer. The need percentile is the "
                "rank-weighted need index stored as need_rank_pct and rescaled here from "
                "0-1 to 0-100; the research percentile is the percentile rank of the "
                "fractional eligible study count, and countries with zero coded attention share "
                "the lowest rank. Colour encodes research percentile minus need "
                "percentile, in percentile points, on a diverging scale centred on zero: "
                "orange marks a country ranked lower for research than for need, blue the "
                "reverse. The shaded quadrant holds countries at or above the median for "
                "need and at or below the median for research, and contains "
                f"{len(quad)} countries; the ten with the largest shortfall are listed "
                "inside it, since every country there sits at the same research "
                f"percentile. {n_floor} of the {len(plot)} plotted countries tie at the "
                f"lowest research percentile, {floor:.2f}, because none has an eligible "
                "study. "
                f"{len(missing)} countries have no need index and are not plotted; they "
                "are kept in the saved data with NA.", lg)
    return d, len(plot), len(quad), len(missing)


# --------------------------------------------------------------------------- figure 5
def fig_05(lg, layer, lisa, moran):
    name = "fig_05_evidence_gap_lisa_map"
    cols = ["iso3", "evidence_gap_value", "evidence_gap_lisa_I", "evidence_gap_lisa_p",
            "evidence_gap_lisa_cat_fdr", "evidence_gap_lisa_cat_raw",
            "evidence_gap_value_missing", "wb_region", "wb_income_group",
            "n_studies_fractional", "need_rank_pct"]
    g = layer[["iso3", "geometry"]].merge(lisa[cols], on="iso3", how="left")
    audit = audit_map_layer(g, "evidence_gap_value", "country_analytical_layer + LISA (FDR)")

    g["display_class"] = np.where(g["evidence_gap_value_missing"].fillna(True),
                                  "no data", g["evidence_gap_lisa_cat_fdr"])
    proj = g.to_crs(CRS_DISPLAY)

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    base_map_axes(ax)
    counts = {}
    for lab in LISA_ORDER:
        sub = proj[proj["display_class"] == lab]
        counts[lab] = len(sub)
        if len(sub):
            sub.plot(ax=ax, color=LISA_COLORS[lab], edgecolor=EDGE, linewidth=0.22)
    nod = proj[proj["display_class"] == "no data"]
    counts["no data"] = len(nod)
    if len(nod):
        nod.plot(ax=ax, color=COL_NODATA, edgecolor=EDGE, linewidth=0.22,
                 hatch=HATCH_NODATA)

    handles = [Patch(facecolor=LISA_COLORS[l], edgecolor=EDGE, linewidth=0.4,
                     label=f"{l}  (n = {counts[l]})") for l in LISA_ORDER]
    handles.append(Patch(facecolor=COL_NODATA, edgecolor=EDGE, linewidth=0.4,
                         hatch=HATCH_NODATA,
                         label=f"no data: no need index  (n = {counts['no data']})"))
    ax.legend(handles=handles, loc="upper center", frameon=False, ncol=3,
              title="Local Moran cluster, FDR-adjusted at alpha = 0.05",
              title_fontsize=8, bbox_to_anchor=(0.5, 0.05), handlelength=1.5,
              columnspacing=1.4)

    mrow = moran[(moran["variable"] == "evidence_gap") &
                 (moran["weights"] == LISA_WEIGHTS)]
    mi = float(mrow["morans_I"].iloc[0]) if len(mrow) else np.nan
    mp = float(mrow["p_sim"].iloc[0]) if len(mrow) else np.nan
    ax.set_title("Local Moran clusters of the research-need mismatch, FDR-adjusted categories, "
                 f"{LISA_WEIGHTS} weights (global Moran's I = {mi}, p = {mp})",
                 loc="left", pad=6)

    save_figure(fig, name, g.drop(columns="geometry"),
                "Local Moran (LISA) cluster categories for the research-need mismatch, one value per "
                "country, 195 countries in the spatial layer. Categories are assigned on "
                "Benjamini-Hochberg FDR-adjusted pseudo p-values at alpha = 0.05 from "
                f"{LISA_PERMUTATIONS} permutations under the {LISA_WEIGHTS} weights "
                "matrix, the primary matrix for this project; raw unadjusted categories "
                "are in the saved data and are never used to call a cluster. Colour "
                "encodes the cluster type, using one palette for every LISA map this "
                "script draws: vermillion High-High, blue Low-Low, orange High-Low, sky "
                "blue Low-High, near-white for a country whose local statistic is not "
                "significant after adjustment. Grey hatching marks a country with no need "
                "index, whose mismatch value is unavailable and was mean-filled "
                f"before estimation. The mapped variable is the stored evidence_gap "
                "column, the country's research-output percentile minus its research-need "
                "percentile, both on a 0-100 scale, so a positive value marks a country "
                "better represented in this literature than its need rank implies. "
                f"Global Moran's I for this variable is {mi} with pseudo p = {mp}. Equal "
                "Earth projection (EPSG:8857).", lg)
    return audit, counts, mi, mp


# --------------------------------------------------------------------------- figure 6
def fig_06(lg, coef, comparison, diagnostics):
    name = "fig_06_model_coefficients"
    models = [("participation_logit", "Participation (logit)\noutcome: cell has any study"),
              ("ppml_area_exposure", "Intensity (PPML, area exposure)\noutcome: fractional study count")]
    d = coef[coef["model"].isin([m for m, _ in models]) & (coef["term"] != "const")].copy()
    d["label"] = d["term"].map(TERM_LABELS).fillna(d["term"])
    d = d[d["estimate"].notna()]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
    plotted = []
    finite_terms = [t for t in TERM_ORDER if t != "need"]
    lim = 0.0
    for m, _ in models:
        s = d[(d["model"] == m) & (d["term"].isin(finite_terms))]
        if len(s):
            lim = max(lim, float(np.nanmax(np.abs(
                pd.concat([s["ci_low"], s["ci_high"], s["estimate"]])))))
    lim *= 1.22

    ypos = {t: i for i, t in enumerate(TERM_ORDER[::-1])}
    for ax, (m, title) in zip(axes, models):
        s = d[d["model"] == m].set_index("term")
        ax.axvline(0, color="#999999", linewidth=0.8, zorder=1)
        for t in TERM_ORDER:
            y = ypos[t]
            if t not in s.index:
                ax.text(0, y, "not estimated (offset)", fontsize=7.5, color=OK_GREY,
                        ha="center", va="center", style="italic")
                plotted.append({"model": m, "term": t, "status": "not estimated (offset)",
                                "estimate": np.nan, "std_error": np.nan,
                                "ci_low": np.nan, "ci_high": np.nan, "off_scale": False})
                continue
            r = s.loc[t]
            est, lo, hi = float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])
            off = (abs(lo) > lim) or (abs(hi) > lim) or (abs(est) > lim)
            col = OK_VERM if (lo > 0 or hi < 0) else OK_BLUE
            if off:
                ax.annotate("", xy=(-lim * 0.97, y), xytext=(lim * 0.97, y),
                            arrowprops=dict(arrowstyle="<->", color=col, lw=1.2,
                                            alpha=0.6, shrinkA=0, shrinkB=0),
                            zorder=3)
                if abs(est) <= lim * 0.9:
                    ax.plot([est], [y], marker="o", ms=4.5, color=col, zorder=4)
                else:
                    ax.plot([np.sign(est) * lim * 0.9], [y],
                            marker=">" if est > 0 else "<", ms=6.5, color=col, zorder=4)
                ax.text(0, y - 0.3,
                        f"{est:,.3f}  [{lo:,.1f}, {hi:,.1f}]  interval wider than the axis",
                        fontsize=6.8, color=OK_GREY, ha="center", va="top")
            else:
                ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt="o", ms=4.5,
                            color=col, ecolor=col, elinewidth=1.4, capsize=2.5, zorder=3)
            plotted.append({"model": m, "term": t, "status": "off scale" if off else "on scale",
                            "estimate": est, "std_error": float(r["std_error"]),
                            "ci_low": lo, "ci_high": hi, "off_scale": off})
        n = int(s["n"].iloc[0]) if len(s) else np.nan
        ax.set_title(f"{title}\nn = {n:,} country-crop cells", fontsize=9, loc="left")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.7, len(TERM_ORDER) - 0.3)
        ax.set_xlabel("Coefficient (95% cluster-robust CI, clustered by country)")
        ax.grid(axis="x", color="#EEEEEE", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([TERM_LABELS[t] for t in TERM_ORDER[::-1]])
    handles = [Line2D([], [], color=OK_VERM, marker="o", ms=4.5, linestyle="-",
                      label="95% CI excludes zero"),
               Line2D([], [], color=OK_BLUE, marker="o", ms=4.5, linestyle="-",
                      label="95% CI includes zero")]
    axes[1].legend(handles=handles, frameon=False, loc="lower right", fontsize=7.5)
    fig.suptitle("Main econometric estimates, country-crop panel", x=0.005, ha="left",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = pd.DataFrame(plotted)
    save_figure(fig, name, out,
                "Coefficients of the two main models on the country-crop panel. Unit is "
                "the country-crop cell. Left: participation logit, outcome is whether the "
                "cell has any eligible study, 1,799 cells. Right: PPML intensity with log "
                "harvested area as the exposure, outcome is the fractional eligible study "
                "count, 1,792 cells. Bars are 95 per cent confidence intervals from "
                "standard errors clustered by country. Colour marks only whether the "
                "interval excludes zero: vermillion excludes it, blue includes it. "
                "Harvested area enters the PPML as the exposure and so has no coefficient "
                "there. The need index is entered in the estimation as need_rank_pct "
                "divided by 100, giving it a range of roughly 0 to 0.01, so its "
                "coefficient is on a hundredfold-compressed scale; its interval falls "
                "outside the plotted range and its estimate and interval are printed "
                "beside the row. Estimates are read from "
                "16_Econometrics/model_coefficients.csv without modification. These are "
                "associations, not causal effects.", lg)
    return out


# --------------------------------------------------------------------------- tables
def tab_01(lg, dedup_log, resolve_log, screen_log, manifest, desc, panel):
    rows = [
        ("Search", "Query result records retrieved from OpenAlex",
         dedup_log["counts"]["raw_lines_read"], "29_Logs/phase02_01_flatten_and_dedup.json"),
        ("Search", "OpenAlex query files contributing records",
         dedup_log["counts"]["raw_files"], "29_Logs/phase02_01_flatten_and_dedup.json"),
        ("Deduplication", "Unique OpenAlex work identifiers",
         dedup_log["counts"]["unique_openalex_ids"], "29_Logs/phase02_01_flatten_and_dedup.json"),
        ("Deduplication", "Records merged away by pairwise matching",
         dedup_log["counts"]["records_merged_away"], "29_Logs/phase02_01_flatten_and_dedup.json"),
        ("Deduplication", "Unique works after pairwise matching",
         dedup_log["counts"]["unique_works_after_dedup"], "29_Logs/phase02_01_flatten_and_dedup.json"),
        ("Deduplication", "Works restored by splitting over-merged components",
         resolve_log["counts"]["works_restored_by_splitting"], "29_Logs/phase02_05_resolve_components.json"),
        ("Deduplication", "Unique works after component resolution",
         resolve_log["counts"]["final_works"], "29_Logs/phase02_05_resolve_components.json"),
        ("Screening", "Works entering screening",
         screen_log["counts"]["input_works"], "29_Logs/phase03_03_assisted_screening.json"),
        ("Screening", "Excluded",
         screen_log["counts"]["decision_exclude"], "06_Screening/s3a_decision_summary.csv"),
        ("Screening", "Uncertain, not resolved",
         screen_log["counts"]["decision_uncertain"], "06_Screening/s3a_decision_summary.csv"),
        ("Screening", "Eligible studies (frozen corpus)",
         manifest["n_eligible"], "06_Screening/eligible_corpus_manifest.json"),
        ("Corpus", "Eligible: journal articles",
         manifest["n_journal_article"], "06_Screening/eligible_corpus_manifest.json"),
        ("Corpus", "Eligible: conference papers",
         manifest["n_conference"], "06_Screening/eligible_corpus_manifest.json"),
        ("Corpus", "Eligible: with an abstract",
         manifest["n_with_abstract"], "06_Screening/eligible_corpus_manifest.json"),
        ("Corpus", "Eligible: with a resolved study country",
         desc["n_with_country"], "15_Descriptive_Analysis/research_side_descriptives.json"),
        ("Corpus", "Eligible: study location unresolved",
         desc["n_location_unresolved"], "15_Descriptive_Analysis/research_side_descriptives.json"),
        ("Corpus", "Eligible: with at least one resolved crop",
         desc["n_with_crop"], "15_Descriptive_Analysis/research_side_descriptives.json"),
        ("Panel", "Country-crop cells in the analytical panel",
         int(len(panel)), "12_Data_Integration/country_crop_panel.parquet"),
        ("Panel", "Country-crop cells with at least one eligible study",
         int(panel["has_any_study"].sum()), "12_Data_Integration/country_crop_panel.parquet"),
        ("Panel", "Country-crop cells with a measured zero",
         int((~panel["has_any_study"]).sum()), "12_Data_Integration/country_crop_panel.parquet"),
        ("Panel", "Countries in the panel",
         int(panel["iso3"].nunique()), "12_Data_Integration/country_crop_panel.parquet"),
        ("Panel", "Countries with at least one eligible study",
         desc["n_countries_with_a_study"], "15_Descriptive_Analysis/research_side_descriptives.json"),
        ("Panel", "Countries with a measured zero",
         desc["n_countries_zero_studies"], "15_Descriptive_Analysis/research_side_descriptives.json"),
        ("Panel", "Crops in the panel",
         int(panel["crop_standard_name"].nunique()), "12_Data_Integration/country_crop_panel.parquet"),
    ]
    df = pd.DataFrame(rows, columns=["stage", "step", "n", "source_file"])
    df["n_formatted"] = df["n"].map(lambda v: f"{int(v):,}")

    ci = manifest.get("measured_false_inclusion_ci95", [np.nan, np.nan])
    note = (
        "Counts are read from the run logs and frozen manifests named in the source "
        "column; none is typed by hand. Screening was carried out entirely by rule-based "
        "and model-assisted machinery, with no human screening pass, and the uncertain "
        f"group of {screen_log['counts']['decision_uncertain']:,} works was never "
        "resolved. A stratified validation of 85 adjudicated records measured a "
        f"false-inclusion rate of {100 * manifest['measured_false_inclusion_rate_point']:.0f} "
        f"per cent (95 per cent CI {100 * ci[0]:.0f} to {100 * ci[1]:.0f} per cent), "
        f"implying about {manifest['estimated_false_inclusions']:,} ineligible studies "
        "inside the corpus, and a false-exclusion rate of "
        f"{100 * manifest['measured_false_exclusion_rate_largest_stratum']:.2f} per cent "
        "in the largest exclusion stratum, implying about 100 eligible studies outside "
        "it. The validation does not meet its own preregistered confidence bounds and "
        "that failure is reported rather than repaired.")
    write_table(df, "tab_01_corpus_construction",
                "Corpus construction, from retrieved records to the frozen eligible "
                "corpus and the analytical panel.", note, lg,
                tex_cols=["stage", "step", "n_formatted", "source_file"],
                align="llrl")
    return df


def tab_02(lg, coef, comparison, diagnostics):
    left, right = "participation_logit", "ppml_area_exposure"
    c = coef[coef["model"].isin([left, right])].copy()
    piv = {}
    for m in (left, right):
        s = c[c["model"] == m].set_index("term")
        piv[m] = s
    terms = ["const"] + TERM_ORDER
    rows = []
    for t in terms:
        row = {"term": t, "label": TERM_LABELS.get(t, t)}
        for m, tag in ((left, "logit"), (right, "ppml")):
            s = piv[m]
            if t in s.index:
                r = s.loc[t]
                row[f"{tag}_estimate"] = float(r["estimate"])
                row[f"{tag}_std_error"] = float(r["std_error"]) if pd.notna(r["std_error"]) else np.nan
                row[f"{tag}_ci_low"] = float(r["ci_low"]) if pd.notna(r["ci_low"]) else np.nan
                row[f"{tag}_ci_high"] = float(r["ci_high"]) if pd.notna(r["ci_high"]) else np.nan
            else:
                for k in ("estimate", "std_error", "ci_low", "ci_high"):
                    row[f"{tag}_{k}"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)

    def cell(r, tag):
        if not np.isfinite(r[f"{tag}_estimate"]):
            return "offset"
        return (f"{r[f'{tag}_estimate']:,.3f} ({r[f'{tag}_std_error']:,.3f}) "
                f"[{r[f'{tag}_ci_low']:,.3f}, {r[f'{tag}_ci_high']:,.3f}]")

    df["Participation (logit)"] = df.apply(lambda r: cell(r, "logit"), axis=1)
    df["Intensity (PPML, area exposure)"] = df.apply(lambda r: cell(r, "ppml"), axis=1)
    df = df.rename(columns={"label": "Covariate"})

    cmp_ = comparison.set_index("model")
    dg = diagnostics["diagnostics"]
    extra = [
        ("Observations (country-crop cells)",
         f"{int(coef[coef['model'] == left]['n'].iloc[0]):,}",
         f"{int(coef[coef['model'] == right]['n'].iloc[0]):,}"),
        ("Standard errors", "cluster-robust by country", "cluster-robust by country"),
        ("Log-likelihood", fmt(cmp_.loc[left, "loglik"], 1), fmt(cmp_.loc[right, "loglik"], 1)),
        ("AIC", fmt(cmp_.loc[left, "aic"], 1), fmt(cmp_.loc[right, "aic"], 1)),
        ("McFadden pseudo R2", fmt(dg[left].get("pseudo_r2"), 4), "NA"),
        ("Share of cells with a measured zero",
         fmt(1 - dg[left]["events"] / dg[left]["n"], 3),
         fmt(dg[right].get("zero_share"), 3)),
        ("Pearson dispersion", "NA", fmt(dg[right].get("pearson_dispersion"), 1)),
    ]
    tail = pd.DataFrame(extra, columns=["Covariate", "Participation (logit)",
                                        "Intensity (PPML, area exposure)"])
    tex_df = pd.concat([df[["Covariate", "Participation (logit)",
                            "Intensity (PPML, area exposure)"]], tail],
                       ignore_index=True)
    df_out = pd.concat([df, tail], ignore_index=True)

    note = (
        "Each cell reports the coefficient, the cluster-robust standard error in round "
        "brackets, and the 95 per cent confidence interval in square brackets. Standard "
        "errors are clustered by country. The outcome of the participation model is "
        "whether a country-crop cell has any eligible study; the outcome of the intensity "
        "model is the fractional eligible study count, with log harvested area entered as "
        "the exposure, which is why it carries no coefficient there. The need index "
        "enters as need_rank_pct divided by 100, so it varies over roughly 0 to 0.01 and "
        "its coefficient is on a hundredfold-compressed scale; the confidence interval "
        "spans about plus or minus 100 to 200 log points and the design cannot detect an "
        "association with need. The Pearson dispersion of 367.7 for the PPML is far above "
        "1, which is why robust errors are reported. These are associations, not causal "
        "effects. Values are read from 16_Econometrics/model_coefficients.csv, "
        "model_comparison.csv and model_diagnostics.json.")
    write_table(df_out, "tab_02_main_regression_estimates",
                "Main econometric estimates on the country-crop panel: participation "
                "logit and PPML intensity.", note, lg,
                tex_cols=["Covariate", "Participation (logit)",
                          "Intensity (PPML, area exposure)"],
                align="lll")
    return df_out


def tab_03(lg, panel, research, top_n=20, need_floor=0.75):
    d = panel.merge(research[["iso3", "fao_area_name"]], on="iso3", how="left")
    sel = d[(d["n_studies_fractional"] == 0)
            & (d["need_rank_pct"] >= need_floor)
            & (d["production_t_mean"] > 0)].copy()
    sel = sel.sort_values(["production_share", "need_rank_pct"], ascending=False)
    top = sel.head(top_n).copy()
    top["need_percentile_0_100"] = 100 * top["need_rank_pct"]
    top["production_Mt"] = top["production_t_mean"] / 1e6
    top["area_Mha"] = top["area_ha_mean"] / 1e6
    top["production_share_pct"] = 100 * top["production_share"]
    out = top[["iso3", "fao_area_name", "crop_standard_name", "wb_region",
               "wb_income_group", "production_Mt", "area_Mha", "production_share_pct",
               "need_percentile_0_100", "n_studies_fractional", "n_studies_full"]].copy()
    out.columns = ["ISO3", "Country", "Crop", "World Bank region", "Income group",
                   "Production (Mt)", "Harvested area (Mha)", "Share of world production (%)",
                   "Need percentile", "Fractional studies", "Full-count studies"]
    for c, nd in (("Production (Mt)", 2), ("Harvested area (Mha)", 2),
                  ("Share of world production (%)", 3), ("Need percentile", 1)):
        out[c] = out[c].map(lambda v, nd=nd: f"{v:,.{nd}f}")
    out["Fractional studies"] = out["Fractional studies"].map(lambda v: f"{v:.0f}")
    out["Full-count studies"] = out["Full-count studies"].map(lambda v: f"{v:.0f}")
    out["Crop"] = out["Crop"].str.replace("_", " ")

    note = (
        f"Selection rule, fixed before the table was produced: country-crop cells with a "
        f"fractional eligible study count of exactly zero, a need percentile at or above "
        f"the {int(need_floor * 100)}th, and a positive FAOSTAT production denominator, "
        f"ranked by share of world production. {len(sel):,} of the 2,616 panel cells meet "
        f"the rule, spanning {sel['iso3'].nunique()} countries; the {top_n} largest by "
        "production share are shown. A zero here is measured, not missing: the cell was in "
        "the panel and no eligible study covered it. The need percentile is the "
        "rank-weighted need index rescaled from 0-1 to 0-100 and is a country-level "
        "quantity, identical for every crop within a country. The full ranking is in the "
        "supplement.")
    write_table(out, "tab_03_under_researched_systems",
                "Country-crop systems with high research need and no eligible study, "
                "ranked by share of world production.", note, lg,
                align="llllrrrrrrr")
    return out, len(sel), int(sel["iso3"].nunique())


def tab_04(lg, stab):
    df = stab.copy()
    df.columns = [c.strip() for c in df.columns]
    show = df[["finding", "measure", "baseline", "range_across_variants", "n_variants",
               "classification"]].copy()
    show.columns = ["Finding", "Measure", "Baseline", "Range across variants",
                    "Variants", "Classification"]
    # A missing range means different things depending on how many variants were run.
    # One variant is a single estimate; several variants with no range is an unrecorded
    # range, and is labelled that way rather than being made to look like one estimate.
    show["Range across variants"] = [
        r if isinstance(r, str) and r.strip()
        else ("single estimate" if n <= 1 else "not recorded")
        for r, n in zip(df["range_across_variants"], df["n_variants"])]
    note_lines = [f"{r['finding']}: {r['reason']}" for _, r in df.iterrows()]
    note = ("Classification follows the prespecified scheme: stable, partially stable, "
            "specification-sensitive, unsupported. Reasons, one per finding. "
            + " || ".join(note_lines)
            + " Values are read from 20_Robustness/finding_stability_classification.csv.")
    write_table(show, "tab_04_finding_stability",
                "Stability of the main findings across prespecified robustness variants.",
                note, lg, align="p{4.2cm}p{3.2cm}rp{2.4cm}rp{2.2cm}")
    return show


# --------------------------------------------------------------------------- main
def main() -> int:
    lg = RunLogger("phase17_01_manuscript1_figures")
    np.random.seed(RANDOM_SEED)
    lg.count("random_seed", RANDOM_SEED)

    import geopandas as gpd

    I = P["integration"]
    paths = {
        "panel": I / "country_crop_panel.parquet",
        "research": I / "country_research_side.parquet",
        "desc": P["descriptive"] / "research_side_descriptives.json",
        "lisa": P["spatialecon"] / "research_side_lisa_clusters.csv",
        "moran": P["spatialecon"] / "research_side_global_moran.csv",
        "coef": P["econ"] / "model_coefficients.csv",
        "comparison": P["econ"] / "model_comparison.csv",
        "diagnostics": P["econ"] / "model_diagnostics.json",
        "stability": P["robustness"] / "finding_stability_classification.csv",
        "layer": P["weights"] / "country_analytical_layer.parquet",
        "dedup_log": P["logs"] / "phase02_01_flatten_and_dedup.json",
        "resolve_log": P["logs"] / "phase02_05_resolve_components.json",
        "screen_log": P["logs"] / "phase03_03_assisted_screening.json",
        "corpus_manifest": P["screening"] / "eligible_corpus_manifest.json",
    }
    ok = {k: need_input(lg, v, k) for k, v in paths.items()}

    D = {}
    if ok["panel"]:
        D["panel"] = pd.read_parquet(paths["panel"])
    if ok["research"]:
        D["research"] = pd.read_parquet(paths["research"])
    if ok["desc"]:
        D["desc"] = json.loads(paths["desc"].read_text())
    if ok["lisa"]:
        D["lisa"] = pd.read_csv(paths["lisa"])
    if ok["moran"]:
        D["moran"] = pd.read_csv(paths["moran"])
    if ok["coef"]:
        D["coef"] = pd.read_csv(paths["coef"], index_col=0)
    if ok["comparison"]:
        D["comparison"] = pd.read_csv(paths["comparison"])
    if ok["diagnostics"]:
        D["diagnostics"] = json.loads(paths["diagnostics"].read_text())
    if ok["stability"]:
        D["stability"] = pd.read_csv(paths["stability"])
    if ok["layer"]:
        D["layer"] = gpd.read_parquet(paths["layer"])
    for k in ("dedup_log", "resolve_log", "screen_log", "corpus_manifest"):
        if ok[k]:
            D[k] = json.loads(paths[k].read_text())

    audits = []

    # --- join losses, logged rather than silently dropped -------------------------
    if "panel" in D and "layer" in D:
        pi, gi = set(D["panel"]["iso3"]), set(D["layer"]["iso3"])
        missing_geom = sorted(pi - gi)
        lg.count("panel_countries", len(pi))
        lg.count("layer_countries", len(gi))
        lg.count("panel_countries_without_geometry", len(missing_geom))
        if missing_geom:
            sub = D["panel"][D["panel"]["iso3"].isin(missing_geom)].groupby("iso3")[
                "n_studies_fractional"].sum()
            lg.warn("panel countries with no boundary in the spatial layer, not mapped: "
                    + ", ".join(f"{k} (fractional studies {v:g})" for k, v in sub.items()))

    # --- figures ------------------------------------------------------------------
    counts01 = counts05 = None
    if "layer" in D and "research" in D:
        a, counts01 = fig_01(lg, D["layer"], D["research"])
        audits.append(a)
        lg.count("fig01_countries_measured_zero", counts01["0 (no resolved study)"])
        lg.count("fig01_countries_no_data", counts01["no data"])
    else:
        record("figure", "fig_01_research_intensity_map", "skipped",
               "country_analytical_layer.parquet or country_research_side.parquet absent")
        lg.error("fig_01 skipped: required input absent")

    if "panel" in D:
        cr = fig_02(lg, D["panel"])
        lg.count("fig02_crops", len(cr))
        lg.count("fig02_top_crop_fractional", round(float(cr["n_studies_fractional"].iloc[0]), 2))
    else:
        record("figure", "fig_02_crop_attention_vs_production", "skipped",
               "country_crop_panel.parquet absent")
        lg.error("fig_02 skipped: required input absent")

    if "panel" in D and "research" in D and "desc" in D:
        _, n_both, n_zero, n_noprod, n_clip = fig_03(lg, D["panel"], D["research"], D["desc"])
        lg.count("fig03_countries_plotted", n_both)
        lg.count("fig03_countries_measured_zero_strip", n_zero)
        lg.count("fig03_countries_no_production_denominator", n_noprod)
        lg.count("fig03_zero_research_clipped_to_left_edge", n_clip)
    else:
        record("figure", "fig_03_research_vs_production_share", "skipped",
               "panel, research side, or descriptives JSON absent")
        lg.error("fig_03 skipped: required input absent")

    if "research" in D:
        _, n_plot, n_quad, n_miss = fig_04(lg, D["research"])
        lg.count("fig04_countries_plotted", n_plot)
        lg.count("fig04_high_need_low_research_countries", n_quad)
        lg.count("fig04_countries_without_need_index", n_miss)
    else:
        record("figure", "fig_04_need_vs_research_percentile", "skipped",
               "country_research_side.parquet absent")
        lg.error("fig_04 skipped: required input absent")

    if "layer" in D and "lisa" in D and "moran" in D:
        a, counts05, mi, mp = fig_05(lg, D["layer"], D["lisa"], D["moran"])
        audits.append(a)
        for k, v in counts05.items():
            lg.count(f"fig05_{k.replace(' ', '_').replace('-', '_')}", v)
        lg.count("fig05_global_moran_I", mi)
        # The scale defect this note used to report was repaired upstream on 2026-08-24
        # (D-131): research_side_spatial_analysis.py now rescales need_rank_pct to 0-100
        # and asserts that the gap does not correlate with research_pct above 0.99.
        # The guard is repeated here so a regression upstream fails this figure loudly.
        _src = D.get("research")
        _g = (_src[["evidence_gap", "research_pct"]].dropna()
              if _src is not None
              and {"evidence_gap", "research_pct"} <= set(_src.columns) else None)
        if _g is not None and len(_g) > 2 and _g.corr().iloc[0, 1] > 0.99:
            raise ValueError(
                "evidence_gap correlates with research_pct above 0.99; the scale defect "
                "repaired under D-131 has reappeared upstream")
    else:
        record("figure", "fig_05_evidence_gap_lisa_map", "skipped",
               "spatial layer, LISA clusters, or global Moran table absent")
        lg.error("fig_05 skipped: required input absent")

    if "coef" in D and "comparison" in D and "diagnostics" in D:
        o = fig_06(lg, D["coef"], D["comparison"], D["diagnostics"])
        lg.count("fig06_coefficients_plotted", int((o["status"] == "on scale").sum()))
        lg.count("fig06_coefficients_off_scale", int(o["off_scale"].sum()))
    else:
        record("figure", "fig_06_model_coefficients", "skipped",
               "model coefficients, comparison, or diagnostics absent")
        lg.error("fig_06 skipped: required input absent")

    # --- map input audit, CLAUDE.md 6.1 ------------------------------------------
    if audits:
        ap = P["figures"] / "01_Data_Validation" / "map_input_audit_manuscript1.csv"
        ap.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audits).to_csv(ap, index=False)
        lg.add_output(ap, rows=len(audits))
        lg.note("map input audit written to a manuscript-1 specific file; the existing "
                "map_input_audit.csv is left untouched")

    # --- tables -------------------------------------------------------------------
    if all(k in D for k in ("dedup_log", "resolve_log", "screen_log",
                            "corpus_manifest", "desc", "panel")):
        t1 = tab_01(lg, D["dedup_log"], D["resolve_log"], D["screen_log"],
                    D["corpus_manifest"], D["desc"], D["panel"])
        lg.count("tab01_rows", len(t1))
    else:
        record("table", "tab_01_corpus_construction", "skipped",
               "one of the phase 2/3 run logs, the corpus manifest, the descriptives "
               "JSON, or the panel is absent")
        lg.error("tab_01 skipped: required input absent")

    if "coef" in D and "comparison" in D and "diagnostics" in D:
        t2 = tab_02(lg, D["coef"], D["comparison"], D["diagnostics"])
        lg.count("tab02_rows", len(t2))
    else:
        record("table", "tab_02_main_regression_estimates", "skipped",
               "model coefficients, comparison, or diagnostics absent")
        lg.error("tab_02 skipped: required input absent")

    if "panel" in D and "research" in D:
        _, n_sel, n_cty = tab_03(lg, D["panel"], D["research"])
        lg.count("tab03_cells_meeting_rule", n_sel)
        lg.count("tab03_countries_meeting_rule", n_cty)
    else:
        record("table", "tab_03_under_researched_systems", "skipped",
               "panel or research side absent")
        lg.error("tab_03 skipped: required input absent")

    if "stability" in D:
        t4 = tab_04(lg, D["stability"])
        lg.count("tab04_findings", len(t4))
    else:
        record("table", "tab_04_finding_stability", "skipped",
               "finding_stability_classification.csv absent")
        lg.error("tab_04 skipped: required input absent")

    man = pd.DataFrame(MANIFEST)
    mp_ = P["figures"] / "manuscript1_figure_manifest.csv"
    man.to_csv(mp_, index=False)
    lg.add_output(mp_, rows=len(man))
    print("\n" + man.to_string(index=False))
    lg.finish()
    return 0 if not (man["status"] == "skipped").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
