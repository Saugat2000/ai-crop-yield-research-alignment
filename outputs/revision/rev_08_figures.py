"""Revision figures.

  item  6 - predicted probability of participation against log harvested area
  item  7 - concentration (Gini) by period
  item  8 - need-index component correlation heatmap
  item  9 - LISA cluster map with inset panels for Europe and Africa
  Step 22 - study identification and analytical sample flow

Styling matches the existing figures: Okabe-Ito colours, 9 pt base font, 200 dpi,
no internal analysis shorthand in any label.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

# The submission folder lives outside this replication repository. When it is absent
# (the normal case for anyone running this repo) figures are written here only.
_SUB = ROOT / "Final Manuscript" / "Manuscript 1" / "09_Wiley_Submission"
SUB = _SUB if _SUB.is_dir() else None
OUT = HERE
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 200})
OKB, OKV, OKG, OKY = "#0072B2", "#D55E00", "#009E73", "#E69F00"
X = ["log_area", "rd", "need9", "log_gdp_pc", "tertiary", "internet", "log_population"]


def save(fig, name):
    for d in [d for d in (OUT, SUB) if d is not None]:
        fig.savefig(d / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def main():
    lg = RunLogger("rev_08_figures")
    e = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet").merge(
        pd.read_parquet(ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet")
        [["iso3", "need9_rank_pct"]].rename(columns={"need9_rank_pct": "need9"}),
        on="iso3", how="left")

    # ------------------------------------------------- item 6: predicted probability
    M = sm.add_constant(e[X].astype(float), has_constant="add")
    m = sm.Logit(e.studied.astype(float), M).fit(disp=0, cov_type="cluster",
                                                 cov_kwds={"groups": e.iso3.to_numpy()})
    grid = np.linspace(e.log_area.quantile(.01), e.log_area.quantile(.99), 200)
    base = {c: e[c].mean() for c in X}
    Xg = pd.DataFrame({c: np.repeat(base[c], len(grid)) for c in X})
    Xg["log_area"] = grid
    Xg = sm.add_constant(Xg, has_constant="add")[M.columns]
    pr = m.predict(Xg)
    xb = Xg.values @ m.params.values
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xg.values, m.cov_params().values, Xg.values))
    lo = 1 / (1 + np.exp(-(xb - 1.96 * se))); hi = 1 / (1 + np.exp(-(xb + 1.96 * se)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(grid, lo, hi, color=OKB, alpha=0.18, lw=0)
    ax.plot(grid, pr, color=OKB, lw=1.8)
    q1, q3 = e.log_area.quantile(.25), e.log_area.quantile(.75)
    ax.axvspan(q1, q3, color="0.85", alpha=0.35, lw=0, zorder=0)
    ax.set_xlabel("Log harvested area (hectares)")
    ax.set_ylabel("Predicted probability a country-crop system is studied")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.hist(e.log_area, bins=45, color="0.55", alpha=0.35, lw=0)
    ax2.set_yticks([]); ax2.set_ylabel("")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.text(0.02, 0.96, "Shaded band: interquartile range of harvested area\n"
                        "Other covariates held at their sample means",
            transform=ax.transAxes, va="top", fontsize=7.5, color="0.25")
    save(fig, "fig_A3_predicted_probability.pdf")
    pd.DataFrame({"log_area": grid, "predicted_probability": pr,
                  "ci_low": lo, "ci_high": hi}).to_csv(
        OUT / "fig_A3_predicted_probability_data.csv", index=False)

    # ------------------------------------------------- item 7: concentration by period
    tg = pd.read_csv(OUT / "temporal_gini.csv")
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    full = tg[~tg.period.str.contains("partial")]
    part = tg[tg.period.str.contains("partial")]
    ax.plot(full.period, full.gini_all_countries, "o-", color=OKB, lw=1.8, ms=6,
            label="Complete periods")
    if len(part):
        ax.plot(part.period, part.gini_all_countries, "o", color=OKV, ms=7,
                label="Partial year (2026)")
        ax.plot([full.period.iloc[-1], part.period.iloc[0]],
                [full.gini_all_countries.iloc[-1], part.gini_all_countries.iloc[0]],
                ":", color=OKV, lw=1.4)
    for _, r in tg.iterrows():
        ax.annotate(f"{r.gini_all_countries:.3f}\n({int(r.studies)} studies)",
                    (r.period, r.gini_all_countries), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=7)
    ax.set_ylabel("Gini coefficient of country research counts")
    ax.set_xlabel("Publication period")
    ax.set_ylim(0.80, 0.93); ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_A4_temporal_concentration.pdf")

    # ------------------------------------------------- item 8: component heatmap
    sp = pd.read_csv(OUT / "need_component_corr_spearman.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(sp.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sp))); ax.set_yticks(range(len(sp)))
    ax.set_xticklabels(sp.columns, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(sp.index, fontsize=7.5)
    for i in range(len(sp)):
        for j in range(len(sp)):
            v = sp.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if abs(v) > 0.55 else "0.15")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Spearman correlation", fontsize=8)
    ax.set_title("Research-need components, direction-harmonised percentile ranks", fontsize=9)
    save(fig, "fig_A5_need_component_correlations.pdf")

    # ------------------------------------------------- item 9: LISA map with insets
    lay = gpd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    lisa = pd.read_csv(ROOT / "31_Presubmission_Audit" / "gap9_lisa_clusters.csv")
    g = lay.merge(lisa, on="iso3", how="left")
    CAT = {"High-High": "#B2182B", "Low-Low": "#2166AC",
           "High-Low": "#EF8A62", "Low-High": "#67A9CF"}
    qcol = "lisa_cat_gap9"
    if qcol not in g.columns:
        print("  LISA category column not found; skipping inset map")
    else:
        gg = g.to_crs("+proj=eqearth")
        fig = plt.figure(figsize=(7.2, 5.6))
        axm = fig.add_axes([0.0, 0.30, 1.0, 0.70])
        gg.plot(ax=axm, color="#F2F2F2", edgecolor="white", linewidth=0.25)
        for lab, col in CAT.items():
            s = gg[gg[qcol] == lab]
            if len(s):
                s.plot(ax=axm, color=col, edgecolor="white", linewidth=0.25,
                       label=f"{lab} (n = {len(s)})")
        axm.set_axis_off()
        axm.legend(frameon=False, loc="lower left", fontsize=7.5, ncol=2)
        for i, (name, bnds) in enumerate([
                ("Western and northern Europe", (-11, 35, 32, 66)),
                ("Central and eastern Africa", (8, -12, 48, 22))]):
            ax = fig.add_axes([0.06 + i * 0.50, 0.02, 0.40, 0.30])
            w = g.to_crs("EPSG:4326")
            wsub = w.cx[bnds[0]:bnds[2], bnds[1]:bnds[3]].to_crs("+proj=eqearth")
            wsub.plot(ax=ax, color="#F2F2F2", edgecolor="white", linewidth=0.3)
            for lab, col in CAT.items():
                s = wsub[wsub[qcol] == lab]
                if len(s):
                    s.plot(ax=ax, color=col, edgecolor="white", linewidth=0.3)
            ax.set_axis_off()
            ax.set_title(name, fontsize=8)
        save(fig, "fig_04_v3_mismatch_lisa_insets.pdf")

    # ------------------------------------------------- Step 22: sample flow
    flow = [("Records retrieved from OpenAlex", "102,892"),
            ("Unique works after duplicate resolution", "43,543"),
            ("Eligible studies after screening", "7,045"),
            ("Studies with accepted country evidence\nand a FAOSTAT-matched crop", "2,031"),
            ("Fractional study-equivalents inside\nthe 2,616-cell panel", "2,020.2"),
            ("Country-crop cells carrying research", "485 of 2,616"),
            ("Cells in the estimation sample\n(all covariates observed)", "1,799")]
    drops = ["59,349 duplicate records removed", "36,498 screened out on six eligibility criteria",
             "5,014 studies without usable country or crop evidence",
             "10.9 study-equivalents on cells outside the panel", None,
             "817 cells dropped, R&D expenditure binding", None]
    fig, ax = plt.subplots(figsize=(6.4, 8.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(flow) * 2 + 1); ax.set_axis_off()
    for i, ((lab, n), dr) in enumerate(zip(flow, drops)):
        y = (len(flow) - i) * 2 - 0.5
        ax.add_patch(FancyBboxPatch((0.4, y - 0.62), 5.6, 1.24, boxstyle="round,pad=0.06",
                                    fc="#EAF2F8", ec=OKB, lw=1.0))
        ax.text(0.7, y, lab, va="center", ha="left", fontsize=8)
        ax.text(5.75, y, n, va="center", ha="right", fontsize=8.5, fontweight="bold", color=OKB)
        if i < len(flow) - 1:
            ax.add_patch(FancyArrowPatch((3.2, y - 0.66), (3.2, y - 1.34),
                                         arrowstyle="-|>", mutation_scale=11, color="0.45", lw=1.0))
        if dr:
            ax.text(6.25, y - 1.0, dr, va="center", ha="left", fontsize=7, color=OKV)
    ax.set_title("Study identification and analytical sample flow", fontsize=9.5, pad=6)
    save(fig, "fig_A6_sample_flow.pdf")

    for f in ["fig_A3_predicted_probability.pdf", "fig_A4_temporal_concentration.pdf",
              "fig_A5_need_component_correlations.pdf", "fig_A6_sample_flow.pdf"]:
        lg.add_output(OUT / f)
    lg.finish()
    print("\nrev_08_figures complete")


if __name__ == "__main__":
    main()
