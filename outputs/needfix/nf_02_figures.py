"""Regenerate Figures 4 and 5 on the corrected nine-component coverage floor."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import patheffects as pe
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402
SUB = ROOT / "Final Manuscript" / "Manuscript 1" / "09_Wiley_Submission"
PNG = HERE / "png"; PNG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 300})
OKB, OKV = "#0072B2", "#D55E00"
COL = {"High-High": "#b2182b", "Low-Low": "#2166ac", "High-Low": "#f4a582",
       "Low-High": "#92c5de", "Not significant": "#f0f0f0"}


def save(fig, name):
    for d in (HERE, SUB):
        fig.savefig(d / name, bbox_inches="tight", dpi=300)
    fig.savefig(PNG / name.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig); print(f"  wrote {name}")


def main():
    lg = RunLogger("nf_02_figures")
    lay = gpd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    lisa = pd.read_csv(HERE / "gap9_corrected_lisa.csv")

    # ---------------- Figure 4: need percentile against research percentile ----------
    d = lisa.copy()
    d["need_pct"] = d.research_pct - d.gap
    mn, mr = d.need_pct.median(), d.research_pct.median()
    quad = (d.need_pct >= mn) & (d.research_pct <= mr)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.axvspan(mn, 100, ymin=0, ymax=(mr / 100), color=OKV, alpha=0.10, lw=0)
    ax.scatter(d.loc[~quad, "need_pct"], d.loc[~quad, "research_pct"], s=16, c=OKB, alpha=.65, lw=0)
    ax.scatter(d.loc[quad, "need_pct"], d.loc[quad, "research_pct"], s=18, c=OKV, alpha=.85, lw=0)
    ax.axvline(mn, color="0.55", lw=.8, ls="--"); ax.axhline(mr, color="0.55", lw=.8, ls="--")
    for iso in ["USA", "CHN", "IND", "BRA", "NGA", "ETH", "COD", "SDN", "NER", "TCD"]:
        r = d[d.iso3 == iso]
        if len(r):
            ax.annotate(iso, (r.need_pct.iloc[0], r.research_pct.iloc[0]), xytext=(3, 3),
                        textcoords="offset points", fontsize=7)
    ax.set_xlabel("Research-need percentile (nine-component index)")
    ax.set_ylabel("Research-output percentile")
    ax.text(.985, .02, f"high need, low research: n = {int(quad.sum())}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=OKV)
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save(fig, "fig_04_v2_need_vs_research.pdf")
    print(f"    countries={len(d)}  quadrant={int(quad.sum())}")

    # ---------------- Figure 5: LISA map with insets ---------------------------------
    g = lay.merge(lisa[["iso3", "lisa_cat_gap9"]], on="iso3", how="left")
    g["cls"] = g.lisa_cat_gap9.where(g.lisa_cat_gap9.notna(), "No need index")
    gp = g.to_crs("+proj=eqearth")
    fig = plt.figure(figsize=(9.2, 6.9))
    axm = fig.add_axes([0.0, 0.40, 1.0, 0.60])
    for cat, col in COL.items():
        s = gp[gp.cls == cat]
        if len(s): s.plot(ax=axm, color=col, edgecolor="white", linewidth=.25)
    nd = gp[gp.cls == "No need index"]
    if len(nd): nd.plot(ax=axm, color="#d9d9d9", edgecolor="white", linewidth=.25, hatch="///")
    axm.set_axis_off()
    hs = [Patch(facecolor=c, label=f"{k}  (n = {int((g.cls==k).sum())})") for k, c in COL.items()]
    hs.append(Patch(facecolor="#d9d9d9", hatch="///",
                    label=f"No need index  (n = {int((g.cls=='No need index').sum())})"))
    axm.legend(handles=hs, loc="lower left", frameon=False, ncol=2, fontsize=7.2,
               bbox_to_anchor=(.005, .02), handlelength=1.4, columnspacing=1.1, labelspacing=.35)
    yb = axm.get_ylim(); axm.set_ylim(yb[0] * .62, yb[1] * 1.02)

    REG = [("(a) Western and northern Europe", (-11, 35, 32, 71), [.055, .02, .395, .335],
            "High-High", {"LUX": (.6, -1.4), "BEL": (-1.1, .4), "NLD": (.5, 1.0), "CHE": (.3, -.8),
                          "SVN": (1.2, -.6), "SVK": (.8, .5), "EST": (.6, .3), "DNK": (-.6, .9),
                          "CZE": (.2, .6)}),
           ("(b) Central and eastern Africa", (8, -12, 48, 23), [.552, .02, .395, .335],
            "Low-Low", {"COG": (-2.2, 1.4), "GAB": (-2.4, -.6), "COD": (1.0, -1.0), "DJI": (2.6, .8)})]
    for title, (lo0, la0, lo1, la1), rect, focus, nudge in REG:
        axi = fig.add_axes(rect)
        box = gpd.GeoSeries.from_wkt(
            [f"POLYGON(({lo0} {la0},{lo1} {la0},{lo1} {la1},{lo0} {la1},{lo0} {la0}))"], crs="EPSG:4326")
        cp = gpd.clip(g.to_crs("EPSG:4326"), box.iloc[0]).to_crs("+proj=eqearth")
        for cat, col in COL.items():
            s = cp[cp.cls == cat]
            if len(s): s.plot(ax=axi, color=col, edgecolor="white", linewidth=.35)
        nds = cp[cp.cls == "No need index"]
        if len(nds): nds.plot(ax=axi, color="#d9d9d9", edgecolor="white", linewidth=.35, hatch="///")
        lab = cp[cp.cls == focus]
        for _, r in lab.iterrows():
            c = r.geometry.representative_point()
            dx, dy = nudge.get(r.iso3, (0., 0.))
            axi.annotate(r.iso3, (c.x + dx * 90000, c.y + dy * 90000), ha="center", va="center",
                         fontsize=5.4, color="white", fontweight="bold",
                         path_effects=[pe.withStroke(linewidth=1.3, foreground="#00000088")])
        bp = box.to_crs("+proj=eqearth").total_bounds
        axi.set_xlim(bp[0], bp[2]); axi.set_ylim(bp[1], bp[3])
        axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_visible(True); sp.set_edgecolor("0.35"); sp.set_linewidth(.9)
        tot = int((g.cls == focus).sum())
        axi.set_title(f"{title}: {len(lab)} of the {tot} {focus} countries", fontsize=7.8, pad=3)
        box.to_crs("+proj=eqearth").boundary.plot(ax=axm, color="0.25", linewidth=.9, linestyle="--")
        print(f"    inset {focus}: {len(lab)} of {tot} shown")
    save(fig, "fig_05_v2_mismatch_lisa.pdf")
    for f in ["fig_04_v2_need_vs_research.pdf", "fig_05_v2_mismatch_lisa.pdf"]:
        lg.add_output(HERE / f)
    lg.finish()


if __name__ == "__main__":
    main()
