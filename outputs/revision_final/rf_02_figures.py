"""Final revision figures.

  Figure 4  (fig_05_v2_mismatch_lisa.pdf)      - inset panels for Europe and Africa
  Figure 2  (fig_03_v2_scale_alignment.pdf)    - visible open markers for zero-research units
  Figure A5 (fig_A4_temporal_concentration.pdf)- honest y-axis, full-period reference line

Style matches the existing figures: 9 pt base font, Okabe-Ito accents, the LISA palette
unchanged, 300 dpi PDF plus a 150 dpi PNG preview for each.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, ConnectionPatch
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

SUB = ROOT / "Final Manuscript" / "Manuscript 1" / "09_Wiley_Submission"
OUT = HERE
PNG = OUT / "png_previews"; PNG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 300})
OKB, OKV, OKG = "#0072B2", "#D55E00", "#009E73"
COL = {"High-High": "#b2182b", "Low-Low": "#2166ac", "High-Low": "#f4a582",
       "Low-High": "#92c5de", "Not significant": "#f0f0f0"}


def save(fig, name):
    for d in (OUT, SUB):
        fig.savefig(d / name, bbox_inches="tight", dpi=300)
    fig.savefig(PNG / name.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {name} (+ PNG preview)")


def fig4_insets(lg):
    """Figure 4: world LISA map with zoomed insets over the two cluster regions."""
    lay = gpd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    lisa = pd.read_csv(ROOT / "31_Presubmission_Audit" / "gap9_lisa_clusters.csv")
    g = lay.merge(lisa, on="iso3", how="left")
    g["cls"] = g["lisa_cat_gap9"].where(g["lisa_cat_gap9"].notna(), "No need index")
    gp = g.to_crs("+proj=eqearth")

    fig = plt.figure(figsize=(9.2, 6.9))
    axm = fig.add_axes([0.0, 0.40, 1.0, 0.60])
    for cat, col in COL.items():
        s = gp[gp.cls == cat]
        if len(s):
            s.plot(ax=axm, color=col, edgecolor="white", linewidth=0.25)
    nd = gp[gp.cls == "No need index"]
    if len(nd):
        nd.plot(ax=axm, color="#d9d9d9", edgecolor="white", linewidth=0.25, hatch="///")
    axm.set_axis_off()
    hs = [Patch(facecolor=c, label=f"{k}  (n = {int((g.cls == k).sum())})") for k, c in COL.items()]
    hs.append(Patch(facecolor="#d9d9d9", hatch="///",
                    label=f"No need index  (n = {int((g.cls == 'No need index').sum())})"))
    axm.legend(handles=hs, loc="lower left", frameon=False, ncol=2, fontsize=7.2,
               bbox_to_anchor=(0.005, 0.02), handlelength=1.4, columnspacing=1.1,
               labelspacing=0.35)
    # trim the empty polar margins so the inset row sits close under the map
    yb = axm.get_ylim()
    axm.set_ylim(yb[0] * 0.62, yb[1] * 1.02)

    REGIONS = [
        ("(a) Western and northern Europe", (-11, 35, 32, 71), [0.055, 0.02, 0.395, 0.335],
         "High-High", {"LUX": (0.6, -1.4), "BEL": (-1.1, 0.4), "NLD": (0.5, 1.0),
                       "CHE": (0.3, -0.8), "SVN": (1.2, -0.6), "SVK": (0.8, 0.5),
                       "EST": (0.6, 0.3), "DNK": (-0.6, 0.9), "CZE": (0.2, 0.6)}),
        ("(b) Central and eastern Africa", (8, -12, 48, 23), [0.552, 0.02, 0.395, 0.335],
         "Low-Low", {"COG": (-2.2, 1.4), "GAB": (-2.4, -0.6), "COD": (1.0, -1.0),
                     "DJI": (2.6, 0.8)}),
    ]
    from matplotlib import patheffects as pe
    for title, (lon0, lat0, lon1, lat1), rect, focus, nudge in REGIONS:
        axi = fig.add_axes(rect)
        box = gpd.GeoSeries.from_wkt(
            [f"POLYGON(({lon0} {lat0},{lon1} {lat0},{lon1} {lat1},{lon0} {lat1},{lon0} {lat0}))"],
            crs="EPSG:4326")
        # clip rather than select, so overseas territories do not stretch the extent
        clipped = gpd.clip(g.to_crs("EPSG:4326"), box.iloc[0])
        cp = clipped.to_crs("+proj=eqearth")
        for cat, col in COL.items():
            sset = cp[cp.cls == cat]
            if len(sset):
                sset.plot(ax=axi, color=col, edgecolor="white", linewidth=0.35)
        nds = cp[cp.cls == "No need index"]
        if len(nds):
            nds.plot(ax=axi, color="#d9d9d9", edgecolor="white", linewidth=0.35, hatch="///")
        lab = cp[cp.cls == focus]
        for _, r in lab.iterrows():
            c = r.geometry.representative_point()
            dx, dy = nudge.get(r.iso3, (0.0, 0.0))
            axi.annotate(r.iso3, (c.x + dx * 90000, c.y + dy * 90000), ha="center", va="center",
                         fontsize=5.4, color="white", fontweight="bold",
                         path_effects=[pe.withStroke(linewidth=1.3, foreground="#00000088")])
        bp = box.to_crs("+proj=eqearth").total_bounds
        axi.set_xlim(bp[0], bp[2]); axi.set_ylim(bp[1], bp[3])
        axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_visible(True); sp.set_edgecolor("0.35"); sp.set_linewidth(0.9)
        axi.set_title(f"{title}: {len(lab)} of the {int((g.cls == focus).sum())} {focus} countries",
                      fontsize=7.8, pad=3)
        box.to_crs("+proj=eqearth").boundary.plot(ax=axm, color="0.25", linewidth=0.9,
                                                  linestyle="--")
    save(fig, "fig_05_v2_mismatch_lisa.pdf")
    for _, r in g[g.cls.isin(["High-High", "Low-Low"])].iterrows():
        pass
    print("    High-High:", ", ".join(sorted(g[g.cls == "High-High"].iso3)))
    print("    Low-Low:  ", ", ".join(sorted(g[g.cls == "Low-Low"].iso3)))


def fig2_markers(lg):
    """Figure 2: hollow, larger zero-research markers with a dark edge."""
    panel = pd.read_parquet(ROOT / "12_Data_Integration" / "country_crop_panel.parquet")
    tot_r = panel["n_studies_fractional"].sum(); tot_a = panel["area_ha_mean"].sum()
    cells = panel.dropna(subset=["area_ha_mean"]).copy()
    cells = cells[cells["area_ha_mean"] > 0]
    cells["rs"] = 100 * cells["n_studies_fractional"] / tot_r
    cells["as_"] = 100 * cells["area_ha_mean"] / tot_a
    cty = panel.groupby("iso3").agg(nf=("n_studies_fractional", "sum"),
                                    a=("area_ha_mean", "sum")).reset_index()
    cty = cty[cty["a"] > 0]
    cty["rs"] = 100 * cty["nf"] / tot_r; cty["as_"] = 100 * cty["a"] / tot_a

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ax, dd, ttl, labels in (
            (axes[0], cells, "Country-crop systems",
             [("CHN", "wheat"), ("USA", "maize"), ("IND", "rice"), ("BRA", "sugarcane")]),
            (axes[1], cty, "Countries", ["USA", "IND", "CHN", "BRA", "NGA", "IDN"])):
        pos = dd[dd["rs"] > 0]; zer = dd[dd["rs"] == 0]
        floor = max(pos["rs"].min() / 4, 1e-5)
        ax.scatter(pos["as_"], pos["rs"], s=12, c=OKB, alpha=0.55, lw=0,
                   label=f"research present (n = {len(pos)})")
        # hollow marker, dark edge, 1.5x area: unmistakable against filled points in print
        # zeros sit on an off-scale band: shaded strip plus hollow markers, so they cannot
        # be misread as small positive values on the logarithmic axis
        ax.axhspan(floor / 1.9, floor * 1.9, color="#EDEDED", zorder=0, lw=0)
        ax.scatter(zer["as_"], np.full(len(zer), floor), s=14, marker="v",
                   facecolors="none", edgecolors="#4D4D4D", linewidths=0.55, alpha=0.85,
                   label=f"zero research (n = {len(zer)})", zorder=3)
        lims = [dd["as_"].min() * 0.5, max(dd["as_"].max(), pos["rs"].max()) * 1.6]
        ax.plot(lims, lims, ls="--", c="0.35", lw=1.1, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(floor / 3.2, pos["rs"].max() * 3.0)
        ax.text(0.012, floor * 2.4, "zero (off scale)", transform=ax.get_yaxis_transform(),
                fontsize=6.4, color="0.35", va="bottom")
        ax.set_xlabel("Share of world harvested area, % (log scale)")
        ax.set_title(ttl, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left", fontsize=7.5)
        if ttl.startswith("Country-crop"):
            OFF = {("CHN", "wheat"): (-46, 4), ("USA", "maize"): (4, 4),
                   ("IND", "rice"): (4, -9), ("BRA", "sugarcane"): (4, -2)}
            for iso, crop in labels:
                r = dd[(dd.iso3 == iso) & (dd.crop_standard_name == crop)]
                if len(r) and r.rs.iloc[0] > 0:
                    ax.annotate(f"{iso}-{crop}", (r.as_.iloc[0], r.rs.iloc[0]),
                                xytext=OFF.get((iso, crop), (3, 2)),
                                textcoords="offset points", fontsize=6.5)
            oil = dd[(dd.iso3 == "IDN") & (dd.crop_standard_name == "oilpalm")]
            if len(oil):
                ax.annotate("IDN-oilpalm", (oil.as_.iloc[0], floor), xytext=(3, 4),
                            textcoords="offset points", fontsize=6.5, color="#4D4D4D")
        else:
            for iso in labels:
                r = dd[dd.iso3 == iso]
                if len(r) and r.rs.iloc[0] > 0:
                    ax.annotate(iso, (r.as_.iloc[0], r.rs.iloc[0]), xytext=(3, 2),
                                textcoords="offset points", fontsize=6.5)
    axes[0].set_ylabel("Share of fractional research, % (log scale)")
    fig.tight_layout()
    save(fig, "fig_03_v2_scale_alignment.pdf")


def figA5_axis(lg):
    """Figure A5: y-axis extended so a four-point Gini move is not visually inflated."""
    tg = pd.read_csv(ROOT / "outputs" / "revision" / "temporal_gini.csv")
    full = tg[~tg.period.str.contains("partial")]
    part = tg[tg.period.str.contains("partial")]
    REF = 0.848
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(REF, color=OKG, lw=1.1, ls="--",
               label=f"Full-period Gini ({REF:.3f})")
    ax.plot(full.period, full.gini_all_countries, "o-", color=OKB, lw=1.8, ms=6,
            label="Complete periods")
    if len(part):
        ax.plot(part.period, part.gini_all_countries, "o", color=OKV, ms=7,
                label="Partial year (2026)")
        ax.plot([full.period.iloc[-1], part.period.iloc[0]],
                [full.gini_all_countries.iloc[-1], part.gini_all_countries.iloc[0]],
                ":", color=OKV, lw=1.4)
    for _, r in tg.iterrows():
        ax.annotate(f"{r.gini_all_countries:.3f}", (r.period, r.gini_all_countries),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_ylim(0.50, 1.00)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("Gini coefficient of country research counts")
    ax.set_xlabel("Publication period")
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, 0.03, "Axis spans 0.50 to 1.00; concentration is high in every period",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.35")
    fig.tight_layout()
    save(fig, "fig_A4_temporal_concentration.pdf")
    print("    y-axis now 0.50-1.00; reference line at the full-period Gini 0.848")


def main():
    lg = RunLogger("rf_02_figures")
    print("FIGURE REVISIONS")
    fig4_insets(lg)
    fig2_markers(lg)
    figA5_axis(lg)
    for f in ["fig_05_v2_mismatch_lisa.pdf", "fig_03_v2_scale_alignment.pdf",
              "fig_A4_temporal_concentration.pdf"]:
        lg.add_output(OUT / f)
    lg.finish()


if __name__ == "__main__":
    main()
