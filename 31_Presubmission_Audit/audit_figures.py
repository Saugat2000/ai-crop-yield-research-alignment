"""Rebuild the five figures affected by the pre-submission audit.

fig_04_v2: need9 percentile vs research percentile (56-country quadrant)
fig_05_v2: observed-sample LISA map of the scale-excluded mismatch
fig_03_v2: research share vs harvested-area share, cells and countries (two panels)
fig_02_v2: crop attention share vs harvested-area share (appendix)
fig_06_v2: coefficient plot for the revised Table 5
No internal analysis shorthand appears in any figure; captions live in the manuscript.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

OUT = HERE.parent / "Final Manuscript" / "Manuscript 1" / "09_Wiley_Submission"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 200})
OKB, OKV, OKG = "#0072B2", "#D55E00", "#009E73"


def main() -> int:
    lg = RunLogger("audit_05_figures")
    panel = pd.read_parquet(P["integration"] / "country_crop_panel.parquet")
    n9 = pd.read_parquet(HERE / "need_index_scale_excluded.parquet")
    lisa = pd.read_csv(HERE / "gap9_lisa_clusters.csv")
    lay = gpd.read_parquet(P["weights"] / "country_analytical_layer.parquet")

    # ---------------- fig_04_v2: need vs research percentiles -----------------------
    d = n9[n9["need9_rank_pct"].notna()].copy()
    d["research_pct"] = d["n_studies_fractional"].rank(pct=True) * 100
    d["need_pct"] = d["need9_rank_pct"] * 100
    mn, mr = d["need_pct"].median(), d["research_pct"].median()
    quad = (d["need_pct"] >= mn) & (d["research_pct"] <= mr)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.axvspan(mn, 100, ymin=0, ymax=(mr / 100), color=OKV, alpha=0.10, lw=0)
    ax.scatter(d.loc[~quad, "need_pct"], d.loc[~quad, "research_pct"], s=16, c=OKB, alpha=0.65, lw=0)
    ax.scatter(d.loc[quad, "need_pct"], d.loc[quad, "research_pct"], s=18, c=OKV, alpha=0.85, lw=0)
    ax.axvline(mn, color="0.55", lw=0.8, ls="--"); ax.axhline(mr, color="0.55", lw=0.8, ls="--")
    for iso in ["USA", "CHN", "IND", "BRA", "NGA", "ETH", "COD", "SDN", "NER", "TCD"]:
        r = d[d.iso3 == iso]
        if len(r):
            ax.annotate(iso, (r.need_pct.iloc[0], r.research_pct.iloc[0]),
                        xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Research-need percentile (nine-component index)")
    ax.set_ylabel("Research-output percentile")
    ax.text(0.985, 0.02, f"high need, low research: n = {int(quad.sum())}",
            transform=ax.transAxes, ha="right", fontsize=8, color=OKV)
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_04_v2_need_vs_research.pdf"); plt.close(fig)
    lg.count("fig04_quadrant_n", int(quad.sum()))

    # ---------------- fig_05_v2: observed-sample LISA map ---------------------------
    g = lay.merge(lisa, on="iso3", how="left")
    COL = {"High-High": "#b2182b", "Low-Low": "#2166ac", "High-Low": "#f4a582",
           "Low-High": "#92c5de", "Not significant": "#f0f0f0"}
    g["cls"] = g["lisa_cat_gap9"].where(g["lisa_cat_gap9"].notna(), "No need index")
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    gp = g.to_crs("+proj=eqearth")
    for cat, col in COL.items():
        sub = gp[gp.cls == cat]
        if len(sub): sub.plot(ax=ax, color=col, edgecolor="white", linewidth=0.25)
    nd = gp[gp.cls == "No need index"]
    if len(nd): nd.plot(ax=ax, color="#d9d9d9", edgecolor="white", linewidth=0.25, hatch="///")
    ax.set_axis_off()
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=c, label=f"{k}  (n = {int((g.cls == k).sum())})") for k, c in COL.items()]
    hs.append(Patch(facecolor="#d9d9d9", hatch="///", label=f"No need index  (n = {int((g.cls=='No need index').sum())})"))
    ax.legend(handles=hs, loc="lower left", frameon=False, ncol=3, fontsize=7.5,
              bbox_to_anchor=(0.02, -0.06))
    fig.tight_layout(); fig.savefig(OUT / "fig_05_v2_mismatch_lisa.pdf"); plt.close(fig)

    # ---------------- fig_03_v2: research vs harvested-area share -------------------
    tot_r = panel["n_studies_fractional"].sum()
    tot_a = panel["area_ha_mean"].sum()
    cells = panel.dropna(subset=["area_ha_mean"]).copy()
    cells = cells[cells["area_ha_mean"] > 0]
    cells["rs"] = 100 * cells["n_studies_fractional"] / tot_r
    cells["as_"] = 100 * cells["area_ha_mean"] / tot_a
    cty = panel.groupby("iso3").agg(nf=("n_studies_fractional", "sum"),
                                    a=("area_ha_mean", "sum")).reset_index()
    cty = cty[cty["a"] > 0]
    cty["rs"] = 100 * cty["nf"] / tot_r; cty["as_"] = 100 * cty["a"] / tot_a
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=False)
    for ax, dd, ttl, labels in (
            (axes[0], cells, "Country-crop systems",
             [("CHN", "wheat"), ("USA", "maize"), ("IND", "rice"), ("BRA", "sugarcane")]),
            (axes[1], cty, "Countries", ["USA", "IND", "CHN", "BRA", "NGA", "IDN"])):
        pos = dd[dd["rs"] > 0]
        zer = dd[dd["rs"] == 0]
        floor = max(pos["rs"].min() / 4, 1e-5)
        ax.scatter(pos["as_"], pos["rs"], s=12, c=OKB, alpha=0.55, lw=0)
        ax.scatter(zer["as_"], np.full(len(zer), floor), s=9, marker="v", c=OKV, alpha=0.5,
                   lw=0, label=f"zero research (n = {len(zer)})")
        lims = [min(pos["as_"].min(), floor) * 0.6, max(dd["as_"].max(), pos["rs"].max()) * 1.6]
        ax.plot(lims, lims, ls="--", c="0.5", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
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
                ax.annotate("IDN-oilpalm", (oil.as_.iloc[0], floor),
                            xytext=(3, 2), textcoords="offset points", fontsize=6.5, color=OKV)
        else:
            for iso in labels:
                r = dd[dd.iso3 == iso]
                if len(r) and r.rs.iloc[0] > 0:
                    ax.annotate(iso, (r.as_.iloc[0], r.rs.iloc[0]),
                                xytext=(3, 2), textcoords="offset points", fontsize=6.5)
    axes[0].set_ylabel("Share of fractional research, % (log scale)")
    fig.tight_layout(); fig.savefig(OUT / "fig_03_v2_scale_alignment.pdf"); plt.close(fig)
    lg.count("fig03_cells_plotted", len(cells))

    # ---------------- fig_02_v2: crop attention vs area share -----------------------
    cr = panel.groupby("crop_standard_name").agg(nf=("n_studies_fractional", "sum"),
                                                 area=("world_area_ha", "first")).reset_index()
    cr["att"] = 100 * cr.nf / cr.nf.sum(); cr["ash"] = 100 * cr.area / cr.area.sum()
    cr = cr.sort_values("att", ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ypos = np.arange(len(cr)); h = 0.38
    ax.barh(ypos + h / 2, cr["att"], height=h, color=OKB, label="Share of research attention")
    ax.barh(ypos - h / 2, cr["ash"], height=h, color=OKG, label="Share of world harvested area")
    ax.set_yticks(ypos); ax.set_yticklabels([c.replace("_", " ") for c in cr.crop_standard_name])
    ax.set_xlabel("%"); ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_02_v2_crop_attention_area.pdf"); plt.close(fig)

    # ---------------- fig_06_v2: coefficient plot for revised Table 5 ---------------
    t5 = pd.read_csv(HERE / "table5_need9_primary.csv")
    NAME = {"log_area": "log harvested area", "log_gdp_pc": "log GDP per capita, PPP",
            "rd": "R&D expenditure, % of GDP", "tertiary": "tertiary enrolment (share)",
            "internet": "internet users (share)", "need9": "research-need index (0-1)",
            "log_population": "log population"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    yp = np.arange(len(t5))[::-1]
    for ax, pre, ttl in ((axes[0], "logit", "Participation (logit)"),
                         (axes[1], "ppml", "Intensity (PPML, area exposure)")):
        for i, (_, r) in enumerate(t5.iterrows()):
            b, lo, hi = r[f"{pre}_b"], r[f"{pre}_lo"], r[f"{pre}_hi"]
            if pd.isna(b):
                ax.text(0, yp[i], "offset", va="center", ha="center", fontsize=7, color="0.4")
                continue
            sig = (lo > 0) or (hi < 0)
            c = OKV if sig else OKB
            ax.plot([lo, hi], [yp[i], yp[i]], color=c, lw=1.4)
            ax.plot(b, yp[i], "o", color=c, ms=4)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_title(ttl, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(yp); axes[0].set_yticklabels([NAME[t] for t in t5.term])
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([0], [0], color=OKV, marker="o", ms=4, label="95% CI excludes zero"),
                            Line2D([0], [0], color=OKB, marker="o", ms=4, label="95% CI includes zero")],
                   frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_06_v2_coefficients.pdf"); plt.close(fig)

    for f in ["fig_04_v2_need_vs_research.pdf", "fig_05_v2_mismatch_lisa.pdf",
              "fig_03_v2_scale_alignment.pdf", "fig_02_v2_crop_attention_area.pdf",
              "fig_06_v2_coefficients.pdf"]:
        lg.add_output(OUT / f)
        (HERE / f).write_bytes((OUT / f).read_bytes())
    print("five figures rebuilt")
    lg.finish(); return 0


if __name__ == "__main__":
    sys.exit(main())
