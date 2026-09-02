"""Regenerate every figure affected by the study-location coding correction.

Figure 1 map, Figure 3 scale alignment, Figure 4 need vs research, Figure 5 LISA with insets,
Figure 6 predicted probability, Figure A1 crop attention. Figure 2 (sample flow) is rebuilt by
gf_04 because its counts are text. Styling matches the existing figures exactly.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import patheffects as pe
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402
# The submission folder lives outside this replication repository. When it is absent
# (the normal case for anyone running this repo) figures are written here only.
_SUB = ROOT / "Final Manuscript" / "Manuscript 1" / "09_Wiley_Submission"
SUB = _SUB if _SUB.is_dir() else None
PNG = HERE / "png"; PNG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 300})
OKB, OKV, OKG = "#0072B2", "#D55E00", "#009E73"
SEQ = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]
COL_ZERO = "#F0F0F0"
LISA_COL = {"High-High": "#b2182b", "Low-Low": "#2166ac", "High-Low": "#f4a582",
            "Low-High": "#92c5de", "Not significant": "#f0f0f0"}
X = ["log_area", "rd", "need", "log_gdp_pc", "tertiary", "internet", "log_population"]
BREAKS = [0, 1, 5, 20, 100, np.inf]


def save(fig, name):
    for d in [d for d in (HERE, SUB) if d is not None]:
        fig.savefig(d / name, bbox_inches="tight", dpi=300)
    fig.savefig(PNG / name.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig); print(f"  wrote {name}")


def main():
    lg = RunLogger("gf_03_figures")
    pan = pd.read_parquet(HERE / "country_crop_panel_corrected.parquet")
    lisa = pd.read_csv(HERE / "gap_corrected_lisa.csv")
    lay = gpd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    corr = pd.read_parquet(ROOT / "outputs" / "revision" / "need_index_corrected_floor.parquet")
    lg.add_input(HERE / "country_crop_panel_corrected.parquet")
    cty = pan.groupby("iso3", as_index=False).n_studies_fractional.sum()

    # ---------------- Figure 1: choropleth ----------------
    g = lay.merge(cty, on="iso3", how="left")
    g["n"] = g.n_studies_fractional.fillna(0.0)
    def cls(v):
        if v == 0: return 0
        for i,(a,b) in enumerate(zip(BREAKS[:-1], BREAKS[1:]), start=1):
            if a < v <= b: return i
        return len(BREAKS)-1
    g["cls"] = g.n.map(cls)
    gp = g.to_crs("+proj=eqearth")
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    labels = ["0 (no resolved study)", "0 < n ≤ 1", "1 < n ≤ 5", "5 < n ≤ 20",
              "20 < n ≤ 100", "n > 100"]
    cols = [COL_ZERO] + SEQ
    for i, c in enumerate(cols):
        s = gp[gp.cls == i]
        if len(s): s.plot(ax=ax, color=c, edgecolor="white", linewidth=0.25)
    ax.set_axis_off()
    ax.set_title("Fractional count of eligible AI crop-yield studies by country, 2000–2026",
                 fontsize=9.5)
    hs = [Patch(facecolor=cols[i], edgecolor="0.6", linewidth=.3,
                label=f"{labels[i]}  (n = {int((g.cls==i).sum())})") for i in range(len(cols))]
    hs.append(Patch(facecolor="#d9d9d9", hatch="///", label="no data: external data unavailable  (n = 0)"))
    ax.legend(handles=hs, loc="lower center", frameon=False, ncol=3, fontsize=7.2,
              bbox_to_anchor=(0.5, -0.14), title="Fractional eligible studies per country",
              title_fontsize=7.6)
    save(fig, "fig_01_research_intensity_map.pdf")

    # ---------------- Figure 3: scale alignment ----------------
    tot_r, tot_a = pan.n_studies_fractional.sum(), pan.area_ha_mean.sum()
    cells = pan.dropna(subset=["area_ha_mean"]); cells = cells[cells.area_ha_mean > 0].copy()
    cells["rs"] = 100*cells.n_studies_fractional/tot_r; cells["as_"] = 100*cells.area_ha_mean/tot_a
    ct = pan.groupby("iso3").agg(nf=("n_studies_fractional","sum"), a=("area_ha_mean","sum")).reset_index()
    ct = ct[ct.a > 0].copy(); ct["rs"] = 100*ct.nf/tot_r; ct["as_"] = 100*ct.a/tot_a
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ax, dd, ttl, labs in ((axes[0], cells, "Country-crop systems",
            [("CHN","wheat"),("USA","maize"),("IND","rice"),("BRA","sugarcane")]),
            (axes[1], ct, "Countries", ["USA","IND","CHN","BRA","NGA","IDN"])):
        pos = dd[dd.rs > 0]; zer = dd[dd.rs == 0]
        floor = max(pos.rs.min()/4, 1e-5)
        ax.axhspan(floor/1.9, floor*1.9, color="#EDEDED", zorder=0, lw=0)
        ax.scatter(pos.as_, pos.rs, s=12, c=OKB, alpha=.55, lw=0,
                   label=f"research present (n = {len(pos)})")
        ax.scatter(zer.as_, np.full(len(zer), floor), s=14, marker="v", facecolors="none",
                   edgecolors="#4D4D4D", linewidths=.55, alpha=.85, zorder=3,
                   label=f"zero research (n = {len(zer)})")
        lims = [dd.as_.min()*0.5, max(dd.as_.max(), pos.rs.max())*1.6]
        ax.plot(lims, lims, ls="--", c="0.35", lw=1.1, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(floor/3.2, pos.rs.max()*3.0)
        ax.text(.012, floor*2.4, "zero (off scale)", transform=ax.get_yaxis_transform(),
                fontsize=6.4, color="0.35", va="bottom")
        ax.set_xlabel("Share of world harvested area, % (log scale)")
        ax.set_title(ttl, fontsize=9); ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left", fontsize=7.5)
        if ttl.startswith("Country-crop"):
            OFF={("CHN","wheat"):(-46,4),("USA","maize"):(4,4),("IND","rice"):(4,-9),("BRA","sugarcane"):(4,-2)}
            for iso,crop in labs:
                r = dd[(dd.iso3==iso)&(dd.crop_standard_name==crop)]
                if len(r) and r.rs.iloc[0] > 0:
                    ax.annotate(f"{iso}-{crop}", (r.as_.iloc[0], r.rs.iloc[0]),
                                xytext=OFF.get((iso,crop),(3,2)), textcoords="offset points", fontsize=6.5)
            oil = dd[(dd.iso3=="IDN")&(dd.crop_standard_name=="oilpalm")]
            if len(oil): ax.annotate("IDN-oilpalm", (oil.as_.iloc[0], floor), xytext=(3,4),
                                     textcoords="offset points", fontsize=6.5, color="#4D4D4D")
        else:
            for iso in labs:
                r = dd[dd.iso3==iso]
                if len(r) and r.rs.iloc[0] > 0:
                    ax.annotate(iso, (r.as_.iloc[0], r.rs.iloc[0]), xytext=(3,2),
                                textcoords="offset points", fontsize=6.5)
    axes[0].set_ylabel("Share of fractional research, % (log scale)")
    fig.tight_layout(); save(fig, "fig_03_v2_scale_alignment.pdf")

    # ---------------- Figure 4: need vs research ----------------
    d = lisa.copy()
    mn, mr = d.need_pct.median(), d.research_pct.median()
    quad = (d.need_pct >= mn) & (d.research_pct <= mr)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.axvspan(mn, 100, ymin=0, ymax=(mr/100), color=OKV, alpha=.10, lw=0)
    ax.scatter(d.loc[~quad,"need_pct"], d.loc[~quad,"research_pct"], s=16, c=OKB, alpha=.65, lw=0)
    ax.scatter(d.loc[quad,"need_pct"], d.loc[quad,"research_pct"], s=18, c=OKV, alpha=.85, lw=0)
    ax.axvline(mn, color="0.55", lw=.8, ls="--"); ax.axhline(mr, color="0.55", lw=.8, ls="--")
    for iso in ["USA","CHN","IND","BRA","NGA","ETH","COD","SDN","NER","TCD"]:
        r = d[d.iso3==iso]
        if len(r): ax.annotate(iso,(r.need_pct.iloc[0],r.research_pct.iloc[0]),xytext=(3,3),
                               textcoords="offset points", fontsize=7)
    ax.set_xlabel("Research-need percentile (nine-component index)")
    ax.set_ylabel("Research-output percentile")
    ax.text(.985,.02,f"high need, low research: n = {int(quad.sum())}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=OKV)
    ax.set_xlim(-2,102); ax.set_ylim(-2,102); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); save(fig, "fig_04_v2_need_vs_research.pdf")

    # ---------------- Figure 5: LISA with insets ----------------
    gg = lay.merge(lisa[["iso3","lisa_cat_gap9"]], on="iso3", how="left")
    gg["cls"] = gg.lisa_cat_gap9.where(gg.lisa_cat_gap9.notna(), "No need index")
    gpp = gg.to_crs("+proj=eqearth")
    fig = plt.figure(figsize=(9.2, 6.9))
    axm = fig.add_axes([0.0, .40, 1.0, .60])
    for cat,c in LISA_COL.items():
        s = gpp[gpp.cls==cat]
        if len(s): s.plot(ax=axm, color=c, edgecolor="white", linewidth=.25)
    nd = gpp[gpp.cls=="No need index"]
    if len(nd): nd.plot(ax=axm, color="#d9d9d9", edgecolor="white", linewidth=.25, hatch="///")
    axm.set_axis_off()
    hs=[Patch(facecolor=c,label=f"{k}  (n = {int((gg.cls==k).sum())})") for k,c in LISA_COL.items()]
    hs.append(Patch(facecolor="#d9d9d9",hatch="///",label=f"No need index  (n = {int((gg.cls=='No need index').sum())})"))
    axm.legend(handles=hs, loc="lower left", frameon=False, ncol=2, fontsize=7.2,
               bbox_to_anchor=(.005,.02), handlelength=1.4, columnspacing=1.1, labelspacing=.35)
    yb=axm.get_ylim(); axm.set_ylim(yb[0]*.62, yb[1]*1.02)
    REG=[("(a) Western and northern Europe",(-11,35,32,71),[.055,.02,.395,.335],"High-High",
          {"LUX":(.6,-1.4),"BEL":(-1.1,.4),"NLD":(.5,1.0),"CHE":(.3,-.8),"SVN":(1.2,-.6),
           "SVK":(.8,.5),"EST":(.6,.3),"DNK":(-.6,.9),"CZE":(.2,.6)}),
         ("(b) Central and eastern Africa",(8,-12,48,23),[.552,.02,.395,.335],"Low-Low",
          {"COG":(-2.2,1.4),"GAB":(-2.4,-.6),"COD":(1.0,-1.0),"DJI":(2.6,.8)})]
    for title,(lo0,la0,lo1,la1),rect,focus,nudge in REG:
        axi=fig.add_axes(rect)
        box=gpd.GeoSeries.from_wkt([f"POLYGON(({lo0} {la0},{lo1} {la0},{lo1} {la1},{lo0} {la1},{lo0} {la0}))"],crs="EPSG:4326")
        cp=gpd.clip(gg.to_crs("EPSG:4326"), box.iloc[0]).to_crs("+proj=eqearth")
        for cat,c in LISA_COL.items():
            s=cp[cp.cls==cat]
            if len(s): s.plot(ax=axi,color=c,edgecolor="white",linewidth=.35)
        nds=cp[cp.cls=="No need index"]
        if len(nds): nds.plot(ax=axi,color="#d9d9d9",edgecolor="white",linewidth=.35,hatch="///")
        lab=cp[cp.cls==focus]
        for _,r in lab.iterrows():
            c=r.geometry.representative_point(); dx,dy=nudge.get(r.iso3,(0.,0.))
            axi.annotate(r.iso3,(c.x+dx*90000,c.y+dy*90000),ha="center",va="center",fontsize=5.4,
                         color="white",fontweight="bold",
                         path_effects=[pe.withStroke(linewidth=1.3,foreground="#00000088")])
        bp=box.to_crs("+proj=eqearth").total_bounds
        axi.set_xlim(bp[0],bp[2]); axi.set_ylim(bp[1],bp[3]); axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values(): sp.set_visible(True); sp.set_edgecolor("0.35"); sp.set_linewidth(.9)
        tot=int((gg.cls==focus).sum())
        axi.set_title(f"{title}: {len(lab)} of the {tot} {focus} countries", fontsize=7.8, pad=3)
        box.to_crs("+proj=eqearth").boundary.plot(ax=axm,color="0.25",linewidth=.9,linestyle="--")
        print(f"    inset {focus}: {len(lab)} of {tot}")
    save(fig, "fig_05_v2_mismatch_lisa.pdf")

    # ---------------- Figure 6: predicted probability ----------------
    e0=pd.read_parquet(ROOT/"16_Econometrics"/"estimation_sample.parquet")
    keep=[c for c in e0.columns if c not in ("n_studies_fractional","n_studies_full","studied","has_any_study","need")]
    e=e0[keep].merge(pan[["iso3","crop_standard_name","n_studies_fractional"]],
                     on=["iso3","crop_standard_name"],how="left").merge(
        corr[["iso3","need9_floor9"]].rename(columns={"need9_floor9":"need"}),on="iso3",how="left")
    e["n_studies_fractional"]=e.n_studies_fractional.fillna(0.0)
    e["studied"]=(e.n_studies_fractional>0).astype(float); e=e.dropna(subset=["need"])
    M=sm.add_constant(e[X].astype(float),has_constant="add")
    m=sm.Logit(e.studied.astype(float),M).fit(disp=0,cov_type="cluster",cov_kwds={"groups":e.iso3.to_numpy()})
    grid=np.linspace(e.log_area.quantile(.01),e.log_area.quantile(.99),200)
    base={c:e[c].mean() for c in X}
    Xg=pd.DataFrame({c:np.repeat(base[c],len(grid)) for c in X}); Xg["log_area"]=grid
    Xg=sm.add_constant(Xg,has_constant="add")[M.columns]
    pr=m.predict(Xg); xb=Xg.values@m.params.values
    se=np.sqrt(np.einsum("ij,jk,ik->i",Xg.values,m.cov_params().values,Xg.values))
    lo=1/(1+np.exp(-(xb-1.96*se))); hi=1/(1+np.exp(-(xb+1.96*se)))
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    ax.fill_between(grid,lo,hi,color=OKB,alpha=.18,lw=0); ax.plot(grid,pr,color=OKB,lw=1.8)
    q1,q3=e.log_area.quantile(.25),e.log_area.quantile(.75)
    ax.axvspan(q1,q3,color="0.85",alpha=.35,lw=0,zorder=0)
    ax.set_xlabel("Log harvested area (hectares)")
    ax.set_ylabel("Predicted probability a country-crop system is studied"); ax.set_ylim(0,1)
    ax2=ax.twinx(); ax2.hist(e.log_area,bins=45,color="0.55",alpha=.35,lw=0); ax2.set_yticks([])
    ax.set_zorder(ax2.get_zorder()+1); ax.patch.set_visible(False)
    ax.text(.02,.96,"Shaded band: interquartile range of harvested area\nOther covariates held at their sample means",
            transform=ax.transAxes,va="top",fontsize=7.5,color="0.25")
    save(fig,"fig_A3_predicted_probability.pdf")

    # ---------------- Figure A1: crop attention vs area ----------------
    cr=pan.groupby("crop_standard_name",as_index=False).agg(att=("n_studies_fractional","sum"),
                                                            ar=("area_ha_mean","sum"))
    cr["att"]=100*cr.att/cr.att.sum(); cr["ash"]=100*cr.ar/cr.ar.sum()
    cr=cr.sort_values("att",ascending=False).head(15).sort_values("att")
    y=np.arange(len(cr)); h=0.38
    fig,ax=plt.subplots(figsize=(6.6,5.2))
    ax.barh(y+h/2,cr.att,height=h,color=OKB,label="Share of research attention")
    ax.barh(y-h/2,cr.ash,height=h,color=OKG,label="Share of world harvested area")
    ax.set_yticks(y); ax.set_yticklabels([c.replace("_"," ") for c in cr.crop_standard_name],fontsize=8)
    ax.set_xlabel("%"); ax.legend(frameon=False,fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); save(fig,"fig_02_v2_crop_attention_area.pdf")

    for f in ["fig_01_research_intensity_map.pdf","fig_03_v2_scale_alignment.pdf",
              "fig_04_v2_need_vs_research.pdf","fig_05_v2_mismatch_lisa.pdf",
              "fig_A3_predicted_probability.pdf","fig_02_v2_crop_attention_area.pdf"]:
        lg.add_output(HERE/f)
    lg.finish()


if __name__ == "__main__":
    main()
