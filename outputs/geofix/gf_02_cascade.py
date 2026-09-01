"""Regenerate every reported quantity on the corrected study-location coding."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")
from libpysal.weights import KNN
import esda

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402
PERM, SEED = 9999, 20260730
X = ["log_area", "rd", "need", "log_gdp_pc", "tertiary", "internet", "log_population"]
LAB = {0: "Not significant", 1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2*np.arange(1, n+1)-n-1).dot(x)/(n*x.sum())) if n and x.sum() else np.nan
def fdr(p, a=0.05):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); out = np.zeros(m, bool)
    ok = p[o] <= a*(np.arange(1, m+1)/m)
    if ok.any(): out[o[:np.max(np.where(ok)[0])+1]] = True
    return out
def knn_w(gdf, k):
    return KNN.from_dataframe(gdf.set_geometry(gpd.points_from_xy(gdf.centroid_lon, gdf.centroid_lat),
                              crs="EPSG:4326").to_crs("+proj=eqearth"), k=k)
def logit(df, xs, y="studied"):
    M = sm.add_constant(df[xs].astype(float), has_constant="add")
    return sm.Logit(df[y].astype(float), M).fit(disp=0, cov_type="cluster",
                                                cov_kwds={"groups": df.iso3.to_numpy()})
def ppml(df, xs, offset=True, y="n_studies_fractional"):
    d = df[df.area_ha_mean.fillna(0) > 0].copy()
    M = sm.add_constant(d[xs].astype(float), has_constant="add")
    off = np.log(d.area_ha_mean.astype(float)) if offset else None
    return sm.GLM(d[y].astype(float), M, family=sm.families.Poisson(), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": d.iso3.to_numpy()}), d


def main():
    lg = RunLogger("gf_02_cascade")
    pan = pd.read_parquet(HERE / "country_crop_panel_corrected.parquet")
    scc = pd.read_parquet(HERE / "study_country_crop_corrected.parquet")
    lay = gpd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    corr = pd.read_parquet(ROOT / "outputs" / "revision" / "need_index_corrected_floor.parquet")
    lg.add_input(HERE / "country_crop_panel_corrected.parquet")
    out = {}

    # ------------------------------------------------ 5.1 descriptive
    cty = pan.groupby("iso3", as_index=False).n_studies_fractional.sum()
    tot = cty.n_studies_fractional.sum()
    top = cty.sort_values("n_studies_fractional", ascending=False)
    out["countries_with_research"] = int((cty.n_studies_fractional > 0).sum())
    out["countries_zero"] = int((cty.n_studies_fractional == 0).sum())
    out["cells_with_research"] = int(pan.has_any_study.sum())
    out["gini_panel199"] = gini(cty.n_studies_fractional)
    out["gini_cond"] = gini(cty[cty.n_studies_fractional > 0].n_studies_fractional)
    out["top1_share"] = 100*top.n_studies_fractional.iloc[0]/tot
    out["top3_share"] = 100*top.n_studies_fractional.head(3).sum()/tot
    out["top20_share"] = 100*top.n_studies_fractional.head(20).sum()/tot
    out["top3_names"] = list(top.iso3.head(3)); out["top3_vals"] = list(top.n_studies_fractional.head(3).round(1))
    crop = pan.groupby("crop_standard_name", as_index=False).n_studies_fractional.sum()
    ct = crop.sort_values("n_studies_fractional", ascending=False)
    out["top3crop_share"] = 100*ct.n_studies_fractional.head(3).sum()/crop.n_studies_fractional.sum()
    out["top3crop"] = list(zip(ct.crop_standard_name.head(3), ct.n_studies_fractional.head(3).round(1)))
    out["gini_crop"] = gini(crop.n_studies_fractional)
    inc = pan.groupby("wb_income_group", as_index=False).n_studies_fractional.sum()
    inc["share"] = 100*inc.n_studies_fractional/tot
    print("5.1 DESCRIPTIVE")
    for k in ["countries_with_research","countries_zero","cells_with_research","gini_panel199",
              "gini_cond","top1_share","top3_share","top20_share","top3_names","top3_vals",
              "top3crop_share","top3crop","gini_crop"]:
        v = out[k]; print(f"   {k:24s} {round(v,4) if isinstance(v,float) else v}")
    print("   income shares:", {r.wb_income_group: round(r.share,1) for _,r in inc.iterrows()})

    # ------------------------------------------------ 5.2 correlations
    pos = pan[pan.area_ha_mean > 0]
    out["sp_cell_area"] = pos.n_studies_fractional.corr(pos.area_ha_mean, method="spearman")
    out["sp_cell_prod"] = pos.n_studies_fractional.corr(pos.production_t_mean, method="spearman")
    cc = pan.groupby("iso3", as_index=False).agg(r=("n_studies_fractional","sum"),
                                                 a=("area_ha_mean","sum"), p=("production_t_mean","sum"))
    cc = cc[cc.a > 0]
    out["sp_ctry_area"] = cc.r.corr(cc.a, method="spearman")
    out["sp_ctry_prod"] = cc.r.corr(cc.p, method="spearman")
    print(f"\n5.2 Spearman cell area {out['sp_cell_area']:.3f} prod {out['sp_cell_prod']:.3f} | "
          f"country area {out['sp_ctry_area']:.3f} prod {out['sp_ctry_prod']:.3f}")
    zero = pos[pos.n_studies_fractional == 0]
    out["zero_cells_posarea"] = len(zero); out["pos_area_cells"] = len(pos)
    out["zero_area_share"] = 100*zero.area_ha_mean.sum()/pos.area_ha_mean.sum()
    print(f"    zero-research cells with positive area {len(zero)} of {len(pos)}, "
          f"holding {out['zero_area_share']:.1f}% of panel area")

    # ------------------------------------------------ 5.3 / 5.5 mismatch and spatial
    g = lay.merge(cty, on="iso3", how="left").merge(
        corr[["iso3","need9_floor9"]], on="iso3", how="left")
    g["n_studies_fractional"] = g.n_studies_fractional.fillna(0.0)
    g["research_pct"] = g.n_studies_fractional.rank(pct=True)*100
    g["gap"] = g.research_pct - pd.to_numeric(g.need9_floor9, errors="coerce")*100
    o = g[g.gap.notna()].reset_index(drop=True)
    mn, mr = (o.research_pct-o.gap).median(), o.research_pct.median()
    o["need_pct"] = o.research_pct - o.gap
    out["n_gap"] = len(o)
    out["quadrant"] = int(((o.need_pct >= mn) & (o.research_pct <= mr)).sum())
    out["ties"] = int((o.n_studies_fractional == 0).sum())
    print(f"\n5.3 mismatch n={out['n_gap']} quadrant={out['quadrant']} ties={out['ties']}")
    for k in (4, 6, 8):
        w = knn_w(o, k); w.transform = "r"
        mi = esda.Moran(o.gap.values, w, permutations=PERM)
        out[f"moran_k{k}"] = mi.I; out[f"moran_p{k}"] = mi.p_sim
        if k == 6:
            lm = esda.Moran_Local(o.gap.values, w, permutations=PERM, seed=SEED)
            cat = np.where(fdr(lm.p_sim), lm.q, 0)
            o["lisa_cat_gap9"] = [LAB[c] for c in cat]
            out["HH"], out["LL"] = int((cat==1).sum()), int((cat==3).sum())
            out["HL"], out["LH"] = int((cat==4).sum()), int((cat==2).sum())
            go = esda.G_Local(o.gap.values, w, permutations=PERM, seed=SEED)
            rg = fdr(go.p_sim)
            out["hot"], out["cold"] = int((rg&(go.Zs>0)).sum()), int((rg&(go.Zs<0)).sum())
        print(f"    Moran k={k}: {mi.I:.4f} (p={mi.p_sim:.4f})")
    print(f"    LISA {out['HH']}/{out['LL']}/{out['HL']}/{out['LH']}  "
          f"Getis hot={out['hot']} cold={out['cold']}")
    print(f"    HH: {', '.join(sorted(o[o.lisa_cat_gap9=='High-High'].iso3))}")
    print(f"    LL: {', '.join(sorted(o[o.lisa_cat_gap9=='Low-Low'].iso3))}")
    o[["iso3","gap","research_pct","need_pct","lisa_cat_gap9"]].to_csv(HERE/"gap_corrected_lisa.csv", index=False)

    # ------------------------------------------------ 5.4 regressions
    e0 = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    keep = [c for c in e0.columns if c not in ("n_studies_fractional","n_studies_full","studied","has_any_study","need")]
    e = e0[keep].merge(pan[["iso3","crop_standard_name","n_studies_fractional"]],
                       on=["iso3","crop_standard_name"], how="left").merge(
        corr[["iso3","need9_floor9"]].rename(columns={"need9_floor9":"need"}), on="iso3", how="left")
    e["n_studies_fractional"] = e.n_studies_fractional.fillna(0.0)
    e["studied"] = (e.n_studies_fractional > 0).astype(float)
    e = e.dropna(subset=["need"])
    print(f"\n5.4 estimation {len(e)} cells / {e.iso3.nunique()} countries / {int(e.studied.sum())} events")
    m = logit(e, X); ci = m.conf_int()
    p1, _ = ppml(e, [x for x in X if x!="log_area"]); c1 = p1.conf_int()
    p2, _ = ppml(e, X, offset=False); c2 = p2.conf_int()
    ba, sa = p2.params["log_area"], p2.bse["log_area"]
    wald = ((ba-1)/sa)**2; pw = 1-stats.chi2.cdf(wald,1)
    cd = pd.get_dummies(e.crop_standard_name, prefix="crop", drop_first=True)
    e2 = pd.concat([e, cd], axis=1)
    mfe = logit(e2, X+list(cd.columns))
    pfe, _ = ppml(e2, [x for x in X if x!="log_area"]+list(cd.columns)); cfe = pfe.conf_int()
    rows=[]
    for mod,r_,cx,xs in [("participation",m,ci,X),("intensity_offset",p1,c1,[x for x in X if x!="log_area"]),
                         ("intensity_free_area",p2,c2,X),("participation_cropFE",mfe,mfe.conf_int(),X),
                         ("intensity_cropFE",pfe,cfe,[x for x in X if x!="log_area"])]:
        for t in xs:
            rows.append(dict(model=mod,term=t,estimate=r_.params[t],se=r_.bse[t],
                             ci_low=cx.loc[t,0],ci_high=cx.loc[t,1],n=int(r_.nobs)))
    pd.DataFrame(rows).to_csv(HERE/"model_estimates_geofix.csv", index=False)
    out["pseudoR2"]=1-m.llf/m.llnull; out["events"]=int(e.studied.sum())
    print(f"    participation log_area {m.params['log_area']:.3f} [{ci.loc['log_area',0]:.3f},{ci.loc['log_area',1]:.3f}]"
          f"  need {m.params['need']:.3f} [{ci.loc['need',0]:.3f},{ci.loc['need',1]:.3f}]  rd {m.params['rd']:.3f}")
    print(f"    intensity need {p1.params['need']:.3f} [{c1.loc['need',0]:.3f},{c1.loc['need',1]:.3f}]  rd {p1.params['rd']:.3f}")
    print(f"    elasticity {ba:.3f} [{c2.loc['log_area',0]:.3f},{c2.loc['log_area',1]:.3f}] Wald {wald:.2f} p={pw:.4f}"
          f"  free need {p2.params['need']:.3f} [{c2.loc['need',0]:.3f},{c2.loc['need',1]:.3f}]")
    print(f"    cropFE participation log_area {mfe.params['log_area']:.3f}  intensity need {pfe.params['need']:.3f}")
    me = m.get_margeff(at="overall", method="dydx")
    ame = pd.DataFrame({"term":[t for t in m.params.index if t!="const"],"ame":me.margeff,"se":me.margeff_se})
    ame["ci_low"]=ame.ame-1.959964*ame.se; ame["ci_high"]=ame.ame+1.959964*ame.se
    ame["p_value"]=2*(1-stats.norm.cdf(np.abs(ame.ame/ame.se)))
    iq=[]
    for t in X:
        q1,q3=e[t].quantile(.25),e[t].quantile(.75)
        a1,a3=e.copy(),e.copy(); a1[t]=q1; a3[t]=q3
        pr1=m.predict(sm.add_constant(a1[X].astype(float),has_constant="add")).mean()
        pr3=m.predict(sm.add_constant(a3[X].astype(float),has_constant="add")).mean()
        iq.append(dict(term=t,p25=q1,p75=q3,iqr_dprob=pr3-pr1))
    ame=ame.merge(pd.DataFrame(iq),on="term"); ame.to_csv(HERE/"ame_geofix.csv",index=False)
    print("    AME:", {r.term:(round(r.ame,4),round(100*r.iqr_dprob,1)) for _,r in ame.iterrows()})
    out["wald"]=wald; out["wald_p"]=pw
    pd.DataFrame([out]).to_json(HERE/"headline_numbers.json", orient="records", indent=1)
    for f in ["gap_corrected_lisa.csv","model_estimates_geofix.csv","ame_geofix.csv","headline_numbers.json"]:
        lg.add_output(HERE/f)
    lg.finish()


if __name__ == "__main__":
    main()
