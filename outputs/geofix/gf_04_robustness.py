"""Rerun the robustness suite on the corrected study-location coding, so Table 8's
baselines and ranges come from the same data as the main results."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import statsmodels.api as sm
warnings.filterwarnings("ignore")
from libpysal.weights import KNN
import esda

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402
PERM, SEED = 9999, 20260730
X = ["log_area", "rd", "need", "log_gdp_pc", "tertiary", "internet", "log_population"]


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2*np.arange(1, n+1)-n-1).dot(x)/(n*x.sum())) if n and x.sum() else np.nan
def knn_w(gdf, k):
    return KNN.from_dataframe(gdf.set_geometry(gpd.points_from_xy(gdf.centroid_lon, gdf.centroid_lat),
                              crs="EPSG:4326").to_crs("+proj=eqearth"), k=k)
def logit(df, xs, y="studied"):
    M = sm.add_constant(df[xs].astype(float), has_constant="add")
    return sm.Logit(df[y].astype(float), M).fit(disp=0, cov_type="cluster",
                                                cov_kwds={"groups": df.iso3.to_numpy()})


def main():
    lg = RunLogger("gf_04_robustness")
    scc = pd.read_parquet(HERE / "study_country_crop_corrected.parquet")
    pan = pd.read_parquet(HERE / "country_crop_panel_corrected.parquet")
    sld = pd.read_parquet(ROOT / "12_Data_Integration" / "study_level_dataset.parquet")
    lay = gpd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    corr = pd.read_parquet(ROOT / "outputs" / "revision" / "need_index_corrected_floor.parquet")
    n11 = pd.read_parquet(ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet")
    lg.add_input(HERE / "study_country_crop_corrected.parquet")
    iso199 = pan[["iso3"]].drop_duplicates()
    meta = sld.set_index("openalex_id")

    # ---------------- Gini variants ----------------
    def gini_of(s, weight="fractional_weight"):
        g = s.groupby("iso3", as_index=False)[weight].sum()
        full = iso199.merge(g, on="iso3", how="left").fillna({weight: 0})
        return gini(full[weight])
    gv = {}
    gv["baseline (fractional)"] = gini_of(scc)
    full_cnt = scc.assign(one=1.0)
    gv["full counting"] = gini_of(full_cnt, "one")
    cw = scc.copy()
    cw["cw"] = cw.fractional_weight * cw.openalex_id.map(meta.cited_by_count).fillna(0)
    gv["citation weighted"] = gini_of(cw, "cw")
    jr = scc[scc.openalex_id.map(meta.type).isin(["article"])]
    gv["journal only"] = gini_of(jr)
    for drop, lab in [(["USA"], "excl. USA"), (["CHN"], "excl. China"),
                      (["USA","CHN"], "excl. USA+China"), (["USA","CHN","IND"], "excl. top three")]:
        gv[lab] = gini(iso199[~iso199.iso3.isin(drop)].merge(
            scc[~scc.iso3.isin(drop)].groupby("iso3",as_index=False).fractional_weight.sum(),
            on="iso3", how="left").fillna(0).fractional_weight)
    for lo,hi,lab in [(2000,2014,"2000-2014"),(2015,2019,"2015-2019"),(2020,2026,"2020-2026")]:
        gv[lab] = gini_of(scc[(scc.publication_year>=lo)&(scc.publication_year<=hi)])
    ML = (sld.title.fillna("")+" "+sld.abstract.fillna("")).str.contains(
        r"machine learning|deep learning|neural network|random forest|gradient boost|support vector|"
        r"XGBoost|LSTM|CNN|artificial intelligence", case=False, regex=True, na=False)
    mlids = set(sld.loc[ML,"openalex_id"])
    gv["ML-confirmed subset"] = gini_of(scc[scc.openalex_id.isin(mlids)])
    conf = scc[~scc.openalex_id.map(meta.type).isin(["conference-paper","conference-abstract","preprint"])]
    gv["excl. conference/preprint"] = gini_of(conf)
    print("GINI VARIANTS")
    for k,v in gv.items(): print(f"   {k:28s} {v:.4f}")
    gvals=[v for v in gv.values() if not np.isnan(v)]
    print(f"   -> baseline {gv['baseline (fractional)']:.3f}  range [{min(gvals):.3f}, {max(gvals):.3f}]  n={len(gvals)}")

    # ---------------- model variants ----------------
    e0 = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    keep=[c for c in e0.columns if c not in ("n_studies_fractional","n_studies_full","studied","has_any_study","need")]
    e = e0[keep].merge(pan[["iso3","crop_standard_name","n_studies_fractional"]],
                       on=["iso3","crop_standard_name"], how="left")
    e["n_studies_fractional"]=e.n_studies_fractional.fillna(0.0)
    e["studied"]=(e.n_studies_fractional>0).astype(float)
    e9=e.merge(corr[["iso3","need9_floor9"]].rename(columns={"need9_floor9":"need"}),on="iso3",how="left").dropna(subset=["need"])
    e11=e.merge(n11[["iso3","need_rank_pct"]].rename(columns={"need_rank_pct":"need"}),on="iso3",how="left").dropna(subset=["need"])
    mv={}
    for lab,df,extra in [("baseline",e9,None),("11-component index",e11,None)]:
        m=logit(df,X); mv[lab]=dict(area=m.params["log_area"],rd=m.params["rd"],need=m.params["need"])
    for lab,df in [("crop FE",e9),("region FE",e9)]:
        col="crop_standard_name" if lab=="crop FE" else "wb_region"
        dd=pd.get_dummies(df[col],prefix=col[:4],drop_first=True)
        d2=pd.concat([df,dd],axis=1); m=logit(d2,X+list(dd.columns))
        mv[lab]=dict(area=m.params["log_area"],rd=m.params["rd"],need=m.params["need"])
    for grp,d_ in e9.groupby("wb_income_group"):
        if d_.studied.nunique()<2 or d_.studied.sum()<12: continue
        try:
            m=logit(d_,X); mv[f"income: {grp}"]=dict(area=m.params["log_area"],rd=m.params["rd"],need=m.params["need"])
        except Exception: pass
    for lab,drop in [("excl. USA",["USA"]),("excl. China",["CHN"]),("excl. India",["IND"])]:
        d_=e9[~e9.iso3.isin(drop)]; m=logit(d_,X)
        mv[lab]=dict(area=m.params["log_area"],rd=m.params["rd"],need=m.params["need"])
    mvdf=pd.DataFrame(mv).T
    print("\nMODEL VARIANTS")
    print(mvdf.round(3).to_string())
    for k in ["area","rd","need"]:
        v=mvdf[k].dropna()
        print(f"   {k}: baseline {mv['baseline'][k]:.3f}  range [{v.min():.3f}, {v.max():.3f}]  n={len(v)}")

    # ---------------- spatial variants ----------------
    cty=pan.groupby("iso3",as_index=False).n_studies_fractional.sum()
    g=lay.merge(cty,on="iso3",how="left").merge(corr[["iso3","need9_floor9"]],on="iso3",how="left").merge(
        n11[["iso3","need_rank_pct"]],on="iso3",how="left")
    g["n_studies_fractional"]=g.n_studies_fractional.fillna(0.0)
    sv={}
    for idx,col in [("9-comp","need9_floor9"),("11-comp","need_rank_pct")]:
        for meth in ["average","min","max","dense"]:
            rp=g.n_studies_fractional.rank(pct=True,method=meth)*100
            gg=g.assign(gap=rp-pd.to_numeric(g[col],errors="coerce")*100)
            o=gg[gg.gap.notna()].reset_index(drop=True)
            for k in ([4,6,8] if meth=="average" else [6]):
                w=knn_w(o,k); w.transform="r"
                sv[f"{idx} {meth} k={k}"]=esda.Moran(o.gap.values,w,permutations=999).I
    zs=np.log1p(g.n_studies_fractional); zs=(zs-zs.mean())/zs.std()
    gg=g.assign(gap=zs*100/4-pd.to_numeric(g.need9_floor9,errors="coerce")*100/4)
    o=gg[gg.gap.notna()].reset_index(drop=True); w=knn_w(o,6); w.transform="r"
    sv["9-comp standardised log count"]=esda.Moran(o.gap.values,w,permutations=999).I
    print("\nSPATIAL VARIANTS"); 
    for k,v in sv.items(): print(f"   {k:34s} {v:.4f}")
    svals=list(sv.values())
    print(f"   -> baseline {sv['9-comp average k=6']:.3f}  range [{min(svals):.3f}, {max(svals):.3f}]  n={len(svals)}")

    pd.DataFrame([{"quantity":"gini","baseline":gv["baseline (fractional)"],"lo":min(gvals),"hi":max(gvals),"n":len(gvals)},
                  {"quantity":"logit_area","baseline":mv["baseline"]["area"],"lo":mvdf.area.min(),"hi":mvdf.area.max(),"n":len(mvdf)},
                  {"quantity":"logit_rd","baseline":mv["baseline"]["rd"],"lo":mvdf.rd.min(),"hi":mvdf.rd.max(),"n":len(mvdf)},
                  {"quantity":"logit_need","baseline":mv["baseline"]["need"],"lo":mvdf.need.min(),"hi":mvdf.need.max(),"n":len(mvdf)},
                  {"quantity":"moran","baseline":sv["9-comp average k=6"],"lo":min(svals),"hi":max(svals),"n":len(svals)},
                 ]).to_csv(HERE/"robustness_ranges.csv",index=False)
    pd.DataFrame(gv,index=["gini"]).T.to_csv(HERE/"gini_variants.csv")
    mvdf.to_csv(HERE/"model_variants.csv")
    pd.Series(sv).to_csv(HERE/"spatial_variants.csv")
    for f in ["robustness_ranges.csv","gini_variants.csv","model_variants.csv","spatial_variants.csv"]:
        lg.add_output(HERE/f)
    lg.finish()


if __name__ == "__main__":
    main()
