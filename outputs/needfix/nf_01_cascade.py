"""Adopt the corrected nine-component coverage floor as the primary need index.

The published floor counted all eleven recorded indicators, including the two
agricultural-scale shares, which are never missing. Every country therefore received two
free components toward the five-component threshold, and five countries were indexed on
as few as one observed need component. The index value excluded agricultural scale; the
inclusion rule did not.

This rebuilds the full cascade on the corrected floor: mismatch measure, descriptive
counts, regressions, marginal effects, and spatial statistics. The eleven-indicator rule
becomes the sensitivity variant.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

OUT = HERE
SEED, PERM = 20260730, 9999
X = ["log_area", "rd", "need", "log_gdp_pc", "tertiary", "internet", "log_population"]


def fdr(p, a=0.05):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    thr = a * (np.arange(1, n + 1) / n)
    ok = p[o] <= thr
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    r = np.zeros(n, bool)
    if k: r[o[:k]] = True
    return r


def logit(df, xs, y="studied"):
    M = sm.add_constant(df[xs].astype(float), has_constant="add")
    return sm.Logit(df[y].astype(float), M).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": df["iso3"].to_numpy()})


def ppml(df, xs, use_offset=True, y="n_studies_fractional"):
    d = df[df.area_ha_mean.fillna(0) > 0].copy()
    M = sm.add_constant(d[xs].astype(float), has_constant="add")
    off = np.log(d.area_ha_mean.astype(float)) if use_offset else None
    return sm.GLM(d[y].astype(float), M, family=sm.families.Poisson(), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": d.iso3.to_numpy()}), d


def main():
    lg = RunLogger("nf_01_cascade")
    corr = pd.read_parquet(ROOT / "outputs" / "revision" / "need_index_corrected_floor.parquet")
    n9 = pd.read_parquet(ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet")
    layer = pd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    e0 = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    lg.add_input(ROOT / "16_Econometrics" / "estimation_sample.parquet")

    print("=" * 74)
    print("CORRECTED COVERAGE FLOOR: five of the NINE primary need components")
    print("=" * 74)
    print(f"countries indexed, corrected floor : {int(corr.need9_floor9.notna().sum())}")
    print(f"countries indexed, published floor : {int(corr.need9_rank_pct.notna().sum())}")
    drop = corr[corr.need9_rank_pct.notna() & corr.need9_floor9.isna()]
    print(f"countries dropped                  : {', '.join(drop.iso3)}")
    print(f"  their observed need components   : {list(drop.obs9)}")

    # ---------------------------------------------------------------- descriptive cascade
    d = layer[["iso3"]].merge(
        n9[["iso3", "n_studies_fractional"]], on="iso3", how="left").merge(
        corr[["iso3", "need9_floor9", "need9_rank_pct"]], on="iso3", how="left")
    d["n_studies_fractional"] = d.n_studies_fractional.fillna(0.0)

    # research percentile is ranked over ALL layer countries, then differenced against the
    # need percentile where one exists. This matches the archived pipeline: the research
    # ranking must not depend on which countries happen to carry a need index.
    d["research_pct"] = d.n_studies_fractional.rank(pct=True) * 100
    rows = []
    for tag, col in [("corrected (188)", "need9_floor9"), ("published (193)", "need9_rank_pct")]:
        dd = d.copy()
        dd["need_pct"] = pd.to_numeric(dd[col], errors="coerce") * 100
        dd["gap"] = dd.research_pct - dd.need_pct
        sub = dd[dd.gap.notna()].reset_index(drop=True)
        mn, mr = sub.need_pct.median(), sub.research_pct.median()
        quad = int(((sub.need_pct >= mn) & (sub.research_pct <= mr)).sum())
        ties = int((sub.n_studies_fractional == 0).sum())
        rows.append(dict(index=tag, countries=len(sub), high_need_low_research=quad,
                         tied_at_zero=ties))
        print(f"\n{tag}: n={len(sub)}  high-need/low-research quadrant={quad}  tied at zero={ties}")
        if tag.startswith("corrected"):
            corrected = sub.copy()
        else:
            published = sub.copy()
    pd.DataFrame(rows).to_csv(OUT / "descriptive_counts.csv", index=False)

    # ---------------------------------------------------------------- regressions
    print("\n" + "-" * 74)
    print("REGRESSIONS on the corrected index")
    print("-" * 74)
    # the estimation sample already carries an eleven-component "need"; drop it so the
    # corrected nine-component index takes that name
    e = e0.drop(columns=["need"]).merge(
        corr[["iso3", "need9_floor9"]].rename(columns={"need9_floor9": "need"}),
        on="iso3", how="left")
    n_before = len(e)
    e = e.dropna(subset=["need"])
    print(f"estimation cells {n_before} -> {len(e)}   countries {e.iso3.nunique()}")

    est = []
    m = logit(e, X); ci = m.conf_int()
    for t in X:
        est.append(dict(model="participation", term=t, estimate=m.params[t], se=m.bse[t],
                        ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1], n=int(m.nobs)))
    print(f"  participation: log_area {m.params['log_area']:.3f} "
          f"[{ci.loc['log_area',0]:.3f},{ci.loc['log_area',1]:.3f}]  "
          f"need {m.params['need']:.3f} [{ci.loc['need',0]:.3f},{ci.loc['need',1]:.3f}]  "
          f"rd {m.params['rd']:.3f}  pseudoR2 {1-m.llf/m.llnull:.3f}  events {int(e.studied.sum())}")

    Xo = [x for x in X if x != "log_area"]
    p1, dp = ppml(e, Xo); ci1 = p1.conf_int()
    for t in Xo:
        est.append(dict(model="intensity_offset", term=t, estimate=p1.params[t], se=p1.bse[t],
                        ci_low=ci1.loc[t, 0], ci_high=ci1.loc[t, 1], n=int(p1.nobs)))
    print(f"  intensity(offset): need {p1.params['need']:.3f} "
          f"[{ci1.loc['need',0]:.3f},{ci1.loc['need',1]:.3f}]  rd {p1.params['rd']:.3f} "
          f"[{ci1.loc['rd',0]:.3f},{ci1.loc['rd',1]:.3f}]  n={int(p1.nobs)}")

    p2, _ = ppml(e, X, use_offset=False); ci2 = p2.conf_int()
    ba, sa = p2.params["log_area"], p2.bse["log_area"]
    w = ((ba - 1) / sa) ** 2; pw = 1 - stats.chi2.cdf(w, 1)
    for t in X:
        est.append(dict(model="intensity_free_area", term=t, estimate=p2.params[t], se=p2.bse[t],
                        ci_low=ci2.loc[t, 0], ci_high=ci2.loc[t, 1], n=int(p2.nobs)))
    print(f"  free elasticity: {ba:.3f} (SE {sa:.3f}, CI [{ci2.loc['log_area',0]:.3f},"
          f"{ci2.loc['log_area',1]:.3f}])  Wald chi2 {w:.2f} p={pw:.4f}")
    print(f"    need under free elasticity: {p2.params['need']:.3f} "
          f"[{ci2.loc['need',0]:.3f},{ci2.loc['need',1]:.3f}]")

    cd = pd.get_dummies(e.crop_standard_name, prefix="crop", drop_first=True)
    e2 = pd.concat([e, cd], axis=1)
    mfe = logit(e2, X + list(cd.columns)); cife = mfe.conf_int()
    for t in X:
        est.append(dict(model="participation_cropFE", term=t, estimate=mfe.params[t],
                        se=mfe.bse[t], ci_low=cife.loc[t, 0], ci_high=cife.loc[t, 1],
                        n=int(mfe.nobs)))
    pfe, _ = ppml(e2, Xo + list(cd.columns)); cipfe = pfe.conf_int()
    for t in Xo:
        est.append(dict(model="intensity_cropFE", term=t, estimate=pfe.params[t], se=pfe.bse[t],
                        ci_low=cipfe.loc[t, 0], ci_high=cipfe.loc[t, 1], n=int(pfe.nobs)))
    print(f"  crop FE: participation log_area {mfe.params['log_area']:.3f}  "
          f"intensity need {pfe.params['need']:.3f} "
          f"[{cipfe.loc['need',0]:.3f},{cipfe.loc['need',1]:.3f}]")
    pd.DataFrame(est).to_csv(OUT / "model_estimates_corrected.csv", index=False)

    # ---------------------------------------------------------------- marginal effects
    me = m.get_margeff(at="overall", method="dydx")
    ame = pd.DataFrame({"term": [t for t in m.params.index if t != "const"],
                        "ame": me.margeff, "se": me.margeff_se})
    ame["ci_low"] = ame.ame - 1.959964 * ame.se
    ame["ci_high"] = ame.ame + 1.959964 * ame.se
    ame["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(ame.ame / ame.se)))
    iqr = []
    Mfull = sm.add_constant(e[X].astype(float), has_constant="add")
    for t in X:
        q1, q3 = e[t].quantile(.25), e[t].quantile(.75)
        a1, a3 = e.copy(), e.copy(); a1[t] = q1; a3[t] = q3
        p_1 = m.predict(sm.add_constant(a1[X].astype(float), has_constant="add")).mean()
        p_3 = m.predict(sm.add_constant(a3[X].astype(float), has_constant="add")).mean()
        iqr.append(dict(term=t, p25=q1, p75=q3, iqr_dprob=p_3 - p_1))
    ame = ame.merge(pd.DataFrame(iqr), on="term")
    ame.to_csv(OUT / "ame_corrected.csv", index=False)
    print("\n  average marginal effects (delta method, clustered):")
    for _, r in ame.iterrows():
        print(f"    {r.term:15s} AME {r.ame:7.4f} (SE {r.se:.4f}, p={r.p_value:.4f})  "
              f"IQR {100*r.iqr_dprob:5.1f} pp")

    # ---------------------------------------------------------------- spatial
    print("\n" + "-" * 74)
    print("SPATIAL on the corrected index")
    print("-" * 74)
    geom = gpd.GeoSeries.from_wkb(layer.geometry) if layer.geometry.dtype == object else layer.geometry
    cent = gpd.GeoDataFrame(layer.copy(), geometry=geom, crs="EPSG:4326")
    cxy = np.column_stack([cent.centroid_lon.values, cent.centroid_lat.values])
    proj = gpd.GeoDataFrame(layer[["iso3"]],
                            geometry=gpd.points_from_xy(cxy[:, 0], cxy[:, 1]),
                            crs="EPSG:4326").to_crs("+proj=eqearth")
    ids = layer.iso3.tolist()
    sub = corrected
    pos = [ids.index(i) for i in sub.iso3]
    g = gpd.GeoDataFrame(sub[["iso3"]], geometry=gpd.points_from_xy(
        proj.geometry.x.values[pos], proj.geometry.y.values[pos]))
    srows = []
    for k in [4, 6, 8]:
        w = KNN.from_dataframe(g, k=k, ids=sub.iso3.tolist())
        assert list(w.id_order) == sub.iso3.tolist()
        w.transform = "r"
        mo = Moran(sub.gap.values, w, permutations=PERM)
        srows.append(dict(k=k, n=len(sub), morans_I=mo.I, p_sim=mo.p_sim))
        print(f"  Moran's I  k={k}: {mo.I:.4f} (p = {mo.p_sim:.4f}, n = {len(sub)})")
    pd.DataFrame(srows).to_csv(OUT / "moran_corrected.csv", index=False)

    w6 = KNN.from_dataframe(g, k=6, ids=sub.iso3.tolist()); w6.transform = "r"
    lm = Moran_Local(sub.gap.values, w6, permutations=PERM, seed=SEED)
    rej = fdr(lm.p_sim)
    q = np.where(rej, lm.q, 0)
    CAT = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low", 0: "Not significant"}
    sub["lisa_cat_gap9"] = [CAT[int(x)] for x in q]
    counts = sub.lisa_cat_gap9.value_counts()
    print(f"  LISA: HH={int((q==1).sum())} LL={int((q==3).sum())} "
          f"HL={int((q==4).sum())} LH={int((q==2).sum())}")
    print(f"    High-High: {', '.join(sorted(sub[sub.lisa_cat_gap9=='High-High'].iso3))}")
    print(f"    Low-Low  : {', '.join(sorted(sub[sub.lisa_cat_gap9=='Low-Low'].iso3))}")
    go = G_Local(sub.gap.values, w6, permutations=PERM, seed=SEED)
    rg = fdr(go.p_sim)
    print(f"  Getis-Ord: hot={int((rg & (go.Zs>0)).sum())} cold={int((rg & (go.Zs<0)).sum())}")
    sub[["iso3", "gap", "research_pct", "need_pct", "lisa_cat_gap9"]].to_csv(
        OUT / "gap9_corrected_lisa.csv", index=False)

    for f in ["descriptive_counts.csv", "model_estimates_corrected.csv", "ame_corrected.csv",
              "moran_corrected.csv", "gap9_corrected_lisa.csv"]:
        lg.add_output(OUT / f)
    lg.finish()
    print("\nnf_01_cascade complete")


if __name__ == "__main__":
    main()
