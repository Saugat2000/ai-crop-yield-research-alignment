"""Revision analyses: spatial weight construction, tie sensitivity, and LM diagnostics.

  Step 15 / R1-M8 - the published k-nearest-neighbour matrix is built on polygon
                    centroids projected to Equal Earth, an equal-AREA projection whose
                    planar distances are not great-circle distances. This rebuilds the
                    weights on geodesic (haversine) distance and reruns every spatial
                    statistic.
  Step 16 / R1-Mo1 - 71 of 193 countries tie at the lowest research percentile. This
                    tests the mismatch and Moran results under alternative tie
                    conventions.
  item 16          - Lagrange Multiplier tests for spatial lag and spatial error on the
                    participation model residuals.
  R1-M3            - spatial statistics recomputed on the corrected 188-country floor.
"""
from __future__ import annotations
import sys, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
warnings.filterwarnings("ignore")
from libpysal.weights import KNN, W as PysalW
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

OUT = HERE
SEED = 20260730
PERM = 9999
EARTH_KM = 6371.0088


def fdr(p, a=0.05):
    """Benjamini-Hochberg, returning the boolean reject vector."""
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    thr = a * (np.arange(1, n + 1) / n)
    passed = p[o] <= thr
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    rej = np.zeros(n, bool)
    if k:
        rej[o[:k]] = True
    return rej


def haversine_xyz(lon, lat):
    """Unit-sphere Cartesian coordinates: Euclidean distance there is monotone in
    great-circle distance, so k-nearest neighbours are the true geodesic neighbours."""
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def knn_from_points(xy, ids, k):
    g = gpd.GeoDataFrame({"id": ids}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]))
    return KNN.from_dataframe(g, k=k, ids=list(ids))


def knn_geodesic(lon, lat, ids, k):
    from scipy.spatial import cKDTree
    P3 = haversine_xyz(np.asarray(lon), np.asarray(lat))
    tree = cKDTree(P3)
    _, idx = tree.query(P3, k=k + 1)
    nb = {ids[i]: [ids[j] for j in idx[i][1:]] for i in range(len(ids))}
    # id_order must be pinned: libpysal alphabetises dict keys otherwise, which would
    # silently misalign the data vector against the weights.
    return PysalW(nb, id_order=list(ids), silence_warnings=True)


def moran(y, w, ids=None):
    if ids is not None:
        assert list(w.id_order) == list(ids), "weights id_order does not match the data order"
    w.transform = "r"
    m = Moran(y, w, permutations=PERM)
    return m.I, m.p_sim


def lisa_counts(y, w, a=0.05, ids=None):
    if ids is not None:
        assert list(w.id_order) == list(ids), "weights id_order does not match the data order"
    w.transform = "r"
    lm = Moran_Local(y, w, permutations=PERM, seed=SEED)
    rej = fdr(lm.p_sim, a)
    q = np.where(rej, lm.q, 0)
    return {"HH": int((q == 1).sum()), "LL": int((q == 3).sum()),
            "LH": int((q == 2).sum()), "HL": int((q == 4).sum())}, q, lm


def main():
    lg = RunLogger("rev_04_spatial")
    layer = pd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    n9 = pd.read_parquet(ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet")
    corr = pd.read_parquet(OUT / "need_index_corrected_floor.parquet")
    lg.add_input(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    lg.add_input(ROOT / "31_Presubmission_Audit" / "need_index_scale_excluded.parquet")

    geom = gpd.GeoSeries.from_wkb(layer.geometry) if layer.geometry.dtype == object else layer.geometry
    gdf = gpd.GeoDataFrame(layer.copy(), geometry=geom, crs="EPSG:4326")
    cent_ee = gdf.to_crs("+proj=eqearth").geometry.centroid
    ee_xy = np.column_stack([cent_ee.x.values, cent_ee.y.values])
    cent_deg = gdf.geometry.centroid
    ids = layer.iso3.tolist()

    # ------------------------------------------------- guard: reproduce published matrix
    stored = pickle.load(open(ROOT / "14_Spatial_Weights" / "spatial_weights.pkl", "rb"))["knn_k6"]
    rebuilt = knn_from_points(ee_xy, ids, 6)
    same = sum(set(stored.neighbors[i]) == set(rebuilt.neighbors[i]) for i in ids)
    print(f"GUARD: Equal Earth rebuild reproduces published knn_k6 for {same}/{len(ids)} countries")

    geo6 = knn_geodesic(cent_deg.x.values, cent_deg.y.values, ids, 6)
    agree = sum(set(stored.neighbors[i]) == set(geo6.neighbors[i]) for i in ids)
    print(f"Geodesic knn_k6 agrees with the published Equal Earth matrix for "
          f"{agree}/{len(ids)} countries ({100*agree/len(ids):.1f}%)")

    # ------------------------------------------------- assemble the mismatch variable
    d = layer[["iso3"]].merge(n9[["iso3", "need9_rank_pct", "gap9", "n_studies_fractional",
                                  "research_pct", "research_pct_min"]], on="iso3", how="left")
    d = d.merge(corr[["iso3", "need9_floor9", "obs9"]], on="iso3", how="left")
    obs = d.gap9.notna()
    print(f"\nobserved gap9 countries in the spatial layer: {int(obs.sum())}")

    rows = []
    for k in [4, 6, 8]:
        sub = d[obs].reset_index(drop=True)
        idx = sub.iso3.tolist()
        pos = [ids.index(i) for i in idx]
        w_ee = knn_from_points(ee_xy[pos], idx, k)
        w_geo = knn_geodesic(cent_deg.x.values[pos], cent_deg.y.values[pos], idx, k)
        for nm, w in [("equal_earth", w_ee), ("geodesic", w_geo)]:
            I, p = moran(sub.gap9.values, w, idx)
            rows.append(dict(variable="gap9", weights=nm, k=k, n=len(sub), morans_I=I, p_sim=p))
            print(f"  gap9 k={k} {nm:12s} I = {I:.4f} (p = {p:.4f})")

    # corrected 188-country floor
    obs2 = d.need9_floor9.notna() & d.n_studies_fractional.notna()
    sub2 = d[obs2].reset_index(drop=True)
    rp = sub2.n_studies_fractional.rank(pct=True) * 100
    gap_corr = rp - sub2.need9_floor9 * 100
    idx2 = sub2.iso3.tolist(); pos2 = [ids.index(i) for i in idx2]
    for nm, w in [("equal_earth", knn_from_points(ee_xy[pos2], idx2, 6)),
                  ("geodesic", knn_geodesic(cent_deg.x.values[pos2], cent_deg.y.values[pos2], idx2, 6))]:
        I, p = moran(gap_corr.values, w, idx2)
        rows.append(dict(variable="gap9_corrected_floor", weights=nm, k=6, n=len(sub2),
                         morans_I=I, p_sim=p))
        print(f"  gap9 corrected floor (n={len(sub2)}) {nm:12s} I = {I:.4f} (p = {p:.4f})")

    # ------------------------------------------------- Step 16: tie conventions
    sub = d[obs].reset_index(drop=True)
    idx = sub.iso3.tolist(); pos = [ids.index(i) for i in idx]
    w_ee = knn_from_points(ee_xy[pos], idx, 6)
    w_geo = knn_geodesic(cent_deg.x.values[pos], cent_deg.y.values[pos], idx, 6)
    nz = (sub.n_studies_fractional.fillna(0) == 0).sum()
    print(f"\ncountries tied at zero research: {nz} of {len(sub)}")
    ties = {
        "average (published)": sub.n_studies_fractional.rank(pct=True, method="average"),
        "min": sub.n_studies_fractional.rank(pct=True, method="min"),
        "max": sub.n_studies_fractional.rank(pct=True, method="max"),
        "dense": sub.n_studies_fractional.rank(pct=True, method="dense"),
    }
    need_pct = sub.gap9.values / 100.0  # placeholder, replaced below
    need_p = (sub.n_studies_fractional.rank(pct=True, method="average") * 100 - sub.gap9)
    for nm, rk in ties.items():
        g = rk * 100 - need_p
        for wn, w in [("equal_earth", w_ee), ("geodesic", w_geo)]:
            I, p = moran(g.values, w, idx)
            rows.append(dict(variable=f"gap9_tie_{nm}", weights=wn, k=6, n=len(g),
                             morans_I=I, p_sim=p))
        I6, p6 = moran(g.values, w_ee, idx)
        print(f"  tie rule {nm:20s} I = {I6:.4f} (p = {p6:.4f})")
    # standardised continuous research count as a zero-preserving alternative
    zcount = (np.log1p(sub.n_studies_fractional.fillna(0)) -
              np.log1p(sub.n_studies_fractional.fillna(0)).mean()) / \
             np.log1p(sub.n_studies_fractional.fillna(0)).std()
    galt = zcount * 100 / 4 - need_p / 4
    I, p = moran(galt.values, w_ee, idx)
    rows.append(dict(variable="gap9_log_count_standardised", weights="equal_earth", k=6,
                     n=len(galt), morans_I=I, p_sim=p))
    print(f"  log-count standardised alternative I = {I:.4f} (p = {p:.4f})")

    pd.DataFrame(rows).to_csv(OUT / "spatial_weight_and_tie_sensitivity.csv", index=False)

    # ------------------------------------------------- LISA under both weight definitions
    lrows = []
    for wn, w in [("equal_earth", w_ee), ("geodesic", w_geo)]:
        cnt, q, lm = lisa_counts(sub.gap9.values, w, ids=idx)
        cnt["weights"] = wn; cnt["n"] = len(sub)
        lrows.append(cnt)
        print(f"\nLISA ({wn}): HH={cnt['HH']} LL={cnt['LL']} HL={cnt['HL']} LH={cnt['LH']}")
        if wn == "geodesic":
            sub2b = sub.copy(); sub2b["lisa_q_geodesic"] = q
            sub2b[["iso3", "gap9", "lisa_q_geodesic"]].to_csv(
                OUT / "lisa_geodesic_clusters.csv", index=False)
        g = G_Local(sub.gap9.values, w, permutations=PERM, seed=SEED)
        rej = fdr(g.p_sim)
        print(f"  Getis-Ord: hot={int((rej & (g.Zs>0)).sum())} cold={int((rej & (g.Zs<0)).sum())}")
    pd.DataFrame(lrows).to_csv(OUT / "lisa_weight_comparison.csv", index=False)

    # ------------------------------------------------- item 16: LM tests
    e = pd.read_parquet(ROOT / "16_Econometrics" / "estimation_sample.parquet")
    e = e.merge(n9[["iso3", "need9_rank_pct"]].rename(columns={"need9_rank_pct": "need9"}),
                on="iso3", how="left")
    X = sm.add_constant(e[["log_area", "rd", "need9", "log_gdp_pc", "tertiary",
                           "internet", "log_population"]].astype(float), has_constant="add")
    m = sm.Logit(e.studied.astype(float), X).fit(disp=0, cov_type="cluster",
                                                 cov_kwds={"groups": e.iso3.to_numpy()})
    e["resid"] = e.studied.astype(float) - m.predict(X)
    cm = e.groupby("iso3", as_index=False).agg(resid=("resid", "mean"),
                                               log_area=("log_area", "mean"),
                                               rd=("rd", "mean"), need9=("need9", "mean"),
                                               log_gdp_pc=("log_gdp_pc", "mean"),
                                               tertiary=("tertiary", "mean"),
                                               internet=("internet", "mean"),
                                               log_population=("log_population", "mean"))
    cm = cm[cm.iso3.isin(ids)].reset_index(drop=True)
    cids = cm.iso3.tolist(); cpos = [ids.index(i) for i in cids]
    from spreg import OLS
    lmrows = []
    for wn, w in [("equal_earth", knn_from_points(ee_xy[cpos], cids, 6)),
                  ("geodesic", knn_geodesic(cent_deg.x.values[cpos], cent_deg.y.values[cpos], cids, 6))]:
        w.transform = "r"
        yv = cm[["resid"]].values
        Xv = cm[["log_area", "rd", "need9", "log_gdp_pc", "tertiary", "internet",
                 "log_population"]].values
        ols = OLS(yv, Xv, w=w, spat_diag=True, moran=True,
                  name_y="country_mean_residual", name_x=["log_area", "rd", "need9",
                  "log_gdp_pc", "tertiary", "internet", "log_population"], name_w=wn)
        for lab, key in [("LM (lag)", "lm_lag"), ("LM (error)", "lm_error"),
                         ("Robust LM (lag)", "rlm_lag"), ("Robust LM (error)", "rlm_error"),
                         ("Moran's I (error)", "moran_res")]:
            v = getattr(ols, key, None)
            if v is not None:
                lmrows.append(dict(weights=wn, test=lab, statistic=float(v[0]),
                                   p_value=float(v[-1]), n=len(cm)))
    lmt = pd.DataFrame(lmrows)
    lmt.to_csv(OUT / "lm_spatial_diagnostics.csv", index=False)
    print("\nLM SPATIAL DIAGNOSTICS on country-mean participation residuals")
    print(lmt.round(4).to_string(index=False))

    for f in ["spatial_weight_and_tie_sensitivity.csv", "lisa_weight_comparison.csv",
              "lisa_geodesic_clusters.csv", "lm_spatial_diagnostics.csv"]:
        lg.add_output(OUT / f)
    lg.finish()
    print("\nrev_04_spatial complete")


if __name__ == "__main__":
    main()
