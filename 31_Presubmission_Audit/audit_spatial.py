"""Pre-submission audit M6/XV + scale-excluded mismatch: observed-data spatial analysis.

Reruns the mismatch spatial statistics using only countries with an observed mismatch
value, with weights rebuilt on the valid sample, so no mean-filled unit can enter a
reported cluster. Also compares weight matrices on a common connected subsample (queen's
non-island set) so matrix sensitivity is not confounded with sample composition, and runs
the scale-excluded mismatch through the same machinery.
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import esda
from libpysal.weights import KNN, Queen, DistanceBand

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_Project_Management"))
from project_config import P, RANDOM_SEED, RunLogger  # noqa: E402

PERM = 9999
LAB = {0: "Not significant", 1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}


def fdr(p, alpha=0.05):
    p = np.asarray(p, float); out = np.zeros_like(p, bool)
    order = np.argsort(p); m = len(p)
    passed = p[order] <= alpha * (np.arange(1, m + 1) / m)
    if passed.any(): out[order[:np.max(np.where(passed)[0]) + 1]] = True
    return out


def knn_w(gdf, k):
    return KNN.from_dataframe(gdf.set_geometry(gpd.points_from_xy(gdf.centroid_lon, gdf.centroid_lat), crs="EPSG:4326").to_crs("+proj=eqearth"), k=k)


def run(y, w, tag, rows_g, rows_l=None):
    w.transform = "r"
    mi = esda.Moran(y, w, permutations=PERM)
    rows_g.append({"analysis": tag, "n": w.n, "morans_I": round(float(mi.I), 4),
                   "p_sim": float(mi.p_sim)})
    if rows_l is not None:
        lm = esda.Moran_Local(y, w, permutations=PERM, seed=RANDOM_SEED)
        cat = np.where(fdr(lm.p_sim), lm.q, 0)
        rows_l.append({"analysis": tag, "n_sig_fdr": int((cat > 0).sum()),
                       "HH": int((cat == 1).sum()), "LL": int((cat == 3).sum()),
                       "HL": int((cat == 4).sum()), "LH": int((cat == 2).sum())})
        return cat
    return None


def main() -> int:
    lg = RunLogger("audit_03_spatial")
    np.random.seed(RANDOM_SEED)
    lay = gpd.read_parquet(P["weights"] / "country_analytical_layer.parquet")
    panel = pd.read_parquet(P["integration"] / "country_crop_panel.parquet")
    n9 = pd.read_parquet(HERE / "need_index_scale_excluded.parquet")[
        ["iso3", "need_rank_pct", "need9_rank_pct", "n_components_observed"]]
    for f in ("country_analytical_layer.parquet",): lg.add_input(P["weights"] / f)

    cty = panel.groupby("iso3", as_index=False)["n_studies_fractional"].sum()
    g = lay.merge(cty, on="iso3", how="left").merge(n9, on="iso3", how="left")
    g["n_studies_fractional"] = g["n_studies_fractional"].fillna(0.0)
    g["research_pct"] = g["n_studies_fractional"].rank(pct=True) * 100
    g["gap11"] = g["research_pct"] - pd.to_numeric(g["need_rank_pct"], errors="coerce") * 100
    g["gap9"] = g["research_pct"] - pd.to_numeric(g["need9_rank_pct"], errors="coerce") * 100

    rows_g, rows_l = [], []

    # --- observed-only baseline mismatch (drop mean-filled units) -------------------
    obs = g[g["gap11"].notna()].reset_index(drop=True)
    lg.count("countries_observed_gap11", len(obs))
    for k in (4, 6, 8):
        cat = run(obs["gap11"].values, knn_w(obs, k), f"gap11_observed_knn{k}",
                  rows_g, rows_l if k == 6 else None)
        if k == 6:
            obs["lisa_cat_gap11"] = [LAB[c] for c in cat]
    q = Queen.from_dataframe(obs, use_index=False, silence_warnings=True)
    ni = [i for i in range(len(obs)) if i not in q.islands]
    run(obs["gap11"].values, knn_w(obs, 6), "gap11_observed_knn6_dup", rows_g)  # stability dup
    # queen on its connected subset
    obs_q = obs.iloc[ni].reset_index(drop=True)
    run(obs_q["gap11"].values, Queen.from_dataframe(obs_q, use_index=False,
        silence_warnings=True), "gap11_observed_queen_connected", rows_g)

    # --- XV: common connected sample, all matrix types ------------------------------
    for k in (4, 6, 8):
        run(obs_q["gap11"].values, knn_w(obs_q, k), f"gap11_commonsample_knn{k}", rows_g)
    lg.count("common_sample_n", len(obs_q))

    # --- scale-excluded mismatch ----------------------------------------------------
    o9 = g[g["gap9"].notna()].reset_index(drop=True)
    cat9 = run(o9["gap9"].values, knn_w(o9, 6), "gap9_observed_knn6", rows_g, rows_l)
    o9["lisa_cat_gap9"] = [LAB[c] for c in cat9]
    for k in (4, 8):
        run(o9["gap9"].values, knn_w(o9, k), f"gap9_observed_knn{k}", rows_g)
    gs = esda.G_Local(o9["gap9"].values, knn_w(o9, 6), permutations=PERM, star=True,
                      seed=RANDOM_SEED)
    gsig = fdr(gs.p_sim)
    lg.count("gap9_getis_hot", int(((gs.Zs > 0) & gsig).sum()))
    lg.count("gap9_getis_cold", int(((gs.Zs < 0) & gsig).sum()))

    # --- Mo1: component-coverage subset ---------------------------------------------
    o7 = g[g["gap11"].notna() & (g["n_components_observed"] >= 7)].reset_index(drop=True)
    run(o7["gap11"].values, knn_w(o7, 6), "gap11_needcov_ge7_knn6", rows_g)

    G = pd.DataFrame(rows_g); L = pd.DataFrame(rows_l)
    G.to_csv(HERE / "spatial_audit_global.csv", index=False)
    L.to_csv(HERE / "spatial_audit_lisa.csv", index=False)
    o9[["iso3", "gap9", "lisa_cat_gap9"]].to_csv(HERE / "gap9_lisa_clusters.csv", index=False)
    obs[["iso3", "gap11", "lisa_cat_gap11"]].to_csv(HERE / "gap11_observed_lisa_clusters.csv", index=False)
    for f in ("spatial_audit_global.csv", "spatial_audit_lisa.csv"):
        lg.add_output(HERE / f)
    print(G.to_string(index=False)); print(); print(L.to_string(index=False))
    # cluster membership shifts vs published mean-filled analysis
    pub = pd.read_csv(P["spatialecon"] / "research_side_lisa_clusters.csv")[
        ["iso3", "evidence_gap_lisa_cat_fdr"]]
    cmp = pub.merge(obs[["iso3", "lisa_cat_gap11"]], on="iso3", how="outer")
    diff = cmp[(cmp.evidence_gap_lisa_cat_fdr != cmp.lisa_cat_gap11)
               & ~(cmp.evidence_gap_lisa_cat_fdr.isna() & cmp.lisa_cat_gap11.isna())]
    print("\ncluster-category changes vs published mean-filled run:")
    print(diff.to_string(index=False) if len(diff) else "  NONE")
    lg.finish(); return 0


if __name__ == "__main__":
    sys.exit(main())
