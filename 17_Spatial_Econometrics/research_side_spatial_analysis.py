"""Phase 9c — spatial structure of the RESEARCH side and of the evidence gap.

The need-side run (phase09_02) established the machinery on data that does not depend on
the corpus. This is the same machinery on the quantities the paper is about: where
AI-based crop-yield research is done, and where it is done relative to need.

Conventions carried over unchanged from the need-side run, so the two are comparable:
9,999 permutations, alpha 0.05, kNN k=6 as the primary matrix, Benjamini-Hochberg across
the local statistics, and every alternative weights matrix reported rather than the one
that gives the neatest map.

Two rules matter here more than anywhere else in the project.

**Zero is not missing.** A country with no eligible study has `n_studies_fractional = 0`.
It enters every statistic as a zero. Filling it with a mean, or dropping it, would remove
exactly the observations the paper exists to describe.

**Islands are handled explicitly.** Under row-standardised queen contiguity a country with
no land neighbour has an undefined spatial lag and is dropped, silently, by most software.
kNN k=6 retains every country, which is why it is primary. The queen result is reported
beside it with the count of countries it loses.

Nothing here is called a cluster unless the FDR-adjusted local statistic supports it.
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RANDOM_SEED, RunLogger  # noqa: E402

warnings.filterwarnings("ignore")

PERMUTATIONS = 9999
ALPHA = 0.05
LISA_LABEL = {0: "Not significant", 1: "High-High", 2: "Low-High",
              3: "Low-Low", 4: "High-Low"}


def fdr(p, alpha=ALPHA):
    """Benjamini-Hochberg. Returns a boolean mask of rejections."""
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p, dtype=bool)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = order.size
    passed = p[order] <= alpha * (np.arange(1, m + 1) / m)
    if passed.any():
        out[order[:np.max(np.where(passed)[0]) + 1]] = True
    return out


def main() -> int:
    lg = RunLogger("phase09_03_research_side_spatial")
    np.random.seed(RANDOM_SEED)
    lg.count("random_seed", RANDOM_SEED)

    lay_p = P["weights"] / "country_analytical_layer.parquet"
    panel_p = P["integration"] / "country_crop_panel.parquet"
    need_p = P["indices"] / "country_need_indices.parquet"
    for f in (lay_p, panel_p, need_p):
        lg.add_input(f)

    lay = gpd.read_parquet(lay_p)
    panel = pd.read_parquet(panel_p)
    need = pd.read_parquet(need_p)

    # Country-level research side, aggregated from the country-crop panel so that the
    # fractional weights are preserved exactly.
    cty = panel.groupby("iso3", as_index=False).agg(
        n_studies_fractional=("n_studies_fractional", "sum"),
        n_studies_full=("n_studies_full", "sum"),
        citation_weighted=("citation_weighted_fractional", "sum"),
        crop_area_ha=("area_ha_mean", "sum"),
        crop_production_t=("production_t_mean", "sum"),
        cells_with_study=("has_any_study", "sum"),
        cells_total=("has_any_study", "size"))
    cty["research_intensity_per_mha"] = np.where(
        cty["crop_area_ha"] > 0,
        cty["n_studies_fractional"] / (cty["crop_area_ha"] / 1e6), np.nan)
    cty["log_studies"] = np.log1p(cty["n_studies_fractional"])
    cty["cell_coverage_pct"] = 100 * cty["cells_with_study"] / cty["cells_total"]

    g = lay.merge(cty, on="iso3", how="left").merge(
        need[["iso3", "need_rank_pct", "need_equal_pct", "need_pca_pct",
              "need_entropy_pct"]], on="iso3", how="left")

    # A country in the spatial layer with no panel row has no eligible study, which is a
    # zero. A country with no need index has MISSING external data. The two are kept apart.
    for c in ("n_studies_fractional", "n_studies_full", "citation_weighted"):
        g[c] = g[c].fillna(0.0)
    g["log_studies"] = np.log1p(g["n_studies_fractional"])
    lg.count("countries_in_layer", len(g))
    lg.count("countries_with_zero_studies", int((g["n_studies_fractional"] == 0).sum()))
    lg.count("countries_missing_need_index", int(g["need_rank_pct"].isna().sum()))

    # Evidence gap: research percentile minus need percentile. Positive means a country is
    # better represented in the literature than its need rank implies. Defined only where
    # the need index exists, so it is NA rather than a guess for the rest.
    #
    # Both terms must be on the SAME scale. `need_rank_pct` is a 0-1 proportion despite the
    # `_pct` suffix, while `research_pct` is 0-100. Subtracting them directly made the
    # "evidence gap" equal to the research percentile minus a number below 1: it correlated
    # with research_pct at 0.99995 and measured no gap at all. The rescale is asserted
    # rather than assumed, because the column name is what caused the error.
    g["research_pct"] = g["n_studies_fractional"].rank(pct=True) * 100
    _nr = pd.to_numeric(g["need_rank_pct"], errors="coerce")
    if _nr.max(skipna=True) > 1.5:
        raise ValueError(
            f"need_rank_pct max is {_nr.max():.3f}; this code expects a 0-1 proportion. "
            "Check the scale before computing the evidence gap.")
    g["need_pct_0_100"] = _nr * 100.0
    g["evidence_gap"] = g["research_pct"] - g["need_pct_0_100"]
    _r = g[["evidence_gap", "research_pct"]].dropna().corr().iloc[0, 1]
    if _r > 0.99:
        raise ValueError(
            f"evidence_gap correlates with research_pct at {_r:.5f}; the two terms are not "
            "on the same scale and the gap is not measuring a gap.")
    lg.count("evidence_gap_corr_with_research_pct", round(float(_r), 4))
    lg.count("countries_with_evidence_gap", int(g["evidence_gap"].notna().sum()))

    with open(P["weights"] / "spatial_weights.pkl", "rb") as fh:
        W = pickle.load(fh)

    VARS = {
        "log_studies": "Research output, log(1 + fractional study count)",
        "research_intensity_per_mha": "Studies per million hectares harvested",
        "citation_weighted": "Citation-weighted fractional study count",
        "evidence_gap": "Evidence gap, research percentile minus need percentile",
        "cell_coverage_pct": "Share of a country's crop systems with any study",
    }

    import esda

    global_rows, lisa_rows, frames, getis_rows = [], [], {}, []
    for vkey, label in VARS.items():
        for wname, w in W.items():
            ids = list(w.id_order)
            sub = g.set_index("iso3").reindex(ids)
            y = pd.to_numeric(sub[vkey], errors="coerce")
            n_missing = int(y.isna().sum())
            if y.notna().sum() < 20 or y.dropna().nunique() < 3:
                lg.warn(f"{vkey} on {wname}: too few usable values, skipped")
                continue
            yv = y.fillna(y.mean()).values
            mi = esda.Moran(yv, w, permutations=PERMUTATIONS)
            global_rows.append({
                "variable": vkey, "label": label, "weights": wname,
                "n": w.n, "n_islands": len(w.islands), "n_missing_filled": n_missing,
                "morans_I": round(float(mi.I), 4), "expected_I": round(float(mi.EI), 4),
                "z_sim": round(float(mi.z_sim), 3), "p_sim": float(mi.p_sim),
                "permutations": PERMUTATIONS})

            if wname != "knn_k6":
                continue

            lm = esda.Moran_Local(yv, w, permutations=PERMUTATIONS, seed=RANDOM_SEED)
            sig_raw, sig_fdr = lm.p_sim < ALPHA, fdr(lm.p_sim, ALPHA)
            cat_raw = np.where(sig_raw, lm.q, 0)
            cat_fdr = np.where(sig_fdr, lm.q, 0)
            frames[vkey] = pd.DataFrame({
                "iso3": ids,
                f"{vkey}_value": y.values,
                f"{vkey}_lisa_I": lm.Is,
                f"{vkey}_lisa_p": lm.p_sim,
                f"{vkey}_lisa_cat_raw": [LISA_LABEL[c] for c in cat_raw],
                f"{vkey}_lisa_cat_fdr": [LISA_LABEL[c] for c in cat_fdr],
                f"{vkey}_value_missing": y.isna().values})
            lisa_rows.append({
                "variable": vkey, "weights": wname,
                "n_sig_raw": int(sig_raw.sum()), "n_sig_fdr": int(sig_fdr.sum()),
                "HH_fdr": int((cat_fdr == 1).sum()), "LL_fdr": int((cat_fdr == 3).sum()),
                "HL_fdr": int((cat_fdr == 4).sum()), "LH_fdr": int((cat_fdr == 2).sum()),
                "n_missing_filled": n_missing,
                "note": "categories reported on FDR-adjusted p only; raw counts shown for "
                        "comparison and never used to call a cluster"})

            # Getis-Ord Gi*, which answers a different question from local Moran: where are
            # the high-value and low-value concentrations, rather than where is a value
            # like or unlike its neighbours.
            gs = esda.G_Local(yv, w, permutations=PERMUTATIONS, star=True, seed=RANDOM_SEED)
            gsig = fdr(gs.p_sim, ALPHA)
            getis_rows.append({
                "variable": vkey, "weights": wname,
                "n_hot_fdr": int(((gs.Zs > 0) & gsig).sum()),
                "n_cold_fdr": int(((gs.Zs < 0) & gsig).sum()),
                "n_sig_raw": int((gs.p_sim < ALPHA).sum())})
            frames[vkey][f"{vkey}_getis_z"] = gs.Zs
            frames[vkey][f"{vkey}_getis_p"] = gs.p_sim
            frames[vkey][f"{vkey}_getis_sig_fdr"] = gsig

    G = pd.DataFrame(global_rows)
    gp = P["spatialecon"] / "research_side_global_moran.csv"
    G.to_csv(gp, index=False)
    lg.add_output(gp, rows=len(G))

    L = pd.DataFrame(lisa_rows)
    lp = P["spatialecon"] / "research_side_local_moran_summary.csv"
    L.to_csv(lp, index=False)
    lg.add_output(lp, rows=len(L))

    GT = pd.DataFrame(getis_rows)
    gtp = P["spatialecon"] / "research_side_getis_ord_summary.csv"
    GT.to_csv(gtp, index=False)
    lg.add_output(gtp, rows=len(GT))

    merged = None
    for f in frames.values():
        merged = f if merged is None else merged.merge(f, on="iso3", how="outer")
    if merged is not None:
        merged = merged.merge(
            g[["iso3", "wb_region", "wb_income_group", "n_studies_fractional",
               "need_rank_pct", "evidence_gap"]], on="iso3", how="left")
        cp = P["spatialecon"] / "research_side_lisa_clusters.csv"
        merged.to_csv(cp, index=False)
        lg.add_output(cp, rows=len(merged))

    piv = G.pivot_table(index="variable", columns="weights", values="morans_I")
    sp = P["spatialecon"] / "research_side_moran_weight_sensitivity.csv"
    piv.round(4).to_csv(sp)
    lg.add_output(sp, rows=len(piv))
    spread = (piv.max(axis=1) - piv.min(axis=1))
    lg.count("max_moran_spread_across_matrices", round(float(spread.max()), 4))

    cp2 = P["integration"] / "country_research_side.parquet"
    g.drop(columns=["geometry"], errors="ignore").to_parquet(cp2, index=False)
    lg.add_output(cp2, rows=len(g))

    for _, r in G[G["weights"] == "knn_k6"].iterrows():
        lg.count(f"moran_{r['variable']}_knn_k6", r["morans_I"])
        lg.count(f"moran_p_{r['variable']}_knn_k6", r["p_sim"])

    q = W["queen"]
    lg.count("queen_islands_dropped", len(q.islands))

    print("\nGLOBAL MORAN'S I — RESEARCH SIDE")
    print(G[["variable", "weights", "n", "n_islands", "n_missing_filled",
             "morans_I", "z_sim", "p_sim"]].to_string(index=False))
    print("\nLOCAL MORAN (kNN k=6, 9,999 permutations), FDR-adjusted")
    print(L[["variable", "n_sig_raw", "n_sig_fdr", "HH_fdr", "LL_fdr", "HL_fdr",
             "LH_fdr"]].to_string(index=False))
    print("\nGETIS-ORD Gi*, FDR-adjusted")
    print(GT.to_string(index=False))
    print("\nMORAN'S I BY WEIGHTS MATRIX")
    print(piv.round(3).to_string())
    print(f"\nqueen contiguity would drop {len(q.islands)} countries with no land neighbour; "
          f"kNN k=6 retains all {W['knn_k6'].n}")
    lg.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
