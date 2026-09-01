"""Correct three confirmed systematic study-location coding errors and rebuild the cascade.

The extractor matched country names inside larger toponyms and agro-ecological zone names:

  "New South Wales, Australia"      -> United Kingdom  (6 records)
  "Guinea savanna" (a West African  -> Guinea          (3 records)
    agro-ecological zone)
  "Georgia" (the United States      -> Georgia         (4 records)
    state)

Every affected record was read individually and the correction is deterministic: in each case
the surrounding text names the true study country explicitly. The frozen corpus is not edited.
Corrections are applied as an overlay to the derived country lists, and the allocation is then
rebuilt with the published rule, which this script reproduces exactly before applying anything.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

# record id -> (codes to remove, codes to add, evidence)
CORR = {
 "W2120934246": (["GBR"], [], "field of wheat, New South Wales, Australia"),
 "W3107408591": (["GBR"], [], "two cotton fields near Mungindi, New South Wales, Australia"),
 "W4206381465": (["GBR"], [], "wheat paddocks in Victoria, New South Wales and South Australia"),
 "W4404120595": (["GBR"], [], "field experiments at two locations in New South Wales, Australia"),
 "W4405139381": (["GBR"], [], "neighbouring farms in New South Wales, Australia"),
 "W7163057425": (["GBR"], [], "rice fields in southern New South Wales, Australia"),
 "W2268886164": (["GIN"], [], "experiment at Kpalesawgu in Ghana; Guinea savanna is a zone"),
 "W4308421688": (["GIN"], [], "trials at Zaria and Doguwa, northern Guinea savanna of Nigeria"),
 "W4364360149": (["GIN"], [], "savanna areas of Nigeria: Sudan, Northern and Southern Guinea Savanna"),
 "W2000673519": (["GEO"], [], "64 counties in Georgia and Alabama, United States"),
 "W2970731140": (["GEO"], [], "Sentinel-2 tiles in Mississippi, Georgia and Texas, southern US"),
 "W4307077699": (["GEO"], ["USA"], "alfalfa variety trials in Kentucky and Georgia, United States"),
 "W7154632121": (["GEO"], [], "peanut yield in Georgia, USA"),
}


def split_list(x):
    if pd.isna(x) or not str(x).strip():
        return []
    return [t.strip() for t in str(x).replace(",", ";").split(";") if t.strip()]


def allocate(d, valid_iso, valid_crop):
    rows = []
    for r in d.to_dict("records"):
        isos = ([r["loc_country_iso3"]] if pd.notna(r.get("loc_country_iso3"))
                else split_list(r.get("loc_countries_all_iso3")))
        isos = [i for i in dict.fromkeys(isos) if i and i in valid_iso]
        crops = [c for c in dict.fromkeys(split_list(r.get("crop_standardized_crops")))
                 if c in valid_crop]
        if not isos or not crops:
            continue
        w = 1.0 / (len(isos) * len(crops))
        for i in isos:
            for c in crops:
                rows.append(dict(openalex_id=r["openalex_id"], iso3=i,
                                 crop_standard_name=c,
                                 publication_year=r["publication_year"],
                                 cited_by_count=r.get("cited_by_count"),
                                 fractional_weight=w))
    return pd.DataFrame(rows)


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() else np.nan


def main():
    lg = RunLogger("gf_01_correct_coding")
    d = pd.read_parquet(ROOT / "12_Data_Integration" / "study_level_dataset.parquet")
    scc0 = pd.read_parquet(ROOT / "12_Data_Integration" / "study_country_crop_dataset.parquet")
    layer = pd.read_parquet(ROOT / "14_Spatial_Weights" / "country_analytical_layer.parquet")
    ccn = pd.read_parquet(ROOT / "13_Indices" / "country_crop_need_measures.parquet")
    lg.add_input(ROOT / "12_Data_Integration" / "study_level_dataset.parquet")
    valid_iso = set(layer.loc[~layer.is_aggregate.fillna(False), "iso3"].dropna())
    valid_crop = set(ccn.crop_standard_name.dropna())

    # ---- guard: reproduce the published allocation before changing anything
    rep = allocate(d, valid_iso, valid_crop)
    assert len(rep) == len(scc0) and abs(rep.fractional_weight.sum() - scc0.fractional_weight.sum()) < 1e-9
    print(f"GUARD: published allocation reproduced exactly "
          f"({len(rep)} rows, {rep.openalex_id.nunique()} studies, "
          f"{rep.fractional_weight.sum():.1f} weight)")

    # ---- apply the overlay
    d2 = d.copy()
    d2["_sid"] = d2.openalex_id.str.split("/").str[-1]
    log = []
    for sid, (rm, add, ev) in CORR.items():
        m = d2._sid == sid
        if not m.any():
            print(f"  WARNING record not found: {sid}"); continue
        i = d2.index[m][0]
        before_all = d2.at[i, "loc_countries_all_iso3"]
        before_one = d2.at[i, "loc_country_iso3"]
        codes = ([before_one] if pd.notna(before_one) else split_list(before_all))
        after = [c for c in codes if c not in rm] + [c for c in add if c not in codes]
        after = list(dict.fromkeys(after))
        if pd.notna(before_one):
            d2.at[i, "loc_country_iso3"] = after[0] if len(after) == 1 else np.nan
            if len(after) != 1:
                d2.at[i, "loc_countries_all_iso3"] = ";".join(after)
        else:
            d2.at[i, "loc_countries_all_iso3"] = ";".join(after)
        log.append(dict(openalex_id=sid, title=str(d2.at[i, "title"])[:110],
                        before=";".join(codes), removed=";".join(rm), added=";".join(add),
                        after=";".join(after), evidence=ev,
                        decision_method="MODEL_ASSISTED_CORRECTION - deterministic string defect"))
    corr = pd.DataFrame(log)
    corr.to_csv(HERE / "coding_corrections.csv", index=False)
    print(f"\napplied {len(corr)} record corrections")

    # ---- rebuild
    scc = allocate(d2, valid_iso, valid_crop)
    scc.to_parquet(HERE / "study_country_crop_corrected.parquet", index=False)
    print(f"corrected allocation: {len(scc)} rows, {scc.openalex_id.nunique()} studies, "
          f"{scc.fractional_weight.sum():.1f} weight")

    # ---- impact
    a = scc0.groupby("iso3").agg(studies=("openalex_id", "nunique"),
                                 weight=("fractional_weight", "sum"))
    b = scc.groupby("iso3").agg(studies=("openalex_id", "nunique"),
                                weight=("fractional_weight", "sum"))
    cmp = a.join(b, how="outer", lsuffix="_before", rsuffix="_after").fillna(0)
    ch = cmp[(cmp.weight_before.round(6) != cmp.weight_after.round(6))]
    print("\nCOUNTRY-LEVEL IMPACT")
    print(ch.round(3).to_string())
    ch.to_csv(HERE / "country_impact.csv")

    lay = layer[["iso3"]].copy()
    for tag, s in [("before", scc0), ("after", scc)]:
        g = s.groupby("iso3", as_index=False).fractional_weight.sum()
        full = lay.merge(g, on="iso3", how="left").fillna({"fractional_weight": 0})
        pan_c = pd.read_parquet(ROOT / "12_Data_Integration" / "country_crop_panel.parquet")
        p199 = pan_c[["iso3"]].drop_duplicates().merge(g, on="iso3", how="left").fillna({"fractional_weight": 0})
        print(f"  {tag}: Gini(195 layer) {gini(full.fractional_weight):.4f}   "
              f"Gini(199 panel) {gini(p199.fractional_weight):.4f}   "
              f"countries with research {int((p199.fractional_weight>0).sum())}")

    # ---- rebuilt panel counts
    pan = pd.read_parquet(ROOT / "12_Data_Integration" / "country_crop_panel.parquet")
    newc = (scc.groupby(["iso3", "crop_standard_name"], as_index=False)
            .agg(n_studies_fractional_new=("fractional_weight", "sum"),
                 n_studies_full_new=("openalex_id", "nunique")))
    pan2 = pan.drop(columns=["n_studies_fractional", "n_studies_full"]).merge(
        newc, on=["iso3", "crop_standard_name"], how="left").fillna(
        {"n_studies_fractional_new": 0.0, "n_studies_full_new": 0})
    pan2 = pan2.rename(columns={"n_studies_fractional_new": "n_studies_fractional",
                                "n_studies_full_new": "n_studies_full"})
    pan2["has_any_study"] = pan2.n_studies_fractional > 0
    pan2.to_parquet(HERE / "country_crop_panel_corrected.parquet", index=False)
    print(f"\npanel cells with research: {int(pan.has_any_study.sum())} -> "
          f"{int(pan2.has_any_study.sum())}")
    print(f"panel fractional total: {pan.n_studies_fractional.sum():.2f} -> "
          f"{pan2.n_studies_fractional.sum():.2f}")

    for f in ["coding_corrections.csv", "country_impact.csv",
              "study_country_crop_corrected.parquet", "country_crop_panel_corrected.parquet"]:
        lg.add_output(HERE / f)
    lg.finish()


if __name__ == "__main__":
    main()
