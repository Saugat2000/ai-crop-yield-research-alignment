# Data dictionary

Every dataset is Parquet unless noted. Column names are given for the variables the reported
analysis reads; the panel files carry additional descriptive columns not used in any model.

## Unit of analysis

The unit is the **country-crop system**: one country paired with one crop. The panel is
199 countries × 25 crops, restricted to systems FAOSTAT reports, giving **2,616** cells. A cell
with no eligible study is a real zero, not a missing value, and stays distinguishable from a cell
whose external data are unavailable.

---

## `12_Data_Integration/study_level_dataset.parquet` — 7,045 rows, one per eligible study

| Column | Meaning |
|---|---|
| `openalex_id` | OpenAlex work identifier; the join key to the bibliographic source |
| `doi`, `title`, `publication_year`, `type`, `venue_name` | Bibliographic metadata as returned by OpenAlex |
| `cited_by_count` | OpenAlex citation count at the freeze date, 2026-07-30 |
| `loc_study_scope` | Coded scope of the empirical study area: single-country, multi-country, regional, global, unresolved |
| `loc_country_iso3`, `loc_countries_all_iso3` | Countries the study empirically examines, ISO 3166-1 alpha-3 |
| `loc_source_section`, `loc_evidence_text` | Where in the document the location was read, and the verbatim sentence supporting it |
| `loc_confidence`, `loc_verification`, `loc_coded_by` | Coder confidence, verification status, and coder identity for the location value |
| `crop_standardized_crops`, `crop_faostat_items`, `crop_in_faostat` | Crops examined, mapped to FAOSTAT items |
| `crop_evidence_text`, `crop_confidence`, `crop_verification` | The same evidence trail for the crop value |
| `institution_countries`, `first_author_country` | Affiliation countries, used **only** for the local-leadership measure |

Study location is coded from the study-area, data, and methods sections. It is never inferred
from author affiliation. Local leadership uses institutional affiliation country only.

## `12_Data_Integration/study_country_crop_dataset.parquet` — 2,966 rows

The fractional allocation. One row per (study, country, crop) triple, for the **2,031** studies
that carry accepted country evidence and a FAOSTAT-matched crop.

| Column | Meaning |
|---|---|
| `fractional_weight` | 1 / (`n_countries_in_study` × `n_crops_in_study`); the weights of one study sum to 1 |
| `n_countries_in_study`, `n_crops_in_study` | Denominators of that weight |
| `location_confidence`, `location_cue_type`, `crop_confidence` | Carried through from the study-level coding |

Fractional counting is primary throughout. Full counting (`n_studies_full`) appears only in
descriptive comparisons and is always labelled.

**Corrected version: `outputs/geofix/study_country_crop_corrected.parquet`** — 2,954 rows, after
the three systematic location-coding repairs. Use this one.

## `12_Data_Integration/country_crop_panel.parquet` — 2,616 rows, the analytical panel

| Column | Meaning |
|---|---|
| `iso3`, `crop_standard_name` | The cell identifier |
| `area_ha_mean`, `production_t_mean`, `yield_t_ha_mean` | FAOSTAT means over the reference years |
| `n_ref_years` | Number of FAOSTAT years behind those means |
| `area_share_global`, `production_share_global` | Cell share of world harvested area and production |
| `yield_volatility_cv` | Coefficient of variation of yield over the reference window |
| `n_studies_fractional` | Research attention, fractional counting — the outcome |
| `n_studies_full` | Full counting, descriptive comparison only |
| `has_any_study` | Participation outcome. In the corrected panel 483 cells are 1 and 2,133 are 0 |
| `need_rank_pct`, `need_equal_pct`, `need_pca_pct`, `need_entropy_pct` | Research-need percentile under each aggregation rule |
| `wb_region`, `wb_income_group` | World Bank classifications |

109 of the 2,616 cells carry no FAOSTAT record at all (`n_ref_years` = 0; both `area_ha_mean` and
`production_t_mean` null). A further 3 have production but no harvested area, so `area_ha_mean` is
null in 112 cells; those drop out of any area-scaled statistic. 2,343 cells have harvested area
above zero, and 1,866 of those carry no eligible study.

**Corrected version: `outputs/geofix/country_crop_panel_corrected.parquet`** — same 2,616 rows,
regenerated attention columns. Use this one.

## `16_Econometrics/estimation_sample.parquet` — 1,799 rows

The panel joined to World Bank capacity indicators and reduced to cells with complete covariates.
Regression variables:

| Column | Meaning |
|---|---|
| `studied` | 1 if the cell carries at least one eligible study, else 0. **The stored column is the pre-correction value and marks 447 cells.** The reported models rebuild it from `outputs/geofix/country_crop_panel_corrected.parquet`, which gives **446**; that is the number in `headline_numbers.json` and in the manuscript's Table 5 note. |
| `log_area` | Natural log of `area_ha_mean` |
| `log_production`, `log_gdp_pc`, `log_population` | Natural logs of the corresponding levels |
| `rd` | R&D expenditure as a percentage of GDP, World Bank `GB.XPD.RSDV.GD.ZS`, left on its own scale (0.006 to 6.35 in sample) |
| `tertiary` | Tertiary gross enrolment ratio, `SE.TER.ENRR`, divided by 100 (can exceed 1) |
| `internet` | Internet users as a share of population, `IT.NET.USER.ZS`, divided by 100 |
| `need` | Research-need index, nine components, rank aggregation, on 0–1 |
| `*_year` | The reference year of each World Bank indicator for that country |

Other World Bank codes read by `run_models.py`: `NY.GDP.PCAP.PP.KD` (GDP per capita, PPP,
constant), `SP.POP.TOTL` (population), `SL.AGR.EMPL.ZS` (agricultural employment share). For each
country the most recent non-missing value is taken and its year kept alongside it.

The intensity models drop 7 further cells with zero exposure, giving n = 1,792.

## `13_Indices/country_need_indices.parquet` — 199 rows

The eleven-component index. Components: undernourishment, moderate-or-severe food insecurity,
cereal-import dependency, dietary-energy adequacy, agricultural employment share, agricultural
value added as a share of GDP, agricultural value added per worker, warming since baseline,
area-weighted yield volatility, and the two agricultural-scale shares (`crop_area_share_global`,
`crop_production_share_global`).

`need_rank` is primary; `need_equal`, `need_pca`, `need_entropy` are the alternative
standardisations. `n_components_observed` records the coverage behind each country's value.

**The manuscript's primary index excludes the two agricultural-scale components**, so that need
is not partly a restatement of scale:

- `31_Presubmission_Audit/need_index_scale_excluded.parquet` — the nine-component index
- `outputs/revision/need_index_corrected_floor.parquet` — the same index under the corrected
  coverage floor (`need9_floor9`, `obs9`), which indexes **188** countries. **This is the
  version used in the reported models.**

## `14_Spatial_Weights/`

| File | Contents |
|---|---|
| `country_analytical_layer.parquet` | 195-country boundary layer, Natural Earth admin-0, EPSG:4326 |
| `spatial_weights.pkl` | Five row-standardised weight matrices: kNN k=6 (primary), k=4, k=8, queen contiguity, distance band |

Islands and disconnected components are handled by the k-nearest-neighbour construction, which
gives every country exactly k neighbours; the queen-contiguity variant is reported alongside so
the effect of that choice is visible.

## `06_Screening/s3a_screening_decisions.parquet` — 43,543 rows

`openalex_id`, `s3a_decision`, `s3a_reason` — the screening outcome for every deduplicated work.
These are the three columns the analysis reads. The project's full bookkeeping worksheet carries
further coder columns and is available from the authors.

## External source data included here

| File | Source | Licence |
|---|---|---|
| `10_External_Data/FAOSTAT/faostat_country_crop_year.parquet` | FAOSTAT QCL (Production, Crops and Livestock), normalised | CC BY 4.0 |
| `10_External_Data/World_Bank/wb_indicators_long.csv` | World Bank Open Data indicator API, long format | CC BY 4.0 |

Both were downloaded 2026-07-30 and are redistributed here under those licences with
attribution. The larger FAOSTAT bulk archives are not redistributed: food security (FS),
macro-statistics key indicators (MK), employment indicators in agriculture (OEA), and temperature
change (ET). Each is a normalised "All Data" archive downloadable from
<https://bulks-faostat.fao.org/production/> or the domain pages at
<https://www.fao.org/faostat/en/#data>. They are not needed to reproduce any reported number: the
country-level values they feed are already in `13_Indices/country_need_indices.parquet`, and the
nine-component index used in the reported models is in
`outputs/revision/need_index_corrected_floor.parquet`. World Bank indicators can be re-pulled from
<https://api.worldbank.org/v2/country/all/indicator/><code>?format=json for the six codes listed
above.
