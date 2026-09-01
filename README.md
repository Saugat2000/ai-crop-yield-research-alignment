# Global Geography of AI-Based Crop-Yield Research: Agricultural Scale, Research Need, and Scientific Capacity

Replication data and code for the manuscript submitted to *Advances in Agriculture* (Wiley).

**Authors:** Prasamsha Poudel (Agriculture and Forestry University, Nepal) and Saugat Khanal
(Department of Agricultural Economics, Oklahoma State University, USA). Both authors
contributed equally.

## What this repository contains

The study codes the empirical location of 7,045 eligible AI/ML crop-yield studies
(2000–2026) from what each study reports about its own study area, allocates them
fractionally over 2,616 country-crop systems (199 countries × 25 crops), and compares the
resulting research attention with agricultural importance (FAOSTAT production and harvested
area), an eleven-component research-need index, and national scientific-capacity indicators,
using participation and count regressions and global and local spatial statistics.

This repository holds the derived datasets, the analysis code exactly as run for the
reported results, and the outputs behind every table and figure in the manuscript.

## Layout

| Folder | Contents |
|---|---|
| `01_Project_Management/` | `project_config.py` — paths, constants, random seed, run logging (all scripts import it; paths resolve relative to the repository, so nothing needs configuring) |
| `06_Screening/` | Screening decisions for all 43,543 deduplicated works (`s3a_screening_decisions.parquet`: `openalex_id`, `s3a_decision`, `s3a_reason` — the columns the analysis reads), decision summary, frozen-corpus manifest (SHA-256) |
| `12_Data_Integration/` | **The core datasets.** `study_country_crop_dataset.parquet` (study → country-crop fractional allocations), `country_crop_panel.parquet` (the 2,616-cell analytical panel), `study_level_dataset.parquet` (one row per eligible study) |
| `13_Indices/` | `country_need_indices.parquet` — the research-need indices (rank aggregation primary; equal, entropy, PCA alternatives) |
| `14_Spatial_Weights/` | `country_analytical_layer.parquet` (195-country boundary layer, Natural Earth admin-0) and `spatial_weights.pkl` (the five row-standardised weight matrices: kNN k=6 primary; k=4, k=8, queen, distance band) |
| `15_Descriptive_Analysis/` | Script + outputs for Section 5.1–5.2 descriptives |
| `16_Econometrics/` | Scripts (`run_models.py`, `model_scaffolds.py`), the estimation sample, and model outputs for Section 5.4 (Table 5) |
| `17_Spatial_Econometrics/` | Script + global/local Moran, Getis–Ord, and LISA outputs for Section 5.5 (Table 6 spatial rows, Figure 4) |
| `20_Robustness/` | Script + the robustness suite outputs for Section 5.6 (Table 6) |
| `21_Figures/` | Figure script, the six manuscript figures (PDF), and the plotted data + caption sidecar for each figure |
| `22_Tables/` | Machine-readable versions of the manuscript tables |
| `29_Logs/` | The run logs the figure script reads for the corpus-construction table; new runs write their logs here |

## Reproducing the results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python 15_Descriptive_Analysis/research_side_descriptives.py   # Section 5.1-5.2
python 17_Spatial_Econometrics/research_side_spatial_analysis.py  # Section 5.5 (9,999 permutations, fixed seed)
python 16_Econometrics/run_models.py                           # Section 5.4  (see note below)
python 20_Robustness/run_robustness.py                         # Section 5.6
python 21_Figures/make_manuscript1_figures.py                  # Figures 1-4, A1-A2 + table CSVs
```

Every script logs inputs (with checksums), outputs, row counts, and the random seed to
`29_Logs/`. The permutation seed is fixed in `project_config.py`, so spatial p-values
reproduce exactly.

**Note on `run_models.py`:** the script rebuilds its estimation sample from a raw World Bank
extract (`10_External_Data/World_Bank/wb_indicators_long.csv`) that is not included here;
the World Bank indicator codes are listed at the top of the script and the extract can be
re-downloaded from the public World Bank API. The derived `16_Econometrics/estimation_sample.parquet`
**is** included, and every downstream result (robustness, spatial, figures) runs from the
included files without it.

## Script → manuscript mapping

| Script | Main outputs | Manuscript exhibit |
|---|---|---|
| `research_side_descriptives.py` | `research_side_descriptives.json`, most-under/over-researched cells | §5.1–5.3 values |
| `run_models.py` | `model_coefficients.csv`, `model_diagnostics.json` | Table 5, Figure A2 |
| `research_side_spatial_analysis.py` | `research_side_global_moran.csv`, `..._lisa_clusters.csv`, local/Getis summaries | §5.5, Figure 4 |
| `run_robustness.py` | `concentration_robustness.csv`, `model_robustness.csv`, `finding_stability_classification.csv` | Table 6 |
| `make_manuscript1_figures.py` | `fig_01`–`fig_06` (+ data/caption sidecars), `tab_01`–`tab_04` CSVs | Figures 1–4, A1–A2; Tables 1, 3, 5, 6 inputs |

## Pre-submission audit (2026-08-29): revised primary specifications

`31_Presubmission_Audit/` holds the scripts and outputs of the pre-submission audit that
revised the reported specifications. In the submitted manuscript:

- the **primary research-need index excludes the two agricultural-scale components**
  (share of global harvested area and production); `audit_need_index.py` rebuilds it and
  `need_index_scale_excluded.parquet` stores it. The original eleven-component index
  (`13_Indices/`) is the sensitivity variant.
- the **primary regression estimates** are `table5_need9_primary.csv`: the scale-excluded index, no standalone yield-volatility
  regressor. Crop and region fixed-effect, country-level, allocation-scope, and
  coverage-threshold variants are in `model_audit_results.csv`.
- the **primary spatial statistics** are observed-sample results with weights rebuilt on
  countries whose mismatch value is observed (`spatial_audit_global.csv`,
  `gap9_lisa_clusters.csv`); the earlier mean-filled outputs in `17_Spatial_Econometrics/`
  are superseded for the mismatch variable.
- cross-crop comparisons use **harvested area**; production is compared within crops
  (`within_crop_benchmarks.csv`).
- figure files `21_Figures/fig_0*_v2*.pdf` are the submitted versions.

## Pre-submission revision (2026-08-31): reviewer-driven analyses

`outputs/revision/` holds the scripts and outputs of a second pre-submission audit conducted
as two skeptical reviewer passes. The reports are `Reviewer_1_PreSubmission_Report.md`,
`Reviewer_2_PreSubmission_Report.md`, `Additional_Analysis_Audit.md`, and
`PreSubmission_Action_Log.md` in the manuscript folder. Findings that changed the manuscript:

- **Study-location and crop coding were validated for the first time** (`rev_05`, `rev_09`).
  A 210-record stratified probability sample was adjudicated by model-assisted review, which
  is a first pass requiring human confirmation and is labelled as such everywhere. Country
  allocation is correct for 90.4% of adjudicated records (95% CI 84.3-94.3); 62.0% of sampled
  unresolved records name a study area the coder did not recover.
- **The PPML exposure offset was tested and rejected** (`rev_01`): the harvested-area
  elasticity is 0.849 (95% CI 0.754-0.944), and unit elasticity is rejected at p = 0.002.
  Under the free specification the research-need coefficient is no longer distinguishable
  from zero, so the manuscript no longer reports a positive need association.
- **The need-index coverage floor was corrected** (`rev_03`): the published floor counted all
  eleven recorded indicators, including the two never-missing agricultural-scale shares.
  Applied to the nine primary components it indexes 188 rather than 193 countries; results
  are unchanged.
- **Spatial weights were rebuilt on great-circle distance** (`rev_04`): neighbour sets change
  for 86 of 195 countries and the clustering result is unchanged.
- **The panel universe is now stated exactly** (`rev_06`): 2,590 FAOSTAT cells, plus 30 for
  four territories FAOSTAT reports inside France, minus 4, giving 2,616; 109 cells carry no
  FAOSTAT denominator.
- Crop fixed effects, an expanded sample without R&D, nested block models, average marginal
  effects, crop-specific and period-specific models, quasi-Poisson comparison, rank-tie
  sensitivity, and LM spatial diagnostics are all reported (`rev_01`, `rev_02`, `rev_04`).

The title was changed to the neutral form because the nested block comparison supports a
capacity-over-need ordering for participation but separates neither block for intensity.

## Data sources and licences

| Source | Used for | Licence |
|---|---|---|
| [OpenAlex](https://openalex.org) | Bibliographic corpus | CC0 |
| [FAOSTAT](https://www.fao.org/faostat/) | Production, harvested area, need components | CC BY 4.0 |
| [World Bank Open Data](https://data.worldbank.org) | Capacity indicators | CC BY 4.0 |
| [Natural Earth](https://www.naturalearthdata.com) | Country boundaries (admin-0) | Public domain |

Derived datasets in this repository are released under CC BY 4.0; code under the MIT
Licence (see `LICENSE`). The screening worksheet here carries the three columns used by the
analysis; the project's full bookkeeping worksheet is available from the authors.

## Citation

See `CITATION.cff`. Please cite the manuscript once published; until then, cite this
repository.
