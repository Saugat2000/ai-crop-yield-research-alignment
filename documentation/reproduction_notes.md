# Reproduction notes

## What runs, and what it needs

Every script resolves its paths from `01_Project_Management/project_config.py`, which sets
`PROJECT_ROOT` from its own location. Nothing needs configuring and no script contains an
absolute path. Clone the repository anywhere and run from the repository root.

| Script | Needs | Produces |
|---|---|---|
| `21_Figures/export_plotted_data.py` | included files only | the plotted data behind each figure; asserts 18 values against `headline_numbers.json` |
| `15_Descriptive_Analysis/research_side_descriptives.py` | included files only | §5.1–5.2 descriptives |
| `17_Spatial_Econometrics/research_side_spatial_analysis.py` | included files only | global and local Moran, Getis–Ord, LISA (9,999 permutations, fixed seed) |
| `20_Robustness/run_robustness.py` | included files only | concentration and model robustness suites |
| `16_Econometrics/run_models.py` | included files only | first-release model estimates |
| `outputs/geofix/gf_02_cascade.py` → `gf_05_appendix_tables.py` | included files only | **the reported results** |
| `outputs/geofix/gf_01_correct_coding.py` | included files only | the corrected allocation |
| `outputs/revision/rev_01_models.py` … `rev_10_tables.py` | included files only | the reviewer-pass analyses |
| `outputs/graphical_abstract/ga_01_build.py` | included files only | the graphical abstract |

The two external extracts these scripts read — `10_External_Data/World_Bank/wb_indicators_long.csv`
and `10_External_Data/FAOSTAT/faostat_country_crop_year.parquet` — are included, so no download is
needed to reproduce any reported number.

## What is not reproducible from this repository

**Corpus construction upstream of screening.** The OpenAlex harvest, deduplication, and
full-text retrieval ran against live APIs on 2026-07-30 and are not rerunnable to the same result:
OpenAlex is updated continuously, so a rerun today returns a different record set. The repository
starts from the frozen output of that stage —
`06_Screening/s3a_screening_decisions.parquet`, 43,543 screened works, with a SHA-256 manifest in
`06_Screening/eligible_corpus_manifest.json`. Every downstream number follows from it.

**Study-level coding.** Location and crop coding was model-assisted first-pass extraction over
full texts, some of which are behind paywalls and cannot be redistributed. The coded values, the
verbatim evidence text for each, the coder, the confidence, and the verification status are all in
`12_Data_Integration/study_level_dataset.parquet`, so the coding can be audited without rerunning it.

**Colour-vision check on the submitted PDFs.** `outputs/revision_final/rf_03_cvd_check.py` reads
figure PDFs. It falls back to `21_Figures/` when the submission folder is absent, which is the
normal case here.

## Numerical reproducibility

The permutation seed is fixed in `project_config.py` (`RANDOM_SEED = 20260730`), so the spatial
p-values reproduce exactly. Model estimates are deterministic given the estimation sample.
Every script writes a run log to `29_Logs/` recording start and end time, input files with
SHA-256 checksums, output files, row counts, package versions, and the seed.

## Known inconsistencies between the manuscript and these outputs

Three values in the manuscript body were not updated when the study-location coding correction
was applied. They are recorded here rather than silently changed, because the compiled PDF in
`manuscript/` is the version that was submitted.

| Manuscript location | Printed | Value in `headline_numbers.json` |
|---|---|---|
| §Analytical Sample (`.tex` lines 418, 422) | 485 studied cells, 2,131 empty | 483 and 2,133 |
| Table 5 note (`.tex` line 764) | 447 studied cells, 1,352 unstudied | 446 and 1,353 |
| Appendix (`.tex` line 1306) | 122 countries, 485 cells | 120 and 483 |

The abstract, §5.2, and every table and figure carry the corrected values (483, 2,133, 446, 120).
Derived shares are unaffected: 483/2,616 still rounds to 18.5%, and 483/2,507 to 19.3%.
