# Where Is AI-Based Crop-Yield Research Done? A Study-Location Analysis of 2,616 Country-Crop Systems

Replication data, analysis code, and the submitted manuscript for a study of where AI-based
crop-yield research is actually conducted.

- **Prasamsha Poudel** — Agriculture and Forestry University, Nepal
- **Saugat Khanal** — Department of Agricultural Economics, Oklahoma State University, USA

Both authors contributed equally.

## Overview

Machine learning now dominates crop-yield prediction, but the geography of that evidence has not
been measured. Bibliometric studies record where authors work. This study records where the
agriculture being studied is located.

The unit of analysis is the **country-crop system**, not the article. Screening 43,543
deduplicated works gave 7,045 eligible AI/ML crop-yield studies published from 2000 to 2026. Each
was assigned to the countries and crops it empirically examines, read from its own reported study
area rather than from author affiliation. The 2,031 studies carrying accepted country evidence and
a FAOSTAT-matched crop were allocated fractionally across 2,616 country-crop systems spanning 199
countries and 25 crops. That attention was then compared with harvested area, a nine-component
research-need index that excludes agricultural scale, and national scientific-capacity indicators,
using participation and count regressions together with global and local spatial statistics.

What the analysis finds: 120 of 199 countries carry at least one eligible study and 483 of the
2,616 systems carry any research, with a Gini coefficient of country research counts of 0.849.
The 1,866 unstudied systems that do have cropland hold 19.1% of the panel's harvested area.
Harvested area has the largest association with whether a system is studied. Attention scales
sublinearly with area. Research and development expenditure predicts both participation and
intensity; the association with measured research need is not identified.

## Citation

The manuscript has been **prepared for and submitted to *Advances in Agriculture* (Wiley)**. It is
not yet accepted or published, so there is no DOI, volume, issue, or article number. Until then,
cite this repository — see `CITATION.cff`.

## Repository contents

| Folder | Contents |
|---|---|
| `manuscript/` | The compiled manuscript as submitted (PDF, 26 pages), its LaTeX source, the bibliography, and compilation instructions |
| `documentation/` | Data dictionary, output precedence, reproduction notes. **Read `output_precedence.md` before quoting any number from `outputs/`** |
| `01_Project_Management/` | `project_config.py` — paths, constants, seed, run logging. Every script imports it; paths resolve relative to the repository, so nothing needs configuring |
| `06_Screening/` | Screening decisions for all 43,543 deduplicated works, the decision summary, and the frozen-corpus SHA-256 manifest |
| `10_External_Data/` | The FAOSTAT crop extract and World Bank indicator extract the models read (both CC BY 4.0, redistributed with attribution) |
| `12_Data_Integration/` | The core datasets: study-level (7,045 rows), fractional allocation, and the 2,616-cell analytical panel |
| `13_Indices/` | Research-need indices and the underlying country-crop need measures |
| `14_Spatial_Weights/` | The 195-country boundary layer and five row-standardised weight matrices (kNN k=6 primary; k=4, k=8, queen, distance band) |
| `15_Descriptive_Analysis/` | Descriptives script and outputs |
| `16_Econometrics/` | Model scripts, the estimation sample, and first-release estimates |
| `17_Spatial_Econometrics/` | Global and local Moran, Getis–Ord, and LISA |
| `20_Robustness/` | Robustness suite |
| `21_Figures/` | The ten submitted figure PDFs, the plotted data behind each, the export script, and the figure manifest |
| `22_Tables/` | Machine-readable versions of the manuscript tables |
| `29_Logs/` | Run logs; new runs write here |
| `31_Presubmission_Audit/` | Pre-submission audit scripts and outputs (2026-08-29) |
| `outputs/` | The four revision stages. `outputs/geofix/` is authoritative |

## Data

### Included here

Derived datasets — the coded study-level dataset, the fractional allocation, the country-crop
panel, the need indices, the estimation sample, the spatial weights, and every model, spatial, and
robustness output — are released under CC BY 4.0.

Two source extracts are included because their licences permit redistribution with attribution:

| File | Source | Downloaded | Licence |
|---|---|---|---|
| `10_External_Data/FAOSTAT/faostat_country_crop_year.parquet` | FAOSTAT, Production: Crops and Livestock (QCL) | 2026-07-30 | CC BY 4.0 |
| `10_External_Data/World_Bank/wb_indicators_long.csv` | World Bank Open Data indicator API | 2026-07-30 | CC BY 4.0 |

With these, every reported number can be reproduced from this repository alone.

### Not included, and how to get it

| Not included | Why | How to reconstruct |
|---|---|---|
| The raw OpenAlex harvest | Continuously updated; a rerun today returns a different record set | The frozen result is `06_Screening/s3a_screening_decisions.parquet` (43,543 works with `openalex_id`), checksummed in `06_Screening/eligible_corpus_manifest.json`. Each work is retrievable from `https://api.openalex.org/works/<openalex_id>` |
| Full texts of the coded studies | Copyright; many are paywalled | Not needed. The coded values, the verbatim evidence sentence for each, the coder, the confidence, and the verification status are all in `12_Data_Integration/study_level_dataset.parquet` |
| FAOSTAT bulk archives for food security, macro-statistics, employment, and temperature change | Large; not needed downstream | The country-level values they produce are already in `13_Indices/country_need_indices.parquet`. Re-download from <https://www.fao.org/faostat/en/#data> |
| Wiley LaTeX class and style files | Redistribution not permitted by the template licence | See `manuscript/README.md` |

Source data are used under their own licences and remain the property of their providers. No
redistribution right is claimed over anything beyond the two CC BY 4.0 extracts named above.

## Reproduction

```bash
git clone https://github.com/Saugat2000/ai-crop-yield-research-alignment.git
cd ai-crop-yield-research-alignment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then, from the repository root:

**1. Verify the analytical data and the headline results.** This checks 18 reported quantities
against `outputs/geofix/headline_numbers.json` and exits non-zero on any disagreement.

```bash
python 21_Figures/export_plotted_data.py
```

**2. Reproduce the reported models.** `outputs/geofix/` is the authoritative stage; these
rebuild it end to end.

```bash
python outputs/geofix/gf_01_correct_coding.py     # corrected study-location allocation
python outputs/geofix/gf_02_cascade.py            # panel, need index, models, Gini, Moran, LISA
python outputs/geofix/gf_04_robustness.py         # robustness suite
python outputs/geofix/gf_05_appendix_tables.py    # appendix tables
```

**3. Reproduce the spatial results.**

```bash
python 17_Spatial_Econometrics/research_side_spatial_analysis.py   # 9,999 permutations, fixed seed
```

**4. Reproduce the figures.**

```bash
python outputs/geofix/gf_03_figures.py            # Figures 1-5 and A3
python outputs/revision_final/rf_02_figures.py    # Figure A4
python outputs/revision/rev_08_figures.py         # Figures A5 and A6
```

**5. Reproduce the descriptives, first-release models, and tables.**

```bash
python 15_Descriptive_Analysis/research_side_descriptives.py
python 16_Econometrics/run_models.py
python 20_Robustness/run_robustness.py
python outputs/revision/rev_10_tables.py
```

Every script writes a run log to `29_Logs/` with input checksums, output row counts, package
versions, and the random seed. Paths are relative throughout; no script contains an absolute path.

## Software

Python 3.9.6 produced the reported results. No R is used.

```
esda==2.5.1        geopandas==1.0.1    libpysal==4.8.1    matplotlib==3.9.4
numpy==1.26.4      pandas==2.3.3       pyarrow==21.0.0    scipy==1.12.0
shapely==2.0.7     statsmodels==0.14.6
```

Pinned in `requirements.txt`. `geopandas` needs GEOS and PROJ, which its wheels bundle.

## Main manuscript outputs

| Manuscript exhibit | File | Produced by |
|---|---|---|
| Headline values throughout | `outputs/geofix/headline_numbers.json` | `outputs/geofix/gf_02_cascade.py` |
| Table 1, corpus construction | `22_Tables/tab_01_corpus_construction.csv` | `21_Figures/make_manuscript1_figures.py` |
| Main regression table | `outputs/geofix/model_estimates_geofix.csv`, `outputs/geofix/tables/tab6_regression.tex` | `outputs/geofix/gf_02_cascade.py` |
| Marginal effects | `outputs/geofix/ame_geofix.csv`, `outputs/geofix/tables/tab7_ame.tex` | `outputs/geofix/gf_02_cascade.py` |
| Main robustness table | `outputs/geofix/robustness_ranges.csv`, `outputs/geofix/tables/tab8_robust.tex` | `outputs/geofix/gf_04_robustness.py` |
| Concentration and Gini | `outputs/geofix/gini_variants.csv` | `outputs/geofix/gf_02_cascade.py` |
| Spatial results, Moran and LISA | `outputs/geofix/spatial_variants.csv`, `outputs/geofix/gap_corrected_lisa.csv` | `outputs/geofix/gf_02_cascade.py` |
| Figures 1–6, A3–A6 | `21_Figures/fig_*.pdf` with `*_data.csv` sidecars | see `21_Figures/manuscript1_figure_manifest.csv` |
| Appendix tables A2–A10 | `outputs/geofix/tables/`, `outputs/revision/tables/`, `outputs/revision_final/tables/` | `gf_05`, `rev_10`, `rf_01` |
| Coding validation rates | `outputs/revision/validation_headline_rates.csv` | `outputs/revision/rev_09_validation_rates.py` |
| PPML elasticity test | `outputs/revision/ppml_area_elasticity_test.csv` | `outputs/revision/rev_01_models.py` |

## Data availability

The bibliographic corpus was assembled from the public OpenAlex database. Agricultural production
and harvested-area data are from FAOSTAT, research-need components from FAOSTAT food security,
macroeconomic, employment, and temperature domains, scientific-capacity indicators from World Bank
Open Data, and country boundaries from Natural Earth. All sources are publicly available. The
coded study-level dataset, the derived country-crop panel, the analysis code, and the outputs
behind every table and figure are available in this repository.

## Sources and licences

| Source | Used for | Licence |
|---|---|---|
| [OpenAlex](https://openalex.org) | Bibliographic corpus | CC0 |
| [FAOSTAT](https://www.fao.org/faostat/) | Production, harvested area, need components | CC BY 4.0 |
| [World Bank Open Data](https://data.worldbank.org) | Capacity indicators | CC BY 4.0 |
| [Natural Earth](https://www.naturalearthdata.com) | Country boundaries, admin-0 | Public domain |

Code in this repository is under the MIT Licence (`LICENSE`); derived datasets under CC BY 4.0.

## A note on coding and its limits

Study location is coded from each study's study-area, data, and methods sections, never from
author affiliation. Local leadership uses institutional affiliation country only. One researcher
did the coding, so no inter-rater statistic between two people is reported; reliability evidence
comes from a stratified re-verification audit of 210 records, whose rates are in
`outputs/revision/validation_headline_rates.csv`. Model-assisted extraction was a first pass
requiring human confirmation, and is labelled as such wherever it appears.
