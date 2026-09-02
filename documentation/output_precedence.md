# Which output file is authoritative

The reported results were revised four times after the first replication release. Each revision
kept its own output folder so the record of what changed stays readable. Only the **last** stage
is reported in the manuscript. A file from an earlier stage is a record of that stage, not a
current result.

Read this table before quoting any number from `outputs/`.

| Stage | Folder | Date | What it changed | Status |
|---|---|---|---|---|
| 1 | `15_…`–`22_…` phase folders | 2026-08-28 | First replication release | Superseded for every quantity revised below |
| 2 | `31_Presubmission_Audit/` | 2026-08-29 | Need index rebuilt without the two agricultural-scale components; observed-sample spatial statistics; crop fixed effects | Superseded, except `need_index_scale_excluded.parquet` and `sample_flow.csv` |
| 3 | `outputs/revision/` | 2026-08-31 | Reviewer passes: coding validation sample, PPML offset test, need-index coverage floor, geodesic weights, panel accounting | Current for validation rates, `temporal_gini.csv`, `need_component_corr_*.csv`, `need_index_corrected_floor.parquet` |
| 4 | `outputs/revision_final/` | 2026-09-01 | Appendix tables A8–A10, colour-vision check, Figure A4 | Current for those exhibits |
| 5 | `outputs/needfix/` | 2026-09-01 | Nine-component coverage floor adopted as primary | **Superseded by stage 6** |
| 6 | `outputs/geofix/` | 2026-09-01 | Three systematic study-location coding errors repaired; every downstream result regenerated | **Authoritative. Every headline number in the manuscript comes from here.** |

## The single verification anchor

`outputs/geofix/headline_numbers.json` holds every headline quantity in the manuscript as a
machine-readable record. `21_Figures/export_plotted_data.py` and
`outputs/graphical_abstract/ga_01_build.py` both assert against it and refuse to write on
disagreement. If you want to check one number, check it there.

## Two files that disagree between stages

Stage 5 and stage 6 both produced a LISA classification. They differ:

| File | High-High | Low-Low | High-Low | Low-High | Status |
|---|---|---|---|---|---|
| `outputs/needfix/gap9_corrected_lisa.csv` | 18 | 6 | 1 | 2 | superseded |
| `outputs/geofix/gap_corrected_lisa.csv` | 18 | 6 | **2** | 2 | **reported** |

The manuscript and `headline_numbers.json` report the stage-6 values. The same applies to the
model estimates: use `outputs/geofix/model_estimates_geofix.csv`, not
`outputs/needfix/model_estimates_corrected.csv`.
