# Manuscript

| File | What it is |
|---|---|
| `Poudel_Khanal_AI_Crop_Yield_Research_Manuscript.pdf` | The compiled manuscript as submitted, 26 pages |
| `Poudel_Khanal_AIA_manuscript.tex` | Its LaTeX source |
| `manuscript1_wiley.bib` | The bibliography |
| `Poudel_Khanal_AIA_manuscript.bbl` | The formatted bibliography, so the document compiles without the journal's `.bst` |

## Compiling

The source is written against the journal's own LaTeX template. Four template files are **not**
redistributed here, because Wiley's template licence does not permit it:

`USG.cls`, `NJDnatbib.sty`, `lettersp.sty`, `wileyNJD-Chicago.bst`, and the `images/` logo set

Download them from the *Advances in Agriculture* author guidelines page
(<https://onlinelibrary.wiley.com/page/journal/9403/homepage/author-guidelines>) and put them
beside the `.tex` file. Then, from this folder:

```bash
cp ../21_Figures/fig_01_research_intensity_map.pdf \
   ../21_Figures/fig_0{2,3,4,5,6}_v2_*.pdf \
   ../21_Figures/fig_A{3,4,5,6}_*.pdf .
pdflatex Poudel_Khanal_AIA_manuscript
bibtex   Poudel_Khanal_AIA_manuscript
pdflatex Poudel_Khanal_AIA_manuscript
pdflatex Poudel_Khanal_AIA_manuscript
```

The ten figure PDFs in `21_Figures/` are byte-identical to the files submitted with the
manuscript. Reading the PDF requires none of this.

## Status

Prepared for submission to *Advances in Agriculture* (Wiley). Not yet accepted or published: no
DOI, volume, issue, or article number exists for it. See `documentation/reproduction_notes.md`
for three values in the body that the coding correction superseded.
