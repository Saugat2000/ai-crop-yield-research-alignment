"""Colour-vision-deficiency check for every submitted figure.

Each figure PDF is rasterised and passed through Machado, Oliveira and Fernandes (2009)
severity-1.0 simulation matrices for deuteranopia, protanopia, and tritanopia. For each
figure the palette colours actually used are recovered and every pair is compared in CIE
Lab space before and after simulation. A pair separated by more than the just-noticeable
threshold under normal vision but collapsing below it under simulation is reported.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import fitz
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "01_Project_Management"))
from project_config import P, RunLogger  # noqa: E402

SUB = ROOT / "Final Manuscript" / "Manuscript 1" / "09_Wiley_Submission"
OUT = HERE / "cvd"; OUT.mkdir(exist_ok=True)

# Machado et al. (2009), severity 1.0
M = {
 "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                           [0.280085, 0.672501,  0.047413],
                           [-0.011820, 0.042940, 0.968881]]),
 "protanopia":   np.array([[0.152286, 1.052583, -0.204868],
                           [0.114503, 0.786281,  0.099216],
                           [-0.003882,-0.048116, 1.051998]]),
 "tritanopia":   np.array([[1.255528,-0.076749, -0.178779],
                           [-0.078411,0.930809,  0.147602],
                           [0.004733, 0.691367,  0.303900]]),
}
FIGS = {
 "Figure 1  (research output map)":        "fig_01_research_intensity_map.pdf",
 "Figure 2  (scale alignment)":            "fig_03_v2_scale_alignment.pdf",
 "Figure 3  (need vs research)":           "fig_04_v2_need_vs_research.pdf",
 "Figure 4  (LISA clusters + insets)":     "fig_05_v2_mismatch_lisa.pdf",
 "Figure A1 (crop attention vs area)":     "fig_02_v2_crop_attention_area.pdf",
 "Figure A2 (coefficients)":               "fig_06_v2_coefficients.pdf",
 "Figure A3 (sample flow)":                "fig_A6_sample_flow.pdf",
 "Figure A4 (predicted probability)":      "fig_A3_predicted_probability.pdf",
 "Figure A5 (concentration by period)":    "fig_A4_temporal_concentration.pdf",
 "Figure A6 (component correlations)":     "fig_A5_need_component_correlations.pdf",
}
THRESHOLD = 12.0   # Lab distance below which two fills read as the same colour in print


def srgb_to_lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def simulate(rgb, kind):
    lin = srgb_to_lin(rgb.astype(float))
    out = lin @ M[kind].T
    out = np.clip(out, 0, 1)
    srgb = np.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)
    return np.clip(srgb * 255, 0, 255)


def to_lab(rgb):
    lin = srgb_to_lin(np.asarray(rgb, float))
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def palette(path, top=9):
    """Recover the dominant saturated fills actually used in the figure."""
    d = fitz.open(path)
    pix = d[0].get_pixmap(dpi=110)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    flat = img.reshape(-1, 3)
    q = (flat // 8 * 8)
    keep = ~((q.max(1) - q.min(1) < 18) | (q.min(1) > 235))   # drop greys and near-white
    q = q[keep]
    if not len(q):
        return [], img
    cols, counts = np.unique(q, axis=0, return_counts=True)
    o = np.argsort(-counts)[:top]
    return [tuple(int(v) for v in cols[i]) for i in o if counts[i] > 60], img


def main():
    lg = RunLogger("rf_03_cvd_check")
    print("COLOUR-VISION-DEFICIENCY CHECK (Machado et al. 2009, severity 1.0)")
    print(f"pairs are flagged when Lab separation falls below {THRESHOLD:.0f} after simulation\n")
    rows, any_fail = [], False
    for label, fname in FIGS.items():
        p = SUB / fname
        if not p.exists():
            print(f"  {label:42s} FILE MISSING"); continue
        pal, img = palette(p)
        verdicts = []
        for kind in M:
            worst, worst_pair = 1e9, None
            for i in range(len(pal)):
                for j in range(i + 1, len(pal)):
                    a, b = np.array(pal[i]), np.array(pal[j])
                    d0 = np.linalg.norm(to_lab(a) - to_lab(b))
                    if d0 < THRESHOLD:          # already similar under normal vision
                        continue
                    d1 = np.linalg.norm(to_lab(simulate(a, kind)) - to_lab(simulate(b, kind)))
                    if d1 < worst:
                        worst, worst_pair = d1, (pal[i], pal[j])
            ok = worst >= THRESHOLD or worst_pair is None
            verdicts.append((kind, ok, worst if worst_pair else np.nan, worst_pair))
        passed = all(v[1] for v in verdicts)
        any_fail |= not passed
        status = "PASS" if passed else "REVIEW"
        print(f"  {label:42s} {status}   colours={len(pal)}")
        for kind, ok, worst, pair in verdicts:
            if not ok:
                print(f"        {kind}: closest pair separation {worst:.1f} "
                      f"({pair[0]} vs {pair[1]})")
        rows.append(dict(figure=label, file=fname, n_colours=len(pal), passed=passed,
                         **{k: (f"{w:.1f}" if not np.isnan(w) else "n/a")
                            for k, _, w, _ in verdicts}))
        # save a deuteranopia preview
        sim = simulate(img.astype(float), "deuteranopia").astype(np.uint8)
        Image.fromarray(sim).save(OUT / fname.replace(".pdf", "_deuteranopia.png"))
    import pandas as pd
    pd.DataFrame(rows).to_csv(HERE / "cvd_check.csv", index=False)
    print(f"\nsimulated previews in {OUT}")
    print("overall:", "all figures pass" if not any_fail else "at least one figure needs review")
    lg.add_output(HERE / "cvd_check.csv")
    lg.finish()


if __name__ == "__main__":
    main()
