# Lunar Rock-Abundance Estimation From Daytime Orbital Images

| | |
| --- | --- |
| Final rank | 1st |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 5 |
| Last submission | 2026-07-05 |

## Problem statement

### Lunar Rock-Abundance Estimation from Daytime Orbital Images

**Domain: Computer Vision — per-image scalar estimation from a single grayscale orbital image patch.**

You are given anonymized **100×100 grayscale patches** of the Moon's daytime surface. For each, estimate its **rock abundance** — the fraction of ground covered by exposed rocks.

### What makes it hard — and new

The rocks are **smaller than a single pixel** and cast no shadow at noon, so the answer simply **isn't in the image**. It is normally measured by a **nighttime thermal sensor that is withheld here** — all the daytime view leaves behind is faint texture. There is nothing to detect, segment, or translate: only to *infer*. It is a deliberately **information-limited** task — how far can a model reach toward a quantity the camera physically cannot see?

The score keeps that honest: the easy optical shortcut is **subtracted out** (prediction and truth are both de-trended against a brightness/albedo baseline), and skill is judged on the **worst unseen regions**, not the average. A brightness detector scores ~0; only sub-pixel structure brightness can't explain earns credit.

### Why it isn't the existing literature

Prior lunar work either reads rock abundance straight from the **thermal or radar signal** (withheld here) or **detects boulders resolved in much sharper imagery** (here they are sub-pixel, and the patches carry no location or scale). Both work where the signal is *visible*; this measures inference where it is *not*. That exact combination — sub-pixel, withheld-modality, shortcut-removed, cross-region — has no public benchmark.

### Data

All files are under `./dataset/public/`.

| File | Contents |
| --- | --- |
| `patches.npy` | A `uint8` NumPy array of shape `(12000, 100, 100)` — one 100×100 grayscale daytime patch per row. The **row index is the `tile_uid`** (row `k` is the patch for `tile_uid = k`). Patches are anonymized: randomly reoriented (rotations/reflections, which do not change the rock fraction) and carry no location, scale, or acquisition metadata. |
| `train.csv` | 8,000 labeled training rows (columns below). |
| `test.csv` | 4,000 rows to predict (columns below). |
| `sample_submission.csv` | A ready-to-edit submission template: 4,000 rows, one per test `tile_uid`, with `rock_abundance` pre-filled with a placeholder constant. Replace the placeholders with your predictions and keep the format identical. |

**Columns** (the fields of the three CSVs):

| Column | Type | Appears in | Description |
| --- | --- | --- | --- |
| `tile_uid` | integer | `train.csv`, `test.csv`, `sample_submission.csv` | Patch identifier; equals the row index into `patches.npy` (values 0–11999). |
| `rock_abundance` | float | `train.csv`, `sample_submission.csv` | The label — areal fraction of ground covered by exposed rocks, in `[0, 1]`. Provided in `train.csv` (your training target) and as a placeholder in `sample_submission.csv`; **withheld from `test.csv`** (it is the value you predict). |

The 4,000 test patches come from terrain regions **disjoint** from the training regions, so your estimator must generalize to provinces it has never seen labeled. Across the data, `rock_abundance` has a median of ≈0.006 and is strongly right-skewed, with a long tail of rock-rich fresh-crater terrain reaching ≈0.14.

### Submission

Write `./working/submission.csv` with **exactly two columns, in this order**: `tile_uid,rock_abundance`. Include **one row for every `tile_uid` in `test.csv`** — all 4,000 test ids, with no missing, extra, or duplicate rows (order does not matter). A correctly-formatted file looks like this:

```
tile_uid,rock_abundance
3,0.0054
7,0.0231
12,0.0061
58,0.0009
...
11998,0.0087
```

Each `rock_abundance` is your estimate of that patch's rock areal fraction: a finite number in `[0, 1]` (the same units as the `train.csv` labels). The grader raises an error on any submission with the wrong columns or order, a missing / extra / duplicate `tile_uid`, or a non-numeric, NaN, or out-of-range value.

### Scoring

The score is a single value in [0, 1] (higher is better) built from rank agreement between your predicted rock abundance and the withheld thermal-derived truth, with the optically-explainable part removed:

1. A surface brightness / fresh-ejecta-albedo baseline is computed for every test image. Both your predicted ranks and the true ranks are orthogonalized against this baseline (a cubic partialling of the baseline rank). This removes credit for the easy "bright fresh crater" shortcut: an estimator that only tracks image brightness or ejecta albedo scores near zero.
2. **Pooled term (weight 0.25):** the rank correlation between the residual predicted and residual true rock abundance over all 4,000 test images.
3. **Worst-regions term (weight 0.75):** the test images are grouped into their held-out terrain regions; the residual rank correlation is computed within each region, and the term is the mean of the two lowest regional values. This rewards estimators whose skill holds up in the hardest unseen provinces rather than only on average.

The final score is `0.25 * pooled + 0.75 * worst_two_regions`, with negative correlations floored at zero. A constant or brightness-only prediction scores about 0; recovering the truth exactly would score 1.

### What to use

This is a single-image scalar-regression vision task. Train an image model (fine-tune a pretrained backbone or train from scratch) on the provided patches and labels, reading the faint morphological cues that survive into daytime optics. No external data, internet access, or location lookup is permitted at inference — predict from the provided patches with your own model.
