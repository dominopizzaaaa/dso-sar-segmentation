# Assumptions — Thresholding and VLM/LLM Components

This document records the assumptions baked into the SAR segmentation
pipeline, focusing on the two areas that rely on tuned values or external
model behaviour: **brightness thresholding** (P10/P11) and the
**GeoChat + Groq VLM/LLM** threshold-derivation step.

---

## 1. SAR Brightness Assumptions

The entire post-processing pipeline rests on one physical premise:

> **In an amplitude SAR image normalised to [0, 1], land-cover classes occupy
> characteristic brightness ranges because of how each surface scatters radar
> energy.**

| Class | Typical normalised brightness | Physical reason |
|-------|------------------------------|-----------------|
| Water | very dark (< 0.10) | Smooth surface → specular reflection away from sensor |
| Road | dark (0.05 – 0.20) | Smooth asphalt → mostly specular, weak return |
| Bareland | medium (0.20 – 0.50) | Rough soil/rock → moderate diffuse backscatter |
| Building | bright (0.40 – 0.80) | Corner/double-bounce reflection → strong return |
| Vegetation | medium-variable | Volume scattering from canopy |

**Assumption:** the 2nd–98th percentile normalisation applied at load time
produces comparable brightness distributions across scenes. This holds for
Umbra GEC products processed by the same ground segment, but **breaks for
imagery from a different sensor, processing chain, or acquisition geometry**
(see multi-angle notes below).

---

## 2. P10 Scene-Type Confidence Multipliers

P10 multiplies each class's soft probability by a scene-type-specific weight.
These weights encode prior knowledge about what each environment should
contain. They are **hand-tuned, not learned**.

| Scene type | building | road | vegetation | water | bareland |
|------------|---------:|-----:|-----------:|------:|---------:|
| mine | 0.02 | 0.30 | 0.30 | 0.50 | 2.00 |
| port | 1.00 | 1.00 | 0.90 | 1.30 | 0.80 |
| urban | 1.20 | 1.10 | 1.00 | 0.80 | 0.70 |

**Assumptions:**

- **Mine scenes contain almost no buildings.** The ×0.02 multiplier nearly
  eliminates building predictions. This is correct for open-pit mines
  (Bingham, Nevada, Kalgoorlie) but **would wrongly suppress buildings in a
  mine scene that contains a processing plant or town.**
- **Mines are predominantly bareland.** The ×2.0 bareland boost assumes the
  dominant surface is exposed rock/soil.
- **Ports have significant water.** The ×1.3 water boost assumes a coastline
  is present.
- A scene must be assignable to exactly one of three types. Scenes that mix
  types (e.g. an urban area adjacent to a large mine) are not well modelled.

If a new scene does not fit mine/port/urban cleanly, the multipliers should be
revisited rather than forced into the nearest bucket.

---

## 3. P11 Brightness Thresholds

P11's watershed edge carving reclassifies pixels inside large road blobs using
two thresholds:

- `road_thresh` — pixels **darker** than this stay **road**
- `bareland_thresh` — pixels **brighter** than this become **building**;
  pixels in between become **bareland**

Hard-coded defaults:

| Scene type | road_thresh | bareland_thresh |
|------------|------------:|----------------:|
| mine | 0.12 | 0.50 |
| port | 0.15 | 0.45 |
| urban | 0.14 | 0.42 |

**Assumptions:**

- Mine terrain is darker overall, so `road_thresh` is set lower (0.12) to avoid
  labelling dark bareland as road.
- These thresholds are only applied to **road blobs larger than 2000 pixels**
  (`CARVE_MIN_AREA`). Smaller road regions are left untouched — the assumption
  is that small road predictions are usually correct and only large blobs are
  the result of terrace/mixed-terrain confusion.
- A blob with internal brightness standard deviation **below 0.10** is treated
  as uniform and reclassified wholesale by its mean brightness; above 0.10 it
  is carved with watershed. The 0.10 cutoff is empirical.

---

## 4. Water Connectivity Fix Assumptions

The flood-fill water fix labels a region as water if it is:

1. **dark** — below the 12th percentile of scene brightness, AND
2. **smooth** — local variance below 0.003, AND
3. **edge-connected** — the connected component touches the image border, AND
4. **large** — the connected sea component is ≥ 5000 pixels.

**Assumptions:**

- **Large water bodies reach the image edge.** This is true for coastal
  port/urban scenes but **would miss an inland lake fully enclosed within the
  frame.**
- The 12th-percentile darkness cutoff assumes water is among the darkest 12% of
  pixels. Very bright scenes (lots of buildings) or very dark scenes (mostly
  water) may need this adjusted.

---

## 5. GeoChat + Groq VLM/LLM Assumptions

When `--scene-type auto` is used, the scene type and thresholds are derived by
a two-model chain instead of the hard-coded tables.

### 5.1 Pipeline

```
SAR thumbnail ──► GeoChat (7B VLM) ──► natural-language scene description
                                          │
                                          ▼
                  Groq LLaMA-4 ──► JSON {scene_type, road_thresh, bareland_thresh}
```

### 5.2 GeoChat assumptions

- GeoChat is given a **downsampled thumbnail** (max side ≈ 512 px), not the full
  image. The assumption is that scene *type* is recognisable at low resolution.
- The prompt **explicitly tells GeoChat the SAR brightness convention** ("bright
  = strong backscatter…"). Without this, GeoChat reasons optically and produces
  unreliable descriptions. The quality of the downstream thresholds depends on
  GeoChat correctly describing dominant land cover, water presence, and building
  density.

### 5.3 Groq LLaMA-4 assumptions

- The system prompt **constrains the numeric output ranges** (road 0.08–0.20,
  bareland 0.35–0.58). The LLM is assumed to reason *within* these physically
  motivated bounds rather than inventing values.
- Output is **forced to JSON.** Parsing assumes the model returns valid JSON
  (after stripping markdown fences). Malformed output triggers the fallback.
- **Validation finding:** across the 10 development scenes, VLM-derived
  thresholds landed within **±0.03** of the hand-tuned values. This is the
  empirical basis for trusting the auto mode — it is **not guaranteed** to hold
  on scene types not represented in development.

### 5.4 Determinism and reproducibility

- GeoChat is run with `do_sample=False` (greedy decoding) → deterministic.
- Groq is run with `temperature=0.1` → near-deterministic but **not perfectly
  reproducible**. Re-running auto mode may shift a threshold by a small amount.
- For a fully reproducible run, use an explicit `--scene-type mine|port|urban`
  instead of `auto`.

### 5.5 Failure / fallback behaviour

If **anything** in the VLM chain fails (model load error, network failure,
malformed JSON, Groq timeout), the pipeline:

1. prints a `[VLM] ERROR` message, and
2. **falls back to `scene_type = urban`** with the urban threshold table.

**Assumption:** urban is the safest default because its multipliers and
thresholds are the least aggressive (no near-zero suppression like mine's
building ×0.02). A wrong urban fallback degrades quality gracefully rather than
catastrophically erasing a class. If the true scene is a mine, the fallback
will leave buildings over-predicted — review the output and re-run with an
explicit `--scene-type mine` if this happens.

### 5.6 Network assumptions

- Groq is reached via **`curl` subprocess**, not Python `urllib`. On the
  development cluster the compute-node HTTP proxy blocked `urllib` but allowed
  `curl`. On a machine with normal outbound HTTPS this distinction does not
  matter, but the `curl` binary must be present.
- A valid Groq API key must be at the path given by `--groq-key`
  (default `./groq_key.txt`).

---

## 6. Summary of What Must Be Re-Checked for a New Sensor/Region

If this pipeline is applied to imagery **outside the Umbra X-band development
set**, the following assumptions are most likely to need re-tuning:

1. **Percentile normalisation** — confirm 2/98 stretch gives comparable
   brightness (Section 1).
2. **Scene-type multipliers** — confirm mine/port/urban buckets still apply
   (Section 2).
3. **Brightness thresholds** — re-tune `road_thresh` / `bareland_thresh`, or
   rely on `--scene-type auto` and validate the VLM output (Sections 3, 5).
4. **Water edge-connectivity** — fails for enclosed inland water (Section 4).
5. **VLM ±0.03 validation** — does not transfer automatically; re-validate
   (Section 5.3).
