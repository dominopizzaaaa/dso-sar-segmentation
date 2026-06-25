#!/usr/bin/env python3
"""
run_single_image.py
===================================================================
Run the full SAR semantic segmentation pipeline on a SINGLE SAR image.

Pipeline chain:
    AlignEarth (P8)  ->  P10 soft confidence  ->  P11 watershed carving

Usage:
    python run_single_image.py --input  path/to/scene.tif \
                               --output path/to/output_dir \
                               --scene-type {mine,port,urban,auto}

    # auto scene-type uses the GeoChat + Groq VLM to derive thresholds
    # mine/port/urban use the hard-coded threshold table (no VLM needed)

Outputs written to --output:
    <name>_p8.png    raw AlignEarth prediction (colour map)
    <name>_p10.png   after soft-confidence refinement
    <name>_final.png final P11 result with building outlines
    <name>_final.npy final class array (uint8, 0-5)

===================================================================
PIPELINE OVERVIEW (read this before diving into the functions below)
===================================================================
The pipeline runs in three stages, each one cleaning up the previous
stage's mistakes using a different kind of evidence:

  STAGE 1 (P8)  load_alignearth_model() + run_alignearth()
      A CLIP-style vision-language model (AlignEarth, a SAR-specific
      encoder distilled from optical CLIP) slides a 448x448 window
      across the full scene and predicts one of 6 classes per pixel,
      purely from learned visual patterns. This stage knows nothing
      about brightness rules or scene type -- it's the model's raw,
      "first guess" semantic segmentation. Output: a hard label map
      (one integer 0-5 per pixel) called pred_p8.

  STAGE 2 (P10) apply_p10()
      The hard P8 labels get converted into "soft" per-class
      probabilities (via Gaussian blurring -- see apply_p10's
      docstring), then those probabilities get reweighted using
      (a) fixed scene-type priors (e.g. "mines rarely have real
      buildings"), and (b) real SAR brightness/texture statistics
      computed directly from the image. The probabilities are then
      re-collapsed to a refined hard label map, pred_p10. This stage
      fixes systematic model biases (e.g. bright rocks misread as
      buildings) using domain knowledge the model itself doesn't have.

  STAGE 3 (P11) apply_p11()
      Two further fixes, both geometry/brightness driven rather than
      learned: (a) a flood-fill pass that finds large, smooth, dark
      regions touching the image border and forces them to "water"
      (catches big water bodies the model may have mislabeled), and
      (b) "edge carving" -- for any class-2 (road) blob made of mixed
      brightness, run a watershed transform to split it into the
      sub-regions it should actually be (road/bareland/building),
      since AlignEarth's coarse 448px tiling tends to blur several
      adjacent materials into one label. Output: pred_p11, the final
      class map, plus add_building_outlines() draws dark edge lines
      around buildings purely for visual clarity in the output PNG.

Net effect: P8 = "what does the model think this looks like",
P10 = "what does this look like once we apply domain priors and
brightness evidence", P11 = "clean up the geometry so blobs don't
straddle multiple real materials".
===================================================================
"""

import argparse
import gc
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from scipy.ndimage import (
    label as cc_label,
    uniform_filter,
    distance_transform_edt,
    binary_erosion,
    gaussian_filter,
    sobel,
)
from skimage.morphology import disk
from skimage.segmentation import watershed
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

# ===================================================================
# CONFIGURATION
# ===================================================================

# Path to the SegEarth-OV-2 codebase (contains the AlignEarth model code)
SEGEARTH_PATH = os.environ.get(
    "SEGEARTH_PATH", str(Path(__file__).parent / "models" / "SegEarth-OV-2")
)
sys.path.insert(0, SEGEARTH_PATH)

# Six-class taxonomy
CLASS_NAMES = ["background", "building", "road", "vegetation", "water", "bareland"]
N_CLS = 6
PALETTE = np.array(
    [
        [0, 0, 0],        # 0 background  - black
        [255, 0, 0],      # 1 building    - red
        [255, 255, 0],    # 2 road        - yellow
        [0, 128, 0],      # 3 vegetation  - green
        [0, 0, 255],      # 4 water       - blue
        [139, 90, 43],    # 5 bareland    - brown
    ],
    dtype=np.uint8,
)

# AlignEarth tiling parameters
AE_TILE = 448      # tile size extracted from the full-res image
AE_STRIDE = 224    # 50% overlap between adjacent tiles
AE_INPUT = 224     # size each tile is resized to before the model

# P10 scene-type confidence multipliers (see ASSUMPTIONS.md)
# These are fixed, hand-picked priors -- NOT learned from data. They
# encode domain knowledge like "open-pit mines almost never contain
# real buildings, so any 'building' prediction there is probably a
# misread bright rock pile" (hence the 0.02 multiplier for class 1
# under "mine" -- it suppresses that class's probability by 98%).
CONFIDENCE_ADJUST = {
    "mine":  {1: 0.02, 2: 0.3, 3: 0.3, 4: 0.5, 5: 2.0},
    "port":  {1: 1.0,  2: 1.0, 3: 0.9, 4: 1.3, 5: 0.8},
    "urban": {1: 1.2,  2: 1.1, 3: 1.0, 4: 0.8, 5: 0.7},
}

# P11 hard-coded brightness thresholds per scene type (see ASSUMPTIONS.md)
# In SAR, brightness roughly correlates with surface roughness: smooth
# surfaces (water, asphalt roads) scatter the radar signal away from
# the sensor and look dark; rough/vertical surfaces (buildings) scatter
# strongly back and look bright. These per-scene-type cutoffs say
# "below this brightness = road-like, above this = building-like,
# in between = bareland".
BRIGHTNESS_THRESHOLDS = {
    "mine":  {"road": 0.12, "bareland": 0.50},
    "port":  {"road": 0.15, "bareland": 0.45},
    "urban": {"road": 0.14, "bareland": 0.42},
}

CARVE_MIN_AREA = 2000  # only carve road blobs larger than this (pixels)


# ===================================================================
# STAGE 0 - SAR LOADING & NORMALISATION
# ===================================================================
def load_sar(tif_path):
    """
    Read a SAR GeoTIFF and normalise it to a [0, 1] brightness scale.

    WHAT IT DOES:
      1. Opens the .tif with rasterio and reads band 1 (SAR images are
         single-channel -- pure backscatter intensity, no colour).
      2. `valid = sar > 0` marks pixels with real data (some SAR scenes
         have 0-value padding around the actual footprint -- those
         pixels should never be classified as anything and get masked
         out later).
      3. Computes the 2nd and 98th percentile brightness across only
         the valid pixels (p2, p98). This is a robust way to find
         "typical" dark and bright bounds while ignoring extreme
         outlier pixels (sensor noise spikes, etc.) that a plain
         min/max would be thrown off by.
      4. Linearly rescales every pixel so that p2 -> 0.0 and p98 -> 1.0,
         then clips anything outside [0, 1]. This is a standard
         "percentile stretch" -- it makes brightness comparable across
         different scenes that might have very different raw intensity
         ranges (e.g. due to different sensor gain settings).

    RETURNS:
      sar_norm : float32 array (H, W), brightness rescaled to [0, 1]
      valid    : bool array (H, W), True where the original pixel had
                 real (non-zero) data
      H, W     : the image's height and width in pixels
    """
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        sar = src.read(1).astype(np.float32)
    valid = sar > 0
    p2, p98 = np.percentile(sar[valid], 2), np.percentile(sar[valid], 98)
    sar_norm = np.clip((sar - p2) / (p98 - p2 + 1e-6), 0, 1)
    return sar_norm, valid, H, W


# ===================================================================
# STAGE 1 - ALIGNEARTH INFERENCE (P8)
# ===================================================================
def load_alignearth_model(device, class_file, segearth_path):
    """
    Load the AlignEarth model (a CLIP-style vision-language encoder,
    specifically the SAR-adapted variant of SegEarth-OV-2's open-
    vocabulary segmentation framework) onto the given device (CPU/GPU)
    and prepare it for inference.

    KEY DETAIL -- the directory change:
      SegEarth-OV-2's own model code hardcodes the checkpoint path as
      the *relative* string 'checkpoint/AlignEarth-SAR-ViT-B-16.pt'.
      A relative path is resolved against the current working
      directory at the moment it's opened -- so this function
      temporarily `os.chdir()`s into the SegEarth-OV-2 folder before
      constructing the model (so that relative path correctly finds
      checkpoint/AlignEarth-SAR-ViT-B-16.pt sitting inside that
      folder), then changes back to wherever the script was running
      from afterward. `class_file` (the list of class names this
      model should predict) is resolved to an *absolute* path first,
      specifically so that step doesn't break once the cwd changes.

    PARAMETERS PASSED TO SegEarthSegmentation (the actual model class,
    defined in SegEarth-OV-2's segearth_segmentor.py):
      clip_type="AlignEarth"   -- selects the SAR-distilled encoder
                                   branch (as opposed to plain CLIP,
                                   RemoteCLIP, BLIP, etc. -- this
                                   codebase supports many backbones,
                                   we only ever use this one)
      vit_type="ViT-B/16"      -- the underlying transformer's size/
                                   patch-size variant
      model_type="SCLIP"       -- the segmentation head/strategy used
                                   to turn the model's patch features
                                   into a per-pixel class map
      name_path=class_file_abs -- the 6 class names this model should
                                   score against (background, building,
                                   road, vegetation, water, bareland)
      logit_scale=100          -- a temperature-like scaling factor
                                   applied to the model's raw
                                   similarity scores before softmax;
                                   higher = more confident/peaked
                                   predictions
      cls_token_lambda=0       -- weight given to the model's global
                                   [CLS] token vs. local patch tokens;
                                   0 means rely purely on local patch
                                   evidence (better for per-pixel
                                   segmentation than the whole-image
                                   summary)
      ignore_residual=False    -- whether to skip a particular
                                   residual-connection adjustment
                                   inside the model's attention layers
      feature_up=False         -- whether to run the optional
                                   SimFeatUp upsampling module (skipped
                                   here -- not needed by our 448px
                                   tiling approach)
      device=device            -- explicitly passed so the model loads
                                   on CPU or GPU correctly; without
                                   this, SegEarthSegmentation defaults
                                   to assuming a GPU is always present

    RETURNS: the model, moved to `device`, forced to float32 precision
    (.float()), and set to evaluation mode (.eval(), which disables
    dropout/batch-norm-update behaviour used only during training).
    Forcing fp32 here matters on CPU specifically: several PyTorch CPU
    kernels (matmul, conv2d) simply have no implementation for fp16
    ("Half") tensors, since fp16 is a GPU-only speed optimisation.
    """
    import os
    # Resolve class_file to absolute path BEFORE changing directory
    class_file_abs = str(os.path.abspath(class_file))
    # Must cd into SegEarth-OV-2 so "checkpoint/AlignEarth-SAR-ViT-B-16.pt" resolves
    orig_dir = os.getcwd()
    os.chdir(segearth_path)
    from segearth_segmentor import SegEarthSegmentation
    print("  Loading AlignEarth model...", flush=True)
    model = SegEarthSegmentation(
        clip_type="AlignEarth",
        vit_type="ViT-B/16",
        model_type="SCLIP",
        name_path=class_file_abs,
        logit_scale=100,
        cls_token_lambda=0,
        ignore_residual=False,
        feature_up=False,
        device=device,
    )
    os.chdir(orig_dir)
    return model.to(device).float().eval()


def run_alignearth(model, sar_norm, H, W, device):
    """
    Run sliding-window AlignEarth inference across the full SAR scene
    and return a hard (single integer per pixel) label map. This is
    "P8" -- the model's raw first-pass semantic segmentation, before
    any of the P10/P11 refinement.

    WHY TILING IS NEEDED:
    SAR scenes here are enormous (tens of thousands of pixels per
    side) but the model only accepts small, fixed-size inputs
    (AE_INPUT = 224x224). So the image is processed in overlapping
    448x448 tiles (AE_TILE), each downsized to 224x224 before going
    into the model, and the resulting per-tile predictions are
    stitched back together into a full-resolution map.

    STEP-BY-STEP:

    1. CLIP normalisation constants (`mean`, `std`):
       AlignEarth's vision tower is architecturally a CLIP ViT-B/16,
       which expects 3-channel input normalised by these exact
       per-channel mean/std values -- the same constants used when the
       original CLIP model was trained on natural optical photos. They
       aren't SAR-specific; they're inherited because this model's
       conv layer weights were shaped for that input distribution.

    2. `transform(patch_rgb)`:
       Resizes a 448x448 patch down to 224x224 (AE_INPUT, the model's
       expected input size) using Lanczos resampling, applies the
       mean/std normalisation above, and rearranges the array from
       HWC (height, width, channel) to CHW (channel, height, width) --
       the layout PyTorch convolution layers expect. The struct.pack/
       np.frombuffer round-trip is a defensive workaround: it forces a
       byte-for-byte copy through Python's struct module rather than
       letting numpy and torch share memory directly, which sidesteps
       a known ABI (binary interface) mismatch that can occur between
       certain numpy and torch versions.

    3. Faking 3 channels from 1:
       `sar_rgb = np.stack([...] * 3, axis=-1)` duplicates the single
       grayscale SAR brightness channel three times, producing a fake
       "RGB" image where R=G=B=SAR brightness. This satisfies the
       model's structural requirement for 3-channel input (inherited
       from CLIP/optical-image pretraining) without needing to retrain
       or reshape the first conv layer.

    4. Building tile positions:
       A grid of (y, x) top-left tile coordinates is generated with
       50% overlap (AE_STRIDE = 224, half of AE_TILE = 448) so that no
       seam falls exactly on a real object boundary every time. The
       extra clamping logic after the main grid loop makes sure tiles
       are also placed flush against the bottom and right edges of the
       image even if the image's dimensions aren't an exact multiple
       of the stride -- otherwise a strip along those edges would
       never get covered by any tile.

    5. Per-tile inference loop:
       For each tile: skip it entirely if it's pure zero-padding
       (`patch.max() == 0` -- nothing to classify). Otherwise, run it
       through the model (`model.forward_feature`) to get raw class
       logits, upsample those logits back from the model's internal
       resolution to the full 448x448 tile size (`F.interpolate`,
       bilinear), and convert logits to per-class probabilities with
       softmax. These per-pixel, per-class probabilities are
       accumulated into `prob_sum` at this tile's location, and
       `count_map` tracks how many overlapping tiles touched each
       pixel (since tiles overlap by 50%, most pixels get covered by
       more than one tile).

    6. Averaging overlapping predictions:
       `prob_sum / count_map` averages together every tile's opinion
       about each pixel (pixels near the middle of an object, covered
       by many tiles, get an averaged, smoothed-out probability;
       `np.maximum(count_map, 1)` just guards against a division by
       zero for any pixel that somehow got zero coverage).

    7. Final collapse to hard labels:
       `.argmax(axis=0)` picks whichever of the 6 classes has the
       highest averaged probability for each pixel, producing the
       final single-integer-per-pixel P8 label map that gets passed
       into apply_p10() next.
    """
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def transform(patch_rgb):
        im = Image.fromarray(patch_rgb).resize((AE_INPUT, AE_INPUT), Image.LANCZOS)
        arr = (np.array(im).astype(np.float32) / 255.0 - mean) / std
        t = arr.transpose(2, 0, 1)
        # struct.pack round-trip avoids numpy/torch ABI incompatibility
        raw = struct.pack(f"{t.size}f", *t.flatten().tolist())
        arr2 = np.frombuffer(raw, dtype=np.float32).reshape(t.shape).copy()
        return torch.tensor(arr2.tolist(), dtype=torch.float32)

    sar_rgb = np.stack([(sar_norm * 255).astype(np.uint8)] * 3, axis=-1)
    prob_sum = np.zeros((N_CLS, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    # Build tile positions (with edge clamping)
    positions = []
    for y in range(0, H - AE_TILE + 1, AE_STRIDE):
        for x in range(0, W - AE_TILE + 1, AE_STRIDE):
            positions.append((y, x))
    for x in range(0, W - AE_TILE + 1, AE_STRIDE):
        if (H - AE_TILE, x) not in positions:
            positions.append((H - AE_TILE, x))
    for y in range(0, H - AE_TILE + 1, AE_STRIDE):
        if (y, W - AE_TILE) not in positions:
            positions.append((y, W - AE_TILE))
    if (H - AE_TILE, W - AE_TILE) not in positions:
        positions.append((H - AE_TILE, W - AE_TILE))

    print(f"  AlignEarth: {len(positions)} tiles", flush=True)
    for i, (y, x) in enumerate(positions):
        patch = sar_rgb[y:y + AE_TILE, x:x + AE_TILE]
        if patch.max() == 0:
            continue
        tensor = transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model.forward_feature(tensor)
            logits_up = F.interpolate(
                logits.float(), size=(AE_TILE, AE_TILE),
                mode="bilinear", align_corners=False,
            )
            probs = F.softmax(logits_up, dim=1).squeeze(0)
            p = probs.detach().cpu().to(torch.float32)
            raw = struct.pack(f"{p.numel()}f", *p.flatten().tolist())
            arr = np.frombuffer(raw, dtype=np.float32).reshape(
                N_CLS, AE_TILE, AE_TILE
            ).copy()
            prob_sum[:, y:y + AE_TILE, x:x + AE_TILE] += arr
            count_map[y:y + AE_TILE, x:x + AE_TILE] += 1
        if i % 200 == 0:
            print(f"    {i}/{len(positions)}", flush=True)

    count_map = np.maximum(count_map, 1)
    return (prob_sum / count_map[np.newaxis]).argmax(axis=0).astype(np.uint8)


# ===================================================================
# STAGE 2 - P10 SOFT CONFIDENCE REFINEMENT
# ===================================================================
def guided_filter(guide, src, radius, eps):
    """
    A "guided filter" -- a classic image-processing technique that
    sharpens/smooths one image (`src`) using the structure of a second,
    sharper image (`guide`) as a reference, while staying close to
    src's own values. Conceptually it's a local linear regression: in
    every small sliding window of size (2*radius+1), it finds the best
    local linear relationship "src ~ a * guide + b" (the standard
    least-squares slope/intercept formulas: a = Cov(guide, src) /
    Var(guide), b = mean(src) - a * mean(guide)), then re-expresses
    src through that local relationship. The effect: src ends up
    following guide's sharp edges instead of staying blurry, while
    still tracking src's own overall values.

    In apply_p10(), `guide` is the real SAR brightness image and `src`
    is one class's blurry, Gaussian-smoothed soft-probability map --
    so this step "snaps" each class's fuzzy probability map onto the
    actual sharp edges visible in the SAR data, rather than leaving it
    blurred from the earlier Gaussian-blur step.

    `eps` is a small regularisation constant preventing division by
    zero when a window has near-zero brightness variance (a perfectly
    flat patch of guide).
    """
    size = 2 * radius + 1
    mg = uniform_filter(guide, size)
    ms = uniform_filter(src, size)
    mgs = uniform_filter(guide * src, size)
    mgg = uniform_filter(guide * guide, size)
    a = (mgs - mg * ms) / (mgg - mg * mg + eps)
    b = ms - a * mg
    return uniform_filter(a, size) * guide + uniform_filter(b, size)


def apply_p10(pred_raw, sar_norm, valid, scene_type):
    """
    "P10" -- refine the P8 hard label map using a combination of
    (a) manufactured soft probabilities, (b) fixed scene-type priors,
    and (c) real per-pixel SAR brightness/texture statistics.

    IMPORTANT CAVEAT: `pred_raw` here is the *hard* P8 output (one
    integer per pixel) -- AlignEarth's original softmax probabilities
    from run_alignearth() are NOT passed in. So the "soft" probabilities
    built in step (1) below are reconstructed from the hard labels via
    blurring, not the model's true confidence scores.

    STEP-BY-STEP:

    (1) Hard labels -> soft probabilities via Gaussian blur:
        For each class c, take the binary mask (pred_raw == c) -- 1.0
        where this pixel was labelled c, 0.0 elsewhere -- and blur it
        with a Gaussian kernel (sigma=3). A pixel deep inside a solid
        region of class c stays close to 1.0 after blurring (all its
        neighbours agree); a pixel near a class boundary gets pulled
        towards an intermediate value, since some of its blurred
        neighbourhood belonged to a different class. After blurring
        all 6 classes independently, the per-pixel values are
        renormalised to sum to 1 (a genuine probability distribution).
        This step is purely geometric -- "how far is this pixel from
        a class boundary" -- not based on any model confidence.

    (2) Scene-type confidence multipliers:
        Multiply each class's soft probability by a fixed prior from
        CONFIDENCE_ADJUST (e.g. in "mine" scenes, class 1/building
        gets multiplied by 0.02 -- a hard-coded belief that real
        buildings are extremely rare in open-pit mines, so most
        "building" predictions there are probably misclassified
        bright rock/rubble). Renormalise afterwards.

    (3) SAR intensity decomposition -- genuine per-pixel statistics:
        - Road penalty: pixels currently labelled road (class 2) that
          are unusually BRIGHT (> 0.10) get their road-probability
          reduced, scaled linearly by how far over 0.10 they are
          (roads should look dark in SAR; a bright "road" pixel is
          suspicious).
        - Mine-specific building/texture check: computes a genuine
          local statistic -- the coefficient of variation (local std
          / local mean) of SAR brightness in an 11x11 window, using
          the identity Var = E[X^2] - E[X]^2 via two uniform_filter
          passes. Low local texture variance for a "building" pixel
          (i.e. smooth, not rough) reduces its building-probability,
          since real buildings tend to scatter SAR signal with more
          local roughness/variation than smooth bright ground.
        - Water penalty: bright pixels labelled water (class 4) get
          penalised similarly to the road case (water should be dark).
        - Building dark-penalty: dim pixels labelled building get a
          mild probability reduction scaled by how dark they are.
        Renormalise after all of these.

    (4) Guided-filter sharpening:
        Run every class's soft-probability map through guided_filter()
        against the real SAR brightness image, snapping each class's
        fuzzy probability boundaries onto the SAR image's actual sharp
        edges. Renormalise once more.

    (5) Re-argmax:
        Collapse the final, reweighted soft probabilities back down to
        a single hard label per pixel (whichever class now has the
        highest probability), and zero out any pixel that was outside
        the originally valid SAR footprint (`~valid`).

    Net effect: this stage corrects systematic model biases that
    AlignEarth's purely visual, tile-based predictions can't account
    for on their own -- e.g. "bright + textured = building" being a
    bad rule specifically inside an open-pit mine.
    """
    H, W = sar_norm.shape
    conf_adj = CONFIDENCE_ADJUST.get(scene_type, {})

    # (1) soften hard labels into soft probabilities
    soft = np.zeros((N_CLS, H, W), dtype=np.float32)
    for c in range(N_CLS):
        soft[c] = gaussian_filter((pred_raw == c).astype(np.float32), sigma=3)
    s = soft.sum(0, keepdims=True); s[s == 0] = 1; soft /= s

    # (2) scene-type confidence multipliers
    for cls_id, mult in conf_adj.items():
        if mult != 1.0:
            soft[cls_id] *= mult
    s = soft.sum(0, keepdims=True); s[s == 0] = 1; soft /= s
    gc.collect()

    # (3) SAR intensity decomposition
    rm = (pred_raw == 2); bm = (pred_raw == 1); wm = (pred_raw == 4)
    if rm.any():
        rr = np.ones((H, W), dtype=np.float32)
        br = rm & (sar_norm > 0.10)
        rr[br] = np.clip(1.0 - (sar_norm[br] - 0.10) / 0.15, 0.05, 1.0)
        soft[2] *= rr
    if bm.any() and scene_type == "mine":
        lm = uniform_filter(sar_norm, 11); lsq = uniform_filter(sar_norm ** 2, 11)
        cov = np.sqrt(np.maximum(lsq - lm ** 2, 0)) / (lm + 1e-6)
        br2 = np.ones((H, W), dtype=np.float32); lt = bm & (cov < 0.8)
        br2[lt] = np.clip(cov[lt] / 0.8, 0.05, 1.0); soft[1] *= br2
    if wm.any():
        wr = np.ones((H, W), dtype=np.float32)
        bw = wm & (sar_norm > 0.1)
        wr[bw] = np.clip(1.0 - (sar_norm[bw] - 0.1) / 0.2, 0.05, 1.0)
        soft[4] *= wr
    if bm.any():
        bb = np.ones((H, W), dtype=np.float32)
        db = bm & (sar_norm < 0.2)
        bb[db] = np.clip(sar_norm[db] / 0.2, 0.1, 1.0)
        soft[1] *= bb
    s = soft.sum(0, keepdims=True); s[s == 0] = 1; soft /= s
    gc.collect()

    # (4) guided-filter sharpening using SAR edges
    for c in range(N_CLS):
        soft[c] = np.clip(guided_filter(sar_norm, soft[c], 3, 0.001), 0, 1)
    s = soft.sum(0, keepdims=True); s[s == 0] = 1; soft /= s

    # (5) re-argmax
    pred = np.argmax(soft, axis=0).astype(np.uint8)
    pred[~valid] = 0
    del soft; gc.collect()
    return pred


# ===================================================================
# STAGE 3 - P11 WATERSHED EDGE CARVING
# ===================================================================
def carve_blob(blob_mask, sar_norm, thresh):
    """
    Subdivide a single connected blob of road (class 2) pixels into
    the multiple real materials it probably actually contains, using
    a watershed transform seeded by SAR brightness. Called from
    apply_p11() only for blobs whose brightness is NOT uniform (i.e.
    likely several adjacent materials wrongly merged under one label
    by the coarser P8/P10 stages).

    STEP-BY-STEP:

    1. Crop to a tight bounding box around just this blob (plus a 20px
       margin), so the expensive operations below run on a small patch
       rather than the whole scene.

    2. If the blob's brightness IS actually uniform (std < 0.10) after
       all -- this is a second safety check inside carve_blob itself,
       separate from the std check that decided to call carve_blob in
       the first place -- just relabel the whole blob by its mean
       brightness and return early, skipping watershed entirely.

    3. Seed generation: erode the blob inward by ~3px (`disk(3)`) to
       discard unreliable edge pixels, then split the eroded interior
       into three brightness categories purely by the scene-type
       thresholds: dark (`dk`, below thresh["road"]), bright (`br`,
       above thresh["bareland"]), and medium (`md`, in between). Each
       category is connected-component labelled separately (so
       multiple disconnected dark patches become distinct seeds, not
       one merged seed) and folded into one combined `markers` array
       with globally unique integer IDs per seed.

    4. Watershed transform: treats the (lightly blurred, to reduce
       noise) SAR brightness surface as literal terrain -- dark =
       valley, bright = peak -- and simulates flooding outward from
       every seed simultaneously. Each seed's "water" claims
       neighbouring pixels until it would meet another seed's water,
       at which point a boundary forms there. `mask=bc` constrains the
       flood to stay inside the original blob.

    5. Reclassify each resulting watershed region by its own actual
       mean brightness (not just by which seed-category it grew from,
       since a region can absorb pixels of slightly different
       brightness during flooding) against the same road/bareland
       thresholds, producing the final per-pixel class assignment.

    6. Paste the cropped result back into a full-size array at the
       original coordinates, zeroing anything outside the original
       blob (so this function never touches pixels outside the one
       blob it was asked to carve).
    """
    H, W = sar_norm.shape
    ys, xs = np.where(blob_mask)
    if len(ys) == 0:
        return np.zeros_like(blob_mask, dtype=np.uint8)
    margin = 20
    y0 = max(0, ys.min() - margin); y1 = min(H, ys.max() + margin + 1)
    x0 = max(0, xs.min() - margin); x1 = min(W, xs.max() + margin + 1)
    bc = blob_mask[y0:y1, x0:x1]; sc = sar_norm[y0:y1, x0:x1]
    bs = sar_norm[blob_mask]

    # uniform blob -> single class by mean brightness
    if bs.std() < 0.10:
        result = np.zeros_like(blob_mask, dtype=np.uint8); mb = bs.mean()
        if mb < thresh["road"]:       result[blob_mask] = 2
        elif mb < thresh["bareland"]: result[blob_mask] = 5
        else:                         result[blob_mask] = 1
        return result

    # mixed blob -> watershed seeded from brightness
    be = binary_erosion(bc, disk(3))
    dk = be & (sc < thresh["road"])
    br = be & (sc > thresh["bareland"])
    md = be & (sc >= thresh["road"]) & (sc <= thresh["bareland"])
    markers = np.zeros(bc.shape, dtype=np.int32)
    dl, nd = cc_label(dk)
    for i in range(1, nd + 1): markers[dl == i] = i
    bl2, nb = cc_label(br)
    for i in range(1, nb + 1): markers[bl2 == i] = nd + i
    ml, nm = cc_label(md)
    for i in range(1, nm + 1): markers[ml == i] = nd + nb + i
    if markers.max() == 0:
        result = np.zeros_like(blob_mask, dtype=np.uint8); mb = bs.mean()
        if mb < thresh["road"]:       result[blob_mask] = 2
        elif mb < thresh["bareland"]: result[blob_mask] = 5
        else:                         result[blob_mask] = 1
        return result
    ws = watershed(gaussian_filter(sc, sigma=2), markers=markers, mask=bc)
    rc = np.zeros(bc.shape, dtype=np.uint8)
    for lbl in np.unique(ws):
        if lbl == 0: continue
        region = (ws == lbl) & bc
        if not region.any(): continue
        mb = sc[region].mean()
        if mb < thresh["road"]:       rc[region] = 2
        elif mb < thresh["bareland"]: rc[region] = 5
        else:                         rc[region] = 1
    result = np.zeros_like(blob_mask, dtype=np.uint8)
    result[y0:y1, x0:x1] = rc; result[~blob_mask] = 0
    return result


def apply_p11(pred, sar_norm, valid, scene_type):
    """
    "P11" -- final geometric cleanup pass on top of the P10 result,
    using three independent fixes:

    (1) Flood-fill water connectivity fix:
        Computes local brightness variance in an 11x11 window (same
        Var = E[X^2] - E[X]^2 trick as in apply_p10) to find regions
        that are both dark AND smooth/low-texture (`lv < 0.003`) --
        the SAR signature of calm open water, as opposed to merely
        dark land (e.g. dark asphalt, which still has texture). Among
        those candidate water pixels, it specifically keeps only the
        connected components that TOUCH the image border (`border`
        mask) and exceed 5000 pixels in size -- the logic being that a
        genuine open-water body (sea, large river) is large and almost
        always connects out to the edge of any cropped satellite
        scene, whereas a small, landlocked dark/smooth patch is more
        likely a different material that happens to look superficially
        similar. Any pixel passing all these checks gets force-set to
        class 4 (water), unless it was already vegetation (class 3) --
        a defensive guard against overwriting likely-correct
        vegetation pixels that can sometimes share the "dark and
        smooth" signature.

    (2) Watershed edge carving on large road blobs:
        For every connected blob of class-2 (road) pixels larger than
        CARVE_MIN_AREA (2000px): if the blob's brightness is uniform
        (std < 0.10), just relabel the whole thing by its mean
        brightness against the scene's thresholds. If the brightness
        is mixed (multiple real materials likely merged into one
        label), call carve_blob() to split it via watershed -- see
        that function's docstring for the full mechanism.

    (3) Small-fragment cleanup:
        For each class, find connected components smaller than 30
        pixels -- almost certainly noise/misclassification specks
        rather than real features -- and reassign each such fragment's
        pixels to whatever class is closest to them spatially, using
        `distance_transform_edt(..., return_indices=True)` (a
        Euclidean distance transform that, alongside the distance
        itself, also returns the *coordinates* of the nearest
        "non-fragment" pixel for every fragment pixel -- those
        coordinates are then used to look up and copy that nearest
        pixel's class). This is skipped if there are either zero or
        an enormous number (>10000) of such tiny fragments, the latter
        being a safety valve against pathological cases where this
        cleanup would be extremely slow for little benefit.

    Finally, any pixel outside the original valid SAR footprint
    (`~valid`) is forced back to background (0), since none of the
    fixes above should be able to manufacture a real class label
    inside a no-data region.
    """
    H, W = sar_norm.shape
    pred_out = pred.copy()
    thresh = BRIGHTNESS_THRESHOLDS[scene_type]

    # (1) flood-fill water connectivity fix
    lm = uniform_filter(sar_norm, 11); lsq = uniform_filter(sar_norm ** 2, 11)
    lv = np.maximum(lsq - lm ** 2, 0); del lm, lsq
    dt = float(np.percentile(sar_norm[valid].ravel(), 12))
    wc = (sar_norm < dt) & (lv < 0.003) & valid; del lv
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    wl, _ = cc_label(wc); bl = np.unique(wl[border & (wl > 0)])
    sea = np.isin(wl, bl) & wc; del wl, wc
    sl, _ = cc_label(sea); sz = np.bincount(sl.ravel())
    lg = np.where(sz >= 5000)[0]; lg = lg[lg > 0]
    sf = np.isin(sl, lg); del sl, sea
    pred_out[sf & (pred != 3)] = 4
    print(f"  Water fix: {sf.sum()} px", flush=True)
    del sf; gc.collect()

    # (2) watershed edge carving on large road blobs
    mask = (pred_out == 2)
    if mask.any():
        labs, n = cc_label(mask); sizes = np.bincount(labs.ravel())
        total = 0
        for cid in range(1, n + 1):
            if sizes[cid] < CARVE_MIN_AREA:
                continue
            comp = (labs == cid); bs = sar_norm[comp]
            if bs.std() < 0.10:
                mb = bs.mean()
                if mb < thresh["road"]:       pred_out[comp] = 2
                elif mb < thresh["bareland"]: pred_out[comp] = 5
                else:                         pred_out[comp] = 1
            else:
                carved = carve_blob(comp, sar_norm, thresh)
                cm = (carved > 0) & comp
                if cm.any():
                    pred_out[cm] = carved[cm]
            total += sizes[cid]
        del labs; gc.collect()
        print(f"  Edge carving: {total} px", flush=True)

    # (3) fragment cleanup
    for cls in range(1, N_CLS):
        m = (pred_out == cls); lab, _ = cc_label(m)
        sz = np.bincount(lab.ravel())
        sm = np.where(sz < 30)[0]; sm = sm[sm > 0]
        if len(sm) == 0 or len(sm) > 10000:
            continue
        s2 = np.isin(lab, sm)
        _, (iy, ix) = distance_transform_edt(s2, return_indices=True)
        pred_out[s2] = pred_out[iy[s2], ix[s2]]

    pred_out[~valid] = 0
    return pred_out


def add_building_outlines(pred_out, sar_norm, rgb):
    """
    Purely cosmetic post-processing for the final output PNG: draws a
    thin dark outline around each building (class 1) blob, so building
    footprints are visually easier to pick out against the surrounding
    classes. This does NOT change any pixel's class label in
    `pred_out` -- it only modifies the colour image `rgb` that gets
    saved to disk.

    HOW THE OUTLINE IS FOUND:
      1. Compute the SAR image's local gradient magnitude using Sobel
         operators (`sobel(..., axis=0)` for vertical edges, `axis=1`
         for horizontal; combined via Pythagorean sum) on a lightly
         blurred copy of the SAR brightness -- this highlights actual
         edges/boundaries in the underlying radar data.
      2. For each building blob bigger than 200 pixels: take the
         gradient values within that blob, find their 90th percentile,
         and keep only the pixels whose gradient exceeds that
         threshold (`be`) -- i.e. the strongest, most edge-like 10% of
         pixels within this building.
      3. Erode the blob inward twice with a plus-shaped structuring
         element, then intersect with `be` -- this restricts the
         "edge" pixels to ones that are also a few pixels in from the
         blob's true boundary, producing a clean, thin line rather
         than a noisy scattering of high-gradient pixels anywhere
         inside the building.
      4. Collect all such edge pixels across every building blob into
         one mask, then paint those pixels black ([0, 0, 0]) directly
         in the output RGB image.
    """
    building_mask = (pred_out == 1)
    if not building_mask.any():
        return
    sar_sm = gaussian_filter(sar_norm, sigma=1.0)
    gy = sobel(sar_sm, axis=0); gx = sobel(sar_sm, axis=1)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    bld_lab, n_bld = cc_label(building_mask)
    bld_sizes = np.bincount(bld_lab.ravel())
    edge_mask = np.zeros(pred_out.shape, dtype=bool)
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for bid in range(1, n_bld + 1):
        if bld_sizes[bid] < 200:
            continue
        blob = (bld_lab == bid)
        thr = np.percentile(grad[blob], 90)
        be = blob & (grad > thr)
        b_eroded = binary_erosion(blob, structure=struct)
        b_eroded = binary_erosion(b_eroded, structure=struct)
        be &= b_eroded
        edge_mask |= be
    rgb[edge_mask] = [0, 0, 0]


# ===================================================================
# VLM THRESHOLD DERIVATION (optional, --scene-type auto)
# ===================================================================
def derive_scene_type_vlm(sar_norm, geochat_dir, groq_key_path):
    """
    OPTIONAL alternative to manually specifying --scene-type. Uses a
    two-model chain to automatically guess the scene type (mine/port/
    urban) and derive matching brightness thresholds, instead of using
    the fixed BRIGHTNESS_THRESHOLDS/CONFIDENCE_ADJUST lookup tables
    keyed by a human-supplied scene type.

    Two-stage process:

    1. GeoChat (a remote-sensing-specialised vision-language model,
       loaded locally -- requires real GPU VRAM, not practical on a
       CPU-only laptop) is shown a downsampled thumbnail of the SAR
       scene with a prompt explaining SAR brightness conventions, and
       asked to describe the scene in plain English (dominant land
       cover, water presence, building density, etc.).

    2. That text description is sent to Groq's cloud API (a fast LLM
       inference service), along with a system prompt instructing it
       to output a strict JSON object classifying the scene as mine/
       port/urban and proposing specific road/bareland brightness
       thresholds, given general domain knowledge about what SAR
       brightness ranges typically correspond to which scene types.

    The thresholds are clipped to sane ranges (`np.clip`) and a
    sanity check (`if rt >= bt`) nudges the bareland threshold upward
    if the model returned a nonsensical pair where road >= bareland.

    Falls back to a fixed "urban" scene type with urban's default
    thresholds if ANYTHING in this chain fails (GeoChat not available,
    Groq API error, malformed JSON, etc.) -- wrapped in a broad
    try/except specifically so that a failure in this optional,
    automatic path never crashes the whole pipeline; the user can
    always fall back to manually specifying --scene-type instead.

    NOTE: this function is only ever called when --scene-type=auto.
    For mine/port/urban, the pipeline skips this entirely and goes
    straight to the fixed lookup tables -- no GPU or API key required.
    """
    try:
        # --- GeoChat scene description ---
        sys.path.insert(0, geochat_dir)
        from geochat.model.builder import load_pretrained_model
        from geochat.mm_utils import tokenizer_image_token
        from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from geochat.conversation import conv_templates

        device = torch.device("cuda")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            str(Path(geochat_dir) / "weights" / "GeoChat-7B"), None, "geochat"
        )
        model.eval()

        sc = max(1, max(sar_norm.shape) // 512)
        thumb = (sar_norm[::sc, ::sc] * 255).astype(np.uint8)
        img = Image.fromarray(thumb).convert("RGB")
        image_tensor = image_processor.preprocess(
            [img], crop_size={"height": 504, "width": 504},
            size={"shortest_edge": 504}, return_tensors="pt"
        )["pixel_values"].half().to(device)

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + (
            "A SAR (Synthetic Aperture Radar) satellite image. "
            "Bright areas = strong backscatter (buildings, rough surfaces). "
            "Dark areas = weak backscatter (water, smooth roads, shadow). "
            "Describe the scene: type, dominant land cover, water presence, "
            "building density, brightness patterns."
        )
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(
            conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = model.generate(
                input_ids, images=image_tensor,
                do_sample=False, max_new_tokens=300, use_cache=True,
            )
        desc = tokenizer.decode(
            out[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        print(f"  [GeoChat] {desc[:120]}...", flush=True)

        # --- Groq threshold derivation ---
        groq_key = Path(groq_key_path).read_text().strip()
        system = (
            "You are an expert in SAR image analysis. Given a scene description, "
            "output brightness thresholds for edge carving. In SAR normalised to "
            "[0,1]: roads=dark(0.05-0.20), bareland=medium(0.20-0.50), "
            "buildings=bright(0.40-0.80). Mine/arid: road_thresh LOW (0.10-0.13). "
            "Port/urban: slightly higher (0.14-0.17). Also classify the scene as "
            'one of mine/port/urban. Respond ONLY with JSON: '
            '{"scene_type":"<mine|port|urban>","road_thresh":<float>,'
            '"bareland_thresh":<float>}'
        )
        payload = json.dumps({
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Description:\n{desc}\n\nJSON only."},
            ],
            "max_tokens": 200, "temperature": 0.1,
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(payload); tmp = f.name
        r = subprocess.run(
            ["curl", "-s", "https://api.groq.com/openai/v1/chat/completions",
             "-H", f"Authorization: Bearer {groq_key}",
             "-H", "Content-Type: application/json", "-d", f"@{tmp}"],
            capture_output=True, text=True, timeout=60,
        )
        os.unlink(tmp)
        import re
        raw = json.loads(r.stdout)["choices"][0]["message"]["content"]
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        d = json.loads(clean)
        st = d.get("scene_type", "urban")
        rt = float(np.clip(float(d["road_thresh"]), 0.08, 0.20))
        bt = float(np.clip(float(d["bareland_thresh"]), 0.35, 0.58))
        if rt >= bt:
            bt = min(rt + 0.20, 0.58)
        print(f"  [Groq] scene={st} road<{rt:.3f} bareland<{bt:.3f}", flush=True)
        return st, {"road": rt, "bareland": bt}
    except Exception as e:
        print(f"  [VLM] ERROR ({e}); falling back to scene_type=urban", flush=True)
        return "urban", BRIGHTNESS_THRESHOLDS["urban"]


# ===================================================================
# MAIN
# ===================================================================
def main():
    """
    Entry point: parses CLI arguments, runs the three-stage pipeline
    (P8 -> P10 -> P11) on the given SAR scene in order, and writes
    out four files into --output:
      <name>_p8.png    raw AlignEarth prediction
      <name>_p10.png   after soft-confidence refinement
      <name>_final.png final P11 result, with building outlines drawn
      <name>_final.npy the same final result as a raw numpy array
                        (uint8, values 0-5), for any downstream
                        analysis that needs the class labels directly
                        rather than a colour image

    Device selection: `torch.device("cuda" if torch.cuda.is_available()
    else "cpu")` automatically uses a GPU if one is present and visible
    to PyTorch, otherwise falls back to CPU -- this is what allows the
    exact same script to run unmodified on a GPU server or a CPU-only
    laptop.
    """
    ap = argparse.ArgumentParser(description="Single-image SAR segmentation")
    ap.add_argument("--input", required=True, help="Path to input SAR .tif")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument(
        "--scene-type", default="auto",
        choices=["mine", "port", "urban", "auto"],
        help="'auto' uses GeoChat+Groq VLM; others use the hard-coded table",
    )
    ap.add_argument("--geochat-dir", default=str(Path(__file__).parent / "models" / "GeoChat"))
    ap.add_argument("--groq-key", default=str(Path(__file__).parent / "groq_key.txt"))
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    # Use parent folder name if TIF has a long auto-generated name
    name = out_dir.name if out_dir.name else in_path.stem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    print(f"\n=== Processing {name} ===", flush=True)

    # Stage 0
    sar_norm, valid, H, W = load_sar(in_path)
    print(f"  Loaded SAR: {H}x{W}", flush=True)

    # Determine scene type + thresholds
    if args.scene_type == "auto":
        scene_type, _ = derive_scene_type_vlm(
            sar_norm, args.geochat_dir, args.groq_key
        )
    else:
        scene_type = args.scene_type
    print(f"  Scene type: {scene_type}", flush=True)

    # Write class list for AlignEarth
    class_file = out_dir / "_classes.txt"
    class_file.write_text("\n".join(CLASS_NAMES))

    # Stage 1: AlignEarth
    model = load_alignearth_model(device, str(class_file), SEGEARTH_PATH)
    pred_p8 = run_alignearth(model, sar_norm, H, W, device)
    pred_p8[~valid] = 0
    Image.fromarray(PALETTE[np.clip(pred_p8, 0, 5)]).save(out_dir / f"{name}_p8.png")
    print(f"  Saved {name}_p8.png", flush=True)

    # Stage 2: P10
    pred_p10 = apply_p10(pred_p8, sar_norm, valid, scene_type)
    Image.fromarray(PALETTE[np.clip(pred_p10, 0, 5)]).save(out_dir / f"{name}_p10.png")
    print(f"  Saved {name}_p10.png", flush=True)

    # Stage 3: P11
    pred_p11 = apply_p11(pred_p10, sar_norm, valid, scene_type)
    p11_rgb = PALETTE[np.clip(pred_p11, 0, 5)].copy()
    p11_rgb[~valid] = 0
    add_building_outlines(pred_p11, sar_norm, p11_rgb)
    Image.fromarray(p11_rgb).save(out_dir / f"{name}_final.png")
    np.save(out_dir / f"{name}_final.npy", pred_p11)
    print(f"  Saved {name}_final.png and {name}_final.npy", flush=True)

    print(f"=== Done in {time.time() - t0:.1f}s ===\n", flush=True)


if __name__ == "__main__":
    main()
