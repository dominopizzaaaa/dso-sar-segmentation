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
CONFIDENCE_ADJUST = {
    "mine":  {1: 0.02, 2: 0.3, 3: 0.3, 4: 0.5, 5: 2.0},
    "port":  {1: 1.0,  2: 1.0, 3: 0.9, 4: 1.3, 5: 0.8},
    "urban": {1: 1.2,  2: 1.1, 3: 1.0, 4: 0.8, 5: 0.7},
}

# P11 hard-coded brightness thresholds per scene type (see ASSUMPTIONS.md)
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
    """Read a SAR GeoTIFF and normalise to [0,1] via 2-98 percentile stretch."""
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
    return model.to(device).eval()


def run_alignearth(model, sar_norm, H, W, device):
    """Sliding-window AlignEarth inference. Returns hard label map (H,W)."""
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
    size = 2 * radius + 1
    mg = uniform_filter(guide, size)
    ms = uniform_filter(src, size)
    mgs = uniform_filter(guide * src, size)
    mgg = uniform_filter(guide * guide, size)
    a = (mgs - mg * ms) / (mgg - mg * mg + eps)
    b = ms - a * mg
    return uniform_filter(a, size) * guide + uniform_filter(b, size)


def apply_p10(pred_raw, sar_norm, valid, scene_type):
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
    Use GeoChat + Groq to classify the scene and derive thresholds.
    Returns (scene_type, thresholds_dict). Falls back to 'urban' on error.
    See ASSUMPTIONS.md for the prompt design and fallback behaviour.
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
