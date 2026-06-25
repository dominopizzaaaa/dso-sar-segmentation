# ===================================================================
# Dockerfile — SAR Semantic Segmentation Pipeline (CPU / portable)
#
# Builds a self-contained, CPU-only image with ALL dependencies
# (torch, mmseg, rasterio, scikit-image, open_clip, BLIP/fairscale/timm
# pulled in by SegEarth-OV-2, etc.) so the pipeline runs identically on
# Mac, Windows, and Linux without any manual environment setup.
#
# Built once for linux/amd64 in CI. Apple-Silicon Macs run it through
# emulation; Windows runs it via the WSL2 backend; Linux runs it
# natively.
#
# Model weights are NOT baked in (too large / licensed). They are
# mounted at run time.
# ===================================================================

# Plain Ubuntu 22.04 — no CUDA. Small, runs anywhere.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---- Global numpy<2 lock ----
# torch 2.1.2 is built against numpy 1.x. SegEarth-OV-2's own
# requirements.txt actually pins numpy==2.0.0, but that conflicts with
# torch 2.1.2's compiled ABI (numpy 2.x breaks torch.from_numpy with
# "Failed to initialize NumPy: _ARRAY_API not found"). We deliberately
# override their pin and stay on numpy<2 — torch interop is more
# fundamental and affects every package in this image, not just one
# dependency's stated preference. PIP_CONSTRAINT forces EVERY pip/mim
# install for the rest of the build to honor this.
RUN echo "numpy<2.0" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# ---- System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        git \
        curl \
        libgl1 \
        libglib2.0-0 \
        gdal-bin \
        libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# ---- Python dependencies ----
RUN pip install --no-cache-dir "numpy<2.0" "setuptools<81" wheel

# torch CPU build (2.1.2 — matches SegEarth-OV-2's own requirements.txt
# pin, and matches the mmcv cpu/torch2.1 wheel below)
RUN pip install --no-cache-dir \
        torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cpu

# Core scientific / imaging stack, plus everything SegEarth-OV-2's own
# requirements.txt declares (its bundled BLIP module needs transformers/
# fairscale/timm; opencv and openpyxl/safetensors are used by its own
# data + checkpoint loading code). Versions follow SegEarth-OV-2's
# requirements.txt where they don't conflict with the numpy<2 pin above.
# Source: https://github.com/earth-insights/SegEarth-OV-2/blob/main/requirements.txt
#   numpy==2.0.0 in that file is intentionally NOT followed — see note above.
RUN pip install --no-cache-dir \
        pillow \
        rasterio \
        scipy \
        scikit-image \
        open_clip_torch \
        "ftfy==6.2.3" \
        "regex==2024.9.11" \
        "tqdm==4.65.2" \
        huggingface_hub \
        "transformers==4.44.2" \
        "einops==0.8.0" \
        "fairscale==0.4.13" \
        "timm==1.0.9" \
        "safetensors==0.4.5" \
        "opencv-python-headless==4.8.0.76" \
        "openpyxl==3.1.5" \
        "matplotlib==3.8.4" \
        "fsspec==2024.3.1"

# MMSegmentation — install mmcv from OpenMMLab's prebuilt CPU wheel index
# (matched to torch 2.1) so nothing compiles from source. Versions match
# SegEarth-OV-2's requirements.txt (mmcv==2.1.0, mmengine==0.10.4,
# mmsegmentation==1.2.2).
RUN pip install --no-cache-dir -U openmim && \
    mim install "mmengine==0.10.4" && \
    pip install --no-cache-dir mmcv==2.1.0 \
        -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html && \
    pip install --no-cache-dir "mmsegmentation==1.2.2"

# Belt-and-suspenders: reassert numpy<2 in case mim's pip pass bumped it,
# then verify numpy<->torch interop AND every dependency SegEarth-OV-2's
# BLIP module needs (transformers, fairscale, timm) actually import.
# This fails the build loudly instead of surfacing as a runtime crash
# on someone's laptop three steps later.
RUN pip install --no-cache-dir "numpy<2.0" && \
    python -c "\
import numpy as np, torch, mmcv, mmseg, mmengine, rasterio, skimage, open_clip; \
import transformers, fairscale, timm, einops, safetensors, openpyxl, cv2, matplotlib; \
from transformers.activations import ACT2FN; \
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper; \
assert np.__version__.startswith('1.'), 'numpy must be <2, got ' + np.__version__; \
assert torch.from_numpy(np.zeros((2,2), dtype='float32')).sum().item() == 0.0; \
print('OK | numpy', np.__version__, '| torch', torch.__version__, '| mmcv', mmcv.__version__, '| mmseg', mmseg.__version__, '| transformers', transformers.__version__, '| timm', timm.__version__)"

# ---- Project code ----
WORKDIR /app
COPY run_single_image.py run_batch.sh ./
COPY docs ./docs

# models/, input_scenes/, results/ are mounted at run time,
# so they are intentionally NOT copied into the image.

# Default command prints help. Override at run time with real arguments.
ENTRYPOINT ["python", "run_single_image.py"]
CMD ["--help"]
