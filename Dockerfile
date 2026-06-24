# ===================================================================
# Dockerfile — SAR Semantic Segmentation Pipeline (CPU / portable)
#
# Builds a self-contained, CPU-only image with ALL dependencies
# (torch, mmseg, rasterio, scikit-image, open_clip, etc.) so the
# pipeline runs identically on Mac, Windows, and Linux without any
# manual environment setup.
#
# Built once for linux/amd64 in CI. Apple-Silicon Macs run it through
# emulation; Windows runs it via the WSL2 backend; Linux runs it
# natively.
#
# Model weights are NOT baked in (too large / licensed). They are
# mounted at run time — see the run commands below / DOCKER.md.
# ===================================================================

# Plain Ubuntu 22.04 — no CUDA. Small, runs anywhere.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

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
# numpy pinned <2.0 to avoid the ABI incompatibility.
RUN pip install --no-cache-dir "numpy<2.0" "setuptools<81" wheel

# torch CPU build (2.1.2 — matches the mmcv cpu/torch2.1 wheel below)
RUN pip install --no-cache-dir \
        torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cpu

# Core scientific / imaging stack
RUN pip install --no-cache-dir \
        pillow \
        rasterio \
        scipy \
        scikit-image \
        open_clip_torch \
        ftfy \
        regex \
        tqdm \
        huggingface_hub

# MMSegmentation — install mmcv from OpenMMLab's prebuilt CPU wheel index
# (matched to torch 2.1) so nothing compiles from source.
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    pip install --no-cache-dir mmcv==2.1.0 \
        -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html && \
    pip install --no-cache-dir mmsegmentation

# Sanity check at build time — fail the CI build early if imports break.
RUN python -c "import torch, mmcv, mmseg, mmengine, rasterio, skimage, open_clip; \
print('torch', torch.__version__, '| mmcv', mmcv.__version__, '| mmseg', mmseg.__version__)"

# ---- Project code ----
WORKDIR /app
COPY run_single_image.py run_batch.sh ./
COPY docs ./docs

# models/, input_scenes/, results/ are mounted at run time,
# so they are intentionally NOT copied into the image.

# Default command prints help. Override at run time with real arguments.
ENTRYPOINT ["python", "run_single_image.py"]
CMD ["--help"]
