# Model Weights

Place model weights here (not bundled — large and licensed).

```
models/
├── SegEarth-OV-2/                          # AlignEarth codebase + checkpoint
│   └── checkpoint/AlignEarth-SAR-ViT-B-16.pt
└── GeoChat/                                # only for --scene-type auto
    └── weights/GeoChat-7B/
```

Copy from the cluster:
    cp -r ~/sar_data/SegEarth-OV-2  models/
    cp -r ~/sar_data/GeoChat        models/
