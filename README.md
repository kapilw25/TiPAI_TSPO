# TiPAI-TSPO

Tournament Inpainting for Patch-Level Alignment in Text-to-Image

## Quick Start (GPU Instance)

```bash
# 1. Setup
cd TiPAI_TSPO
python -m venv venv_tipai
source venv_tipai/bin/activate
pip install -r requirements.txt

# 2. Create logs directory
mkdir -p logs

# 3. Generate images (SDXL on A10-24GB, FLUX on A6000-48GB+)
python -u src/m01_generate_images.py --model sdxl 2>&1 | tee logs/m01_sdxl.log
# python -u src/m01_generate_images.py --model flux --push-hf 2>&1 | tee logs/m01_flux.log

# 4. Download SAM checkpoint (required for signal extraction)
mkdir -p models && wget -O models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 5. Extract 3 signals (SAM + Grad-CAM + Gaps)
python -u src/m02_extract_signals.py 2>&1 | tee logs/m02.log

# 6. Create risk maps (RED overlay on bad segments)
python -u src/m03_create_risk_maps.py 2>&1 | tee logs/m03.log

# 7. Create pairs
python -u src/m04_create_pairs.py 2>&1 | tee logs/m04.log

# 8. Launch demo
python src/m08_demo_app.py --port 7860
```

## HuggingFace Dataset

Dataset: [kapilw25/TiPAI-POC-Faithfulness](https://huggingface.co/datasets/kapilw25/TiPAI-POC-Faithfulness)

- Modules m02-m05 automatically download from HuggingFace if not available locally
- Use `--push-hf` flag with m01 to push after generation

## Documentation

| Document | Description |
|----------|-------------|
| [Proposal](literature/proposal_TiPAI_TSPO.pdf) | Original research proposal |
| [Plan Overview](literature/plan/plan_overview.md) | System design & roadmap |
| [Stage A](literature/plan/plan_stage_A.md) | Auditor-Scorer training |
| [Stage B](literature/plan/plan_stage_B.md) | Auditor-Inpaint + TSPO |
| [Stage C](literature/plan/plan_stage_C.md) | Calibration & deployment |
| [Dataset Prep](literature/plan/plan_dataset_prep.md) | DETONATE dataset creation |
| [POC Plan](iter/iter_1/poc_heatmap_demo.md) | 12-hour POC implementation |

## Requirements

- CUDA GPU (no CPU fallback)
- Python 3.10+
- ~40GB disk space for models + outputs
