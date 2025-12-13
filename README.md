# TiPAI-TSPO

Tournament Inpainting for Patch-Level Alignment in Text-to-Image

## Quick Start (GPU Instance)

```bash
# 1. Setup
cd TiPAI_TSPO
python -m venv venv_tipai
source venv_tipai/bin/activate
pip install -r requirements.txt

# 2. Generate images + push to HuggingFace (~2 hrs for 80 images)
python src/m01_generate_images.py --push-hf

# 3. Score with CLIP (~30 min)
# Downloads from HF if not local
python src/m02_clip_scoring.py

# 4. Create heatmaps (~10 min)
python src/m03_create_heatmaps.py

# 5. Create pairs (~5 min)
python src/m04_create_pairs.py

# 6. Launch demo
python src/m05_demo_app.py --port 7860
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
