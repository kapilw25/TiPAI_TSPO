# TiPAI-TSPO: Full Research Roadmap

## Research Structure (Mirror CITA Paper)

```
Phase 1: DETONATE Dataset     → plan.md (current)
Phase 2: Model Training       → Train 5 baselines + TSPO
Phase 3: Evaluation           → 5 benchmarks, radar/heatmap
Phase 4: Paper                → NeurIPS/CVPR submission
```

---

## Model Comparison Matrix

| Model | Type | Patch-Aware | Tournament |
|-------|------|-------------|------------|
| SD 3.5 vanilla | Baseline | No | No |
| ImageReward | Scalar reward | No | No |
| DiffusionDPO | Preference | No | No |
| RLHF-Diffusion | PPO | No | No |
| **TSPO (ours)** | Tournament | **Yes** | **Yes** |

---

## Evaluation Benchmarks (5)

| Benchmark | Metric | Source |
|-----------|--------|--------|
| Faithfulness | VQAScore / TIFA | T2I-CompBench |
| Policy Compliance | Detector F1 | DETONATE test |
| Patch Attribution | Grad-CAM IoU | Manual annotation |
| Generation Quality | FID / CLIP-Score | COCO-30K |
| Compute Efficiency | GPU-hours / sample | Training logs |

---

## Expected Results Table

| Model | Faithfulness | Policy | Patch IoU | FID↓ | GPU-hrs |
|-------|-------------|--------|-----------|------|---------|
| SD 3.5 | 0.65 | 0.40 | N/A | 12.5 | 0 |
| ImageReward | 0.72 | 0.45 | N/A | 11.8 | 20 |
| DiffusionDPO | 0.74 | 0.50 | N/A | 11.2 | 25 |
| RLHF-Diffusion | 0.76 | 0.55 | N/A | 10.8 | 100 |
| **TSPO** | **0.82** | **0.75** | **0.68** | **10.2** | 30 |

*(Hypothetical targets - to be validated)*

---

## Deliverables

| Phase | Output | Location |
|-------|--------|----------|
| 1 | DETONATE dataset (20K pairs) | HuggingFace |
| 2 | 5 trained models | HuggingFace |
| 3 | Evaluation scripts + results | GitHub |
| 4 | Paper draft | Overleaf |

---

## Key Claim

> TSPO achieves **patch-level alignment** with **tournament efficiency**, outperforming DiffusionDPO on policy compliance while matching RLHF quality at 3x lower compute.
