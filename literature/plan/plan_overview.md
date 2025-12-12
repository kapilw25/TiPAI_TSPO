# TiPAI-TSPO: Complete Research Roadmap

## Key Deviation from Proposal

| Proposal | Our Version | Reason |
|----------|-------------|--------|
| L_policy (NSFW, weapon, symbol, logo, age) | L_faithfulness (prompt-failing regions) | Immigration-safe |
| Policy thresholds tau_P | Faithfulness thresholds tau_F only | No NSFW data needed |
| Per-class risk detection | Single faithfulness risk | HuggingFace/GitHub safe |

**Our "policy" IS faithfulness**: detecting regions that violate the PROMPT, not platform safety rules.

---

## What is TiPAI-TSPO?

**Layman Explanation**: A spell-checker for AI images

```
Text Spell-Checker:                    Image Spell-Checker (TiPAI):

"The cta sat on the mat"               Prompt: "red car on beach"
       ^                               Image: [blue car on beach]
       |                                              ^
  "cta" is wrong!                                     |
  Suggests: "cat"                      "car is wrong color!"
                                       Fixes: [red car on beach]
```

---

## System Overview

```
+-------------------------------------------------------------------------+
|                        TiPAI-TSPO SYSTEM                                |
+-------------------------------------------------------------------------+

                    +-------------------+
                    |      PROMPT       |
                    | "red car on beach"|
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    |   Base Diffuser   |
                    |   (SD 3.5, etc)   |
                    +---------+---------+
                              |
                              v
         +--------------------+--------------------+
         |                                         |
         v                                         v
+------------------+                     +------------------+
|   Stage A        |                     |   Stage B        |
|   JUDGE          |                     |   ARTIST + COACH |
|                  |                     |                  |
| "Is this good?"  |<------------------->| "Fix this region"|
| "Where is bad?"  |      scores         | "Try these knobs"|
+--------+---------+                     +--------+---------+
         |                                        |
         |         +------------------+           |
         +-------->|   Stage C        |<----------+
                   |   CALIBRATION    |
                   |                  |
                   | "Trust scores?"  |
                   | "Accept edit?"   |
                   +--------+---------+
                            |
                            v
                   +------------------+
                   |   Final Image    |
                   | (faithful to     |
                   |  prompt)         |
                   +------------------+
```

---

## Preliminaries and Notation

```
+-------------------------------------------------------------------+
|  Core Notation (from proposal):                                    |
|                                                                    |
|  p           = prompt (text)                                       |
|  {x_t}       = diffusion trajectory (pixel space)                  |
|  {z_t}       = latent trajectory (VAE space)                       |
|  T           = total timesteps (e.g., 50)                          |
|  t           = current timestep (T=noise, 0=clean)                 |
|                                                                    |
|  Phi_base    = base diffusion model (frozen)                       |
|  x_ctrl      = control output (unedited)                           |
|  I_(t-1)     = decoded audit view at step t-1                      |
|                                                                    |
|  R           = region (bounding box)                               |
|  m           = binary mask (1 = edit here)                         |
|  K_t         = number of regions at timestep t                     |
+-------------------------------------------------------------------+
```

### Diffusion Process

```
t = T (start)                    t = 0 (end)
Pure noise                       Clean image
    |                               |
    v                               v
[////////]  -->  [shapes]  -->  [details]  -->  [final]
                     ^               ^
                     |               |
              We can audit and edit at these points
```

### Region Mining

```
+-------------------------------------------------------------------+
|  How we find "bad" regions in I_(t-1):                             |
|                                                                    |
|  1. Stage A Risk Heatmap                                           |
|     - High risk = low faithfulness to prompt                       |
|                                                                    |
|  2. CLIP Grad-CAM                                                  |
|     - Where does image NOT match prompt?                           |
|                                                                    |
|  3. Object Detection Gap                                           |
|     - Prompt says "dog" but no dog detected                        |
|                                                                    |
|  Fuse these signals --> mine K regions with masks                  |
+-------------------------------------------------------------------+
```

---

## Three Training Stages

### Stage A: Auditor-Scorer (As)

```
Goal: Train a JUDGE that can:
  1. Score whole images (S)
  2. Score individual patches (S_R)
  3. Generate risk heatmaps (r)

+-------------------+     +-------------------+
|  Prompt + Image   | --> |   Auditor-Scorer  |
|                   |     |       (As)        |
+-------------------+     +---------+---------+
                                    |
                    +---------------+---------------+
                    |               |               |
                    v               v               v
              +---------+     +---------+     +---------+
              | Global  |     |  Patch  |     |  Risk   |
              | Score S |     | Scores  |     | Heatmap |
              | (0-1)   |     | S_R     |     |  r      |
              +---------+     +---------+     +---------+

Training: DETONATE dataset (20K chosen/rejected pairs)
Losses: L_pair + L_patch + L_faithfulness + L_sal
```

**See**: `plan_stage_A.md` for details

---

### Stage B: Auditor-Inpaint (Ag) + TSPO

```
Goal: Train an ARTIST that can fix regions + COACH that picks settings

B1: Pretrain Ag (Artist)
    - Input: image with masked region + noise
    - Output: fixed region (faithful to prompt)

B2: Tournament (Competition)
    - Generate N=5 candidate fixes
    - Include control (do nothing) as C_0
    - Score all with Stage A
    - Accept only if CLEARLY better (guards)

B3: TSPO (Coach)
    - Learn which "knobs" work best
    - Knobs: mask dilation, CFG, noise, inversion depth
    - Reward actions that lead to winners

+-------------------+     +-------------------+     +-------------------+
|   Bad Region      | --> |   TSPO Policy     | --> |   N Candidates    |
|   (R, mask m)     |     |   (pick knobs)    |     |   C_1 ... C_N     |
+-------------------+     +-------------------+     +---------+---------+
                                                              |
                                                              v
                                                    +-------------------+
                          +-------------------------|   Tournament      |
                          |                         |   + Guards        |
                          v                         +-------------------+
                    +-----------+                             |
                    |  Winner?  |<----------------------------+
                    +-----------+
                          |
              +-----------+-----------+
              |                       |
              v                       v
        +-----------+           +-----------+
        |  Accept   |           |   Keep    |
        |  & Blend  |           |  Control  |
        +-----------+           +-----------+
```

**See**: `plan_stage_B.md` for details

---

### Stage C: Calibration & Deployment

```
Goal: Tune decision rules for reliable operation

1. Calibrate Scores
   - Raw scores --> probabilities
   - Methods: Platt scaling, Isotonic regression

2. Choose Operating Points
   - delta: margin required to beat control
   - tau_F(t): faithfulness threshold per timestep
   - seam_threshold: max boundary artifact

3. Deploy
   - Full inference algorithm
   - Per-patch, per-step decisions
   - Seamless latent blending

+-------------------+
|  Raw Score S=0.8  |
+-------------------+
         |
         v (Platt scaling)
+-------------------+
|  Calibrated p=0.75|  "75% chance this is truly better"
+-------------------+
         |
         v (compare to threshold)
+-------------------+
|  p > delta + p_0? |
|  F > tau_F(t)?    |
+-------------------+
         |
    +----+----+
    |         |
   YES        NO
    |         |
    v         v
 ACCEPT    REJECT
```

**See**: `plan_stage_C.md` for details

---

## Research Phases

| Phase | Task | Output | Time |
|-------|------|--------|------|
| **Phase 1** | DETONATE Dataset | 20K pairs + patches | 2-3 weeks |
| **Phase 2** | Stage A Training | Auditor-Scorer model | 1 week |
| **Phase 3** | Stage B Training | Ag + TSPO policy | 2-3 weeks |
| **Phase 4** | Stage C Calibration | Tuned system | 1 week |
| **Phase 5** | Evaluation | Benchmarks + paper | 2 weeks |
| **Phase 6** | Paper Writing | NeurIPS/CVPR draft | 2-3 weeks |

---

## Model Comparison Matrix

| Model | Type | Patch-Aware | Tournament | Our Advantage |
|-------|------|-------------|------------|---------------|
| SD 3.5 vanilla | Baseline | No | No | - |
| ImageReward | Scalar reward | No | No | Patch-level |
| DiffusionDPO | Preference | No | No | Patch + Tournament |
| RLHF-Diffusion | PPO | No | No | Efficiency |
| **TSPO (ours)** | Tournament | **Yes** | **Yes** | All above |

---

## Evaluation Benchmarks

| Benchmark | Metric | What it measures |
|-----------|--------|------------------|
| VQAScore | 0-1 | Text-image faithfulness |
| TIFA | 0-1 | VQA-based faithfulness |
| T2I-CompBench | Multiple | Compositional generation |
| FID | Lower=better | Image quality |
| CLIP-Score | Higher=better | Semantic alignment |
| Patch IoU | 0-1 | Localization accuracy |

---

## Expected Results

| Model | Faithfulness | Patch IoU | FID | GPU-hrs |
|-------|-------------|-----------|-----|---------|
| SD 3.5 | 0.65 | N/A | 12.5 | 0 |
| ImageReward | 0.72 | N/A | 11.8 | 20 |
| DiffusionDPO | 0.74 | N/A | 11.2 | 25 |
| RLHF-Diffusion | 0.76 | N/A | 10.8 | 100 |
| **TSPO (ours)** | **0.82** | **0.68** | **10.2** | 30 |

*(Hypothetical targets - to be validated)*

---

## Deliverables

| Phase | Output | Location |
|-------|--------|----------|
| 1 | DETONATE dataset (20K pairs) | HuggingFace |
| 2-4 | Trained models (As, Ag, TSPO) | HuggingFace |
| 5 | Evaluation scripts + results | GitHub |
| 6 | Paper draft | Overleaf |

---

## Key Claim

> TiPAI-TSPO achieves **patch-level alignment** with **tournament efficiency**, outperforming DiffusionDPO on faithfulness while matching RLHF quality at 3x lower compute.

---

## File Structure

```
iter/iter_1/
|-- plan_overview.md        # This file (high-level roadmap)
|-- plan_dataset_prep.md    # DETONATE dataset creation
|-- plan_stage_A.md         # Auditor-Scorer training
|-- plan_stage_B.md         # Auditor-Inpaint + TSPO
|-- plan_stage_C.md         # Calibration & deployment
```

---

## Quick Reference: What Each Component Does

| Component | Layman Analogy | Input | Output |
|-----------|----------------|-------|--------|
| **As (Scorer)** | Essay grader | Image + Prompt | Score + Risk map |
| **Ag (Inpainter)** | Photo editor | Masked region + Prompt | Fixed region |
| **TSPO (Policy)** | Camera settings expert | Scene info | Knob settings |
| **Tournament** | American Idol | N candidates | Winner |
| **Guards** | Quality control | Scores | Accept/Reject |
| **Calibration** | Thermometer tuning | Raw scores | Probabilities |
