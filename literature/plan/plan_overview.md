# TiPAI-TSPO: Complete Research Roadmap

## Implementation Strategy: Incremental Complexity

### Options (Do B First, Then A if Needed)

| Option | What to Do | Effort | Output |
|--------|------------|--------|--------|
| **C: Just Ship** | Notebook + random 5 picks + threshold | ~0 GPU hrs | Working product, no paper |
| **B: Simplified** ⬅️ START HERE | Use HF scores + Train TSPO only | ~40 GPU hrs | Working system, Tier-2 paper |
| **A: Full Professor** | Train Stage A scorer + TSPO + Calibration | ~170 GPU hrs | Tier-1 paper |

### Why TSPO Over Other *PO Algorithms?

| Algorithm | Preference Pairs? | Approach | Why TSPO is Better |
|-----------|-------------------|----------|-------------------|
| **PPO** | ❌ No (reward model) | Actor-critic RL | Needs critic, less stable |
| **GRPO** | ❌ No (group relative) | Group comparison | No preference pairs |
| **DPO** | ✅ Yes | **Pairwise** (2 items) | Only compares 2 items |
| **TSPO** | ✅ Yes | **Listwise** (N=5 items) | Ranks all candidates + diversity + compute |

**TSPO = DPO's preference pairs + Listwise ranking + Diversity/Compute regularizers**

Like CITA beat DPO with instruction-conditioning + KL control,
TSPO beats DPO with listwise ranking + diversity + compute efficiency.

---

### Phase 1: Option B (Simplified TiPAI-TSPO)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: SIMPLIFIED (Use HF Scores + Train TSPO Only)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FROZEN (from notebook):              TRAIN (only novelty):                 │
│  ─────────────────────────            ─────────────────────                 │
│  • NudeNet detection                  • TSPO Policy                         │
│  • HF classifiers (scoring)             → Learn which CFG/seed/steps work   │
│  • SD Inpainting                        → ~40 GPU hours                     │
│  • Guards (just threshold)                                                  │
│                                                                              │
│  Pick best candidate using HF scores (no Stage A training!)                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Option A (Full Professor's Proposal) - OPTIONAL

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: FULL (Add Stage A Scorer) - Only if Phase 1 results insufficient  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ADD training:                        WHY:                                  │
│  ─────────────                        ────                                  │
│  • Stage A Scorer                     • Judge seam quality (HF can't)       │
│    → Train on DETONATE pairs          • Judge faithfulness (HF can't)       │
│    → ~20 GPU hours                    • Composite score for Tier-1 paper    │
│                                                                              │
│  • Calibration                        • Interpretable thresholds            │
│    → Platt scaling                    • Mathematical guarantees             │
│    → ~10 GPU hours                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Comparison: Phase 1 vs Phase 2

| Aspect | Phase 1 (HF Scores) | Phase 2 (Trained Scorer) |
|--------|---------------------|--------------------------|
| Detects NSFW | ✅ Yes | ✅ Yes |
| Judges seam quality | ❌ No | ✅ Yes |
| Judges scene match | ❌ No | ✅ Yes |
| GPU Hours | ~40 | ~170 |
| Paper Target | Tier-2 | Tier-1 |

---

## Research Focus: AI Safety (NSFW Detection + Censoring)

| Aspect | Description |
|--------|-------------|
| **Goal** | Detect and censor NSFW/toxic regions in AI-generated images |
| **Method** | Patch-level detection + Tournament-based inpainting |
| **Dataset** | T2ISafety (CVPR 2025) - 70K prompts, 68K annotated images |
| **Baselines** | Safe Latent Diffusion, Erasing Concepts, NudeNet, etc. |

---

## Tier-1 Conference Requirements (CRITICAL)

**Lesson from CITA**: To get accepted at Tier-1 venues (NeurIPS, CVPR, ICCV), we must show:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIER-1 ACCEPTANCE REQUIREMENTS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WHAT CITA DID (Tier-1 Standard):                                           │
│  ────────────────────────────────                                           │
│  1. TRAINED all baselines (DPO, GRPO, PPO) from same checkpoint             │
│  2. Showed training curves for ALL methods (not just CITA)                  │
│  3. Demonstrated CITA's margin grew HIGHER during training                  │
│  4. Final evaluation showed CITA >> others (86.7% vs 56.1%)                │
│                                                                              │
│  WHAT TiPAI-TSPO MUST DO (Same Standard):                                   │
│  ────────────────────────────────────────                                   │
│  1. TRAIN all baselines from same checkpoint (NOT frozen)                   │
│  2. Show training curves for ALL methods                                    │
│  3. Demonstrate TiPAI-TSPO margin grows HIGHER during training              │
│  4. Final evaluation shows TiPAI-TSPO >> others                             │
│                                                                              │
│  ✗ INSUFFICIENT: Compare against frozen NudeNet only                        │
│  ✓ REQUIRED: Train Safe LD, Erasing Concepts, etc. and compare curves       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Dynamics Comparison (Like CITA)

```
Safety Margin (during training)
│
│      ──────────────────────  TiPAI-TSPO (should be HIGHEST)
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   Safe Latent Diffusion (trained)
│  · · · · · · · · · · · · ·   Erasing Concepts (trained)
│  - - - - - - - - - - - - -   Forget-Me-Not (trained)
│
│────────────────────────────────────────────► Training Steps
     0    1K   2K   3K   4K   5K

KEY: All baselines TRAINED, all have training curves, TiPAI-TSPO wins
```

### Baselines: TRAINED vs FROZEN

| Baseline | Status | Training Required | Notes |
|----------|--------|-------------------|-------|
| **Safe Latent Diffusion** | MUST TRAIN | Yes | Concept suppression loss |
| **Erasing Concepts** | MUST TRAIN | Yes | Fine-tuning erasure |
| **Forget-Me-Not** | MUST TRAIN | Yes | Attention resteering |
| **SAFREE** | FROZEN (OK) | No | Training-free by design |
| **NudeNet + Inpaint** | FROZEN (reference) | No | Inference-only baseline |
| **TiPAI-TSPO (ours)** | MUST TRAIN | Yes | Our method |

**Note**: SAFREE and NudeNet are frozen by design, but we still need 3+ trained baselines for fair comparison.

### Scope Impact

| Aspect | Original Plan | Tier-1 Requirement |
|--------|--------------|-------------------|
| Baselines trained | 0 (all frozen) | 3+ TRAINED |
| Training curves shown | TiPAI-TSPO only | ALL methods |
| Evaluation type | Final scores only | Training dynamics + Final |
| Estimated time | 6-8 weeks | 12-16 weeks |
| GPU hours | ~100 | ~400 |

---

## Visa Guardrails (Immigration-Safe Research)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VISA GUARDRAILS                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ✓ ALLOWED (Safe for F-1 Visa)          ✗ NOT ALLOWED (Risky)               │
│  ─────────────────────────────          ─────────────────────               │
│  • Download existing datasets           • Generate new NSFW content          │
│    (T2ISafety, MMA-Diffusion)           • Create adversarial attacks         │
│  • Train DETECTION models               • Store unencrypted NSFW locally     │
│  • Train CENSORING/recovery models      • Distribute NSFW content            │
│  • Publish DEFENSE papers               • Bypass safety filters              │
│  • Use university compute               • Use personal accounts for NSFW     │
│                                                                              │
│  KEY PRINCIPLE: We DETECT and FIX, we don't GENERATE                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Action | Risk Level | Recommendation |
|--------|------------|----------------|
| Download T2ISafety dataset | Low | Use university compute |
| Train on existing NSFW images | Low-Medium | Document research purpose |
| Generate new NSFW images | **High** | **DO NOT DO** |
| Publish defense/detection paper | Low | Standard academic practice |

---

## Building TiPAI-TSPO on NudeNet Notebook (Layman Explanation)

### What the Notebook Does (Current - Works!)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  NOTEBOOK PIPELINE (Simple Chef)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: NudeNet looks at image                                             │
│          → "I see exposed breast at (x=100, y=200)"                         │
│          → Outputs: bounding boxes                                          │
│                                                                              │
│  Step 2: HF Classifiers double-check                                        │
│          → "Yes, this image is 85% NSFW"                                    │
│          → Confirms NudeNet's detection                                     │
│                                                                              │
│  Step 3: Generate ONE mask (white region to cover)                          │
│          → Dilate the boxes a bit                                           │
│                                                                              │
│  Step 4: SD Inpainting fills in mask                                        │
│          → "Put modest clothing here"                                       │
│          → Outputs: ONE censored image                                      │
│                                                                              │
│  DONE. Take what you get.                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Analogy**: Like a chef who makes ONE dish and serves it. No choice, no quality check.

---

### What Professor's Proposal ADDS (Three Stages)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PROFESSOR'S TiPAI-TSPO (Smart Restaurant)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  KEEP: NudeNet + HF Classifiers (detection) ← FROZEN, don't touch           │
│                                                                              │
│  ADD Stage A: "Food Critic" (Auditor-Scorer)                                │
│  ─────────────────────────────────────────────                              │
│  Instead of just "is it NSFW?", ask "HOW safe is it? (0-100)"               │
│                                                                              │
│  • Input: Any image                                                         │
│  • Output: Safety SCORE (e.g., "this image is 23% safe")                    │
│  • Also: WHERE is it unsafe? (risk heatmap)                                 │
│                                                                              │
│  Training: Show 100K pairs of (unsafe, safe) images                         │
│            → Learn "safe image should score HIGHER than unsafe"             │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ADD Stage B: "5 Chefs Competition" (Tournament + TSPO)                     │
│  ─────────────────────────────────────────────────────────                  │
│                                                                              │
│  B1: Train SD Inpainter to be timestep-aware                                │
│      (different settings for different situations)                          │
│                                                                              │
│  B2: TOURNAMENT - Instead of 1 output, make 5 different ones:               │
│      ┌───────────────────────────────────────────────────────┐              │
│      │  Chef 1: CFG=7, seed=123   → Candidate 1 (score: 72)  │              │
│      │  Chef 2: CFG=9, seed=456   → Candidate 2 (score: 85)  │ ← WINNER!   │
│      │  Chef 3: CFG=11, seed=789  → Candidate 3 (score: 68)  │              │
│      │  Chef 4: CFG=5, seed=111   → Candidate 4 (score: 55)  │              │
│      │  Chef 5: Do nothing        → Control C_0 (score: 20)  │              │
│      └───────────────────────────────────────────────────────┘              │
│                                                                              │
│      Use Stage A's scorer to rate each candidate!                           │
│      Pick the one with HIGHEST safety score!                                │
│                                                                              │
│  B3: GUARDS - Only accept winner if it's CLEARLY better than doing nothing  │
│      → If Score(winner) - Score(control) < threshold, keep original         │
│      → Prevents bad edits from making things WORSE                          │
│                                                                              │
│  B4: TSPO Policy - Learn WHICH settings (knobs) work best                   │
│      → "For this type of image, CFG=9 usually wins"                         │
│      → Reduces the 5 tries needed over time                                 │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ADD Stage C: "Quality Control Manager" (Calibration)                       │
│  ────────────────────────────────────────────────────                       │
│                                                                              │
│  Tune the thresholds so decisions are RELIABLE:                             │
│  • δ (delta): "How much better must winner be?" (e.g., +10 points)          │
│  • τ_P: "Minimum safety score to accept" (e.g., must be > 70%)              │
│  • τ_F: "Minimum faithfulness" (don't change the image too much)            │
│                                                                              │
│  Like a manager who says: "Only serve the dish if it scores > 80            │
│  AND it's at least 10 points better than the original"                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Side-by-Side Comparison

| Aspect | Notebook (Current) | TiPAI-TSPO (Professor's) |
|--------|-------------------|--------------------------|
| **Detection** | NudeNet + HF ✓ | Same (FROZEN) |
| **Scoring** | Binary (NSFW/safe) | Continuous (0-100) |
| **Inpaint outputs** | 1 | 5 (tournament) |
| **Quality control** | None (take what you get) | Guards (reject bad edits) |
| **Learning** | None (all frozen) | Stage A Scorer + TSPO Policy |

---

### Visual Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  INPUT IMAGE (NSFW)                                                         │
│        │                                                                     │
│        ▼                                                                     │
│  ┌──────────────────────────────────────────┐                               │
│  │  NudeNet + HF (FROZEN from notebook)     │                               │
│  │  "Breast detected at (100, 200)"         │                               │
│  └──────────────────────┬───────────────────┘                               │
│                         │                                                    │
│                         ▼                                                    │
│  ┌──────────────────────────────────────────┐                               │
│  │  Stage A: Scorer (NEW - TRAINED)         │                               │
│  │  "Original image safety = 20/100"        │                               │
│  │  "Risk map shows problem at breast area" │                               │
│  └──────────────────────┬───────────────────┘                               │
│                         │                                                    │
│                         ▼                                                    │
│  ┌──────────────────────────────────────────┐                               │
│  │  Stage B: Tournament (NEW)               │                               │
│  │                                          │                               │
│  │  SD Inpaint x 5 different settings:      │                               │
│  │  C1=72, C2=85✓, C3=68, C4=55, C0=20     │                               │
│  │                                          │                               │
│  │  Winner: C2 (score 85)                   │                               │
│  │  Margin: 85-20 = 65 > δ=10 ✓            │                               │
│  └──────────────────────┬───────────────────┘                               │
│                         │                                                    │
│                         ▼                                                    │
│  ┌──────────────────────────────────────────┐                               │
│  │  Stage C: Calibration (NEW)              │                               │
│  │  "Score 85 > threshold 70? ✓"            │                               │
│  │  "Margin 65 > required 10? ✓"            │                               │
│  │  → ACCEPT edit, output C2                │                               │
│  └──────────────────────────────────────────┘                               │
│                         │                                                    │
│                         ▼                                                    │
│  OUTPUT: Best censored image (C2)                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### TL;DR

| What | Analogy |
|------|---------|
| **Notebook** | Chef makes 1 dish, you eat it |
| **Stage A** | Hire a food critic to score dishes |
| **Stage B** | Make 5 chefs compete, pick winner |
| **Stage C** | Manager sets quality standards |

**Professor's proposal = Notebook + Quality Control System**

---

## POC: Working Prototype

**Notebook**: `literature/NSFW_patch_detection_generation_COPY.ipynb`

**Dataset**: [T2ISafety (CVPR 2025)](https://huggingface.co/datasets/OpenSafetyLab/t2i_safety_dataset) - 70K prompts, 68K annotated images

### Pipeline Output (4-panel)

![POC Output 1](../nudity_patch_1.png)

| Panel | Content | TiPAI Equivalent |
|-------|---------|------------------|
| Original | Input NSFW image | Input to Auditor |
| Outlined | Red boxes on detected regions | Region mining output |
| Mask | Binary mask (dilated) | Risk heatmap |
| Censored | SD-Inpainted safe version | Painter output |

### Pipeline Output (5-panel with uncombined masks)

![POC Output 2](../nudity_patch_2.png)

| Panel | Content | Purpose |
|-------|---------|---------|
| Uncombined Masks | Individual boxes per detection | Fine-grained localization |
| Combined Mask | Merged bounding box | Coherent inpainting region |

### POC Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POC PIPELINE (Notebook)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT IMAGE (potentially NSFW)                                              │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐             │
│  │  NudeNet    │───▶│  HF Classifiers  │───▶│  Ensemble Vote  │             │
│  │  Detector   │    │  (Falconsai +    │    │  (score > 0.7)  │             │
│  │             │    │   AdamCodd)      │    │                 │             │
│  └─────────────┘    └──────────────────┘    └────────┬────────┘             │
│       │                                              │                       │
│       │ bounding boxes                               │ is_nsfw?              │
│       ▼                                              ▼                       │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │              MASK GENERATION                         │                    │
│  │  • Uncombined: individual boxes per detection        │                    │
│  │  • Combined: merged bounding box + dilation          │                    │
│  └─────────────────────────────────────────────────────┘                    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │         STABLE DIFFUSION INPAINTING                  │                    │
│  │  Model: runwayml/stable-diffusion-inpainting         │                    │
│  │  Prompt: "modest clothing covering exposed areas"    │                    │
│  │  Steps: 60, Strength: 0.9, Guidance: 9.5            │                    │
│  └─────────────────────────────────────────────────────┘                    │
│       │                                                                      │
│       ▼                                                                      │
│  SAFE OUTPUT (NSFW regions replaced with clothing)                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## CRITICAL INSIGHT: Build ON TOP of NudeNet, Don't Replace It

### Problem with Current src/ Code

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CURRENT CODE IS FOR FAITHFULNESS, NOT SAFETY!                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  src/m01_generate_images.py  ─┐                                             │
│  src/m02_extract_signals.py   ├── Detects "red car vs purple car"           │
│  src/m03_create_risk_maps.py ─┘   NOT nude regions!                         │
│                                                                              │
│  Risk map output (scene_01_v1_seed42_risk_map.png):                         │
│  • Highlights color change (red → purple)                                   │
│  • Wrong object outlines                                                    │
│  • USELESS for NSFW detection                                               │
│                                                                              │
│  Meanwhile, NudeNet + Inpaint (POC notebook):                               │
│  • Detects nude regions PERFECTLY                                           │
│  • SD Inpainting covers them WELL                                           │
│  • Already working!                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Code to DELETE

```bash
# These files are for FAITHFULNESS, not SAFETY - DELETE or ARCHIVE
src/m01_generate_images.py      # Generates faithful/unfaithful pairs
src/m02_extract_signals.py      # Extracts CLIP signals (not NSFW)
src/m03_create_risk_maps.py     # Creates risk maps for color changes
src/m04_create_pairs.py         # Creates pairs for faithfulness
utilities/signals.py            # CLIP-based signals (not NudeNet)
```

### Code to BUILD (Phase 1 - Simplified)

```bash
# Phase 1: Use HF scores + Train TSPO only
src/m01_nsfw_detection.py       # Wrapper around NudeNet + HF ensemble (FROZEN)
src/m02_multi_candidate.py      # Generate N=5 inpaint candidates
src/m03_hf_scorer.py            # Use HF classifiers to score (NO TRAINING)
src/m04_tournament.py           # Pick best candidate based on HF scores
src/m05_guards.py               # Accept/reject logic (threshold, NO TRAINING)
src/m06_tspo_policy.py          # TRAIN: Learn which settings work (ONLY TRAINING!)
src/m07_inference.py            # Full pipeline
```

### Code to BUILD (Phase 2 - Full) - OPTIONAL

```bash
# Phase 2: Add Stage A Scorer training (only if Phase 1 insufficient)
src/m03_stage_a_scorer.py       # TRAIN: Score safety + seam + faithfulness
src/m08_calibration.py          # Platt scaling for thresholds
```

---

## Integration with T2ISafety Dataset

**Dataset**: [OpenSafetyLab/t2i_safety_dataset](https://huggingface.co/datasets/OpenSafetyLab/t2i_safety_dataset)

### T2ISafety Taxonomy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    T2ISafety TAXONOMY (CVPR 2025)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TOXICITY (6 tasks)              FAIRNESS (3 tasks)      PRIVACY (3 tasks)  │
│  ──────────────────              ─────────────────       ─────────────────  │
│  • Sexual content ◄── PRIMARY    • Gender bias           • Public figures   │
│  • Hate speech                   • Age bias              • Personal IDs     │
│  • Humiliation                   • Race bias             • IP violation     │
│  • Violence                                                                 │
│  • Illegal activity                                                         │
│  • Disturbing content                                                       │
│                                                                              │
│  Total: 70K prompts, 68K annotated images                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Download

```bash
# Option 1: huggingface-cli
pip install huggingface_hub
huggingface-cli download OpenSafetyLab/t2i_safety_dataset --local-dir ./t2i_safety_data

# Option 2: Python
from datasets import load_dataset
dataset = load_dataset("OpenSafetyLab/t2i_safety_dataset")

# Combine split files after download
cat train.zip.part-a* > train.zip
unzip train.zip
```

### Create Training Pairs

```python
from datasets import load_dataset

# Load T2ISafety (toxicity subset for NSFW)
dataset = load_dataset("OpenSafetyLab/t2i_safety_dataset")

# Filter for toxicity domain
train_data = [
    item for item in dataset['train']
    if item['base_category'] == 'toxicity'
]

# Each item has:
# - image: the unsafe image
# - image_category: 'sexual', 'violence', etc.
# - conversations: [{"from": "assistant", "value": "unsafe sexual"}]

def create_training_pair(nsfw_image_path):
    """Create (unsafe, safe) pair for Stage A training."""

    # Unsafe = original NSFW
    unsafe = Image.open(nsfw_image_path)

    # Safe = censored version (use notebook pipeline)
    original, outlined, uncombined_mask, combined_mask, censored = process_image(nsfw_image_path)
    safe = censored

    # Patch labels = mask (1 where NSFW was detected)
    patch_labels = combined_mask

    return {
        'unsafe': unsafe,
        'safe': safe,
        'patch_labels': patch_labels,
        'category': 'toxicity'
    }
```

---

## Evaluation Plan: Learning from CITA (LLM) → TiPAI-TSPO (T2I)

### CITA Project Reference (LLM Alignment)

Our prior work **CITA** demonstrated instruction-conditioned alignment for LLMs. We apply the same evaluation philosophy to TiPAI-TSPO.

**CITA Evaluation Radar (5 Axes):**

```
                         TruthfulQA (M₂)
                              +0.054
                                │
                                │
    Cond. Safety (M₃)           │           ECLIPTICA (M₁)
         +0.475 ────────────────┼──────────── +0.172
                               ╱│╲
                              ╱ │ ╲
                             ╱  │  ╲
                            ╱   │   ╲
    Length Ctrl (M₄)       ╱    │    ╲      LITMUS (AQI-M₅)
         +0.164 ──────────╱─────┴─────╲───── +26.388


    Results (Average Radius = Overall Performance):
    ┌─────────────────────────────────────┐
    │  CITA:  86.7%  ████████████████████ │  ◄── WINNER
    │  DPO:   56.1%  ███████████          │
    │  GRPO:  36.1%  ███████              │
    │  PPO:   20.4%  ████                 │
    └─────────────────────────────────────┘
```

---

### TiPAI-TSPO Training Pipeline (T2I Safety)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TiPAI-TSPO TRAINING PIPELINE (T2I Safety)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SD 3.5 / FLUX ──▶ Base ──┬──▶ Safe Latent Diffusion  (Concept suppression) │
│  (Pretrained)      │      │                                                  │
│                    │      ├──▶ Erasing Concepts       (Concept erasure)      │
│                    │      │                                                  │
│                    │      ├──▶ NudeNet + Inpaint      (Detection + fix)      │
│                    │      │                                                  │
│                    │      ├──▶ SAFREE                 (Training-free)        │
│                    │      │                                                  │
│                    │      └──▶ TiPAI-TSPO             ◄── OUR METHOD         │
│                              (Patch-level + Tournament + Guards)             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Baselines to Compare (Safety-Focused) - TIER-1 COMPLIANT

**CRITICAL**: For Tier-1, we must TRAIN competing methods and compare training curves.

| Method | Paper | Approach | TRAIN? | Training Curve? |
|--------|-------|----------|--------|-----------------|
| SD 3.5 vanilla | - | None | ❌ No | N/A |
| **Safe Latent Diffusion** | CVPR 2023 | Concept suppression | ✅ **YES** | ✅ **REQUIRED** |
| **Erasing Concepts** | ICCV 2023 | Fine-tuning erasure | ✅ **YES** | ✅ **REQUIRED** |
| **Forget-Me-Not** | CVPR 2024 | Attention resteering | ✅ **YES** | ✅ **REQUIRED** |
| **NudeNet + Inpaint** | - | Detection + inpaint | ❌ Frozen | N/A (reference) |
| **SAFREE** | arXiv 2024 | Training-free | ❌ Frozen | N/A (by design) |
| **TiPAI-TSPO (ours)** | - | Patch + Tournament | ✅ **YES** | ✅ **REQUIRED** |

**Feature Comparison:**

| Method | Patch-Aware | Tournament | Post-hoc Fix | Guards |
|--------|-------------|------------|--------------|--------|
| Safe Latent Diffusion | No | No | No | No |
| Erasing Concepts | No | No | No | No |
| Forget-Me-Not | No | No | No | No |
| NudeNet + Inpaint | Yes | No | Yes | No |
| SAFREE | No | No | No | No |
| **TiPAI-TSPO (ours)** | **Yes** | **Yes** | **Yes** | **Yes** |

---

### Primary Evaluation Metric: Mitigation Rate

| Metric | What it Measures | Industry Standard? |
|--------|------------------|-------------------|
| **Mitigation Rate / Erasure Proportion** | % NSFW successfully removed | ✅ Yes ([arXiv benchmark](https://arxiv.org/html/2502.12527v1)) |
| **Nudity Removal Rate (NRR)** | Reduction in exposed body parts | ✅ Yes |

**Focus on Mitigation Rate** - this is the primary metric for Phase 1.

### Why TSPO Beats DPO (Both Use Preference Pairs)

| Feature | DPO | TSPO | Why TSPO Wins |
|---------|-----|------|---------------|
| Comparison | Pairwise (2 items) | **Listwise (N=5 items)** | Ranks all candidates, not just 2 |
| Diversity | ❌ Not considered | ✅ **Diversity regularizer** | Encourages varied candidates |
| Compute | ❌ Not considered | ✅ **Compute regularizer** | Learns to use fewer tries |
| Tournament | ❌ | ✅ | Picks best from N candidates |

**Research evidence:** [ADPO](https://arxiv.org/html/2510.18913) shows listwise beats pairwise by 12-93%.

### Secondary Metrics (Optional)

| Axis | Benchmark | What it Measures | Metric |
|------|-----------|------------------|--------|
| **M₁** | T2ISafety (Sexual) | Sexual content detection | Precision/Recall/F1 |
| **M₂** | T2ISafety (Violence) | Violence detection | Precision/Recall/F1 |
| **M₃** | Patch IoU | Localization accuracy | IoU with ground truth |
| **M₄** | Censoring Quality (FID) | Image quality after censoring | Lower is better |

---

### Radar Chart Design for TiPAI-TSPO

```
                    Sexual Detection (M₁)
                            │
                            │
    Mitigation Rate (M₅)    │         Violence Detection (M₂)
         ───────────────────┼─────────────────
                           ╱│╲
                          ╱ │ ╲
                         ╱  │  ╲
                        ╱   │   ╲
    FID Quality (M₄)   ╱    │    ╲        Patch IoU (M₃)
         ─────────────╱─────┴─────╲────── Localization


    Expected Results:
    ┌─────────────────────────────────────┐
    │  TiPAI-TSPO: 85%  ██████████████████│  ◄── TARGET
    │  NudeNet+Inp: 72% ██████████████    │
    │  Safe LD:     65% ████████████      │
    │  Erasing:     60% ██████████        │
    │  SAFREE:      55% █████████         │
    │  SD 3.5:      30% █████             │
    └─────────────────────────────────────┘
```

---

### Detailed Benchmark Specifications

#### M₁: T2ISafety Sexual Detection

| Sub-category | # Images | Metric |
|--------------|----------|--------|
| Sexual violence | ~300 | Precision/Recall |
| Pornography | ~300 | Precision/Recall |
| Nudity | ~300 | Precision/Recall |

#### M₂: T2ISafety Violence Detection

| Sub-category | # Images | Metric |
|--------------|----------|--------|
| Physical harm | ~300 | Precision/Recall |
| Bloody/graphic | ~300 | Precision/Recall |
| Domestic violence | ~300 | Precision/Recall |

#### M₃: Patch IoU (Localization - OUR UNIQUE METRIC)

```
┌─────────────────────────────────────────┐
│  Ground Truth Mask    Predicted Mask    │
│  (actual NSFW region) (detected region) │
│  ┌───────────┐        ┌───────────┐     │
│  │     ██    │        │    ███    │     │
│  │    ████   │   vs   │   █████   │     │
│  │     ██    │        │    ███    │     │
│  └───────────┘        └───────────┘     │
│                                          │
│  IoU = Intersection / Union              │
│  Higher = Better localization            │
└─────────────────────────────────────────┘
```

#### M₄: Censoring Quality (FID)

| Score | Quality |
|-------|---------|
| < 15 | Excellent (seamless censoring) |
| 15-30 | Good |
| 30-50 | Acceptable |
| > 50 | Poor (visible artifacts) |

#### M₅: Mitigation Rate

```
Mitigation Rate = (# NSFW images successfully censored) / (# total NSFW images)

Success criteria:
  - NSFW region covered/removed
  - No new NSFW content introduced
  - Image still looks natural
```

---

### CITA vs TiPAI-TSPO Comparison

| Aspect | CITA (LLM) | TiPAI-TSPO (T2I Safety) |
|--------|------------|-------------------------|
| **Domain** | Text generation | Image safety |
| **Base Model** | Llama-3.1-8B | SD 3.5 / FLUX |
| **Baselines** | PPO, DPO, GRPO | Safe LD, Erasing, NudeNet |
| **Unique Feature** | Instruction-conditioned + KL | Patch-level + Tournament + Guards |
| **# Eval Axes** | 5 | 5 |
| **Radar Metric** | Average % | Average % |
| **Key Win** | 86.7% vs DPO 56.1% | Target: 85% vs NudeNet 72% |

---

### Expected Results

| Model | Sexual (M₁) | Violence (M₂) | Patch IoU (M₃) | FID (M₄) | Mitig. (M₅) | Avg % |
|-------|-------------|---------------|----------------|----------|-------------|-------|
| SD 3.5 (no safety) | 0.20 | 0.25 | N/A | N/A | 0.00 | 30% |
| SAFREE | 0.55 | 0.50 | N/A | 18.0 | 0.45 | 55% |
| Erasing Concepts | 0.60 | 0.55 | N/A | 22.0 | 0.55 | 60% |
| Safe Latent Diff. | 0.65 | 0.60 | N/A | 20.0 | 0.60 | 65% |
| NudeNet + Inpaint | 0.75 | 0.68 | 0.55 | 16.0 | 0.72 | 72% |
| **TiPAI-TSPO (ours)** | **0.88** | **0.82** | **0.72** | **12.0** | **0.85** | **85%** |

*(Hypothetical targets - to be validated)*

---

### Implementation Checklist (TIER-1 COMPLIANT)

**Phase 0: Dataset Preparation**
| Task | Status | Notes |
|------|--------|-------|
| Download T2ISafety dataset | ❌ | Use university compute |
| Create training pairs (unsafe, safe) | ❌ | For all methods |

**Phase 1: Train ALL Baselines (TIER-1 REQUIREMENT)**
| Task | Status | Notes | GPU Hours |
|------|--------|-------|-----------|
| Train Safe Latent Diffusion | ❌ | CVPR 2023 loss function | ~30 |
| Train Erasing Concepts | ❌ | ICCV 2023 fine-tuning | ~30 |
| Train Forget-Me-Not | ❌ | CVPR 2024 attention | ~30 |
| Train TiPAI-TSPO Stage A | ❌ | Our method | ~20 |
| Train TiPAI-TSPO Stage B | ❌ | Our method | ~40 |

**Phase 2: Training Curves (TIER-1 REQUIREMENT)**
| Task | Status | Notes |
|------|--------|-------|
| Log Safety Margin during training (ALL methods) | ❌ | TensorBoard |
| Log Eval Loss during training (ALL methods) | ❌ | TensorBoard |
| Generate comparison plots (like CITA) | ❌ | matplotlib |

**Phase 3: Final Evaluation**
| Task | Status | Notes |
|------|--------|-------|
| Implement NudeNet + Inpaint baseline | ✓ | POC notebook (frozen) |
| Implement SAFREE baseline | ❌ | Training-free (frozen) |
| Compute Sexual Detection (M₁) | ❌ | All methods |
| Compute Violence Detection (M₂) | ❌ | All methods |
| Compute Patch IoU (M₃) | ❌ | Our unique metric |
| Compute FID (M₄) | ❌ | Standard metric |
| Compute Mitigation Rate (M₅) | ❌ | All methods |
| Generate radar chart | ❌ | matplotlib |

**Total Estimated GPU Hours**: ~150-200 (A100)

---

### Key Differentiator

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WHY TiPAI-TSPO SHOULD WIN                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Safe Latent Diffusion:           TiPAI-TSPO:                               │
│  ─────────────────────            ───────────                               │
│  • Suppresses during generation   • Detects + fixes post-generation         │
│  • May miss some NSFW             • Patch-level precision                   │
│  • No localization                • Explicit risk heatmaps                  │
│  • Single output                  • Tournament (best of N)                  │
│                                   • Guards (reject bad fixes)               │
│                                                                              │
│  KEY ADVANTAGE: Works on ANY image (not just during generation)             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Research Phases (INCREMENTAL APPROACH)

### PHASE 1: Simplified TiPAI-TSPO (~40 GPU hrs) ⬅️ START HERE

| Step | Task | Output | GPU Hours |
|------|------|--------|-----------|
| 1.1 | Setup notebook as base (NudeNet + HF + SD Inpaint) | Working pipeline | 0 |
| 1.2 | Add multi-candidate generation (N=5) | 5 candidates per image | 0 |
| 1.3 | Add tournament (pick best by HF score) | Best candidate selection | 0 |
| 1.4 | Add guards (threshold-based, no training) | Accept/reject logic | 0 |
| 1.5 | **Train TSPO policy** (ONLY TRAINING) | Learn best settings | ~40 |
| 1.6 | Evaluate vs NudeNet+Inpaint baseline | Comparison results | ~5 |

**Phase 1 Total**: ~45 GPU hours → Working system, possible Tier-2 paper

---

### PHASE 2: Full Professor's Proposal (~130 GPU hrs) - OPTIONAL

| Step | Task | Output | GPU Hours |
|------|------|--------|-----------|
| 2.1 | Train Stage A Scorer (seam + faithfulness) | Trained scorer | ~20 |
| 2.2 | Train ALL baselines (Safe LD, Erasing, Forget-Me-Not) | 3 trained baselines | ~90 |
| 2.3 | Add Calibration (Platt scaling) | Interpretable thresholds | ~10 |
| 2.4 | Generate Training Curves (ALL methods) | Plots like CITA | 0 |
| 2.5 | Final Evaluation vs trained baselines | Benchmarks + radar | ~10 |

**Phase 2 Total**: ~130 GPU hours → Tier-1 paper

---

### Decision Point After Phase 1

```
IF Phase 1 results are good enough (e.g., 80%+ mitigation rate):
  → Submit to Tier-2 venue OR
  → Skip to Phase 2 for Tier-1

IF Phase 1 results show seam/faithfulness issues:
  → Phase 2 is needed (train Stage A scorer)
```

**Combined Total**: ~175 GPU hours (A100)

---

## Key Claims

### Phase 1 Claim (Simplified - Tier-2)

> TiPAI-TSPO uses **tournament-based inpainting** with **TSPO policy optimization** to select the best censoring candidate from N=5 options, outperforming single-output NudeNet+Inpaint baseline on NSFW mitigation rate.

**Phase 1 Evidence:**
1. TSPO learns better settings than random (ablation)
2. Tournament (N=5) beats single output (N=1)
3. Mitigation rate improvement over baseline

---

### Phase 2 Claim (Full - Tier-1) - OPTIONAL

> TiPAI-TSPO achieves **patch-level NSFW detection and censoring** with **tournament-based optimization** and **trained composite scorer**, demonstrating **superior training dynamics** and **better final performance** on safety, seam quality, and faithfulness compared to Safe Latent Diffusion, Erasing Concepts, and Forget-Me-Not.

**Phase 2 Evidence Required:**
1. Training curves showing TiPAI-TSPO's margin > all baselines
2. Final evaluation showing TiPAI-TSPO >> others on 5 axes
3. Ablation studies on Tournament, Guards, TSPO, Stage A Scorer

