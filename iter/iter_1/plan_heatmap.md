# TiPAI: Object-Level Faithfulness Detection with Risk Maps

## Goal

```
Show: "We can automatically find WHERE an image fails the prompt at OBJECT level"

Input:  Prompt = "a red car on the beach"
        Image  = [blue car on beach]

Output:
+-------------------+     +-------------------+
|                   |     |  Risk Map         |
|   [BLUE CAR]      | --> |  (RED overlay on  |
|   [BEACH]         |     |   car region)     |
+-------------------+     +-------------------+

Key: RED = issue detected (wrong color, missing object, low score)
     No overlay = correct object matching baseline prompt
```

---

## Architecture: 3-Signal Approach

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SIGNAL EXTRACTION (Object-Level, NOT Patch-Level)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Signal 1: SAM Segmentation + Per-Object CLIP Scores                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  SAM segments image → CLIP scores each segment against prompt   │    │
│  │  Output: objects with (bbox, label, label_score, color, score)  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Signal 2: CLIP Grad-CAM Attention Maps                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Backprop through CLIP to get attention heatmap                 │    │
│  │  Output: HxW attention map showing where CLIP "looks"           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Signal 3: Object Gap Detection (Expected vs Detected)                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Parse prompt → expected objects                                │    │
│  │  Compare with detected objects                                  │    │
│  │  Output: gaps (missing, wrong_color, wrong_object)              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
TiPAI_TSPO/
├── src/
│   ├── m01_generate_images.py      # SDXL or FLUX image generation
│   ├── m02_extract_signals.py      # SAM + CLIP + Gaps (3 signals)
│   ├── m03_create_risk_maps.py     # RED-only overlay on bad segments
│   ├── m04_create_pairs.py         # Chosen/rejected pairs from signals
│   ├── m08_demo_app.py             # Gradio interface
│   │
│   ├── utils/
│   │   ├── clip_utils.py           # CLIP scoring + Grad-CAM
│   │   ├── sam_utils.py            # SAM segmentation wrapper
│   │   ├── prompt_parser.py        # Extract objects from prompts
│   │   └── hf_utils.py             # HuggingFace dataset utils
│   │
│   └── legacy/                     # Old patch-based approach
│       ├── m02_clip_scoring.py     # OLD: 7x7 patch CLIP scores
│       └── m03_create_heatmaps.py  # OLD: patch-based heatmaps
│
├── outputs/
│   ├── m01_images_sdxl/            # SDXL generated images
│   ├── m01_images_flux/            # FLUX generated images
│   ├── m02_gradcam/                # Grad-CAM numpy files
│   ├── m03_risk_maps/              # Risk map visualizations
│   ├── m04_pairs/                  # Comparison images
│   └── centralized.db              # All data (images, signals, risk_maps, pairs)
│
├── data/
│   ├── prompts.json                # FLUX: 20 base × 4 variations (5 objects)
│   └── prompts_sdxl.json           # SDXL: 20 base × 4 variations (3 objects)
│
├── models/
│   └── sam_vit_b_01ec64.pth        # SAM checkpoint
│
└── requirements.txt
```

---

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE (Object-Level Signals)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: Generate Images (m01)                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
│  │ prompts.json │ --> │ SDXL / FLUX  │ --> │  80 Images   │             │
│  │ (20 base ×   │     │ --model flux │     │  + text      │             │
│  │  4 variations│     │ seed=42      │     │  overlay     │             │
│  └──────────────┘     └──────────────┘     └──────────────┘             │
│                                                                          │
│  STEP 2: Extract 3 Signals (m02)                                        │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
│  │ Each Image   │ --> │ SAM + CLIP   │ --> │ signals      │             │
│  │              │     │ + Grad-CAM   │     │ table in DB  │             │
│  │              │     │ + Gap detect │     │              │             │
│  └──────────────┘     └──────────────┘     └──────────────┘             │
│                                                                          │
│  STEP 3: Create Risk Maps (m03)                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
│  │ Signals from │ --> │ RED overlay  │ --> │ Risk map     │             │
│  │ DB (objects, │     │ ONLY on bad  │     │ PNG files    │             │
│  │ gaps)        │     │ segments     │     │              │             │
│  └──────────────┘     └──────────────┘     └──────────────┘             │
│                                                                          │
│  STEP 4: Create Pairs (m04)                                             │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  v0_original ────┬──► vs v1_attribute → ATTRIBUTE    │               │
│  │  (always chosen) ├──► vs v2_object    → OBJECT       │               │
│  │                  └──► vs v3_spatial   → SPATIAL      │               │
│  │                                                      │               │
│  │  Result: 60 pairs (20 base × 3 failure types)       │               │
│  └──────────────────────────────────────────────────────┘               │
│                                                                          │
│  STEP 5: Demo App (m08)                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
│  │ Gradio UI    │ --> │ Browse pairs │ --> │ View risk    │             │
│  │              │     │ with gaps    │     │ maps         │             │
│  └──────────────┘     └──────────────┘     └──────────────┘             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Model Configurations (m01)

```
┌───────────────────────────────────────────────────────────────────┐
│  SDXL vs FLUX                                                      │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  SDXL + Refiner:                                                  │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  • Base: stabilityai/stable-diffusion-xl-base-1.0           │  │
│  │  • Refiner: stabilityai/stable-diffusion-xl-refiner-1.0     │  │
│  │  • vRAM: ~18GB (both models)                                │  │
│  │  • GPU: A10-24GB sufficient                                 │  │
│  │  • Steps: 40 (80% base, 20% refiner)                        │  │
│  │  • Multi-object: LIMITED (misses objects even with refiner) │  │
│  │  • Prompts: prompts_sdxl.json (3 objects per scene)         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  FLUX.1-dev (RECOMMENDED for multi-object):                       │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  • Model: black-forest-labs/FLUX.1-dev (gated, needs token) │  │
│  │  • vRAM: ~30GB+                                             │  │
│  │  • GPU: A6000-40GB or larger                                │  │
│  │  • Multi-object: Excellent prompt following (4-5 of 5 obj)  │  │
│  │  • Prompts: prompts.json (5 objects per scene)              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Usage:                                                           │
│    python src/m01_generate_images.py --model sdxl --refiner      │
│    python src/m01_generate_images.py --model flux                │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Module 2: Extract Signals (NEW)

```
┌───────────────────────────────────────────────────────────────────┐
│  m02_extract_signals.py (3-Signal Extraction)                      │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Input:  outputs/m01_images_sdxl/*.png or m01_images_flux/*.png   │
│          outputs/centralized.db (reads: images table)             │
│  Output: outputs/centralized.db (table: signals)                  │
│          outputs/m02_gradcam/{image_id}_gradcam.npy               │
│                                                                    │
│  SIGNAL 1: SAM + Per-Object CLIP                                  │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  SAM segments image into objects                            │  │
│  │  For each segment:                                          │  │
│  │    - classify_object() → label, label_score                 │  │
│  │    - detect_color() → color, color_score                    │  │
│  │    - compute_clip_score(segment, phrase) → prompt_score     │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  SIGNAL 2: CLIP Grad-CAM                                          │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  compute_gradcam(image, prompt) → HxW attention map         │  │
│  │  Save as .npy file for later visualization                  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  SIGNAL 3: Object Gap Detection                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  parse_prompt() → expected objects with attributes          │  │
│  │  Compare expected vs detected:                              │  │
│  │    - missing: object not found                              │  │
│  │    - wrong_color: found but wrong color                     │  │
│  │    - wrong_object: different object in same location        │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  DB Schema (signals table):                                       │
│  ┌──────────┬────────┬─────────────┬─────────────┬──────────────┐│
│  │ image_id │ prompt │ objects_json│ gradcam_path│ gaps_json    ││
│  ├──────────┼────────┼─────────────┼─────────────┼──────────────┤│
│  │ attr_01_ │ "red   │ [{segment_id│ m02_gradcam/│ [{object_noun││
│  │ v1_seed42│ car..."│  bbox, label│ attr_01...  │  issue,      ││
│  │          │        │  color,...}]│ .npy        │  expected,...││
│  └──────────┴────────┴─────────────┴─────────────┴──────────────┘│
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Module 3: Create Risk Maps (NEW)

```
┌───────────────────────────────────────────────────────────────────┐
│  m03_create_risk_maps.py (RED-Only Overlay)                        │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Input:  signals table (SAM masks, CLIP scores, gaps from m02)    │
│  Output: outputs/m03_risk_maps/{image_id}_risk_map.png            │
│          outputs/centralized.db (table: risk_maps)                │
│                                                                    │
│  KEY DIFFERENCE from old patch-based approach:                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  OLD: Green/Yellow/Red colormap on all patches              │  │
│  │  NEW: RED overlay ONLY on segments with issues              │  │
│  │       No overlay for correct objects (clean image)          │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Logic:                                                           │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  v0_original: NO overlay (clean reference image)            │  │
│  │                                                             │  │
│  │  v1/v2/v3 Variations: RED overlay on segments with:         │  │
│  │    1. Missing objects (gap detected)                        │  │
│  │    2. Wrong attributes (color mismatch)                     │  │
│  │    3. Low CLIP score (< v0_score * threshold)               │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Visual Output Example:                                           │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  "a red car on beach" (baseline) vs "blue car" (generated)  │  │
│  │                                                             │  │
│  │  +-------------------+     +-------------------+             │  │
│  │  |  Original Image   |     |  Risk Map         |             │  │
│  │  |                   |     |                   |             │  │
│  │  |  [BLUE CAR]       |     |  [RED OVERLAY]    |  ← issue   │  │
│  │  |  [BEACH]          |     |  [BEACH]          |  ← clean   │  │
│  │  +-------------------+     +-------------------+             │  │
│  │                                                             │  │
│  │  Shows: "2 bad segments" (truck + sky marked RED)           │  │
│  │  RED = issue detected | No overlay = correct                │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Module 4: Create Pairs (Training Data)

```
┌───────────────────────────────────────────────────────────────────┐
│  m04_create_pairs.py (Chosen/Rejected Training Pairs)              │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Input:  signals table + risk_maps table                          │
│  Output: outputs/m04_pairs/*.png (2x2 comparison grids)           │
│          outputs/centralized.db (table: pairs)                    │
│                                                                    │
│  Creates 3 pairs per base prompt:                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  v0_original (chosen) vs v1_attribute (rejected)            │  │
│  │    → attribute failure (wrong color/size/material)          │  │
│  │                                                             │  │
│  │  v0_original (chosen) vs v2_object (rejected)               │  │
│  │    → object failure (wrong/swapped object)                  │  │
│  │                                                             │  │
│  │  v0_original (chosen) vs v3_spatial (rejected)              │  │
│  │    → spatial failure (wrong position/count)                 │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Total: 20 prompts × 3 pairs = 60 pairs                          │
│                                                                    │
│  Output Visual (2x2 comparison grid):                             │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  +------------------+------------------+                     │  │
│  │  |  CHOSEN (v0)     |  REJECTED (v1)   |                     │  │
│  │  |  score: 0.85     |  score: 0.62     |                     │  │
│  │  |  gaps: 0         |  gaps: 2         |                     │  │
│  │  +------------------+------------------+                     │  │
│  │  |  Chosen Risk Map |  Rejected Risk   |                     │  │
│  │  |  (no overlay)    |  Map (RED areas) |                     │  │
│  │  +------------------+------------------+                     │  │
│  │                                                             │  │
│  │  Title: "ATTRIBUTE FAILURE | Score Gap: 0.23"               │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  PURPOSE: Training data for Auditor-Scorer model (Stage A)        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Model learns: chosen (v0) should score higher than rejected │  │
│  │  Signals + pairs → supervised training signal                │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

```
┌───────────────────────────────────────────────────────────────────┐
│  centralized.db Tables                                             │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  images (from m01):                                               │
│  ┌────────────┬──────────┬───────────┬───────┬─────────────────┐ │
│  │ image_id   │ base_id  │ variation │ model │ baseline_prompt │ │
│  ├────────────┼──────────┼───────────┼───────┼─────────────────┤ │
│  │ attr_01_v1 │ attr_01  │v1_attrib  │ flux  │ "red car..."    │ │
│  │ _seed42    │          │           │       │                 │ │
│  └────────────┴──────────┴───────────┴───────┴─────────────────┘ │
│  + generation_prompt, prompt_category, seed, image_path          │
│                                                                    │
│  signals (from m02):                                              │
│  ┌────────────┬─────────────┬─────────────┬──────────┬──────────┐│
│  │ image_id   │ objects_json│ gradcam_path│ gaps_json│avg_score ││
│  ├────────────┼─────────────┼─────────────┼──────────┼──────────┤│
│  │ attr_01_v1 │ [{bbox,...}]│ m02_gradcam │ [{issue, │ 0.52     ││
│  │ _seed42    │             │ /...npy     │ ...}]    │          ││
│  └────────────┴─────────────┴─────────────┴──────────┴──────────┘│
│                                                                    │
│  risk_maps (from m03):                                            │
│  ┌────────────┬───────────────┬──────────────┬──────────────────┐│
│  │ image_id   │ risk_map_path │ num_bad_segs │ v0_ref_score     ││
│  ├────────────┼───────────────┼──────────────┼──────────────────┤│
│  │ attr_01_v1 │ m03_risk_maps │ 1            │ 0.89             ││
│  │ _seed42    │ /...png       │              │                  ││
│  └────────────┴───────────────┴──────────────┴──────────────────┘│
│                                                                    │
│  pairs (from m04):                                                │
│  ┌───────────┬─────────────┬─────────────┬─────────────┬────────┐│
│  │ pair_id   │failure_type │ chosen_id   │rejected_id  │score_  ││
│  │           │             │             │             │gap     ││
│  ├───────────┼─────────────┼─────────────┼─────────────┼────────┤│
│  │ pair_attr │ attribute   │ attr_01_v0  │attr_01_v1   │ 0.37   ││
│  │ _01_attr  │             │ _seed42     │_seed42      │        ││
│  └───────────┴─────────────┴─────────────┴─────────────┴────────┘│
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Quick Start Commands

```bash
# Setup
cd /lambda/nfs/DiskUsEast1/TiPAI_TSPO
source venv_tipai/bin/activate
mkdir -p logs

# Run pipeline (SDXL+refiner - A10-24GB, limited multi-object)
python -u src/m01_generate_images.py --model sdxl --refiner 2>&1 | tee logs/m01_sdxl_$(date +%Y%m%d_%H%M%S).log

# Run pipeline (FLUX - A6000-40GB+, RECOMMENDED for multi-object)
python -u src/m01_generate_images.py --model flux 2>&1 | tee logs/m01_flux_$(date +%Y%m%d_%H%M%S).log

# Signal extraction → Risk maps → Pairs (same for both models)
python -u src/m02_extract_signals.py 2>&1 | tee logs/m02_$(date +%Y%m%d_%H%M%S).log
python -u src/m03_create_risk_maps.py 2>&1 | tee logs/m03_$(date +%Y%m%d_%H%M%S).log
python -u src/m04_create_pairs.py 2>&1 | tee logs/m04_$(date +%Y%m%d_%H%M%S).log

# Demo
python src/m08_demo_app.py --port 7860  # Browse pre-generated pairs
python src/m08_demo_app.py --live       # Live CLIP scoring (GPU required)
```

---

## Why Object-Level > Patch-Level

```
┌───────────────────────────────────────────────────────────────────┐
│  PATCH-LEVEL (OLD)              vs    OBJECT-LEVEL (NEW)          │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  7×7 grid patches               vs    SAM-detected objects        │
│  • Fixed grid doesn't           vs    • Segments follow object    │
│    follow object boundaries           boundaries naturally        │
│                                                                    │
│  Green/Yellow/Red colormap      vs    RED-only on issues          │
│  • Distracting, hard to         vs    • Clean: issues stand out   │
│    see actual issues                                              │
│                                                                    │
│  No semantic understanding      vs    Object classification       │
│  • Just image-text similarity   vs    • Knows what object it is   │
│                                       • Knows expected vs found   │
│                                                                    │
│  No gap detection               vs    Explicit gap detection      │
│  • Can't say "car is blue       vs    • "Expected: red car        │
│    instead of red"                      Found: blue car"          │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Success Criteria

```
┌───────────────────────────────────────────────────────────────────┐
│  PASS if:                                                          │
│    - 80 images generated (20 base × 4 variations)                 │
│    - Signals extracted for all images (objects, gradcam, gaps)    │
│    - Risk maps show RED overlay on wrong segments only            │
│    - v0_original has no overlay (clean reference)                 │
│    - Gap detection correctly identifies wrong color/object        │
│    - 60 pairs created with failure_type labels                    │
│    - Demo shows pairs with risk maps and gap details              │
│                                                                    │
│  KEY VALIDATIONS:                                                  │
│    - "blue car" variation → gap: {issue: wrong_color, expected:   │
│      red, found: blue}                                            │
│    - "bicycle" variation → gap: {issue: wrong_object, expected:   │
│      car, found: bicycle}                                         │
│    - Risk map highlights the correct segment (car, not beach)     │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Next Steps (Stage A Training)

```
┌───────────────────────────────────────────────────────────────────┐
│  After POC validation:                                             │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  m05_train_stage_a.py (Auditor-Scorer)                            │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Input: 3 signals from m02                                  │  │
│  │  Output: Global faithfulness score S (learned)              │  │
│  │                                                             │  │
│  │  Architecture: MLP that takes signal features → score       │  │
│  │  Training: Use v0 vs v1/v2/v3 pairs as supervision          │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  m06_train_stage_b.py (Inpainter + TSPO)                          │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Use risk maps to guide inpainting of bad regions           │  │
│  │  TSPO loss to improve generator                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  m07_calibrate.py (Stage C)                                       │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Calibrate Auditor-Scorer for deployment                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```
