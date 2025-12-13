# POC: Patch-Level Faithfulness Heatmap Demo

## Goal (12 Hours)

```
Show professor: "We can automatically find WHERE an image fails the prompt"

Input:  Prompt = "a red car on the beach"
        Image  = [blue car on beach]

Output:
+-------------------+     +-------------------+
|                   |     |  Faithfulness     |
|   [BLUE CAR]      | --> |  Heatmap          |
|   [BEACH]         |     |  [RED=BAD]        |
+-------------------+     +-------------------+

Professor sees: "The blue car region is highlighted as unfaithful!"
```

---

## Timeline (12 Hours)

| Hour | Task | Output |
|------|------|--------|
| 0-1 | Setup environment | `venv_tipai/` |
| 1-3 | Generate 80 images (20 prompts x 4) | `outputs/images/` |
| 3-5 | CLIP scoring (global + patch) | `outputs/scores/` |
| 5-7 | Heatmap generation | `outputs/heatmaps/` |
| 7-9 | Gradio demo app | `src/demo.py` |
| 9-11 | Create pairs + visualizations | `outputs/pairs/` |
| 11-12 | Polish + prepare presentation | slides |

---

## File Structure

```
TiPAI_TSPO/
├── src/
│   ├── m01_generate_images.py      # SD image generation
│   ├── m02_clip_scoring.py         # Global + patch CLIP scores
│   ├── m03_create_heatmaps.py      # Visualize patch scores
│   ├── m04_create_pairs.py         # Best vs worst pairs
│   └── m05_demo_app.py             # Gradio interface
│
├── outputs/
│   ├── m01_images/                 # Generated images
│   ├── m02_scores/                 # CLIP scores (DB)
│   ├── m03_heatmaps/               # Heatmap visualizations
│   ├── m04_pairs/                  # Chosen/rejected pairs
│   └── centralized.db              # All data
│
├── data/
│   └── prompts.json                # 20 test prompts
│
└── requirements.txt
```

---

## Libraries (requirements.txt)

```
torch>=2.0.0
transformers>=4.35.0
diffusers>=0.24.0
open-clip-torch>=2.23.0
gradio>=4.0.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    POC PIPELINE (Systematic Variations)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: Generate Images (m01)                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
│  │ prompts.json │ --> │    SDXL      │ --> │  80 Images   │             │
│  │ (20 base ×   │     │  seed=42     │     │  (PNG files) │             │
│  │  4 variations│     │  (fixed)     │     │              │             │
│  └──────────────┘     └──────────────┘     └──────────────┘             │
│                                                                          │
│  STEP 2: CLIP Scoring (m02)                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
│  │ Each Image   │ --> │ CLIP ViT-L/14│ --> │ Global +     │             │
│  │              │     │              │     │ 49 Patch     │             │
│  │              │     │              │     │ Scores       │             │
│  └──────────────┘     └──────────────┘     └──────────────┘             │
│                                                                          │
│  STEP 3: Heatmap (m03)                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
│  │ Patch Scores │ --> │ Interpolate  │ --> │ Red/Green    │             │
│  │ (7×7 grid)   │     │ + Colormap   │     │ Overlay      │             │
│  └──────────────┘     └──────────────┘     └──────────────┘             │
│                                                                          │
│  STEP 4: Create Pairs (m04) - VARIATION-BASED                           │
│  ┌──────────────────────────────────────────────────────┐               │
│  │  v0_original ────┬──► vs v1_attribute → ATTRIBUTE    │               │
│  │  (always chosen) ├──► vs v2_object    → OBJECT       │               │
│  │                  └──► vs v3_spatial   → SPATIAL      │               │
│  │                                                      │               │
│  │  Result: 60 pairs (20 base × 3 failure types)       │               │
│  └──────────────────────────────────────────────────────┘               │
│                                                                          │
│  STEP 5: Demo App (m05)                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
│  │ Gradio UI    │ --> │ Browse pairs │ --> │ View heatmaps│             │
│  └──────────────┘     └──────────────┘     └──────────────┘             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module 1: Generate Images

```
┌───────────────────────────────────────────────────────────────────┐
│  m01_generate_images.py (Systematic Variations)                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Input:  data/prompts.json (80 prompts = 20 base × 4 variations)  │
│  Output: outputs/m01_images/{prompt_id}_seed42.png                │
│          outputs/centralized.db (table: images)                   │
│                                                                    │
│  THE 4 VARIATION TYPES:                                           │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  BASE: "a red car on the beach"                             │  │
│  │                                                             │  │
│  │  v0_original:  "a red car on the beach"      (CHOSEN)      │  │
│  │  v1_attribute: "a BLUE car on the beach"     (rejected)    │  │
│  │  v2_object:    "a red BICYCLE on the beach"  (rejected)    │  │
│  │  v3_spatial:   "a red car IN THE OCEAN"      (rejected)    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Flow:                                                            │
│          ┌───────────────┐                                        │
│          │ Load 80       │                                        │
│          │ prompts       │                                        │
│          └───────┬───────┘                                        │
│                  │                                                │
│                  v                                                │
│          ┌───────────────┐                                        │
│          │ Load SDXL     │                                        │
│          │ (CUDA only)   │                                        │
│          └───────┬───────┘                                        │
│                  │                                                │
│                  v                                                │
│          ┌───────────────┐                                        │
│          │ For each      │                                        │
│          │ prompt:       │                                        │
│          │ Generate with │                                        │
│          │ seed=42       │                                        │
│          └───────┬───────┘                                        │
│                  │                                                │
│                  v                                                │
│          ┌───────────────┐                                        │
│          │ 80 images     │                                        │
│          │ total         │                                        │
│          └───────────────┘                                        │
│                                                                    │
│  DB Schema (images table):                                        │
│  ┌──────────┬──────────┬───────────┬──────────┬─────────────────┐│
│  │ image_id │ base_id  │ variation │ seed     │ path            ││
│  ├──────────┼──────────┼───────────┼──────────┼─────────────────┤│
│  │ attr_01_ │ attr_01  │v0_original│ 42       │ m01_images/...  ││
│  │ v0_seed42│          │           │          │                 ││
│  └──────────┴──────────┴───────────┴──────────┴─────────────────┘│
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Module 2: CLIP Scoring

```
+-------------------------------------------------------------------+
|  m02_clip_scoring.py                                               |
+-------------------------------------------------------------------+

Input:  outputs/m01_images/*.png
        outputs/centralized.db (reads: images)
Output: outputs/centralized.db (table: scores)

Flow:
        +-------------------+
        |   Load CLIP       |
        |   ViT-L/14        |
        +---------+---------+
                  |
                  v
        +-------------------+
        |   For each image: |
        +---------+---------+
                  |
        +---------+---------+---------+
        |                             |
        v                             v
+---------------+             +---------------+
| Global Score  |             | Patch Scores  |
|               |             |               |
| text_emb =    |             | Split image   |
| CLIP(prompt)  |             | into 7x7 grid |
|               |             |               |
| img_emb =     |             | For each cell:|
| CLIP(image)   |             | patch_emb =   |
|               |             | CLIP(patch)   |
| score =       |             |               |
| cos_sim(t,i)  |             | patch_score = |
+-------+-------+             | cos_sim(t,p)  |
        |                     +-------+-------+
        |                             |
        +-------------+---------------+
                      |
                      v
              +---------------+
              | Save to DB    |
              | (scores table)|
              +---------------+

DB Schema (scores table):
+----------+--------+--------------+----------------------+
| image_id | prompt | global_score | patch_scores (JSON)  |
+----------+--------+--------------+----------------------+
| img_001  | "red.."| 0.72         | [0.8, 0.3, 0.9, ...] |
+----------+--------+--------------+----------------------+

Patch Grid:
+-----+-----+-----+-----+-----+-----+-----+
|  0  |  1  |  2  |  3  |  4  |  5  |  6  |
+-----+-----+-----+-----+-----+-----+-----+
|  7  |  8  |  9  | 10  | 11  | 12  | 13  |
+-----+-----+-----+-----+-----+-----+-----+
| 14  | 15  | 16  | 17  | 18  | 19  | 20  |
+-----+-----+-----+-----+-----+-----+-----+
| 21  | 22  | 23  | 24  | 25  | 26  | 27  |
+-----+-----+-----+-----+-----+-----+-----+
| 28  | 29  | 30  | 31  | 32  | 33  | 34  |
+-----+-----+-----+-----+-----+-----+-----+
| 35  | 36  | 37  | 38  | 39  | 40  | 41  |
+-----+-----+-----+-----+-----+-----+-----+
| 42  | 43  | 44  | 45  | 46  | 47  | 48  |
+-----+-----+-----+-----+-----+-----+-----+

Each cell gets a CLIP score (0-1)
```

---

## Module 3: Create Heatmaps

```
+-------------------------------------------------------------------+
|  m03_create_heatmaps.py                                            |
+-------------------------------------------------------------------+

Input:  outputs/centralized.db (reads: scores)
Output: outputs/m03_heatmaps/{image_id}_heatmap.png

Flow:
        +-------------------+
        |   Load scores     |
        |   from DB         |
        +---------+---------+
                  |
                  v
        +-------------------+
        |   For each image: |
        +---------+---------+
                  |
                  v
        +-------------------+
        | Reshape 49 scores |
        |   to 7x7 grid     |
        +---------+---------+
                  |
                  v
        +-------------------+
        | Interpolate to    |
        | image size (512x) |
        +---------+---------+
                  |
                  v
        +-------------------+
        | Apply colormap    |
        | HIGH=green=good   |
        | LOW=red=bad       |
        +---------+---------+
                  |
                  v
        +-------------------+
        | Overlay on image  |
        | (alpha blend)     |
        +---------+---------+
                  |
                  v
        +-------------------+
        | Save heatmap PNG  |
        +-------------------+

Visual Output:
+-------------------+     +-------------------+     +-------------------+
|  Original Image   |  +  |  Score Heatmap    |  =  |  Overlay Result   |
|                   |     |  (interpolated)   |     |                   |
|  [BLUE CAR]       |     |  [RED REGION]     |     |  [BLUE CAR]       |
|  [BEACH]          |     |  [GREEN REGION]   |     |  highlighted red  |
+-------------------+     +-------------------+     +-------------------+

Color Scale:
    0.0 -------- 0.5 -------- 1.0
    RED         YELLOW       GREEN
    (bad)       (medium)     (good)
```

---

## Module 4: Create Pairs (Variation-Based)

```
┌───────────────────────────────────────────────────────────────────┐
│  m04_create_pairs.py (VARIATION-BASED, not score-based)           │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Input:  outputs/centralized.db (reads: scores, images)           │
│  Output: outputs/centralized.db (table: pairs)                    │
│          outputs/m04_pairs/{pair_id}_comparison.png               │
│                                                                    │
│  PAIRING LOGIC (3 pairs per base prompt):                         │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                                                             │  │
│  │  v0_original ────┬──► vs v1_attribute → "attribute" failure│  │
│  │  (always chosen) │                                          │  │
│  │                  ├──► vs v2_object    → "object" failure   │  │
│  │                  │                                          │  │
│  │                  └──► vs v3_spatial   → "spatial" failure  │  │
│  │                                                             │  │
│  │  Result: 20 base × 3 pair types = 60 pairs                 │  │
│  │  Each pair has KNOWN failure_type (no VQAScore needed!)    │  │
│  │                                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Comparison Visualization:                                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  ATTRIBUTE FAILURE (color/size/material)                    │  │
│  │  Score Gap: 0.37                                            │  │
│  ├─────────────────────────────────────────────────────────────┤  │
│  │                           │                                 │  │
│  │  CHOSEN (v0)              │  REJECTED (v1)                  │  │
│  │  "red car on beach"       │  "BLUE car on beach"            │  │
│  │  score: 0.89              │  score: 0.52                    │  │
│  │  ┌─────────────────┐      │  ┌─────────────────┐            │  │
│  │  │   [RED CAR]     │      │  │   [BLUE CAR]    │            │  │
│  │  │   [BEACH]       │      │  │   [BEACH]       │            │  │
│  │  └─────────────────┘      │  └─────────────────┘            │  │
│  │                           │                                 │  │
│  │  Heatmap: mostly green    │  Heatmap: car region RED       │  │
│  │                           │                                 │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  DB Schema (pairs table):                                         │
│  ┌───────────┬─────────┬─────────────┬─────────────┬────────────┐│
│  │ pair_id   │ base_id │failure_type │ chosen_id   │rejected_id ││
│  ├───────────┼─────────┼─────────────┼─────────────┼────────────┤│
│  │ pair_attr │ attr_01 │ attribute   │ attr_01_v0  │attr_01_v1  ││
│  │ _01_attr  │         │             │ _seed42     │_seed42     ││
│  └───────────┴─────────┴─────────────┴─────────────┴────────────┘│
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Module 5: Demo App

```
+-------------------------------------------------------------------+
|  m05_demo_app.py (Gradio)                                          |
+-------------------------------------------------------------------+

Interface:
+---------------------------------------------------------------+
|                     TiPAI POC Demo                             |
+---------------------------------------------------------------+
|                                                                |
|  Enter Prompt: [________________________________] [Generate]   |
|                                                                |
+---------------------------------------------------------------+
|                                                                |
|  Generated Images:                                             |
|  +-------+ +-------+ +-------+ +-------+                       |
|  | img1  | | img2  | | img3  | | img4  |                       |
|  | 0.82  | | 0.71  | | 0.89  | | 0.55  |                       |
|  +-------+ +-------+ +-------+ +-------+                       |
|                                                                |
+---------------------------------------------------------------+
|                                                                |
|  Best vs Worst:                                                |
|  +-------------------+     +-------------------+               |
|  | CHOSEN (0.89)     |     | REJECTED (0.55)   |               |
|  | [image + heatmap] |     | [image + heatmap] |               |
|  +-------------------+     +-------------------+               |
|                                                                |
+---------------------------------------------------------------+
|                                                                |
|  Patch Analysis (Rejected Image):                              |
|  +-------------------------------------------+                 |
|  |  Heatmap shows car region has LOW score   |                 |
|  |  because prompt said RED but image is BLUE|                 |
|  +-------------------------------------------+                 |
|                                                                |
+---------------------------------------------------------------+

Flow:
        +-------------------+
        | User enters prompt|
        +---------+---------+
                  |
                  v
        +-------------------+
        | Generate 4 images |
        | with SD           |
        +---------+---------+
                  |
                  v
        +-------------------+
        | CLIP score all    |
        | (global + patch)  |
        +---------+---------+
                  |
                  v
        +-------------------+
        | Create heatmaps   |
        +---------+---------+
                  |
                  v
        +-------------------+
        | Display results   |
        | in Gradio UI      |
        +-------------------+
```

---

## Prompts Structure (data/prompts.json)

```
┌───────────────────────────────────────────────────────────────────┐
│  80 PROMPTS = 20 base × 4 variations (Systematic Approach)        │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  EXAMPLE BASE PROMPT (attr_01):                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  v0_original:  "a shiny red sports car on a sandy beach"   │  │
│  │  v1_attribute: "a shiny BLUE sports car on a sandy beach"  │  │
│  │  v2_object:    "a shiny red BICYCLE on a sandy beach"      │  │
│  │  v3_spatial:   "a shiny red sports car IN THE OCEAN"       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  5 CATEGORIES (4 base prompts each = 20 base):                    │
│                                                                    │
│  Object Presence (4 base × 4 var = 16 prompts):                   │
│  - obj_01: "a golden retriever and a tabby cat..."               │
│  - obj_02: "a vintage bicycle leaning against a tree..."         │
│  - obj_03: "a steaming cup of coffee on a table..."              │
│  - obj_04: "a white swan gliding across a lake..."               │
│                                                                    │
│  Attribute Binding (4 base × 4 var = 16 prompts):                 │
│  - attr_01: "a shiny red sports car on a beach..."               │
│  - attr_02: "a cozy blue cottage with a white door..."           │
│  - attr_03: "a ripe yellow banana and a green apple..."          │
│  - attr_04: "a sleek black cat with orange eyes..."              │
│                                                                    │
│  Counting (4 base × 4 var = 16 prompts):                          │
│  - count_01: "three red apples on a plate..."                    │
│  - count_02: "two playful dogs in a park..."                     │
│  - count_03: "five colorful balloons floating..."                │
│  - count_04: "four wooden chairs around a table..."              │
│                                                                    │
│  Spatial Relations (4 base × 4 var = 16 prompts):                 │
│  - spatial_01: "a fluffy cat sitting on a cardboard box..."      │
│  - spatial_02: "an elegant lamp next to a bed..."                │
│  - spatial_03: "a red ball under a wooden chair..."              │
│  - spatial_04: "an open book in front of a laptop..."            │
│                                                                    │
│  Compositional (4 base × 4 var = 16 prompts):                     │
│  - comp_01: "a professional chef cooking in a kitchen..."        │
│  - comp_02: "a young child reading under a tree..."              │
│  - comp_03: "an astronaut riding a horse on mars..."             │
│  - comp_04: "a friendly robot serving coffee..."                 │
│                                                                    │
│  TOTAL: 20 base × 4 variations = 80 prompts → 80 images          │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Expected Output Examples

```
┌───────────────────────────────────────────────────────────────────┐
│  Example: attr_01 (Attribute Binding)                              │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  4 VARIATIONS (all generated with seed=42):                       │
│                                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐│
│  │ v0_original │  │v1_attribute │  │  v2_object  │  │v3_spatial ││
│  │             │  │             │  │             │  │           ││
│  │ [RED CAR]   │  │ [BLUE CAR]  │  │ [BICYCLE]   │  │ [CAR IN   ││
│  │ [BEACH]     │  │ [BEACH]     │  │ [BEACH]     │  │  OCEAN]   ││
│  │             │  │             │  │             │  │           ││
│  │ score: 0.89 │  │ score: 0.52 │  │ score: 0.61 │  │score: 0.48││
│  │ (CHOSEN)    │  │ (rejected)  │  │ (rejected)  │  │(rejected) ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘│
│                                                                    │
│  3 PAIRS CREATED:                                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Pair 1: v0 vs v1 → failure_type = "attribute" (wrong color)│  │
│  │ Pair 2: v0 vs v2 → failure_type = "object" (wrong object)  │  │
│  │ Pair 3: v0 vs v3 → failure_type = "spatial" (wrong place)  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  HEATMAP for v1_attribute (rejected):                             │
│  ┌─────────────────────────────────────┐                          │
│  │ . . . . . . . │  Legend:            │                          │
│  │ . . . . . . . │  . = green (good)   │                          │
│  │ . . X X . . . │  X = red (bad)      │                          │
│  │ . . X X . . . │                     │                          │
│  │ . . X X . . . │  Car region is RED  │                          │
│  │ . . . . . . . │  because BLUE ≠ RED │                          │
│  │ . . . . . . . │  (attribute fail)   │                          │
│  └─────────────────────────────────────┘                          │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Professor Presentation Points

```
+-------------------------------------------------------------------+
|  Key Messages for Professor:                                       |
+-------------------------------------------------------------------+

1. "We can automatically detect WHERE an image fails"
   - Show heatmaps with bad regions highlighted
   - No human labeling needed (CLIP does it)

2. "This enables patch-level preference learning"
   - Existing methods (DPO) only have image-level scores
   - We know WHICH region caused the rejection

3. "This is the foundation for TSPO"
   - Stage A (this POC) --> Stage B (inpainting) --> Stage C (deployment)
   - POC proves Stage A concept works

4. "Immigration-safe approach"
   - No NSFW content
   - Pure faithfulness (text-image alignment)
   - Uses established metrics (CLIP, VQAScore)

+-------------------------------------------------------------------+
|  Demo Script (3 minutes):                                          |
+-------------------------------------------------------------------+

1. [30s] Enter prompt: "a red car on the beach"
2. [30s] Show 4 generated images with scores
3. [30s] Highlight best (red car) vs worst (wrong color)
4. [60s] Show heatmaps - "see how car region is red in the bad one?"
5. [30s] "This patch-level signal is what TSPO will use to fix images"
```

---

## Quick Start Commands

```bash
# Setup
cd /Users/kapilwanaskar/Downloads/research_projects/TiPAI_TSPO
python -m venv venv_tipai
source venv_tipai/bin/activate
pip install -r requirements.txt

# Run pipeline
python src/m01_generate_images.py    # ~2 hours (GPU)
python src/m02_clip_scoring.py       # ~30 min
python src/m03_create_heatmaps.py    # ~10 min
python src/m04_create_pairs.py       # ~5 min
python src/m05_demo_app.py           # Launch Gradio

# Or: Run demo directly (generates on-the-fly)
python src/m05_demo_app.py --live
```

---

## Success Criteria

```
┌───────────────────────────────────────────────────────────────────┐
│  POC PASS if:                                                      │
│    - 80 images generated (20 base × 4 variations)                 │
│    - 60 pairs created (20 base × 3 failure types)                 │
│    - v0_original consistently scores HIGHER than v1/v2/v3         │
│    - Heatmaps show LOW scores where variation differs from v0     │
│    - Gradio demo runs and shows pairs with heatmaps               │
│    - Professor says "I see the concept"                           │
│                                                                    │
│  POC FAIL if:                                                      │
│    - v0 doesn't score higher than variations (no discrimination)  │
│    - Heatmaps don't localize the changed region                   │
│    - CLIP scores are random (no correlation with prompt fidelity) │
│                                                                    │
│  KEY ADVANTAGE of Systematic Variations:                          │
│    - We KNOW which region should be red (the changed part)        │
│    - If heatmap highlights wrong region → CLIP issue, not data    │
│    - No VQAScore needed → reproducible, deterministic             │
└───────────────────────────────────────────────────────────────────┘
```

---

## Next Steps After POC

```
┌───────────────────────────────────────────────────────────────────┐
│  SCALING PLAN (Systematic Variations)                              │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  POC → Paper Dataset → Full Training                              │
│                                                                    │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │
│  │ POC         │   │ Paper       │   │ Training    │              │
│  │             │   │             │   │             │              │
│  │ 20 base     │   │ 5,000 base  │   │ TSPO on     │              │
│  │ × 4 var     │→→→│ × 4 var     │→→→│ 20K pairs   │              │
│  │ = 80 images │   │ = 20K images│   │             │              │
│  │ = 60 pairs  │   │ = 15K pairs │   │             │              │
│  │             │   │             │   │             │              │
│  │ CLIP only   │   │ CLIP only   │   │ Auditor-    │              │
│  │ No training │   │ No training │   │ Scorer MLP  │              │
│  └─────────────┘   └─────────────┘   └─────────────┘              │
│                                                                    │
│  PROMPT SOURCE for Paper:                                         │
│  - T2I-CompBench (6K)                                             │
│  - TIFA-v1.0 (4K)                                                 │
│  - DrawBench (2K)                                                 │
│  - GenAI-Bench (3K)                                               │
│  → 5,000 selected base prompts × 4 variations = 20,000 images    │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```
