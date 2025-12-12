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
+-------------------------------------------------------------------------+
|                         POC PIPELINE                                     |
+-------------------------------------------------------------------------+

STEP 1: Generate Images
+-------------------+     +-------------------+     +-------------------+
|   prompts.json    | --> |  Stable Diffusion | --> |   80 Images       |
|   (20 prompts)    |     |  (4 per prompt)   |     |   (PNG files)     |
+-------------------+     +-------------------+     +-------------------+

STEP 2: CLIP Scoring
+-------------------+     +-------------------+     +-------------------+
|   Each Image      | --> |   CLIP ViT-L/14   | --> |   Global Score    |
|                   |     |                   |     |   (0-1)           |
+-------------------+     +-------------------+     +-------------------+
        |
        v
+-------------------+     +-------------------+     +-------------------+
|   Split into      | --> |   CLIP Score      | --> |   Patch Scores    |
|   7x7 = 49 patches|     |   Each Patch      |     |   (49 values)     |
+-------------------+     +-------------------+     +-------------------+

STEP 3: Heatmap
+-------------------+     +-------------------+     +-------------------+
|   Patch Scores    | --> |   Interpolate     | --> |   Heatmap Image   |
|   (49 values)     |     |   to Image Size   |     |   (color overlay) |
+-------------------+     +-------------------+     +-------------------+

STEP 4: Create Pairs
+-------------------+     +-------------------+     +-------------------+
|   4 Images per    | --> |   Sort by Global  | --> |   Best = Chosen   |
|   Prompt          |     |   CLIP Score      |     |   Worst = Rejected|
+-------------------+     +-------------------+     +-------------------+

STEP 5: Demo
+-------------------+     +-------------------+     +-------------------+
|   Gradio App      | --> |   User enters     | --> |   Shows heatmap   |
|                   |     |   prompt          |     |   + pairs         |
+-------------------+     +-------------------+     +-------------------+
```

---

## Module 1: Generate Images

```
+-------------------------------------------------------------------+
|  m01_generate_images.py                                            |
+-------------------------------------------------------------------+

Input:  data/prompts.json
Output: outputs/m01_images/{prompt_id}_{seed}.png
        outputs/centralized.db (table: images)

Flow:
        +-------------------+
        |   Load Prompts    |
        |   (20 prompts)    |
        +---------+---------+
                  |
                  v
        +-------------------+
        |   Load SD Model   |
        |   (SDXL or SD3.5) |
        +---------+---------+
                  |
                  v
        +-------------------+
        |   For each prompt:|
        |   Generate 4 imgs |
        |   (seeds 0,1,2,3) |
        +---------+---------+
                  |
                  v
        +-------------------+
        |   Save to disk    |
        |   + log to DB     |
        +---------+---------+
                  |
                  v
        +-------------------+
        |   80 images total |
        +-------------------+

DB Schema (images table):
+----------+--------+------+------------------+
| image_id | prompt | seed | path             |
+----------+--------+------+------------------+
| img_001  | "red.."| 0    | m01_images/...   |
+----------+--------+------+------------------+
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

## Module 4: Create Pairs

```
+-------------------------------------------------------------------+
|  m04_create_pairs.py                                               |
+-------------------------------------------------------------------+

Input:  outputs/centralized.db (reads: scores)
Output: outputs/centralized.db (table: pairs)
        outputs/m04_pairs/{prompt_id}_comparison.png

Flow:
        +-------------------+
        |   For each prompt |
        |   (20 prompts)    |
        +---------+---------+
                  |
                  v
        +-------------------+
        |   Get 4 images    |
        |   + global scores |
        +---------+---------+
                  |
                  v
        +-------------------+
        |   Sort by score   |
        |   descending      |
        +---------+---------+
                  |
                  v
        +-------------------+
        |   Chosen = Best   |
        |   Rejected = Worst|
        +---------+---------+
                  |
                  v
        +-------------------+
        | Create comparison |
        | visualization     |
        +-------------------+

Comparison Visualization:
+---------------------------------------------------------------+
|  Prompt: "a red car on the beach"                              |
+---------------------------------------------------------------+
|                           |                                    |
|  CHOSEN (score: 0.89)     |  REJECTED (score: 0.52)           |
|  +-------------------+    |  +-------------------+             |
|  |                   |    |  |                   |             |
|  |   [RED CAR]       |    |  |   [BLUE CAR]      |             |
|  |   [BEACH]         |    |  |   [BEACH]         |             |
|  |                   |    |  |                   |             |
|  +-------------------+    |  +-------------------+             |
|                           |                                    |
|  Heatmap: mostly green    |  Heatmap: car region red          |
+---------------------------------------------------------------+

DB Schema (pairs table):
+---------+--------+------------+---------------+--------+--------+
| pair_id | prompt | chosen_img | rejected_img  | c_score| r_score|
+---------+--------+------------+---------------+--------+--------+
| pair_01 | "red.."| img_003    | img_001       | 0.89   | 0.52   |
+---------+--------+------------+---------------+--------+--------+
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

## 20 Test Prompts (data/prompts.json)

```
+-------------------------------------------------------------------+
|  Categories (from DETONATE plan):                                  |
+-------------------------------------------------------------------+

Object Presence (4 prompts):
1. "a dog and a cat sitting together"
2. "a bicycle next to a tree"
3. "a cup of coffee on a wooden table"
4. "a bird flying over a lake"

Attribute Binding (4 prompts):
5. "a red car on the beach"
6. "a blue house with a white door"
7. "a yellow banana and a green apple"
8. "a black cat with orange eyes"

Counting (4 prompts):
9. "three red apples on a plate"
10. "two dogs playing in a park"
11. "five balloons floating in the sky"
12. "four chairs around a table"

Spatial Relations (4 prompts):
13. "a cat sitting on top of a box"
14. "a lamp next to a bed"
15. "a ball under a chair"
16. "a book in front of a computer"

Compositional (4 prompts):
17. "a chef cooking in a modern kitchen"
18. "a child reading a book under a tree"
19. "an astronaut riding a horse on mars"
20. "a robot serving coffee in a cafe"
```

---

## Expected Output Examples

```
+-------------------------------------------------------------------+
|  Example 1: Attribute Binding Failure                              |
+-------------------------------------------------------------------+

Prompt: "a red car on the beach"

Generated Images:
+----------+----------+----------+----------+
| img_1    | img_2    | img_3    | img_4    |
| red car  | blue car | red car  | gray car |
| beach    | beach    | parking  | beach    |
| score:   | score:   | score:   | score:   |
| 0.89     | 0.52     | 0.61     | 0.48     |
+----------+----------+----------+----------+
   BEST                              WORST

Heatmap for img_4 (worst):
+-------------------+
| . . . . . . . |   Legend:
| . . . . . . . |   . = high score (green)
| . . X X . . . |   X = low score (red)
| . . X X . . . |
| . . X X . . . |   The car region (X) has
| . . . . . . . |   low score because
| . . . . . . . |   gray != red
+-------------------+


+-------------------------------------------------------------------+
|  Example 2: Counting Failure                                       |
+-------------------------------------------------------------------+

Prompt: "three red apples on a plate"

Generated Images:
+----------+----------+----------+----------+
| img_1    | img_2    | img_3    | img_4    |
| 3 apples | 2 apples | 3 apples | 5 apples |
| red      | red      | green    | red      |
| score:   | score:   | score:   | score:   |
| 0.91     | 0.65     | 0.58     | 0.62     |
+----------+----------+----------+----------+
   BEST                   WORST

Heatmap for img_3 (worst):
+-------------------+
| . . . . . . . |   The apple regions show
| . . X X X . . |   low scores because
| . . X X X . . |   green != red
| . . X X X . . |   (color attribute fail)
| . . . . . . . |
| . . . . . . . |
+-------------------+
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
+-------------------------------------------------------------------+
|  POC PASS if:                                                      |
|    - 80 images generated without errors                            |
|    - Heatmaps clearly show low-score regions                       |
|    - Best/worst pairs are visually distinguishable                 |
|    - Gradio demo runs and responds in < 60s                        |
|    - Professor says "I see the concept"                            |
|                                                                    |
|  POC FAIL if:                                                      |
|    - CLIP scores don't correlate with visual quality               |
|    - Heatmaps look random (not localized)                          |
|    - All images score similarly (no discrimination)                |
+-------------------------------------------------------------------+
```

---

## Next Steps After POC

```
POC (12 hrs) --> Mini Dataset (2 days) --> Full Stage A (1 week)

+-------------------+     +-------------------+     +-------------------+
|  POC: 80 images   | --> |  Mini: 1K images  | --> |  Full: 20K images |
|  No training      |     |  Simple training  |     |  Full Stage A     |
|  CLIP only        |     |  MLP head on CLIP |     |  All 4 losses     |
+-------------------+     +-------------------+     +-------------------+
```
