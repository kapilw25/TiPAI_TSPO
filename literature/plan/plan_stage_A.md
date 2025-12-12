# Stage A: Auditor-Scorer Training Plan

## Deviation from Proposal

| Proposal | Our Version | Reason |
|----------|-------------|--------|
| L_policy (NSFW, weapon, symbol, logo, age) | L_faithfulness (prompt-failing regions) | Immigration-safe, no NSFW data needed |
| Policy thresholds tau_P | Faithfulness thresholds tau_F only | Same reason |
| Per-class risk heatmaps (NSFW, weapon, etc.) | Single faithfulness heatmap | No policy classes in our dataset |

**Why this change?**
- Original proposal requires NSFW/weapon/symbol detection datasets
- Such datasets cannot be pushed to HuggingFace/GitHub
- Immigration risk for non-immigrant researchers in USA
- Our "policy" IS faithfulness: detecting regions that violate the PROMPT

---

## What is the Auditor-Scorer?

**Layman Analogy**: A teacher grading student essays.

```
Student Essay (Image) + Assignment Prompt (Text)
                    |
                    v
              Teacher (Auditor)
                    |
        +-----------+-----------+
        |                       |
        v                       v
   Overall Grade         Margin Comments
   (Global Score)        (Patch Scores)

   "B- overall"          "This paragraph is weak"
                         "Good intro"
                         "Wrong conclusion"
```

The Auditor-Scorer does the same for images:
- **Global Score**: "This image gets 0.7/1.0 for matching the prompt"
- **Patch Scores**: "The car region is wrong (0.2), the background is correct (0.9)"

---

## System Architecture

```
+---------------------------------------------------------------------+
|                         AUDITOR-SCORER                               |
+---------------------------------------------------------------------+
                                |
        +-----------------------+-----------------------+
        |                       |                       |
        v                       v                       v
+---------------+       +---------------+       +---------------+
|  TEXT ENCODER |       | IMAGE ENCODER |       | PATCH ENCODER |
|  (CLIP ViT-L) |       |  (CLIP ViT-L) |       |  (CLIP ViT-L) |
|    FROZEN     |       |    FROZEN     |       |    FROZEN     |
+-------+-------+       +-------+-------+       +-------+-------+
        |                       |                       |
        v                       v                       v
  text_emb (768-d)        img_emb (768-d)       patch_embs (N x 768-d)
        |                       |                       |
        +----------+------------+                       |
                   |                                    |
                   v                                    v
           +---------------+                    +---------------+
           |  GLOBAL HEAD  |                    |  PATCH HEAD   |
           |     (MLP)     |                    |     (MLP)     |
           |   TRAINABLE   |                    |   TRAINABLE   |
           +-------+-------+                    +-------+-------+
                   |                                    |
                   v                                    v
            Global Score (0-1)                  Patch Scores [s1, s2, ..., sN]
                   |                                    |
                   +----------------+-------------------+
                                    |
                                    v
                            +---------------+
                            | SALIENCY HEAD |
                            |  (Decoder)    |
                            |   TRAINABLE   |
                            +-------+-------+
                                    |
                                    v
                            Risk Heatmap (H x W)
```

**What's FROZEN vs TRAINABLE?**

```
FROZEN = Pre-trained weights, don't change
         (Like using a calculator - you trust it works)

TRAINABLE = Weights we update during training
            (Like teaching a new skill)

Why freeze CLIP?
- CLIP already understands images and text (trained on 400M pairs)
- We just add small "head" networks on top to make decisions
- Faster training, less compute, less data needed
```

---

## The Three Training Heads

### Head 1: Global Scorer

```
Purpose: Rate the WHOLE image (0 to 1)

Example:
+-----------------------------------------------------------+
|  Prompt: "a red car on the beach"                         |
|                                                           |
|  Image A: Red car, beach background --> Score: 0.92       |
|  Image B: Blue car, beach background --> Score: 0.61      |
|  Image C: Red car, parking lot --> Score: 0.55            |
+-----------------------------------------------------------+

Architecture:
    [text_emb + img_emb] --> Linear(1536->256) --> ReLU --> Linear(256->1) --> Sigmoid
                                       |
                                       v
                                Global Score (0-1)
```

### Head 2: Patch Scorer

```
Purpose: Rate EACH REGION of the image (0 to 1)

Example:
+-----------------------------------------------------------+
|  Prompt: "a red car on the beach"                         |
|  Image: Blue car on beach                                 |
|                                                           |
|  +---------+---------+                                    |
|  | Sky     | Sky     |  --> Patch scores:                 |
|  | (0.8)   | (0.8)   |                                    |
|  +---------+---------+      Sky: 0.8 (irrelevant)         |
|  | BLUE    | Beach   |      Beach: 0.9 (correct)          |
|  | CAR     | (0.9)   |      Car: 0.2 (WRONG COLOR!)       |
|  | (0.2)   |         |                                    |
|  +---------+---------+                                    |
+-----------------------------------------------------------+

Architecture:
    For each patch:
    [text_emb + patch_emb] --> Linear(1536->256) --> ReLU --> Linear(256->1) --> Sigmoid
                                        |
                                        v
                                 Patch Score (0-1)
```

### Head 3: Saliency Map Generator

```
Purpose: Create a HEATMAP showing "risky" regions (prompt-failing)

Example:
+-----------------------------------------------------------+
|  Input: Image with patches scored                         |
|                                                           |
|  +---------+---------+      +---------+---------+         |
|  | 0.8     | 0.8     |      | .       | .       |         |
|  +---------+---------+  --> +---------+---------+         |
|  | 0.2     | 0.9     |      | XXXXX   | .       |         |
|  +---------+---------+      +---------+---------+         |
|       Patch Scores              Risk Heatmap              |
|                            (X = High Risk, . = Low)       |
+-----------------------------------------------------------+

Architecture:
    Patch Scores --> Interpolate to image size --> Risk Heatmap

    Risk = 1 - Score (low score = high risk)
```

---

## The Four Training Losses

### Loss 1: L_pair (Pairwise Ranking)

```
Goal: Chosen image should score HIGHER than rejected image

Layman: "The good essay should get a better grade than the bad essay"

+-----------------------------------------------------------+
|  Prompt: "three red apples"                               |
|                                                           |
|  Chosen Image: 3 red apples --> Score: 0.95               |
|  Rejected Image: 2 red apples --> Score: 0.60             |
|                                                           |
|  L_pair = log(1 + exp(-(0.95 - 0.60)))                    |
|         = log(1 + exp(-0.35))                             |
|         = 0.53  (wants to push gap wider)                 |
+-----------------------------------------------------------+

Formula (from proposal):
    L_pair = (1/N) * sum[ log(1 + exp(-(S_chosen - S_rejected))) ]

    This is soft margin ranking loss (not hard margin)
```

### Loss 2: L_patch (Patch-Level Ranking)

```
Goal: Good patches in chosen should score HIGHER than bad patches in rejected

Layman: "Good paragraphs should outscore bad paragraphs"

+-----------------------------------------------------------+
|  Prompt: "a red car on the beach"                         |
|                                                           |
|  Chosen Image: Red car, beach                             |
|    - Car patch score: 0.90                                |
|    - Beach patch score: 0.85                              |
|                                                           |
|  Rejected Image: Blue car, beach                          |
|    - Car patch score: 0.30 (wrong color!)                 |
|    - Beach patch score: 0.85                              |
|                                                           |
|  L_patch compares: 0.90 (chosen car) vs 0.30 (rejected)   |
+-----------------------------------------------------------+

Formula (from proposal):
    L_patch = (1/N) * sum_n[ (1/K) * sum_k[ log(1 + exp(-(S_Rk_chosen - S_Rk_rejected))) ] ]
```

### Loss 3: L_faithfulness (Binary Classification)

**NOTE**: This replaces L_policy from the proposal (see Deviation section above)

```
Goal: Detect if a patch is "unfaithful to prompt" or not

Layman: "Is this paragraph following the assignment? Yes/No."

+-----------------------------------------------------------+
|  For each patch in rejected image:                        |
|                                                           |
|  Car patch (should be red, is blue):                      |
|    - Predicted score: 0.3 (model thinks 30% faithful)     |
|    - Label: 0 (unfaithful)                                |
|    - BCE Loss: -log(1-0.3) = 0.36                         |
|                                                           |
|  Beach patch (correct):                                   |
|    - Predicted score: 0.85                                |
|    - Label: 1 (faithful)                                  |
|    - BCE Loss: -log(0.85) = 0.16                          |
|                                                           |
|  L_faithfulness = average of all BCE losses               |
+-----------------------------------------------------------+

Formula:
    L_faithfulness = BCE(patch_scores, patch_labels)

    Where patch_labels come from DETONATE dataset annotations
```

### Loss 4: L_sal (Saliency Supervision)

```
Goal: Risk heatmap should match ground-truth unfaithful regions

Layman: "Your highlighted regions should match the answer key"

+-----------------------------------------------------------+
|  Predicted Heatmap:          Ground Truth Mask:           |
|  +---------+---------+      +---------+---------+         |
|  | .       | .       |      | .       | .       |         |
|  +---------+---------+      +---------+---------+         |
|  | XXXXX   | .       |      | XXXXX   | .       |         |
|  +---------+---------+      +---------+---------+         |
|                                                           |
|  L_sal = 1 - Dice(predicted, ground_truth)                |
|        = 1 - 0.95 = 0.05  (good overlap!)                 |
+-----------------------------------------------------------+

Formula (from proposal):
    L_sal = (1/N) * sum_n[ 1 - Dice(r_predicted, m_ground_truth) ]

    Dice = 2 * |A intersect B| / (|A| + |B|)
```

### Total Loss

```
+-----------------------------------------------------------+
|                                                           |
|  L_A = lambda_1 * L_pair                                  |
|      + lambda_2 * L_patch                                 |
|      + lambda_3 * L_faithfulness   <-- replaces L_policy  |
|      + lambda_4 * L_sal                                   |
|                                                           |
|  Suggested weights:                                       |
|    lambda_1 = 1.0   (primary: image-level ranking)        |
|    lambda_2 = 1.0   (primary: patch-level ranking)        |
|    lambda_3 = 0.5   (secondary: classification)           |
|    lambda_4 = 0.1   (auxiliary: heatmap supervision)      |
|                                                           |
+-----------------------------------------------------------+
```

---

## Training Data Flow

```
+---------------------------------------------------------------------+
|                        DETONATE DATASET                              |
|                        (20K pairs)                                   |
+---------------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------------+
|  Each training sample contains:                                      |
|                                                                      |
|  {                                                                   |
|    prompt: "a red car on the beach",                                 |
|    chosen_image: <high_VQAScore_image>,                              |
|    rejected_image: <low_VQAScore_image>,                             |
|    chosen_score: 0.92,                                               |
|    rejected_score: 0.61,                                             |
|    unfaithful_patches: [                                             |
|      { bbox: [x1,y1,x2,y2], category: "wrong_attribute" }            |
|    ]                                                                 |
|  }                                                                   |
+---------------------------------------------------------------------+
                                |
                                v
+---------------------------------------------------------------------+
|                      TRAINING BATCH                                  |
|                                                                      |
|  Batch Size: 32 pairs                                                |
|                                                                      |
|  For each pair:                                                      |
|    1. Encode prompt --> text_emb                                     |
|    2. Encode chosen image --> img_emb_chosen                         |
|    3. Encode rejected image --> img_emb_rejected                     |
|    4. Extract patches from both --> patch_embs                       |
|    5. Compute global scores for both                                 |
|    6. Compute patch scores for both                                  |
|    7. Generate risk heatmap for rejected                             |
|    8. Compute all 4 losses                                           |
|    9. Backpropagate (only through trainable heads!)                  |
+---------------------------------------------------------------------+
```

---

## Step-by-Step Training Recipe (from Proposal)

### Step 1: Encode Features

```
+-----------------------------------------------------------+
|  Input: (prompt p, image I) and (prompt p, patch I_R)     |
|                                                           |
|  1. text_emb = CLIP_text(p)          # 768-d              |
|  2. img_emb = CLIP_image(I)          # 768-d              |
|  3. patch_embs = CLIP_image(I_R)     # K x 768-d          |
|                                                           |
|  All from frozen CLIP ViT-L/14                            |
+-----------------------------------------------------------+
```

### Step 2: Predict Scores and Risk Maps

```
+-----------------------------------------------------------+
|  1. Global Score:                                         |
|     S = GlobalHead(text_emb, img_emb)                     |
|                                                           |
|  2. Patch Scores:                                         |
|     S_R = PatchHead(text_emb, patch_embs)   # K scores    |
|                                                           |
|  3. Risk Heatmap:                                         |
|     r = SaliencyHead(features)              # H x W       |
+-----------------------------------------------------------+
```

### Step 3: Train with Curriculum

```
+-----------------------------------------------------------+
|  Curriculum Strategy (from proposal):                     |
|                                                           |
|  Early epochs:                                            |
|    - Use high-margin pairs (big score gap)                |
|    - Easy examples first                                  |
|    - Example: VQAScore 0.9 vs 0.3                         |
|                                                           |
|  Later epochs:                                            |
|    - Anneal to include hard pairs                         |
|    - Small score gaps                                     |
|    - Example: VQAScore 0.7 vs 0.6                         |
|                                                           |
|  Why? Prevents early collapse, learns coarse then fine    |
+-----------------------------------------------------------+
```

### Step 4: Validate

```
+-----------------------------------------------------------+
|  Validation Metrics (from proposal):                      |
|                                                           |
|  1. Global Pair-AUC                                       |
|     - "How often S_chosen > S_rejected?"                  |
|     - Target: > 0.85                                      |
|                                                           |
|  2. Patch Pair-AUC                                        |
|     - "How often good_patch > bad_patch?"                 |
|     - Target: > 0.80                                      |
|                                                           |
|  3. Per-class ROC-AUC (for faithfulness categories)       |
|     - Object presence, attribute, counting, spatial, text |
|     - Target: > 0.75 per class                            |
|                                                           |
|  4. ECE (Expected Calibration Error) for S                |
|     - "Are confidence scores reliable?"                   |
|     - Target: < 0.10                                      |
+-----------------------------------------------------------+
```

---

## Outputs for Stage B

```
+-----------------------------------------------------------+
|  After Stage A training, we have:                         |
|                                                           |
|  1. Calibration-ready Global Score S(p, I)                |
|     - Input: prompt + full image                          |
|     - Output: 0-1 faithfulness score                      |
|                                                           |
|  2. Calibration-ready Patch Score S_R(p, I_R)             |
|     - Input: prompt + image crop                          |
|     - Output: 0-1 faithfulness score per patch            |
|                                                           |
|  3. Risk Heatmap r                                        |
|     - Input: image features                               |
|     - Output: H x W heatmap of unfaithful regions         |
|                                                           |
|  4. Reason Vector (optional)                              |
|     - Which faithfulness category failed                  |
|     - e.g., "wrong_attribute", "missing_object"           |
+-----------------------------------------------------------+

These outputs feed into Stage B:
- Scorer judges tournament candidates
- Risk map guides which regions to inpaint
- Reason helps debug failures
```

---

## Training Pipeline Diagram

```
                              +------------------+
                              |    DETONATE      |
                              |    Dataset       |
                              |   (20K pairs)    |
                              +--------+---------+
                                       |
                                       v
                              +------------------+
                              |   DataLoader     |
                              |  batch_size=32   |
                              +--------+---------+
                                       |
           +---------------------------+---------------------------+
           |                           |                           |
           v                           v                           v
    +------------+              +------------+              +------------+
    |   Prompt   |              |   Chosen   |              |  Rejected  |
    |            |              |   Image    |              |   Image    |
    +-----+------+              +-----+------+              +-----+------+
          |                           |                           |
          v                           v                           v
    +------------+              +------------+              +------------+
    | CLIP Text  |              | CLIP Image |              | CLIP Image |
    |  Encoder   |              |  Encoder   |              |  Encoder   |
    |  (frozen)  |              |  (frozen)  |              |  (frozen)  |
    +-----+------+              +-----+------+              +-----+------+
          |                           |                           |
          |         +-----------------+                           |
          |         |                                             |
          v         v                                             v
       +--+----+----+--+                                  +-------+-------+
       |  Global Head  |                                  |  Global Head  |
       |   (train)     |                                  |    (train)    |
       +-------+-------+                                  +-------+-------+
               |                                                  |
               v                                                  v
         Score_chosen                                      Score_rejected
               |                                                  |
               +------------------------+-------------------------+
                                        |
                                        v
                                  +-----------+
                                  |  L_pair   |
                                  +-----+-----+
                                        |
               +------------------------+------------------------+
               |                        |                        |
               v                        v                        v
         +-----------+           +-----------+            +-----------+
         |  L_patch  |           | L_faithful|            |   L_sal   |
         +-----------+           +-----------+            +-----------+
               |                        |                        |
               +------------------------+------------------------+
                                        |
                                        v
                                  +-----------+
                                  |  L_total  |
                                  |   = L_A   |
                                  +-----+-----+
                                        |
                                        v
                                  +-----------+
                                  |  Backward |
                                  |   Pass    |
                                  +-----+-----+
                                        |
                                        v
                                  +-----------+
                                  |  Update   |
                                  |   Heads   |
                                  |  (AdamW)  |
                                  +-----------+
```

---

## Hyperparameters

```
+-----------------------------------------------------------+
|  Training Configuration:                                  |
|                                                           |
|  - Epochs: 10-20                                          |
|  - Batch size: 32                                         |
|  - Learning rate: 1e-4                                    |
|  - Optimizer: AdamW (weight_decay=0.01)                   |
|  - Scheduler: Cosine annealing with warmup                |
|  - Warmup steps: 500                                      |
|  - GPU: 1x A100 (40GB)                                    |
|  - Estimated time: 4-6 hours                              |
|                                                           |
|  Loss Weights:                                            |
|  - lambda_1 (L_pair): 1.0                                 |
|  - lambda_2 (L_patch): 1.0                                |
|  - lambda_3 (L_faithfulness): 0.5                         |
|  - lambda_4 (L_sal): 0.1                                  |
+-----------------------------------------------------------+
```

---

## Success Criteria

```
+-----------------------------------------------------------+
|  PASS if:                                                 |
|    - Global Pair-AUC > 0.85                               |
|    - Patch Pair-AUC > 0.80                                |
|    - ECE < 0.10                                           |
|    - Training converges within 20 epochs                  |
|                                                           |
|  FAIL if:                                                 |
|    - Global Pair-AUC < 0.70                               |
|    - Loss doesn't decrease after 5 epochs                 |
|    - OOM errors (reduce batch size to 16)                 |
|                                                           |
|  If FAIL, try:                                            |
|    1. Reduce to ViT-B/16 instead of ViT-L/14              |
|    2. Increase training data                              |
|    3. Tune loss weights                                   |
|    4. Add dropout (0.1) for regularization                |
+-----------------------------------------------------------+
```

---

## Next: Stage B

```
After Stage A completes:

+-----------------------------------------------------------+
|  Stage A Outputs:                                         |
|    - Trained Auditor-Scorer (As)                          |
|    - Global scorer: S(p, I)                               |
|    - Patch scorer: S_R(p, I_R)                            |
|    - Risk heatmap: r                                      |
|                                                           |
|                         |                                 |
|                         v                                 |
|                                                           |
|  Stage B Uses These To:                                   |
|    - Score tournament candidates                          |
|    - Guide which regions to fix via inpainting            |
|    - Train TSPO policy to propose better candidates       |
+-----------------------------------------------------------+
```
