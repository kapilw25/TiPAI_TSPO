# DETONATE Dataset Creation Plan (v4)

## Overview

**Goal**: Create 20K+ preference pairs for TiPAI auditor-scorer training

**Strategy**: Pure Faithfulness (maximum citations, 100% immigration safe)

**Anchor Papers**: TIFA (300+ citations), VQAScore (ECCV'24 Best Paper), T2I-CompBench (200+ citations)

---

## Key Reframe

> **"Objectionable patch"** = region where image FAILS the prompt (causes rejection)

NOT NSFW. This is the TIFA/VQAScore definition used by Google Imagen3, DALL-E 3 evals.

---

## Professor's Training Pipeline

```
Step 1: Dataset (20K+ accept/reject pairs)
    ↓
Step 2: Patch Extraction (pre-trained detectors)
    ↓
Step 3: Classifier Design (patch-level scorer)
    ↓
Step 4: Semantic Tokenizer (if classifier underperforms)
    ↓
Step 5: TSPO - Policy learns from preference

Note: TSPO is concrete. Scoring function needs exploration.
```

---

## Dataset Structure

```
DETONATE/
├── data/
│   ├── object_presence/      # 5K pairs
│   ├── attribute_binding/    # 5K pairs
│   ├── counting/             # 4K pairs
│   ├── spatial_relations/    # 3K pairs
│   └── text_rendering/       # 3K pairs
├── patches/
│   ├── missing_objects/
│   ├── wrong_attributes/
│   ├── count_errors/
│   └── spatial_errors/
├── metadata.parquet
└── README.md
```

---

## 5 Faithfulness Categories

| Category | Pairs | Prompt Example | Chosen | Rejected | Patch = |
|----------|-------|----------------|--------|----------|---------|
| **Object Presence** | 5K | "a dog and a cat" | Both present | Cat missing | Missing cat region |
| **Attribute Binding** | 5K | "red car, blue house" | Correct colors | Swapped colors | Wrong-colored object |
| **Counting** | 4K | "three apples" | 3 apples | 2 or 5 apples | Extra/missing apple |
| **Spatial Relations** | 3K | "cat on table" | Cat ON table | Cat UNDER table | Misplaced cat |
| **Text Rendering** | 3K | "sign says STOP" | Readable STOP | Garbled text | Text region |

---

## Source Prompts

| Source | Count | Category | License | Citations |
|--------|-------|----------|---------|-----------|
| **T2I-CompBench** | 6K | All categories | MIT | 200+ |
| **TIFA-v1.0** | 4K | Object, Attribute, Count | Apache 2.0 | 300+ |
| **DrawBench** | 2K | Spatial, Text | Apache 2.0 | 500+ |
| **MSCOCO captions** | 5K | Object, Attribute | CC BY 4.0 | 10K+ |
| **GenAI-Bench** | 3K | Compositional | MIT | CVPR'24 |

**Total**: 20K prompts → 80K images (4 per prompt) → 20K pairs

---

## Pair Generation Pipeline

```python
from vqascore import VQAScorer
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large")
scorer = VQAScorer(model="clip-flant5-xxl")

def generate_pair(prompt: str) -> tuple:
    # Generate 4 candidates
    images = [pipe(prompt).images[0] for _ in range(4)]

    # Score with VQAScore (TIFA alternative)
    scores = [scorer(prompt, img) for img in images]

    # Best = chosen, Worst = rejected
    chosen = images[argmax(scores)]
    rejected = images[argmin(scores)]

    return chosen, rejected, max(scores), min(scores)
```

---

## Patch Extraction Pipeline

### What is an "Objectionable Patch"?

```
Prompt: "a red car on the beach"
Chosen:  Red car ✓, Beach ✓  → Score: 0.92
Rejected: Blue car ✗, Beach ✓ → Score: 0.61

Objectionable Patch = The BLUE CAR (caused score drop)
```

### Extraction Methods

| Method | Detects | Output |
|--------|---------|--------|
| **CLIP Grad-CAM** | Semantic drift regions | Heatmap |
| **TIFA VQA** | Attribute mismatches | "Is car red?" → No → BBox |
| **Object Detector** | Missing objects | BBox of expected location |
| **Difference Map** | Chosen vs Rejected diff | Pixel-level mask |

### Architecture

```
Input: (Prompt, Chosen, Rejected)
    │
    ├──► CLIP Grad-CAM(Rejected, Prompt)
    │    → Highlight low-alignment regions
    │
    ├──► TIFA Questions: "Is there a dog?", "Is car red?"
    │    → Failed questions → Localize with GradCAM
    │
    ├──► Object Detection Gap
    │    → Expected objects not detected → Mark as missing
    │
    └──► |Chosen - Rejected| pixel diff
         → High-diff regions = likely patches

    ▼
Fusion: Weighted union → NMS → Top 5 patches per pair
```

### Patch Extractor Code

```python
class FaithfulnessPatchExtractor:
    def __init__(self):
        self.clip = load_clip("ViT-L/14")
        self.tifa = TIFAScorer()
        self.detector = load_yolo("yolov8l")

    def extract(self, prompt, chosen, rejected):
        patches = []

        # 1. CLIP Grad-CAM on rejected image
        gradcam = clip_gradcam(rejected, prompt)
        low_align_regions = threshold(gradcam, t=0.3)
        patches.extend(regions_to_patches(low_align_regions, "semantic_drift"))

        # 2. TIFA question-answering
        questions = self.tifa.generate_questions(prompt)
        for q, expected_ans in questions:
            actual_ans = self.tifa.answer(rejected, q)
            if actual_ans != expected_ans:
                # Localize the failed attribute
                bbox = gradcam_for_question(rejected, q)
                patches.append(Patch(bbox, category="attribute_fail", question=q))

        # 3. Missing object detection
        expected_objects = parse_nouns(prompt)
        detected = self.detector(rejected)
        for obj in expected_objects:
            if obj not in detected:
                patches.append(Patch(category="missing_object", label=obj))

        # 4. Pixel difference (simple but effective)
        diff = np.abs(chosen - rejected)
        high_diff_regions = threshold(diff.mean(axis=2), t=50)
        patches.extend(regions_to_patches(high_diff_regions, "pixel_diff"))

        # Deduplicate and rank
        patches = nms(patches, iou_threshold=0.5)
        return sorted(patches, key=lambda p: p.confidence, reverse=True)[:5]
```

---

## Auditor-Scorer Architecture

```
Prompt ────► CLIP Text Encoder ────► t_emb (768-d)
                                         │
                                    cosine_sim ──► Alignment Score
                                         │
Image ─────► CLIP Image Encoder ───► i_emb (768-d)
    │
    └──────► Patch Extractor ──────► [p1, p2, ..., pN]
                                         │
                                    ┌────▼────┐
                                    │  Patch  │
                                    │ Scorer  │──► Patch Scores [s1, s2, ..., sN]
                                    │  (MLP)  │
                                    └─────────┘
                                         │
                                         ▼
                                    Risk Heatmap (overlay on image)
```

### Scoring Function (To Be Explored)

```python
class AuditorScorer(nn.Module):
    def __init__(self):
        self.clip = load_clip("ViT-L/14", frozen=True)
        self.patch_scorer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, prompt, patches):
        # Global alignment
        t_emb = self.clip.encode_text(prompt)
        i_emb = self.clip.encode_image(image)
        alignment = cosine_sim(t_emb, i_emb)

        # Patch-level scores
        patch_embs = [self.clip.encode_image(p) for p in patches]
        patch_scores = [self.patch_scorer(emb) for emb in patch_embs]

        # Combined score (exploration needed)
        # Option 1: alignment * (1 - max(patch_scores))
        # Option 2: alignment - sum(patch_scores)
        # Option 3: learned fusion

        return alignment, patch_scores
```

---

## Verification Protocol

| Category | Verification Question | Answer Type | Agreement Target |
|----------|----------------------|-------------|------------------|
| Object Presence | "Is there a [object]?" | Yes/No | 95%+ |
| Attribute Binding | "Is the [object] [color]?" | Yes/No | 90%+ |
| Counting | "How many [object]?" | Number | 85%+ |
| Spatial Relations | "Is [A] [relation] [B]?" | Yes/No | 85%+ |
| Text Rendering | "What does the sign say?" | Text | 80%+ |

**Human verification**: 10% random sample (2K pairs) via Prolific
**Cost estimate**: 2K × $0.10 × 3 annotators = $600

---

## Implementation Phases

| Phase | Task | Output | Compute |
|-------|------|--------|---------|
| **1** | Download T2I-CompBench, TIFA, DrawBench prompts | 20K prompts | CPU |
| **2** | Generate 4 images per prompt (SD 3.5) | 80K images | 40 GPU-hrs |
| **3** | Score with VQAScore | 80K scores | 20 GPU-hrs |
| **4** | Create pairs (best vs worst) | 20K pairs | CPU |
| **5** | Extract patches | 100K patches | 10 GPU-hrs |
| **6** | Human verification (10%) | Validated subset | $600 |
| **7** | Package for HuggingFace | DETONATE dataset | - |

**Total**: ~70 GPU-hours on A100 + $600 annotation

---

## Why This Wins

| Factor | Value |
|--------|-------|
| **Citation anchor** | TIFA (300+), VQAScore (ECCV'24), T2I-CompBench (200+) |
| **Industry validation** | Google Imagen3 uses VQAScore |
| **Immigration safe** | 100% - no NSFW, no hate, no violence |
| **Easy verification** | Binary VQA questions |
| **Novel contribution** | First PATCH-LEVEL faithfulness dataset |

---

## Differentiation from Existing Work

| Dataset | What it provides | DETONATE adds |
|---------|-----------------|---------------|
| TIFA | Prompts + VQA questions | **Preference pairs + patches** |
| T2I-CompBench | Prompts + categories | **Chosen/rejected images + patches** |
| Pick-a-Pic | Image pairs (human pref) | **Patch-level localization** |
| ImageReward | Scalar scores | **Patch-level attribution** |

**DETONATE = First dataset with patch-level faithfulness annotations for preference learning**

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| VQAScore model bias | Cross-validate with TIFA, human spot-check |
| Patch extraction noise | Multiple methods + NMS fusion |
| Low inter-annotator agreement | Focus on binary questions, 3 annotators |
| Compute cost | Use SD 3.5 Medium for initial experiments |
