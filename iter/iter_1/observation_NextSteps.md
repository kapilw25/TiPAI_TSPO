# POC Iteration 1: Observations & Next Steps

**Date**: 2024-12-13
**Status**: Gap detection conceptually correct, but CLIP accuracy insufficient

---

## 1. Current Pipeline

```
prompts.json (20 scenes × 4 variations = 80 prompts)
       │
       ▼
m01_generate_images.py (FLUX.1-dev) → 80 images
       │
       ▼
m02_extract_signals.py (SAM + CLIP) → signals table
       │
       ▼
m03_create_risk_maps.py (RED outline) → risk maps
```

---

## 2. Observed Issues

### 2.1 False Positives: Too Many Objects Outlined

**scene_01_v1** (color change: red car → purple car)
- **Expected**: Only purple car outlined
- **Actual**: Car, palm tree, umbrella, surfboard ALL outlined
- **Root cause**: CLIP detects wrong colors for multiple objects (noise)

**scene_01_v2** (object swap: car → motorcycle)
- **Expected**: Only motorcycle outlined
- **Actual**: Motorcycle, palm tree, surfboard ALL outlined
- **Root cause**: CLIP misclassifies segments, creates false gaps

### 2.2 False Negatives: Missing Outlines

**scene_02_v1** (color change: golden retriever → black retriever)
- **Expected**: Black retriever outlined
- **Actual**: NO outline at all
- **Root cause**: CLIP failed to detect color mismatch

**scene_02_v2** (object swap: retriever → hamster)
- **Expected**: Hamster outlined
- **Actual**: NO outline at all
- **Root cause**: "retriever" marked as "missing" (segment_id=-1), hamster not in expected list so ignored

### 2.3 Spatial Detection Not Implemented

**scene_01_v3** (spatial: car on beach → car floating in water)
- **Expected**: Car outlined (wrong position)
- **Actual**: No outline
- **Root cause**: Current system only detects object presence and color, not spatial relations

---

## 3. Why This Matters (Research Constraint)

From `plan_overview.md` - Stage A Goal:

```
Goal: Train a JUDGE that can:
  Input:  Prompt + Image (ONLY these two)
  Output: Score + Risk heatmap

The system must detect: "the car is blue, not red"
just from comparing IMAGE against PROMPT
```

### Prompt-Diff Approach is CHEATING

| Approach | Why Invalid |
|----------|-------------|
| Compare v0 vs v1 prompts | In real world, you don't have v0 |
| Know which object changed | That's the system's JOB to discover |

**The v1/v2/v3 variations exist ONLY to create training data with known ground-truth issues.**

The detection system must work with ONLY:
- `baseline_prompt` (what user asked for)
- `image` (what was generated)

---

## 4. Current Detection Logic (Conceptually Correct)

```
baseline_prompt → parse → expected: [red car, blue umbrella, white surfboard, ...]
                              │
                              ▼
image → SAM segments → CLIP classify → detected: [purple car, blue umbrella, ...]
                              │
                              ▼
                    Compare expected vs detected
                              │
                              ▼
                    Gap: car has wrong_color (expected=red, found=purple)
                              │
                              ▼
                    RED outline on car segment
```

**The logic is RIGHT. The problem is CLIP accuracy.**

---

## 5. Root Cause Analysis

| Issue | Cause | Impact |
|-------|-------|--------|
| CLIP color detection noisy | Detects wrong colors for correct objects | False positives |
| CLIP object classification noisy | Misclassifies segments | False positives/negatives |
| Object swap detection gap | "Missing" objects have segment_id=-1 | False negatives |
| Spatial detection missing | Only checks presence/color, not position | v3 not detected |

---

## 6. Improvement Options (Without Cheating)

### Option A: Better CLIP Prompts (Quick Win)

Current:
```python
classify_object(segment, ["car", "umbrella", "surfboard"])
detect_color(segment)  # Returns single best color
```

Improved:
```python
# Compare specific prompts
score_red = clip_score(segment, "a photo of a red car")
score_blue = clip_score(segment, "a photo of a blue car")
if abs(score_red - score_blue) > threshold:
    # Confident about color
```

### Option B: Higher Thresholds

Current: `OBJECT_MATCH_THRESHOLD = 0.55`

Improved:
- Only flag if confidence > 0.7
- Require significant color score difference (e.g., > 0.1)

### Option C: Ensemble Queries

```python
# Multiple prompts per segment
prompts = [
    f"a {expected_color} {noun}",
    f"a photo of a {expected_color} {noun}",
    f"this is a {expected_color} {noun}"
]
scores = [clip_score(segment, p) for p in prompts]
avg_score = mean(scores)  # More robust
```

### Option D: Object Swap Detection Fix

Current issue: When car → motorcycle, "car" is missing (segment_id=-1), motorcycle ignored.

Fix:
```python
# For each detected segment NOT matching any expected object
for det in detected_objects:
    if det.label not in expected_nouns:
        # This is a WRONG OBJECT - flag it
        gaps.append(ObjectGap(
            issue="wrong_object",
            expected="one of expected objects",
            found=det.label,
            segment_id=det.segment_id  # Now we can outline it!
        ))
```

### Option E: Accept POC Limitations

- Document current accuracy (~40-60% correct)
- Show concept works even if noisy
- Plan Stage A training to learn better detection

---

## 7. Recommended Next Steps

### Immediate (This Iteration)

1. **Implement Option D** (Object swap fix) - catches v2 variations
2. **Implement Option A** (Better CLIP prompts) - reduces noise
3. **Re-run m02 + m03** and validate

### Future (Stage A Training)

1. Use current POC data as noisy training signal
2. Train Auditor-Scorer to learn patch-level risk
3. Fine-tune on DETONATE dataset

---

## 8. Key Insight

> The gap detection LOGIC is correct. The IMPLEMENTATION relies on CLIP, which isn't accurate enough for fine-grained color/object detection. Stage A training exists precisely to learn a better detector than zero-shot CLIP.

**Current POC demonstrates the concept. Stage A training will improve accuracy.**

---

## 9. Files to Modify

| File | Change |
|------|--------|
| `src/utils/clip_utils.py` | Better color comparison prompts |
| `src/m02_extract_signals.py` | Add wrong_object detection for swaps |
| `src/m02_extract_signals.py` | Higher confidence thresholds |

---

## 10. Success Criteria for Next Iteration

| Metric | Current | Target |
|--------|---------|--------|
| v1 (color change) detection | ~50% | >70% |
| v2 (object swap) detection | ~20% | >60% |
| v3 (spatial) detection | 0% | Deferred to Stage A |
| False positive rate | High | <30% |
