# TiPAI Dataset Creation Plan (v5 - Systematic Variations)

## Overview

**Goal**: Create 20K+ preference pairs for TiPAI auditor-scorer training

**Strategy**: Systematic Variation Approach (based on TIFA benchmark categories)

**Anchor Papers**: TIFA (300+ citations), VQAScore (ECCV'24 Best Paper), T2I-CompBench (200+ citations)

---

## Key Reframe

> **"Objectionable patch"** = region where image FAILS the prompt (causes rejection)

NOT NSFW. This is the TIFA/VQAScore definition used by Google Imagen3, DALL-E 3 evals.

---

## Why Systematic Variations > Random Seeds

| Aspect | Random Seeds (OLD) | Systematic Variations (NEW) |
|--------|-------------------|----------------------------|
| Failure Type | Unknown | Known (we control it) |
| Interpretability | "This image is worse" | "This image has wrong color" |
| Reproducibility | Seed-dependent | Deterministic |
| Annotation Cost | Need VQAScore | Built-in labels |
| Research Value | Limited | Category-specific analysis |

---

## The 4 Variation Types

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SYSTEMATIC VARIATION STRUCTURE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  BASE PROMPT: "a shiny red sports car parked on a sandy beach"      │
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐                         │
│  │  v0_original    │    │  v1_attribute   │                         │
│  │  (CHOSEN)       │    │  (REJECTED)     │                         │
│  │                 │    │                 │                         │
│  │  "red car on    │    │  "BLUE car on   │  ← Wrong COLOR          │
│  │   beach"        │    │   beach"        │                         │
│  └─────────────────┘    └─────────────────┘                         │
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐                         │
│  │  v2_object      │    │  v3_spatial     │                         │
│  │  (REJECTED)     │    │  (REJECTED)     │                         │
│  │                 │    │                 │                         │
│  │  "red BICYCLE   │    │  "red car IN    │  ← Wrong LOCATION       │
│  │   on beach"     │    │   THE OCEAN"    │                         │
│  └─────────────────┘    └─────────────────┘                         │
│         ↑                                                           │
│    Wrong OBJECT                                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pairing Logic (3 Pairs per Base Prompt)

```
┌──────────────────────────────────────────────────────────────────┐
│                      PAIR CREATION                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   v0_original ─────┬────► vs v1_attribute  →  ATTRIBUTE FAILURE  │
│   (always chosen)  │                                              │
│                    ├────► vs v2_object     →  OBJECT FAILURE     │
│                    │                                              │
│                    └────► vs v3_spatial    →  SPATIAL FAILURE    │
│                                                                   │
│   Result: 3 pairs per base prompt, each with KNOWN failure type  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Layman Example:**
- Prompt: "red car on beach"
- Pair 1: Correct red car (chosen) vs Blue car (rejected) → "attribute" failure
- Pair 2: Correct red car (chosen) vs Red bicycle (rejected) → "object" failure
- Pair 3: Correct red car (chosen) vs Car in ocean (rejected) → "spatial" failure

---

## Dataset Scale

| Stage | Base Prompts | × Variations | × Seeds | = Images | Pairs |
|-------|--------------|--------------|---------|----------|-------|
| **POC** | 20 | 4 | 1 | 80 | 60 |
| **Paper** | 5,000 | 4 | 1 | 20,000 | 15,000 |

---

## 5 Faithfulness Categories

| Category | Base Prompts | v0 (Chosen) | v1 (Attribute) | v2 (Object) | v3 (Spatial) |
|----------|--------------|-------------|----------------|-------------|--------------|
| **Object Presence** | 1,250 | "dog and cat" | "green cat" | "dog and bird" | "cat on roof" |
| **Attribute Binding** | 1,250 | "red car" | "blue car" | "red bicycle" | "car behind" |
| **Counting** | 1,000 | "three apples" | "shiny apples" | "three oranges" | "five apples" |
| **Spatial Relations** | 750 | "cat on table" | "brown cat" | "dog on table" | "cat under table" |
| **Compositional** | 750 | "chef in kitchen" | "chef in red" | "waiter cooking" | "chef in bathroom" |

---

## Source Prompts (for Paper Scale)

| Source | Count | Category | License | Citations |
|--------|-------|----------|---------|-----------|
| **T2I-CompBench** | 6K | All categories | MIT | 200+ |
| **TIFA-v1.0** | 4K | Object, Attribute, Count | Apache 2.0 | 300+ |
| **DrawBench** | 2K | Spatial, Text | Apache 2.0 | 500+ |
| **GenAI-Bench** | 3K | Compositional | MIT | CVPR'24 |

**Selection**: 5,000 base prompts × 4 variations = 20,000 images

---

## SDXL Model Choice: Base Only vs Base + Refiner

**Reference**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

| Factor | Option A: Base Only | Option B: Base + Refiner |
|--------|---------------------|--------------------------|
| **vRAM** | ~12-14GB | ~20-24GB |
| **Speed** | ~20s/image | ~35s/image |
| **Quality** | Good enough for POC | Best quality |
| **Complexity** | Simple | More code |

**POC Decision: Option A (Base only)**

Reasoning:
1. POC goal = demonstrate patch-level heatmaps, not maximum image quality
2. Refiner adds complexity without changing the faithfulness concept
3. Saves vRAM → can use cheaper A10-24GB instance
4. Paper-scale can add refiner later for quality improvement

---

## Why This Dataset Design Wins

| Factor | Value |
|--------|-------|
| **Failure types known** | No VQAScore annotation needed |
| **Deterministic** | Fixed seed (42), reproducible |
| **Citation anchor** | TIFA categories, T2I-CompBench style |
| **Immigration safe** | 100% - no NSFW, pure faithfulness |
| **Scalable** | Just add more base prompts |
