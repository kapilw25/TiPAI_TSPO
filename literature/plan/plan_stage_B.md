# Stage B: Auditor-Inpaint + TSPO Training Plan

## Deviation from Proposal

| Proposal | Our Version | Reason |
|----------|-------------|--------|
| Policy safety P_i in scoring | Skip P_i, use F_i (faithfulness) only | Immigration-safe, no NSFW detection |
| tau_P thresholds | tau_F thresholds only | Same reason |
| Risk includes NSFW/weapon | Risk = unfaithfulness to prompt | Same reason |

---

## Big Picture

```
Stage A = We trained a JUDGE (Auditor-Scorer)
          - Can grade images (0-1)
          - Can identify bad regions (risk heatmap)

Stage B = We train an ARTIST (Auditor-Inpaint) + COACH (TSPO)
          - Artist (Ag) fixes bad regions
          - Coach (TSPO) learns which "brush settings" work best
```

---

## B1: Pretrain Ag (The Artist)

### What is Ag?

**Layman Analogy**: Teaching a photo editor to fix specific regions

```
Imagine Photoshop's "Content-Aware Fill" but SMARTER:
- It knows WHAT the region should look like (from prompt)
- It knows WHEN in the generation process it's editing (timestep)
- It blends seamlessly with surrounding pixels
```

### Input Formation (Context + Noise Inside Mask)

```
What the Artist (Ag) receives:

+---------------------------+     +---------------------------+
|                           |     |                           |
|   Original Image          |     |   Input to Ag             |
|   (with bad region)       |     |                           |
|   +-------+               |     |   +-------+               |
|   | BLUE  |  Beach        | --> |   |///////|  Beach        |
|   | CAR   |               |     |   |NOISE//|               |
|   +-------+               |     |   +-------+               |
|                           |     |      ^                    |
+---------------------------+     |      |                    |
                                  |   Mask says: "fix here"   |
                                  +---------------------------+

Formula (from proposal):
    X = I_(t-1) * (1 - m) + epsilon * m

    Where:
    - I_(t-1) = current image (keep outside mask)
    - m = binary mask (1 = region to fix)
    - epsilon ~ N(0, sigma_t^2) = noise scaled by timestep
    - t = chosen from a timestep band
```

### Why Timestep-Aware?

```
+-------------------------------------------------------------------+
|  Diffusion generates images step-by-step:                          |
|                                                                    |
|  t=T (start)     t=T/2 (middle)      t=0 (end)                    |
|  Pure noise  --> Coarse shapes  --> Fine details                  |
|                                                                    |
|  Ag must know WHERE in this process we're editing:                 |
|                                                                    |
|  - Early (t high): Can make big changes, coarse features           |
|  - Late (t low): Only small tweaks, fine details                   |
|                                                                    |
|  This is why Ag takes timestep t as input!                         |
+-------------------------------------------------------------------+
```

### Target: What Should Ag Produce?

```
+---------------------------+
|                           |
|   Target Output           |
|                           |
|   +-------+               |
|   | RED   |  Beach        |  <-- From DETONATE "chosen" image
|   | CAR   |               |      The GOOD version of this region
|   +-------+               |
|                           |
+---------------------------+

Y = I+_R  (the patch from the CHOSEN image in the pair)
```

### Ag Architecture

```
+-------------------------------------------------------------------+
|                        AUDITOR-INPAINT (Ag)                        |
+-------------------------------------------------------------------+
                                |
        +-----------------------+-----------------------+
        |                       |                       |
        v                       v                       v
+---------------+       +---------------+       +---------------+
|    Context    |       |     Mask      |       |   Timestep    |
| I*(1-m)+noise |       |      m        |       |      t        |
+-------+-------+       +-------+-------+       +-------+-------+
        |                       |                       |
        +----------+------------+-----------+-----------+
                   |                        |
                   v                        v
           +---------------+        +---------------+
           |  U-Net or     |        |   Timestep    |
           |  Transformer  |        |   Embedding   |
           |   Backbone    |<-------+               |
           +-------+-------+        +---------------+
                   |
                   v
           +---------------+
           |    Prompt     |
           |   Embedding   |-------> Cross-attention
           +---------------+
                   |
                   v
           +---------------+
           |   Predicted   |
           |   Inpainted   |
           |    Region     |
           +---------------+
```

### Training Losses for Ag (Pretrain Phase)

```
+-------------------------------------------------------------------+
|  L_inpaint = beta_1 * ||predicted - target||_1    (pixel match)   |
|            + beta_2 * (1 - CLIP(prompt, predicted)) (semantic)    |
|            + beta_3 * Risk(predicted)              (faithfulness) |
|            + beta_4 * LPIPS_boundary(pred, target) (seam quality) |
+-------------------------------------------------------------------+

Layman explanation:
1. ||pred - target||_1: "Does it look like the target pixel-by-pixel?"
2. CLIP alignment: "Does it match the prompt meaning?"
3. Risk: "Is it now faithful?" (scored by Stage A)
4. LPIPS_boundary: "Are the edges smooth? No visible seam?"
```

### Preference Loss (Contrastive)

```
+-------------------------------------------------------------------+
|  L_pref = -log[ exp(<f(pred), f(I+)>/tau) /                       |
|                 (exp(<f(pred), f(I+)>/tau) + exp(<f(pred), f(I-)>/tau)) ]
|                                                                    |
|  Where:                                                            |
|    f() = feature extractor (e.g., CLIP image encoder)             |
|    I+ = chosen region (good)                                       |
|    I- = rejected region (bad)                                      |
|    tau = temperature                                               |
|                                                                    |
|  Layman: "Prediction should be similar to good, different from bad"|
+-------------------------------------------------------------------+
```

### Total Pretrain Loss

```
+-------------------------------------------------------------------+
|                                                                    |
|  L_G^pre = L_inpaint + beta_5 * L_pref                            |
|                                                                    |
|  Suggested weights:                                                |
|    beta_1 = 1.0   (L1 reconstruction)                             |
|    beta_2 = 0.5   (CLIP alignment)                                |
|    beta_3 = 0.3   (faithfulness risk)                             |
|    beta_4 = 0.2   (boundary smoothness)                           |
|    beta_5 = 0.5   (preference contrastive)                        |
|                                                                    |
+-------------------------------------------------------------------+
```

### Short Inversion & Latent Blending (Seam-Free)

```
Problem: If Ag edits in RGB, how do we put it back into diffusion?

Solution: "Short DDIM Inversion" (d steps)

+-------------------+     +-------------------+     +-------------------+
|  Edited Region    | --> |  Invert back to   | --> |  Blend in latent  |
|  (RGB space)      |     |  latent space     |     |  space            |
|  from Ag          |     |  (d steps DDIM)   |     |                   |
+-------------------+     +-------------------+     +-------------------+

Blending formula:
    z_(t-1) = (1 - alpha*m) * z_ctrl + (alpha*m) * z_edit

Where:
    - z_ctrl = original latent (control)
    - z_edit = inverted edited region
    - m = feathered mask (soft edges)
    - alpha = alpha(t), varies by timestep

Why feathered mask?
    Hard edges --> visible seams
    Soft edges --> smooth blending
```

### Pretrain Recipe

```
+-------------------------------------------------------------------+
|  Step 1: Sample a pair (prompt, I+, I-) from DETONATE              |
|                                                                    |
|  Step 2: Mine region R with mask m from I- (rejected)              |
|                                                                    |
|  Step 3: Choose random timestep t from band [t_min, t_max]         |
|                                                                    |
|  Step 4: Form input X = I- * (1-m) + noise * m                     |
|                                                                    |
|  Step 5: Predict I_hat = Ag(X, m, prompt, t)                       |
|                                                                    |
|  Step 6: Compute L_G^pre using target Y = I+_R                     |
|                                                                    |
|  Step 7: Update Ag weights                                         |
+-------------------------------------------------------------------+
```

### Pretrain Validation Metrics

```
+-------------------------------------------------------------------+
|  1. Patch Risk Reduction (delta_r)                                 |
|     - "Did the fix reduce unfaithfulness?"                         |
|     - Compare Risk(before) vs Risk(after)                          |
|     - Target: delta_r > 0.2                                        |
|                                                                    |
|  2. CLIP Score Improvement (delta_CLIP)                            |
|     - "Did semantic alignment improve?"                            |
|     - Target: delta_CLIP > 0.05                                    |
|                                                                    |
|  3. Boundary LPIPS                                                 |
|     - "Are seams invisible?"                                       |
|     - Measure LPIPS on a ring around the mask                      |
|     - Target: LPIPS_boundary < 0.1                                 |
+-------------------------------------------------------------------+
```

---

## B2: Tournament and Guards (Competition with Safety Net)

### What is the Tournament?

**Layman Analogy**: American Idol audition with a "control" contestant

```
Situation: We found a bad region to fix
Strategy:  Generate N=5 different fixes, pick the best one
Safety:    Always include the ORIGINAL as a contestant (C_0)
           Only accept a fix if it's CLEARLY better than original
```

### Candidate Generation

```
For each bad region detected:

                    +-------------------+
                    |   Policy (TSPO)   |
                    |   draws N actions |
                    +--------+----------+
                             |
         +-------+-------+-------+-------+-------+
         |       |       |       |       |       |
         v       v       v       v       v       v
       a_1     a_2     a_3     a_4     a_5    (actions = knob settings)
         |       |       |       |       |
         v       v       v       v       v
    +------+ +------+ +------+ +------+ +------+
    | Ag   | | Ag   | | Ag   | | Ag   | | Ag   |
    +------+ +------+ +------+ +------+ +------+
         |       |       |       |       |
         v       v       v       v       v
       C_1     C_2     C_3     C_4     C_5    (candidate fixes)
         |       |       |       |       |
         +-------+-------+---+---+-------+
                             |
                             v
                    +---------------+
                    |     C_0       |  <-- ALWAYS included!
                    |   (control)   |      The "do nothing" option
                    |   Original    |
                    +---------------+
```

### State for Policy

```
+-------------------------------------------------------------------+
|  State s = (prompt, z_(t-1), I_(t-1), mask m, timestep t)          |
|                                                                    |
|  The policy sees:                                                  |
|    - What the user asked for (prompt)                              |
|    - Current latent state                                          |
|    - Current decoded image                                         |
|    - Which region needs fixing (mask)                              |
|    - Where we are in generation (timestep)                         |
+-------------------------------------------------------------------+
```

### Scoring Each Candidate

```
For each candidate C_i (i = 0, 1, ..., N):

+-------------------------------------------------------------------+
|  Compose full image: I_composed = Paste(I_(t-1), C_i, region R)   |
|                                                                    |
|  Score with Stage A:                                               |
|  (S_i, F_i, B_i) = Auditor_Scorer(prompt, I_composed)             |
|                                                                    |
|  Where:                                                            |
|    S_i = Overall score (0-1)                                       |
|    F_i = Faithfulness score (0-1)                                  |
|    B_i = Seam/boundary quality = exp(-kappa * LPIPS_boundary)      |
+-------------------------------------------------------------------+

Note: We skip P_i (policy safety) from the proposal due to immigration concerns.
Our F_i (faithfulness) serves as our "policy" - images must match the prompt.
```

### Guarded Margin Selection (The Safety Net)

```
"Only accept a fix if it's CLEARLY better than doing nothing"

+-------------------------------------------------------------------+
|  Utility for candidate i:                                          |
|                                                                    |
|  u_i = (S_i - S_0 - delta)_+  *  1[F_i >= tau_F(t)]  *  B_i       |
|        ------------------       -----------------       ---        |
|        Must beat control        Must be faithful        Seam       |
|        by margin delta          above threshold         quality    |
|                                                                    |
|  Where:                                                            |
|    (x)_+ = max(0, x)                                               |
|    delta = margin required (e.g., 0.05)                            |
|    tau_F(t) = faithfulness threshold (varies by timestep)          |
|    B_i = boundary quality (0-1)                                    |
+-------------------------------------------------------------------+

Selection rule:
    i* = argmax_i(u_i)

    if u_i* > 0:
        ACCEPT candidate C_i* (it won!)
    else:
        KEEP control C_0 (no edit, stay safe)
```

### Why This is "Monotone Non-Regression"

```
+-------------------------------------------------------------------+
|  Key Guarantee: We NEVER make things worse!                        |
|                                                                    |
|  Case 1: A candidate beats control by margin                       |
|          --> We accept it (improvement!)                           |
|                                                                    |
|  Case 2: No candidate beats control                                |
|          --> We keep control (no change)                           |
|          --> Score stays the same (not worse)                      |
|                                                                    |
|  This is like "innocent until proven guilty":                      |
|  The edit must PROVE it's better before we accept it.              |
+-------------------------------------------------------------------+
```

### Guard Flow Diagram

```
                        All Candidates Scored
                        (S_0, S_1, ..., S_N)
                               |
                               v
                    +---------------------+
                    | For each i > 0:     |
                    | Check S_i > S_0 +   |
                    |        delta?       |
                    +----------+----------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
        +----------+                      +----------+
        |   YES    |                      |    NO    |
        | (passes  |                      | (fails   |
        |  margin) |                      |  margin) |
        +----+-----+                      +----+-----+
             |                                 |
             v                                 |
    +------------------+                       |
    | Check: F_i >=    |                       |
    |        tau_F(t)? |                       |
    +--------+---------+                       |
             |                                 |
      +------+------+                          |
      |             |                          |
      v             v                          |
  +-------+    +--------+                      |
  |  YES  |    |   NO   |                      |
  +---+---+    +---+----+                      |
      |            |                           |
      v            +---------------------------+
  +--------+                                   |
  | Compute|                                   |
  | u_i    |                                   |
  +---+----+                                   |
      |                                        |
      v                                        v
+-------------+                        +---------------+
| Pick best   |                        | All u_i <= 0  |
| i* = argmax |                        |               |
+------+------+                        +-------+-------+
       |                                       |
       v                                       v
+-------------+                        +---------------+
| ACCEPT C_i* |                        | KEEP CONTROL  |
| Short-invert|                        | C_0 (no edit) |
| & blend     |                        +---------------+
+-------------+
```

---

## B3: TSPO - Learning Which Knobs Work Best

### What is TSPO?

**Layman Analogy**: Training a photographer to choose good camera settings

```
The photographer (policy) must learn:
- "For this type of scene, use these settings"
- "If the first few shots failed, try something different"
- "Don't waste film on expensive settings unless necessary"
```

### What are the "Knobs" (Actions)?

```
+-------------------------------------------------------------------+
|  Each action a_i is a combination of settings:                     |
|                                                                    |
|  1. Mask dilation/feathering                                       |
|     - How much to expand the mask? (pixels: 0, 5, 10, 20)          |
|     - Soft or hard edges? (feather: 0, 3, 7 pixels)                |
|                                                                    |
|  2. Inside-mask CFG (Classifier-Free Guidance)                     |
|     - How strongly to follow prompt inside mask?                   |
|     - Values: 3, 5, 7, 10, 15                                      |
|     - Higher = more faithful but potentially artifacts             |
|                                                                    |
|  3. Prompt token emphasis                                          |
|     - Weight certain words higher                                  |
|     - e.g., "a (RED:1.5) car" emphasizes RED                       |
|                                                                    |
|  4. Latent noise jitter                                            |
|     - How much randomness in the fix?                              |
|     - Scale: 0.0, 0.1, 0.2, 0.5                                    |
|                                                                    |
|  5. Random seed                                                    |
|     - Different seed = different result                            |
|     - For diversity in candidates                                  |
|                                                                    |
|  6. Short inversion depth (d)                                      |
|     - How many steps to invert? (d: 1, 3, 5, 10)                   |
|     - More = better blend, but more compute                        |
|                                                                    |
|  7. Light LoRA routing (optional)                                  |
|     - Which fine-tuned adapter to use?                             |
|     - e.g., "realistic", "artistic", "none"                        |
+-------------------------------------------------------------------+
```

### Policy Architecture

```
+-------------------------------------------------------------------+
|                         TSPO POLICY (pi_theta)                     |
+-------------------------------------------------------------------+
                                |
        +-----------------------+-----------------------+
        |                       |                       |
        v                       v                       v
+---------------+       +---------------+       +---------------+
|    Prompt     |       |  Image/Latent |       |  Timestep +   |
|   Embedding   |       |   Features    |       |    Mask       |
+-------+-------+       +-------+-------+       +-------+-------+
        |                       |                       |
        +-----------+-----------+-----------+-----------+
                    |
                    v
            +---------------+
            |     MLP or    |
            |  Transformer  |
            |    Encoder    |
            +-------+-------+
                    |
                    v
            +---------------+
            |   Action      |
            | Distribution  |
            | (per knob)    |
            +-------+-------+
                    |
        +-----------+-----------+-----------+
        |           |           |           |
        v           v           v           v
   +--------+  +--------+  +--------+  +--------+
   | Mask   |  |  CFG   |  | Noise  |  | Depth  |
   | dilation| | inside |  | jitter |  |   d    |
   +--------+  +--------+  +--------+  +--------+

Each knob has its own categorical distribution.
Sample one value per knob to form action a.
```

### Credits: Who Gets Rewarded?

```
After tournament, we know the utilities. How to credit each candidate?

Method 1: Leave-one-out advantage
    A_i = u_i - max(u_j for j != i)

    "How much worse would tournament be without candidate i?"

Method 2: Soft credits (softmax)
    w_i = softmax(u_i / tau) - 1/N

    "Proportional credit based on relative utility"

+-------------------------------------------------------------------+
|  Example:                                                          |
|  Utilities: u_0=0.0, u_1=0.3, u_2=0.8, u_3=0.5, u_4=0.2           |
|                                                                    |
|  Winner: C_2 (u=0.8)                                               |
|                                                                    |
|  Leave-one-out credits:                                            |
|    A_1 = 0.3 - 0.8 = -0.5  (worse than winner)                    |
|    A_2 = 0.8 - 0.5 = +0.3  (beat 2nd place!) <-- POSITIVE         |
|    A_3 = 0.5 - 0.8 = -0.3  (worse than winner)                    |
|    A_4 = 0.2 - 0.8 = -0.6  (much worse)                           |
|                                                                    |
|  Only the winner (and maybe close 2nd) get positive credit.        |
+-------------------------------------------------------------------+
```

### TSPO Objective

```
+-------------------------------------------------------------------+
|                                                                    |
|  L_TSPO = - sum_i[ w_i * log(pi_theta(a_i | s)) ]  (reward good)  |
|           - beta * H[pi_theta(. | s)]               (entropy)     |
|           + lambda_c * Cost({a_i})                  (cheap pref)  |
|           - lambda_d * sum_{i<j}[ d(C_i, C_j) ]     (diversity)   |
|                                                                    |
+-------------------------------------------------------------------+

Term by term:
1. Main term: Increase probability of actions that won
   - w_i > 0 for winners --> increase log(pi(a_i))
   - w_i < 0 for losers --> decrease log(pi(a_i))

2. Entropy regularization: Keep exploring
   - Don't collapse to always picking same action
   - beta controls exploration vs exploitation

3. Cost penalty: Prefer cheaper actions
   - More inversion steps (d) = more compute
   - Cost({a}) might penalize high d values

4. Diversity bonus: Encourage varied candidates
   - d(C_i, C_j) = feature distance between candidates
   - Want candidates to be different, not 5 copies
```

### TSPO Training Loop

```
+-------------------------------------------------------------------+
|  For each audited region (R, m) at timestep t:                     |
|                                                                    |
|  1. Form state s = (prompt, z_(t-1), I_(t-1), m, t)               |
|                                                                    |
|  2. Sample N actions from policy:                                  |
|     a_1, a_2, ..., a_N ~ pi_theta(a | s)                          |
|                                                                    |
|  3. Generate candidates:                                           |
|     C_i = Ag(s; a_i) for i = 1..N                                 |
|     C_0 = control (no edit)                                       |
|                                                                    |
|  4. Score all candidates with Stage A:                             |
|     (S_i, F_i, B_i) = As(prompt, Compose(I, C_i, R))              |
|                                                                    |
|  5. Compute utilities u_0, u_1, ..., u_N                          |
|                                                                    |
|  6. Pick winner i* = argmax(u_i)                                  |
|                                                                    |
|  7. Compute credits w_1, ..., w_N                                 |
|                                                                    |
|  8. Compute Cost and diversity terms                              |
|                                                                    |
|  9. Update policy: theta <- theta - lr * grad(L_TSPO)             |
|                                                                    |
|  10. (Optional) Log (a_i, u_i, winner) for analysis               |
+-------------------------------------------------------------------+
```

### Hyperparameters

```
+-------------------------------------------------------------------+
|  TSPO Configuration:                                               |
|                                                                    |
|  - N = 5 candidates per tournament                                 |
|  - beta = 0.01 (entropy regularization)                           |
|  - lambda_c = 0.1 (cost penalty)                                  |
|  - lambda_d = 0.05 (diversity bonus)                              |
|  - Learning rate: 1e-4                                            |
|  - Optimizer: Adam                                                 |
|                                                                    |
|  Guard thresholds:                                                 |
|  - delta = 0.05 (margin over control)                             |
|  - tau_F(t) = 0.6 + 0.2 * (1 - t/T)  (stricter late)             |
|                                                                    |
|  Cost weights:                                                     |
|  - d=1: cost=0.1                                                  |
|  - d=3: cost=0.3                                                  |
|  - d=5: cost=0.5                                                  |
|  - d=10: cost=1.0                                                 |
+-------------------------------------------------------------------+
```

---

## Full Stage B Pipeline Diagram

```
+-------------------------------------------------------------------------+
|                           STAGE B OVERVIEW                               |
+-------------------------------------------------------------------------+

                         +------------------+
                         |  Detected Bad    |
                         |  Region (R, m)   |
                         |  from Stage A    |
                         +--------+---------+
                                  |
                                  v
                    +---------------------------+
                    |     TSPO Policy (pi)      |
                    |  "Which knobs to try?"    |
                    +-------------+-------------+
                                  |
                    +-------------+-------------+
                    |      |      |      |      |
                    v      v      v      v      v
                  a_1    a_2    a_3    a_4    a_5     (N=5 actions)
                    |      |      |      |      |
                    v      v      v      v      v
                +-----+ +-----+ +-----+ +-----+ +-----+
                | Ag  | | Ag  | | Ag  | | Ag  | | Ag  |
                +--+--+ +--+--+ +--+--+ +--+--+ +--+--+
                   |      |      |      |      |
                   v      v      v      v      v
                 C_1    C_2    C_3    C_4    C_5     (N candidates)
                   |      |      |      |      |
                   +------+------+------+------+
                                 |
                                 v
                   +-------------+-------------+
                   |            C_0            |     (control)
                   |       (do nothing)        |
                   +---------------------------+
                                 |
                                 v
                   +---------------------------+
                   |     Stage A Scorer (As)   |
                   |   Score each: (S, F, B)   |
                   +-------------+-------------+
                                 |
                                 v
                   +---------------------------+
                   |    Compute utilities      |
                   |    u_i for each candidate |
                   +-------------+-------------+
                                 |
                                 v
                   +---------------------------+
                   |    Guarded Selection      |
                   |    i* = argmax(u_i)       |
                   +-------------+-------------+
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
         +---------------+               +---------------+
         |  u_i* > 0     |               |  u_i* <= 0    |
         |  Winner found |               |  No winner    |
         +-------+-------+               +-------+-------+
                 |                               |
                 v                               v
         +---------------+               +---------------+
         | Short-invert  |               |  Keep control |
         | C_i* and      |               |  C_0 (no edit)|
         | blend into    |               +---------------+
         | z_(t-1)       |
         +-------+-------+
                 |
                 v
         +---------------+
         | Log for TSPO: |
         | (a_i, u_i,    |
         | winner, cost) |
         +-------+-------+
                 |
                 v
         +---------------+
         | Update policy |
         | theta via     |
         | L_TSPO        |
         +---------------+
```

---

## Training Schedule

```
+-------------------------------------------------------------------+
|  Phase 1: Pretrain Ag (B1)                                         |
|    - Epochs: 10-20                                                 |
|    - Data: DETONATE pairs                                          |
|    - Loss: L_G^pre                                                 |
|    - Time: ~10 GPU-hours (A100)                                    |
|                                                                    |
|  Phase 2: Joint Training (B2 + B3)                                 |
|    - Epochs: 20-50                                                 |
|    - Data: Generated during diffusion runs                         |
|    - Loss: L_TSPO (for policy) + optional Ag fine-tune             |
|    - Time: ~30 GPU-hours (A100)                                    |
|                                                                    |
|  Note: Can interleave Ag updates with TSPO updates                 |
+-------------------------------------------------------------------+
```

---

## Success Criteria

```
+-------------------------------------------------------------------+
|  PASS if:                                                          |
|    - Ag reduces patch risk by > 20% on average                     |
|    - Tournament win rate > 50% (candidate beats control)           |
|    - TSPO learns: win rate increases over training                 |
|    - Boundary LPIPS < 0.1 (seamless edits)                         |
|                                                                    |
|  FAIL if:                                                          |
|    - Ag makes things worse (risk increases)                        |
|    - Tournament never finds winners                                |
|    - TSPO collapses to single action                               |
|    - Visible seams in outputs                                      |
|                                                                    |
|  If FAIL, try:                                                     |
|    1. Increase N (more candidates)                                 |
|    2. Lower delta (easier to beat control)                         |
|    3. More diverse action space                                    |
|    4. Longer Ag pretraining                                        |
+-------------------------------------------------------------------+
```

---

## Outputs for Stage C

```
+-------------------------------------------------------------------+
|  After Stage B training, we have:                                  |
|                                                                    |
|  1. Trained Auditor-Inpaint (Ag)                                   |
|     - Can fix regions given mask + prompt + timestep               |
|                                                                    |
|  2. Trained TSPO Policy (pi_theta)                                 |
|     - Proposes good knob settings for tournaments                  |
|     - Learns compute-quality tradeoffs                             |
|                                                                    |
|  3. Tournament Logs                                                |
|     - (action, utility, winner) tuples                             |
|     - Used for calibration in Stage C                              |
|                                                                    |
|  Stage C will:                                                     |
|    - Calibrate scores for reliable decisions                       |
|    - Choose operating points (delta, tau_F)                        |
|    - Deploy the full system                                        |
+-------------------------------------------------------------------+
```

---

## Summary Table

| Component | What it does | Layman Analogy |
|-----------|--------------|----------------|
| **B1: Ag** | Fixes regions given mask + prompt | Photoshop Content-Aware Fill (smart) |
| **B2: Tournament** | Generate N fixes, pick best | American Idol audition |
| **B2: Guards** | Only accept if CLEARLY better | "Innocent until proven guilty" |
| **B3: TSPO** | Learn which settings work | Photographer learning camera settings |
| **Credits** | Reward winning actions | "MVP gets the bonus" |
| **Diversity** | Don't generate identical fixes | "Try different approaches" |
| **Cost penalty** | Prefer cheaper settings | "Don't waste resources" |
