# Stage C: Calibration and Deployment Plan

## Deviation from Proposal

| Proposal | Our Version | Reason |
|----------|-------------|--------|
| Policy thresholds tau_P | Skipped (no NSFW detection) | Immigration-safe |
| Per-class calibration (NSFW, weapon, etc.) | Single faithfulness calibration | No policy classes |

---

## Big Picture

```
Stage A = We trained a JUDGE (Auditor-Scorer)
Stage B = We trained an ARTIST (Ag) + COACH (TSPO)

Stage C = We TUNE THE DECISION RULES
          - Turn raw scores into reliable probabilities
          - Choose threshold knobs (delta, tau_F)
          - Deploy the full system
```

**Layman Analogy**: Calibrating a thermometer

```
Before calibration:
    Thermometer says 70F but actual is 68F
    Thermometer says 90F but actual is 88F

After calibration:
    We learn the mapping: displayed --> actual
    Now we can trust the readings!

Same for our scorer:
    Raw score 0.8 might mean 75% chance of true improvement
    Calibration learns this mapping
```

---

## Why Calibration Matters

```
+-------------------------------------------------------------------+
|  Problem: Raw scores are NOT probabilities                         |
|                                                                    |
|  Example:                                                          |
|    - Scorer says S = 0.8                                           |
|    - Does this mean 80% chance of being good?                      |
|    - No! It's just a raw number from the neural network            |
|                                                                    |
|  Solution: Calibrate scores to match actual win rates              |
|                                                                    |
|  After calibration:                                                |
|    - If calibrated score p_hat = 0.8                               |
|    - Then ~80% of the time, this candidate truly is better         |
+-------------------------------------------------------------------+
```

---

## Score Calibration Methods

### Method 1: Platt Scaling

```
+-------------------------------------------------------------------+
|  Idea: Fit a sigmoid to map raw scores to probabilities            |
|                                                                    |
|  Formula:                                                          |
|      p_hat = sigmoid((S - b) / T)                                  |
|            = 1 / (1 + exp(-(S - b) / T))                           |
|                                                                    |
|  Parameters:                                                       |
|      b = bias (shifts the curve left/right)                        |
|      T = temperature (controls steepness)                          |
|                                                                    |
|  Example:                                                          |
|      Raw S = 0.7, b = 0.5, T = 0.2                                 |
|      p_hat = sigmoid((0.7 - 0.5) / 0.2) = sigmoid(1.0) = 0.73     |
+-------------------------------------------------------------------+

Visual:
                    p_hat
                     1.0 |              ___________
                         |           __/
                         |        __/
                     0.5 |      _/
                         |    _/
                         | __/
                     0.0 |/___________________________
                         0.0          b          1.0   S
                                      ^
                                      |
                               Inflection point
```

### Method 2: Isotonic Regression

```
+-------------------------------------------------------------------+
|  Idea: Non-parametric, learns any monotonic mapping                |
|                                                                    |
|  How it works:                                                     |
|    1. Sort all (score, actual_win) pairs by score                  |
|    2. Fit a monotonically increasing step function                 |
|    3. At each score level, estimate win probability                |
|                                                                    |
|  Advantage: More flexible than Platt                               |
|  Disadvantage: Needs more data, can overfit                        |
+-------------------------------------------------------------------+

Visual:
                    p_hat
                     1.0 |                    ____
                         |               ____|
                         |           ___|
                     0.5 |      ____|
                         |  ___|
                         |__|
                     0.0 |____________________________
                         0.0                     1.0   S

                         (Step function that only goes up)
```

### Calibration Data

```
+-------------------------------------------------------------------+
|  Source: Held-out tournament results from Stage B                  |
|                                                                    |
|  For each tournament, we have:                                     |
|    - Scores: S_0 (control), S_1, ..., S_N (candidates)            |
|    - Winner: which candidate was actually accepted                 |
|    - Ground truth: was the winner truly better? (human or VQA)     |
|                                                                    |
|  Create calibration pairs:                                         |
|    (S_candidate - S_control, did_it_actually_win?)                 |
|                                                                    |
|  Example:                                                          |
|    Tournament 1: S_ctrl=0.5, S_cand=0.75, winner=cand, true_win=1  |
|    Tournament 2: S_ctrl=0.6, S_cand=0.65, winner=cand, true_win=0  |
|                                                                    |
|  --> Calibrate on these (score_gap, actual_outcome) pairs          |
+-------------------------------------------------------------------+
```

### Calibration Recipe

```
+-------------------------------------------------------------------+
|  Step 1: Collect held-out tournament data                          |
|    - Run ~1000 tournaments on validation set                       |
|    - Record (S_cand, S_ctrl, winner, ground_truth_quality)        |
|                                                                    |
|  Step 2: Fit calibration mapping                                   |
|    - Option A: Platt scaling (fit b, T)                           |
|    - Option B: Isotonic regression                                 |
|                                                                    |
|  Step 3: Validate calibration                                      |
|    - Compute ECE (Expected Calibration Error)                      |
|    - Target: ECE < 0.05                                            |
|                                                                    |
|  Step 4: Store calibration parameters                              |
|    - Save (b, T) or isotonic lookup table                          |
+-------------------------------------------------------------------+
```

---

## Choosing Operating Points (Interpretable Knobs)

### The Three Knobs

```
+-------------------------------------------------------------------+
|  1. Margin delta: "How much better must candidate be?"             |
|                                                                    |
|     u_i = (S_i - S_0 - delta)_+ * ...                             |
|                        ^^^^^                                       |
|                        this knob                                   |
|                                                                    |
|     delta = 0.0: Accept any improvement (aggressive)               |
|     delta = 0.1: Need 10% better to accept (conservative)          |
|     delta = 0.2: Need 20% better (very conservative)               |
|                                                                    |
|  Choose delta such that:                                           |
|     Pr[true win | S_cand >= S_ctrl + delta] >= 0.9                |
|     "90% of accepted edits should be real improvements"            |
+-------------------------------------------------------------------+

+-------------------------------------------------------------------+
|  2. Faithfulness threshold tau_F(t): "Minimum quality required"    |
|                                                                    |
|     1[F_i >= tau_F(t)]                                             |
|              ^^^^^^^                                               |
|              this knob                                             |
|                                                                    |
|     tau_F varies by timestep:                                      |
|       - Early (t high): tau_F = 0.6 (lenient, can fix later)      |
|       - Middle: tau_F = 0.8 (stricter)                            |
|       - Late (t low): tau_F = 0.7 (slightly relaxed to avoid       |
|                                    over-sanitization)              |
|                                                                    |
|     From proposal: stricter mid-trajectory, relaxed at end         |
+-------------------------------------------------------------------+

+-------------------------------------------------------------------+
|  3. Seam threshold: "Maximum allowed boundary artifact"            |
|                                                                    |
|     B_i = exp(-kappa * LPIPS_boundary)                            |
|                                                                    |
|     kappa controls how much we penalize seams                      |
|     Also set max allowed LPIPS_boundary (e.g., 0.15)              |
|                                                                    |
|     If LPIPS_boundary > threshold: reject edit regardless          |
+-------------------------------------------------------------------+
```

### Threshold Schedule Across Timesteps

```
+-------------------------------------------------------------------+
|  Timestep-aware thresholds (T total steps):                        |
|                                                                    |
|  t/T = 1.0 (start, pure noise)                                     |
|    - tau_F = 0.5 (lenient, everything is fuzzy anyway)            |
|    - delta = 0.03 (small margin needed)                           |
|                                                                    |
|  t/T = 0.5 (middle, shapes forming)                                |
|    - tau_F = 0.8 (strict, this is when structure is set)          |
|    - delta = 0.05 (medium margin)                                 |
|                                                                    |
|  t/T = 0.1 (near end, fine details)                                |
|    - tau_F = 0.7 (slightly relaxed to allow finishing touches)    |
|    - delta = 0.08 (higher margin, be conservative)                |
+-------------------------------------------------------------------+

Visual:
    tau_F
    0.9 |       ____
        |      /    \
    0.7 |     /      \____
        |    /
    0.5 |___/
        |________________________________
        t=T            t=T/2           t=0
        (start)        (middle)        (end)
```

### Operating Point Selection Recipe

```
+-------------------------------------------------------------------+
|  For margin delta:                                                 |
|                                                                    |
|  1. Plot precision-recall curve on calibration data                |
|     - x-axis: candidate score - control score                      |
|     - y-axis: precision (true positive rate at this threshold)     |
|                                                                    |
|  2. Find delta where precision >= 0.9                              |
|     "At least 90% of accepted edits are true improvements"         |
|                                                                    |
|  3. Check recall is acceptable (> 0.5)                             |
|     "We're not rejecting too many good edits"                      |
+-------------------------------------------------------------------+

+-------------------------------------------------------------------+
|  For tau_F per timestep:                                           |
|                                                                    |
|  1. Bucket calibration data by timestep ranges                     |
|                                                                    |
|  2. For each bucket, find tau_F where:                             |
|     - False negative rate < 0.3 (don't miss too many good edits)  |
|     - False positive rate < 0.1 (don't accept bad edits)          |
|                                                                    |
|  3. Smooth across timesteps to avoid jumpy behavior                |
+-------------------------------------------------------------------+
```

---

## Deployed Decision Rule

### Per-Patch, Per-Step Algorithm

```
+-------------------------------------------------------------------+
|  At each audited timestep t, for each region (R, m):               |
|                                                                    |
|  1. Sample N candidates via policy pi_theta:                       |
|     a_1, ..., a_N ~ pi_theta(. | state)                           |
|     C_1, ..., C_N = Ag(state; a_i)                                |
|     C_0 = control (no edit)                                        |
|                                                                    |
|  2. Score with CALIBRATED As:                                      |
|     (S_i, F_i, B_i) = As(prompt, Compose(I, C_i, R))              |
|     p_hat_i = Calibrate(S_i)  <-- apply Platt/isotonic            |
|                                                                    |
|  3. Check guards and compute utility:                              |
|     For each candidate i:                                          |
|       pass_margin = (p_hat_i - p_hat_0) >= delta                  |
|       pass_faithful = F_i >= tau_F(t)                             |
|       pass_seam = LPIPS_boundary(C_i) < seam_threshold            |
|                                                                    |
|       if pass_margin AND pass_faithful AND pass_seam:             |
|         u_i = (p_hat_i - p_hat_0 - delta) * B_i                   |
|       else:                                                        |
|         u_i = 0                                                    |
|                                                                    |
|  4. Select:                                                        |
|     i* = argmax(u_i)                                              |
|     if u_i* > 0:                                                  |
|       ACCEPT C_i*                                                 |
|       Short-invert and blend into z_(t-1)                         |
|     else:                                                          |
|       KEEP C_0 (control, no edit)                                 |
|                                                                    |
|  5. Log tournament for potential recalibration                     |
+-------------------------------------------------------------------+
```

### Decision Flow Diagram

```
                    +-------------------+
                    | For each region   |
                    | at timestep t     |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Sample N actions  |
                    | from TSPO policy  |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Generate N cands  |
                    | + control C_0     |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Score with As     |
                    | (calibrated)      |
                    +---------+---------+
                              |
                              v
              +---------------+---------------+
              |                               |
              v                               v
    +-----------------+             +-----------------+
    | Check guards    |             | Check guards    |
    | for each cand   |             | for control     |
    +-----------------+             +-----------------+
              |                               |
              v                               v
    +-----------------+             +-----------------+
    | Compute u_i for |             | u_0 = 0         |
    | passing cands   |             | (by definition) |
    +---------+-------+             +-----------------+
              |
              v
    +-------------------+
    | i* = argmax(u_i)  |
    +---------+---------+
              |
              v
    +-------------------+
    | u_i* > 0 ?        |
    +---------+---------+
              |
      +-------+-------+
      |               |
      v               v
  +-------+       +-------+
  |  YES  |       |  NO   |
  +---+---+       +---+---+
      |               |
      v               v
  +----------+    +----------+
  | ACCEPT   |    | KEEP     |
  | C_i*     |    | CONTROL  |
  | Blend in |    | (no edit)|
  +----------+    +----------+
```

---

## Coupling to the Diffusion Scheduler (Seam-Free Edits)

### Which Timesteps to Audit?

```
+-------------------------------------------------------------------+
|  Not every timestep needs auditing (too expensive!)                |
|                                                                    |
|  T_audit = subset of timesteps to check                            |
|                                                                    |
|  Strategy from proposal:                                           |
|    - Audit every 3rd step (or every 5th)                          |
|    - Coarse scales early (when structure is forming)              |
|    - Fine scales late (when details are being added)              |
|                                                                    |
|  Example for T=50 steps:                                           |
|    T_audit = {50, 47, 44, 41, ..., 8, 5, 2}                        |
|    (every 3rd step)                                                |
+-------------------------------------------------------------------+
```

### Short DDIM Inversion

```
+-------------------------------------------------------------------+
|  When Ag edits in RGB, we need to put it back into latent space    |
|                                                                    |
|  DDIM Inversion:                                                   |
|    - Take edited RGB region                                        |
|    - Run diffusion BACKWARDS for d steps                           |
|    - Get latent z_edit that would produce this RGB                 |
|                                                                    |
|  "Short" = only d steps (not full T):                              |
|    - d=1: very fast, but might not blend well                      |
|    - d=5: good balance                                             |
|    - d=10: best blending, but slower                               |
|                                                                    |
|  The value of d is one of the TSPO knobs (learned!)                |
+-------------------------------------------------------------------+
```

### Latent Blending with Feathered Masks

```
+-------------------------------------------------------------------+
|  After inversion, blend edited latent with control:                |
|                                                                    |
|  z_(t-1) = (1 - alpha * m_feather) * z_ctrl                       |
|          + (alpha * m_feather) * z_edit                            |
|                                                                    |
|  Where:                                                            |
|    m_feather = feathered (soft-edge) version of mask              |
|    alpha = alpha(t) = blending strength (varies by timestep)       |
|                                                                    |
|  Feathering:                                                       |
|    Hard mask:      Feathered mask:                                 |
|    [1 1 1 1]       [0.2 0.5 0.5 0.2]                              |
|    [1 1 1 1]  -->  [0.5 1.0 1.0 0.5]                              |
|    [1 1 1 1]       [0.5 1.0 1.0 0.5]                              |
|    [1 1 1 1]       [0.2 0.5 0.5 0.2]                              |
|                                                                    |
|  Soft edges = seamless blending!                                   |
+-------------------------------------------------------------------+
```

### Alpha Schedule

```
+-------------------------------------------------------------------+
|  alpha(t) controls how much of the edit to use:                    |
|                                                                    |
|  alpha = 1.0: fully use the edit                                   |
|  alpha = 0.5: 50/50 blend                                          |
|  alpha = 0.0: ignore the edit (use control)                        |
|                                                                    |
|  Typical schedule:                                                 |
|    - Early timesteps: alpha = 0.8 (mostly edit, structures change)|
|    - Late timesteps: alpha = 0.6 (more conservative, keep details)|
|                                                                    |
|  Why lower alpha late?                                             |
|    - Late edits should be subtle                                   |
|    - Don't want to destroy fine details from earlier              |
+-------------------------------------------------------------------+
```

---

## Complexity and Compute-Quality Tradeoff

### Cost Breakdown

```
+-------------------------------------------------------------------+
|  For each audited timestep, cost scales as:                        |
|                                                                    |
|  O(K_t * (N * C_Ag + C_As + d * C_DDIM_inv))                      |
|                                                                    |
|  Where:                                                            |
|    K_t = number of regions to audit at timestep t                  |
|    N = number of candidates (typically 5)                          |
|    C_Ag = cost of one inpainting with Ag                          |
|    C_As = cost of scoring with As                                 |
|    d = DDIM inversion steps                                        |
|    C_DDIM_inv = cost of one inversion step                        |
+-------------------------------------------------------------------+
```

### Optimizations

```
+-------------------------------------------------------------------+
|  1. Batch inpainting and scoring                                   |
|     - Generate all N candidates in one forward pass                |
|     - Score all candidates in one batch                            |
|                                                                    |
|  2. Cache text/image features                                      |
|     - CLIP text embedding: compute once per prompt                 |
|     - Reuse across all timesteps and regions                       |
|                                                                    |
|  3. TSPO learns to reduce cost                                     |
|     - Cost penalty in L_TSPO                                       |
|     - Policy learns to favor cheap actions (low d)                |
|     - Only uses expensive actions when needed                      |
|                                                                    |
|  4. Early stopping                                                 |
|     - If control is already very good, skip tournament            |
|     - If no risky regions detected, skip audit                    |
+-------------------------------------------------------------------+
```

### Quality-Latency Pareto Frontier

```
+-------------------------------------------------------------------+
|  TSPO learns to balance quality vs speed:                          |
|                                                                    |
|  Quality                                                           |
|     ^                                                              |
|     |        * (high quality, high cost)                          |
|     |      *                                                       |
|     |    *    <-- Pareto frontier                                  |
|     |   *                                                          |
|     |  *                                                           |
|     | *   (low quality, low cost)                                  |
|     +---------------------------------> Latency                    |
|                                                                    |
|  TSPO pushes towards the Pareto frontier:                          |
|    - For easy regions: use cheap settings                          |
|    - For hard regions: use expensive settings                      |
+-------------------------------------------------------------------+
```

---

## Full Algorithm (Annotated)

```
+-------------------------------------------------------------------+
|  TiPAI-TSPO Inference Algorithm                                    |
+-------------------------------------------------------------------+

Input: prompt p, base diffuser Phi_base, T steps
Output: final image I_0

Initialize: z_T ~ N(0, I)  # pure noise

for t = T, T-1, ..., 1:

    # 1. CONTROL STEP
    z_ctrl_(t-1) = Phi_base(z_t, p, t)
    I_(t-1) = Decode(z_ctrl_(t-1))  # for auditing

    # 2. CHECK IF AUDIT STEP
    if t not in T_audit:
        z_(t-1) = z_ctrl_(t-1)
        continue

    # 3. MINE REGIONS
    R_t = MineRegions(I_(t-1), p)  # using Stage A risk map
    # R_t = {(R_k, m_k)} for k = 1..K_t

    # 4. FOR EACH REGION
    for (R, m) in R_t:

        # 4a. GENERATE CANDIDATES
        state = (p, z_(t-1), I_(t-1), m, t)
        actions = [Sample(pi_theta, state) for _ in range(N)]
        candidates = [Ag(state, a) for a in actions]
        C_0 = Crop(I_(t-1), R)  # control
        candidates = [C_0] + candidates  # include control

        # 4b. SCORE (batched)
        scores = []
        for i, C in enumerate(candidates):
            I_composed = Paste(I_(t-1), C, R)
            (S, F, B) = As(p, I_composed)
            p_hat = Calibrate(S)
            scores.append((p_hat, F, B))

        # 4c. SELECT
        utilities = []
        for i in range(1, N+1):  # skip control
            p_hat_i, F_i, B_i = scores[i]
            p_hat_0, _, _ = scores[0]  # control

            pass_all = (p_hat_i - p_hat_0 >= delta) and
                       (F_i >= tau_F(t)) and
                       (B_i >= seam_threshold)

            if pass_all:
                u_i = (p_hat_i - p_hat_0 - delta) * B_i
            else:
                u_i = 0
            utilities.append(u_i)

        i_star = argmax(utilities)

        if utilities[i_star] > 0:
            # ACCEPT: short-invert and blend
            C_star = candidates[i_star + 1]
            z_edit = ShortDDIMInvert(C_star, d=actions[i_star].d)
            m_feather = Feather(m)
            alpha = alpha_schedule(t)
            z_(t-1) = (1 - alpha*m_feather) * z_ctrl_(t-1)
                    + (alpha*m_feather) * z_edit
        # else: keep z_ctrl_(t-1) (already set)

        # 4d. LOG
        Log(actions, utilities, winner=i_star)

    # 5. TSPO UPDATE (optional, online)
    if enough_logs_accumulated:
        Update(pi_theta, logs)  # minimize L_TSPO

# Final decode
I_0 = Decode(z_0)
return I_0
```

---

## Calibration Maintenance

```
+-------------------------------------------------------------------+
|  Calibration can drift over time (data distribution changes)       |
|                                                                    |
|  Periodic recalibration:                                           |
|    1. Collect new tournament logs (every 1000 images)              |
|    2. Check ECE on recent data                                     |
|    3. If ECE > 0.1, re-fit calibration                            |
|    4. Update (b, T) or isotonic table                              |
|                                                                    |
|  Continual learning:                                               |
|    - TSPO updates online from tournament logs                      |
|    - Judge (As) can be fine-tuned periodically                    |
+-------------------------------------------------------------------+
```

---

## Success Criteria

```
+-------------------------------------------------------------------+
|  PASS if:                                                          |
|    - ECE < 0.05 (well-calibrated scores)                          |
|    - Precision at delta >= 0.9                                     |
|    - Recall at delta >= 0.5                                        |
|    - No visible seams in output images                             |
|    - Latency overhead < 2x base model                              |
|                                                                    |
|  FAIL if:                                                          |
|    - ECE > 0.15 (badly calibrated)                                |
|    - Too conservative (recall < 0.3)                               |
|    - Too aggressive (precision < 0.7)                              |
|    - Visible artifacts                                             |
|                                                                    |
|  If FAIL, try:                                                     |
|    1. Collect more calibration data                                |
|    2. Use isotonic instead of Platt (more flexible)               |
|    3. Adjust delta/tau_F manually based on visual inspection      |
|    4. Reduce N or audit frequency for latency                     |
+-------------------------------------------------------------------+
```

---

## Summary: The Complete TiPAI-TSPO System

```
+-------------------------------------------------------------------+
|                    COMPLETE SYSTEM OVERVIEW                        |
+-------------------------------------------------------------------+

+-------------+     +-------------+     +-------------+
|   Stage A   |     |   Stage B   |     |   Stage C   |
|   Auditor-  | --> |   Auditor-  | --> | Calibration |
|   Scorer    |     |   Inpaint   |     |   & Deploy  |
|   (As)      |     |   (Ag+TSPO) |     |             |
+-------------+     +-------------+     +-------------+
      |                   |                   |
      v                   v                   v
  "Judge"            "Artist +            "Tuned
  images              Coach"              Decisions"


At inference time:

    Prompt + Noise
         |
         v
    +----------+
    | Diffuse  | <-------------------+
    | one step |                     |
    +----+-----+                     |
         |                           |
         v                           |
    +----------+                     |
    | Decode   |                     |
    | to RGB   |                     |
    +----+-----+                     |
         |                           |
         v                           |
    +----------+     +----------+    |
    | Mine     |---->| Stage A  |    |
    | Regions  |     | Score    |    |
    +----+-----+     +----------+    |
         |                |          |
         v                v          |
    +----------+     +----------+    |
    | Generate |     | Guard &  |    |
    | N cands  |<--->| Select   |    |
    | (Ag+TSPO)|     | (Calib)  |    |
    +----------+     +----+-----+    |
                          |          |
                     +----+----+     |
                     |         |     |
                     v         v     |
                 [Accept]  [Reject]  |
                     |         |     |
                     v         v     |
                 +------+  +------+  |
                 |Blend |  | Keep |  |
                 |Edit  |  |Ctrl  |  |
                 +--+---+  +--+---+  |
                    |         |      |
                    +----+----+      |
                         |           |
                         v           |
                    z_(t-1) --------+
                         |
                    (loop until t=0)
                         |
                         v
                    Final Image
```

---

## File Dependencies

```
+-------------------------------------------------------------------+
|  Stage C depends on:                                               |
|                                                                    |
|  From Stage A:                                                     |
|    - Trained Auditor-Scorer (As) weights                          |
|    - Risk heatmap function                                         |
|                                                                    |
|  From Stage B:                                                     |
|    - Trained Auditor-Inpaint (Ag) weights                         |
|    - Trained TSPO policy (pi_theta) weights                       |
|    - Tournament logs for calibration data                         |
|                                                                    |
|  Stage C produces:                                                 |
|    - Calibration parameters (b, T) or isotonic table              |
|    - Operating points (delta, tau_F schedule, seam threshold)     |
|    - Complete inference script                                     |
+-------------------------------------------------------------------+
```
