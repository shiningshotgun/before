# Remnant Decision-Trace & Return Composition Analysis

This document details the decision-trace investigation conducted on the Remnant prototype to determine whether counterfactual dream transformations alter internal agent mechanics (actions, Q-learning updates, per-action memory lookups, and policy blending) when terminal cumulative return contrasts are near zero.

---

## 1. Passive-Listener Architecture & Non-Interference Equivalence Gate

### The Instrumentation Challenge
Investigating internal agent decisions during execution risks altering the execution trajectory itself. Specifically:
- `QAgent.act()` queries `memory.value(s, a)` for all candidate actions $a \in \{0, 1, 2, 3\}$ when choosing greedily.
- When episodic memory exceeds `lookup_sample=400`, each `memory.value()` call invokes `self._rng.sample(...)`, which advances `memory._rng`.
- On exploratory random actions (`rng.random() < self.eps`), the policy selects a random action without evaluating candidate values.
- If an active logging wrapper evaluates candidate values during an exploratory step, `memory._rng` advances out-of-schedule, immediately desynchronizing the memory sampler and corrupting subsequent trajectory paths.

### Passive Observer Solution
The architecture implements strictly passive listener callbacks:
1. `remnant/memory.py` attaches an optional `trace_listener(state, action, value, pool_indices)` invoked only inside existing `value()` calls.
2. `remnant/agent.py` attaches `decision_listener` and `learn_listener`. On exploratory steps (`is_random=True`), the listener receives `None` for candidate values without calling `memory.value()`.
3. Calls to `getstate()` and `append()` are read-only/append-only and consume zero RNG draws.

### The Equivalence Gate (`tests/test_equivalence_gate.py`)
To prove non-interference honestly, `test_passive_logging_deterministic_equivalence` asserts 100% full-state deterministic equality between running with logging enabled vs. disabled on Seed 42:
- Exact identity of full per-phase action sequences (`live_actions`, `dream_actions`, `return_actions`).
- Exact identity of terminal Q-tables (`final_q`).
- Exact identity of episodic memory contents (`final_memory_entries`).
- Exact identity of memory internal RNG state (`final_memory_rng_state`).
- Exact identity of terminal states for every RNG stream (`live_rng`, `transform_rng`, `dream_act_rng`, `dream_env_rng`, `return_act_rng`).
- Exact identity of summary returns, coverage, and contrasts.

All assertions pass unconditionally.

---

## 2. Four Descriptive Measures Across Valid Seeds

The full trace dataset is recorded in `checkpoints/stage4_runs_v4.json` across all five prespecified seeds. Seeds 13 and 23 are excluded under prespecified Validity Rule 1 (realized dream equals live regime at suspension). Below are the verified metrics across the three valid seeds (Seeds 42, 77, 99):

| Metric | Seed 42 | Seed 77 | Seed 99 |
| :--- | :---: | :---: | :---: |
| **Realized Dream vs Live Regime** | Objects 0 & 1 differ | Object 1 differs | Objects 0 & 3 differ |
| **First Reward Divergence in Dream** | **Step 1** (collected Obj 0: $+1.0$ vs $-1.0$) | **None** (Obj 1 never collected in dream) | **Step 40** (collected Obj 0: $+1.0$ vs $-1.0$) |
| **Dream Action Agreement (Pre-diff)** | 100% (1/1 steps) | 100% (60/60 steps) | 100% (40/40 steps) |
| **Dream Action Agreement (Post-diff)** | **96.6%** (57/59 steps) | N/A (0 post-diff steps) | **100%** (20/20 steps) |
| **Dream Total Updates $\sum \|\Delta Q\|$ (Dream / Untrans)** | **0.7767** / **0.1728** | **0.5662** / **0.5662** | **0.5010** / **0.1975** |
| **Post-Dream Q-Table $L_1$ Distance** | **0.6154** | **0.0000** | **0.3118** |
| **First Return Action Divergence (Dream vs Untrans)** | Fischer: **Step 71**<br>Mal: **Step 94** | Fischer: **None**<br>Mal: **None** | Fischer: **Step 44**<br>Mal: **Step 41** |
| **First Return Action Divergence (Dream vs No-Dream)** | **Step 2** | **Step 5** | **Step 3** |
| **Return State Alignment (Dream vs Untrans)** | 173–184 / 200 identical steps (97.7%–98.9% match at identical states) | 200 / 200 identical steps (100% action match) | 98–113 / 200 identical steps (95.9%–97.3% match at identical states) |
| **Return Memory Lookups with Dream Entries** | **73.0%** (1060 / 1452 lookups) | **73.2%** (1054 / 1440 lookups) | **72.7%** (1062 / 1460 lookups) |
| **Greedy Steps Where Memory Shifted Argmax** | 50.0% – 51.8% | 51.6% – 52.2% | 46.4% – 48.8% |
| **Argmax Shifts on Pools Containing Dream Entries** | **100%** (0 shifts on pools without dream entries) | **100%** (0 shifts on pools without dream entries) | **100%** (0 shifts on pools without dream entries) |
| **Argmax Agreement Dream vs Untrans in Return** | 90.2% – 93.3% | 100% | 69.3% – 72.3% |
| **Cumulative Return Contrasts (Dream - Untrans)** | Fischer: **+0.00**<br>Mal: **+0.00** | Fischer: **+0.00**<br>Mal: **+0.00** | Fischer: **-0.50**<br>Mal: **+0.50** |

---

## 3. Return-Phase Reward Composition Findings

Executing `return_composition_analysis.py` decomposes the 200-step return phase into step costs ($-0.01$ per move) and discrete item collection events:

### Seed 42: Exact Reward Invariance Despite Action Divergence
- In both Fischer and Mal conditions, the step-by-step reward sequence of `dream` and `dream_untransformed` was **100% identical across all 200 return steps** (`reward-differing steps: 0`).
- Both arms executed exactly 12 item collections at identical time steps:
  - Step 1: Obj 1
  - Step 5: Obj 2
  - Step 47: Obj 3
  - Step 60: Obj 1
  - Step 66: Obj 2
  - Step 120: Obj 3
  - Step 126: Obj 1
  - Step 130: Obj 2
  - Step 164: Obj 0
  - Step 176: Obj 3
  - Step 180: Obj 1
  - Step 187: Obj 2
- Total step cost was identical ($-1.880$).
- *Observation:* Although the agent's actions diverged at step 71 (Fischer) and step 94 (Mal), these differences involved exploratory local loops between collection milestones that did not alter the timing or sequence of object collections. This equal return outcome is recorded as an empirical observation; no specific mechanism for the equality has been experimentally isolated.

### Seed 77: Complete Trajectory Identity (Mathematical Null)
- In Seed 77, the realized dream differed from the live regime only on Object 1.
- During the 60 dream steps, the agent visited and collected Objects 0, 2, and 3, but never encountered Object 1.
- Because no altered meaning was experienced, Q-learning updates were identical ($\sum |\Delta Q| = 0.5662$), post-dream Q-tables were identical ($L_1 = 0.0000$), and the return trajectory was 100% identical across all 200 steps.

### Seed 99: Meaning Shift Propagation
In Seed 99, Object 0 was collected during the dream at step 40, inducing post-dream Q-divergence ($L_1 = 0.3118$). In the return phase:
- **Fischer Target (Return difference: $-0.50$):**
  - Transformed dream collected Object 3 twice (steps 52, 132), but did not collect Object 0.
  - Untransformed dream collected Object 0 at step 91 ($+0.99$ net) and Object 3 once (step 157).
  - Net collection reward was $+0.50$ higher in untransformed dream ($5.50$ vs $5.00$).
- **Mal Target (Return difference: $+0.50$):**
  - Transformed dream collected Object 3 three times (steps 52, 81, 132).
  - Untransformed dream collected Object 3 twice (steps 91, 157).
  - Net collection reward was $+0.50$ higher in transformed dream ($7.50$ vs $7.00$).

---

## 4. Methodological Scope & Single-Draw Limits

1. **Association vs. Causation in Memory Shifts:**
   - On greedy evaluation steps, memory shifted the chosen action ($\text{argmax}(Q_{\text{blended}}) \ne \text{argmax}(Q_{\text{raw}})$) on ~50% of steps.
   - In 100% of these cases, the sampled lookup pool happened to contain $\ge 1$ dream-phase entry.
   - However, because dream entries were present in ~73% of all lookup pools by base rate, and no counterfactual ablation without dream entries has been run, this pattern is reported strictly as an observed association without causal attribution.
2. **Single Paired Draws:**
   - All results reflect single paired draws per seed under fixed tabular hyperparameters. They do not constitute general statistical distributions and must not be extrapolated beyond this specific gridworld protocol.
