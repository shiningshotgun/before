# before - Empirical Baseline & Verification Report

Note: This report's runs predate the project rename from Remnant to `before`, so historical execution paths (`~/Documents/remnant`) and naming are retained as recorded.

This document records the measured baseline behavior, pass criteria rationale, and empirical results derived on this machine for Remnant (including paired dream-arm RNG streams, dream rollout coverage logging, Mal target live-regime guards, scoped test collection, and decision-trace analysis). All runs were executed strictly using `.venv` Python 3.11 inside `~/Documents/remnant`, with gates enforcing assertions with nonzero exits on failure.

---

## 1. Stage 1 & Stage 2 Measured Baseline

### Prespecified Criteria & Rationale
- **Stage 1 Evaluation Return Threshold (> 1.0):**
  In the default 5x5 gridworld layout, positive rewards are placed at objects 0 (+1.0), 2 (+0.5), and 3 (+0.5), while object 1 carries a penalty (-1.0). Total optimal collection yields +2.0 gross reward minus ~0.12 step costs (12 steps). A random policy achieves negative cumulative returns (~ -0.6). A proficient tabular Q-agent collecting the positive items achieves an average return > 1.0. The baseline evaluation is greedy (`eps = 0.0`) over 50 evaluation episodes and **strictly does not mutate episodic memory** (`record_memory=False`).
- **Stage 2 World Model Accuracy & Coverage (> 0.80):**
  Chronological 90/10 split over 18,000 logged transitions. The model predicts both next state ($s'$) and reward ($r$). Evaluated against held-out transitions with dropped unseen transitions tracked explicitly. Threshold requires `next_state_accuracy >= 0.80`, `reward_accuracy >= 0.80`, and `coverage >= 0.80`.

### Measured Results (`run_stage12.py`)
```json
{
  "seed": 7,
  "resumed_from_checkpoint": true,
  "train_episodes": 300,
  "transitions_logged": 18000,
  "stage1_avg_eval_return": 1.4,
  "stage1_threshold": 1.0,
  "stage1_pass": true,
  "stage2_heldout_next_state_accuracy": 1.0,
  "stage2_heldout_reward_accuracy": 1.0,
  "stage2_heldout_joint_accuracy": 1.0,
  "stage2_heldout_seen": 1795,
  "stage2_heldout_dropped_unseen": 5,
  "stage2_heldout_coverage": 0.997,
  "stage2_accuracy_threshold": 0.8,
  "stage2_coverage_threshold": 0.8,
  "stage2_pass": true,
  "wall_seconds": 2.21,
  "max_rss_mb": 67.05
}
```

---

## 2. Stage 3 Dream Machinery

The dream environment runs an episode backed entirely by the learned transition and reward tables of the `WorldModel`. It tracks transition coverage and known/unknown state-action pairs, failing if any rule violation occurs or if unknown transitions are encountered.

### Measured Results (`run_stage3.py`)
```json
{
  "seed": 11,
  "injected_start": "pos=(0, 0), col=[]",
  "dream_steps": 60,
  "dream_total_reward": 1.4,
  "known_steps": 60,
  "unknown_steps": 0,
  "coverage": 1.0,
  "rule_violations": 0,
  "logged_same_format": true,
  "memory_entries_total": 18060,
  "stage3_pass": true
}
```

---

## 3. Stage 4 Full Run Protocol & Results (Paired-Stream Protocol)

### Prespecified Protocols
1. **Condition Integrity:**
   - The learned dream is sampled **FIRST** via per-object recombination from the agent's experienced regimes (`recombine=True`).
   - The experimenter constructs the outer target relative to the realized dream:
     - **Fischer arm:** Outer target $:=$ realized dream meanings.
     - **Mal arm:** Outer target $:=$ controlled mismatch constructed by swapping two uncollected objects that possess differing values in the realized dream.
   - **Disclosed Match/Mismatch Relation:** The match/mismatch relation is experimenter-constructed around the realized dream; the dream does not forecast the future.
2. **Three Prespecified Validity Rules:**
   - **Rule 1:** Realized dream meanings must differ from the live regime at suspension. If the realized dream equals the live regime, the run is excluded under this prespecified rule and reported as excluded (never resampled).
   - **Rule 2:** The Mal mismatch must differ on at least one reachable, uncollected object in the suspended state (quantified in report).
   - **Rule 3:** The Mal outer target must differ from the live regime at suspension. If `mal_target == live_regime`, the run is excluded under Rule 3.
3. **Mid-Episode Suspension & Resume:**
   - Structural suspension at step $t=5$ of episode 300, guaranteeing $\ge 2$ uncollected objects by horizon layout.
   - Return phase resumes from the exact suspended state (preserving position and inventory). Episode boundaries during the 200 return steps are tracked for viewer segmentation.
4. **Three Paired Arms & Within-Target Contrasts:**
   - **`dream`:** Transformed dream episode inside the learned model with realized dream meanings.
   - **`dream_untransformed`:** Faithful-replay dream episode inside the learned model with the live regime at suspension.
   - **`no_dream`:** Paired control skipping dream execution entirely (0 transitions, 0 memory entries added, 0 Q-table updates).
   - **Contrasts Computed Within Target:**
     - Overall Dream Effect: $\text{Dream} - \text{No-Dream}$
     - Untransformed Dream Baseline Effect: $\text{Dream Untransformed} - \text{No-Dream}$
     - Transformation Attribution: $\text{Dream} - \text{Dream Untransformed}$
5. **Rollout Coverage Tracking:**
   - For every dream rollout, `known_steps` ($(s, a)$ in model counts), `unknown_steps`, and `coverage` ($\text{known} / \text{total}$) are explicitly logged. Across all valid runs in v3, coverage was 100% (60/60 known steps).
6. **Paired RNG Stream Architecture:**
   - `live_rng`: `seed * 1000 + 1` (generates identical pre-suspension history for all arms).
   - `transform_rng`: `seed * 1000 + 4` (samples realized dream).
   - **Dream Pair Randomness:** Both `dream` and `dream_untransformed` instantiate their own distinct `random.Random` objects initialized from **identical** seeds:
     - `dream_act_rng`: `seed * 1000 + 2` (agent action randomness during dream).
     - `dream_env_rng`: `seed * 1000 + 5` (model successor transition sampling randomness during dream).
   - `return_act_rng`: `seed * 1000 + 3` (identical initial state across all three arms during return).

### Measured Results across Prespecified Seeds (`[13, 23, 42, 77, 99]`)

Results recorded cleanly in `checkpoints/stage4_runs_v3.json` (`stage4_runs.json` and `stage4_runs_v2.json` preserved intact).

#### Excluded Seeds
- **Seed 13:** Excluded under Rule 1 (realized dream `{0: -1.0, 1: 1.0, 2: 0.5, 3: -0.5}` equals live regime at suspension).
- **Seed 23:** Excluded under Rule 1 (realized dream `{0: -1.0, 1: 1.0, 2: 0.5, 3: -0.5}` equals live regime at suspension).

#### Valid Seeds Summary Table (Single Paired Draws)

| Seed | Target Condition | Outer Target Meanings | Arm | Return | Dream Coverage (known/total) | Contrasts |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | **Fischer** | `{0: 1.0, 1: -1.0, 2: 0.5, 3: 0.5}` | `dream`<br>`dream_untrans`<br>`no_dream` | **-1.50**<br>**-1.50**<br>**-2.00** | 1.000 (60/60)<br>1.000 (60/60)<br>N/A | Dream vs NoDream: **+0.50**<br>Untrans vs NoDream: **+0.50**<br>Transformation Effect: **+0.00** |
| | **Mal** | `{0: -1.0, 1: 1.0, 2: 0.5, 3: 0.5}`<br>*(swap 0 & 1, $\Delta=2.0$)* | `dream`<br>`dream_untrans`<br>`no_dream` | **+4.50**<br>**+4.50**<br>**+4.00** | 1.000 (60/60)<br>1.000 (60/60)<br>N/A | Dream vs NoDream: **+0.50**<br>Untrans vs NoDream: **+0.50**<br>Transformation Effect: **+0.00** |
| **77** | **Fischer** | `{0: -1.0, 1: -1.0, 2: 0.5, 3: -0.5}` | `dream`<br>`dream_untrans`<br>`no_dream` | **-3.00**<br>**-3.00**<br>**-2.50** | 1.000 (60/60)<br>1.000 (60/60)<br>N/A | Dream vs NoDream: **-0.50**<br>Untrans vs NoDream: **-0.50**<br>Transformation Effect: **+0.00** |
| | **Mal** | `{0: 0.5, 1: -1.0, 2: -1.0, 3: -0.5}`<br>*(swap 0 & 2, $\Delta=1.5$)* | `dream`<br>`dream_untrans`<br>`no_dream` | **-6.00**<br>**-6.00**<br>**-8.50** | 1.000 (60/60)<br>1.000 (60/60)<br>N/A | Dream vs NoDream: **+2.50**<br>Untrans vs NoDream: **+2.50**<br>Transformation Effect: **+0.00** |
| **99** | **Fischer** | `{0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5}` | `dream`<br>`dream_untrans`<br>`no_dream` | **+5.00**<br>**+5.50**<br>**+4.50** | 1.000 (60/60)<br>1.000 (60/60)<br>N/A | Dream vs NoDream: **+0.50**<br>Untrans vs NoDream: **+1.00**<br>Transformation Effect: **-0.50** |
| | **Mal** | `{0: 0.5, 1: 1.0, 2: 1.0, 3: 0.5}`<br>*(swap 0 & 2, $\Delta=0.5$)* | `dream`<br>`dream_untrans`<br>`no_dream` | **+7.50**<br>**+7.00**<br>**+6.50** | 1.000 (60/60)<br>1.000 (60/60)<br>N/A | Dream vs NoDream: **+1.00**<br>Untrans vs NoDream: **+0.50**<br>Transformation Effect: **+0.50** |

---

## 4. Reading and Interpreting the Results

- **Shared Draw Schedules:** Because `dream` and `dream_untransformed` now start from identical RNG seeds for both environment transitions and agent action selection, any divergence between them reflects the impact of experiencing the transformed meaning rewards vs. untransformed rewards during the dream episode.
- **Single Paired Draws:** Each seed contributes one paired draw per arm. In Seeds 42 and 77, the transformation attribution ($\text{Dream} - \text{Dream Untransformed}$) was exactly $0.00$ (the agent's trajectory was unaffected by the meaning differences during the 60 dream steps). In Seed 99, the transformation attribution was $-0.50$ under Fischer target and $+0.50$ under Mal target.
- **No Causal Overreach:** These numbers are single paired draws under this specific tabular gridworld protocol; finite single draws do not establish general benefit or harm and do not rule out implementation or environment limits.

---

## 5. Decision-Trace Analysis (Internal Divergence Investigation)

### Passive Logging Architecture & Equivalence Gate
To investigate whether dream transformation changes internal mechanics (actions, Q-learning updates, per-action lookups, blend contributions, memory sampling) despite terminal return contrasts being near zero:
1. **Passive Observers:** Instrumented callbacks inside `before/memory.py` (`trace_listener`) and `before/agent.py` (`decision_listener`, `learn_listener`).
2. **Strict Non-Interference:** On exploratory random actions (`rng.random() < self.eps`), the policy does not evaluate action values or call `memory.value()`. Instrumentation records `is_random=True` and `None` for values without invoking `memory.value()`, preventing desynchronization of `memory._rng`.
3. **Equivalence Gate Passed (`tests/test_equivalence_gate.py`):** Verified 100% full-state deterministic equivalence between logging-on and logging-off on seed 42 (identical live/dream/return action sequences, final Q-tables, episodic memory contents, memory RNG state, and all stream final RNG states). In addition, summary returns and coverage across all seeds in v4 match the recorded v3 baseline numbers as a separately verified check.
4. **Trace Artifact:** Full step-by-step traces recorded to `checkpoints/stage4_runs_v4.json` across all 5 prespecified seeds (`[13, 23, 42, 77, 99]`).

### Four Descriptive Measures

#### Measure 1: Action Divergence During Dream Window (60 steps)
- **Seed 42:**
  - Realized dream: `{0: 1.0, 1: -1.0, 2: 0.5, 3: 0.5}` vs Live at suspension: `{0: -1.0, 1: 1.0, 2: 0.5, 3: -0.5}`.
  - First reward difference occurs at **step 1** (agent collects item 0: receives $+1.0$ in transformed dream vs $-1.0$ in untransformed dream).
  - Pre-difference window (steps 0-0): 100% action match (1/1 steps).
  - Post-difference window (steps 1-59): **96.6% action match** (57/59 steps). The agent chose the same action in 58/60 dream steps overall.
- **Seed 77:**
  - Realized dream: `{0: -1.0, 1: -1.0, 2: 0.5, 3: -0.5}` vs Live at suspension: `{0: -1.0, 1: 1.0, 2: 0.5, 3: -0.5}`.
  - The only difference between realized dream and live regime is object 1 (both live regimes value object 0 at -1.0, object 2 at 0.5, object 3 at -0.5).
  - During the 60 dream steps, the agent collected objects 0, 2, and 3, but **never encountered or collected object 1**.
  - As a result, the agent received identical rewards throughout: first reward difference step = **None**.
  - Pre-difference window: 100% action match (60/60 steps). Zero divergence during dream.
- **Seed 99:**
  - Realized dream: `{0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5}` vs Live at suspension: `{0: -1.0, 1: 1.0, 2: 0.5, 3: -0.5}`.
  - Differences on objects 0 and 3.
  - First reward difference occurs at **step 40** (agent collects object 0).
  - Pre-difference window (steps 0-39): 100% action match (40/40 steps).
  - Post-difference window (steps 40-59): 100% action match (20/20 steps). The trajectory remained identical throughout all 60 steps despite receiving a different reward on step 40, because the remaining steps were exploratory/transitional.

#### Measure 2: Learning Difference During Dream Window
- **Seed 42:**
  - Transformed dream $\sum |\Delta Q| = 0.7767$; Untransformed dream $\sum |\Delta Q| = 0.1728$.
  - Post-dream Q-table $L_1$ distance ($\sum |Q_{\text{dream}}(s, a) - Q_{\text{untrans}}(s, a)|$) = **0.6154**.
  - The meaning difference produced substantial Q-value updating differences during the dream episode.
- **Seed 77:**
  - Transformed dream $\sum |\Delta Q| = 0.5662$; Untransformed dream $\sum |\Delta Q| = 0.5662$.
  - Post-dream Q-table $L_1$ distance = **0.0000** (identical updates because object 1 was never collected in dream).
- **Seed 99:**
  - Transformed dream $\sum |\Delta Q| = 0.5010$; Untransformed dream $\sum |\Delta Q| = 0.1975$.
  - Post-dream Q-table $L_1$ distance = **0.3118**.

#### Measure 3: Return-Phase Difference (200 steps)
- **Trajectory State & Action Alignment:**
  - **Seed 42:**
    - Fischer: Dream and untransformed arms visited identical states on 173/200 return steps (action agreement at those states: 169/173 = 97.7%; overall return action agreement: 91.0%; first action divergence at step 71). Dream vs No-Dream diverged immediately at step 2 (51.5% agreement).
    - Mal: Identical states on 184/200 steps (action agreement: 182/184 = 98.9%; overall return action agreement: 95.0%; first action divergence at step 94).
  - **Seed 77:**
    - Fischer & Mal: Identical states on 200/200 return steps (100% action agreement; first action divergence = None). Dream vs No-Dream diverged at step 5 (~45% agreement).
  - **Seed 99:**
    - Fischer: Identical states on 98/200 return steps (action agreement at those states: 94/98 = 95.9%; overall return action agreement: 74.5%; first action divergence at step 44).
    - Mal: Identical states on 113/200 return steps (action agreement at those states: 110/113 = 97.3%; overall return action agreement: 77.5%; first action divergence at step 41).

#### Measure 4: Memory Usage & Blend Impact
- **Dream Entries in Return-Phase Memory Sampling:**
  - At return onset, memory contains 17,945 live transitions (299 full episodes plus 5 suspended steps) + 60 dream transitions = 18,005 entries (dream entries are ~0.33% of memory).
  - Each greedy evaluation samples `lookup_sample=400` entries with uniform replacement/sample.
  - Expected dream entries per sample: $400 \times \frac{60}{18,005} \approx 1.33$.
  - Measured across all seeds: **72.7% - 73.2% of all memory lookups sampled $\ge 1$ dream-phase entry** (mean sampled count: 1.30 - 1.32 entries per lookup pool).
- **Policy Blending & Argmax Shifts:**
  - Average absolute memory blend impact $|Q_{\text{blended}} - Q_{\text{raw}}| = 0.023 - 0.033$.
  - On greedy evaluation steps, memory shifted the chosen action ($\text{argmax}(Q_{\text{blended}}) \ne \text{argmax}(Q_{\text{raw}})$) on **46.4% - 52.2% of greedy steps**.
  - **Observed Association:** In all observed instances where an argmax shift occurred (e.g. 82/164 steps in Seed 42 Fischer, 85/164 in Seed 42 Mal, 84/161 in Seed 77 Fischer, 83/161 in Seed 77 Mal, 77/166 in Seed 99 Fischer, 81/166 in Seed 99 Mal), the sampled lookup pool happened to contain $\ge 1$ dream-phase entry (0 shifts occurred in pools without dream entries). However, because dream entries were present in ~73% of all lookup pools by base rate and no controlled counterfactual comparison has been run, this observation is reported strictly as an empirical association without causal attribution.
  - On jointly greedy steps in the return phase, dream and untransformed arms agreed on argmax choice 90.2% - 93.3% of the time in Seed 42, 100% in Seed 77, and 69.3% - 72.3% in Seed 99.

### Key Analytical Takeaways
1. **The Transformation Does Induce Internal Divergence:** In seeds where the agent collects an object whose meaning changed (Seeds 42 and 99), the transformation produces measurable divergence in Q-updates ($\sum |\Delta Q|$ difference of $0.3 - 0.6$), post-dream Q-tables ($L_1 > 0.3$), and return-phase action choices (diverging at steps 71 and 44).
2. **Return-Phase Return Patterns Across Seeds:**
   - In **Seed 77**, the changed object was simply not visited during the 60 dream steps, yielding mathematically zero difference between transformed and untransformed dream runs ($L_1 = 0.0$, 100% identical trajectory).
   - In **Seed 42**, the cumulative return sums came out equal (-1.5 vs -1.5 in Fischer, +4.5 vs +4.5 in Mal) despite substantial internal Q-table divergence ($L_1 = 0.6154$) and downstream action divergence starting at step 71/94. This equal return outcome is recorded as an empirical observation; no specific mechanism for the equality has been experimentally isolated or tested.
   - In **Seed 99**, divergence took effect after step 40, resulting in return contrasts of -0.50 in Fischer and +0.50 in Mal.
3. **No-Dream vs Dream Contrast:** Across all valid seeds, having *any* dream (transformed or untransformed) caused immediate behavioral divergence from the no-dream control by return step 2-5, confirming that the dream episode actively perturbs the agent's memory and subsequent policy.

---

## 6. Test Suite & Scoped Collection

Configured `pytest.ini` with `testpaths = tests` to strictly prevent collecting historical snapshots in `checkpoints/`.
Includes foundation tests and deterministic equivalence gate tests.

```
tests/test_foundations.py: 14 passed
tests/test_equivalence_gate.py: 2 passed
Total: 16 passed in 76.24s (measured on this machine)
```
