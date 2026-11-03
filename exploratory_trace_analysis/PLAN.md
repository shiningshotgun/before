# Exploratory Post-Hoc Trace Analysis Plan

## 1. Scope & Status Statement

This document defines the post-hoc exploratory analysis plan for the closed `before` pilot dataset located in `repeated_draws/` (note: the pilot predates the project rename from Remnant to `before`). 

- **Independence from Closed Pilot:** The repeated paired draws pilot was prespecified in `prespec_repeated_draws.md` with an analysis-once commitment. Its conclusions and artifacts are immutable. Everything produced in `exploratory_trace_analysis/` is post-hoc, exploratory, and hypothesis-generating.
- **No Causal or Performance Claims:** Findings are strictly descriptive of observational traces within this specific discrete tabular gridworld under fixed hyperparameters. No claims of causal benefit, cognitive advantage, or general population validity are made.
- **Scope Boundaries:** No new seeds, runs, replays, or model evaluations will be performed. Nested dream layers and external environments remain out of scope.

---

## 2. Logged Fields Audit: Capabilities & Limitations

Inspection of the raw trace files (`repeated_draws/seed_<seed>.json`) and the driver instrumentation confirms the following field availability:

### Available Quantities
1. **Dream Phase (60 steps per dream arm):**
   - `q_table_before_dream`: Full serialized Q-table list of dictionaries (`pos`, `col`, `action`, `val`) at dream start.
   - `q_updates_dream`: Ordered list of 60 step dictionaries containing `step`, `state` (`[pos, col]`), `action`, `reward`, `next_state`, `old_q`, `target`, `new_q`, `delta`.
   - `q_table_after_dream`: Full serialized Q-table at dream completion.
   - **Feasibility Determination:** **CONFIRMED FEASIBLE.** The per-dream-update Q-table trajectory $Q_t$ for $t \in [0, 60]$ can be reconstructed exactly by applying the recorded transitions to $Q_0$ and verifying $Q_{60} == Q_{\text{after}}$.
   - *Note:* The paired `no_dream` arm logs `len(q_updates_dream) == 0` and identical before/after Q-tables.

2. **Return Phase (200 steps per arm):**
   - `return_trace`: 200 step records per arm (`step`, `state`, `chosen_action`, `is_random`, `q_raws`, `m_vals`, `blended_vals`, `memory_lookups`, `reward`, `next_state`).
   - `chosen_action`: Integer $\in \{0, 1, 2, 3\}$.
   - `is_random`: Boolean indicating whether $\epsilon$-greedy exploration triggered `rng.randrange(...)`.
   - `q_raws`, `m_vals`, `blended_vals`: Numerical dictionaries/lists populated when `is_random == False`.

### Omitted Quantities & Observational Boundaries
1. **Per-Step RNG Internal State:** While `final_rng_states` is logged at episode end, the per-step internal state of `random.Random` is **not logged per step**. Consequently, matched-state divergences where `is_random == True` cannot be inspected for underlying pseudo-random draw values beyond the boolean indicator `is_random`.
2. **Attribution Limitation:** Same-state action differences cannot be attributed to value differences unless `is_random == False` on both paired arms and `blended_vals` argmax differs without ties.

---

## 3. Joint Classification Scheme (Seed $\times$ Target)

To prevent forcing data into an assumed monotonic hierarchy, each seed $\times$ target pair ($N = 27 \times 2 = 54$) is classified by the 4-tuple of observed binary indicators:
$$(q_{\text{div}}, a_{\text{div}}, s_{\text{div}}, r_{\text{div}})$$
where:
- $q_{\text{div}} \in \{0, 1\}$: $L_1(Q_{\text{dream}}, Q_{\text{untrans}}) > 0$ post-dream.
- $a_{\text{div}} \in \{0, 1\}$: Any return action mismatch: $\exists t \in [0, 199], a^{(d)}_t \neq a^{(u)}_t$.
- $s_{\text{div}} \in \{0, 1\}$: Return-phase collection schedule difference (timing, order, or identity of collected items).
- $r_{\text{div}} \in \{0, 1\}$: Return contrast nonzero: $R^{(d)}_{\text{return}} - R^{(u)}_{\text{return}} \neq 0$.

### Monotone Hierarchy Profiles
- **Profile (0, 0, 0, 0) — Exact Null (Seed-77 pattern):** Zero Q divergence, zero action divergence, identical collection schedule, zero return contrast.
- **Profile (1, 0, 0, 0) — Latent Divergence:** Q table modified by dream, but return actions never diverge (divergent Q states never visited, or argmax unaffected).
- **Profile (1, 1, 0, 0) — Behavioral Drift without Schedule Impact (Seed-42 pattern):** Q diverged, actions diverged, but collection schedule and return remain identical.
- **Profile (1, 1, 1, 0) — Schedule Divergence without Return Impact:** Actions and collection timing/order diverged, but cumulative return is identical.
- **Profile (1, 1, 1, 1) — Complete Divergence:** Q, actions, schedule, and return all diverged.

### Non-Nested & Irregular Profiles
Any observation outside the monotonic funnel is classified as an explicit irregular profile and reported individually:
- **Profile (1, 1, 0, 1):** Return changed without schedule change (e.g., step-cost discrepancy from longer paths between identical item collections).
- **Profile (0, 1, *, *):** Action divergence despite zero post-dream Q divergence (e.g., memory lookup divergence or unaligned exploration).
- **Profile (*, 0, 1, *):** Schedule divergence without action divergence (inconsistent in a deterministic simulator; flag as anomaly if observed).
- **Profile (*, 0, 0, 1):** Return divergence without action divergence (inconsistent; flag as anomaly if observed).

---

## 4. Matched-State Comparison & Propagation Methodology

For every valid seed and target condition where action divergence occurs ($a_{\text{div}} = 1$):
1. **First Divergence Step ($t_{\text{div}}$):**
   - The smallest step index $t \in [0, 199]$ where $a^{(d)}_t \neq a^{(u)}_t$.
   - By definition, trajectories match up to $t_{\text{div}}$, so $s^{(d)}_{t_{\text{div}}} = s^{(u)}_{t_{\text{div}}}$.
2. **Matched-State Categorization at $t_{\text{div}}$:**
   - **Random Exploration Branch:** If `is_random == True` in either `dream` or `dream_untransformed`.
   - **Greedy Value Shift:** If `is_random == False` in both arms, and $\text{argmax}(\text{blended\_vals}^{(d)}) \neq \text{argmax}(\text{blended\_vals}^{(u)})$.
   - **Tie-Breaking Ambiguity:** If `is_random == False` in both arms, but multiple actions achieve maximum blended value.
   - **Insufficient Data:** If required fields (`blended_vals`, `is_random`) are missing or unreadable.
3. **Trajectory Divergence Persistence & Rejoining:**
   - For all steps $t > t_{\text{div}}$:
     - Record state match: $I_s(t) = \mathbb{I}(s^{(d)}_t == s^{(u)}_t)$.
     - Record action match: $I_a(t) = \mathbb{I}(a^{(d)}_t == a^{(u)}_t)$.
   - **State Persistence Metric:** Fraction of post-divergence steps in different states:
     $$P_{\text{state}} = \frac{1}{200 - t_{\text{div}}} \sum_{t=t_{\text{div}}}^{199} (1 - I_s(t))$$
   - **Rejoining Definition:**
     - *Transient State Intersection:* Paired trajectories visit the same state at step $t$ ($I_s(t) = 1$) but subsequently diverge.
     - *Absorbing Rejoin:* Trajectories match in both state and action permanently from step $t_{\text{rejoin}}$ onward:
       $$\forall t \ge t_{\text{rejoin}}, \quad s^{(d)}_t = s^{(u)}_t \land a^{(d)}_t = a^{(u)}_t$$

---

## 5. Asymmetry Case Studies

Two focal cases with asymmetric target outcomes will be analyzed via full step-by-step trajectory reconstruction:
1. **Seed 3233:** Schedule divergence observed in Fischer target, but NOT in Mal target.
2. **Seed 9126:** Schedule divergence observed in Mal target, but NOT in Fischer target.

For both seeds, the reconstruction will trace:
- Suspension state and realized dream regime vs live regime.
- Which specific items were swapped under Mal target.
- The 60-step dream Q updates and their spatial overlap with the return trajectory.
- The return trajectory pathing, item collection events, and terminal return composition.

---

## 6. Stop & Falsification Criteria

To maintain rigorous experimental boundaries, analysis stops immediately if any of the following occur:
1. **Data Corruption Gate:** If any of the 27 valid JSON files fails schema validation, is unreadable, or lacks `arms`, `fischer_target`, `mal_target`, or full 200-step return traces.
2. **Anchor Reproduction Gate:** If the independent parser does not reproduce the independently verified numbers from the closed pilot:
   - Post-dream Q $L_1 > 0$ count: exactly 14 per target.
   - Action divergence count: exactly 12 Fischer, 11 Mal.
   - First action divergence mean step: $71.8 \pm 0.1$ (Fischer), $63.4 \pm 0.1$ (Mal).
   - Archetype partition: exactly 13 null, 6 pattern, 8 divergent.
   - Schedule divergence count: exactly 8 per target (7 shared).
   - Return contrast distribution: exact match of counts and values.
3. **Q Reconstruction Verification Gate:** If reconstructed $Q_{60}$ fails to match `q_table_after_dream` within numerical precision ($10^{-5}$) on any valid run.

---

## 7. Synthetic Fixture Specifications

Four synthetic fixtures are defined to test the independent parser prior to evaluating real traces:

1. **`fixture_null` (Null Archetype):**
   - Construct identical Q tables ($L_1 = 0$), identical 200-step return traces with identical greedy actions and collection events.
   - Expected Output: `q_div=0, act_div=0, sched_div=0, ret_div=0`. Persistence = 0.0, Rejoined = True ($t=0$).
2. **`fixture_divergent_persistent` (Complete Divergent Archetype):**
   - Construct $L_1 > 0$ Q tables. Return traces match up to step 20 ($t_{\text{div}} = 20$). At step 20, arm `dream` takes action 0 (greedy shift), arm `untrans` takes action 1.
   - Trajectories remain in separate grid coordinates through step 199 with different collected items and unequal returns.
   - Expected Output: `q_div=1, act_div=1, sched_div=1, ret_div=1`. Persistence = 1.0, Absorbing Rejoin = False.
3. **`fixture_rejoining` (Absorbing Rejoin Archetype):**
   - Construct $L_1 > 0$ Q tables. Return traces match up to step 10 ($t_{\text{div}} = 10$). Actions diverge from step 10 to 29 (different local paths), but rejoin at state `[3, 3]` at step 30 ($t_{\text{rejoin}} = 30$) and remain identical through step 199.
   - Expected Output: `q_div=1, act_div=1, sched_div=0, ret_div=0`. Persistence < 1.0, Absorbing Rejoin = True ($t_{\text{rejoin}} = 30$).
4. **`fixture_non_nested_latent` (Latent Q Divergence / Non-Nested):**
   - Construct $L_1 > 0$ Q tables, but differences are placed exclusively on unvisited states.
   - All 200 return steps have identical states, actions, collections, and returns.
   - Expected Output: `q_div=1, act_div=0, sched_div=0, ret_div=0`. Profile: Latent Divergence.
