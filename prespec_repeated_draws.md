# Remnant: Repeated Paired Draws Pilot Prespecification

This document records the exact prespecification for an exploratory pilot of repeated paired draws under the unchanged Remnant protocol. It is written and committed before launching any experiment runs. The prespecification is immutable: if it changes later, the pilot stops and reports rather than adapting.

---

## 1. Protocol & Seed Generation

### Seed Generation
Generated via `.venv/bin/python -c "import random; print(random.Random(20261101).sample(range(1000, 10000), 35))"`. Exactly thirty-five unique seeds, disjoint from the published five (`[13, 23, 42, 77, 99]`):

```python
PRESPECIFIED_SEEDS = [
    2491, 2896, 7456, 1978, 1711, 1826, 4591, 5500, 6396, 3052,
    3327, 8314, 9004, 2490, 3233, 1022, 6792, 1431, 9056, 7356,
    2692, 9126, 1111, 4054, 2784, 6406, 5048, 3998, 6035, 6333,
    1478, 3563, 4915, 7763, 5876
]
```

### Unchanged Experimental Protocol
- **Simulator & Layout:** Default 5x5 gridworld (`remnant/world.py`).
- **Regimes:** Regime 0 (episodes 0–149) and Regime 1 (episodes 150–299) unchanged.
- **Mid-Episode Suspension:** Step $t=5$ of episode 299.
- **Dream Transformation:** Per-object learned recombination (`learned_transform(..., recombine=True)`).
- **Target Conditions:** Fischer (outer target := realized dream) and Mal (controlled swap of uncollected objects).
- **Three Paired Arms:** `dream` (transformed), `dream_untransformed` (faithful replay), and `no_dream` (control), paired on identical initial RNG states per seed.
- **Validity Rules:**
  - Rule 1: Realized dream must differ from live regime at suspension.
  - Rule 2: Mal mismatch must differ on at least one uncollected object.
  - Rule 3: Mal target must differ from live regime at suspension.
  Exclusions are recorded with rule and reason, counted, and never replaced.
- **Logging:** Full passive trace logging (`log_traces=True`), gated by the verified full-state non-interference implementation.

---

## 2. Prespecified Outcomes

### Primary Outcome
Per target condition (Fischer, Mal), the incidence across **valid seeds** of any return-phase collection-schedule difference between `dream` (transformed) and `dream_untransformed` (faithful replay).
- A collection-schedule difference is defined as any divergence in the identity, timing (step number), or count of collected objects across the 200 return steps.
- Reported with a standard Wilson 95% score interval:
  $$\hat{p} \pm \frac{z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
  with $z = 1.95996$.
- **Interpretation Guard:** This interval is descriptive of seed-to-seed sampling variability under this exact discrete gridworld protocol (this grid, regimes, horizon, hyperparameters). It is not a population claim, a causal-benefit claim, or an advantage claim.

### Secondary Outcomes
1. **Return Contrasts:** Per-target distribution (mean, median, range) of terminal cumulative return differences ($\text{Return}_{\text{dream}} - \text{Return}_{\text{dream\_untrans}}$).
2. **Archetype Incidence:**
   - **Seed-42 Pattern:** Post-dream Q-table divergence ($L_1 > 0$) accompanied by an identical return-phase collection schedule.
   - **Seed-77 Pattern:** Exact mathematical null (realized dream differed on objects never visited during dream; dream updates $\sum |\Delta Q|$ identical; post-dream $L_1 = 0$; return trajectory 100% identical).
3. **First Divergence Step:** Distribution of the first return step where the chosen action diverges between `dream` and `dream_untransformed`.

### Planning Expectations (Not Guarantees)
Historically, 2 of 5 seeds were excluded under Validity Rule 1. Across 35 candidate seeds, approximately 20 valid draws are anticipated, yielding a Wilson 95% interval half-width near 0.22 in the worst-case ($p=0.5$). The actual valid count is accepted and reported as-is.

---

## 3. Fixed Budget & Stop Rules

- **Cumulative Caps:**
  - **Wall Clock:** 45 minutes (2700 seconds) cumulative across resumes.
  - **Storage:** 1.5 GB cumulative of new files in `repeated_draws/`.
- **Enforcement:**
  - Tracked across resumes in `repeated_draws/manifest.json`.
  - When either cap is reached: the driver stops launching new seeds immediately.
  - If a seed is interrupted mid-execution, only that seed's incomplete temporary file (`repeated_draws/tmp_<seed>.json`) is deleted. Completed files are never deleted or modified.
  - Remaining seeds are reported as `NOT_RUN` (never replaced).
- **No Outcome Peeking:** Analysis is executed exactly once, after the batch has terminated, over completed outputs only.
