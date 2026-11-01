"""Dream transformation source.
Supports:
- 'none': faithful replay of base meanings
- 'manual': operator-set meanings (DEBUG CONTROL ONLY)
- 'learned_variation': samples a plausible transformation from the experienced
  distribution of regime rewards (optionally recombining per-object values across regimes).
"""
import random

def learned_transform(experienced_regimes: dict, rng: random.Random, recombine: bool = True):
    """Samples transformed per-object rewards from experienced regimes.
    
    experienced_regimes: dict of regime_id -> {obj_idx: reward_value}
    recombine: If True, draws each object's reward independently from what that object
               was observed to yield across experienced regimes.
               If False, samples an entire experienced regime as a coherent bundle.
    """
    assert experienced_regimes, "experienced_regimes cannot be empty for learned_variation"
    
    # Get all object indices seen across all regimes
    all_objs = set()
    for obj_map in experienced_regimes.values():
        all_objs.update(obj_map.keys())
    all_objs = sorted(all_objs)

    if not recombine:
        regime_id = rng.choice(list(experienced_regimes.keys()))
        return dict(experienced_regimes[regime_id])
    
    transformed = {}
    for obj in all_objs:
        # Collect candidate reward values experienced for this object
        candidates = [
            obj_map[obj]
            for obj_map in experienced_regimes.values()
            if obj in obj_map
        ]
        if candidates:
            transformed[obj] = rng.choice(candidates)
        else:
            transformed[obj] = 0.0
    return transformed


def transform(meanings, rng, mode='none', manual=None, experienced_regimes=None, recombine=True):
    if mode == 'none':
        return dict(meanings)
    if mode == 'manual':
        assert manual is not None, "manual mode requires operator-set meanings (debug control)"
        return dict(manual)
    if mode == 'learned_variation':
        assert experienced_regimes is not None, (
            "learned_variation requires experienced_regimes extracted from the agent's history"
        )
        return learned_transform(experienced_regimes, rng, recombine=recombine)
    raise ValueError(f"Unknown transform mode: {mode}")
