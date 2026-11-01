"""Episodic memory: one buffer for live and dream transitions alike.
No origin field. The agent is never told which entries came from where.
"""
import math, random

class EpisodicMemory:
    def __init__(self, temperature=0.1, lookup_sample=400):
        self.entries = []          # (state, action, outcome, time)
        self.temperature = temperature
        self.lookup_sample = lookup_sample
        self._rng = random.Random(0)
        self.trace_listener = None  # Passive observer: fn(state, action, value, pool_indices)

    def append(self, state, action, outcome, t):
        self.entries.append((state, action, outcome, t))

    def value(self, state, action):
        """Similarity-weighted value lookup (NEC pattern) over all entries."""
        if len(self.entries) > self.lookup_sample:
            indices = self._rng.sample(range(len(self.entries)), self.lookup_sample)
            pool = [self.entries[i] for i in indices]
        else:
            indices = list(range(len(self.entries)))
            pool = self.entries

        num, den = 0.0, 0.0
        for s, a, out, _ in pool:
            sim = self._sim(state, s)
            if a == action and sim > 0:
                w = math.exp(sim / self.temperature)
                num += w * out
                den += w
        val = num / den if den else None

        if self.trace_listener is not None:
            self.trace_listener(state, action, val, indices)

        return val

    @staticmethod
    def _sim(s1, s2):
        (p1, c1), (p2, c2) = s1, s2
        dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + len(c1 ^ c2)
        return 1.0 / (1.0 + dist)

    def get_rng_state(self):
        """Return the internal RNG state as a serializable tuple/list."""
        raw = self._rng.getstate()
        return (raw[0], list(raw[1]), raw[2])

    def set_rng_state(self, state_tuple):
        """Restore the internal RNG state."""
        st = (state_tuple[0], tuple(state_tuple[1]), state_tuple[2])
        self._rng.setstate(st)
