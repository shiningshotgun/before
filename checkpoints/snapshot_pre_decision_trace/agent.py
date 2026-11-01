"""Tabular Q-learning agent. The decision loop is source-agnostic:
it reads a state, consults policy + memory, acts, receives an outcome.
The environment it is connected to can be the simulator or the learned model.
"""
import random

class QAgent:
    def __init__(self, n_actions=4, lr=0.3, gamma=0.95, eps=0.2, memory=None, memory_blend=0.5):
        self.q = {}
        self.n_actions, self.lr, self.gamma, self.eps = n_actions, lr, gamma, eps
        self.memory = memory
        self.memory_blend = memory_blend

    def qval(self, s, a):
        v = self.q.get((s, a), 0.0)
        if self.memory is not None:
            mv = self.memory.value(s, a)
            if mv is not None:
                v = (1 - self.memory_blend) * v + self.memory_blend * mv
        return v

    def act(self, s, rng: random.Random):
        if rng.random() < self.eps:
            return rng.randrange(self.n_actions)
        vals = [self.qval(s, a) for a in range(self.n_actions)]
        best = max(vals)
        cands = [a for a, v in enumerate(vals) if v == best]
        return rng.choice(cands)

    def learn(self, s, a, r, s_next):
        old = self.q.get((s, a), 0.0)
        target = r + self.gamma * max(self.qval(s_next, an) for an in range(self.n_actions))
        self.q[(s, a)] = old + self.lr * (target - old)

    def to_dict(self):
        """Serialize QAgent state to plain json-serializable dict (NO pickle)."""
        serialized_q = []
        for (s, a), val in self.q.items():
            pos, col = s
            serialized_q.append({
                'pos': list(pos),
                'collected': sorted(list(col)),
                'action': a,
                'val': val
            })
        mem_entries = []
        mem_rng = None
        if self.memory is not None:
            for s, a, out, t in self.memory.entries:
                pos, col = s
                mem_entries.append({
                    'pos': list(pos),
                    'collected': sorted(list(col)),
                    'action': a,
                    'outcome': out,
                    't': t
                })
            mem_rng = self.memory.get_rng_state()
            mem_lookup_sample = self.memory.lookup_sample
            mem_temperature = self.memory.temperature
        return {
            'q': serialized_q,
            'n_actions': self.n_actions,
            'lr': self.lr,
            'gamma': self.gamma,
            'eps': self.eps,
            'memory_blend': self.memory_blend,
            'memory_entries': mem_entries,
            'memory_rng_state': mem_rng,
            'memory_lookup_sample': mem_lookup_sample,
            'memory_temperature': mem_temperature
        }

    @classmethod
    def from_dict(cls, data):
        """Reconstruct QAgent from plain json-serializable dict."""
        from remnant.memory import EpisodicMemory
        mem = EpisodicMemory(
            temperature=data.get('memory_temperature', 0.1),
            lookup_sample=data.get('memory_lookup_sample', 400)
        )
        for e in data.get('memory_entries', []):
            s = (tuple(e['pos']), frozenset(e['collected']))
            mem.append(s, e['action'], e['outcome'], e['t'])
        if data.get('memory_rng_state') is not None:
            mem.set_rng_state(data['memory_rng_state'])
        
        agent = cls(
            n_actions=data['n_actions'],
            lr=data['lr'],
            gamma=data['gamma'],
            eps=data['eps'],
            memory=mem,
            memory_blend=data['memory_blend']
        )
        for item in data.get('q', []):
            s = (tuple(item['pos']), frozenset(item['collected']))
            agent.q[(s, item['action'])] = item['val']
        return agent
