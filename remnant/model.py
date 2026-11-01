"""Learned world model: transition and reward tables estimated from the
agent's own logged experience. Distinct from the simulator's actual rules;
contains the agent's own understanding, including its errors. This model -
not the simulator - generates the dream.
"""
from collections import defaultdict
import random

class WorldModel:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))   # (s,a) -> s' -> n
        self.reward_sum = defaultdict(float)                  # (s,a) -> sum r
        # Optional regime tracking (experimenter-side metadata only)
        # regime_id -> obj_idx -> list of observed object reward increments
        self.regime_obj_rewards = defaultdict(lambda: defaultdict(list))

    def observe(self, s, a, r, s_next, regime=None):
        self.counts[(s, a)][s_next] += 1
        self.reward_sum[(s, a)] += r
        if regime is not None:
            # Check if an object was collected in this transition
            diff = s_next[1] - s[1]
            if diff:
                obj_idx = next(iter(diff))
                # Store r - step_cost (or store raw r, we can resolve step_cost)
                self.regime_obj_rewards[regime][obj_idx].append(r)

    def sample_next(self, s, a, rng: random.Random):
        succ = self.counts.get((s, a))
        if not succ:
            return s, 0.0
        total = sum(succ.values())
        x = rng.uniform(0, total)
        for sn, n in succ.items():
            x -= n
            if x <= 0:
                return sn, self.reward_sum[(s, a)] / total
        sn = next(iter(succ))
        return sn, self.reward_sum[(s, a)] / total

    def predict(self, s, a):
        succ = self.counts.get((s, a))
        if not succ:
            return None, None
        sn = max(succ, key=succ.get)
        return sn, self.reward_sum[(s, a)] / sum(succ.values())

    def accuracy_on(self, held_out, reward_tolerance=0.05):
        """Evaluate accuracy of argmax s' and predicted reward r on held-out transitions."""
        seen = 0
        correct_next = 0
        correct_reward = 0
        correct_both = 0
        dropped_unseen = 0

        for s, a, r, sn in held_out:
            psn, pr = self.predict(s, a)
            if psn is not None:
                seen += 1
                sn_match = (psn == sn)
                r_match = (abs(pr - r) <= reward_tolerance)
                if sn_match:
                    correct_next += 1
                if r_match:
                    correct_reward += 1
                if sn_match and r_match:
                    correct_both += 1
            else:
                dropped_unseen += 1

        acc_next = correct_next / seen if seen else 0.0
        acc_reward = correct_reward / seen if seen else 0.0
        acc_both = correct_both / seen if seen else 0.0
        total_transitions = seen + dropped_unseen
        coverage = seen / total_transitions if total_transitions else 0.0

        return {
            'next_state_accuracy': acc_next,
            'reward_accuracy': acc_reward,
            'joint_accuracy': acc_both,
            'seen_count': seen,
            'dropped_unseen_count': dropped_unseen,
            'total_count': total_transitions,
            'coverage': coverage
        }

    def infer_step_cost(self):
        """Estimate step cost from transitions where no object was collected."""
        non_collect_r = []
        for (s, a), sn_counts in self.counts.items():
            for sn, cnt in sn_counts.items():
                if not (sn[1] - s[1]):
                    # No object collected
                    r_avg = self.reward_sum[(s, a)] / sum(sn_counts.values())
                    non_collect_r.extend([r_avg] * cnt)
        if non_collect_r:
            return sum(non_collect_r) / len(non_collect_r)
        return -0.01

    def infer_object_locations(self):
        """Infer object index to grid location mapping purely from transitions."""
        obj_locs = {}
        for (s, a), sn_counts in self.counts.items():
            for sn in sn_counts:
                diff = sn[1] - s[1]
                if diff:
                    obj_idx = next(iter(diff))
                    obj_locs[obj_idx] = sn[0]
        return obj_locs

    def get_experienced_regime_meanings(self):
        """Return mean object reward for each experienced regime."""
        step_cost = self.infer_step_cost()
        regime_meanings = {}
        for reg, objs in self.regime_obj_rewards.items():
            regime_meanings[reg] = {}
            for obj_idx, r_list in objs.items():
                # Reward received was step_cost + object_reward, so obj_reward = r - step_cost
                obj_r = (sum(r_list) / len(r_list)) - step_cost
                regime_meanings[reg][obj_idx] = round(obj_r, 4)
        return regime_meanings
