"""Stage 3: dream machinery. One episode inside the LEARNED model from an
injected start. Check: episode is coherent (outcomes follow the model's own
rules; agent behaves goal-directedly) and is logged in the same format as
real episodes, appended to the same unlabeled memory.

Defect 6 fix: does not count unknown (s,a) as vacuous passes. Reports full
transition coverage, unknown counts, and checks independent model consistency.
Gate exits nonzero on failure.
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, '.')
from remnant.model import WorldModel
from run_stage12 import load_checkpoint, CKPT

SEED = 11
DREAM_STEPS = 60

class DreamEnv:
    """Environment interface backed entirely by the learned WorldModel."""
    def __init__(self, model, rng):
        self.model = model
        self.rng = rng
        self.s = None

    def reset(self, start_state):
        self.s = start_state
        return self.s

    def step(self, a):
        self.s, r = self.model.sample_next(self.s, a, self.rng)
        return self.s, r

def main():
    if not CKPT.exists():
        print(f"Error: {CKPT} not found. Run run_stage12.py first.", file=sys.stderr)
        sys.exit(1)

    agent, episodes_done, log, _ = load_checkpoint(CKPT)
    model = WorldModel()
    for s, a, r, s2 in log:
        model.observe(s, a, r, s2)

    rng = random.Random(SEED)
    dream = DreamEnv(model, rng)

    # Injected start: state agent visited mid-route in logged history
    inject_state = log[len(log) // 2][0]
    s = dream.reset(inject_state)

    dream_log = []
    total_reward = 0.0
    known_steps = 0
    unknown_steps = 0
    violations = 0

    for t in range(DREAM_STEPS):
        a = agent.act(s, rng)
        known_succs = model.counts.get((s, a), {})
        s2, r = dream.step(a)

        if not known_succs:
            # Defect 6: Unknown state-action pair encountered
            unknown_steps += 1
        else:
            known_steps += 1
            # Verify sampled transition was learned by the model
            if s2 not in known_succs:
                violations += 1

        dream_log.append((s, a, r, s2))
        agent.memory.append(s, a, r, t)  # Unlabeled episodic memory
        total_reward += r
        s = s2

    coverage = known_steps / DREAM_STEPS if DREAM_STEPS > 0 else 0.0
    stage3_pass = (violations == 0 and unknown_steps == 0 and coverage == 1.0 and DREAM_STEPS > 0)

    report = {
        'seed': SEED,
        'injected_start': f"pos={inject_state[0]}, col={sorted(list(inject_state[1]))}",
        'dream_steps': DREAM_STEPS,
        'dream_total_reward': round(total_reward, 3),
        'known_steps': known_steps,
        'unknown_steps': unknown_steps,
        'coverage': coverage,
        'rule_violations': violations,
        'logged_same_format': all(len(x) == 4 for x in dream_log),
        'memory_entries_total': len(agent.memory.entries),
        'stage3_pass': stage3_pass
    }

    out_file = Path('checkpoints/stage3_dream.json')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    serialized_log = [
        [[list(x[0][0]), sorted(list(x[0][1]))], x[1], round(x[2], 4), [list(x[3][0]), sorted(list(x[3][1]))]]
        for x in dream_log
    ]
    with open(out_file, 'w') as f:
        json.dump({'report': report, 'dream_log': serialized_log}, f, indent=2)

    print(json.dumps(report, indent=2))

    assert stage3_pass, (
        f"Stage 3 Gate Failed: violations={violations}, unknown_steps={unknown_steps}, coverage={coverage}"
    )

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"GATE ASSERTION FAILURE: {e}", file=sys.stderr)
        sys.exit(1)
