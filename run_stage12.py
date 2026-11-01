"""Stages 1-2: live training baseline + learned world-model accuracy.
Checkpointed and resumable via plain JSON (no pickle). Logs seed, config, transitions,
and strictly enforces gates with nonzero exits upon failure.

Pass criteria justification (prespecified):
- Stage 1: The default gridworld has positive rewards at objects 0 (+1.0), 2 (+0.5), 3 (+0.5)
  and penalty at object 1 (-1.0). Total optimal collection yields +2.0 gross reward minus
  step costs (~0.12). A random agent achieves negative returns (~ -0.6). A proficient agent
  collecting positive objects achieves > 1.0. Baseline threshold is set to 1.0.
- Stage 2: WorldModel transition and reward accuracy must exceed 0.80 (80%) on held-out transitions,
  with transition coverage > 0.80.
"""
import json, random, resource, sys, time
from pathlib import Path
sys.path.insert(0, '.')
from before.world import default_world
from before.agent import QAgent
from before.memory import EpisodicMemory
from before.model import WorldModel

CONFIG = {
    'seed': 7,
    'train_episodes': 300,
    'eval_episodes': 50,
    'max_steps': 60,
    'checkpoint_path': 'checkpoints/stage1.json',
    'pass_criteria': {
        'stage1_threshold': 1.0,
        'stage1_rationale': 'Proficient agent collecting positive items yields gross +2.0 minus path steps; baseline threshold 1.0 ensures non-trivial policy.',
        'stage2_accuracy_threshold': 0.8,
        'stage2_coverage_threshold': 0.8,
        'stage2_rationale': 'Held-out next-state and reward predictions must both exceed 80% with at least 80% coverage to verify model accuracy without vacuous pass.'
    }
}

SEED = CONFIG['seed']
TRAIN_EPISODES = CONFIG['train_episodes']
EVAL_EPISODES = CONFIG['eval_episodes']
MAX_STEPS = CONFIG['max_steps']
CKPT = Path(CONFIG['checkpoint_path'])

def serialize_log(log):
    out = []
    for s, a, r, s2 in log:
        out.append([
            [list(s[0]), sorted(list(s[1]))],
            a,
            r,
            [list(s2[0]), sorted(list(s2[1]))]
        ])
    return out

def deserialize_log(data):
    out = []
    for item in data:
        s = (tuple(item[0][0]), frozenset(item[0][1]))
        a = item[1]
        r = item[2]
        s2 = (tuple(item[3][0]), frozenset(item[3][1]))
        out.append((s, a, r, s2))
    return out

def run_episode(env, agent, rng, log, learn=True, record_memory=True):
    s = env.reset((0, 0))
    total, t = 0.0, 0
    while t < MAX_STEPS:
        a = agent.act(s, rng)
        s2, r = env.step(a)
        if log is not None:
            log.append((s, a, r, s2))
        if record_memory and agent.memory is not None:
            agent.memory.append(s, a, r, t)
        if learn:
            agent.learn(s, a, r, s2)
        total += r
        s = s2
        t += 1
    return total

def save_checkpoint(path: Path, agent: QAgent, episodes_done: int, log: list, rng: random.Random):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix('.tmp')
    raw_state = rng.getstate()
    serializable_rng = (raw_state[0], list(raw_state[1]), raw_state[2])
    payload = {
        'agent': agent.to_dict(),
        'episodes_done': episodes_done,
        'log': serialize_log(log),
        'rng_state': serializable_rng,
        'config': CONFIG
    }
    with open(tmp_path, 'w') as f:
        json.dump(payload, f)
    tmp_path.replace(path)

def load_checkpoint(path: Path):
    with open(path, 'r') as f:
        data = json.load(f)
    agent = QAgent.from_dict(data['agent'])
    episodes_done = data['episodes_done']
    log = deserialize_log(data['log'])
    rng = random.Random()
    raw_state = data['rng_state']
    rng_state = (raw_state[0], tuple(raw_state[1]), raw_state[2])
    rng.setstate(rng_state)
    return agent, episodes_done, log, rng

def main():
    resumed = False
    if CKPT.exists():
        agent, start_ep, log, rng = load_checkpoint(CKPT)
        resumed = True
    else:
        rng = random.Random(SEED)
        agent = QAgent(memory=EpisodicMemory())
        start_ep, log = 0, []

    world = default_world()
    t0 = time.time()
    for ep in range(start_ep, TRAIN_EPISODES):
        run_episode(world, agent, rng, log, learn=True, record_memory=True)

    # Checkpoint agent + progress + log + exact rng state
    save_checkpoint(CKPT, agent, TRAIN_EPISODES, log, rng)

    # Stage-1 evaluation check: greedy eval return WITHOUT mutating episodic memory
    eval_rng = random.Random(SEED + 1)
    agent.eps = 0.0
    mem_len_before = len(agent.memory.entries) if agent.memory else 0
    returns = [run_episode(world, agent, eval_rng, log=None, learn=False, record_memory=False)
               for _ in range(EVAL_EPISODES)]
    mem_len_after = len(agent.memory.entries) if agent.memory else 0
    assert mem_len_before == mem_len_after, "Defect 8 violation: evaluation mutated episodic memory!"

    avg_return = sum(returns) / len(returns)
    stage1_threshold = CONFIG['pass_criteria']['stage1_threshold']
    stage1_pass = avg_return > stage1_threshold

    # Stage-2 check: learned model accuracy on held-out transitions
    cut = int(0.9 * len(log))
    model = WorldModel()
    for s, a, r, s2 in log[:cut]:
        model.observe(s, a, r, s2)

    eval_stats = model.accuracy_on(log[cut:], reward_tolerance=0.01)
    stage2_acc_threshold = CONFIG['pass_criteria']['stage2_accuracy_threshold']
    stage2_cov_threshold = CONFIG['pass_criteria']['stage2_coverage_threshold']
    stage2_pass = (eval_stats['next_state_accuracy'] >= stage2_acc_threshold and
                   eval_stats['reward_accuracy'] >= stage2_acc_threshold and
                   eval_stats['coverage'] >= stage2_cov_threshold)

    # macOS ru_maxrss is reported in bytes; convert to MB
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    if sys.platform != 'darwin':
        max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    report = {
        'seed': SEED,
        'resumed_from_checkpoint': resumed,
        'train_episodes': TRAIN_EPISODES,
        'transitions_logged': len(log),
        'stage1_avg_eval_return': round(avg_return, 3),
        'stage1_threshold': stage1_threshold,
        'stage1_rationale': CONFIG['pass_criteria']['stage1_rationale'],
        'stage1_pass': stage1_pass,
        'stage2_heldout_next_state_accuracy': round(eval_stats['next_state_accuracy'], 3),
        'stage2_heldout_reward_accuracy': round(eval_stats['reward_accuracy'], 3),
        'stage2_heldout_joint_accuracy': round(eval_stats['joint_accuracy'], 3),
        'stage2_heldout_seen': eval_stats['seen_count'],
        'stage2_heldout_dropped_unseen': eval_stats['dropped_unseen_count'],
        'stage2_heldout_coverage': round(eval_stats['coverage'], 3),
        'stage2_accuracy_threshold': stage2_acc_threshold,
        'stage2_coverage_threshold': stage2_cov_threshold,
        'stage2_rationale': CONFIG['pass_criteria']['stage2_rationale'],
        'stage2_pass': stage2_pass,
        'wall_seconds': round(time.time() - t0, 2),
        'max_rss_mb': round(max_rss_mb, 2),
    }
    print(json.dumps(report, indent=2))

    # Assert gates and exit nonzero on failure
    assert stage1_pass, f"Stage 1 Gate Failed: avg_return {avg_return} <= {stage1_threshold}"
    assert stage2_pass, f"Stage 2 Gate Failed: accuracy or coverage below threshold"

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"GATE ASSERTION FAILURE: {e}", file=sys.stderr)
        sys.exit(1)
