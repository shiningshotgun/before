"""CLI viewer to step through post-return trajectories and compare conditions.
Renders 5x5 ASCII grid world, agent position, objects, collected set, and running reward.
"""
import json, sys
from pathlib import Path

WALLS = {(1, 2), (2, 2), (3, 2)}
OBJECT_LOCS = {0: (0, 4), 1: (4, 0), 2: (4, 4), 3: (2, 0)}
ACTION_NAMES = ['NORTH', 'SOUTH', 'WEST', 'EAST']

def render_grid(pos, collected):
    lines = []
    lines.append("+---" * 5 + "+")
    for r in range(5):
        row_str = "|"
        for c in range(5):
            cell = (r, c)
            if cell == pos:
                row_str += " A |"
            elif cell in WALLS:
                row_str += "###|"
            else:
                obj_found = None
                for idx, o_pos in OBJECT_LOCS.items():
                    if o_pos == cell:
                        obj_found = idx
                        break
                if obj_found is not None:
                    if obj_found in collected:
                        row_str += f" x{obj_found}|"
                    else:
                        row_str += f" O{obj_found}|"
                else:
                    row_str += "   |"
        lines.append(row_str)
        lines.append("+---" * 5 + "+")
    return "\n".join(lines)

def view_runs(filename='checkpoints/stage4_runs_v3.json'):
    report_file = Path(filename)
    if not report_file.exists():
        for fallback in ['checkpoints/stage4_runs_v2.json', 'checkpoints/stage4_runs.json']:
            if Path(fallback).exists():
                report_file = Path(fallback)
                break
        if not report_file.exists():
            print(f"No run records found at {filename} or fallbacks.")
            return

    with open(report_file) as f:
        data = json.load(f)

    print("=" * 80)
    print(f"STAGE 4 RUNS OVERVIEW ({report_file.name})")
    print("=" * 80)
    for r in data['runs']:
        seed = r['seed']
        status = r['status']
        if status == 'EXCLUDED':
            print(f"Seed {seed:02d}: EXCLUDED -> {r['exclusion_reason']}")
        else:
            print(f"Seed {seed:02d}: VALID")
            print(f"  Suspended pos: {r['suspended_pos']}, uncollected: {r['uncollected_objs']}")
            print(f"  Realized dream: {r['realized_dream']}")
            for tgt in ['fischer_target', 'mal_target']:
                tgt_meanings = r[tgt]
                arm_data = r['arms'][tgt]
                d = arm_data['dream']
                u = arm_data.get('dream_untransformed', {})
                n = arm_data['no_dream']
                c = arm_data.get('contrasts', {})
                print(f"  [{tgt}] Meanings: {tgt_meanings}")
                print(f"    dream (transformed):   ret={d['return_cumreward']:6.2f} (cov={d.get('coverage')})")
                if u:
                    print(f"    dream (untransformed): ret={u['return_cumreward']:6.2f} (cov={u.get('coverage')})")
                print(f"    no_dream:              ret={n['return_cumreward']:6.2f}")
                if c:
                    print(f"    Contrasts: dream - nodream = {c.get('dream_minus_nodream'):+.2f} | "
                          f"untrans - nodream = {c.get('dream_untransformed_minus_nodream'):+.2f} | "
                          f"transformation effect (dream - untrans) = {c.get('dream_minus_dream_untransformed'):+.2f}")
    print("=" * 80)

if __name__ == '__main__':
    fname = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/stage4_runs_v3.json'
    view_runs(fname)
