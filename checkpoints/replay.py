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

def view_runs():
    report_file = Path('checkpoints/stage4_runs.json')
    if not report_file.exists():
        print("Run stage 4 first to generate run records.")
        return

    with open(report_file) as f:
        data = json.load(f)

    print("=" * 70)
    print("STAGE 4 RUNS OVERVIEW (Prespecified Seeds)")
    print("=" * 70)
    for r in data['runs']:
        seed = r['seed']
        status = r['status']
        if status == 'EXCLUDED':
            print(f"Seed {seed:02d}: EXCLUDED -> {r['exclusion_reason']}")
        else:
            print(f"Seed {seed:02d}: VALID")
            print(f"  Suspended pos: {r['suspended_pos']}, uncollected: {r['uncollected_objs']}")
            print(f"  Realized dream: {r['realized_dream']}")
            print(f"  Fischer target: {r['fischer_target']}")
            f_arm = r['arms']['fischer_target']
            print(f"    Fischer target return: Dream={f_arm['dream']['return_cumreward']}, NoDream={f_arm['no_dream']['return_cumreward']}, Contrast={f_arm['within_target_contrast_dream_minus_nodream']}")
            print(f"  Mal target: {r['mal_target']} (swapped {r['mal_swapped_pair']}, diff={r['mal_swap_diff']})")
            m_arm = r['arms']['mal_target']
            print(f"    Mal target return:     Dream={m_arm['dream']['return_cumreward']}, NoDream={m_arm['no_dream']['return_cumreward']}, Contrast={m_arm['within_target_contrast_dream_minus_nodream']}")
    print("=" * 70)

if __name__ == '__main__':
    view_runs()
