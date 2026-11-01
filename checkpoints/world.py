"""Grid world: the place (fixed landmarks, walls) plus a hidden meaning layer.
Meanings (object rewards, door destinations) are observable only by acting.
Changing them adds no explicit marker to the observation stream.
"""
from dataclasses import dataclass, field

NORTH, SOUTH, WEST, EAST = 0, 1, 2, 3
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

@dataclass
class GridWorld:
    height: int
    width: int
    walls: frozenset          # blocked cells
    object_cells: tuple       # cells holding objects, in a stable order
    object_rewards: dict      # meaning layer: object index -> reward
    step_cost: float = -0.01
    # runtime state
    pos: tuple = (0, 0)
    collected: frozenset = frozenset()

    def reset(self, start=(0, 0)):
        self.pos = start
        self.collected = frozenset()
        return self.state()

    def state(self):
        return (self.pos, self.collected)

    def step(self, action):
        dr, dc = ACTIONS[action]
        cand = (self.pos[0] + dr, self.pos[1] + dc)
        if 0 <= cand[0] < self.height and 0 <= cand[1] < self.width and cand not in self.walls:
            self.pos = cand
        reward = self.step_cost
        if self.pos in self.object_cells:
            idx = self.object_cells.index(self.pos)
            if idx not in self.collected:
                reward += self.object_rewards[idx]
                self.collected = self.collected | {idx}
        return self.state(), reward

    def change_meanings(self, new_rewards: dict):
        """Silent meaning swap: no marker is added to any observation."""
        self.object_rewards = dict(new_rewards)


def default_world():
    # 5x5 with a central wall column and two gaps; four objects.
    walls = frozenset({(1, 2), (2, 2), (3, 2)})
    objs = ((0, 4), (4, 0), (4, 4), (2, 0))
    rewards = {0: 1.0, 1: -1.0, 2: 0.5, 3: 0.5}
    return GridWorld(5, 5, walls, objs, rewards)
