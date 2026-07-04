"""Single source of truth for player movement physics constants, shared by
the real engine (engine/physics_engine.py) and the coarse grid model
(generation/pathfinding.py). These two must use identical numbers -- the
grid model simulates the same parabolic jump arc the real engine executes,
so any drift between them reintroduces the "grid approves a jump the real
physics can't make" class of bug this file exists to prevent.
"""

MOVE_SPEED = 260.0
JUMP_SPEED = 560.0
GRAVITY = 900.0
DT = 1 / 60.0
PLAYER_W, PLAYER_H = 28, 40
