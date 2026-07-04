"""
Coarse grid-based reachability + route planning.

This is a simplified kinematic model of the player (grid cells, no ramp
incline nuance) used for two things:
  1. The validator's semantic reachability check (generation/validator.py) --
     "is this scene solvable at all," before it ever touches real physics.
  2. The genuine solver's route planning (solvers/genuine.py) -- a sequence
     of waypoints, later converted to discrete move/jump actions and executed
     in the real pymunk engine, which is the ground truth for what actually
     happens (collisions, death, event log).

Jump edges are validated by actually simulating the parabolic arc (see
`_simulate_jump`) using the exact same constants as the real engine
(harness/constants.py) -- an earlier version approximated jumps with a
straight-line-of-cells check, which let the grid approve edges the real
physics couldn't make (a real jump rises fast early and arcs over, so a
straight-line interpolation systematically underestimates height near the
start of the arc and can miss the player bonking its head on a platform
the true arc passes into). Simulating the real equations of motion removes
that whole mismatch class instead of re-tuning approximation constants.

Ramps and platforms are both treated as solid rectangles here (the incline is
a rendering/physics detail, not a reachability-graph detail) -- documented
simplification, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.constants import DT, GRAVITY, JUMP_SPEED, MOVE_SPEED, PLAYER_H

CELL = 20  # px
FALL_MAX_DY = 420  # px, max drop considered survivable/traversable for pathing
# Closed-form max rise of the real jump arc (v^2/2g), with a small safety
# margin for discrete-timestep simulation -- used only for the "grab a
# floating collectible with a straight-up hop" case (see below).
MAX_VERTICAL_RISE = 0.9 * (JUMP_SPEED**2) / (2 * GRAVITY)


def _rect_cells(x, y, w, h):
    """All grid (col,row) cells overlapped by an axis-aligned rect."""
    c0, c1 = int(x // CELL), int((x + w) // CELL)
    r0, r1 = int(y // CELL), int((y + h) // CELL)
    return [(c, r) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1)]


@dataclass
class GridWorld:
    cols: int
    rows: int
    solid: set  # (col,row) cells blocked by platform/ramp/locked-door
    lethal: set  # (col,row) cells that kill on entry (lethal hazard)
    key_cells: dict  # (col,row) -> key_id
    pickup_cells: dict  # (col,row) -> pickup_id
    door_cells: dict  # (col,row) -> door object dict (only while locked; removed from `solid` dynamically)
    zone_cells: dict  # (col,row) -> zone_id
    start_cell: tuple
    jump_cache: dict = field(default_factory=dict)  # (from_cell, direction, frozenset(held_keys)) -> landing cell|None


def build_grid(scene: dict) -> GridWorld:
    w, h = scene["world"]["width"], scene["world"]["height"]
    cols, rows = int(w // CELL) + 1, int(h // CELL) + 1
    solid, lethal = set(), set()
    key_cells, pickup_cells, door_cells, zone_cells = {}, {}, {}, {}

    for obj in scene["objects"]:
        t = obj["type"]
        if t in ("platform", "ramp"):
            for c in _rect_cells(obj["x"], obj["y"], obj["width"], obj["height"]):
                solid.add(c)
        elif t == "door":
            cells = _rect_cells(obj["x"], obj["y"], obj["width"], obj["height"])
            for c in cells:
                door_cells[c] = obj
            # Door cells are NOT added to `solid` here. `solid` is a flat set
            # of coordinates with no per-object provenance, so if a door cell
            # coincides with e.g. a ground cell (a door reaching down to floor
            # level), discarding it when the door opens would also erase the
            # ground's solidity at that coordinate. Door solidity is instead
            # computed fresh each reachability pass from `door_cells` +
            # current held_keys (see reachability_closure), unioned with the
            # permanent `solid` set built from platforms/ramps only.
        elif t == "hazard":
            cells = _rect_cells(obj["x"], obj["y"], obj["width"], obj["height"])
            if obj["lethal"]:
                lethal.update(cells)
            # non-lethal hazards are walkable; no grid effect beyond a collision event
        elif t == "key":
            # Objectives reference the object's `id` (e.g. "key_1"), not its
            # `key_id` (e.g. "k1", which only matters for door-matching) --
            # keep both so reachability_closure can report the former while
            # using the latter to unlock doors.
            for c in _rect_cells(obj["x"] - obj["radius"], obj["y"] - obj["radius"], 2 * obj["radius"], 2 * obj["radius"]):
                key_cells[c] = {"id": obj["id"], "key_id": obj["key_id"]}
        elif t == "pickup":
            for c in _rect_cells(obj["x"] - obj["radius"], obj["y"] - obj["radius"], 2 * obj["radius"], 2 * obj["radius"]):
                pickup_cells[c] = obj["id"]

    gz = scene["goal_zone"]
    for c in _rect_cells(gz["x"], gz["y"], gz["width"], gz["height"]):
        zone_cells[c] = "goal_zone"

    sx, sy = scene["player_start"]["x"], scene["player_start"]["y"]
    # Use the foot line (y + PLAYER_H), not the top-left corner, so a player
    # resting exactly on a surface lands in the free cell just above it
    # rather than one row too high (which would never satisfy _is_standing).
    start_cell = (int(sx // CELL), int((sy + PLAYER_H - 1) // CELL))
    return GridWorld(cols, rows, solid, lethal, key_cells, pickup_cells, door_cells, zone_cells, start_cell)


def _is_standing(grid: GridWorld, c, r, solid_now: set) -> bool:
    if (c, r) in solid_now or (c, r) in grid.lethal:
        return False
    if not (0 <= c < grid.cols and 0 <= r < grid.rows):
        return False
    return (c, r + 1) in solid_now or r + 1 >= grid.rows


def _line_clear(grid: GridWorld, c0, r0, c1, r1, solid_now: set) -> bool:
    """Coarse straight-line-of-cells check used to approve a jump arc."""
    steps = max(abs(c1 - c0), abs(r1 - r0), 1)
    for i in range(steps + 1):
        c = round(c0 + (c1 - c0) * i / steps)
        r = round(r0 + (r1 - r0) * i / steps)
        if (c, r) in grid.lethal:
            return False
        if (c, r) in solid_now and (c, r) != (c1, r1):
            return False
    return True


def _resolve_start_cell(grid: GridWorld, cell: tuple, solid_now: set) -> tuple:
    """The genuine solver replans from the player's live position every hop,
    and mid-settle after landing the body can be a few px penetrated into a
    platform -- transiently placing the computed cell inside solid ground
    instead of the standing cell just above it. Pop upward out of solid
    ground first (this transient case), then fall back to the normal
    "already in open air, drop to the nearest surface" handling.
    """
    c, r = cell
    if (c, r) in solid_now:
        while r > 0 and (c, r) in solid_now:
            r -= 1
        cell = (c, r)
    if not _is_standing(grid, *cell, solid_now=solid_now):
        landing = _fall_target(grid, cell[0], cell[1], solid_now)
        cell = landing if landing else cell
    return cell


def _fall_target(grid: GridWorld, c, r, solid_now: set):
    for dr in range(1, FALL_MAX_DY // CELL + 1):
        rr = r + dr
        if rr >= grid.rows:
            return None
        if (c, rr) in grid.lethal:
            return None
        if (c, rr) in solid_now:
            return (c, rr - 1) if _is_standing(grid, c, rr - 1, solid_now) else None
    return None


def _simulate_jump(grid: GridWorld, from_cell: tuple, direction: int, solid_now: set, held_keys: frozenset = None):
    """Simulate the exact jump arc the real engine's controller executes:
    jump immediately (vy=-JUMP_SPEED), constant horizontal speed, standard
    gravity, stepped at the same DT as the real engine. Returns the landing
    standing cell if the arc lands cleanly there without first clipping a
    solid/lethal cell, else None. `direction` is +1 (right) or -1 (left).

    The genuine/adversarial solvers replan very frequently (every hop, or
    every stalled attempt), and `find_path` calls this for every BFS node it
    expands -- the same (from_cell, direction) simulation gets re-run
    thousands of times across a solve with an unchanged door-lock state.
    `held_keys` fully determines door-lock state (see build_grid), so it's a
    sound cache key; pass it (as a frozenset) to memoize on `grid`.
    """
    cache_key = (from_cell, direction, held_keys) if held_keys is not None else None
    if cache_key is not None and cache_key in grid.jump_cache:
        return grid.jump_cache[cache_key]

    x = from_cell[0] * CELL + CELL / 2
    y = from_cell[1] * CELL + CELL / 2
    vx = direction * MOVE_SPEED
    vy = -JUMP_SPEED
    t = 0.0
    result = None
    # Real total flight time back to launch height is 2*JUMP_SPEED/GRAVITY
    # (~1.24s at current constants); cap a bit above that to allow landing
    # somewhat below launch height, not the arbitrary 3.0s originally used.
    max_t = 2 * JUMP_SPEED / GRAVITY + 1.0
    while t < max_t:
        vy += GRAVITY * DT
        x += vx * DT
        y += vy * DT
        t += DT
        c, r = int(x // CELL), int(y // CELL)
        if not (0 <= c < grid.cols and 0 <= r < grid.rows):
            break
        if (c, r) in grid.lethal or (c, r) in solid_now:
            break  # clipped a hazard, wall, or platform (incl. its underside)
        if vy >= 0 and _is_standing(grid, c, r, solid_now):
            result = (c, r)  # descending and landed on solid ground
            break

    if cache_key is not None:
        grid.jump_cache[cache_key] = result
    return result


def reachability_closure(scene: dict, grid: GridWorld | None = None):
    """Fixed-point BFS: repeatedly explore reachable standing cells with the
    currently-held key set (doors matching a held key become passable), pick
    up any keys/pickups encountered, and repeat until nothing new is reached.

    Returns (reachable_cells: set, reached_ids: set[str], held_keys: set[str]).
    `reached_ids` contains object `id`s (keys/pickups) and zone ids
    ("goal_zone") whose cell was visited -- these are the same ids the
    objective predicate and event log use. `held_keys` separately tracks
    `key_id`s (the door-matching field, distinct from object id).
    """
    grid = grid or build_grid(scene)
    held_keys: set = set()
    reached_ids: set = set()
    reachable: set = set()

    while True:
        solid_now = set(grid.solid)
        for c, obj in grid.door_cells.items():
            currently_locked = obj["locked"] and obj["key_id"] not in held_keys
            if currently_locked:
                solid_now.add(c)

        start = _resolve_start_cell(grid, grid.start_cell, solid_now)
        held_keys_frozen = frozenset(held_keys)

        visited = {start}
        queue = [start]
        while queue:
            c, r = queue.pop()
            for dc in (-1, 1):
                nc, nr = c + dc, r
                if _is_standing(grid, nc, nr, solid_now) and (nc, nr) not in visited:
                    visited.add((nc, nr))
                    queue.append((nc, nr))
                elif (0 <= nc < grid.cols and (nc, nr) not in solid_now and (nc, nr) not in grid.lethal):
                    fall_to = _fall_target(grid, nc, nr, solid_now)
                    if fall_to and fall_to not in visited:
                        visited.add(fall_to)
                        queue.append(fall_to)
            for direction in (-1, 1):
                landing = _simulate_jump(grid, (c, r), direction, solid_now, held_keys_frozen)
                if landing and landing not in visited:
                    visited.add(landing)
                    queue.append(landing)

        # A key/pickup doesn't need to be a *landing* spot -- a player can
        # jump straight up from any standing cell, grab it mid-air, and fall
        # back down. Without this, any collectible floating in open air
        # (not resting on its own platform) is wrongly marked unreachable,
        # since the main BFS above only ever visits standing/landing cells.
        collectible_cells = set(grid.key_cells) | set(grid.pickup_cells)
        for (c, r) in list(visited):
            for dr in range(1, int(MAX_VERTICAL_RISE // CELL) + 1):
                cell = (c, r - dr)
                if cell in collectible_cells and _line_clear(grid, c, r, c, r - dr, solid_now):
                    visited.add(cell)

        new_keys = set()
        new_ids = set()
        for cell in visited:
            if cell in grid.key_cells:
                new_keys.add(grid.key_cells[cell]["key_id"])
                new_ids.add(grid.key_cells[cell]["id"])
            if cell in grid.pickup_cells:
                new_ids.add(grid.pickup_cells[cell])
            if cell in grid.zone_cells:
                new_ids.add(grid.zone_cells[cell])

        if visited <= reachable and new_keys <= held_keys:
            reachable = visited
            reached_ids |= new_ids
            break
        reachable = visited
        reached_ids |= new_ids
        held_keys |= new_keys

    return reachable, reached_ids, held_keys


def find_path(grid: GridWorld, start_cell: tuple, held_keys: set, target_id: str):
    """Single BFS pass (not the fixed-point closure -- used by the genuine
    solver, which already knows which keys it currently holds and re-plans
    after each pickup) from start_cell to whichever cell satisfies target_id
    (a key/pickup/door's object id, or "goal_zone").

    Returns a list of (cell, edge_type) pairs -- edge_type is the edge used
    to *reach* that cell: "start" for the first entry, else "walk"/"fall"/
    "jump"/"grab" (a synthetic final hop for collectibles reached via
    vertical hop, per the same jump-and-grab logic used in
    reachability_closure). The edge type is load-bearing for the controller:
    a "jump" between two cells at the *same* row (a flat jump across a gap)
    has no row delta to infer a jump from, so the caller must be told
    explicitly rather than guessing from row differences. Returns None if
    unreachable under the current held_keys.
    """
    solid_now = set(grid.solid)
    for c, obj in grid.door_cells.items():
        if obj["locked"] and obj["key_id"] not in held_keys:
            solid_now.add(c)

    start = _resolve_start_cell(grid, start_cell, solid_now)
    held_keys_frozen = frozenset(held_keys)

    parent = {start: None}
    edge_type = {start: "start"}
    queue = [start]

    def cell_matches(cell):
        if target_id == "goal_zone":
            return grid.zone_cells.get(cell) == "goal_zone"
        if cell in grid.key_cells and grid.key_cells[cell]["id"] == target_id:
            return True
        if cell in grid.pickup_cells and grid.pickup_cells[cell] == target_id:
            return True
        return False

    def _reconstruct(cell):
        path = [cell]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        path.reverse()
        return [(c, edge_type[c]) for c in path]

    if cell_matches(start):
        return [(start, "start")]

    while queue:
        c, r = queue.pop(0)
        neighbors = []
        for dc in (-1, 1):
            nc, nr = c + dc, r
            if _is_standing(grid, nc, nr, solid_now):
                neighbors.append(((nc, nr), "walk"))
            elif 0 <= nc < grid.cols and (nc, nr) not in solid_now and (nc, nr) not in grid.lethal:
                fall_to = _fall_target(grid, nc, nr, solid_now)
                if fall_to:
                    neighbors.append((fall_to, "fall"))
        for direction in (-1, 1):
            landing = _simulate_jump(grid, (c, r), direction, solid_now, held_keys_frozen)
            if landing:
                neighbors.append((landing, "jump"))

        for n, etype in neighbors:
            if n not in parent:
                parent[n] = (c, r)
                edge_type[n] = etype
                if cell_matches(n):
                    return _reconstruct(n)
                queue.append(n)

    # not found as a standing/landing cell -- check the vertical-grab case
    for (c, r) in list(parent):
        for dr in range(1, int(MAX_VERTICAL_RISE // CELL) + 1):
            cell = (c, r - dr)
            if cell_matches(cell) and _line_clear(grid, c, r, c, r - dr, solid_now):
                parent[cell] = (c, r)
                edge_type[cell] = "grab"
                return _reconstruct(cell)

    return None
