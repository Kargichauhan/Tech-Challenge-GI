"""
Corridor-perspective first-person renderer: a deliberately scoped-down
stand-in for a Wolfenstein-style raycaster, and the honest bridge toward the
challenge's stated vision-policy action space (move forward/back/left/right
+ mouse look), not a claim of a full 3D game.

Why not a real raycaster. That technique casts a fan of rays across a
horizontal field of view into a 2D top-down map of walls, and reads depth
off where each ray first hits a wall. It needs a genuine second spatial axis
(left/right) with real geometry in it. This world doesn't have one: it's a
side-scrolling platformer, one horizontal axis (x) plus gravity (y). Casting
a fan of rays here would have every ray in the fan hit the exact same point,
since there is nothing to either side, so a literal port of the algorithm
would be motion with no information behind it.

What's built instead: a single forward-looking view along the player's
current heading (+x or -x). Objects at different x-distances genuinely come
into and out of view as the player advances. The depth is real, but it's
earned through time (approaching an object) rather than through a second
spatial axis at a single instant. Each object is perspective-scaled (nearer
is larger) and vertically positioned by its actual world y relative to the
player's eye line, so a platform that's genuinely higher up still reads as
higher up. This is closer to an on-rails corridor-runner view than a maze
FPS, and is presented as exactly that.

Cosmetic 2.5D dressing (side wall panels, a receding floor grid, a drifting
parallax backdrop) is layered on top purely as framing: explicitly
decorative, not derived from or claiming to represent any real geometry
(there are no actual side walls in this world). It's drawn first, and every
real feature (platforms, hazards, doors, the goal zone) is drawn on top of
it, so what's real versus decoration is a strict, visible layering order,
not something a viewer has to guess at.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from harness.render.scene_renderer import COLORS, _draw_text

pygame.init()
pygame.font.init()

FOCAL = 320.0  # perspective scale constant: scale = FOCAL / (distance + FOCAL)
MIN_SCALE = 0.04
MAX_RENDER_DISTANCE = 1400.0
EYE_HEIGHT_FRAC = 0.55  # horizon line as a fraction of frame height

SKY_TOP = (14, 15, 20)
SKY_BOTTOM = (28, 30, 40)
GROUND_NEAR = (46, 42, 34)
GROUND_FAR = (24, 24, 28)

# Cosmetic-dressing constants: pure framing, not derived from real geometry.
OPENING_WIDTH_FRAC = 0.42   # width of the "tunnel opening" at the horizon, as a fraction of frame width
OPENING_HEIGHT_FRAC = 0.5   # height of that opening band around the horizon line
WALL_COLOR_NEAR = (58, 56, 64)
WALL_COLOR_FAR = (30, 29, 34)
WALL_SEAM_COLOR = (18, 18, 22)
N_WALL_SEAMS = 4
FLOOR_GRID_COLOR = (18, 17, 15)
N_FLOOR_LINES = 7
N_FLOOR_VERTICALS = 5
PARALLAX_COLOR = (20, 21, 28)
PARALLAX_PERIOD_PX = 2000.0  # world-x distance for one full drift cycle of the backdrop

_FEATURE_COLORS = {
    "platform": COLORS["platform"],
    "ramp": COLORS["ramp"],
    "door_locked": COLORS["door_locked"],
    "door_open": COLORS["door_open"],
    "hazard": COLORS["hazard"],
    "key": COLORS["key"],
    "pickup": COLORS["pickup"],
    "goal_zone": COLORS["goal_zone"],
}


def infer_heading(action: str, previous_heading: str) -> str:
    """Same convention as claude_agent/navigator.py's last_direction and
    solvers/common.py's hop execution: only move_left/move_right actually
    change facing; jump/noop continue whatever direction was most recently
    walked, defaulting to rightward if nothing has moved yet."""
    if action in ("move_left", "move_right"):
        return action
    return previous_heading


def _scale_for_distance(distance: float) -> float:
    return max(MIN_SCALE, FOCAL / (distance + FOCAL))


def _visible_features(scene: dict, engine, heading: str) -> list[dict]:
    px, py = engine.player_body.position
    direction = 1 if heading == "move_right" else -1
    features = []

    for obj in scene["objects"]:
        if obj["type"] == "platform" and obj["id"] == "ground":
            continue  # the ground plane is drawn separately, not as a feature
        if obj["id"] in engine.collected_ids:
            continue
        center_x = obj["x"] + obj.get("width", 0) / 2 if "width" in obj else obj["x"]
        dx = (center_x - px) * direction
        if dx <= 0 or dx > MAX_RENDER_DISTANCE:
            continue
        center_y = obj["y"] + obj.get("height", 0) / 2 if "height" in obj else obj["y"]
        color_key = obj["type"]
        if obj["type"] == "door":
            is_open = not obj["locked"]
            for shape, meta in engine.shape_meta.items():
                if meta["type"] == "door" and meta["id"] == obj["id"]:
                    is_open = shape.sensor
            color_key = "door_open" if is_open else "door_locked"
        size = obj.get("width", obj.get("radius", 20) * 2)
        height = obj.get("height", obj.get("radius", 20) * 2)
        features.append({
            "distance": dx, "center_y": center_y, "size": size, "height": height,
            "color": _FEATURE_COLORS.get(color_key, COLORS["text_dim"]), "type": obj["type"],
        })

    gz = scene["goal_zone"]
    gz_center_x = gz["x"] + gz["width"] / 2
    gz_dx = (gz_center_x - px) * direction
    if 0 < gz_dx <= MAX_RENDER_DISTANCE:
        features.append({
            "distance": gz_dx, "center_y": gz["y"] + gz["height"] / 2, "size": gz["width"],
            "height": gz["height"], "color": COLORS["goal_zone"], "type": "goal_zone",
        })

    features.sort(key=lambda f: -f["distance"])  # far first (painter's algorithm)
    return features


def _draw_parallax_backdrop(surf, w: int, horizon_y: int, player_x: float):
    """A drifting silhouette in the sky band, purely decorative background
    parallax, not tied to any real object. Drift is slow (a fraction of the
    player's actual world position) so it reads as "distant," not as a real
    tracked feature."""
    offset = (player_x * 0.08) % PARALLAX_PERIOD_PX
    peak_h = horizon_y * 0.35
    spacing = 220
    start = -int(offset) % spacing - spacing
    x = start
    while x < w + spacing:
        points = [(x, horizon_y), (x + spacing * 0.5, horizon_y - peak_h), (x + spacing, horizon_y)]
        pygame.draw.polygon(surf, PARALLAX_COLOR, points)
        x += spacing


def _draw_corridor_walls(surf, w: int, h: int, horizon_y: int):
    """Two flat-shaded side-wall trapezoids framing a central 'opening' at
    the horizon, plus a few perspective seam lines: the classic corridor-
    game depth trick. No real walls exist in this world; this only ever
    frames the same real content, never substitutes for it."""
    opening_w = w * OPENING_WIDTH_FRAC
    opening_h = h * OPENING_HEIGHT_FRAC
    ol, orr = w / 2 - opening_w / 2, w / 2 + opening_w / 2
    ot, ob = horizon_y - opening_h / 2, horizon_y + opening_h / 2

    left_wall = [(0, 0), (ol, ot), (ol, ob), (0, h)]
    right_wall = [(w, 0), (orr, ot), (orr, ob), (w, h)]
    pygame.draw.polygon(surf, WALL_COLOR_NEAR, left_wall)
    pygame.draw.polygon(surf, WALL_COLOR_NEAR, right_wall)

    for i in range(1, N_WALL_SEAMS + 1):
        t = i / (N_WALL_SEAMS + 1)
        color = tuple(int(WALL_COLOR_NEAR[c] + (WALL_COLOR_FAR[c] - WALL_COLOR_NEAR[c]) * t) for c in range(3))
        lx = 0 + (ol - 0) * t
        ly_top = 0 + (ot - 0) * t
        ly_bot = h + (ob - h) * t
        pygame.draw.line(surf, color, (lx, ly_top), (lx, ly_bot), width=1)
        rx = w + (orr - w) * t
        pygame.draw.line(surf, color, (rx, ly_top), (rx, ly_bot), width=1)

    pygame.draw.line(surf, WALL_SEAM_COLOR, (ol, ot), (ol, ob), width=2)
    pygame.draw.line(surf, WALL_SEAM_COLOR, (orr, ot), (orr, ob), width=2)


def _draw_floor_grid(surf, w: int, h: int, horizon_y: int):
    """Receding tile lines on the floor band, denser near the horizon: a
    decorative perspective cue layered on the real floor gradient, not a
    claim about world geometry."""
    floor_h = h - horizon_y
    if floor_h <= 0:
        return
    for i in range(1, N_FLOOR_LINES + 1):
        t = (i / N_FLOOR_LINES) ** 2  # denser spacing near the horizon
        y = horizon_y + floor_h * t
        pygame.draw.line(surf, FLOOR_GRID_COLOR, (0, y), (w, y), width=1)

    vp = (w / 2, horizon_y)
    for i in range(N_FLOOR_VERTICALS + 1):
        bx = w * i / N_FLOOR_VERTICALS
        pygame.draw.line(surf, FLOOR_GRID_COLOR, vp, (bx, h), width=1)


def render_first_person_frame(scene: dict, engine, heading: str = "move_right", caption: str | None = None) -> pygame.Surface:
    w, h = scene["world"]["width"], scene["world"]["height"]
    surf = pygame.Surface((w, h))
    horizon_y = int(h * EYE_HEIGHT_FRAC)
    px, py = engine.player_body.position

    for row in range(horizon_y):
        t = row / max(horizon_y, 1)
        color = tuple(int(SKY_TOP[i] + (SKY_BOTTOM[i] - SKY_TOP[i]) * t) for i in range(3))
        pygame.draw.line(surf, color, (0, row), (w, row))
    for row in range(horizon_y, h):
        t = (row - horizon_y) / max(h - horizon_y, 1)
        color = tuple(int(GROUND_NEAR[i] + (GROUND_FAR[i] - GROUND_NEAR[i]) * (1 - t)) for i in range(3))
        pygame.draw.line(surf, color, (0, row), (w, row))

    _draw_parallax_backdrop(surf, w, horizon_y, px)
    _draw_floor_grid(surf, w, h, horizon_y)
    _draw_corridor_walls(surf, w, h, horizon_y)

    for feature in _visible_features(scene, engine, heading):
        scale = _scale_for_distance(feature["distance"])
        fw = max(2, int(feature["size"] * scale))
        fh = max(2, int(feature["height"] * scale))
        screen_y = horizon_y + (feature["center_y"] - py) * scale
        rect = pygame.Rect(0, 0, fw, fh)
        rect.center = (w // 2, int(screen_y))
        fog = min(1.0, feature["distance"] / MAX_RENDER_DISTANCE)
        color = tuple(int(c * (1 - 0.6 * fog)) for c in feature["color"])
        if feature["type"] in ("key", "pickup"):
            pygame.draw.circle(surf, color, rect.center, max(2, fw // 2))
        else:
            pygame.draw.rect(surf, color, rect)

    _draw_text(surf, f"facing: {'right' if heading == 'move_right' else 'left'}", (10, 8), size=16)
    if caption:
        _draw_text(surf, caption, (10, h - 30), size=18)

    return surf
