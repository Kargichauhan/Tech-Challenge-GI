"""
Headless pygame renderer: draws a scene + live engine state to an off-screen
Surface, frame by frame, for the GIF pipeline (render/gif_builder.py). No
display/window is ever created (SDL_VIDEODRIVER=dummy), so this runs fine on
a server with no GPU/X11.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from harness.engine.physics_engine import PLAYER_H, PLAYER_W

pygame.init()
pygame.font.init()

COLORS = {
    "bg": (24, 26, 34),
    "platform": (90, 96, 112),
    "ramp": (110, 96, 78),
    "door_locked": (196, 110, 40),
    "door_open": (70, 150, 90),
    "hazard": (200, 60, 60),
    "key": (235, 200, 60),
    "pickup": (70, 140, 220),
    "goal_zone": (80, 210, 130),
    "player": (240, 240, 245),
    "text": (235, 235, 240),
    "text_dim": (160, 165, 175),
    "caption_bg": (10, 11, 15),
    "exploit": (230, 90, 90),
    "clean": (90, 200, 130),
}

_FONT_CACHE: dict = {}


def _font(size: int):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.SysFont("dejavusansmono,couriernew,monospace", size)
    return _FONT_CACHE[size]


def _draw_text(surf, text, pos, size=18, color=None, center=False):
    color = color or COLORS["text"]
    img = _font(size).render(text, True, color)
    rect = img.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surf.blit(img, rect)
    return rect


def _draw_player(surf, rect):
    """A small robot instead of a plain bar: a head (with eyes and an
    antenna) on top of a body block."""
    color = COLORS["player"]
    head_h = max(10, int(rect.height * 0.4))
    head_rect = pygame.Rect(rect.x, rect.y, rect.width, head_h)
    body_rect = pygame.Rect(rect.x, rect.y + head_h, rect.width, rect.height - head_h)
    pygame.draw.rect(surf, color, body_rect, border_radius=3)
    pygame.draw.rect(surf, color, head_rect, border_radius=4)

    antenna_x = rect.centerx
    pygame.draw.line(surf, color, (antenna_x, rect.y - 5), (antenna_x, rect.y), width=2)
    pygame.draw.circle(surf, COLORS["goal_zone"], (antenna_x, rect.y - 5), 2)

    eye_y = rect.y + head_h // 2
    eye_dx = max(3, rect.width // 4)
    pygame.draw.circle(surf, COLORS["bg"], (rect.centerx - eye_dx, eye_y), 2)
    pygame.draw.circle(surf, COLORS["bg"], (rect.centerx + eye_dx, eye_y), 2)


def render_scene_frame(scene: dict, engine=None, caption: str | None = None) -> pygame.Surface:
    """One frame: the static scene geometry, plus (if `engine` given) live
    state: player position, which doors are open, which keys/pickups are
    already collected, whether the player is inside goal_zone."""
    w, h = scene["world"]["width"], scene["world"]["height"]
    surf = pygame.Surface((w, h))
    surf.fill(COLORS["bg"])

    collected = engine.collected_ids if engine else set()
    open_doors = set()
    if engine:
        for shape, meta in engine.shape_meta.items():
            if meta["type"] == "door" and shape.sensor:
                open_doors.add(meta["id"])

    gz = scene["goal_zone"]
    zone_color = COLORS["goal_zone"]
    pygame.draw.rect(surf, zone_color, (gz["x"], gz["y"], gz["width"], gz["height"]), width=3)

    for obj in scene["objects"]:
        t = obj["type"]
        if t in ("platform", "ramp"):
            color = COLORS[t]
            pygame.draw.rect(surf, color, (obj["x"], obj["y"], obj["width"], obj["height"]))
        elif t == "door":
            is_open = obj["id"] in open_doors or not obj["locked"]
            color = COLORS["door_open"] if is_open else COLORS["door_locked"]
            alpha_rect = pygame.Rect(obj["x"], obj["y"], obj["width"], obj["height"])
            if is_open:
                pygame.draw.rect(surf, color, alpha_rect, width=3)
            else:
                pygame.draw.rect(surf, color, alpha_rect)
        elif t == "hazard":
            color = COLORS["hazard"]
            pygame.draw.rect(surf, color, (obj["x"], obj["y"], obj["width"], obj["height"]))
            # hazard hatching for legibility even in a still frame
            for i in range(0, obj["width"], 8):
                pygame.draw.line(surf, COLORS["bg"], (obj["x"] + i, obj["y"] + obj["height"]),
                                  (obj["x"] + i + 6, obj["y"]), width=1)
        elif t == "key":
            if obj["id"] in collected:
                continue
            pygame.draw.circle(surf, COLORS["key"], (int(obj["x"]), int(obj["y"])), obj["radius"])
        elif t == "pickup":
            if obj["id"] in collected:
                continue
            pygame.draw.circle(surf, COLORS["pickup"], (int(obj["x"]), int(obj["y"])), obj["radius"])

    if engine:
        px, py = engine.player_body.position
        rect = pygame.Rect(0, 0, PLAYER_W, PLAYER_H)
        rect.center = (int(px), int(py))
        _draw_player(surf, rect)
    else:
        ps = scene["player_start"]
        rect = pygame.Rect(ps["x"], ps["y"], PLAYER_W, PLAYER_H)
        _draw_player(surf, rect)

    if caption:
        _draw_text(surf, caption, (10, 8), size=20)

    return surf


def surface_to_array(surf: pygame.Surface) -> np.ndarray:
    """pygame's array3d is (W,H,3); imageio/GIF wants (H,W,3)."""
    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
