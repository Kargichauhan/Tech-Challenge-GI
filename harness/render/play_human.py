#!/usr/bin/env python3
"""
Human-playable mode: an actual interactive window, not just a solver trace.
Arrow keys to move, space/up to jump.

Jump is triggered on key-down, not "held": physics_engine.step() treats
move and jump as mutually exclusive within a single tick (choosing "jump"
zeroes horizontal velocity for that tick, see navigator.py's docstring for
the full explanation of why). Holding jump down every frame would repeatedly
re-zero horizontal velocity and feel broken; firing it for exactly the one
tick where the key was first pressed (matching how the deterministic solvers
and the Claude navigator both use it) gives a normal one-tap-per-jump feel.

Needs a real display (X11/Wayland/etc.); run locally, not over a headless
SSH session. harness/render/scene_renderer.py forces SDL_VIDEODRIVER=dummy
at import time for headless GIF rendering; we grab a real video driver via
pygame.display.init() *before* importing it, since pygame.display.init() is
a documented no-op on later calls once the display module is initialized,
so scene_renderer's own pygame.init() can't downgrade us back to dummy.

Run:  source .venv/bin/activate && python3 -m harness.render.play_human
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.pop("SDL_VIDEODRIVER", None)
import pygame  # noqa: E402

pygame.display.init()
pygame.font.init()

from harness.engine.physics_engine import Engine  # noqa: E402
from harness.generation.deterministic import generate_valid_scene  # noqa: E402
from harness.render.scene_renderer import render_scene_frame  # noqa: E402

RESULT_MESSAGE = {"success": "YOU WIN", "death": "YOU DIED", "timeout": "TIMED OUT"}


def play(scene: dict) -> str | None:
    engine = Engine(scene)
    w, h = scene["world"]["width"], scene["world"]["height"]
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"playing: {scene['id']}, arrows to move, space/up to jump")
    clock = pygame.time.Clock()

    pending_jump = False
    running = True
    while running and engine.result is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE,):
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_UP):
                    pending_jump = True

        if not running:
            break

        keys = pygame.key.get_pressed()
        if pending_jump:
            action = "jump"
            pending_jump = False
        elif keys[pygame.K_LEFT]:
            action = "move_left"
        elif keys[pygame.K_RIGHT]:
            action = "move_right"
        else:
            action = "noop"

        engine.step(action)
        surf = render_scene_frame(scene, engine)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    if engine.result is not None:
        font = pygame.font.SysFont(None, 64)
        img = font.render(RESULT_MESSAGE[engine.result], True, (240, 240, 245))
        rect = img.get_rect(center=(w // 2, h // 2))
        screen.blit(img, rect)
        pygame.display.flip()
        pygame.time.wait(1500)
    return engine.result


def main():
    driver = pygame.display.get_driver()
    if driver in ("dummy", "offscreen"):
        print(
            f"No real display found (SDL fell back to the '{driver}' driver). "
            "This needs an actual X11/Wayland session (or X-forwarding over SSH: "
            "ssh -X) to show a window. Run it on a machine with a display attached."
        )
        sys.exit(1)

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    combo = ["ramp", "door", "key", "hazard", "pickup"]
    scene, _report = generate_valid_scene(seed, primitive_combo=combo)
    print(f"Playing scene {scene['id']!r}. Arrow keys to move, space/up to jump, Esc to quit.")
    result = play(scene)
    print(f"Result: {result}")
    pygame.quit()


if __name__ == "__main__":
    main()
