"""
Ground-truth pymunk physics engine. Executes a validated scene spec tick by
tick, drives the player via discrete controller-style actions (mirroring the
vision-policy's move-forward/back/left/right action space, adapted to 2D:
move_left/move_right/jump/noop), and appends structured events to the event
log per schema/event_log_schema.md.

Hazards, doors-while-locked, platforms, and ramps are all solid (blocking)
shapes -- per the approved event-log design, the player is "supposed to be
blocked by" all of them, which is what makes a boundary_clip (tunneling
through one) a meaningful exploit rather than expected behavior.
"""

from __future__ import annotations

import math

import pymunk

from harness.constants import DT, JUMP_SPEED, MOVE_SPEED, PLAYER_H, PLAYER_W

MAX_TICKS = 60 * 30  # 30s wall-clock timeout per episode

ACTIONS = ("move_left", "move_right", "jump", "noop")

CT_PLAYER = 1
CT_PLAYER_FEET = 2
CT_SOLID = 3       # platform, ramp
CT_DOOR = 4
CT_HAZARD = 5
CT_KEY = 6
CT_PICKUP = 7
CT_ZONE = 8


def _ramp_poly(obj):
    """Right-triangle poly: 'direction' is which side is the LOW side."""
    w, h = obj["width"], obj["height"]
    if obj["direction"] == "right":
        # low side on the right: high-left corner, sloping down to the right
        pts = [(0, 0), (0, h), (w, h)]
    else:
        pts = [(w, 0), (0, h), (w, h)]
    return pts


class Engine:
    def __init__(self, scene: dict):
        self.scene = scene
        self.space = pymunk.Space()
        self.space.gravity = tuple(scene["world"]["gravity"])
        # The player is driven kinematically (we assign body.velocity every
        # tick rather than applying forces). Pymunk's idle-body auto-sleep
        # can put a stationary body to sleep; a plain velocity assignment on
        # a sleeping body does not necessarily wake/activate it, which can
        # permanently freeze the player mid-level. Disable sleeping entirely.
        self.space.sleep_time_threshold = float("inf")
        self.event_log: list[dict] = []
        self.boundary_clips: list[dict] = []  # diagnostic log, not gameplay events
        self.t = 0.0
        self.tick_count = 0
        self.result: str | None = None  # None (running) | "success" | "death" | "timeout"
        self.grounded_count = 0
        self.held_keys: set[str] = set()
        self.collected_ids: set[str] = set()
        self.zone_inside = False
        self.shape_meta: dict = {}  # pymunk Shape -> {"id","type","obj"}
        self._to_remove: list = []
        # Persistent (not per-tick) touching state. Pymunk fires `begin` only
        # once when contact *starts*, not on every tick contact continues --
        # so a per-tick-only record would treat every tick of legitimate
        # ongoing contact (e.g. sliding down along a wall after a normal
        # collision) as "no collision event this tick," misclassifying it as
        # a boundary_clip. Tracking begin/separate persistently distinguishes
        # "still touching, as pymunk sees it" from "the swept path crossed
        # this shape and pymunk never once recorded contact" (a real clip).
        self._currently_touching: set = set()
        self._build()

    # -- construction ---------------------------------------------------

    def _add_static(self, poly_pts, pos, collision_type, sensor, meta):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = pos
        shape = pymunk.Poly(body, poly_pts)
        shape.sensor = sensor
        shape.collision_type = collision_type
        shape.friction = 1.0
        self.space.add(body, shape)
        self.shape_meta[shape] = meta
        return shape

    def _build(self):
        for obj in self.scene["objects"]:
            t = obj["type"]
            if t == "platform":
                pts = [(0, 0), (obj["width"], 0), (obj["width"], obj["height"]), (0, obj["height"])]
                self._add_static(pts, (obj["x"], obj["y"]), CT_SOLID, False, {"id": obj["id"], "type": t, "obj": obj})
            elif t == "ramp":
                pts = _ramp_poly(obj)
                self._add_static(pts, (obj["x"], obj["y"]), CT_SOLID, False, {"id": obj["id"], "type": t, "obj": obj})
            elif t == "door":
                pts = [(0, 0), (obj["width"], 0), (obj["width"], obj["height"]), (0, obj["height"])]
                shape = self._add_static(pts, (obj["x"], obj["y"]), CT_DOOR, not obj["locked"],
                                          {"id": obj["id"], "type": t, "obj": obj})
            elif t == "hazard":
                pts = [(0, 0), (obj["width"], 0), (obj["width"], obj["height"]), (0, obj["height"])]
                self._add_static(pts, (obj["x"], obj["y"]), CT_HAZARD, False, {"id": obj["id"], "type": t, "obj": obj})
            elif t == "key":
                pts = self._circle_poly(obj["radius"])
                self._add_static(pts, (obj["x"], obj["y"]), CT_KEY, True, {"id": obj["id"], "type": t, "obj": obj})
            elif t == "pickup":
                pts = self._circle_poly(obj["radius"])
                self._add_static(pts, (obj["x"], obj["y"]), CT_PICKUP, True, {"id": obj["id"], "type": t, "obj": obj})

        gz = self.scene["goal_zone"]
        pts = [(0, 0), (gz["width"], 0), (gz["width"], gz["height"]), (0, gz["height"])]
        self._add_static(pts, (gz["x"], gz["y"]), CT_ZONE, True, {"id": "goal_zone", "type": "zone", "obj": gz})

        ps = self.scene["player_start"]
        # Infinite moment of inertia -- torque from asymmetric collisions
        # (e.g. landing slightly off-center) can never rotate the body. Without
        # this the player can tip over, which then misaligns the feet-sensor
        # shape (assumed to point straight down) and breaks grounded detection.
        body = pymunk.Body(1.0, float("inf"))
        body.position = (ps["x"] + PLAYER_W / 2, ps["y"] + PLAYER_H / 2)
        main_shape = pymunk.Poly.create_box(body, (PLAYER_W, PLAYER_H))
        main_shape.collision_type = CT_PLAYER
        main_shape.friction = 0.0
        # Same width as the main box, not inset -- if a corner/edge contact is
        # wide enough to physically hold the main shape up (pinning velocity
        # via contact resolution), the feet sensor must be wide enough to
        # detect that same contact. An inset sensor can miss a "just barely
        # caught the edge" landing that the main shape nonetheless rests on,
        # leaving grounded_count stuck at 0 while gravity is a no-op --
        # a real deadlock (can't jump, can't fall) for the controller.
        feet_shape = pymunk.Poly(body, [
            (-PLAYER_W / 2, PLAYER_H / 2 - 4), (PLAYER_W / 2, PLAYER_H / 2 - 4),
            (PLAYER_W / 2, PLAYER_H / 2 + 4), (-PLAYER_W / 2, PLAYER_H / 2 + 4),
        ])
        feet_shape.sensor = True
        feet_shape.collision_type = CT_PLAYER_FEET
        self.space.add(body, main_shape, feet_shape)
        self.player_body = body
        self.player_shape = main_shape
        self.player_feet_shape = feet_shape

        self._register_handlers()

    @staticmethod
    def _circle_poly(radius, sides=8):
        return [(radius * math.cos(2 * math.pi * i / sides), radius * math.sin(2 * math.pi * i / sides)) for i in range(sides)]

    # -- collision handlers ----------------------------------------------

    def _emit(self, event, id_, meta):
        self.event_log.append({"event": event, "id": id_, "t": round(self.t, 4), "meta": meta})

    def _register_handlers(self):
        def feet_begin(arbiter, space, data):
            self.grounded_count += 1
            return True

        def feet_separate(arbiter, space, data):
            self.grounded_count = max(0, self.grounded_count - 1)

        self.space.on_collision(CT_PLAYER_FEET, CT_SOLID, begin=feet_begin, separate=feet_separate)
        self.space.on_collision(CT_PLAYER_FEET, CT_DOOR, begin=feet_begin, separate=feet_separate)

        def hazard_begin(arbiter, space, data):
            shape = arbiter.shapes[1]
            self._currently_touching.add(shape)
            meta = self.shape_meta[shape]
            self._emit("collision", meta["id"], {"other_type": "hazard"})
            if meta["obj"]["lethal"] and self.result is None:
                self._emit("death", None, {"cause": "hazard"})
                self.result = "death"
            return True

        def hazard_separate(arbiter, space, data):
            self._currently_touching.discard(arbiter.shapes[1])

        self.space.on_collision(CT_PLAYER, CT_HAZARD, begin=hazard_begin, separate=hazard_separate)

        def solid_begin(arbiter, space, data):
            shape = arbiter.shapes[1]
            self._currently_touching.add(shape)
            meta = self.shape_meta.get(shape)
            if meta:
                self._emit("collision", meta["id"], {"other_type": meta["type"]})
            return True

        def solid_separate(arbiter, space, data):
            self._currently_touching.discard(arbiter.shapes[1])

        self.space.on_collision(CT_PLAYER, CT_SOLID, begin=solid_begin, separate=solid_separate)
        self.space.on_collision(CT_PLAYER, CT_DOOR, begin=solid_begin, separate=solid_separate)

        def key_begin(arbiter, space, data):
            shape = arbiter.shapes[1]
            meta = self.shape_meta[shape]
            if meta["id"] not in self.collected_ids:
                self.collected_ids.add(meta["id"])
                self.held_keys.add(meta["obj"]["key_id"])
                self._emit("pickup", meta["id"], {})
                self._to_remove.append(shape)
                self._maybe_open_doors()
            return False

        self.space.on_collision(CT_PLAYER, CT_KEY, begin=key_begin)

        def pickup_begin(arbiter, space, data):
            shape = arbiter.shapes[1]
            meta = self.shape_meta[shape]
            if meta["id"] not in self.collected_ids:
                self.collected_ids.add(meta["id"])
                self._emit("pickup", meta["id"], {})
                self._to_remove.append(shape)
            return False

        self.space.on_collision(CT_PLAYER, CT_PICKUP, begin=pickup_begin)

        def zone_begin(arbiter, space, data):
            self._emit("zone_enter", "goal_zone", {})
            self.zone_inside = True
            return False

        def zone_separate(arbiter, space, data):
            if self.zone_inside:
                self._emit("zone_exit", "goal_zone", {})
            self.zone_inside = False

        self.space.on_collision(CT_PLAYER, CT_ZONE, begin=zone_begin, separate=zone_separate)

    def _maybe_open_doors(self):
        for shape, meta in self.shape_meta.items():
            if meta["type"] == "door" and meta["obj"]["locked"] and not shape.sensor:
                if meta["obj"]["key_id"] in self.held_keys:
                    shape.sensor = True
                    self._emit("door_open", meta["id"], {})

    # -- stepping ---------------------------------------------------------

    def step(self, action: str):
        """Advance one tick. Returns True if the episode is over."""
        if self.result is not None:
            return True

        vx = {"move_left": -MOVE_SPEED, "move_right": MOVE_SPEED}.get(action, 0.0)
        vy = self.player_body.velocity.y
        if action == "jump" and self.grounded_count > 0:
            vy = -JUMP_SPEED
        self.player_body.velocity = (vx, vy)

        prev_pos = self.player_body.position
        self._to_remove = []
        self.space.step(DT)
        for shape in self._to_remove:
            self.space.remove(shape)
            del self.shape_meta[shape]
        curr_pos = self.player_body.position

        self._check_boundary_clip(prev_pos, curr_pos)

        self.t += DT
        self.tick_count += 1

        from harness.schema.objective_eval import evaluate
        if self.result is None and evaluate(self.scene["objective"], self.event_log):
            self.result = "success"
        if self.result is None and self.tick_count >= MAX_TICKS:
            self.result = "timeout"
        return self.result is not None

    def _check_boundary_clip(self, prev_pos, curr_pos):
        """Native continuous-collision check (pymunk segment_query), per the
        approved event_log_schema.md design -- not a bbox/trajectory heuristic.
        If the swept path from prev_pos to curr_pos crosses a shape that is
        currently solid (blocking), and pymunk's own arbiter system does not
        currently consider the player to be touching it (see
        `_currently_touching` -- begin/separate tracked persistently, not
        just this tick's *new* begin events, since pymunk only fires `begin`
        once when contact starts, not on every tick contact continues), the
        player tunneled through it.
        """
        if prev_pos == curr_pos:
            return
        hits = self.space.segment_query(prev_pos, curr_pos, PLAYER_W / 2.0, pymunk.ShapeFilter())
        for hit in hits:
            shape = hit.shape
            if shape is self.player_shape or shape is self.player_feet_shape:
                continue
            meta = self.shape_meta.get(shape)
            if not meta:
                continue
            currently_solid = not shape.sensor
            if currently_solid and shape not in self._currently_touching:
                self.boundary_clips.append({
                    "t": round(self.t, 4), "object_id": meta["id"], "object_type": meta["type"],
                    "prev_pos": tuple(prev_pos), "curr_pos": tuple(curr_pos),
                })

    def run(self, policy, max_ticks: int = MAX_TICKS):
        """policy: callable(engine) -> action string. Runs until done or
        max_ticks. Returns (result, event_log, boundary_clips, tick_count)."""
        while self.result is None and self.tick_count < max_ticks:
            action = policy(self)
            self.step(action)
        if self.result is None:
            self.result = "timeout"
        return self.result, self.event_log, self.boundary_clips, self.tick_count
