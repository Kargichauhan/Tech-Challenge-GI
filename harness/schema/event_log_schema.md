# Event-log schema & objective predicate language

This is the mechanism that everything downstream (validator's semantic pass,
genuine solver's success check, adversarial-probe's target, and generation-policy
adaptation stats) reads from. Reviewing before it's load-bearing.

## 1. Event log

The physics engine runs a scene tick-by-tick and appends structured events to
a flat, ordered list as they occur. Every event is a dict:

```json
{"event": "<event_type>", "id": "<object_or_zone_id>", "t": 4.2333, "meta": {}}
```

- `event`: one of the 6 types below (closed set for now).
- `id`: the id of the object/zone the event concerns. Always present except
  for `death`, where it's `null` (death is about the player, not an object).
- `t`: simulation time in seconds when it occurred (float, monotonic increasing
  across the log -- this ordering is what lets us later detect "did X happen
  before Y" even though the objective predicate itself, see below, does NOT
  check order).
- `meta`: event-specific extra detail, for debugging/GIF captions only --
  never read by the predicate evaluator. e.g. `{"cause": "hazard"}` for death,
  `{"other_type": "hazard"}` for collision.

### Event types

| event        | fires when                                                        | `id` refers to        |
|--------------|--------------------------------------------------------------------|------------------------|
| `collision`  | player's body contacts any solid object (platform/ramp/door/hazard)| the object touched     |
| `pickup`     | player overlaps a `key` or `pickup` object (object is then removed)| the key/pickup object  |
| `door_open`  | a door's `key_id` matches a key that has been picked up            | the door               |
| `zone_enter` | player's body enters `goal_zone`'s rectangle                       | `"goal_zone"`          |
| `zone_exit`  | player's body leaves `goal_zone`'s rectangle after entering it     | `"goal_zone"`          |
| `death`      | player touches a `hazard` with `lethal: true`                      | `null` (see `meta.cause`) |

Notes / deliberate simplifications:
- `door_open` fires once, globally, the instant the matching key is picked up
  -- doors are not proximity-gated. This is intentional: it's the simplest
  rule that still creates a real dependency (door blocks the path until then),
  and it's exactly the kind of rule an adversarial solver should try to route
  around (e.g. "do I even need the door if I can clip past it or go around?").
- `collision` fires for every solid contact, including hazards. A hazard
  collision emits **both** `collision` (id=hazard, meta.other_type="hazard")
  **and**, if lethal, a terminal `death` event. Non-lethal hazards only ever
  emit `collision`.
- `death` is not part of the success objective language (see below) -- it's a
  separate hard-stop signal the engine checks every tick: if a `death` event
  occurs, the episode ends immediately as a **failure**, regardless of what
  the objective predicate says. This keeps "objective" (what does success
  mean) and "failure" (did you die) as two independent, non-overlapping
  concepts, which is what let step 4 target the objective specifically.

## 2. Objective predicate language

An objective is a boolean expression tree evaluated against the **set** of
events seen so far in the log (order-blind by design -- see below for why
that's the point). Four node types:

```json
{"event": "pickup", "id": "key_1"}
{"and": [ <objective>, <objective>, ... ]}
{"or":  [ <objective>, <objective>, ... ]}
{"not": <objective>}
```

`{"event": E, "id": I}` evaluates to true iff some event in the log has
`event == E and id == I`. It does not care when, or how many times, or what
else happened. `and`/`or`/`not` combine sub-predicates the obvious way.

Example -- "pick up the key AND enter the goal zone":

```json
{"and": [
  {"event": "pickup", "id": "key_1"},
  {"event": "zone_enter", "id": "goal_zone"}
]}
```

### Why order-blind is deliberate, not an oversight

The scene above *implies* an order via physical construction (door blocks the
straight path to `goal_zone` until the key is picked up), but the objective
predicate itself only checks "did both events happen, ever." That gap is
exactly what step 4 (adversarial probe) is built to find and report:

1. **Clip through a solid boundary** -- if the physics/collision handling has
   any gap (tunneling at high velocity, a corner case in ramp/door geometry),
   the solver can reach `goal_zone` without ever triggering `door_open`,
   satisfying the `zone_enter` half without the intended dependency.
2. **Reach the zone without doing the intended task** -- if the level has any
   unintended path around the door (geometry gap, jumpable hazard), same
   result via legitimate movement rather than a physics bug.
3. **Satisfy the AND out of intended order** -- e.g. enter `goal_zone` first
   (if that's physically reachable before the door even matters in this
   layout), then wander back for the key. The AND is satisfied; the designer's
   intended causal chain (key unlocks door unlocks path to zone) never
   mattered. This is a predicate-language exploit, not a physics exploit --
   it'd survive even a perfect collision engine.

The validator (step 3) checks the level is *reachable at all*; it does not
check that the *only* path respects intended ordering -- that's explicitly
left as the adversarial probe's job, not folded into validation, so the two
stay conceptually separate (validator = "is this level playable," probe = "is
the objective predicate gameable").

### What the adversarial-probe logs on success

Per the plan, whenever the adversarial solver satisfies the objective, we log
enough to inspect the exploit, not just a pass/fail bit:

```json
{
  "scene_id": "demo_1",
  "adversarial_success": true,
  "event_log": [ ... full log ... ],
  "exploit_class": "order_violation" | "unintended_path" | "boundary_clip" | "none",
  "explanation": "zone_enter occurred at t=1.2s, before pickup(key_1) at t=3.4s -- goal reached before key was collected"
}
```

`exploit_class` is assigned by a small rule-checker that compares the
adversarial run's event order/path against the scene's *intended* dependency
graph (derived directly from the scene spec: door.key_id -> key, goal_zone
behind door). Concretely (this is the actual check, not just examples):

- `order_violation`: the success-relevant events occurred, but a `door_open`
  (or `pickup` of a key referenced by a door on the shortest intended path)
  happened *after* the `zone_enter` that the objective required -- i.e., the
  AND was satisfied out of the sequence the scene geometry implies.
- `unintended_path`: `zone_enter` occurred without the corresponding
  `door_open` ever appearing in the log at all (door never needed).
- `boundary_clip`: detected natively via pymunk, not inferred from a
  trajectory/bbox heuristic. Every tick, for every solid object the player is
  supposed to be blocked by (door while locked, hazard, platform/ramp), the
  engine runs `space.segment_query_first(prev_pos, curr_pos, ...)` between the
  player's previous and current tick position against that object's shape. If
  it reports a hit (the player's continuous path crossed the solid geometry)
  but the discrete collision handler never fired a `collision` event for that
  object on that tick, the player tunneled through it -- flag `boundary_clip`
  and record which object/tick. This was chosen over a bbox-jump proxy
  because it has no tunable threshold and no false-positive/false-negative
  band: either the segment query intersects the shape or it doesn't, and
  pymunk exposes it natively so it costs a couple of lines per tick, not a
  re-architecture. Given this is one of three exploit classes feeding the
  submission's headline verifier-robustness score, the native check earns its
  keep over the cheaper proxy.
- `none`: adversarial solver failed to satisfy the objective, or succeeded via
  the same route a genuine solver would take (no exploit found).

## 3. Where this plugs in later (not built yet)

- **Generator (step 3)**: writes `objective` using this predicate language,
  and must ensure every `id` referenced exists in `objects`/`goal_zone`
  (semantic validation, separate from JSON-schema structural validation).
- **Engine (step 3/4)**: appends to the event log per the table above, and
  runs the per-tick `segment_query_first` continuity check (prev-tick pos ->
  curr-tick pos, against every currently-solid shape) needed for
  `boundary_clip` detection -- this must be built into the tick loop itself,
  not bolted on after, since it needs both positions before pymunk's own
  discrete step overwrites `prev_pos`.
- **Genuine solver**: walks toward whatever the objective's unmet leaves
  require, ignoring exploits.
- **Adversarial solver (step 4)**: explicitly searches for the 3 exploit
  classes above.
- **Generation-policy adaptation (step 7)**: reads `exploit_class` rates per
  primitive combo to down-weight combos that are frequently gamed.
