# Session Writeup: Infinite Environment Generation via an Agent Harness

A narrative account of what was built for the General Intuition tech challenge,
in the order it happened — including the checkpoints, the bugs found along the
way, and the current state. For the architecture reference itself, see
`README.md`; this document is the story of how it got there.

## The brief and the framing

The challenge: build an agent harness that reliably constructs 2D environments
from text commands and can navigate through them, with an eye toward the
research motivation General Intuition stated directly — infinite generation →
code-level verifiable objectives → reward signal → improved generation. The
explicit ask was to mirror that framing back at them using their own language,
not bolt something unrelated on top.

## The plan we locked in

Before writing code, we agreed on a build order and two checkpoints where I'd
stop and show you the design before it became load-bearing for anything else:

1. Hybrid scene schema — fixed structure + extensible primitive library
2. Event-log schema + objective predicate language — **pause for review**
3. Freeform generation + validator (reachability/overlap, reject-or-repair)
4. Adversarial-probe solver + exploit logging — **pause for review**
5. Deterministic, zero-API-key path as the guaranteed demo
6. Per-environment GIF (scene → trace → verifier-robustness result)
7. Verifier-feedback-driven generation-policy adaptation, built last, degrading
   gracefully if time ran short

Naming discipline was explicit from the start: "generation-policy adaptation"
or "verifier-feedback-driven generation," never "self-improvement" or "RSI."

## Step 1 — The scene schema

Six primitives (`platform`, `ramp`, `door`, `key`, `hazard`, `pickup`) plus
structural `player_start`/`goal_zone` fields present in every scene. Doors lock
via `key_id` matching a key's own `key_id`. Built as a JSON Schema
(`harness/schema/scene_schema.py`) with a validator function, tested against a
hand-built example scene before moving on.

## Step 2 — Event log + objective language (checkpoint)

Six event types (`collision`, `pickup`, `door_open`, `zone_enter`, `zone_exit`,
`death`) and a boolean predicate language (`event`/`and`/`or`/`not`) over that
log. Two decisions were deliberate and flagged for your review before I built
anything on top of them:

- **The objective predicate is order-blind by design** — it checks whether
  events happened, not in what sequence. This isn't a gap to close later, it's
  the actual test surface the adversarial probe (step 4) needed. You confirmed
  this explicitly and told me not to add a `sequence` operator, since that
  would split objectives into two tiers and add design surface for no benefit
  at this stage.
- **`boundary_clip` detection** — you pushed back on my first instinct
  (bbox/trajectory heuristic) and asked for a pymunk-native
  `segment_query_first` check instead, since a proxy heuristic sitting under
  the submission's headline verifier-robustness number would be a credibility
  risk. I agreed and rewrote the design before building the engine on top of
  it.

## Step 3 — Freeform generation + validator

Built the deterministic template generator and the validator pipeline
(structural → referential → overlap → reachability, with a repair pass) in
parallel, since the deterministic generator became the guaranteed demo path.
The reachability check runs a coarse grid model that treats platforms/ramps/
locked-doors as solid and simulates jump arcs.

Bugs found and fixed while getting this to actually validate correctly:

- **Foot-position bug**: the player's start cell was computed from the
  top-left corner instead of the foot line, so a player resting exactly on a
  surface never registered as "standing."
- **Door/ground shared-cell bug**: doors and the ground can occupy the same
  grid coordinate; representing solidity as a flat cell set meant "opening" a
  door also erased the ground's solidity at that cell. Fixed by computing door
  solidity fresh each pass instead of baking it into the static solid set.
- **`primitive_combo=[]` truthiness bug**: `[] or default_combo` evaluates to
  the default in Python since an empty list is falsy — silently ignoring a
  caller's explicit "no primitives" request.
- **Platform-height-0 overlap bug**: the generator's height-offset choices
  included 0, placing a "floating" platform directly on top of the ground.
- **Key-id vs key-object-id mismatch**: the reachability closure tracked
  reachable *keys* by their door-matching `key_id`, but the objective
  predicate references the *object's own* `id` — two different strings the
  code was conflating, causing real levels to be misjudged unreachable.

After these fixes: 180/180 pass across 6 primitive combos × 30 seeds.

## Step 4 — Physics engine

pymunk-based engine executing a scene tick-by-tick, emitting the event log,
and implementing the native `segment_query`-based boundary-clip check agreed
in step 2. Hazards, locked doors, platforms, and ramps are all solid —
matching the approved design (the player is "supposed to be blocked by" all
of them, which is what makes tunneling through one meaningful).

## Step 5 — Genuine solver

A greedy/pathfinding solver sharing the grid model with the validator. This
is where most of the session's real debugging happened, because "the solver
mostly works" isn't the same as "the solver works" — each fix below was found
by actually running batches and reading why specific scenes failed, not by
inspection:

- **Jump-timing bug**: the controller waited until near the *landing* x
  before jumping, meaning it would walk straight through anything (a hazard)
  sitting between takeoff and landing before ever leaving the ground.
- **Fall-drift mismatch**: the grid model's "fall off a ledge" edge assumes a
  straight vertical drop, but the real controller kept moving horizontally
  while falling, landing somewhere the model never checked for hazards.
- **Oscillation from full replanning**: replanning from scratch after every
  single hop caused genuine oscillation, since two equally-short routes in
  opposite directions look interchangeable to a fresh search from a slightly
  different landing spot each time. Fixed by committing to a whole planned
  path and only replanning when a hop's actual landing diverges from what was
  planned.
- **Feet-sensor width deadlock**: the feet-sensor used to detect "am I
  grounded" was narrower than the main collision box. A player barely
  catching the very corner of a platform could be held up by the wider main
  shape while the narrower sensor never registered contact — a genuine
  deadlock (can't jump, can't fall).
- **Rotation-tipping bug**: the player body had no rotation lock, so
  collisions could tip it over, which then misaligned the feet sensor.
- **Jump-arc mismatch (the deepest one)**: the grid model originally
  approximated a jump as a straight line between takeoff and landing. Real
  jumps rise fast early and arc over — the straight-line approximation both
  approved jumps the real physics couldn't make and rejected ones it could.
  Replaced with an actual parabolic simulation using the same constants as
  the real engine (`harness/constants.py`), which removed the whole class of
  mismatch rather than re-tuning approximation constants.
- **Pathfinding performance**: jump simulation was being recomputed from
  scratch on every replanning call. Caching keyed on `held_keys` (which fully
  determines door-lock state) cut a batch that was hanging for 5+ minutes down
  to about a second — this became directly relevant later, since the
  generation-policy adaptation loop depends on many fast probe runs.

## Step 6 — Adversarial solver + exploit classifier (checkpoint)

Built the adversarial solver (goes for whatever satisfies the objective
cheapest, preferring the goal zone outright even before intended
prerequisites) and the exploit classifier (`order_violation`,
`unintended_path`, `boundary_clip`, `none`), then paused for review before
wiring the GIF renderer or generation-policy adaptation on top.

You asked for three specific verifications before signing off, and treated any
failure as a bug to fix, not something to just note:

1. **Prove `boundary_clip` and `unintended_path` actually fire.** Building
   hand-crafted test scenes for this surfaced a real detector bug: pymunk
   fires `begin` once when contact *starts*, not on every tick contact
   continues, so a naive per-tick check flagged ordinary sustained contact
   (sliding down a wall after a normal collision) as a tunneling exploit.
   Fixed by tracking persistent touch state instead of per-tick state. Also
   discovered, via direct velocity-injection testing, that pymunk's
   speculative-contact system actively resists tunneling well past the game's
   actual jump/move speeds — `boundary_clip` is verified correct via a direct
   test, but not reachable through normal gameplay at current speeds. Stated
   plainly rather than oversold.
2. **Correlate exploit findings against genuine-solver reliability.** Across a
   30-scene batch, the two scenes where an exploit was found *without* a
   confirmed honest baseline were both in the same combo known to have lower
   genuine-solver reliability — a real, non-coincidental correlation. Added
   `honest_baseline_confirmed` to the probe's output so downstream reporting
   (GIF captions, aggregate stats) distinguishes "confirmed shortcut past a
   provably-solvable level" from the weaker "adversary succeeded, but nobody
   confirmed anyone could solve this honestly."
3. **Check whether the "already there" nudge fallback inflates adversarial
   success.** My first pass at this check had its own bug — it assumed
   `goal_zone` is always the last-completed requirement, but the objective's
   order-blind AND means a different pickup can finish last. Fixed the check,
   then confirmed: 0 of 12 adversarial successes relied on a nudge-only final
   leg.

## Step 7 — GIF renderer

Headless pygame rendering (off-screen surface, `SDL_VIDEODRIVER=dummy`) into
an imageio-assembled GIF: generated scene → agent trace with live event
captions → a verifier-robustness result screen. Verified visually by
extracting and inspecting individual frames, not just trusting the code ran
without errors.

## Deterministic demo (`run_demo.py`)

Six environments across the primitive library, deterministic seeds, zero
external dependency. Last clean run: 6/6 genuine solves, 2 confirmed
`order_violation` exploits found by the adversarial probe.

## Claude-API upgrade layer

Built after you explicitly chose "build code now, capture the live run
later" when no API key was available in-session:

- `scene_generator.py` — text prompt → scene via a **forced, non-strict**
  tool call. Deliberately not `output_config.format` structured outputs: the
  objective predicate is a recursive schema (self-referential via `$ref`),
  and structured outputs explicitly don't support recursive schemas.
- `navigator.py` — Claude picks one discrete action per decision point from a
  text description of game state, stateless per call (no growing
  conversation) to keep cost and latency bounded regardless of episode length.
- Verified as much as possible without a live key: confirmed both the scene
  generator's and navigator's requests are well-formed by sending them
  against a deliberately invalid key and getting the expected 401 (not a
  local construction error) — proof the request reaches the server correctly,
  just not proof of what Claude does with it.

### What happened when you actually ran it

Two real issues surfaced live, both fixed:

1. **Claude returned the `objective` field as a JSON-encoded string** instead
   of a nested object, consistently across all 4 retry attempts — a known
   tool-calling quirk specifically tied to the one recursive part of the
   schema. Fixed with defensive parsing (`json.loads` if it comes back as a
   string) rather than relying on re-prompting to fix a systematic model
   behavior.
2. **A real reporting bug**: the GIF's result screen and the script's console
   output were both showing the *deterministic solver's* result as if it were
   the outcome of the trace on screen — but the trace shown is Claude's own
   play. In your actual test run, Claude's own navigation made 8 decisions
   (repeated `move_right`, never a `jump`) and almost certainly died in the
   hazard within ~2 seconds, while the unrelated deterministic baseline
   solved the same scene successfully. Fixed by giving the result screen a
   primary "CLAUDE'S NAVIGATION: `<result>`" headline with the deterministic
   comparison clearly subordinate underneath, so a viewer can never conflate
   the two. Verified with a regression test that the original deterministic-
   only demo path is unaffected.

A security note from the same testing round: your real API key ended up
pasted into the chat twice (once in a stray shell line, once in an `export`
command). I flagged it each time and recommended rotating it via the
Anthropic console — worth confirming that's done if it hasn't been.

## Generation-policy adaptation

Confirmed with you upfront: 8 environments/round × 3 rounds, sampling from a
**fixed list of six named primitive combos** (not independent per-primitive
weights, to keep the per-combo signal legible). Score per combo:

```
score = 0.5 · genuine_solve_rate + 0.3 · (1 − exploit_rate) + 0.2 · (1 − validator_rejection_rate)
```

normalized into next round's sampling weights, with a floor so no combo ever
hits zero. A trend chart (matplotlib, dark aesthetic matching the GIF
renderer, dataviz-skill-validated categorical palette) shows genuine-solve-
rate, exploit-rate, and validator-rejection-rate across rounds. In the actual
run, harder combos (the 5-primitive full course, `ramp+hazard+pickup`) got
down-weighted toward the floor while simpler, more reliable combos converged
upward — the mechanism visibly doing what it's supposed to.

## README and current state

Wrote `README.md` covering architecture, the three runnable entry points, the
naming discipline, and — stated plainly rather than hidden — the known
limitations: genuine-solver reliability varies by combo (roughly 50% on the
hardest one), `boundary_clip` is verified-correct but not reachable at
current game speeds, and the Claude-API layer needed a live key to fully
verify (since resolved, per above).

## Investigating the navigator death, and where it landed

You asked me to look into why Claude's own play died within 2 seconds without
ever attempting a jump. That investigation found a real design bug, not just
a prompting gap: the navigator held one discrete action for the entire 0.25s
decision window, and choosing "jump" zeroed horizontal velocity for as long
as it was held — so even a perfectly-timed jump decision could only produce
a vertical hop with zero horizontal movement, physically incapable of
clearing any gap or hazard regardless of prompting. The deterministic solvers
never hit this because they re-decide the action every single tick (jump for
exactly one tick, then move for the rest); the navigator can't do that
per-tick replanning without a tool call per tick.

Fixed by composing a single "jump" decision the same way: one tick of
vertical launch, then the rest of the window continuing whatever horizontal
direction was already in motion — so Claude only has to decide *when* to
jump, not sequence the two-tick maneuver itself. Also added a computed
hazard-proximity hint to the observation text. Verified the fix directly
(bypassing the API): a scripted policy using the same fixed action model
against the *exact* scene that killed the recorded run now clears the hazard
cleanly — real horizontal displacement during the jump, no death event.

What's still open: we checked two earlier recorded runs
(`simple_hazard_jump.json`, `key_door_goal_level.json`) to see if either
already showed a Claude success by coincidence — neither did (one died at the
same hazard, the other fell into a bottomless pit and timed out never having
picked up the key), and both predate the fix besides. So there is currently
no recorded run of Claude successfully completing a level end-to-end via its
own navigation. The decision, stated directly in the README's known
limitations rather than glossed over: disclose the verified-but-unconfirmed
state rather than either withholding the failing recording or waiting
indefinitely for a lucky successful run to cherry-pick.

## The pivot: adopting a larger architecture mid-session

After the cross-check above, you raised wanting something closer to "real
life minecraft or something like a space videogame" — I flagged that as a
significant scope jump and recommended against it as a first instinct, then
you pasted a complete external strategy document ("Ludens") proposing a much
larger architecture: a self-playing game-inventor system modeled on published
open-ended-learning research (OMNI-EPIC, ACCEL, POET, PowerPlay, UED/PAIRED,
PLR), with a MAP-Elites archive, regret scoring, an explore/play/work/exploit
loop, and a raycasting first-person renderer bridging toward the challenge's
stated vision-policy action space — explicitly calling the mechanism
"recursive self-improvement," directly contradicting this session's own
naming rule from step 1.

I flagged both the scope risk and the naming contradiction and asked you
directly rather than picking one silently. Your answers: **full pivot to the
new architecture**, and **adopt the new document's framing** for naming. I
then used Plan Mode before writing any code — one Explore pass over the
existing codebase (confirming my own understanding of `pathfinding.py`,
`probe.py`, and `policy_adaptation.py` against the actual files rather than
memory), one Plan sub-agent scoped specifically to the riskiest new pieces
(the invention loop, regret scoring, the archive), then three clarifying
questions before locking the plan: whether "recursive self-improvement"
should apply everywhere or just the new package (you chose scoping it to the
new package only), whether a corridor-perspective renderer was an acceptable
scope-down from "3D game" (you agreed), and how to pace execution across six
phases (you chose two checkpoints rather than one per phase).

**Guiding decision carried through the whole pivot**: nothing in `engine/`,
`solvers/genuine.py`, `solvers/adversarial.py`, `pathfinding.py`, or
`validator.py` changed. The coarse grid model already simulates real jump
physics and already serves as a fast oracle; the continuous pymunk engine is
the only thing capable of producing `boundary_clip`. Everything new is
additive — a new `harness/invention/` package plus a handful of new modules
alongside the existing ones.

### Phase 1 — small, honest additions to what already existed

A literal interactive human-playable mode (`harness/render/play_human.py`) —
jump fires on key-down, not while held, for the same reason the Claude
navigator composes jump the way it does (holding it the whole tick window
would zero horizontal velocity). Surfaced the validator's existing
`seed_attempts` data as an explicit "valid-yield rate" in the demo's output
rather than tracking it twice. Added one demo scene using the challenge's own
example phrasing ("a table with a can on it — pick up the can"), built from
existing `pickup`+`platform` primitives, no schema change.

### Phase 2 — the Q-learner: the one agent that actually trains

`GenuineSolver`/`AdversarialSolver` are both hand-built and never improve.
`harness/solvers/q_learner.py` is a genuinely trained tabular Q-learner over
the same coarse grid graph, reusing `pathfinding.py`'s own transition
primitives rather than a second physics model. Tested across every combo in
the library, including the hardest 5-primitive one: it reliably converges
from strongly negative early-episode returns to a solved policy, and the
converged policy was verified by actually executing it in the real pymunk
engine (not just trusting the training curve) via the same `do_one_hop`
executor the other solvers share. The learning curve for the hardest combo
goes from roughly -2.0 to +1.0 over 3000 episodes in about 2.5 seconds —
a real, unfaked curve, visually confirmed by reading the rendered chart.

### Phase 3 — the invention loop (checkpoint passed)

Built `harness/invention/`'s full registry (mutation, regret, behavior
keying, novelty pre-filtering, the archive, the explore/play/work/exploit
orchestration) and `run_invention_loop.py`, then — per the agreed
checkpoint — ran the deterministic-only loop (no Claude) for several rounds
and read `runs/invention/report.json` by hand before building anything
further on top. That first real run surfaced a genuine bug: the warm-start
seeding round logged its insert result under the key `"inserted"` while every
later round used `"accepted"` — the loop itself was correct (5 of 6 seed
combos really did fill archive cells), but the printed summary said "0/6
accepted," which would have read as a broken seeding pass if not caught.
Fixed by standardizing the key, then re-verified: coverage grew monotonically
round over round (20% → 44% across seed + 6 rounds), regret values landed in
a sane 0.0–0.35 range, and — notably — a few mutations turned up genuinely
new `order_violation`/`unintended_path` exploits the original six combos
never exposed, which is exactly the kind of thing an invention loop should be
able to find that a fixed-list sampler can't.

With that checkpoint passed, `harness/claude_agent/scene_reviser.py` was
layered on top — structurally identical to the proven `scene_generator.py`
pattern (forced tool call, defensive JSON-string parsing, bounded retry).
Verified the fallback path two ways without spending a real API call: running
`--claude` with no key set (confirmed it prints a clear message and proceeds
deterministically), and a mocked client whose `messages.create` always raises
(confirmed `explore_propose` falls through to a real code-mutation move, not
`None`). A live run against a real key is the user's to do, same as
`run_claude_demo.py` always has been.

### Phase 4 — archive visualization

Invoked the dataviz skill again for a sequential palette, since a heatmap is
a different color job than the categorical trend chart from generation-policy
adaptation. Its reference palette's sequential ramp is documented for a light
chart surface (low value → recedes toward a light background); this repo's
charts are all dark-surface, so the ramp is used **reversed** here — low
fitness maps to the ramp's darkest step (blending toward the dark surface),
high fitness to its lightest step (popping against it) — a stated, reasoned
adaptation, not a guess. Empty archive cells get a distinct flat neutral
gray, not "very dark blue," so a genuine 0.0-fitness cell never looks
identical to "never reached." Confirmed visually by reading the rendered PNG:
empty cells clearly read as "--", filled cells' brightness clearly tracks
fitness.

### Phase 5 — the vision-policy bridge (checkpoint passed)

Before building the raycaster, worked out explicitly why a literal
Wolfenstein-style raycaster doesn't apply to a side-scrolling world (no
second spatial axis to cast a ray fan into) and scoped it honestly instead:
a corridor-perspective first-person view, real depth earned through time
rather than a second axis. Built `raycaster.py`, wired it into `gif_builder.py`
as an opt-in `dual_render` flag (default off, so the existing deterministic
and Claude demos are unaffected), and built `dataset_emitter.py` for the
`(frame, action, reward)` export with an explicit, honest mapping onto the
challenge's action vocabulary (strafe fields present but always 0.0, since
this world has no lateral axis; a heading-flip triggers the one real "look
elsewhere" signal). Per the agreed checkpoint, extracted and visually
inspected sample frames from a real dual-render GIF before attempting the
optional reward model — confirmed the first-person door correctly grows and
switches from locked (orange) to open (green) as the player approaches, with
the hazard visible below it.

For the optional reward-model stretch piece, checked what was actually
installed rather than assuming: neither torch nor scikit-learn is a
dependency in this environment, and installing a deep-learning framework
purely for one explicitly-first-to-cut stretch component wasn't worth the
weight. Built a closed-form linear regression (numpy only) on downsampled
frames instead, and reported the result exactly as measured rather than
tuning it to look better: with only ~74 frames from a single playthrough and
a much larger downsampled feature count, the fit is heavily underdetermined
and the held-out R² is strongly negative. That's disclosed plainly in both
the module docstring and the README's implemented-vs-envisioned table as a
real property of training on one episode, not something a different model
choice would fix.

### Phase 6 — packaging

Extended (not replaced) `README.md` and this writeup with the invention
loop's explore/play/work/exploit mapping, the regret formula, a citations
list for the conceptual basis (stated as conceptual basis, not claimed
implementations), the naming section's explicit scoping (new terminology for
`harness/invention/` only; `policy_adaptation.py` keeps its original,
accurate description), an honest implemented-vs-envisioned table, and the new
bugs/limitations from this phase of the session.

## Post-review fixes and a scope re-check

After the pivot's six phases were built and demonstrated, live use surfaced
two real, separate issues:

1. **The invention loop's `--claude` mode looked hung.** A real run took
   268s with zero visible output beyond the round headers, and the report
   showed 0 of 15 candidates had actually used Claude revision — every one
   silently fell back to deterministic mutation. The root problem was
   `make_reviser`'s error handling: any failure (auth error, timeout,
   validation exhaustion) was caught and swallowed with no trace at all,
   so a 100%-fallback run was indistinguishable from a working-but-unlucky
   one. Fixed by: adding a 60s request timeout so a blocked/unreachable
   network fails within a bounded time instead of the SDK's much longer
   default; adding progress callbacks (`on_round_start`, `on_candidate`,
   `on_candidate_done`) so every candidate prints a definite outcome as it
   finishes, not just on failure; and printing the actual exception or
   rejection reason instead of swallowing it. Verified with the same mocked-
   failure test used earlier (a client whose `messages.create` always
   raises) to confirm the timeout parameter is present and the failure
   reason is now visible, plus a regression run of the deterministic-only
   path to confirm the new callbacks don't change its behavior.

2. **You asked for the first-person view to go further into 2.5D/3D.** Given
   the world genuinely has no depth axis, I laid out three real options at
   very different cost/risk (cosmetic 2.5D dressing on the existing
   corridor view; a schema-level "lane" concept enabling true perpendicular
   walls and real strafing, touching the schema/physics/solvers/Q-learner
   state space; or a full 3D engine rebuild) rather than silently picking
   one, since the second and third would have been substantial, risky
   redesigns of already-verified systems. You chose the cosmetic option.
   Built side wall panels with perspective seam lines, a floor grid
   receding to a vanishing point, and a slowly drifting parallax skyline —
   all drawn first, with every real feature (platforms, hazards, doors, the
   goal zone) drawn on top, so the layering makes clear what's real data
   versus decorative framing. Verified visually via rendered test frames and
   the full dual-render GIF pipeline, plus a regression run of the existing
   exploit-classifier tests and the deterministic demo to confirm nothing
   in the physics/verification path was touched.

## Submission review, a "maze" generator, playable browser demos, and the bigger reward-model run

Before sending this off, you asked for a full cross-check against the
original brief with an eye toward what evaluators actually expect, plus a
review of a competing team's submission you'd found, to see what — if
anything — was worth incorporating.

**The brief cross-check** surfaced one urgent, non-content gap (this
directory had never been git-initialized — no way to share it as a repo)
and several real documentation gaps given how explicit the brief is about
reviewer time ("we will not have hours to review it"): no at-a-glance hook,
no explicit requirements↔deliverable mapping table, and the two playable
browser artifacts weren't mentioned in the README at all despite being the
best 10-second "try this" material in the submission. Fixed all of the
content gaps (added an at-a-glance section, a full requirements-mapping
table, and the artifact links up top); left git alone per your explicit
instruction.

Substantively, the cross-check identified the reward-model bridge as the
weakest of the brief's three named "why this matters" pillars — not fatal,
but the one place our honest number was actually bad rather than just
modestly scoped down, and the brief names it as 1 of exactly 3 core
motivations, not a minor detail.

**The competing repo review** (a different team's "Ludens" submission,
JS/browser-based, top-down grid maze rather than a physics platformer) was
substantially strong — a real trained CNN with credible held-out numbers,
a literal "picked up the can" tile-based objective, one-way tiles as a
genuine rules-level (not just layout) mutation. Also found a real
performance bug in it (their `computeRegret` retrains a fresh Q-learner on
every single render tick via the telemetry HUD) and one documentation
overclaim (the reward model predicts a derived "progress" proxy, not the
literal reward value, despite the README's wording). Declined to port their
world model (a different genre — top-down grid, no gravity — is *why* a
literal raycaster works for them and doesn't for us, not a shortfall on our
side) or their browser-calls-a-local-server architecture. Did port one real
idea: separating "goes in the archive" (diversity, unconditional) from "is
worth building on further" (curriculum, gated) as two explicit checks
instead of one conflated fitness score — added `regret.is_frontier_worthy`
and wired it into `play_select_target`, verified live: a scene with a
confirmed exploit correctly kept its archive cell but was excluded from
being picked as a mutation parent, which the old single-score design
couldn't have distinguished.

**A "maze with lava traps and a locked door" generator**
(`harness/generation/maze.py`) — a winding, zigzagging platform layout using
the existing primitives, not a new schema type (a real 2D branching maze
isn't honest to build in a world with one spatial axis). Hit a real bug
immediately: one height offset was exactly 0, embedding a platform inside
the ground plate, which triggered the validator's repair pass to shove the
*entire ground plane* 1120px sideways — the exact trap `deterministic.py`'s
own comments already warned about. Fixed, then tuned gap sizes until genuine
solve rate across random seeds reached roughly 63%, consistent with (not
worse than) the existing hardest combo's reliability.

**Two playable browser artifacts** followed from a real misunderstanding
worth recording: you asked to see the corridor renderer "moving," and what
was first shown was a single scene generated offline in Python and hand-
copied into a static page — not the live "type anything, get an
environment" experience the brief describes. Clarified this explicitly
rather than letting it stand, then built a genuine local (no LLM, no
network — the artifact sandbox blocks external calls regardless) keyword
compiler that runs entirely in the browser: type a prompt, it parses
keywords onto the same primitive vocabulary, and a lightweight JS/Canvas
physics + corridor-view port makes it immediately playable, with a
reachability safety check before handing back any generated layout.

**The bigger reward-model run** (`run_reward_model_v2.py`,
`train_reward_model_cnn.py`) tested whether the original weak result was a
data-volume problem or a model-class problem, by changing one variable at a
time: 30 solved playthroughs across genuinely different scenes (not one),
held out **by scene**, comparing the same linear method against a real
small CNN (torch, CPU build — checked install size first and used the
~200MB CPU-only wheel instead of the multi-GB default CUDA build, since a
tiny toy CNN doesn't need a GPU stack). The result was more interesting than
a clean "yes, fixed" story: R² turned out to be numerically unstable for
this reward signal (the held-out reward variance is near zero, since most
ticks are just the -0.01 step penalty, so any real error explodes when
divided by a near-zero baseline) — reported plainly rather than swapped for
a flattering metric. MAE, which is scale-stable, told the real story: more
data alone made the *linear* model worse (0.245 average error — it overfits
training scenes' specific pixel content and doesn't transfer to new scene
visuals), while a CNN on the identical bigger dataset achieved a genuinely
small held-out error (0.012). The honest conclusion is that the original
result was as much a model-class problem as a data-volume one — a more
useful, if less tidy, finding than the "more data fixes everything"
hypothesis going in.

## A second, deeper bug in the reward-model dataset, found by checking a number that looked too clean

You asked for a full submission review against the brief, comparing a newer
version of a competing team's repo whose README reports honest baseline
comparisons (MAE improvement over a trivial mean-predictor) for their own
reward model. Rather than assume ours would clear the same bar, I computed
it directly — and it didn't. Both the linear and CNN MAE numbers reported
above turned out to be *worse* than a trivial "always guess the training
mean" baseline (0.0024 MAE), and digging into why revealed the actual
problem: the held-out set was **100% one repeated value**. Every one of 685
held-out frames carried exactly the -0.01 step penalty — no pickup, death,
or success ever showed up in it at all.

The root cause was `dataset_emitter.py`'s sampling: it only captures every
4th tick (`TICK_STRIDE`), and pickup/death/success are single-tick events,
so roughly 3 in 4 of them were silently dropped by construction. Across a
30-scene dataset this meant only 6 success frames and 1 pickup frame
survived in the *training* set, and none at all in the 7 held-out scenes —
making the entire evaluation, as originally run, incapable of measuring
anything beyond "predicts roughly -0.01," regardless of model quality. R²
being unstable (the earlier, first-pass finding) turned out to be a symptom
of this deeper problem, not the whole story.

Fixed by always capturing a frame on any tick with a non-default reward,
regardless of stride, then regenerated the dataset and retrained both
models. The corrected held-out set has real variance (reward std 0.10) and
actual examples of the events that matter. Result: the CNN reaches a
genuine positive R² of 0.32. Its *aggregate* MAE (0.020) still ties the
trivial baseline almost exactly — checked and reported honestly rather than
left implicit — because 685 of 697 held-out frames are still the common
step-penalty case, where a trivial baseline is already good. Breaking the
error out by event type is what actually shows the result: on the 12 rare
frames that are pickup/death/success, the CNN's MAE is 0.303 against 0.621
for blind guessing on the same frames — roughly 2× better specifically on
the events a reward model needs to get right, though inconsistently (some
individual predictions land within 0.1 of the true value, several are
missed entirely). This breakdown is now a permanent part of
`train_reward_model_cnn.py`'s output, not a one-off manual check, so future
runs report it automatically rather than requiring the same investigation
again.

Also folded into this pass: an embedded demo GIF at the top of the README
(the same competing repo's `demo.webp` made the absence of one in ours
obvious) and an explicit section mapping the brief's three "why this
matters" bullets 1:1, in their own order and wording, rather than leaving
that connection implicit across scattered sections.
