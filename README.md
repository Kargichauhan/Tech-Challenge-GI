# Infinite Environment Generation via an Agent Harness

### ▶ [**Live, playable dashboard — click and play, no login, no setup**](https://kargichauhan.github.io/Tech-Challenge-GI/)

Type a text prompt, get a generated environment, walk through it in first-person, right in your browser. Hosted permanently from this repo's [`docs/index.html`](docs/index.html) via GitHub Pages — a public URL, not a login-gated link.

**At a glance** (full detail below):
- A **real physics engine** (pymunk), not prompt-level tricks — gravity, jump arcs, collision.
- Text → a validated, playable environment in **one command, no API key needed**.
- A second agent **actively tries to reward-hack the win condition** — and sometimes succeeds; every scene reports whether it was gamed and how (this is the verifier-robustness problem reward-model training lives and dies on).
- Three different agents play the same levels: a hand-built honest solver, a hand-built adversarial solver, and a **tabular Q-learner that trains from scratch** with a real learning curve.
- An **invention loop** that mutates/revises scenes on its own (including changing the objective's shape, not just the layout) and keeps a MAP-Elites archive of what it finds.

**Run it** (guaranteed path — no API key, no network):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pymunk pygame imageio jsonschema networkx matplotlib anthropic
python3 run_demo.py
```

Generates 7 environments across the primitive library, runs the genuine solver and the adversarial verifier-probe on each, and writes one self-contained GIF per environment to `runs/demo/*.gif` (generated scene → agent trace → verifier-robustness result, readable in well under 30 seconds). Fully deterministic, no external dependency — this is the path reviewers should run.

*Everything below this line is detail — the requirements mapping, architecture, and honestly-reported results, in case you want to go deeper.*

---

## Requirements ↔ what we built

| The brief asked for | What's here |
|---|---|
| Harness that constructs environments from text commands | `run_claude_demo.py "<prompt>"` — Claude reads your exact text, emits a scene via forced tool-call; `harness/generation/maze.py`/`deterministic.py` — code-based generators, no LLM required |
| Playable in a game or physics engine | `harness/engine/physics_engine.py` — real pymunk simulation (gravity, jump arcs, collision), not an approximation |
| Agent maneuvers through generated environments | Three independent agents solve the same schema: `solvers/genuine.py` (honest), `solvers/adversarial.py` (exploit-seeking), `solvers/q_learner.py` (trained from scratch) — plus Claude itself via `claude_agent/navigator.py` |
| "Models like Claude perform well at progressing through 2D environments on their own" (the brief's own stated reason for starting in 2D) | Exactly this: `navigator.py` gives Claude a text description of state per decision point (no pixels), and it navigates — directly matching that reasoning, not just citing it |
| Post-training environments (a supply of diverse, verified environments) | `harness/generation/policy_adaptation.py` (weighted resampling) + `harness/invention/` (an invention loop that mutates/revises scenes and archives the results via MAP-Elites) |
| Code-level objectives ("picked up the can from the table") | `harness/schema/`'s event-log + boolean-predicate objective language — checked in code, never guessed from pixels. The exact brief example is a real demo scene in `run_demo.py` |
| Reward model bridge (code-truth → pixel understanding) | `harness/render/dataset_emitter.py` exports real `(frame, action, reward)` tuples in the challenge's own action vocabulary; a real torch CNN reaches held-out **R²=0.32** and roughly halves the error on the rare success/pickup events that actually matter, versus blind guessing — see below for the full, honestly-reported story, including a real sampling bug found and fixed along the way |
| Vision-policy's action space (move fwd/back/left/right, mouse ΔX/ΔY) | `harness/render/raycaster.py` — a first-person view + `dataset_emitter.py`'s explicit mapping onto that exact vocabulary |
| Creativity | The adversarial exploit classifier, the invention loop's objective-shape mutations, the MAP-Elites archive, and the live playable browser dashboard above |
| Clarity / working output | Seven runnable entry points (table below), all verified working, guaranteed path needs no API key |

## Alignment with the brief's "Why This Matters" (verbatim structure)

The brief names exactly three reasons this problem matters to GI's research. Mapped 1:1, in their own order:

1. **Post-training environments** — a supply of diverse environments for training/evaluating a vision policy on specific goals and rewards.
   → `harness/generation/policy_adaptation.py` (weighted resampling over verified combos) **and** `harness/invention/` (a MAP-Elites archive that mutates/revises scenes on its own — the piece we call the "open-ended invention loop," scoped to this package; see Naming below). Both are runnable end to end with no API key.

2. **Code-level objectives** — e.g. "successfully picked up the can from the table," verified in code, not guessed from pixels.
   → `harness/schema/`'s event-log + boolean-predicate objective language. The brief's exact example is a real scene in `run_demo.py` (a `pickup` primitive on a `platform`, objective `{"and": [zone_enter, pickup]}` — checked against the engine's structured event log, never a VLM).

3. **Reward model training** — code-space signals bridging to pixel-based observation.
   → `harness/render/dataset_emitter.py` exports real `(frame, action, reward)` tuples in the challenge's own action vocabulary. Two reward-model passes were run against this data — see "The reward model" below for the full, honestly-reported result, including a real methodology bug we found and fixed mid-session rather than papering over.

A 2D physics-based agent harness: text commands become playable environments, an
agent (deterministic solver, a trained-from-scratch Q-learner, or Claude itself)
navigates them, and a second agent adversarially probes whether the "win"
condition can be gamed. This mirrors the research loop the challenge itself
describes:

```
infinite generation  →  code-level verifiable objectives  →  reward signal (verifier robustness)  →  improved generation
```

Concretely: a scene generator (deterministic, Claude-driven, or code-mutated)
produces levels → every level's objective is a boolean predicate over a
structured event log, not pixels, so success/failure is checkable in code → an
adversarial solver tries to satisfy that predicate as cheaply as possible,
producing a **verifier-robustness score** per environment → those scores feed
back in two ways, at two different scales:

- a weighted-sampling distribution over a **fixed list** of named primitive
  combinations ("**generation-policy adaptation**" — see Naming below), and
- an **invention loop** that mutates/revises scenes freely (not just resampling
  a fixed list), scores them by regret and exploit-resistance, and keeps the
  best-found scene per behavior cell in a MAP-Elites archive — called the
  "open-ended invention loop" in this submission, scoped to exactly that new
  package (see Naming below).

Both mechanisms are real, both are runnable with no API key, and neither
replaces the other — the fixed-combo sampler stays exactly as accurate a
description of itself as it always was.

## The runnable entry points

| Script | What it does | Requires |
|---|---|---|
| `run_demo.py` | The guaranteed demo. Deterministic generation, genuine + adversarial solve, GIF per environment. | nothing but the venv |
| `run_claude_demo.py "<prompt>"` | The upgrade layer: Claude generates the scene from your text prompt *and* navigates it itself (tool-call per decision point), then the same verifier probe runs for comparison. | `ANTHROPIC_API_KEY` |
| `run_generation_adaptation.py` | Runs 3 rounds × 8 environments over a fixed set of named primitive combos, scores each combo, updates a sampling distribution, and plots the trend. | nothing but the venv |
| `run_q_learning_demo.py [seed]` | Trains a tabular Q-learner from scratch against a scene, plots the learning curve, then plays the *trained* policy for real and renders it. | nothing but the venv |
| `run_invention_loop.py [--claude]` | The invention loop: warm-starts from the fixed combos, then mutates/evaluates/archives new scenes round by round (MAP-Elites). `--claude` layers Claude scene revision on top, falling back to code mutation automatically. | nothing but the venv (`ANTHROPIC_API_KEY` only for `--claude`) |
| `run_maze_demo.py [seed]` | The "maze with lava traps and a locked door" pattern — a winding, zigzagging platform layout instead of the simple generator's flat line. | nothing but the venv |
| `run_dataset_export.py [seed]` | Renders a dual (top-down + first-person) GIF and exports a (frame, action, reward) dataset in the challenge's action vocabulary, plus a linear reward-model fit on that one episode. | nothing but the venv |
| `run_reward_model_v2.py [--fresh]` | The bigger reward-model run: many playthroughs across many scenes, held out by scene, comparing the same linear method against a real small CNN. | `torch` (CPU build: `pip install --index-url https://download.pytorch.org/whl/cpu torch`) |
| `python3 -m harness.render.play_human` | An actual interactive window — arrow keys to move, space/up to jump. Needs a real display (X11/Wayland), not a headless SSH session. | nothing but the venv + a display |

The submission does not depend on `run_claude_demo.py` (or `run_invention_loop.py
--claude`) working during review — both are optional upgrade layers, exercised
and verified to reach the live API correctly (see harness/claude_agent/), but the
guaranteed deliverable is the
deterministic path.

## Architecture

```
harness/schema/           hybrid scene schema (fixed structure + extensible primitive
                           library) + event-log schema + objective predicate language
harness/generation/        deterministic template generator, freeform-with-validator
                           pipeline (reachability/overlap checks, reject-or-repair),
                           generation-policy adaptation (weighted sampling over combos),
                           coarse grid model (pathfinding oracle, shared by everything
                           below), MAP-Elites archive heatmap + coverage trend charts
harness/engine/            pymunk physics engine: executes a scene, emits the event
                           log, detects boundary-clip tunneling natively (segment_query)
harness/solvers/           genuine solver (honest, intended order), adversarial solver
                           (cheapest route, ignoring intended order), and a tabular
                           Q-learner (the one agent that actually trains from scratch)
                           — share hop-execution machinery in common.py
harness/verifier/          runs solvers on the same scene, classifies *how* an
                           adversarial success gamed the objective (or didn't)
harness/invention/         the invention loop: code-based scene mutation, regret +
                           exploit-severity scoring, behavior-space keying, novelty
                           pre-filtering, and the MAP-Elites archive itself
harness/render/            headless pygame renderer + GIF assembly, an interactive
                           human-playable mode, a corridor-perspective first-person
                           renderer, and a (frame, action, reward) dataset exporter
harness/claude_agent/      the Claude-API upgrade layer: scene generation and
                           Claude-driven navigation (as before), plus scene_reviser.py
                           (Tier-2 LLM revision for the invention loop)
```

### The scene schema

Six primitives: `platform`, `ramp`, `door`, `key`, `hazard`, `pickup`, plus
structural `player_start` and `goal_zone` fields present in every scene. Doors
lock/unlock via `key_id` matching; a locked door blocks a player, an unlocked one
doesn't. See `harness/schema/scene_schema.py`.

### The event log and objective language

The engine emits a flat, timestamped log of structured events —
`collision`, `pickup`, `door_open`, `zone_enter`, `zone_exit`, `death` — and an
objective is a boolean predicate over that log: `{"event": ..., "id": ...}`
combined with `and`/`or`/`not`. This is deliberately **order-blind**: it checks
whether events happened, not in what sequence. That's not an oversight, it's the
test surface step 4 (the adversarial probe) is built to exploit. Full design
rationale, including why `death` is decoupled from the objective predicate and
why `boundary_clip` uses a native pymunk `segment_query` check rather than a
trajectory heuristic, is in `harness/schema/event_log_schema.md`.

### Genuine vs. adversarial solvers

Both solvers plan routes with a coarse grid model (`harness/generation/pathfinding.py`)
that **simulates the actual jump physics** (same constants as the real engine,
`harness/constants.py`) rather than approximating it — an earlier straight-line
approximation let the grid approve jumps the real physics couldn't make, which
surfaced as several real bugs during development (see Known limitations).

- **Genuine solver** (`solvers/genuine.py`): collects whatever the objective
  requires in the order a human would — keys before doors before the goal zone.
- **Adversarial solver** (`solvers/adversarial.py`): at every step, goes for
  whichever unmet requirement is *cheapest right now*, preferring the goal zone
  outright whenever it's reachable at all — even before intended prerequisites.
  When no legitimate path exists, it rams blindly toward the goal in case the
  discrete physics step ever tunnels through something (`boundary_clip`).

### Verifier robustness

This is the "reward signal (verifier robustness)" arrow in the causal chain
at the top of this document, made concrete: `harness/verifier/probe.py` runs
both solvers fresh on the same scene and `harness/verifier/exploit_classifier.py`
labels what happened — this per-scene label is the actual signal that feeds
both generation-policy adaptation and the invention loop's regret scoring
below, not just a diagnostic printed and discarded.

| `exploit_class` | Meaning |
|---|---|
| `order_violation` | Objective satisfied, but a required event happened out of the order the level's geometry implies (e.g. reached the goal before opening its door). |
| `unintended_path` | Goal reached without ever interacting with a door that appears to gate it. |
| `boundary_clip` | The physics engine's own tunneling — swept path crossed solid geometry with no collision ever registering. Verified as a real, working detector via injected-velocity tests (see `harness/tests/test_exploit_classes.py`), but not reachable through the current action space's normal speeds — pymunk's speculative-contact system resists tunneling well past our jump/move speeds. Kept as a correct safety net, not oversold as an active exploit path today. |
| `none` | No exploit found. |

Every probe result also carries `honest_baseline_confirmed` — whether the
*genuine* solver actually solved the scene. An exploit found on a scene the
genuine solver couldn't confirm solvable is flagged as a weaker claim (there's
no proven-honest baseline the adversary shortcut past), not silently reported
the same as a confirmed one. Both the GIF result screen and the batch report
carry this distinction.

### Generation-policy adaptation

Per round: sample `N=8` scenes from a **fixed list of named primitive
combinations** (not independent per-primitive weights — deliberately, to keep
the signal per combo legible) according to current weights, run the full
genuine+adversarial probe on each, then score each combo:

```
score = 0.5 · genuine_solve_rate + 0.3 · (1 − exploit_rate) + 0.2 · (1 − validator_rejection_rate)
```

normalized into next round's sampling distribution, with a floor so no combo's
weight ever hits zero. `harness/generation/policy_adaptation.py`. This is a
plain weighted-sampling update — never code self-modification.

### The tabular Q-learner — the one agent that actually trains

`GenuineSolver` and `AdversarialSolver` are both hand-built: they play a fixed
strategy on attempt one and never improve with practice. `harness/solvers/q_learner.py`
is the exception — a genuinely trained agent. State is `(cell, frozenset(collected_ids))`
over the same coarse grid `pathfinding.py` already builds; actions are
`move_left`/`move_right`/`jump_left`/`jump_right`; every transition reuses
`pathfinding.py`'s own primitives (`_is_standing`, `_fall_target`, `_simulate_jump`),
not a second physics approximation. Reward comes from re-evaluating the *whole*
objective predicate after every step — unlike the genuine solver, this agent has
no notion of "targets" or intended order at all; it discovers the dependency
order itself through trial and error. Training is fast (order-2500 episodes,
low single-digit seconds) because the state space is the coarse grid, not raw
per-tick physics. `run_q_learning_demo.py` trains one from scratch on a scene,
plots the learning curve, then executes the converged policy for real in the
pymunk engine (via the same `do_one_hop` hop executor the other solvers use) and
renders it. In testing, it reliably converges from strongly negative early
returns to a solved policy across every combo tried, including the hardest
5-primitive one.

## The invention loop: explore, play, work, exploit

Generation-policy adaptation (above) only ever resamples a **fixed list** of six
named combos — it can make "5-primitive full course" more or less likely, but it
can never invent a scene that isn't already on that list, and it never touches
the *shape* of the objective predicate. `harness/invention/` is a second,
independent mechanism that can: it mutates or Claude-revises scenes freely, keeps
a standing archive of the best-found scene per behavior cell, and grows that
archive round over round. This is the piece called the
**"open-ended invention loop"** — see Naming below for exactly what that
scoping means.

Loosely modeled on published open-ended-learning research — ACCEL, POET, PLR,
and UED/PAIRED for the mutate-and-keep-the-hardest-still-solvable idea; POET and
PowerPlay for pairing environment generation with a standing archive of solved
tasks; OMNI-EPIC for the idea of an explicit "inventor" proposing tasks rather
than sampling from a fixed distribution; MAP-Elites (Mouret & Clune) for the
quality-diversity archive itself; Eureka, Voyager, and ADAS/Meta-Agent-Search
for the general shape of an LLM proposing/revising artifacts that get scored by
code rather than by the LLM's own judgment. None of these are implemented as
described in their papers — they're the conceptual basis for a much smaller,
purpose-built version of the same idea, not a reproduction.

- **explore** (`harness/invention/mutation.py`, `novelty.py`) — a fixed registry
  of code-based moves (add/remove/reposition/resize a primitive, add or remove a
  key-door dependency, swap wholesale to a different combo, *and* two moves that
  touch the objective's shape: wrap it in an `or` for an alternate win condition,
  or add a `not` clause requiring a specific hazard never be touched). This last
  pair is the actual new capability — `policy_adaptation.py`'s combos never vary
  the predicate's shape, only which primitives appear. A cheap structural-Jaccard
  pre-filter (`novelty.py`) skips near-duplicates before spending an expensive
  verifier probe on them. Rejected mutations retry with a different move
  (bounded ladder) rather than being discarded; every candidate goes through the
  exact same `validate_and_repair` a freshly-generated scene does.
- **work** (`harness/invention/regret.py` + the existing `verifier/probe.py`,
  unchanged) — `oracle_par_ticks` is a closed-form, cheap estimate (no solver
  execution at all) of near-optimal tick count, reusing `find_path`'s own
  planning step and converting each edge into a tick estimate from
  `harness/constants.py`. Regret is `max(0, (agent_ticks - par) / par)` using the
  genuine solver's real tick count (or the Q-learner's converged episode length,
  where available, as a more classically-faithful "agent return"). **Scoping
  note, stated in the module docstring too**: classic UED regret is
  (oracle return − learner return), which presumes a trainable policy. Most of
  this repo has no trainable policy — this is a repurposed execution-gap metric,
  honest about standing in for the real thing rather than claiming to be it.
- **play** (`harness/invention/loop.py`'s `play_select_target`) — picks which
  archive cell to grow next, biased toward the frontier (empty or weakest-fitness
  cells) rather than polishing an already-strong occupant. This is also where
  `harness/claude_agent/scene_reviser.py` plugs in: same forced-tool-call pattern
  as `scene_generator.py`, revising a parent scene toward more interesting
  difficulty instead of generating from scratch, falling back to code mutation
  automatically on any failure (no key, API error, or exhausted retries) — the
  loop always completes without a key, exactly like `run_claude_demo.py`.
- **exploit** (`harness/invention/archive.py`) — a MAP-Elites archive keyed on a
  **2D grid** (path-length bin × interaction-count bin). A true 4D grid (adding
  rule-shape and irreversibility-count as literal axes, per the original
  strategy doc) would be almost empty at the batch sizes this loop actually
  runs — a scoping decision stated here rather than silently under-delivered.
  Rule-shape signature and irreversibility count are still computed, just kept
  as per-occupant *metadata* instead of grid axes. Fitness rewards moderate,
  frontier-appropriate regret (not zero, not huge) and penalizes exploitability,
  so a cell's occupant is "hard to solve honestly *and* hard to shortcut," not
  just the hardest-looking scene found.

`run_invention_loop.py` warm-starts the archive from `policy_adaptation.py`'s six
combos (the existing, simpler mechanism becomes this one's round-0 seed set,
not something it replaces), then runs explore/work/play rounds, writing
`runs/invention/report.json` plus an archive heatmap and a coverage-over-rounds
trend chart (`harness/generation/archive_heatmap.py`).

## Vision-policy bridge: first-person renderer + dataset export

The challenge's own framing is pixel-observation, first-person action space
(move forward/back/left/right + mouse look). This repo's argument throughout
has been that a 2D physics world with a structured event log doesn't need that
— but the challenge also gestures at a reward model learned from pixels, so
`harness/render/raycaster.py` and `dataset_emitter.py` build a genuine,
honestly-scoped bridge rather than ignoring that half of the brief.

**Why not a real raycaster**: a Wolfenstein-style raycaster casts a fan of rays
into a 2D top-down map of walls and reads depth off where each ray first hits
one — it needs a real second spatial axis (left/right) with actual geometry in
it. This world doesn't have one; it's a side-scrolling platformer, one
horizontal axis plus gravity. A literal port of the algorithm would have every
ray in the fan hit the same point. What's built instead is a **corridor-
perspective first-person view**: a single forward-looking view along the
player's current heading, where objects come into and out of view as the
player advances — real depth, earned through time (approaching an object)
rather than a second spatial axis at a single instant. Each object is
perspective-scaled (nearer = larger) and vertically positioned by its actual
world height relative to the player's eye line.

On top of that real content, a layer of **cosmetic 2.5D dressing** — side
wall panels with perspective seam lines, a floor grid receding to the
vanishing point, and a slowly drifting parallax skyline — makes the view read
much more like an actual corridor/hallway. This is framing, not a claim about
world geometry: there are no real side walls anywhere in this world, and the
dressing is drawn first, with every real feature (platforms, hazards, doors,
the goal zone) drawn on top of it — a strict, visible layering order, so
what's real vs. decorative framing is never ambiguous to a viewer.

`run_dataset_export.py` renders a dual (top-down + first-person) GIF side by
side and exports a per-tick `(frame, action, reward)` dataset in the
challenge's action vocabulary. The mapping is a labeled, approximate bridge,
not a claim of equivalence (this world has no lateral strafe axis and no
camera that turns independently of movement) — see `dataset_emitter.py`'s
docstring for the exact, honest mapping.

### The reward model: two runs, an honest finding

**First pass** (`harness/render/train_reward_model.py`): closed-form linear
regression on downsampled frames from a single ~74-frame playthrough — no new
dependency at all. Heavily underdetermined (far more pixel features than
training examples); disclosed as such rather than dressed up.

**Second pass** (`run_reward_model_v2.py`, `harness/render/train_reward_model_cnn.py`):
tests whether that was a data problem or a model problem, by changing one
variable at a time. Generates 30 solved playthroughs across genuinely
different scenes, holds out 7 scenes' worth **by scene** (never seen in
training — a real generalization test), then compares the same linear method
against a small CNN (torch) on identical data.

The first version of this run surfaced a second, more serious bug before it
surfaced any result worth reporting: `dataset_emitter.py` only samples every
4th tick, and pickup/death/success are single-tick events — so almost every
one of them was silently missed by the sampling. The held-out set that first
run produced was **100% one repeated value** (every one of 685 frames was
exactly the -0.01 step penalty), which made R² meaningless in either
direction, not just noisy. Fixed by always capturing a frame on any tick with
a non-default reward, regardless of stride, then regenerated:

| | held-out MSE | held-out MAE | held-out R² |
|---|---|---|---|
| Same linear method, more data | 40.4 | 0.565 | -3,963 |
| A small CNN (torch), same data | 0.0069 | 0.020 | **0.32** |

With real variance in the held-out set (reward std 0.10, not ~0), R² is now
a legitimate metric, and the CNN's 0.32 is a genuine, positive result on
scenes it never trained on. The linear model, tested the same way, confirms
what the first (degenerate) run couldn't show cleanly: more data alone does
*not* fix it — raw-pixel linear regression overfits training scenes'
specific visual content and gets *worse*, not better, on genuinely new scene
visuals.

The CNN's aggregate MAE (0.020) is almost identical to a trivial
"always guess the training mean" baseline (0.020) — worth stating plainly,
not glossing over. But that aggregate number is dominated by the 685 of 697
held-out frames that are just the common step penalty, where both the model
and the trivial baseline are already good. Broken out by event type instead:
on the 12 rare frames that actually matter (pickup/death/success), the CNN's
MAE is 0.303 versus 0.621 for blind guessing — roughly **2× better on
exactly the events a reward model needs to get right** — though
inconsistently (some individual predictions land within 0.1 of the true
value, a few are missed entirely). Reported exactly as measured: a real,
if partial and honestly-qualified, signal — not the clean story either
"more data" or "a better model" alone would have suggested going in.

## Naming

**"Generation-policy adaptation"** or **"verifier-feedback-driven generation"**
— never "self-improvement" or "RSI" — describes `harness/generation/policy_adaptation.py`
exactly as accurately today as it always did: a sampling distribution over a
fixed combo list, reweighted from measured stats. That description doesn't
change just because a second, more ambitious mechanism exists alongside it.

**"Open-ended invention loop"** is used for `harness/invention/` specifically:
the invention loop invents scenes outside any fixed list, scores them by
regret and exploit-resistance against a standing archive, and grows that
archive round over round without human intervention between rounds. It is
still bounded — a fixed move registry, a fixed archive shape, no code
executes or modifies itself — so it's named for exactly what it does ("the
generator's own output feeds back into what the generator tries next,
without a human picking the next target") rather than reached for a bigger
term the mechanism doesn't need.

## Known limitations (stated plainly, not hidden)

- **Genuine-solver reliability varies by combo.** The 5-primitive combo
  (`ramp+door+key+hazard+pickup`) solves honestly in roughly half of random
  seeds — the hardest combo in the library, since it stacks every failure mode.
  Simpler combos (`hazard` alone, bare platforming) solve close to 100%. This is
  disclosed rather than cherry-picked around; `honest_baseline_confirmed`
  exists specifically so downstream reporting doesn't overclaim on the harder
  cases.
- **`boundary_clip` is a verified-correct but currently dormant exploit path** —
  see the table above. It's real (proven via direct injected-velocity tests),
  just not reachable at the game's actual jump/move speeds because pymunk's
  speculative contacts resist tunneling well past that range.
- **The Claude-driven navigation layer surfaced a real design bug in how
  discrete jump actions compose with continuous physics** — captured and
  fixed during development (see `harness/claude_agent/navigator.py`'s module
  docstring for the full explanation: a single "jump" decision held for the
  whole 0.25s window zeroed horizontal velocity for that entire window,
  making it physically impossible to clear a gap or hazard no matter how well
  timed, independent of any prompting). The attached recording
  (`runs/claude_demo/`) shows Claude's own play failing at a hazard from
  before this fix. We verified via direct scripted testing (bypassing the
  API entirely) that the underlying jump mechanics are now physically
  correct post-fix — a scripted policy using the same fixed action model
  clears the identical hazard that killed the recorded run — but a single
  live run isn't enough to confirm Claude's own decision-timing reliably
  clears it in practice. We're disclosing this rather than cherry-picking a
  lucky run: consistent with this submission's overall emphasis on honest
  verification over polished-looking results.
- **The Q-learner's failed-jump simplification.** `_simulate_jump` returns
  "no landing" both when an arc grazes solid geometry (survivable in the real
  engine — you bonk and fall back) and when it grazes lethal geometry (fatal).
  The Q-learner's training environment doesn't distinguish these — a failed
  jump there is conservatively treated as a wasted move, not death — but the
  trained policy is executed for real afterward, where the real engine's own
  hazard detection is authoritative regardless of what training assumed.
- **The regret metric is a repurposed execution-gap proxy, not classic UED
  regret.** Classic regret is (oracle return − learner return), which
  presumes a trained policy on both sides. Only the Q-learner is actually
  trained in this repo; regret elsewhere compares a real solver run against a
  cheap closed-form estimate, not two learned policies.
- **The MAP-Elites archive is 2D, not the full 4D described in the original
  strategy document.** Rule-shape and irreversibility-count are computed and
  reported as per-occupant metadata, not grid axes — a true 4D grid would be
  almost empty at the batch sizes this loop actually runs.
- **The first-person renderer is a corridor-perspective view, not a full
  raycasting maze-caster.** This world has one true spatial axis; there's no
  second axis to cast a ray fan into. See "Vision-policy bridge" above.
- **A real sampling bug hid in the reward-model dataset until it was checked
  directly.** `dataset_emitter.py`'s tick-stride sampling missed nearly every
  single-tick pickup/death/success event, producing a held-out set that was
  100% one repeated value on the first bigger run — R² and MAE both looked
  meaningless because there was nothing to measure. Fixed by always
  capturing non-default-reward ticks regardless of stride; see "The reward
  model" for the corrected result.
- **More data alone did not fix the linear reward-model probe — it got
  worse.** Raw-pixel linear regression overfits training scenes' specific
  visual content and gets worse, not better, on genuinely new scene visuals
  (held-out MAE 0.565). A small CNN on the identical (corrected) dataset
  reaches a genuine held-out R²=0.32.
- **The CNN's aggregate MAE ties a trivial "guess the mean" baseline** —
  stated plainly rather than left implicit. The real, if partial, signal only
  shows up when broken out by event type: roughly 2× better than blind
  guessing specifically on the rare pickup/death/success frames, diluted to
  a rounding error in the aggregate because those frames are ~2% of the
  held-out set.

## What's implemented vs. envisioned

The original strategy document this pivot is based on describes a larger
system than fit in this session. Stated plainly rather than silently
under-delivered:

| Piece | Envisioned | Built |
|---|---|---|
| Game inventor | Full generative proposal system | Fixed move registry (`mutation.py`) + optional Claude revision (`scene_reviser.py`) |
| Regret | Oracle vs. trained-learner regret | Execution-gap proxy (real solver vs. closed-form estimate); classically-faithful only where the Q-learner applies |
| Archive | 4D (path-length × interactions × rule-shape × irreversibility) | 2D grid (path-length × interactions); the other two as per-occupant metadata |
| Agents | Deep RL policy | Tabular Q-learner over the coarse grid — real training, small state space |
| First-person renderer | Raycasting maze/FPS | Corridor-perspective single-heading view — this world has no second spatial axis |
| Reward model from pixels | CNN, large held-out set | Both, actually: a dependency-free linear pass (one episode) and a real torch CNN (30 scenes, held out by scene) — held-out R²=0.32, ~2× better than blind guessing on the rare events that matter, aggregate MAE ties a trivial baseline (see Known limitations) |
| Dataset action vocabulary | Native move fwd/back/left/right + mouse | Approximate mapping from a 2-axis world; strafe fields present but always 0.0 |

## Notable bugs found and fixed along the way

Several were caught in development and are worth a reviewer's attention as a
sign of how the verification work went, not just the final numbers:

1. **Feet-sensor width mismatch** caused a genuine physics deadlock — a player
   resting on a platform's exact corner could be held up by the main collision
   shape while the narrower feet-sensor never registered contact, freezing the
   controller (can't jump, can't fall).
2. **Boundary-clip false positive**: pymunk fires `begin` once per contact
   period, not every tick, so a naive per-tick check flagged ordinary sustained
   contact (sliding down a wall) as a tunneling exploit. Fixed by tracking
   persistent touch state instead of per-tick state.
3. **Jump-arc mismatch**: the original grid model approximated jump arcs as a
   straight line between takeoff and landing; the real arc rises fast early,
   so the approximation both approved impossible jumps and rejected possible
   ones. Replaced with an actual parabolic simulation using the engine's own
   constants.
4. **Pathfinding performance**: jump simulation was being recomputed from
   scratch on every replanning call; caching keyed on held-keys (which fully
   determines door-lock state) cut a batch that hung for 5+ minutes down to
   about 1 second.
5. **Invention-loop key mismatch**: the warm-start seeding round logged its
   insert-success under `"inserted"` while every later round logged the same
   thing under `"accepted"` — a silent reporting bug (the loop itself was
   fine; the printed "0/6 accepted" for a round that actually filled 5 cells
   was not). Found immediately by reading the first real run's output rather
   than trusting the printed summary, fixed by standardizing the key.

## Repository layout

```
run_demo.py                      guaranteed deterministic demo
run_claude_demo.py                Claude-API upgrade layer demo
run_generation_adaptation.py     generation-policy adaptation loop
run_q_learning_demo.py           tabular Q-learner: train, plot, play the trained policy
run_invention_loop.py            the invention loop (--claude to layer Claude revision)
run_dataset_export.py            dual-render GIF + (frame, action, reward) dataset export
run_reward_model_v2.py            bigger reward-model run: many scenes, held out by scene, linear vs. CNN
run_maze_demo.py                  "a maze with lava traps and a locked door" pattern demo
harness/                         all library code (see Architecture above)
runs/demo/                       output of run_demo.py (GIFs + summary.json)
runs/claude_demo/                output of run_claude_demo.py
runs/q_learning/                 output of run_q_learning_demo.py (learning curve + GIF)
runs/invention/                  output of run_invention_loop.py (report + heatmap + trend)
runs/dataset_export/             output of run_dataset_export.py (dual GIF + dataset)
runs/reward_model_v2/            output of run_reward_model_v2.py (per-scene data + comparison.json)
runs/maze/                       output of run_maze_demo.py
runs/adaptation/                 output of run_generation_adaptation.py (report + trend chart)
```
