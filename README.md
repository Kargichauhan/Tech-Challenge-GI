# Infinite Environment Generation via an Agent Harness

### ▶ [**Dashboard**](https://kargichauhan.github.io/Tech-Challenge-GI/)

![A generated scene being solved honestly, then reward-hacked by the adversarial probe, rendered top-down and first-person side by side](docs/demo.gif)

**At a glance:**
- A real physics engine (pymunk) runs everything: gravity, jump arcs, and collisions are simulated, not scripted.
- One command turns text into a validated, playable environment. No API key needed.
- A second agent actively tries to reward-hack the win condition, and sometimes succeeds. Every scene reports whether it was gamed and how, which is the verifier-robustness problem that reward-model training depends on.
- Three agents play the same levels: a hand-built honest solver, a hand-built adversarial solver, and a tabular Q-learner that trains from scratch with a real learning curve.
- An invention loop mutates and revises scenes on its own, including changing the shape of the objective itself, and keeps a MAP-Elites archive of what it finds.

**Run it** (no API key, no network):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pymunk pygame imageio jsonschema networkx matplotlib anthropic
python3 run_demo.py
```

This generates 7 environments, runs the genuine solver and the adversarial verifier-probe on each, and writes one self-contained GIF per environment to `runs/demo/*.gif` (scene, then agent trace, then verifier-robustness result, under 30 seconds to read). Fully deterministic, no external dependency, the path reviewers should run first.

The full design rationale and the session-by-session story behind these decisions, including every bug found along the way, is in [WRITEUP.md](WRITEUP.md) if you want it. Everything below this line is the reference layer: what maps to the brief, what to run, and how it's built.

---

## Requirements ↔ what's here

| The brief asked for | What's here |
|---|---|
| Harness that constructs environments from text commands | `run_claude_demo.py "<prompt>"` has Claude read your exact text and emit a scene via a forced tool call. `harness/generation/maze.py` and `deterministic.py` are code-based generators that need no LLM at all. |
| Playable in a game or physics engine | `harness/engine/physics_engine.py` runs a real pymunk simulation: gravity, jump arcs, collision. Nothing here is approximated. |
| Agent maneuvers through generated environments | Three independent agents solve the same schema: `solvers/genuine.py` (honest), `solvers/adversarial.py` (exploit-seeking), and `solvers/q_learner.py` (trained from scratch). Claude itself also plays, through `claude_agent/navigator.py`. |
| "Models like Claude perform well at progressing through 2D environments on their own" (the brief's stated reason for starting in 2D) | `navigator.py` gives Claude a plain-text description of the state at each decision point, no pixels, and it navigates from there: a direct match to that reasoning, not just a citation of it. |
| Post-training environments (a supply of diverse, verified environments) | `harness/generation/policy_adaptation.py` (weighted resampling) and `harness/invention/` (an invention loop that mutates and revises scenes, archiving the results via MAP-Elites). |
| Code-level objectives ("picked up the can from the table") | `harness/schema/`'s event log and boolean-predicate objective language check everything in code; nothing is guessed from pixels. The brief's exact example is a real demo scene in `run_demo.py`. |
| Reward model bridge (code-truth → pixel understanding) | `harness/render/dataset_emitter.py` exports real `(frame, action, reward)` tuples in the challenge's own action vocabulary. A real torch CNN reaches held-out **R²=0.32** and roughly halves the error on the rare success/pickup events that matter most, versus blind guessing. |
| Vision-policy's action space (move fwd/back/left/right, mouse ΔX/ΔY) | `harness/render/raycaster.py` renders a first-person view, and `dataset_emitter.py` maps it explicitly onto that exact vocabulary. |
| Creativity | The adversarial exploit classifier, the invention loop's objective-shape mutations, the MAP-Elites archive, and the live playable browser dashboard above. |
| Clarity / working output | Nine runnable entry points (table below), all verified working, guaranteed path needs no API key. |

## The runnable entry points

| Script | What it does | Requires |
|---|---|---|
| `run_demo.py` | The guaranteed demo. Deterministic generation, genuine + adversarial solve, GIF per environment. | nothing but the venv |
| `run_claude_demo.py "<prompt>"` | Claude generates the scene from your text prompt *and* navigates it itself, then the same verifier probe runs for comparison. | `ANTHROPIC_API_KEY` |
| `run_generation_adaptation.py` | 3 rounds × 8 environments over a fixed set of primitive combos, scores each, updates the sampling distribution, plots the trend. | nothing but the venv |
| `run_q_learning_demo.py [seed]` | Trains a tabular Q-learner from scratch, plots the learning curve, then plays the *trained* policy for real and renders it. | nothing but the venv |
| `run_invention_loop.py [--claude]` | The invention loop: warm-starts from the fixed combos, then mutates/evaluates/archives new scenes round by round (MAP-Elites). `--claude` layers Claude scene revision on top, falling back to code mutation automatically. | nothing but the venv (`ANTHROPIC_API_KEY` only for `--claude`) |
| `run_maze_demo.py [seed]` | A "maze with lava traps and a locked door": a winding, zigzagging platform layout. | nothing but the venv |
| `run_dataset_export.py [seed]` | Dual (top-down + first-person) GIF plus a `(frame, action, reward)` dataset export and a linear reward-model fit on that episode. | nothing but the venv |
| `run_reward_model_v2.py [--fresh]` | The bigger reward-model run: many playthroughs, many scenes, held out by scene, linear method vs. a real small CNN. | `torch` (CPU build: `pip install --index-url https://download.pytorch.org/whl/cpu torch`) |
| `python3 -m harness.render.play_human` | An actual interactive window: arrow keys to move, space/up to jump. | nothing but the venv + a display (X11/Wayland) |

`run_claude_demo.py` and `run_invention_loop.py --claude` are optional upgrade layers, verified to reach the live API correctly, but the guaranteed deliverable is the deterministic path above them.

## Architecture

```
harness/schema/            scene schema, event-log schema, objective predicate language
harness/generation/        deterministic + freeform generators, validator, generation-policy
                           adaptation, coarse grid pathfinding oracle, archive charts
harness/engine/            pymunk physics engine: runs a scene, emits the event log,
                           detects boundary-clip tunneling natively (segment_query)
harness/solvers/           genuine solver, adversarial solver, and a tabular Q-learner
                           (the one agent that actually trains from scratch)
harness/verifier/          runs both solvers on the same scene, classifies how (or
                           whether) an adversarial success gamed the objective
harness/invention/         the invention loop: scene mutation, regret/exploit scoring,
                           behavior-space keying, novelty filtering, MAP-Elites archive
harness/render/            headless renderer + GIF assembly, human-playable mode,
                           first-person corridor renderer, dataset exporter
harness/claude_agent/      the Claude-API layer: scene generation, Claude-driven
                           navigation, and scene_reviser.py for the invention loop
```

**Scene schema.** Six primitives (`platform`, `ramp`, `door`, `key`, `hazard`, `pickup`) plus `player_start`/`goal_zone`. Doors lock and unlock via `key_id` matching.

**Event log and objectives.** The engine emits a flat log of `collision`, `pickup`, `door_open`, `zone_enter`, `zone_exit`, `death` events. An objective is a boolean predicate over that log (`and`/`or`/`not`), deliberately order-blind: it checks whether events happened, not in what sequence. That's the exact surface the adversarial probe is built to exploit.

**Solvers and verifier robustness.** Both solvers plan routes with a coarse grid model that simulates real jump physics, using the engine's own constants, not an approximation. The genuine solver collects requirements in the order a human would; the adversarial solver goes for whatever's cheapest and rams the goal zone first whenever reachable. `harness/verifier/probe.py` runs both and classifies the result:

| `exploit_class` | Meaning |
|---|---|
| `order_violation` | Objective satisfied, but a required event happened out of the order the level's geometry implies. |
| `unintended_path` | Goal reached without ever interacting with a door that appears to gate it. |
| `boundary_clip` | The engine's own tunneling: a swept path crossed solid geometry with no collision registering. Verified real via injected-velocity tests, but not reachable at the action space's normal speeds today (see Known limitations). |
| `none` | No exploit found. |

Every result also carries `honest_baseline_confirmed`, so an exploit found on a scene nobody proved solvable honestly is flagged as a weaker claim, not reported the same as a confirmed one.

**Generation-policy adaptation.** Each round samples `N=8` scenes from a fixed list of named combos, probes each, and scores it: `score = 0.5·genuine_solve_rate + 0.3·(1 − exploit_rate) + 0.2·(1 − validator_rejection_rate)`. The result reweights the next round's sampling distribution, with a floor so no combo ever hits zero. A plain weighted-sampling update, not code self-modification.

**The invention loop (explore / play / work / exploit).** A second, independent mechanism in `harness/invention/` that mutates or Claude-revises scenes freely instead of resampling a fixed list, and can change the objective's shape itself (not just which primitives appear). It scores candidates by a regret proxy (real solver ticks vs. a cheap closed-form estimate) and exploit-resistance, then keeps the best scene per behavior cell in a 2D MAP-Elites archive (path-length × interaction-count). Loosely modeled on published open-ended-learning research (ACCEL, POET, PLR, UED/PAIRED, OMNI-EPIC, MAP-Elites, Eureka/Voyager/ADAS): the conceptual basis for a much smaller version of the same idea, not a reproduction of any of them. `run_invention_loop.py` warm-starts from the six combos above as its round-0 seed set.

**Vision-policy bridge.** This world is a side-scrolling platformer with one horizontal axis, so a literal Wolfenstein-style raycaster has no second spatial axis to cast into. Instead `harness/render/raycaster.py` renders a **corridor-perspective first-person view**: real depth earned through time as the player approaches an object, dressed up with cosmetic 2.5D side panels and a parallax skyline that are drawn first, under every real feature, so real and decorative are never ambiguous. `run_dataset_export.py` exports `(frame, action, reward)` tuples in the challenge's action vocabulary, an honest approximate mapping since this world has no lateral strafe axis.

**The reward model.** A closed-form linear pass on one playthrough (no new dependency) was heavily underdetermined by design. A second pass trained a real CNN on 30 solved playthroughs, held out by scene:

| | held-out MSE | held-out MAE | held-out R² |
|---|---|---|---|
| Linear method, more data | 40.4 | 0.565 | -3,963 |
| Small CNN (torch), same data | 0.0069 | 0.020 | **0.32** |

The CNN's aggregate MAE (0.020) ties a trivial "guess the mean" baseline, because 685 of 697 held-out frames are just the common step penalty. Broken out by event type, the real signal shows up: on the 12 rare pickup/death/success frames, the CNN's MAE is 0.303 versus 0.621 for blind guessing, roughly twice as good on exactly the events a reward model needs to get right. A real sampling bug (missing single-tick events under stride-4 sampling) hid this signal entirely on the first run of this data; see WRITEUP.md for that story.

## Naming

**"Generation-policy adaptation"** (never "self-improvement" or "RSI") describes `harness/generation/policy_adaptation.py`: a sampling distribution over a fixed combo list, reweighted from measured stats.

**"Open-ended invention loop"** is `harness/invention/` specifically: it invents scenes outside any fixed list and grows a standing archive round over round, but it's still bounded, a fixed move registry and archive shape, no code that executes or modifies itself.

## Known limitations

- **Genuine-solver reliability varies by combo.** The hardest five-primitive combo solves honestly in roughly half of random seeds; simpler combos solve close to 100%. `honest_baseline_confirmed` exists so downstream reporting doesn't overclaim on the harder cases.
- **`boundary_clip` is verified-correct but currently dormant**, real and tested via injected velocity, but not reachable at the game's actual jump/move speeds since pymunk's speculative contacts resist tunneling well past that range.
- **Regret is a repurposed execution-gap proxy, not classic UED regret.** Classic regret is oracle return minus *trained-learner* return; only the Q-learner here is actually trained, so elsewhere this compares a real solver against a closed-form estimate instead.
- **The MAP-Elites archive is 2D, not the 4D in the original strategy document.** Rule-shape and irreversibility count are computed but kept as per-occupant metadata rather than grid axes, since a true 4D grid would be nearly empty at the batch sizes this loop runs.
- **The first-person renderer is a corridor-perspective view, not a full raycasting maze-caster**, since this world has only one true spatial axis.

Full bug-by-bug detail (feet-sensor deadlocks, jump-arc mismatches, an invention-loop logging bug, the reward-model sampling bug, and more), plus the implemented-vs-envisioned gaps against the original strategy document, are in [WRITEUP.md](WRITEUP.md).
