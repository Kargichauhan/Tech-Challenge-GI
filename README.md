# Infinite Environment Generation via an Agent Harness

### ▶ [**Dashboard**](https://kargichauhan.github.io/Tech-Challenge-GI/)

![A generated scene being solved honestly, then reward-hacked by the adversarial probe, rendered top-down and first-person side by side](docs/demo.gif)

**At a glance:**
- Real physics (pymunk): gravity, jump arcs, collisions. Nothing scripted.
- Text → validated, playable environment. One command, no API key.
- A second agent tries to reward-hack the win condition. Every scene reports whether it got gamed, and how.
- Three agents, same levels: honest solver, adversarial solver, tabular Q-learner (trains from scratch, real learning curve).
- An invention loop mutates scenes on its own, including the objective's shape, and archives what it finds (MAP-Elites).

**Run it**:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pymunk pygame imageio jsonschema networkx matplotlib anthropic
python3 run_demo.py
```

7 environments, genuine + adversarial solve on each, one GIF per environment in `runs/demo/*.gif` (scene → agent trace → verifier-robustness result, under 30s). Deterministic, no external dependency. Run this one first.

Full design rationale and bug-by-bug story: [WRITEUP.md](WRITEUP.md), optional. Below the line: brief mapping, entry points, architecture.

---

## Why this covers the brief

**Creativity.** Scenes come from text two different ways: Claude can read a prompt and generate one directly (`run_claude_demo.py`), or a code-based generator can build one with no LLM involved at all. A second agent then tries to reward-hack whatever the objective is, and a classifier reports how it managed it (or didn't). On top of that, an invention loop mutates and revises scenes on its own, including changing the shape of the objective itself, and keeps an archive of what it finds. All of it is playable live from the dashboard above.

**Clarity.** One command, no API key, and a GIF you can read in under 30 seconds. The mapping and architecture sections below exist so you don't have to go hunting for how any specific part of the brief got addressed.

**Working output.** `run_demo.py` works out of the box and is the guaranteed path. Nine more entry points sit below it, each doing something different: Claude-driven generation, a trained Q-learner, the invention loop, a bigger reward-model run.

The brief's three research motivations map onto this pretty directly, too. Post-training environments are `harness/generation/policy_adaptation.py` and `harness/invention/`. Code-level objectives, like "picked up the can from the table," are `harness/schema/`'s event log and boolean-predicate objective language, checked entirely in code, with the brief's own example built as a real demo scene. And reward model training is `harness/render/dataset_emitter.py`'s `(frame, action, reward)` export, where a real CNN reaches a held-out **R²=0.32** and roughly halves the error on the rare success/pickup events that matter most, compared to blind guessing.

## The runnable entry points

All need only the venv above. Exceptions: `run_claude_demo.py` / `run_invention_loop.py --claude` need `ANTHROPIC_API_KEY`, `run_reward_model_v2.py` needs `torch`, the human-playable mode needs a real display.

| Script | What it does |
|---|---|
| `run_demo.py` | Guaranteed demo: deterministic generation, genuine + adversarial solve, GIF per environment. |
| `run_claude_demo.py "<prompt>"` | Claude generates the scene from your prompt *and* navigates it, verifier probe runs for comparison. |
| `run_generation_adaptation.py` | 3 rounds × 8 environments, scores each combo, updates sampling distribution, plots the trend. |
| `run_q_learning_demo.py [seed]` | Trains a tabular Q-learner from scratch, plots the learning curve, plays the trained policy for real. |
| `run_invention_loop.py [--claude]` | Warm-starts from the fixed combos, then mutates/evaluates/archives new scenes round by round (MAP-Elites). `--claude` layers Claude revision on top, falls back to code mutation automatically. |
| `run_maze_demo.py [seed]` | "Maze with lava traps and a locked door": winding, zigzagging platform layout. |
| `run_dataset_export.py [seed]` | Dual (top-down + first-person) GIF, `(frame, action, reward)` dataset export, linear reward-model fit. |
| `run_reward_model_v2.py [--fresh]` | Bigger reward-model run: many playthroughs, many scenes, held out by scene, linear vs. real CNN. |
| `python3 -m harness.render.play_human` | Interactive window: arrow keys to move, space/up to jump. |

`run_claude_demo.py` and `run_invention_loop.py --claude` are optional upgrades, verified to reach the live API. The guaranteed deliverable is the deterministic path above them.

## Architecture

```
harness/schema/            scene schema, event-log schema, objective predicate language
harness/generation/        generators (deterministic + freeform), validator, generation-
                           policy adaptation, coarse grid pathfinding oracle, archive charts
harness/engine/            pymunk physics engine, event log, boundary-clip detection
harness/solvers/           genuine solver, adversarial solver, tabular Q-learner
harness/verifier/          runs both solvers, classifies exploit type (or none)
harness/invention/         invention loop: mutation, regret/exploit scoring, behavior
                           keying, novelty filtering, MAP-Elites archive
harness/render/            renderer + GIF assembly, human-playable mode, first-person
                           corridor renderer, dataset exporter
harness/claude_agent/      Claude API layer: scene generation, navigation, scene revision
```

**Scene schema.** 6 primitives: `platform`, `ramp`, `door`, `key`, `hazard`, `pickup`, plus `player_start`/`goal_zone`. Doors lock/unlock via `key_id` matching.

**Event log and objectives.** Flat log of `collision`, `pickup`, `door_open`, `zone_enter`, `zone_exit`, `death`. Objective = boolean predicate over the log (`and`/`or`/`not`), order-blind by design: checks whether events happened, not the sequence. That's the surface the adversarial probe exploits.

**Solvers and verifier robustness.** Both plan routes on a coarse grid that simulates real jump physics (same constants as the engine, no approximation). Genuine solver: collects requirements in human order. Adversarial solver: cheapest path, rams the goal zone first if reachable. `harness/verifier/probe.py` runs both, classifies the result:

| `exploit_class` | Meaning |
|---|---|
| `order_violation` | Objective met, but a required event happened out of the implied order. |
| `unintended_path` | Goal reached without touching a door that appears to gate it. |
| `boundary_clip` | Engine tunneling: solid geometry crossed with no collision registered. Verified via injected-velocity tests; not reachable at current move/jump speeds (see Known limitations). |
| `none` | No exploit found. |

Every result also carries `honest_baseline_confirmed`: an exploit found where nobody proved the scene solvable honestly is flagged as a weaker claim, not reported the same as a confirmed one.

**Generation-policy adaptation.** Each round samples `N=8` scenes from a fixed combo list, probes each, scores it: `score = 0.5·genuine_solve_rate + 0.3·(1 − exploit_rate) + 0.2·(1 − validator_rejection_rate)`. Reweights next round's sampling distribution, floor so no combo hits zero. Weighted-sampling update, not code self-modification.

**Invention loop (explore / play / work / exploit).** `harness/invention/` mutates or Claude-revises scenes freely instead of resampling a fixed list, and can change the objective's shape itself. Scores candidates on a regret proxy (real solver ticks vs. closed-form estimate) plus exploit-resistance, archives the best scene per behavior cell in a 2D MAP-Elites grid (path-length × interaction-count). Conceptual basis: ACCEL, POET, PLR, UED/PAIRED, OMNI-EPIC, MAP-Elites, Eureka/Voyager/ADAS, not a reproduction of any of them. `run_invention_loop.py` warm-starts from the six combos above.

**Vision-policy bridge.** World is a side-scroller, one horizontal axis, no second axis for a literal raycaster. Instead: `harness/render/raycaster.py` renders a **corridor-perspective first-person view**, real depth earned through time as the player approaches an object, plus cosmetic 2.5D dressing (side panels, parallax skyline) drawn under every real feature. `run_dataset_export.py` exports `(frame, action, reward)` tuples in the challenge's action vocabulary; approximate mapping, no lateral strafe axis in this world.

**The reward model.** Closed-form linear pass on one playthrough: heavily underdetermined by design. Second pass: real CNN, 30 solved playthroughs, held out by scene.

| | held-out MSE | held-out MAE | held-out R² |
|---|---|---|---|
| Linear, more data | 40.4 | 0.565 | -3,963 |
| Small CNN (torch), same data | 0.0069 | 0.020 | **0.32** |

CNN's aggregate MAE (0.020) ties a trivial "guess the mean" baseline: 685 of 697 held-out frames are just the step penalty. By event type: on the 12 rare pickup/death/success frames, CNN MAE is 0.303 vs. 0.621 for blind guessing, roughly 2x better on the events that matter. A sampling bug (missing single-tick events under stride-4 sampling) hid this signal on the first run; see WRITEUP.md.

## Known limitations

- **Genuine-solver reliability varies by combo.** Hardest 5-primitive combo: ~50% honest solve rate. Simpler combos: ~100%. `honest_baseline_confirmed` exists so reporting doesn't overclaim on the harder cases.
- **`boundary_clip` verified but dormant.** Real, tested via injected velocity, not reachable at current jump/move speeds.
- **Regret is an execution-gap proxy, not classic UED regret.** Only the Q-learner is actually trained; elsewhere this compares a real solver against a closed-form estimate.
- **MAP-Elites archive is 2D, not the 4D in the original strategy doc.** Rule-shape and irreversibility count are computed but kept as per-occupant metadata, not grid axes.
- **First-person renderer is corridor-perspective, not a full raycasting maze-caster.** One true spatial axis in this world.

Full bug list (feet-sensor deadlocks, jump-arc mismatches, an invention-loop logging bug, the reward-model sampling bug, and more) plus the implemented-vs-envisioned gap table: [WRITEUP.md](WRITEUP.md).
