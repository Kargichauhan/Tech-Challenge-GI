# Session Writeup: Infinite Environment Generation via an Agent Harness

**Optional session log, not required reading.** `README.md` is the complete, self-sufficient submission: architecture, requirements mapping, quickstart, results. This document is a narrative account of how it got there, the checkpoints, the bugs found along the way, and the reasoning behind decisions. Read it only if you want the story behind the results, not to evaluate them.

## The brief and the framing

The challenge: build an agent harness that reliably constructs 2D environments from text commands and can navigate through them, with an eye toward the research motivation General Intuition stated directly (infinite generation, code-level verifiable objectives, reward signal, improved generation). The explicit ask was to mirror that framing back at them using their own language, not bolt something unrelated on top.

## The plan we locked in

Before writing code, we agreed on a build order and two checkpoints where I'd stop and show the design before it became load-bearing for anything else:

1. Hybrid scene schema: fixed structure plus extensible primitive library
2. Event-log schema and objective predicate language (**pause for review**)
3. Freeform generation and validator (reachability/overlap, reject-or-repair)
4. Adversarial-probe solver and exploit logging (**pause for review**)
5. Deterministic, zero-API-key path as the guaranteed demo
6. Per-environment GIF (scene, then trace, then verifier-robustness result)
7. Verifier-feedback-driven generation-policy adaptation, built last, meant to degrade gracefully if time ran short

Naming discipline was explicit from the start: "generation-policy adaptation" or "verifier-feedback-driven generation," never "self-improvement" or "RSI."

## Step 1: The scene schema

Six primitives (`platform`, `ramp`, `door`, `key`, `hazard`, `pickup`) plus structural `player_start`/`goal_zone` fields present in every scene. Doors lock via `key_id` matching a key's own `key_id`. Built as a JSON Schema (`harness/schema/scene_schema.py`) with a validator function, tested against a hand-built example scene before moving on.

## Step 2: Event log and objective language (checkpoint)

Six event types (`collision`, `pickup`, `door_open`, `zone_enter`, `zone_exit`, `death`) and a boolean predicate language (`event`/`and`/`or`/`not`) over that log. Two decisions were deliberate and flagged for review before anything got built on top of them:

- **The objective predicate is order-blind by design.** It checks whether events happened, not in what sequence. That's not a gap to close later, it's the actual test surface the adversarial probe (step 4) needed. You confirmed this explicitly and told me not to add a `sequence` operator, since that would split objectives into two tiers and add design surface for no benefit at this stage.
- **`boundary_clip` detection.** You pushed back on my first instinct (a bbox/trajectory heuristic) and asked for a pymunk-native `segment_query_first` check instead, since a proxy heuristic sitting under the submission's headline verifier-robustness number would be a credibility risk. I agreed and rewrote the design before building the engine on top of it.

## Step 3: Freeform generation and validator

Built the deterministic template generator and the validator pipeline (structural, then referential, then overlap, then reachability, with a repair pass) in parallel, since the deterministic generator became the guaranteed demo path. The reachability check runs a coarse grid model that treats platforms, ramps, and locked doors as solid and simulates jump arcs.

Bugs found and fixed while getting this to actually validate correctly:

- **Foot-position bug.** The player's start cell was computed from the top-left corner instead of the foot line, so a player resting exactly on a surface never registered as "standing."
- **Door/ground shared-cell bug.** Doors and the ground can occupy the same grid coordinate. Representing solidity as a flat cell set meant "opening" a door also erased the ground's solidity at that cell. Fixed by computing door solidity fresh each pass instead of baking it into the static solid set.
- **`primitive_combo=[]` truthiness bug.** `[] or default_combo` evaluates to the default in Python, since an empty list is falsy, which silently ignored a caller's explicit "no primitives" request.
- **Platform-height-0 overlap bug.** The generator's height-offset choices included 0, which placed a "floating" platform directly on top of the ground.
- **Key-id vs. key-object-id mismatch.** The reachability closure tracked reachable keys by their door-matching `key_id`, but the objective predicate references the object's own `id`. These are two different strings the code was conflating, which caused real levels to be misjudged unreachable.

After these fixes: 180 of 180 pass across 6 primitive combos times 30 seeds.

## Step 4: Physics engine

A pymunk-based engine that executes a scene tick by tick, emits the event log, and implements the native `segment_query`-based boundary-clip check agreed on in step 2. Hazards, locked doors, platforms, and ramps are all solid, matching the approved design (the player is supposed to be blocked by all of them, which is what makes tunneling through one meaningful).

## Step 5: Genuine solver

A greedy/pathfinding solver sharing the grid model with the validator. This is where most of the session's real debugging happened, because "the solver mostly works" isn't the same as "the solver works." Each fix below was found by actually running batches and reading why specific scenes failed, not by inspection:

- **Jump-timing bug.** The controller waited until near the landing x before jumping, which meant it would walk straight through anything sitting between takeoff and landing (a hazard, say) before ever leaving the ground.
- **Fall-drift mismatch.** The grid model's "fall off a ledge" edge assumes a straight vertical drop, but the real controller kept moving horizontally while falling, landing somewhere the model never checked for hazards.
- **Oscillation from full replanning.** Replanning from scratch after every single hop caused genuine oscillation, since two equally short routes in opposite directions look interchangeable to a fresh search from a slightly different landing spot each time. Fixed by committing to a whole planned path and only replanning when a hop's actual landing diverges from what was planned.
- **Feet-sensor width deadlock.** The feet sensor used to detect "am I grounded" was narrower than the main collision box. A player barely catching the very corner of a platform could be held up by the wider main shape while the narrower sensor never registered contact, a genuine deadlock where the player could neither jump nor fall.
- **Rotation-tipping bug.** The player body had no rotation lock, so collisions could tip it over, which then misaligned the feet sensor.
- **Jump-arc mismatch, the deepest one.** The grid model originally approximated a jump as a straight line between takeoff and landing. Real jumps rise fast early and arc over, so the straight-line approximation both approved jumps the real physics couldn't make and rejected ones it could. Replaced with an actual parabolic simulation using the same constants as the real engine (`harness/constants.py`), which removed the whole class of mismatch rather than re-tuning approximation constants.
- **Pathfinding performance.** Jump simulation was being recomputed from scratch on every replanning call. Caching keyed on `held_keys`, which fully determines door-lock state, cut a batch that was hanging for over five minutes down to about a second. This became directly relevant later, since the generation-policy adaptation loop depends on many fast probe runs.

## Step 6: Adversarial solver and exploit classifier (checkpoint)

Built the adversarial solver, which goes for whatever satisfies the objective cheapest and prefers the goal zone outright even before intended prerequisites, and the exploit classifier (`order_violation`, `unintended_path`, `boundary_clip`, `none`), then paused for review before wiring the GIF renderer or generation-policy adaptation on top.

Three specific verifications were asked for before signing off, and any failure was treated as a bug to fix, not just something to note:

1. **Prove `boundary_clip` and `unintended_path` actually fire.** Building hand-crafted test scenes for this surfaced a real detector bug: pymunk fires `begin` once when contact starts, not on every tick contact continues, so a naive per-tick check flagged ordinary sustained contact, like sliding down a wall after a normal collision, as a tunneling exploit. Fixed by tracking persistent touch state instead of per-tick state. Direct velocity-injection testing also showed that pymunk's speculative-contact system actively resists tunneling well past the game's actual jump/move speeds. `boundary_clip` is verified correct via a direct test, but it isn't reachable through normal gameplay at current speeds, and that's stated as-is rather than oversold.
2. **Correlate exploit findings against genuine-solver reliability.** Across a 30-scene batch, the two scenes where an exploit was found without a confirmed honest baseline were both in the same combo known to have lower genuine-solver reliability, a real, non-coincidental correlation. `honest_baseline_confirmed` was added to the probe's output so downstream reporting (GIF captions, aggregate stats) distinguishes a confirmed shortcut past a provably solvable level from the weaker case where the adversary succeeded but nobody confirmed anyone could solve it honestly.
3. **Check whether the "already there" nudge fallback inflates adversarial success.** My first pass at this check had its own bug: it assumed `goal_zone` is always the last-completed requirement, but the objective's order-blind AND means a different pickup can finish last. Fixed the check, then confirmed 0 of 12 adversarial successes relied on a nudge-only final leg.

## Step 7: GIF renderer

Headless pygame rendering, an off-screen surface with `SDL_VIDEODRIVER=dummy`, assembled by imageio into a GIF: generated scene, then agent trace with live event captions, then a verifier-robustness result screen. Verified visually by extracting and inspecting individual frames, not just trusting that the code ran without errors.

## Deterministic demo (`run_demo.py`)

Six environments across the primitive library, deterministic seeds, zero external dependency. Last clean run: 6 of 6 genuine solves, 2 confirmed `order_violation` exploits found by the adversarial probe.

## Claude-API upgrade layer

Built after you explicitly chose "build code now, capture the live run later" when no API key was available in-session:

- `scene_generator.py`: a text prompt becomes a scene via a forced, non-strict tool call. Deliberately not `output_config.format` structured outputs, since the objective predicate is a recursive schema (self-referential via `$ref`), and structured outputs explicitly don't support recursive schemas.
- `navigator.py`: Claude picks one discrete action per decision point from a text description of game state, stateless per call (no growing conversation) to keep cost and latency bounded regardless of episode length.
- Verified as much as possible without a live key by confirming both the scene generator's and navigator's requests are well-formed: sending them against a deliberately invalid key and getting the expected 401, not a local construction error. That's proof the request reaches the server correctly, though not proof of what Claude does with it.

### What happened when you actually ran it

Two real issues surfaced live, both fixed:

1. **Claude returned the `objective` field as a JSON-encoded string** instead of a nested object, consistently across all 4 retry attempts, a known tool-calling quirk specifically tied to the one recursive part of the schema. Fixed with defensive parsing (`json.loads` if it comes back as a string) rather than relying on re-prompting to fix a systematic model behavior.
2. **A real reporting bug.** The GIF's result screen and the script's console output were both showing the deterministic solver's result as if it were the outcome of the trace on screen, but the trace shown is Claude's own play. In your actual test run, Claude's own navigation made 8 decisions (repeated `move_right`, never a `jump`) and almost certainly died in the hazard within about 2 seconds, while the unrelated deterministic baseline solved the same scene successfully. Fixed by giving the result screen a primary "CLAUDE'S NAVIGATION: `<result>`" headline with the deterministic comparison clearly subordinate underneath, so a viewer can never conflate the two. Verified with a regression test that the original deterministic-only demo path is unaffected.

A security note from the same testing round: your real API key ended up pasted into the chat twice, once in a stray shell line and once in an `export` command. I flagged it each time and recommended rotating it via the Anthropic console. Worth confirming that's done if it hasn't been.

## Generation-policy adaptation

Confirmed with you upfront: 8 environments per round, 3 rounds, sampling from a fixed list of six named primitive combos (not independent per-primitive weights, to keep the per-combo signal legible). Score per combo:

```
score = 0.5 · genuine_solve_rate + 0.3 · (1 − exploit_rate) + 0.2 · (1 − validator_rejection_rate)
```

Normalized into next round's sampling weights, with a floor so no combo ever hits zero. A trend chart (matplotlib, dark aesthetic matching the GIF renderer, dataviz-skill-validated categorical palette) shows genuine-solve rate, exploit rate, and validator-rejection rate across rounds. In the actual run, harder combos (the 5-primitive full course, `ramp+hazard+pickup`) got down-weighted toward the floor while simpler, more reliable combos converged upward. The mechanism visibly did what it's supposed to.

## README and current state

Wrote `README.md` covering architecture, the three runnable entry points, the naming discipline, and the known limitations: genuine-solver reliability varies by combo (roughly 50% on the hardest one), `boundary_clip` is verified correct but not reachable at current game speeds, and the Claude-API layer needed a live key to fully verify (since resolved, per above).

## Investigating the navigator death, and where it landed

You asked me to look into why Claude's own play died within 2 seconds without ever attempting a jump. That investigation found a real design bug, not just a prompting gap: the navigator held one discrete action for the entire 0.25-second decision window, and choosing "jump" zeroed horizontal velocity for as long as it was held. So even a perfectly timed jump decision could only produce a vertical hop with zero horizontal movement, physically incapable of clearing any gap or hazard regardless of prompting. The deterministic solvers never hit this because they re-decide the action every single tick (jump for exactly one tick, then move for the rest); the navigator can't do that per-tick replanning without a tool call per tick.

Fixed by composing a single "jump" decision the same way: one tick of vertical launch, then the rest of the window continuing whatever horizontal direction was already in motion, so Claude only has to decide when to jump, not sequence the two-tick maneuver itself. Also added a computed hazard-proximity hint to the observation text. Verified the fix directly, bypassing the API: a scripted policy using the same fixed action model against the exact scene that killed the recorded run now clears the hazard cleanly, with real horizontal displacement during the jump and no death event.

What's still open: we checked two earlier recorded runs (`simple_hazard_jump.json`, `key_door_goal_level.json`) to see if either already showed a Claude success by coincidence. Neither did (one died at the same hazard, the other fell into a bottomless pit and timed out without ever picking up the key), and both predate the fix besides. So there is currently no recorded run of Claude successfully completing a level end to end via its own navigation. The decision, stated directly in the README's known limitations rather than glossed over, was to disclose the verified-but-unconfirmed state rather than either withholding the failing recording or waiting indefinitely for a lucky successful run to cherry-pick.

## The pivot: adopting a larger architecture mid-session

After the cross-check above, you raised wanting something closer to "real life minecraft or something like a space videogame." I flagged that as a significant scope jump and recommended against it as a first instinct. You then pasted a complete external strategy document ("Ludens") proposing a much larger architecture: a self-playing game-inventor system modeled on published open-ended-learning research (OMNI-EPIC, ACCEL, POET, PowerPlay, UED/PAIRED, PLR), with a MAP-Elites archive, regret scoring, an explore/play/work/exploit loop, and a raycasting first-person renderer bridging toward the challenge's stated vision-policy action space. It explicitly called the mechanism "recursive self-improvement," which directly contradicted this session's own naming rule from step 1.

I flagged both the scope risk and the naming contradiction and asked you directly rather than picking one silently. Your answers: full pivot to the new architecture, and adopt the new document's framing for naming. I then used Plan Mode before writing any code: one Explore pass over the existing codebase (confirming my own understanding of `pathfinding.py`, `probe.py`, and `policy_adaptation.py` against the actual files rather than memory), one Plan sub-agent scoped specifically to the riskiest new pieces (the invention loop, regret scoring, the archive), then three clarifying questions before locking the plan. Should "recursive self-improvement" apply everywhere or just the new package? You chose scoping it to the new package only. Was a corridor-perspective renderer an acceptable scope-down from "3D game"? You agreed. How should execution be paced across six phases? You chose two checkpoints rather than one per phase.

**Guiding decision carried through the whole pivot**: nothing in `engine/`, `solvers/genuine.py`, `solvers/adversarial.py`, `pathfinding.py`, or `validator.py` changed. The coarse grid model already simulates real jump physics and already serves as a fast oracle; the continuous pymunk engine is the only thing capable of producing `boundary_clip`. Everything new is additive: a new `harness/invention/` package plus a handful of new modules alongside the existing ones.

### Phase 1: small, honest additions to what already existed

A literal interactive human-playable mode (`harness/render/play_human.py`). Jump fires on key-down, not while held, for the same reason the Claude navigator composes jump the way it does (holding it the whole tick window would zero horizontal velocity). Surfaced the validator's existing `seed_attempts` data as an explicit "valid-yield rate" in the demo's output rather than tracking it twice. Added one demo scene using the challenge's own example phrasing ("a table with a can on it, pick up the can"), built from existing `pickup`+`platform` primitives, no schema change.

### Phase 2: the Q-learner, the one agent that actually trains

`GenuineSolver` and `AdversarialSolver` are both hand-built and never improve. `harness/solvers/q_learner.py` is a genuinely trained tabular Q-learner over the same coarse grid graph, reusing `pathfinding.py`'s own transition primitives rather than a second physics model. Tested across every combo in the library, including the hardest 5-primitive one, it reliably converges from strongly negative early-episode returns to a solved policy, and the converged policy was verified by actually executing it in the real pymunk engine (not just trusting the training curve) via the same `do_one_hop` executor the other solvers share. The learning curve for the hardest combo goes from roughly -2.0 to +1.0 over 3,000 episodes in about 2.5 seconds, a real, unfaked curve, visually confirmed by reading the rendered chart.

### Phase 3: the invention loop (checkpoint passed)

Built `harness/invention/`'s full registry (mutation, regret, behavior keying, novelty pre-filtering, the archive, the explore/play/work/exploit orchestration) and `run_invention_loop.py`, then, per the agreed checkpoint, ran the deterministic-only loop (no Claude) for several rounds and read `runs/invention/report.json` by hand before building anything further on top. That first real run surfaced a genuine bug: the warm-start seeding round logged its insert result under the key `"inserted"` while every later round used `"accepted"`. The loop itself was correct (5 of 6 seed combos really did fill archive cells), but the printed summary said "0/6 accepted," which would have read as a broken seeding pass if not caught. Fixed by standardizing the key, then re-verified: coverage grew monotonically round over round (20% to 44% across seed plus 6 rounds), regret values landed in a sane 0.0 to 0.35 range, and a few mutations turned up genuinely new `order_violation`/`unintended_path` exploits the original six combos never exposed, exactly the kind of thing an invention loop should be able to find that a fixed-list sampler can't.

With that checkpoint passed, `harness/claude_agent/scene_reviser.py` was layered on top, structurally identical to the proven `scene_generator.py` pattern (forced tool call, defensive JSON-string parsing, bounded retry). Verified the fallback path two ways without spending a real API call: running `--claude` with no key set (confirmed it prints a clear message and proceeds deterministically), and a mocked client whose `messages.create` always raises (confirmed `explore_propose` falls through to a real code-mutation move, not `None`). A live run against a real key is the user's to do, same as `run_claude_demo.py` always has been.

### Phase 4: archive visualization

Invoked the dataviz skill again for a sequential palette, since a heatmap is a different color job than the categorical trend chart from generation-policy adaptation. Its reference palette's sequential ramp is documented for a light chart surface (low value recedes toward a light background), and this repo's charts are all dark-surface, so the ramp is used reversed here: low fitness maps to the ramp's darkest step (blending toward the dark surface), high fitness to its lightest step (popping against it). That's a reasoned adaptation, not a guess. Empty archive cells get a distinct flat neutral gray, not "very dark blue," so a genuine 0.0-fitness cell never looks identical to "never reached." Confirmed visually by reading the rendered PNG: empty cells clearly read as blank, filled cells' brightness clearly tracks fitness.

### Phase 5: the vision-policy bridge (checkpoint passed)

Before building the raycaster, worked out explicitly why a literal Wolfenstein-style raycaster doesn't apply to a side-scrolling world (no second spatial axis to cast a ray fan into) and scoped it honestly instead: a corridor-perspective first-person view, real depth earned through time rather than a second axis. Built `raycaster.py`, wired it into `gif_builder.py` as an opt-in `dual_render` flag (default off, so the existing deterministic and Claude demos are unaffected), and built `dataset_emitter.py` for the `(frame, action, reward)` export with an explicit mapping onto the challenge's action vocabulary (strafe fields present but always 0.0, since this world has no lateral axis; a heading-flip triggers the one real "look elsewhere" signal). Per the agreed checkpoint, extracted and visually inspected sample frames from a real dual-render GIF before attempting the optional reward model, and confirmed the first-person door correctly grows and switches from locked (orange) to open (green) as the player approaches, with the hazard visible below it.

For the optional reward-model stretch piece, checked what was actually installed rather than assuming: neither torch nor scikit-learn is a dependency in this environment, and installing a deep-learning framework purely for one explicitly-first-to-cut stretch component wasn't worth the weight. Built a closed-form linear regression (numpy only) on downsampled frames instead, and reported the result exactly as measured rather than tuning it to look better: with only about 74 frames from a single playthrough and a much larger downsampled feature count, the fit is heavily underdetermined and the held-out R² is strongly negative. That's a real property of training on one episode, not something a different model choice would fix, and both the module docstring and the README's implemented-vs-envisioned table say so.

### Phase 6: packaging

Extended, not replaced, `README.md` and this writeup with the invention loop's explore/play/work/exploit mapping, the regret formula, a citations list for the conceptual basis (stated as conceptual basis, not claimed implementations), the naming section's explicit scoping (new terminology for `harness/invention/` only; `policy_adaptation.py` keeps its original, accurate description), an implemented-vs-envisioned table, and the new bugs and limitations from this phase of the session.

## Post-review fixes and a scope re-check

After the pivot's six phases were built and demonstrated, live use surfaced two real, separate issues:

1. **The invention loop's `--claude` mode looked hung.** A real run took 268 seconds with zero visible output beyond the round headers, and the report showed 0 of 15 candidates had actually used Claude revision; every one silently fell back to deterministic mutation. The root problem was `make_reviser`'s error handling: any failure (auth error, timeout, validation exhaustion) was caught and swallowed with no trace at all, so a 100%-fallback run was indistinguishable from a working-but-unlucky one. Fixed by adding a 60-second request timeout so a blocked or unreachable network fails within a bounded time instead of the SDK's much longer default, adding progress callbacks (`on_round_start`, `on_candidate`, `on_candidate_done`) so every candidate prints a definite outcome as it finishes rather than just on failure, and printing the actual exception or rejection reason instead of swallowing it. Verified with the same mocked-failure test used earlier (a client whose `messages.create` always raises) to confirm the timeout parameter is present and the failure reason is now visible, plus a regression run of the deterministic-only path to confirm the new callbacks don't change its behavior.

2. **You asked for the first-person view to go further into 2.5D/3D.** Given the world genuinely has no depth axis, I laid out three real options at very different cost and risk rather than silently picking one: cosmetic 2.5D dressing on the existing corridor view; a schema-level "lane" concept enabling true perpendicular walls and real strafing, which would touch the schema, physics, solvers, and Q-learner state space; or a full 3D engine rebuild. The second and third would have been substantial, risky redesigns of already-verified systems. You chose the cosmetic option. Built side wall panels with perspective seam lines, a floor grid receding to a vanishing point, and a slowly drifting parallax skyline, all drawn first, with every real feature (platforms, hazards, doors, the goal zone) drawn on top, so the layering makes clear what's real data versus decorative framing. Verified visually via rendered test frames and the full dual-render GIF pipeline, plus a regression run of the existing exploit-classifier tests and the deterministic demo to confirm nothing in the physics/verification path was touched.

## Submission review, a "maze" generator, playable browser demos, and the bigger reward-model run

Before sending this off, you asked for a full cross-check against the original brief with an eye toward what evaluators actually expect, plus a review of a competing team's submission you'd found, to see what, if anything, was worth incorporating.

**The brief cross-check** surfaced one urgent, non-content gap (this directory had never been git-initialized, so there was no way to share it as a repo) and several real documentation gaps given how explicit the brief is about reviewer time ("we will not have hours to review it"): no at-a-glance hook, no explicit requirements-to-deliverable mapping table, and the two playable browser artifacts weren't mentioned in the README at all despite being the best 10-second "try this" material in the submission. Fixed all of the content gaps (added an at-a-glance section, a full requirements-mapping table, and the artifact links up top); left git alone per your explicit instruction.

Substantively, the cross-check identified the reward-model bridge as the weakest of the brief's three named "why this matters" pillars. Not fatal, but the one place our honest number was actually bad rather than just modestly scoped down, and the brief names it as one of exactly three core motivations, not a minor detail.

**The competing repo review** (a different team's "Ludens" submission, JS/browser-based, a top-down grid maze rather than a physics platformer) was substantially strong: a real trained CNN with credible held-out numbers, a literal "picked up the can" tile-based objective, one-way tiles as a genuine rules-level (not just layout) mutation. Also found a real performance bug in it (their `computeRegret` retrains a fresh Q-learner on every single render tick via the telemetry HUD) and one documentation overclaim (the reward model predicts a derived "progress" proxy, not the literal reward value, despite the README's wording). Declined to port their world model, since a different genre (top-down grid, no gravity) is why a literal raycaster works for them and doesn't for us, not a shortfall on our side, and declined their browser-calls-a-local-server architecture too. Did port one real idea: separating "goes in the archive" (diversity, unconditional) from "is worth building on further" (curriculum, gated) as two explicit checks instead of one conflated fitness score. Added `regret.is_frontier_worthy` and wired it into `play_select_target`, verified live: a scene with a confirmed exploit correctly kept its archive cell but was excluded from being picked as a mutation parent, which the old single-score design couldn't have distinguished.

**A "maze with lava traps and a locked door" generator** (`harness/generation/maze.py`), a winding, zigzagging platform layout using the existing primitives, not a new schema type (a real 2D branching maze isn't honest to build in a world with one spatial axis). Hit a real bug immediately: one height offset was exactly 0, embedding a platform inside the ground plate, which triggered the validator's repair pass to shove the entire ground plane 1,120px sideways, the exact trap `deterministic.py`'s own comments already warned about. Fixed, then tuned gap sizes until genuine solve rate across random seeds reached roughly 63%, consistent with (not worse than) the existing hardest combo's reliability.

**Two playable browser artifacts** followed from a real misunderstanding worth recording: you asked to see the corridor renderer "moving," and what was first shown was a single scene generated offline in Python and hand-copied into a static page, not the live "type anything, get an environment" experience the brief describes. Clarified this explicitly rather than letting it stand, then built a genuine local keyword compiler (no LLM, no network; the artifact sandbox blocks external calls regardless) that runs entirely in the browser: type a prompt, it parses keywords onto the same primitive vocabulary, and a lightweight JS/Canvas physics plus corridor-view port makes it immediately playable, with a reachability safety check before handing back any generated layout.

**The bigger reward-model run** (`run_reward_model_v2.py`, `train_reward_model_cnn.py`) tested whether the original weak result was a data-volume problem or a model-class problem, by changing one variable at a time: 30 solved playthroughs across genuinely different scenes (not one), held out by scene, comparing the same linear method against a real small CNN (torch, CPU build; checked install size first and used the roughly 200MB CPU-only wheel instead of the multi-GB default CUDA build, since a tiny toy CNN doesn't need a GPU stack). The result was more interesting than a clean "yes, fixed" story: R² turned out to be numerically unstable for this reward signal, since the held-out reward variance is near zero (most ticks are just the -0.01 step penalty), so any real error explodes when divided by a near-zero baseline. That's reported plainly rather than swapped for a flattering metric. MAE, which is scale-stable, told the real story: more data alone made the linear model worse (0.245 average error; it overfits training scenes' specific pixel content and doesn't transfer to new scene visuals), while a CNN on the identical bigger dataset achieved a genuinely small held-out error (0.012). The honest conclusion is that the original result was as much a model-class problem as a data-volume one, a more useful, if less tidy, finding than the "more data fixes everything" hypothesis going in.

## A second, deeper bug in the reward-model dataset, found by checking a number that looked too clean

You asked for a full submission review against the brief, comparing a newer version of a competing team's repo whose README reports honest baseline comparisons (MAE improvement over a trivial mean-predictor) for their own reward model. Rather than assume ours would clear the same bar, I computed it directly, and it didn't. Both the linear and CNN MAE numbers reported above turned out to be worse than a trivial "always guess the training mean" baseline (0.0024 MAE), and digging into why revealed the actual problem: the held-out set was 100% one repeated value. Every one of 685 held-out frames carried exactly the -0.01 step penalty; no pickup, death, or success ever showed up in it at all.

The root cause was `dataset_emitter.py`'s sampling: it only captures every 4th tick (`TICK_STRIDE`), and pickup, death, and success are single-tick events, so roughly 3 in 4 of them were silently dropped by construction. Across a 30-scene dataset this meant only 6 success frames and 1 pickup frame survived in the training set, and none at all in the 7 held-out scenes, which made the entire evaluation, as originally run, incapable of measuring anything beyond "predicts roughly -0.01," regardless of model quality. R² being unstable, the earlier finding, turned out to be a symptom of this deeper problem, not the whole story.

Fixed by always capturing a frame on any tick with a non-default reward, regardless of stride, then regenerated the dataset and retrained both models. The corrected held-out set has real variance (reward std 0.10) and actual examples of the events that matter. Result: the CNN reaches a genuine positive R² of 0.32. Its aggregate MAE (0.020) still ties the trivial baseline almost exactly, because 685 of 697 held-out frames are still the common step-penalty case, where a trivial baseline is already good. Breaking the error out by event type is what actually shows the result: on the 12 rare frames that are pickup, death, or success, the CNN's MAE is 0.303 against 0.621 for blind guessing on the same frames, roughly twice as good specifically on the events a reward model needs to get right, though inconsistently (some individual predictions land within 0.1 of the true value, several are missed entirely). This breakdown is now a permanent part of `train_reward_model_cnn.py`'s output, not a one-off manual check, so future runs report it automatically rather than requiring the same investigation again.

Also folded into this pass: an embedded demo GIF at the top of the README (the same competing repo's `demo.webp` made the absence of one in ours obvious) and an explicit section mapping the brief's three "why this matters" bullets one to one, in their own order and wording, rather than leaving that connection implicit across scattered sections.

## Making the live demo actually live, and a round of external review

Two more things happened after the GIF/README pass above.

**The live dashboard.** You asked, in caps, for a real live dashboard instead of a static GIF: something clickable from the README that actually works, not a session-bound artifact link. Wrapped the browser prompt-to-scene demo as a standalone HTML document (`docs/index.html`) and enabled GitHub Pages on the `main` branch's `/docs` folder. The first deploy attempt failed three times in a row with a generic "Deployment failed, try again later" error from GitHub's own backend, including after a full delete-and-recreate of the Pages config; the build step succeeded every time, only the final deploy step errored. Rather than keep retrying blindly, published the same demo as a Claude Artifact as an immediate working fallback and asked you how you wanted to proceed. GitHub Pages resolved on its own a short while later and is now confirmed live and public (HTTP 200, no login wall), so the README's hero link points there directly.

**A round of external review.** You brought back a list of nine findings from a review pass on the repo, organized around three goals: clean, clear for the reviewer, and strong. Before doing anything, I diffed the assumptions against the actual code rather than taking them at face value, and one turned out to be more interesting than reported.

The review's guess for the one real bug (`run_maze_demo.py`'s memory use) was that `gif_builder.py` accumulated every frame in a Python list before writing. Measuring it directly showed that wasn't quite it: removing the list and streaming frames straight into imageio's GIF writer left peak memory almost unchanged (about 3.8GB either way). Isolating the render loop from the writer entirely showed the render loop alone holds steady around 80MB regardless of frame count, which meant the leak was inside the GIF writer itself, specifically imageio's legacy Pillow-based multi-writer, not in how frames were being fed to it. Switching to producing frames on a background thread and feeding them one at a time into PIL's own `Image.save(..., save_all=True, append_images=<generator>)` dropped peak memory to about 830MB for the same scene and frame count, verified with `/usr/bin/time -v` and by re-running both demo paths end to end.

The rest of the list held up as reported: the demo-count line said "6" when the code generates 7 environments; "recursive self-improvement" was the one term likely to read as an overclaim to this audience, renamed to "open-ended invention loop" everywhere it appeared in code and docs; the Quickstart command was buried below a long requirements table instead of being part of the README's first screen; and the exploit classifier, which is really the reward-hacking/verifier-robustness angle, was mentioned but not foregrounded as such. All fixed, verified, and pushed. `WRITEUP.md` also picked up its "optional session log" framing in this pass, so it stops competing with the README as the thing to evaluate.
