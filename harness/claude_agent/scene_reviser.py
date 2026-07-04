"""
Tier-2 LLM scene revision for the invention loop (harness/invention/loop.py):
given a parent scene, ask Claude to revise it into something more
interesting, rather than mutating it via the fixed code-based move registry
(harness/invention/mutation.py). Structurally identical to
claude_agent/scene_generator.py's proven pattern: same forced (non-strict)
tool call and schema (the objective predicate is recursive, so structured
outputs don't apply, see scene_generator.py's docstring), same defensive
JSON-string parsing for the "objective returned as a JSON-encoded string"
quirk observed live during that module's own testing, same bounded
validator-rejection retry loop.

Revised scenes get metadata.generator="freeform_llm". Unlike
mutation.propose_mutant, which deliberately preserves the parent's original
generator value since a code-based mutation isn't a new kind of authorship,
a Claude revision genuinely is LLM-authored content, even though it started
from an existing scene rather than a blank prompt.

`make_reviser(client)` returns a `reviser(parent, rng) -> (scene_or_None,
move_name)` closure matching the contract harness/invention/loop.py's
explore_propose expects, returning None on any failure (no key, API error,
or every retry rejected) so the loop falls back to
mutation.propose_mutant_with_retry automatically, exactly like
run_claude_demo.py's relationship to run_demo.py.
"""

from __future__ import annotations

import json

import anthropic

from harness.claude_agent.scene_generator import SCENE_TOOL
from harness.generation.validator import validate_and_repair

MODEL = "claude-opus-4-8"
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_SECONDS = 60.0  # fail loud on a blocked/unreachable network rather than hang on the SDK's own, much longer, default

SYSTEM_PROMPT = """You revise EXISTING 2D platformer scene specifications for a physics-based \
game harness, to make them more interesting to play -- not generating from scratch.

World convention: origin top-left, x increases right, y increases DOWN; gravity is [0, 900].
Primitive library (the only object `type`s allowed): platform, ramp, door, key, hazard, pickup.

You will be given a complete scene as JSON. Revise it with one or more of these goals in mind:
- Make the honest solve meaningfully longer or more involved than a straight line to the \
  goal -- without making it unsolvable. Prefer moderate difficulty increases (a few more \
  obstacles, a longer route) over extreme ones (do not bury the goal behind an impossible gap).
- Introduce objective-predicate variety beyond a single flat AND: an {"or": [...]} branch \
  (an alternate way to win) or a {"not": {"event": "collision", "id": "..."}} clause (win \
  without ever touching a specific hazard) are both valid and often make for a more \
  interesting scene than another AND term.
- Keep every id reference consistent (door key_id <-> key key_id, objective ids <-> object \
  ids) and keep a walkable/jumpable route from player_start to everything the objective \
  positively requires -- a validator checks all of this and will reject an unsolvable scene.
- `metadata.generator` must be "freeform_llm" (this is a revision, but you produced the \
  final content).

Call emit_scene with the complete REVISED scene (not a diff/patch). If a previous attempt \
was rejected, the rejection reasons will be given to you -- fix exactly those issues."""


def revise_scene_via_claude(parent: dict, client: anthropic.Anthropic | None = None):
    """Returns (scene_or_none, report), same contract as
    scene_generator.generate_scene_via_claude."""
    client = client or anthropic.Anthropic()
    messages = [{
        "role": "user",
        "content": f"Revise this scene:\n{json.dumps(parent, default=str)}",
    }]

    report = {"accepted": False}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[SCENE_TOOL],
            tool_choice={"type": "tool", "name": "emit_scene"},
            messages=messages,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            report = {"accepted": False, "attempts": attempt, "error": "no tool_use block in response"}
            break

        scene = dict(tool_use.input)
        if isinstance(scene.get("objective"), str):
            try:
                scene["objective"] = json.loads(scene["objective"])
            except json.JSONDecodeError:
                pass
        scene.setdefault("id", f"revised_{parent.get('id', 'scene')}_{attempt}")
        scene.setdefault("seed", None)
        scene.setdefault("metadata", {})
        scene["metadata"]["generator"] = "freeform_llm"
        scene["metadata"]["prompt"] = f"revision of {parent.get('id')}"

        validated, validation_report = validate_and_repair(scene)
        validation_report["attempts"] = attempt
        if validated is not None:
            return validated, validation_report

        report = validation_report
        if attempt == MAX_ATTEMPTS:
            break

        errors = (
            validation_report.get("structural_errors")
            or validation_report.get("referential_errors")
            or validation_report.get("overlap_errors")
            or validation_report.get("reachability_errors")
            or ["unknown validation failure"]
        )
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": f"Scene rejected by validator: {errors}. Fix exactly these issues and call emit_scene again.",
                "is_error": True,
            }],
        })

    return None, report


def make_reviser(client: anthropic.Anthropic, verbose: bool = True):
    """`verbose=True` prints why a revision attempt fell back to mutation.
    The fallback itself was previously silent (any exception, or an exhausted
    validator-retry ladder, just returned None with no trace), which made a
    100%-fallback run indistinguishable from a working-but-unlucky one. Set
    False to suppress (e.g. if a caller wants to inspect failures itself)."""
    def reviser(parent: dict, rng):
        try:
            scene, report = revise_scene_via_claude(parent, client)
        except Exception as e:
            if verbose:
                print(f"    [claude_revision failed: {type(e).__name__}: {e}]", flush=True)
            return None, None
        if scene is None:
            if verbose:
                reason = report.get("error") or report.get("structural_errors") or report.get("referential_errors") \
                    or report.get("overlap_errors") or report.get("reachability_errors") or "unknown"
                print(f"    [claude_revision rejected after {report.get('attempts', '?')} attempt(s): {reason}]", flush=True)
            return None, None
        return scene, "claude_revision"
    return reviser
