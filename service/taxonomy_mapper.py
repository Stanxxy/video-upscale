"""Map whole-video-analysis pipeline output to frontend ActionType/TechniqueType enums."""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Valid frontend enum values
# ---------------------------------------------------------------------------

VALID_ACTIONS = frozenset({
    "guard_top", "guard_bottom", "half_guard_top", "half_guard_bottom",
    "side_control", "mount", "back_control", "knee_on_belly",
    "turtle_top", "turtle_bottom", "north_south",
    "sweep", "pass", "reversal", "scramble",
    "submission_attempt", "submission_defense",
    "takedown", "takedown_defense", "guard_pull", "guard_retention",
    "escape", "standing", "clinch", "reset", "other",
})

VALID_TECHNIQUES = frozenset({
    # Chokes
    "rear_naked_choke", "guillotine", "darce", "anaconda", "arm_triangle",
    "ezekiel", "loop_choke", "bow_and_arrow", "clock_choke",
    "baseball_bat_choke", "north_south_choke", "paper_cutter",
    "cross_collar_choke", "triangle_choke",
    # Arm locks
    "armbar", "kimura", "americana", "omoplata", "wrist_lock",
    "bicep_slicer", "tarikoplata", "baratoplata",
    # Leg locks
    "heel_hook", "inside_heel_hook", "outside_heel_hook",
    "straight_ankle_lock", "toe_hold", "knee_bar", "calf_slicer",
    "estima_lock",
    # Sweeps
    "scissor_sweep", "flower_sweep", "hip_bump_sweep", "pendulum_sweep",
    "butterfly_sweep", "x_guard_sweep", "single_leg_x_sweep",
    "hook_sweep", "lasso_sweep", "spider_sweep", "de_la_riva_sweep",
    "berimbolo", "sickle_sweep",
    # Passes
    "toreando_pass", "knee_cut_pass", "leg_drag", "over_under_pass",
    "double_under_pass", "stack_pass", "pressure_pass", "long_step_pass",
    "smash_pass", "body_lock_pass", "x_pass",
    # Takedowns
    "single_leg", "double_leg", "high_crotch", "ankle_pick", "snap_down",
    "arm_drag", "body_lock_takedown", "foot_sweep_takedown", "hip_throw",
    "trip", "suplex", "fireman_carry",
    # Escapes
    "bridge_escape", "hip_escape", "elbow_escape", "trap_and_roll",
    "granby_roll", "inversion_escape", "standing_escape",
    # Guard types
    "closed_guard", "open_guard", "butterfly_guard", "spider_guard",
    "lasso_guard", "de_la_riva_guard", "reverse_de_la_riva", "x_guard",
    "single_leg_x", "half_guard", "deep_half_guard", "z_guard",
    "rubber_guard", "worm_guard", "lapel_guard", "seated_guard",
    # Transitions / misc
    "back_take", "mount_transition", "side_control_transition",
    "leg_entanglement", "sprawl", "underhook", "overhook", "frame",
    "posture_break", "grip_break", "other",
})

# ---------------------------------------------------------------------------
# Aliases: common abbreviations / alternate names -> canonical technique
# ---------------------------------------------------------------------------

TECHNIQUE_ALIASES: dict[str, str] = {
    "rnc": "rear_naked_choke",
    "rear_naked": "rear_naked_choke",
    "slx": "single_leg_x",
    "slx_guard": "single_leg_x",
    "single_leg_x_guard": "single_leg_x",
    "dlr": "de_la_riva_guard",
    "dlr_guard": "de_la_riva_guard",
    "de_la_riva": "de_la_riva_guard",
    "rdlr": "reverse_de_la_riva",
    "rdlr_guard": "reverse_de_la_riva",
    "reverse_dlr": "reverse_de_la_riva",
    "reverse_de_la_riva_guard": "reverse_de_la_riva",
    "darce_choke": "darce",
    "d_arce_choke": "darce",
    "d_arce": "darce",
    "anaconda_choke": "anaconda",
    "guillotine_choke": "guillotine",
    "ezekiel_choke": "ezekiel",
    "loop_choke": "loop_choke",
    "bow_and_arrow_choke": "bow_and_arrow",
    "clock_choke": "clock_choke",
    "baseball_bat": "baseball_bat_choke",
    "north_south_choke": "north_south_choke",
    "paper_cutter_choke": "paper_cutter",
    "cross_collar": "cross_collar_choke",
    "cross_choke": "cross_collar_choke",
    "arm_triangle_choke": "arm_triangle",
    "head_and_arm_choke": "arm_triangle",
    "kata_gatame": "arm_triangle",
    "single_leg_takedown": "single_leg",
    "double_leg_takedown": "double_leg",
    "high_crotch_takedown": "high_crotch",
    "foot_sweep": "foot_sweep_takedown",
    "osoto_gari": "foot_sweep_takedown",
    "ouchi_gari": "trip",
    "kouchi_gari": "trip",
    "ankle_lock": "straight_ankle_lock",
    "straight_footlock": "straight_ankle_lock",
    "kneebar": "knee_bar",
    "toehold": "toe_hold",
    "heelhook": "heel_hook",
    "inside_heelhook": "inside_heel_hook",
    "outside_heelhook": "outside_heel_hook",
    "knee_cut": "knee_cut_pass",
    "knee_slice": "knee_cut_pass",
    "knee_slice_pass": "knee_cut_pass",
    "toreando": "toreando_pass",
    "toreano_pass": "toreando_pass",
    "bullfighter_pass": "toreando_pass",
    "over_under": "over_under_pass",
    "double_under": "double_under_pass",
    "long_step": "long_step_pass",
    "body_lock_pass": "body_lock_pass",
    "x_pass": "x_pass",
    "shrimp": "hip_escape",
    "upa": "trap_and_roll",
    "bumpa": "trap_and_roll",
    "seated": "seated_guard",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(text: Optional[str]) -> str:
    """Normalize freeform technique text to snake_case for matching."""
    if not text:
        return ""
    text = text.strip().lower()
    # Replace hyphens and apostrophes with spaces, then collapse
    text = re.sub(r"['\u2019]", "", text)   # remove apostrophes (D'Arce -> DArce)
    text = re.sub(r"[-]", " ", text)         # hyphens to spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)  # strip other non-alnum
    text = re.sub(r"\s+", "_", text.strip()) # spaces to underscores
    return text


def _resolve_technique(normalized: str) -> str:
    """Resolve a normalized technique string to a valid frontend technique enum value."""
    if not normalized:
        return "other"
    # Direct match
    if normalized in VALID_TECHNIQUES:
        return normalized
    # Alias match
    if normalized in TECHNIQUE_ALIASES:
        return TECHNIQUE_ALIASES[normalized]
    # Partial / substring matching for common patterns
    # Try removing trailing words like "_choke", "_pass", etc. if already covered
    return "other"


# ---------------------------------------------------------------------------
# Action inference by category
# ---------------------------------------------------------------------------

# Keywords in the normalized technique that override the default action for a category
_STANDUP_ACTION_KEYWORDS: dict[str, str] = {
    "guard_pull": "guard_pull",
    "standing": "standing",
    "clinch": "clinch",
    "takedown_defense": "takedown_defense",
    "sprawl": "takedown_defense",
}

_GUARD_PLAY_SWEEP_KEYWORDS = frozenset({
    "sweep", "berimbolo",
})

_GUARD_PLAY_GUARD_RETENTION_KEYWORDS = frozenset({
    "guard_retention", "retention",
})

_GUARD_PLAY_GUARD_PULL_KEYWORDS = frozenset({
    "guard_pull",
})

# Guard types that map to half_guard_bottom
_HALF_GUARD_TECHNIQUES = frozenset({
    "half_guard", "deep_half_guard", "z_guard",
})

_POSITIONAL_ACTION_KEYWORDS: dict[str, str] = {
    "mount": "mount",
    "side_control": "side_control",
    "back_control": "back_control",
    "back_take": "back_control",
    "knee_on_belly": "knee_on_belly",
    "north_south": "north_south",
    "turtle": "turtle_top",
}

_DEFENSE_ACTION_KEYWORDS: dict[str, str] = {
    "submission_defense": "submission_defense",
    "reversal": "reversal",
    "scramble": "scramble",
}


def _infer_standup_action(normalized: str) -> str:
    """Infer action for STANDUP_GAME category."""
    for keyword, action in _STANDUP_ACTION_KEYWORDS.items():
        if keyword in normalized:
            return action
    return "takedown"


def _infer_guard_play_action(normalized: str, technique: str) -> str:
    """Infer action for GUARD_PLAY category."""
    for kw in _GUARD_PLAY_GUARD_PULL_KEYWORDS:
        if kw in normalized:
            return "guard_pull"
    for kw in _GUARD_PLAY_GUARD_RETENTION_KEYWORDS:
        if kw in normalized:
            return "guard_retention"
    for kw in _GUARD_PLAY_SWEEP_KEYWORDS:
        if kw in normalized:
            return "sweep"
    # Check if the resolved technique is a sweep
    if technique.endswith("_sweep") or technique == "berimbolo":
        return "sweep"
    # Check for half guard variants
    if technique in _HALF_GUARD_TECHNIQUES:
        return "half_guard_bottom"
    # Check if technique is a guard type -> guard_bottom
    if technique.endswith("_guard") or technique in VALID_TECHNIQUES and "guard" in technique:
        return "guard_bottom"
    return "guard_bottom"


def _infer_positional_action(normalized: str) -> str:
    """Infer action for POSITIONAL_DOMINANCE category."""
    for keyword, action in _POSITIONAL_ACTION_KEYWORDS.items():
        if keyword in normalized:
            return action
    return "side_control"


def _infer_defense_action(normalized: str) -> str:
    """Infer action for DEFENSE_ESCAPES category."""
    for keyword, action in _DEFENSE_ACTION_KEYWORDS.items():
        if keyword in normalized:
            return action
    return "escape"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_CATEGORY_ACTION_MAP = {
    "STANDUP_GAME": _infer_standup_action,
    "GUARD_PASSING": lambda _n: "pass",
    "SUBMISSION_OFFENSE": lambda _n: "submission_attempt",
}


def map_to_frontend_taxonomy(
    category: str,
    specific_technique: Optional[str],
    role: str,
) -> dict[str, str]:
    """Map pipeline output fields to frontend-compatible action and technique values.

    Args:
        category: Pipeline category (e.g. "STANDUP_GAME", "GUARD_PLAY", etc.)
        specific_technique: Freeform technique string from the model
        role: Freeform role string (currently unused but reserved for future)

    Returns:
        {"action": <valid action enum>, "technique": <valid technique enum>}
    """
    normalized = _normalize(specific_technique)
    technique = _resolve_technique(normalized)

    # Determine action based on category
    if category in _CATEGORY_ACTION_MAP:
        action = _CATEGORY_ACTION_MAP[category](normalized)
    elif category == "GUARD_PLAY":
        action = _infer_guard_play_action(normalized, technique)
    elif category == "POSITIONAL_DOMINANCE":
        action = _infer_positional_action(normalized)
    elif category == "DEFENSE_ESCAPES":
        action = _infer_defense_action(normalized)
    else:
        action = "other"

    return {"action": action, "technique": technique}
