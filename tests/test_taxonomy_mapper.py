"""Tests for taxonomy_mapper — maps pipeline output to frontend enum values."""

import pytest
from service.taxonomy_mapper import map_to_frontend_taxonomy


# ---------------------------------------------------------------------------
# STANDUP_GAME category
# ---------------------------------------------------------------------------

class TestStandupGame:
    def test_default_maps_to_takedown(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Some Unknown Standup Move", "athlete in white gi")
        assert result["action"] == "takedown"

    def test_double_leg_takedown(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Double Leg Takedown", "athlete A")
        assert result["action"] == "takedown"
        assert result["technique"] == "double_leg"

    def test_single_leg_takedown(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Single Leg Takedown", "athlete B")
        assert result["action"] == "takedown"
        assert result["technique"] == "single_leg"

    def test_guard_pull(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Guard Pull", "athlete A")
        assert result["action"] == "guard_pull"

    def test_standing(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Standing Exchange", "athlete A")
        assert result["action"] == "standing"

    def test_clinch(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Clinch Work", "athlete A")
        assert result["action"] == "clinch"

    def test_hip_throw(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Hip Throw", "athlete A")
        assert result["action"] == "takedown"
        assert result["technique"] == "hip_throw"

    def test_snap_down(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Snap Down", "athlete A")
        assert result["action"] == "takedown"
        assert result["technique"] == "snap_down"

    def test_ankle_pick(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Ankle Pick", "athlete A")
        assert result["action"] == "takedown"
        assert result["technique"] == "ankle_pick"

    def test_arm_drag(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Arm Drag", "athlete A")
        assert result["action"] == "takedown"
        assert result["technique"] == "arm_drag"

    def test_body_lock_takedown(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Body Lock Takedown", "athlete A")
        assert result["action"] == "takedown"
        assert result["technique"] == "body_lock_takedown"

    def test_foot_sweep(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Foot Sweep", "athlete A")
        assert result["action"] == "takedown"
        assert result["technique"] == "foot_sweep_takedown"

    def test_takedown_defense(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Takedown Defense", "athlete A")
        assert result["action"] == "takedown_defense"

    def test_sprawl(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Sprawl", "athlete A")
        assert result["action"] == "takedown_defense"
        assert result["technique"] == "sprawl"


# ---------------------------------------------------------------------------
# GUARD_PLAY category
# ---------------------------------------------------------------------------

class TestGuardPlay:
    def test_default_maps_to_guard_bottom(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "Some Guard Work", "athlete A")
        assert result["action"] == "guard_bottom"

    def test_scissor_sweep(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "Scissor Sweep", "athlete A")
        assert result["action"] == "sweep"
        assert result["technique"] == "scissor_sweep"

    def test_butterfly_sweep(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "Butterfly Sweep", "athlete A")
        assert result["action"] == "sweep"
        assert result["technique"] == "butterfly_sweep"

    def test_hip_bump_sweep(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "Hip Bump Sweep", "athlete A")
        assert result["action"] == "sweep"
        assert result["technique"] == "hip_bump_sweep"

    def test_closed_guard(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "Closed Guard", "athlete A")
        assert result["action"] == "guard_bottom"
        assert result["technique"] == "closed_guard"

    def test_de_la_riva_guard(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "De La Riva Guard", "athlete A")
        assert result["action"] == "guard_bottom"
        assert result["technique"] == "de_la_riva_guard"

    def test_guard_retention(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "Guard Retention", "athlete A")
        assert result["action"] == "guard_retention"

    def test_guard_pull_from_guard_play(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "Guard Pull to Closed Guard", "athlete A")
        assert result["action"] == "guard_pull"

    def test_berimbolo(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "Berimbolo", "athlete A")
        assert result["action"] == "sweep"
        assert result["technique"] == "berimbolo"

    def test_x_guard_sweep(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "X Guard Sweep", "athlete A")
        assert result["action"] == "sweep"
        assert result["technique"] == "x_guard_sweep"

    def test_spider_guard(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "Spider Guard", "athlete A")
        assert result["action"] == "guard_bottom"
        assert result["technique"] == "spider_guard"

    def test_half_guard(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "Half Guard", "athlete A")
        assert result["action"] == "half_guard_bottom"
        assert result["technique"] == "half_guard"

    def test_deep_half_guard(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "Deep Half Guard", "athlete A")
        assert result["action"] == "half_guard_bottom"
        assert result["technique"] == "deep_half_guard"


# ---------------------------------------------------------------------------
# GUARD_PASSING category
# ---------------------------------------------------------------------------

class TestGuardPassing:
    def test_default_maps_to_pass(self):
        result = map_to_frontend_taxonomy("GUARD_PASSING", "Some Pass Attempt", "athlete A")
        assert result["action"] == "pass"

    def test_knee_cut_pass(self):
        result = map_to_frontend_taxonomy("GUARD_PASSING", "Knee Cut Pass", "athlete A")
        assert result["action"] == "pass"
        assert result["technique"] == "knee_cut_pass"

    def test_toreando_pass(self):
        result = map_to_frontend_taxonomy("GUARD_PASSING", "Toreando Pass", "athlete A")
        assert result["action"] == "pass"
        assert result["technique"] == "toreando_pass"

    def test_leg_drag(self):
        result = map_to_frontend_taxonomy("GUARD_PASSING", "Leg Drag", "athlete A")
        assert result["action"] == "pass"
        assert result["technique"] == "leg_drag"

    def test_pressure_pass(self):
        result = map_to_frontend_taxonomy("GUARD_PASSING", "Pressure Pass", "athlete A")
        assert result["action"] == "pass"
        assert result["technique"] == "pressure_pass"

    def test_body_lock_pass(self):
        result = map_to_frontend_taxonomy("GUARD_PASSING", "Body Lock Pass", "athlete A")
        assert result["action"] == "pass"
        assert result["technique"] == "body_lock_pass"

    def test_stack_pass(self):
        result = map_to_frontend_taxonomy("GUARD_PASSING", "Stack Pass", "athlete A")
        assert result["action"] == "pass"
        assert result["technique"] == "stack_pass"


# ---------------------------------------------------------------------------
# POSITIONAL_DOMINANCE category
# ---------------------------------------------------------------------------

class TestPositionalDominance:
    def test_mount(self):
        result = map_to_frontend_taxonomy("POSITIONAL_DOMINANCE", "Mount Control", "athlete A")
        assert result["action"] == "mount"

    def test_side_control(self):
        result = map_to_frontend_taxonomy("POSITIONAL_DOMINANCE", "Side Control", "athlete A")
        assert result["action"] == "side_control"

    def test_back_control(self):
        result = map_to_frontend_taxonomy("POSITIONAL_DOMINANCE", "Back Control", "athlete A")
        assert result["action"] == "back_control"

    def test_knee_on_belly(self):
        result = map_to_frontend_taxonomy("POSITIONAL_DOMINANCE", "Knee on Belly", "athlete A")
        assert result["action"] == "knee_on_belly"

    def test_north_south(self):
        result = map_to_frontend_taxonomy("POSITIONAL_DOMINANCE", "North South", "athlete A")
        assert result["action"] == "north_south"

    def test_back_take(self):
        result = map_to_frontend_taxonomy("POSITIONAL_DOMINANCE", "Back Take", "athlete A")
        assert result["action"] == "back_control"
        assert result["technique"] == "back_take"

    def test_mount_transition(self):
        result = map_to_frontend_taxonomy("POSITIONAL_DOMINANCE", "Mount Transition", "athlete A")
        assert result["action"] == "mount"
        assert result["technique"] == "mount_transition"

    def test_default_maps_to_side_control(self):
        result = map_to_frontend_taxonomy("POSITIONAL_DOMINANCE", "Unknown Position", "athlete A")
        assert result["action"] == "side_control"

    def test_turtle_top(self):
        result = map_to_frontend_taxonomy("POSITIONAL_DOMINANCE", "Turtle Control", "athlete A")
        assert result["action"] == "turtle_top"


# ---------------------------------------------------------------------------
# SUBMISSION_OFFENSE category
# ---------------------------------------------------------------------------

class TestSubmissionOffense:
    def test_default_maps_to_submission_attempt(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Some Submission", "athlete A")
        assert result["action"] == "submission_attempt"

    def test_rear_naked_choke(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Rear Naked Choke", "athlete A")
        assert result["action"] == "submission_attempt"
        assert result["technique"] == "rear_naked_choke"

    def test_rnc_alias(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "RNC", "athlete A")
        assert result["technique"] == "rear_naked_choke"

    def test_armbar(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Armbar", "athlete A")
        assert result["action"] == "submission_attempt"
        assert result["technique"] == "armbar"

    def test_triangle_choke(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Triangle Choke", "athlete A")
        assert result["technique"] == "triangle_choke"

    def test_kimura(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Kimura", "athlete A")
        assert result["technique"] == "kimura"

    def test_americana(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Americana", "athlete A")
        assert result["technique"] == "americana"

    def test_guillotine(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Guillotine Choke", "athlete A")
        assert result["technique"] == "guillotine"

    def test_darce(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "D'Arce Choke", "athlete A")
        assert result["technique"] == "darce"

    def test_heel_hook(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Heel Hook", "athlete A")
        assert result["technique"] == "heel_hook"

    def test_inside_heel_hook(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Inside Heel Hook", "athlete A")
        assert result["technique"] == "inside_heel_hook"

    def test_outside_heel_hook(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Outside Heel Hook", "athlete A")
        assert result["technique"] == "outside_heel_hook"

    def test_straight_ankle_lock(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Straight Ankle Lock", "athlete A")
        assert result["technique"] == "straight_ankle_lock"

    def test_toe_hold(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Toe Hold", "athlete A")
        assert result["technique"] == "toe_hold"

    def test_knee_bar(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Knee Bar", "athlete A")
        assert result["technique"] == "knee_bar"

    def test_omoplata(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Omoplata", "athlete A")
        assert result["technique"] == "omoplata"

    def test_wrist_lock(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Wrist Lock", "athlete A")
        assert result["technique"] == "wrist_lock"

    def test_bow_and_arrow(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Bow and Arrow Choke", "athlete A")
        assert result["technique"] == "bow_and_arrow"

    def test_ezekiel(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Ezekiel Choke", "athlete A")
        assert result["technique"] == "ezekiel"

    def test_anaconda(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Anaconda Choke", "athlete A")
        assert result["technique"] == "anaconda"

    def test_arm_triangle(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Arm Triangle", "athlete A")
        assert result["technique"] == "arm_triangle"

    def test_calf_slicer(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Calf Slicer", "athlete A")
        assert result["technique"] == "calf_slicer"

    def test_loop_choke(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "Loop Choke", "athlete A")
        assert result["technique"] == "loop_choke"


# ---------------------------------------------------------------------------
# DEFENSE_ESCAPES category
# ---------------------------------------------------------------------------

class TestDefenseEscapes:
    def test_default_maps_to_escape(self):
        result = map_to_frontend_taxonomy("DEFENSE_ESCAPES", "Some Defense", "athlete A")
        assert result["action"] == "escape"

    def test_submission_defense(self):
        result = map_to_frontend_taxonomy("DEFENSE_ESCAPES", "Submission Defense", "athlete A")
        assert result["action"] == "submission_defense"

    def test_bridge_escape(self):
        result = map_to_frontend_taxonomy("DEFENSE_ESCAPES", "Bridge Escape", "athlete A")
        assert result["action"] == "escape"
        assert result["technique"] == "bridge_escape"

    def test_hip_escape(self):
        result = map_to_frontend_taxonomy("DEFENSE_ESCAPES", "Hip Escape", "athlete A")
        assert result["action"] == "escape"
        assert result["technique"] == "hip_escape"

    def test_elbow_escape(self):
        result = map_to_frontend_taxonomy("DEFENSE_ESCAPES", "Elbow Escape", "athlete A")
        assert result["action"] == "escape"
        assert result["technique"] == "elbow_escape"

    def test_trap_and_roll(self):
        result = map_to_frontend_taxonomy("DEFENSE_ESCAPES", "Trap and Roll", "athlete A")
        assert result["action"] == "escape"
        assert result["technique"] == "trap_and_roll"

    def test_granby_roll(self):
        result = map_to_frontend_taxonomy("DEFENSE_ESCAPES", "Granby Roll", "athlete A")
        assert result["action"] == "escape"
        assert result["technique"] == "granby_roll"

    def test_inversion_escape(self):
        result = map_to_frontend_taxonomy("DEFENSE_ESCAPES", "Inversion Escape", "athlete A")
        assert result["action"] == "escape"
        assert result["technique"] == "inversion_escape"

    def test_reversal(self):
        result = map_to_frontend_taxonomy("DEFENSE_ESCAPES", "Reversal", "athlete A")
        assert result["action"] == "reversal"

    def test_scramble(self):
        result = map_to_frontend_taxonomy("DEFENSE_ESCAPES", "Scramble", "athlete A")
        assert result["action"] == "scramble"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_category(self):
        result = map_to_frontend_taxonomy("TOTALLY_UNKNOWN", "Something", "athlete A")
        assert result["action"] == "other"
        assert result["technique"] == "other"

    def test_case_insensitive_technique(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "rear naked choke", "athlete A")
        assert result["technique"] == "rear_naked_choke"

    def test_mixed_case_technique(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "ARMBAR", "athlete A")
        assert result["technique"] == "armbar"

    def test_empty_technique(self):
        result = map_to_frontend_taxonomy("GUARD_PASSING", "", "athlete A")
        assert result["action"] == "pass"
        assert result["technique"] == "other"

    def test_none_technique(self):
        result = map_to_frontend_taxonomy("GUARD_PASSING", None, "athlete A")
        assert result["action"] == "pass"
        assert result["technique"] == "other"

    def test_alias_slx(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "SLX Guard", "athlete A")
        assert result["technique"] == "single_leg_x"

    def test_alias_dlr(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "DLR Guard", "athlete A")
        assert result["technique"] == "de_la_riva_guard"

    def test_alias_rdlr(self):
        result = map_to_frontend_taxonomy("GUARD_PLAY", "RDLR", "athlete A")
        assert result["technique"] == "reverse_de_la_riva"

    def test_result_has_both_keys(self):
        result = map_to_frontend_taxonomy("STANDUP_GAME", "Double Leg", "athlete A")
        assert "action" in result
        assert "technique" in result

    def test_extra_whitespace(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "  Rear Naked Choke  ", "athlete A")
        assert result["technique"] == "rear_naked_choke"

    def test_technique_with_hyphens(self):
        result = map_to_frontend_taxonomy("SUBMISSION_OFFENSE", "North-South Choke", "athlete A")
        assert result["technique"] == "north_south_choke"
