import pytest

from eval.adcc.data_prep.apply_gracie_mapping import (
    MappingError,
    UnknownSequenceTypeError,
    apply_mapping,
    bucket_position,
    compat_classes_for,
    load_mapping,
)

MAPPING = load_mapping()  # the real, checked-in v1.yaml — no synthetic mapping needed for these tests


def _event(sequence_type, starting, movements, ending):
    return {
        "timestamp": "01:00", "sequenceType": sequence_type,
        "startingPosition": starting, "movements": movements, "endingPosition": ending,
    }


class TestUnknownSequenceType:
    def test_raises_on_unmapped_sequence_type(self):
        event = _event("BARREL ROLL", "STANDING", "N/A", "STANDING")
        with pytest.raises(UnknownSequenceTypeError):
            apply_mapping(event, None, MAPPING)


class TestNativeSequenceTypes:
    def test_guard_pass(self):
        event = _event("GUARD PASS", "OPEN GUARD", "KNEE CUT", "SIDE CONTROL")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "guard_pass"
        assert out["outcome"] == "successful"
        assert out["scoreable"] is True
        assert out["track"] == "hard"

    def test_guard_retention_is_kept_as_unsuccessful_guard_pass(self):
        event = _event("GUARD RETENTION", "OPEN GUARD", "FRAME", "OPEN GUARD")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "guard_pass"
        assert out["outcome"] == "unsuccessful"
        assert out["scoreable"] is True

    def test_takedown_defense_is_kept_as_unsuccessful_takedown(self):
        event = _event("TAKEDOWN DEFENSE", "STANDING", "SPRAWL", "STANDING")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "takedown_attempt"
        assert out["outcome"] == "unsuccessful"

    def test_position_escape(self):
        event = _event("POSITION ESCAPE", "SIDE CONTROL", "HIP ESCAPE", "OPEN GUARD")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "escape"
        assert out["outcome"] == "successful"


class TestSubmissionSubtype:
    def test_choke_via_ending_position(self):
        event = _event("SUBMISSION", "BACK", "N/A", "RNC")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "submission_choke"
        assert out["outcome"] == "successful"
        assert out["track"] == "hard"

    def test_arm_lock_via_ending_position(self):
        event = _event("SUBMISSION", "MOUNT", "N/A", "ARM BAR")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "submission_arm_lock"

    def test_leg_lock_via_ending_position(self):
        event = _event("SUBMISSION", "50-50", "N/A", "INSIDE HEEL HOOK")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "submission_leg_lock"

    def test_attempt_prefix_makes_outcome_unsuccessful(self):
        event = _event("SUBMISSION", "BACK", "ATTEMPT (RNC)", "RNC")
        out = apply_mapping(event, None, MAPPING)
        assert out["outcome"] == "unsuccessful"

    def test_other_ending_falls_back_to_movements_keyword(self):
        event = _event("SUBMISSION", "SIDE CONTROL", "ATTEMPT (WRIST LOCK)", "OTHER")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "submission_arm_lock"
        assert out["outcome"] == "unsuccessful"

    def test_truly_ambiguous_other_becomes_unclear_soft_tier(self):
        event = _event("SUBMISSION", "SIDE CONTROL", "DIG", "OTHER")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "submission_unclear"
        assert out["track"] == "soft"

    def test_submission_compat_is_family_wide_regardless_of_subtype(self):
        """§5.1: a gold SUBMISSION row is feasible against ANY submission_*
        prediction class, even an 'unclear'-subtype one."""
        clear = apply_mapping(_event("SUBMISSION", "BACK", "N/A", "RNC"), None, MAPPING)
        unclear = apply_mapping(_event("SUBMISSION", "SIDE CONTROL", "DIG", "OTHER"), None, MAPPING)
        for normalized in (clear, unclear):
            compat = compat_classes_for(normalized, MAPPING)
            assert compat == frozenset({"submission_choke", "submission_arm_lock", "submission_leg_lock"})


class TestPositionEntrySplit:
    def test_ending_back_mints_back_take(self):
        event = _event("POSITION ENTRY", "TURTLE", "WEDGE BACK TAKE", "BACK")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "back_take"
        assert out["scoreable"] is True
        assert out["track"] == "hard"

    def test_repeat_back_take_is_deduped_against_prior_event(self):
        """§4.3 rule 1 de-dup: if the immediately prior gold event already
        ended in BACK, do not mint a second back_take."""
        prev_event = _event("POSITION ENTRY", "TURTLE", "HOOK (INSERT TOP)", "BACK")
        repeat_event = _event("POSITION ENTRY", "BACK", "SWITCH SIDES", "BACK")
        out = apply_mapping(repeat_event, prev_event, MAPPING)
        assert out is None  # dropped, not a second back_take

    def test_back_take_still_mints_if_prior_did_not_end_in_back(self):
        prev_event = _event("POSITION ESCAPE", "SIDE CONTROL", "HIP ESCAPE", "OPEN GUARD")
        event = _event("POSITION ENTRY", "OPEN GUARD", "BERIMBOLO", "BACK")
        out = apply_mapping(event, prev_event, MAPPING)
        assert out["action_class"] == "back_take"

    def test_no_state_change_administrative_is_dropped(self):
        event = _event("POSITION ENTRY", "OPEN GUARD", "N/A", "OPEN GUARD")
        out = apply_mapping(event, None, MAPPING)
        assert out is None

    def test_no_state_change_with_real_movements_is_transition_not_dropped(self):
        event = _event("POSITION ENTRY", "OPEN GUARD", "GRIP (COLLAR)", "OPEN GUARD")
        out = apply_mapping(event, None, MAPPING)
        assert out is not None
        assert out["action_class"] == "transition"
        assert out["scoreable"] is False
        assert out["track"] == "soft"

    def test_leg_entanglement_entry_is_non_scoreable_transition(self):
        event = _event("POSITION ENTRY", "OPEN GUARD", "K-GUARD ENTRY", "K-GUARD")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "transition"
        assert out["scoreable"] is False
        assert out["track"] == "soft"

    def test_guard_starting_ending_pin_target_defaults_to_guard_pass(self):
        event = _event("POSITION ENTRY", "OPEN GUARD", "LOOSE PASSING, KNEE PULL", "SIDE CONTROL")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "guard_pass"

    def test_guard_starting_ending_pin_target_with_sweep_keyword_is_sweep(self):
        event = _event("POSITION ENTRY", "CLOSED GUARD", "SIT DOWN, SWEEP (YOKO SUMI GAESHI)", "MOUNT")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "sweep"
        assert out["scoreable"] is True

    def test_generic_catch_all_is_transition(self):
        event = _event("POSITION ENTRY", "STANDING", "CIRCLE", "TURTLE")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "transition"
        assert out["scoreable"] is False


class TestGuardPassOverGeneralizationFix:
    """A1 (plan Part A1, BLOCKING): a pin-ending POSITION ENTRY row must NOT
    become guard_pass unless the start is a genuine guard-family position
    (top player COMPLETING a pass). These are the exact false-positive rows
    named in the plan — none of them may resolve to guard_pass."""

    def test_mount_to_mount_is_not_guard_pass(self):
        event = _event("POSITION ENTRY", "MOUNT", "ADJUST", "MOUNT")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "transition"
        assert out["scoreable"] is False

    def test_side_control_to_mount_is_not_guard_pass(self):
        event = _event("POSITION ENTRY", "SIDE CONTROL", "KNEE CUT", "MOUNT")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "transition"
        assert out["scoreable"] is False

    def test_north_south_to_north_south_is_not_guard_pass(self):
        event = _event("POSITION ENTRY", "NORTH SOUTH", "SWITCH HIPS", "NORTH SOUTH")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "transition"
        assert out["scoreable"] is False

    def test_back_to_mount_is_not_guard_pass(self):
        event = _event("POSITION ENTRY", "BACK", "SLIDE TO MOUNT", "MOUNT")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "transition"
        assert out["scoreable"] is False

    def test_turtle_to_mount_is_not_guard_pass(self):
        """This row (TURTLE->MOUNT) was the OLD test's asserted true-positive
        before the A1 fix — turtle breakdown is not a guard pass."""
        event = _event("POSITION ENTRY", "TURTLE", "LOOSE PASSING", "MOUNT")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "transition"
        assert out["scoreable"] is False

    def test_front_headlock_to_side_control_is_not_guard_pass(self):
        event = _event("POSITION ENTRY", "FRONT HEADLOCK", "SCRAMBLE", "SIDE CONTROL")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "transition"
        assert out["scoreable"] is False

    def test_stack_to_north_south_is_not_guard_pass(self):
        event = _event("POSITION ENTRY", "STACK", "ADJUST", "NORTH SOUTH")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "transition"
        assert out["scoreable"] is False

    def test_open_guard_to_side_control_top_initiating_is_true_positive_guard_pass(self):
        """The genuine true-positive case survives: guard-family start,
        top-initiating pass footwork/grip vocabulary, pin-target end."""
        event = _event("POSITION ENTRY", "OPEN GUARD", "KNEE CUT, LEG DRAG", "SIDE CONTROL")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "guard_pass"
        assert out["scoreable"] is True
        assert out["track"] == "hard"

    def test_same_position_dedup_is_not_a_pass(self):
        """Same-position dedup analogous to the back_take immediate-repeat
        dedup: start==end pin can never be 'completing a pass'. Uses a
        non-administrative movements string so this row reaches rule 4
        (rather than being dropped earlier by rule 2's admin-movements
        no-state-change drop)."""
        event = _event("POSITION ENTRY", "SIDE CONTROL", "GRIP FIGHT", "SIDE CONTROL")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "transition"
        assert out["scoreable"] is False


class TestTwisterOutOfTaxonomy:
    """A2 (plan Part A2): TWISTER is a spinal/neck crank, not an arm lock —
    removed from the arm_lock list entirely; routes through the same
    out-of-taxonomy/unclear soft-tier path as an unlisted submission."""

    def test_twister_ending_position_is_unclear_not_arm_lock(self):
        event = _event("SUBMISSION", "BACK", "N/A", "TWISTER")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "submission_unclear"
        assert out["track"] == "soft"

    def test_twister_movements_keyword_does_not_resolve_arm_lock(self):
        event = _event("SUBMISSION", "SIDE CONTROL", "ATTEMPT (TWISTER)", "OTHER")
        out = apply_mapping(event, None, MAPPING)
        assert out["action_class"] == "submission_unclear"

    def test_twister_still_compat_family_wide_as_submission(self):
        """Even though subtype is 'unclear', it's still feasible against any
        submission_* prediction per §5.1's family-wide SUBMISSION compat."""
        normalized = apply_mapping(_event("SUBMISSION", "BACK", "N/A", "TWISTER"), None, MAPPING)
        compat = compat_classes_for(normalized, MAPPING)
        assert compat == frozenset({"submission_choke", "submission_arm_lock", "submission_leg_lock"})


class TestPositionBucketGeneralization:
    def test_known_value_hits_frozen_table(self):
        assert bucket_position("HALF GUARD", MAPPING) == "half_guard"

    def test_unseen_value_falls_back_to_generalization_heuristic(self, caplog):
        # Synthetic value never seen in the real corpus, exercising the
        # documented (logged) fallback rather than the frozen table.
        with caplog.at_level("WARNING"):
            bucket = bucket_position("REVERSE ASHI VARIANT", MAPPING)
        assert bucket == "open_guard"
        assert any("generalizing unseen raw position" in rec.message for rec in caplog.records)

    def test_truly_unclassifiable_value_raises(self):
        with pytest.raises(MappingError):
            bucket_position("XYZZY NONSENSE VALUE", MAPPING)


class TestPositionEntryCompat:
    def test_position_entry_derived_classes_are_exact_singleton_compat(self):
        """Unlike SUBMISSION, a POSITION-ENTRY-derived guard_pass/sweep/
        back_take is an exact-class compat singleton (§5.1's many-to-many
        case is SUBMISSION only)."""
        event = _event("POSITION ENTRY", "TURTLE", "BERIMBOLO", "BACK")
        normalized = apply_mapping(event, None, MAPPING)
        assert compat_classes_for(normalized, MAPPING) == frozenset({"back_take"})
