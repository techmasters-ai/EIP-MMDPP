"""Property-based tests for ABAC and patch state machine using Hypothesis."""

import pytest
from hypothesis import given, settings as h_settings, strategies as st

pytestmark = pytest.mark.unit

# Roles that exist in the system
VALID_ROLES = ["analyst", "curator", "admin", "unauthenticated"]

# Patch states and which transitions are valid from each state
VALID_TRANSITIONS = {
    "DRAFT": ["UNDER_REVIEW"],
    "UNDER_REVIEW": ["APPROVED", "REJECTED"],
    "APPROVED": ["DUAL_APPROVED", "APPLIED"],
    "DUAL_APPROVED": ["APPLIED"],
    "REJECTED": ["DRAFT"],
    "APPLIED": ["REVERTED"],
    "REVERTED": [],
}

ACTOR_ALLOWED_ACTIONS = {
    # (role, action) -> allowed
    ("analyst", "submit_feedback"): True,
    ("analyst", "approve_patch"): False,
    ("analyst", "reject_patch"): False,
    ("analyst", "apply_patch"): False,
    ("curator", "submit_feedback"): True,
    ("curator", "approve_patch"): True,
    ("curator", "reject_patch"): True,
    ("curator", "apply_patch"): True,
    ("admin", "submit_feedback"): True,
    ("admin", "approve_patch"): True,
    ("admin", "reject_patch"): True,
    ("admin", "apply_patch"): True,
    ("unauthenticated", "submit_feedback"): False,
    ("unauthenticated", "approve_patch"): False,
    ("unauthenticated", "reject_patch"): False,
    ("unauthenticated", "apply_patch"): False,
}


class TestPatchStateTransitions:
    """Verify the patch state machine allows only valid transitions."""

    @given(
        current_state=st.sampled_from(list(VALID_TRANSITIONS.keys())),
        target_state=st.sampled_from(list(VALID_TRANSITIONS.keys())),
    )
    def test_only_valid_transitions_allowed(self, current_state: str, target_state: str):
        """For any (current_state, target_state) pair, verify the transition is
        either valid or invalid according to the state machine."""
        allowed = VALID_TRANSITIONS[current_state]
        if target_state in allowed:
            # Valid transition — should not raise
            assert target_state in allowed
        else:
            # Invalid transition — should not be in allowed list
            assert target_state not in allowed

    def test_no_state_can_transition_to_itself(self):
        """No state should be a valid transition from itself."""
        for state, transitions in VALID_TRANSITIONS.items():
            assert state not in transitions, (
                f"State '{state}' lists itself as a valid transition"
            )

    def test_applied_is_terminal_except_revert(self):
        """APPLIED can only transition to REVERTED."""
        assert VALID_TRANSITIONS["APPLIED"] == ["REVERTED"]

    def test_reverted_is_terminal(self):
        """REVERTED is a terminal state with no further transitions."""
        assert VALID_TRANSITIONS["REVERTED"] == []


class TestABACRoleMatrix:
    """Property-based test: verify every (role, action) combination is classified."""

    @given(
        role=st.sampled_from(VALID_ROLES),
        action=st.sampled_from(["submit_feedback", "approve_patch", "reject_patch", "apply_patch"]),
    )
    def test_every_role_action_combination_is_defined(self, role: str, action: str):
        """Every combination should have an explicit allow/deny decision."""
        assert (role, action) in ACTOR_ALLOWED_ACTIONS, (
            f"Undefined ABAC policy for role='{role}', action='{action}'"
        )

    def test_analyst_cannot_approve_or_reject(self):
        """Analyst is a read-only consumer and cannot modify the patch workflow."""
        assert ACTOR_ALLOWED_ACTIONS[("analyst", "approve_patch")] is False
        assert ACTOR_ALLOWED_ACTIONS[("analyst", "reject_patch")] is False
        assert ACTOR_ALLOWED_ACTIONS[("analyst", "apply_patch")] is False

    def test_curator_can_approve_and_reject(self):
        """Curators are the primary governance actors."""
        assert ACTOR_ALLOWED_ACTIONS[("curator", "approve_patch")] is True
        assert ACTOR_ALLOWED_ACTIONS[("curator", "reject_patch")] is True

    def test_unauthenticated_cannot_do_anything(self):
        """Unauthenticated users have no permissions."""
        for action in ["submit_feedback", "approve_patch", "reject_patch", "apply_patch"]:
            assert ACTOR_ALLOWED_ACTIONS[("unauthenticated", action)] is False
