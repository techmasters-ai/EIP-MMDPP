"""Unit tests for ``_next_entity_pass_to_dispatch`` — identity-first, field-gated.

This pure helper encodes the entity-pass selection rule used by BOTH the initial
fan-out in ``derive_ontology_graph`` (Site A) and the follow-up dispatcher
``_try_advance_phase`` (Site B):

    IDENTITY passes dispatch first; a FIELD-GROUP pass is only eligible once
    EVERY identity pass is terminal.

Why: field_group passes harvest committed identity-entity names (the C8 anchor
channel) at dispatch time.  If a field pass dispatches concurrently with an
identity pass — before any identity entity is committed — the anchor channel is
inert.  Gating field passes on all-identity-terminal makes the channel live.

Terminality is FAIL-OPEN: a FAILED identity pass is still terminal (it is marked
terminal by the existing bookkeeping), so field passes still proceed.

Fully hermetic: the helper is pure (no DB, no Celery, no I/O).

Run standalone:
    python3 -m pytest tests/unit/test_next_entity_pass_to_dispatch.py -v
"""
from __future__ import annotations

from app.workers.pipeline import _next_entity_pass_to_dispatch


# Standard pass rosters (manifest order: identity entries before field_group).
IDENTITY = ["radar_identity", "missile_identity"]
FIELD = ["radar_power_rf", "missile_kinematics"]


# ---------------------------------------------------------------------------
# Identity-first: while ANY identity pass is not terminal, only identity is eligible
# ---------------------------------------------------------------------------

class TestIdentityFirst:

    def test_nothing_dispatched_yet_picks_first_identity(self):
        """Cold start: first identity pass is selected, never a field pass."""
        got = _next_entity_pass_to_dispatch(IDENTITY, FIELD, set(), set())
        assert got == "radar_identity"

    def test_first_identity_in_flight_picks_second_identity(self):
        """radar_identity in-flight → next eligible is the second identity pass."""
        got = _next_entity_pass_to_dispatch(
            IDENTITY, FIELD,
            terminal_names=set(),
            in_flight_names={"radar_identity"},
        )
        assert got == "missile_identity"

    def test_field_NOT_eligible_while_an_identity_pass_in_flight(self):
        """One identity terminal, the other in-flight → no field pass eligible.

        The only remaining identity (radar_identity) is in-flight, so there is
        no eligible identity pass AND field passes are gated → None.
        This is the core fix: field must NOT slip in before all identity terminal.
        """
        got = _next_entity_pass_to_dispatch(
            IDENTITY, FIELD,
            terminal_names={"missile_identity"},
            in_flight_names={"radar_identity"},
        )
        assert got is None

    def test_field_NOT_eligible_while_all_identity_in_flight(self):
        """Both identity passes in-flight, none terminal → field gated → None."""
        got = _next_entity_pass_to_dispatch(
            IDENTITY, FIELD,
            terminal_names=set(),
            in_flight_names={"radar_identity", "missile_identity"},
        )
        assert got is None


# ---------------------------------------------------------------------------
# Field eligible only once ALL identity passes are terminal
# ---------------------------------------------------------------------------

class TestFieldGatedOnAllIdentityTerminal:

    def test_all_identity_terminal_picks_first_field(self):
        """Both identity terminal, no field started → first field pass selected."""
        got = _next_entity_pass_to_dispatch(
            IDENTITY, FIELD,
            terminal_names={"radar_identity", "missile_identity"},
            in_flight_names=set(),
        )
        assert got == "radar_power_rf"

    def test_all_identity_terminal_skips_in_flight_field(self):
        """First field in-flight → next eligible is the second field pass."""
        got = _next_entity_pass_to_dispatch(
            IDENTITY, FIELD,
            terminal_names={"radar_identity", "missile_identity"},
            in_flight_names={"radar_power_rf"},
        )
        assert got == "missile_kinematics"

    def test_everything_terminal_returns_none(self):
        """All identity + field terminal → nothing left → None."""
        got = _next_entity_pass_to_dispatch(
            IDENTITY, FIELD,
            terminal_names=set(IDENTITY + FIELD),
            in_flight_names=set(),
        )
        assert got is None

    def test_all_terminal_or_in_flight_returns_none(self):
        """Identity terminal, both fields in-flight → nothing eligible → None."""
        got = _next_entity_pass_to_dispatch(
            IDENTITY, FIELD,
            terminal_names=set(IDENTITY),
            in_flight_names=set(FIELD),
        )
        assert got is None


# ---------------------------------------------------------------------------
# FAIL-OPEN: a FAILED identity pass counts as terminal (it is in terminal_names)
# ---------------------------------------------------------------------------

class TestFailOpenFailedIdentityCountsTerminal:

    def test_failed_identity_in_terminal_set_unblocks_field(self):
        """radar_identity FAILED (still terminal) + missile_identity terminal →
        field passes proceed.  The helper does not distinguish FAILED from
        SUCCEEDED — both live in terminal_names — so this is purely a
        consequence of the caller's terminal bookkeeping (FAIL-OPEN preserved).
        """
        # Both identity names are in terminal_names; one of them was FAILED upstream.
        got = _next_entity_pass_to_dispatch(
            IDENTITY, FIELD,
            terminal_names={"radar_identity", "missile_identity"},
            in_flight_names=set(),
        )
        assert got == "radar_power_rf"


# ---------------------------------------------------------------------------
# Edge cases: zero identity, zero field
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_identity_passes_field_immediately_eligible(self):
        """No identity passes at all → all() over empty is True → field eligible
        immediately (no deadlock for identity-less bundles)."""
        got = _next_entity_pass_to_dispatch(
            identity_names=[],
            field_names=FIELD,
            terminal_names=set(),
            in_flight_names=set(),
        )
        assert got == "radar_power_rf"

    def test_zero_identity_zero_field_returns_none(self):
        """Empty rosters → None (no candidates)."""
        got = _next_entity_pass_to_dispatch([], [], set(), set())
        assert got is None

    def test_zero_field_passes_dispatches_identity_only(self):
        """Identity-only bundle (no field passes) dispatches identity normally."""
        got = _next_entity_pass_to_dispatch(
            IDENTITY, [],
            terminal_names={"radar_identity"},
            in_flight_names=set(),
        )
        assert got == "missile_identity"

    def test_zero_field_all_identity_terminal_returns_none(self):
        """Identity-only bundle, all identity terminal → None (Branch 2 takes over)."""
        got = _next_entity_pass_to_dispatch(
            IDENTITY, [],
            terminal_names=set(IDENTITY),
            in_flight_names=set(),
        )
        assert got is None

    def test_accepts_list_terminal_and_in_flight_args(self):
        """Membership-tested args may be any container (list, set, dict-keys)."""
        got = _next_entity_pass_to_dispatch(
            IDENTITY, FIELD,
            terminal_names=["radar_identity", "missile_identity"],
            in_flight_names=["radar_power_rf"],
        )
        assert got == "missile_kinematics"
