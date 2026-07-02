"""repoint the System Dossier profile onto the ``dossier`` catch-all section

Follow-up to the query-profile-sections work. We added a ``dossier`` catch-all
profile section (``query_profiles._project_field_groups`` treats the reserved
section name ``"dossier"`` as "every profiled field across all sections") and
repointed the ``system_dossier`` query profile at it. That repoint was done as
a LIVE-DB row edit, so it was NOT reproducible on a clean deploy: migration
``0022_query_profiles_table.py`` still seeds the OLD ``kind="dossier"`` version
that composes the three ``system_rf_parameters`` / ``system_components`` /
``system_performance`` section profiles.

This migration makes the repoint reproducible. ``upgrade`` flips the
``system_dossier`` row to ``kind="section_properties"`` pointed at the single
``dossier`` catch-all section (``include_associated_systems=true``) and updates
the description. It is idempotent/safe: scoped to
``WHERE profile_key='system_dossier'`` and a no-op if that row is absent.

``downgrade`` restores EXACTLY what ``0022`` seeds for ``system_dossier`` — the
canonical ``kind="dossier"`` definition (see
``0022_query_profiles_table.py::_CANONICAL_PROFILES['system_dossier']`` promoted
through ``_DEFINITION_KEYS``) and its original description — so it is a true
inverse of a clean ``0022`` seed.

Revision ID: 0024
Revises: 0023
Create Date: 2026-07-02
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


# New (repointed) state: catch-all ``dossier`` section, related systems on.
_NEW_DEFINITION = '{"profile_sections": ["dossier"], "include_associated_systems": true}'
_NEW_DESCRIPTION = (
    "Full entity dossier — every profiled field across all sections, "
    "plus related systems."
)

# Original state, mirroring 0022's canonical ``system_dossier`` seed exactly.
# 0022 promotes the canonical profile through ``_DEFINITION_KEYS``
# (target_entity_types, traversals, section_profile_ids, profile_sections,
# profile_subgroup, include_associated_systems, placeholder_query); every key
# present in the canonical entry lands in the ``definition`` JSONB blob. The
# live untouched siblings confirm this full shape, so the faithful inverse
# restores all of it — not just the dossier-defining keys.
_ORIG_DEFINITION = (
    '{"target_entity_types": [], "traversals": [], '
    '"section_profile_ids": ["system_rf_parameters", "system_components", '
    '"system_performance"], "profile_sections": [], '
    '"include_associated_systems": false, "placeholder_query": "e.g. SA-2"}'
)
_ORIG_DESCRIPTION = (
    "Composite report of RF parameters, components, and performance "
    "for the resolved system."
)


def upgrade() -> None:
    op.execute(
        f"""
        UPDATE governance.query_profiles
        SET kind = 'section_properties',
            definition = '{_NEW_DEFINITION}'::jsonb,
            description = '{_NEW_DESCRIPTION}',
            updated_at = now()
        WHERE profile_key = 'system_dossier'
        """
    )


def downgrade() -> None:
    op.execute(
        f"""
        UPDATE governance.query_profiles
        SET kind = 'dossier',
            definition = '{_ORIG_DEFINITION}'::jsonb,
            description = '{_ORIG_DESCRIPTION}',
            updated_at = now()
        WHERE profile_key = 'system_dossier'
        """
    )
