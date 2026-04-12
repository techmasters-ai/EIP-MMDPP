"""Spec §8.4: worker-side bundle loader tests."""
import pytest


def test_load_bundle_manifest_returns_five_passes():
    """Happy path: parses manifest.yaml into a BundleManifest with 5 passes."""
    from app.services.ontology_bundles import load_bundle_manifest, BundleManifest
    m = load_bundle_manifest("air_defense_v3")
    assert isinstance(m, BundleManifest)
    assert m.bundle_key == "air_defense_v3"
    assert len(m.passes) == 5
    assert {p.name for p in m.passes} == {
        "reference", "radar_domain", "missile_domain", "other_systems", "system_links",
    }


def test_load_bundle_manifest_populates_pass_metadata():
    """Pass metadata (kind, input_mode, primary/bridge/etc.) is parsed."""
    from app.services.ontology_bundles import load_bundle_manifest
    m = load_bundle_manifest("air_defense_v3")
    radar = m.find_pass("radar_domain")
    assert radar.required is True
    assert radar.kind == "entities_and_relationships"
    assert radar.input_mode == "document_only"
    assert "PLATFORM" in radar.bridge_entity_types
    assert "RADAR_SYSTEM" in radar.primary_entity_types


def test_resolve_bundle_key_precedence_run_wins():
    """Run tier wins over source and system_default."""
    from app.services.ontology_bundles import resolve_bundle_key
    result = resolve_bundle_key(
        run_key="from_run",
        source_key="from_source",
        system_default="from_system",
    )
    assert result == "from_run"


def test_resolve_bundle_key_precedence_source_when_run_none():
    from app.services.ontology_bundles import resolve_bundle_key
    result = resolve_bundle_key(
        run_key=None,
        source_key="from_source",
        system_default="from_system",
    )
    assert result == "from_source"


def test_resolve_bundle_key_precedence_system_when_both_none():
    from app.services.ontology_bundles import resolve_bundle_key
    result = resolve_bundle_key(
        run_key=None,
        source_key=None,
        system_default="from_system",
    )
    assert result == "from_system"


def test_resolve_bundle_key_raises_when_all_tiers_none():
    """BundleResolutionError when every tier is None — caller bug."""
    from app.services.ontology_bundles import resolve_bundle_key, BundleResolutionError
    with pytest.raises(BundleResolutionError):
        resolve_bundle_key(run_key=None, source_key=None, system_default=None)  # type: ignore[arg-type]


def test_resolve_bundle_key_for_graph_only_inherited_beats_source():
    """Graph-only precedence: run > inherited > source > system_default."""
    from app.services.ontology_bundles import resolve_bundle_key_for_graph_only
    result = resolve_bundle_key_for_graph_only(
        run_key=None,
        inherited_from_run="inherited",
        source_key="from_source",
        system_default="from_system",
    )
    assert result == "inherited"


def test_load_ontology_bundle_key_none_round_trip():
    """load_ontology(bundle_key=None) returns the system default bundle's ontology
    dict and is equivalent to load_ontology(bundle_key='air_defense_v3')."""
    from app.services.ontology_templates import load_ontology
    default = load_ontology()
    named = load_ontology(bundle_key="air_defense_v3")
    assert default == named


def test_load_ontology_prefer_active_regression():
    """load_ontology(prefer_active=True) raises TypeError (prefer_active is gone)."""
    from app.services.ontology_templates import load_ontology
    with pytest.raises(TypeError):
        load_ontology(prefer_active=True)  # type: ignore[call-arg]


def test_describe_bundle_for_display_none_returns_legacy():
    from app.services.ontology_bundles import describe_bundle_for_display, LEGACY_BUNDLE_LABEL
    assert describe_bundle_for_display(None) == LEGACY_BUNDLE_LABEL


def test_describe_bundle_for_display_real_key_passes_through():
    from app.services.ontology_bundles import describe_bundle_for_display
    assert describe_bundle_for_display("air_defense_v3") == "air_defense_v3"


def test_list_available_bundles_includes_air_defense_v3():
    from app.services.ontology_bundles import list_available_bundles
    assert "air_defense_v3" in list_available_bundles()


def test_list_available_bundles_skips_underscore_dirs():
    """Underscore-prefixed dirs (e.g., _shared) are not bundles."""
    from app.services.ontology_bundles import list_available_bundles
    for name in list_available_bundles():
        assert not name.startswith("_")


def test_load_bundle_manifest_unknown_raises():
    """Unknown bundle_key raises UnknownBundleError (or KeyError/FileNotFoundError)."""
    from app.services.ontology_bundles import load_bundle_manifest
    with pytest.raises(Exception):
        load_bundle_manifest("does_not_exist")


# --- Task 3.5: StatusSignals dataclass (spec §7.10) ----------------------

def test_status_signals_shape():
    """StatusSignals is a dataclass with snapshot, is_stale, graph_queryable."""
    from dataclasses import fields

    from app.services.ontology_bundles import StatusSignals

    field_names = {f.name for f in fields(StatusSignals)}
    assert field_names == {"snapshot", "is_stale", "graph_queryable"}


def test_status_signals_can_be_constructed_with_none_snapshot():
    """Happy path: graph_queryable remains meaningful when snapshot is None."""
    from app.services.ontology_bundles import StatusSignals

    s = StatusSignals(snapshot=None, is_stale=False, graph_queryable=False)
    assert s.snapshot is None
    assert s.is_stale is False
    assert s.graph_queryable is False
