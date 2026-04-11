"""Service-side bundle loader. Distinct from the worker-side module
because this side pre-imports the extraction schema modules during
startup so /extract-pass can dispatch by (bundle_key, pass_name) in
constant time. Worker side reads only metadata; service side loads the
Pydantic template classes.

Imported by docker/docling-graph/app/main.py lifespan hook (wired in
Task 2.5, not here)."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


# Module-level cache: (bundle_key, pass_name) -> template class
_template_cache: dict[tuple[str, str], type[BaseModel]] = {}
_manifest_cache: dict[str, dict] = {}

# Inside the docling-graph container, ontology_bundles/ is copied to /app/ontology_bundles.
# Outside the container (during local tests), use a repo-root relative path.
_BUNDLE_ROOT_CANDIDATES = (
    Path("/app/ontology_bundles"),
    Path(__file__).resolve().parent.parent.parent.parent / "ontology_bundles",
)


def _find_bundle_root() -> Path:
    for p in _BUNDLE_ROOT_CANDIDATES:
        if p.is_dir():
            return p
    raise RuntimeError(
        f"Cannot find ontology_bundles directory; tried: {_BUNDLE_ROOT_CANDIDATES}"
    )


def load_bundle_manifest(bundle_key: str) -> dict:
    """Return the parsed manifest.yaml as a dict. Cached on first read."""
    if bundle_key in _manifest_cache:
        return _manifest_cache[bundle_key]
    path = _find_bundle_root() / bundle_key / "manifest.yaml"
    if not path.exists():
        raise KeyError(f"No manifest.yaml for bundle_key={bundle_key!r}")
    with open(path) as f:
        manifest = yaml.safe_load(f)
    _manifest_cache[bundle_key] = manifest
    return manifest


def load_pass_template(bundle_key: str, pass_name: str) -> type[BaseModel]:
    """Return the Pydantic template class declared for (bundle_key, pass_name).
    Raises KeyError if the bundle or pass is unknown."""
    cache_key = (bundle_key, pass_name)
    if cache_key in _template_cache:
        return _template_cache[cache_key]

    manifest = load_bundle_manifest(bundle_key)
    for pass_def in manifest.get("passes", []):
        if pass_def["name"] == pass_name:
            module_name = pass_def["module"]
            template_class_name = pass_def["template_class"]
            # The module path in manifest.yaml is relative, e.g.
            # "extraction_schemas.radar_domain". We import it as
            # "ontology_bundles.<bundle_key>.extraction_schemas.radar_domain".
            full_module = f"ontology_bundles.{bundle_key}.{module_name}"
            module = importlib.import_module(full_module)
            cls = getattr(module, template_class_name)
            _template_cache[cache_key] = cls
            return cls
    raise KeyError(f"Unknown pass_name={pass_name!r} in bundle={bundle_key!r}")


def preload_all_templates() -> None:
    """Iterate every bundle directory and force-import every declared template class.
    Called once at service startup (from main.py lifespan, Task 2.5)."""
    root = _find_bundle_root()
    for bundle_dir in sorted(root.iterdir()):
        if not bundle_dir.is_dir() or bundle_dir.name.startswith("_"):
            continue
        bundle_key = bundle_dir.name
        try:
            manifest = load_bundle_manifest(bundle_key)
        except Exception:
            continue
        for pass_def in manifest.get("passes", []):
            try:
                load_pass_template(bundle_key, pass_def["name"])
            except Exception:
                # Don't let one broken bundle block the whole service startup
                pass
