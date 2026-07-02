import React, { useEffect, useMemo, useState } from "react";
import {
  createQueryProfile,
  deleteQueryProfile,
  getOntology,
  listQueryProfiles,
  listSources,
  updateQueryProfile,
  type OntologyResponse,
  type QueryProfileDefinitionBody,
  type QueryProfileKind,
  type QueryProfileResponse,
  type QueryProfileTraversal,
  type QueryProfileStep,
  type Source,
} from "../api/client";
import { uniqueSorted } from "../utils/ontologyHelpers";

// Mirror of the backend's `_CANONICAL_ROOT_ENTITY_TYPES` frozenset
// (app/schemas/query_profiles.py) — section_properties profiles may only be
// rooted on these canonical classes. Kept here for fast inline feedback; the
// backend remains the authority.
const CANONICAL_ROOT_ENTITY_TYPES = ["RADAR_SYSTEM", "MISSILE_SYSTEM"];

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 100);
}

function readSelectedOptions(event: React.ChangeEvent<HTMLSelectElement>): string[] {
  return Array.from(event.target.selectedOptions).map((option) => option.value);
}

interface ProfileDraft {
  profileKey: string;
  label: string;
  description: string;
  kind: QueryProfileKind;
  enabled: boolean;
  sourceId: string; // "" == Global / unscoped (source_id null)
  rootEntityTypes: string[];
  targetEntityTypes: string[];
  sectionProfileIds: string[];
  profileSections: string[];
  includeAssociatedSystems: boolean;
  traversals: QueryProfileTraversal[];
  placeholderQuery: string;
}

function blankTraversals(): QueryProfileTraversal[] {
  return [{ steps: [{ direction: "out", rel_types: [], min_hops: 1, max_hops: 1 }] }];
}

function blankProfileDraft(): ProfileDraft {
  return {
    profileKey: "",
    label: "",
    description: "",
    kind: "section_properties",
    enabled: true,
    sourceId: "",
    rootEntityTypes: [],
    targetEntityTypes: [],
    sectionProfileIds: [],
    profileSections: [],
    includeAssociatedSystems: false,
    traversals: blankTraversals(),
    placeholderQuery: "",
  };
}

function draftFromProfile(profile: QueryProfileResponse): ProfileDraft {
  const def = profile.definition ?? {};
  const traversals = def.traversals ?? [];
  return {
    profileKey: profile.profile_key,
    label: profile.label,
    description: profile.description ?? "",
    kind: profile.kind,
    enabled: profile.enabled,
    sourceId: profile.source_id ?? "",
    rootEntityTypes: [...profile.root_entity_types],
    targetEntityTypes: [...(def.target_entity_types ?? [])],
    sectionProfileIds: [...(def.section_profile_ids ?? [])],
    profileSections: [...(def.profile_sections ?? [])],
    includeAssociatedSystems: def.include_associated_systems ?? false,
    traversals:
      traversals.length > 0
        ? traversals.map((traversal) => ({
            steps: traversal.steps.map((step) => ({ ...step, rel_types: [...step.rel_types] })),
          }))
        : blankTraversals(),
    placeholderQuery: def.placeholder_query ?? "",
  };
}

function cleanTraversals(traversals: QueryProfileTraversal[]): QueryProfileTraversal[] {
  return traversals
    .map((traversal) => ({
      steps: traversal.steps
        .filter((step) => step.rel_types.length > 0)
        .map((step) => ({
          ...step,
          rel_types: uniqueSorted(step.rel_types),
          min_hops: Math.max(1, Math.min(step.min_hops, step.max_hops)),
          max_hops: Math.max(step.min_hops, step.max_hops),
        })),
    }))
    .filter((traversal) => traversal.steps.length > 0);
}

function definitionFromDraft(draft: ProfileDraft): QueryProfileDefinitionBody {
  return {
    target_entity_types: draft.kind === "section" ? uniqueSorted(draft.targetEntityTypes) : [],
    traversals: draft.kind === "section" ? cleanTraversals(draft.traversals) : [],
    section_profile_ids: draft.kind === "dossier" ? uniqueSorted(draft.sectionProfileIds) : [],
    profile_sections: draft.kind === "section_properties" ? uniqueSorted(draft.profileSections) : [],
    include_associated_systems:
      draft.kind === "section_properties" ? draft.includeAssociatedSystems : false,
    placeholder_query: draft.placeholderQuery.trim() || null,
  };
}

function MultiSelectField(props: {
  label: string;
  options: string[];
  value: string[];
  onChange: (value: string[]) => void;
  disabled?: boolean;
  helperText?: string;
  id: string;
}) {
  const size = Math.min(Math.max(props.options.length, 4), 10);
  return (
    <div className="field" style={{ flex: 1, minWidth: "240px" }}>
      <label htmlFor={props.id}>{props.label}</label>
      <select
        id={props.id}
        multiple
        value={props.value}
        disabled={props.disabled}
        size={size}
        onChange={(event) => props.onChange(readSelectedOptions(event))}
        style={{ minHeight: "10rem" }}
      >
        {props.options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
      {props.helperText && (
        <div className="text-xs text-muted" style={{ marginTop: "0.35rem" }}>
          {props.helperText}
        </div>
      )}
    </div>
  );
}

function OntologyPanel({ ontology }: { ontology: OntologyResponse | null }) {
  if (!ontology) {
    return (
      <div className="alert alert-info">
        Loading the live ontology…
      </div>
    );
  }
  return (
    <section className="card card-body" style={{ marginBottom: "1rem" }}>
      <div className="flex-center gap-sm" style={{ marginBottom: "0.5rem" }}>
        <h2 style={{ margin: 0, fontSize: "1rem" }}>Ontology (read-only)</h2>
        <span className="badge badge-info" style={{ marginLeft: "auto" }}>
          v{ontology.version}
        </span>
      </div>
      <p className="text-sm text-muted" style={{ marginTop: 0 }}>
        Served live from the air_defense_v3 source of truth. Query profiles below are authored directly
        against these entity types, relationship types, and profile sections.
      </p>

      <div style={{ marginBottom: "0.75rem" }}>
        <div style={{ fontWeight: 600, marginBottom: "0.35rem" }}>
          Entity types ({ontology.entity_types.length})
        </div>
        <div className="flex-center gap-sm" style={{ flexWrap: "wrap", justifyContent: "flex-start" }}>
          {ontology.entity_types.map((e) => (
            <span key={e.name} className="badge badge-success" title={e.label}>
              {e.name}
            </span>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: "0.75rem" }}>
        <div style={{ fontWeight: 600, marginBottom: "0.35rem" }}>
          Relationship types ({ontology.relationship_types.length})
        </div>
        <div className="flex-center gap-sm" style={{ flexWrap: "wrap", justifyContent: "flex-start" }}>
          {ontology.relationship_types.map((r) => (
            <span key={r.name} className="badge badge-info">
              {r.name}
            </span>
          ))}
        </div>
      </div>

      <div>
        <div style={{ fontWeight: 600, marginBottom: "0.35rem" }}>
          Profile sections ({ontology.profile_sections.length})
        </div>
        <div className="flex-center gap-sm" style={{ flexWrap: "wrap", justifyContent: "flex-start" }}>
          {ontology.profile_sections.map((s) => (
            <span key={s} className="badge">
              {s}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

export function QueryProfilesPage() {
  const [ontology, setOntology] = useState<OntologyResponse | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [profiles, setProfiles] = useState<QueryProfileResponse[]>([]);
  const [profileDraft, setProfileDraft] = useState<ProfileDraft>(blankProfileDraft());
  const [editingProfileKey, setEditingProfileKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deletingKey, setDeletingKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const entityTypeOptions = useMemo(
    () => uniqueSorted((ontology?.entity_types ?? []).map((e) => e.name)),
    [ontology],
  );
  const relationshipTypeOptions = useMemo(
    () => uniqueSorted((ontology?.relationship_types ?? []).map((r) => r.name)),
    [ontology],
  );
  const profileSectionOptions = ontology?.profile_sections ?? [];

  const sectionProfileKeys = useMemo(
    () =>
      profiles
        .filter((p) => p.kind === "section" || p.kind === "section_properties")
        .map((p) => p.profile_key),
    [profiles],
  );

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [ontologyResult, sourcesResult, profilesResult] = await Promise.all([
        getOntology(),
        listSources().catch(() => [] as Source[]),
        listQueryProfiles(),
      ]);
      setOntology(ontologyResult);
      setSources(sourcesResult);
      setProfiles(profilesResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load query profiles");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const resetForm = () => {
    setProfileDraft(blankProfileDraft());
    setEditingProfileKey(null);
    setError(null);
    setSuccess(null);
  };

  const handleEditProfile = (profile: QueryProfileResponse) => {
    setProfileDraft(draftFromProfile(profile));
    setEditingProfileKey(profile.profile_key);
    setSuccess(null);
    setError(null);
  };

  /**
   * Fast, inline validation of the 4 known business rules before hitting the
   * API. Mirrors the backend's `validate_shape`; the backend 422 remains the
   * authoritative backstop (surfaced readably via formatApiErrorDetail).
   */
  const validateDraft = (): string | null => {
    if (!profileDraft.label.trim()) return "Profile label is required";
    const key = editingProfileKey ?? (profileDraft.profileKey.trim() || slugify(profileDraft.label));
    if (!key) return "Profile key is required";

    if (profileDraft.kind === "section") {
      const hasTraversal = cleanTraversals(profileDraft.traversals).length > 0;
      if (!hasTraversal) {
        return "Section profiles require at least one traversal with a relationship type";
      }
    } else if (profileDraft.kind === "section_properties") {
      if (uniqueSorted(profileDraft.profileSections).length === 0) {
        return "Section-properties profiles require at least one profile section";
      }
      const invalidRoots = uniqueSorted(profileDraft.rootEntityTypes).filter(
        (t) => !CANONICAL_ROOT_ENTITY_TYPES.includes(t),
      );
      if (invalidRoots.length > 0) {
        return `Section-properties root entity types must be one of ${CANONICAL_ROOT_ENTITY_TYPES.join(
          ", ",
        )}; remove: ${invalidRoots.join(", ")}`;
      }
    } else if (profileDraft.kind === "dossier") {
      if (uniqueSorted(profileDraft.sectionProfileIds).length === 0) {
        return "Dossier profiles require at least one section profile";
      }
    }
    return null;
  };

  const handleSaveProfile = async () => {
    const validationError = validateDraft();
    if (validationError) {
      setError(validationError);
      return;
    }
    const profileKey = editingProfileKey ?? (profileDraft.profileKey.trim() || slugify(profileDraft.label));

    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const definition = definitionFromDraft(profileDraft);
      const rootEntityTypes = uniqueSorted(profileDraft.rootEntityTypes);
      const sourceId = profileDraft.sourceId || null;

      if (editingProfileKey) {
        await updateQueryProfile(editingProfileKey, {
          label: profileDraft.label.trim(),
          description: profileDraft.description.trim() || null,
          kind: profileDraft.kind,
          root_entity_types: rootEntityTypes,
          definition,
          source_id: sourceId,
          enabled: profileDraft.enabled,
        });
      } else {
        await createQueryProfile({
          profile_key: profileKey,
          label: profileDraft.label.trim(),
          description: profileDraft.description.trim() || null,
          kind: profileDraft.kind,
          root_entity_types: rootEntityTypes,
          definition,
          source_id: sourceId,
          enabled: profileDraft.enabled,
        });
      }

      await loadData();
      setSuccess(
        editingProfileKey
          ? `Updated query profile "${profileDraft.label.trim()}".`
          : `Created query profile "${profileDraft.label.trim()}".`,
      );
      setProfileDraft(blankProfileDraft());
      setEditingProfileKey(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save query profile");
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteProfile = async (profile: QueryProfileResponse) => {
    const profileKey = profile.profile_key;
    const confirmed = window.confirm(
      `Delete query profile "${profile.label}" (${profileKey})? This cannot be undone.`,
    );
    if (!confirmed) return;
    setDeletingKey(profileKey);
    setError(null);
    setSuccess(null);
    try {
      await deleteQueryProfile(profileKey);
      await loadData();
      if (editingProfileKey === profileKey) {
        setProfileDraft(blankProfileDraft());
        setEditingProfileKey(null);
      }
      setSuccess(`Deleted query profile "${profileKey}".`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete query profile");
    } finally {
      setDeletingKey(null);
    }
  };

  const updateTraversalStep = (
    traversalIndex: number,
    stepIndex: number,
    updater: (step: QueryProfileStep) => QueryProfileStep,
  ) => {
    setProfileDraft((current) => ({
      ...current,
      traversals: current.traversals.map((traversal, currentTraversalIndex) => {
        if (currentTraversalIndex !== traversalIndex) return traversal;
        return {
          ...traversal,
          steps: traversal.steps.map((step, currentStepIndex) =>
            currentStepIndex === stepIndex ? updater(step) : step,
          ),
        };
      }),
    }));
  };

  const removeTraversal = (traversalIndex: number) => {
    setProfileDraft((current) => ({
      ...current,
      traversals: current.traversals.filter((_, index) => index !== traversalIndex),
    }));
  };

  const removeStep = (traversalIndex: number, stepIndex: number) => {
    setProfileDraft((current) => ({
      ...current,
      traversals: current.traversals.map((traversal, currentTraversalIndex) => {
        if (currentTraversalIndex !== traversalIndex) return traversal;
        return {
          ...traversal,
          steps: traversal.steps.filter((_, index) => index !== stepIndex),
        };
      }),
    }));
  };

  const sourceName = (sourceId?: string | null): string => {
    if (!sourceId) return "Global";
    return sources.find((source) => source.id === sourceId)?.name ?? "Scoped source";
  };

  return (
    <div>
      <OntologyPanel ontology={ontology} />

      <section className="card card-body">
        <div className="flex-center gap-sm" style={{ marginBottom: "0.75rem" }}>
          <h2 style={{ margin: 0, fontSize: "1rem" }}>
            {editingProfileKey ? `Edit Query Profile: ${editingProfileKey}` : "Create Query Profile"}
          </h2>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => void loadData()}
            disabled={loading}
            style={{ marginLeft: "auto" }}
          >
            {loading ? "Refreshing…" : "Refresh"}
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={resetForm} disabled={saving}>
            New Profile
          </button>
        </div>

        <p className="text-sm text-muted" style={{ marginTop: 0 }}>
          Query profiles are deterministic exact-graph search modes. Each enabled profile appears as a query
          mode on the Search Documents page.
        </p>

        <div className="field-row" style={{ gap: "1rem" }}>
          <div className="field" style={{ flex: 1 }}>
            <label htmlFor="profile-label">Profile label</label>
            <input
              id="profile-label"
              type="text"
              value={profileDraft.label}
              onChange={(event) =>
                setProfileDraft((current) => {
                  const nextLabel = event.target.value;
                  const nextDraft = { ...current, label: nextLabel };
                  if (!editingProfileKey && (!current.profileKey || current.profileKey === slugify(current.label))) {
                    nextDraft.profileKey = slugify(nextLabel);
                  }
                  return nextDraft;
                })
              }
              placeholder="e.g. System RF Parameters"
            />
          </div>
          <div className="field" style={{ width: "280px" }}>
            <label htmlFor="profile-key">Profile key</label>
            <input
              id="profile-key"
              type="text"
              value={profileDraft.profileKey}
              onChange={(event) =>
                setProfileDraft((current) => ({ ...current, profileKey: slugify(event.target.value) }))
              }
              placeholder="e.g. system_rf_parameters"
              disabled={Boolean(editingProfileKey)}
            />
          </div>
        </div>

        <div className="field">
          <label htmlFor="profile-description">Description</label>
          <textarea
            id="profile-description"
            rows={2}
            value={profileDraft.description}
            onChange={(event) => setProfileDraft((current) => ({ ...current, description: event.target.value }))}
            placeholder="What should this exact search mode return?"
            style={{ width: "100%" }}
          />
        </div>

        <div className="field-row" style={{ gap: "1rem" }}>
          <div className="field" style={{ width: "220px" }}>
            <label htmlFor="profile-kind">Profile kind</label>
            <select
              id="profile-kind"
              value={profileDraft.kind}
              onChange={(event) => {
                const next = event.target.value as QueryProfileKind;
                setProfileDraft((current) => ({ ...current, kind: next }));
              }}
            >
              <option value="section">Section (traversal)</option>
              <option value="section_properties">Section properties (flat schema)</option>
              <option value="dossier">Dossier</option>
            </select>
          </div>
          <div className="field" style={{ width: "220px" }}>
            <label htmlFor="profile-source">Project source</label>
            <select
              id="profile-source"
              value={profileDraft.sourceId}
              onChange={(event) => setProfileDraft((current) => ({ ...current, sourceId: event.target.value }))}
            >
              <option value="">Global (all sources)</option>
              {sources.map((source) => (
                <option key={source.id} value={source.id}>
                  {source.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field" style={{ flex: 1 }}>
            <label htmlFor="profile-placeholder">Search placeholder</label>
            <input
              id="profile-placeholder"
              type="text"
              value={profileDraft.placeholderQuery}
              onChange={(event) => setProfileDraft((current) => ({ ...current, placeholderQuery: event.target.value }))}
              placeholder="e.g. SA-2"
            />
          </div>
        </div>

        <label className="flex-center gap-sm" style={{ marginBottom: "1rem" }}>
          <input
            type="checkbox"
            checked={profileDraft.enabled}
            onChange={(event) => setProfileDraft((current) => ({ ...current, enabled: event.target.checked }))}
          />
          <span className="text-sm">Enabled (expose this profile as a Search Documents query mode)</span>
        </label>

        <div className="field-row" style={{ gap: "1rem", alignItems: "stretch", flexWrap: "wrap" }}>
          <MultiSelectField
            id="profile-root-types"
            label="Root entity types"
            options={entityTypeOptions}
            value={profileDraft.rootEntityTypes}
            onChange={(value) => setProfileDraft((current) => ({ ...current, rootEntityTypes: value }))}
            helperText={
              profileDraft.kind === "section_properties"
                ? "For section-properties profiles, restrict to RADAR_SYSTEM / MISSILE_SYSTEM."
                : "Optional. Restricts which entity types can be resolved as the search root."
            }
          />

          {profileDraft.kind === "section" && (
            <MultiSelectField
              id="profile-target-types"
              label="Target entity types"
              options={entityTypeOptions}
              value={profileDraft.targetEntityTypes}
              onChange={(value) => setProfileDraft((current) => ({ ...current, targetEntityTypes: value }))}
              helperText="Optional. Restricts which entity types can be returned by the traversal."
            />
          )}
          {profileDraft.kind === "section_properties" && (
            <MultiSelectField
              id="profile-sections"
              label="Profile sections"
              options={profileSectionOptions}
              value={profileDraft.profileSections}
              onChange={(value) => setProfileDraft((current) => ({ ...current, profileSections: value }))}
              helperText="Which canonical-class section(s) this profile projects. Options come from the live ontology."
            />
          )}
          {profileDraft.kind === "dossier" && (
            <MultiSelectField
              id="profile-section-ids"
              label="Section profiles"
              options={sectionProfileKeys}
              value={profileDraft.sectionProfileIds}
              onChange={(value) => setProfileDraft((current) => ({ ...current, sectionProfileIds: value }))}
              helperText="Dossier profiles bundle existing section / section_properties profiles into one exact search mode."
            />
          )}
        </div>

        {profileDraft.kind === "section_properties" && (
          <label className="flex-center gap-sm" style={{ marginTop: "0.75rem" }}>
            <input
              type="checkbox"
              checked={profileDraft.includeAssociatedSystems}
              onChange={(event) =>
                setProfileDraft((current) => ({
                  ...current,
                  includeAssociatedSystems: event.target.checked,
                }))
              }
            />
            <span className="text-sm">
              Include related systems via ASSOCIATED_WITH / CUES (used by System Components).
            </span>
          </label>
        )}

        {profileDraft.kind === "section" && (
          <div style={{ marginTop: "1rem" }}>
            <div className="flex-center gap-sm" style={{ marginBottom: "0.75rem" }}>
              <h4 style={{ margin: 0, fontSize: "0.95rem" }}>Traversal Paths</h4>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() =>
                  setProfileDraft((current) => ({
                    ...current,
                    traversals: [
                      ...current.traversals,
                      { steps: [{ direction: "out", rel_types: [], min_hops: 1, max_hops: 1 }] },
                    ],
                  }))
                }
                style={{ marginLeft: "auto" }}
              >
                Add Traversal
              </button>
            </div>

            <div className="results">
              {profileDraft.traversals.map((traversal, traversalIndex) => (
                <div key={`traversal-${traversalIndex}`} className="result-card">
                  <div className="flex-center gap-sm" style={{ marginBottom: "0.75rem" }}>
                    <strong>Traversal {traversalIndex + 1}</strong>
                    {profileDraft.traversals.length > 1 && (
                      <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => removeTraversal(traversalIndex)}
                        style={{ marginLeft: "auto" }}
                      >
                        Remove Traversal
                      </button>
                    )}
                  </div>

                  {traversal.steps.map((step, stepIndex) => (
                    <div
                      key={`step-${traversalIndex}-${stepIndex}`}
                      className="card"
                      style={{ padding: "1rem", marginBottom: "0.75rem" }}
                    >
                      <div className="flex-center gap-sm" style={{ marginBottom: "0.75rem" }}>
                        <strong>Step {stepIndex + 1}</strong>
                        {traversal.steps.length > 1 && (
                          <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            onClick={() => removeStep(traversalIndex, stepIndex)}
                            style={{ marginLeft: "auto" }}
                          >
                            Remove Step
                          </button>
                        )}
                      </div>

                      <div className="field-row" style={{ gap: "1rem", alignItems: "flex-end" }}>
                        <div className="field" style={{ width: "160px" }}>
                          <label>Direction</label>
                          <select
                            value={step.direction}
                            onChange={(event) =>
                              updateTraversalStep(traversalIndex, stepIndex, (current) => ({
                                ...current,
                                direction: event.target.value as "out" | "in",
                              }))
                            }
                          >
                            <option value="out">Outgoing</option>
                            <option value="in">Incoming</option>
                          </select>
                        </div>
                        <div className="field" style={{ width: "120px" }}>
                          <label>Min hops</label>
                          <input
                            type="number"
                            min={1}
                            max={4}
                            value={step.min_hops}
                            onChange={(event) =>
                              updateTraversalStep(traversalIndex, stepIndex, (current) => ({
                                ...current,
                                min_hops: Number(event.target.value) || 1,
                              }))
                            }
                          />
                        </div>
                        <div className="field" style={{ width: "120px" }}>
                          <label>Max hops</label>
                          <input
                            type="number"
                            min={1}
                            max={4}
                            value={step.max_hops}
                            onChange={(event) =>
                              updateTraversalStep(traversalIndex, stepIndex, (current) => ({
                                ...current,
                                max_hops: Number(event.target.value) || 1,
                              }))
                            }
                          />
                        </div>
                      </div>

                      <MultiSelectField
                        id={`step-rel-types-${traversalIndex}-${stepIndex}`}
                        label="Relationship types"
                        options={relationshipTypeOptions}
                        value={step.rel_types}
                        onChange={(value) =>
                          updateTraversalStep(traversalIndex, stepIndex, (current) => ({
                            ...current,
                            rel_types: value,
                          }))
                        }
                        helperText="Select the relationship types this step is allowed to traverse."
                      />
                    </div>
                  ))}

                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() =>
                      setProfileDraft((current) => ({
                        ...current,
                        traversals: current.traversals.map((currentTraversal, index) => {
                          if (index !== traversalIndex) return currentTraversal;
                          return {
                            ...currentTraversal,
                            steps: [
                              ...currentTraversal.steps,
                              { direction: "out", rel_types: [], min_hops: 1, max_hops: 1 },
                            ],
                          };
                        }),
                      }))
                    }
                  >
                    Add Step
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="flex-center gap-sm" style={{ marginTop: "1rem" }}>
          <button
            type="button"
            className="btn btn-primary"
            disabled={saving}
            onClick={() => void handleSaveProfile()}
          >
            {saving ? "Saving…" : editingProfileKey ? "Save Query Profile" : "Create Query Profile"}
          </button>
          <button type="button" className="btn btn-ghost" onClick={resetForm}>
            Clear Form
          </button>
        </div>

        {error && <div className="alert alert-error mt-md">{error}</div>}
        {success && <div className="alert alert-success mt-md">{success}</div>}

        <div style={{ marginTop: "1.5rem" }}>
          <h4 style={{ marginBottom: "0.75rem" }}>Saved Profiles ({profiles.length})</h4>
          {profiles.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-title">No query profiles yet</div>
              <div className="text-muted text-sm">
                Create section, section-properties, and dossier profiles. Enabled profiles appear in Search
                Documents automatically.
              </div>
            </div>
          ) : (
            <div className="results">
              {profiles.map((profile) => (
                <div key={profile.profile_key} className="result-card">
                  <div className="result-card-header">
                    <strong>{profile.label}</strong>
                    <div className="flex-center gap-sm">
                      <span className={`badge ${profile.kind === "dossier" ? "badge-info" : "badge-success"}`}>
                        {profile.kind}
                      </span>
                      {profile.enabled ? (
                        <span className="badge badge-success">Enabled</span>
                      ) : (
                        <span className="badge">Disabled</span>
                      )}
                    </div>
                  </div>
                  <div className="text-xs text-muted" style={{ marginBottom: "0.5rem" }}>
                    {profile.profile_key} · {sourceName(profile.source_id)}
                  </div>
                  {profile.description && (
                    <p className="text-sm" style={{ margin: "0.5rem 0" }}>{profile.description}</p>
                  )}
                  <div className="text-xs text-muted" style={{ marginBottom: "0.75rem" }}>
                    {profile.kind === "section"
                      ? `${(profile.definition.traversals ?? []).length} traversal path(s)`
                      : profile.kind === "section_properties"
                        ? `Sections: ${(profile.definition.profile_sections ?? []).join(", ") || "(none)"}`
                        : `${(profile.definition.section_profile_ids ?? []).length} section profile reference(s)`}
                  </div>
                  <div className="flex-center gap-sm">
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={() => handleEditProfile(profile)}
                      disabled={saving || deletingKey !== null}
                    >
                      Edit
                    </button>
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={() => void handleDeleteProfile(profile)}
                      disabled={saving || deletingKey === profile.profile_key}
                    >
                      {deletingKey === profile.profile_key ? "Deleting…" : "Delete"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
