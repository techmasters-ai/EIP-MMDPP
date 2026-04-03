import React, { useEffect, useMemo, useState } from "react";
import {
  activateQueryProfileRegistry,
  createQueryProfileRegistry,
  createRegistryQueryProfile,
  deleteRegistryQueryProfile,
  getDefaultQueryProfileTemplate,
  listQueryProfileRegistries,
  listSources,
  updateQueryProfileRegistry,
  updateRegistryQueryProfile,
  type QueryProfileDefinition,
  type QueryProfileRegistry,
  type QueryProfileRegistryTemplate,
  type QueryProfileStep,
  type Source,
} from "../api/client";
import { uniqueSorted, normalizeNamedList } from "../utils/ontologyHelpers";

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

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

interface RegistryEditorState {
  name: string;
  description: string;
  sourceId: string;
  ontologyName: string;
  ontologyVersion: string;
  ontologyJson: string;
  isActive: boolean;
}

interface ProfileDraftTraversal {
  steps: QueryProfileStep[];
}

interface ProfileDraft {
  id: string;
  label: string;
  description: string;
  kind: "section" | "dossier";
  exposed: boolean;
  rootEntityTypes: string[];
  targetEntityTypes: string[];
  sectionProfileIds: string[];
  traversals: ProfileDraftTraversal[];
  placeholderQuery: string;
}

function blankProfileDraft(): ProfileDraft {
  return {
    id: "",
    label: "",
    description: "",
    kind: "section",
    exposed: true,
    rootEntityTypes: [],
    targetEntityTypes: [],
    sectionProfileIds: [],
    traversals: [
      {
        steps: [
          {
            direction: "out",
            rel_types: [],
            min_hops: 1,
            max_hops: 1,
          },
        ],
      },
    ],
    placeholderQuery: "",
  };
}

function editorStateFromTemplate(template: QueryProfileRegistryTemplate): RegistryEditorState {
  return {
    name: template.name,
    description: template.description ?? "",
    sourceId: template.source_id ?? "",
    ontologyName: template.ontology_name ?? "",
    ontologyVersion: template.ontology_version ?? "",
    ontologyJson: prettyJson(template.ontology_definition ?? {}),
    isActive: template.is_active ?? false,
  };
}

function editorStateFromRegistry(registry: QueryProfileRegistry): RegistryEditorState {
  return {
    name: registry.name,
    description: registry.description ?? "",
    sourceId: registry.source_id ?? "",
    ontologyName: registry.ontology_name ?? "",
    ontologyVersion: registry.ontology_version ?? "",
    ontologyJson: prettyJson(registry.ontology_definition ?? {}),
    isActive: registry.is_active,
  };
}

function draftFromProfile(profile: QueryProfileDefinition): ProfileDraft {
  return {
    id: profile.id,
    label: profile.label,
    description: profile.description ?? "",
    kind: profile.kind,
    exposed: profile.exposed,
    rootEntityTypes: [...profile.root_entity_types],
    targetEntityTypes: [...profile.target_entity_types],
    sectionProfileIds: [...profile.section_profile_ids],
    traversals:
      profile.traversals.length > 0
        ? profile.traversals.map((traversal) => ({
            steps: traversal.steps.map((step) => ({ ...step, rel_types: [...step.rel_types] })),
          }))
        : blankProfileDraft().traversals,
    placeholderQuery: profile.placeholder_query ?? "",
  };
}

function profileFromDraft(draft: ProfileDraft): QueryProfileDefinition {
  const id = draft.id.trim() || slugify(draft.label);
  return {
    id,
    label: draft.label.trim(),
    description: draft.description.trim() || null,
    kind: draft.kind,
    exposed: draft.exposed,
    root_entity_types: uniqueSorted(draft.rootEntityTypes),
    target_entity_types: draft.kind === "section" ? uniqueSorted(draft.targetEntityTypes) : [],
    traversals:
      draft.kind === "section"
        ? draft.traversals
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
            .filter((traversal) => traversal.steps.length > 0)
        : [],
    section_profile_ids: draft.kind === "dossier" ? uniqueSorted(draft.sectionProfileIds) : [],
    placeholder_query: draft.placeholderQuery.trim() || null,
  };
}

const EMPTY_TEMPLATE: QueryProfileRegistryTemplate = {
  name: "",
  description: "",
  source_id: "",
  ontology_name: "",
  ontology_version: "",
  ontology_definition: {},
  profiles: [],
  is_active: false,
};

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

export function QueryProfileRegistryPage() {
  const [registries, setRegistries] = useState<QueryProfileRegistry[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [defaultTemplate, setDefaultTemplate] = useState<QueryProfileRegistryTemplate | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pageTab, setPageTab] = useState<"editor" | "registries">("editor");
  const [editorTab, setEditorTab] = useState<"ontology" | "profiles">("ontology");
  const [editor, setEditor] = useState<RegistryEditorState>(editorStateFromTemplate(EMPTY_TEMPLATE));
  const [profileDraft, setProfileDraft] = useState<ProfileDraft>(blankProfileDraft());
  const [editingProfileId, setEditingProfileId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [profileSaving, setProfileSaving] = useState(false);
  const [activatingId, setActivatingId] = useState<string | null>(null);
  const [deletingProfileId, setDeletingProfileId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const selectedRegistry = useMemo(
    () => registries.find((registry) => registry.id === selectedId) ?? null,
    [registries, selectedId],
  );

  const activeOntologyReady = Boolean(
    selectedRegistry?.is_active &&
      selectedRegistry.ontology_definition &&
      Object.keys(selectedRegistry.ontology_definition).length > 0,
  );

  const ontologyEntityTypes = useMemo(
    () => normalizeNamedList(selectedRegistry?.ontology_definition?.entity_types),
    [selectedRegistry],
  );

  const ontologyRelationshipTypes = useMemo(
    () => normalizeNamedList(selectedRegistry?.ontology_definition?.relationship_types),
    [selectedRegistry],
  );

  const sectionProfiles = useMemo(
    () => (selectedRegistry?.profiles ?? []).filter((profile) => profile.kind === "section"),
    [selectedRegistry],
  );

  const starterProfiles = useMemo(() => {
    const existingIds = new Set((selectedRegistry?.profiles ?? []).map((profile) => profile.id));
    return (defaultTemplate?.profiles ?? []).filter((profile) => !existingIds.has(profile.id));
  }, [defaultTemplate, selectedRegistry]);

  const loadRegistryIntoEditor = (registry: QueryProfileRegistry) => {
    setSelectedId(registry.id);
    setPageTab("editor");
    setEditorTab("ontology");
    setEditor(editorStateFromRegistry(registry));
    setProfileDraft(blankProfileDraft());
    setEditingProfileId(null);
    setSuccess(null);
    setError(null);
  };

  const resetEditor = () => {
    const template = defaultTemplate ?? EMPTY_TEMPLATE;
    setSelectedId(null);
    setPageTab("editor");
    setEditorTab("ontology");
    setEditor(editorStateFromTemplate(template));
    setProfileDraft(blankProfileDraft());
    setEditingProfileId(null);
    setSuccess(null);
    setError(null);
  };

  const loadData = async (preferredRegistryId?: string | null) => {
    setLoading(true);
    setError(null);
    try {
      const [registriesResult, sourcesResult, templateResult] = await Promise.all([
        listQueryProfileRegistries(),
        listSources().catch(() => []),
        getDefaultQueryProfileTemplate(),
      ]);
      setRegistries(registriesResult);
      setSources(sourcesResult);
      setDefaultTemplate(templateResult);

      const nextSelectedId = preferredRegistryId ?? selectedId;
      const nextSelected =
        (nextSelectedId && registriesResult.find((registry) => registry.id === nextSelectedId)) ||
        registriesResult[0] ||
        null;

      if (nextSelected) {
        loadRegistryIntoEditor(nextSelected);
      } else {
        setSelectedId(null);
        setEditor(editorStateFromTemplate(templateResult));
        setProfileDraft(blankProfileDraft());
        setEditingProfileId(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load query profile registries");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const parseOntologyDefinition = (): Record<string, unknown> | null => {
    if (!editor.ontologyJson.trim()) {
      return null;
    }
    return JSON.parse(editor.ontologyJson) as Record<string, unknown>;
  };

  const handleSaveRegistry = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!editor.name.trim()) {
      setError("Registry name is required");
      return;
    }

    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const ontologyDefinition = parseOntologyDefinition();
      const payload = {
        name: editor.name.trim(),
        description: editor.description.trim() || undefined,
        source_id: editor.sourceId || null,
        ontology_name: editor.ontologyName.trim() || undefined,
        ontology_version: editor.ontologyVersion.trim() || undefined,
        ontology_definition: ontologyDefinition,
        is_active: editor.isActive,
      };

      const saved = selectedId
        ? await updateQueryProfileRegistry(selectedId, payload)
        : await createQueryProfileRegistry({ ...payload, profiles: [] });

      await loadData(saved.id);
      setSuccess(
        selectedId
          ? `Updated registry "${saved.name}".`
          : `Created registry "${saved.name}".`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save registry");
    } finally {
      setSaving(false);
    }
  };

  const handleActivate = async (registryId: string) => {
    setActivatingId(registryId);
    setError(null);
    setSuccess(null);
    try {
      const activated = await activateQueryProfileRegistry(registryId);
      await loadData(activated.id);
      setSuccess(`Activated registry "${activated.name}".`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to activate registry");
    } finally {
      setActivatingId(null);
    }
  };

  const handleEditProfile = (profile: QueryProfileDefinition) => {
    setProfileDraft(draftFromProfile(profile));
    setEditingProfileId(profile.id);
    setEditorTab("profiles");
    setSuccess(null);
    setError(null);
  };

  const handleLoadStarterProfile = (profile: QueryProfileDefinition) => {
    setProfileDraft(draftFromProfile(profile));
    setEditingProfileId(null);
    setEditorTab("profiles");
    setSuccess(null);
    setError(null);
  };

  const handleSaveProfile = async () => {
    if (!selectedRegistry) {
      setError("Save and activate an ontology registry before creating query profiles");
      return;
    }
    if (!activeOntologyReady) {
      setError("Query profiles can only be created on an active registry with a saved ontology definition");
      return;
    }
    if (!profileDraft.label.trim()) {
      setError("Profile label is required");
      return;
    }

    const profile = profileFromDraft(profileDraft);
    if (!profile.id) {
      setError("Profile id is required");
      return;
    }

    setProfileSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const savedRegistry = editingProfileId
        ? await updateRegistryQueryProfile(selectedRegistry.id, editingProfileId, profile)
        : await createRegistryQueryProfile(selectedRegistry.id, profile);

      await loadData(savedRegistry.id);
      setEditorTab("profiles");
      setProfileDraft(blankProfileDraft());
      setEditingProfileId(null);
      setSuccess(
        editingProfileId
          ? `Updated query profile "${profile.label}".`
          : `Added query profile "${profile.label}".`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save query profile");
    } finally {
      setProfileSaving(false);
    }
  };

  const handleDeleteProfile = async (profileId: string) => {
    if (!selectedRegistry) return;
    setDeletingProfileId(profileId);
    setError(null);
    setSuccess(null);
    try {
      const savedRegistry = await deleteRegistryQueryProfile(selectedRegistry.id, profileId);
      await loadData(savedRegistry.id);
      if (editingProfileId === profileId) {
        setProfileDraft(blankProfileDraft());
        setEditingProfileId(null);
      }
      setEditorTab("profiles");
      setSuccess(`Deleted query profile "${profileId}".`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete query profile");
    } finally {
      setDeletingProfileId(null);
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

  return (
    <div>
      <div className="tabs" style={{ marginBottom: "1rem" }}>
        {([
          ["editor", "Ontology & Query Profiles"],
          ["registries", "Registries"],
        ] as const).map(([tabId, label]) => (
          <button
            key={tabId}
            type="button"
            className={`tab-btn${pageTab === tabId ? " active" : ""}`}
            onClick={() => setPageTab(tabId)}
          >
            {label}
          </button>
        ))}
      </div>

      {pageTab === "editor" && (
        <section className="card card-body">
          <div className="flex-center gap-sm" style={{ marginBottom: "0.75rem" }}>
            <h2 style={{ margin: 0, fontSize: "1rem" }}>
              {selectedId ? "Edit Query Profile Registry" : "Create Query Profile Registry"}
            </h2>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={resetEditor}
              style={{ marginLeft: "auto" }}
              disabled={!defaultTemplate}
            >
              Load Current Ontology Template
            </button>
          </div>

          <p className="text-sm text-muted" style={{ marginTop: 0 }}>
            The ontology definition becomes the shared ontology source for ingest, GraphRAG extraction context,
            and ontology-driven search helpers once its registry is active. Query profiles are deterministic search
            endpoints layered on top of that ontology.
          </p>

          <form onSubmit={(event) => void handleSaveRegistry(event)}>
            <div className="field-row" style={{ gap: "1rem" }}>
              <div className="field" style={{ flex: 1 }}>
                <label htmlFor="registry-name">Registry name</label>
                <input
                  id="registry-name"
                  type="text"
                  value={editor.name}
                  onChange={(event) => setEditor((current) => ({ ...current, name: event.target.value }))}
                  placeholder="e.g. Military Systems Ontology"
                />
              </div>
              <div className="field" style={{ width: "180px" }}>
                <label htmlFor="registry-source">Project source</label>
                <select
                  id="registry-source"
                  value={editor.sourceId}
                  onChange={(event) => setEditor((current) => ({ ...current, sourceId: event.target.value }))}
                >
                  <option value="">Global / Unscoped</option>
                  {sources.map((source) => (
                    <option key={source.id} value={source.id}>
                      {source.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="field">
              <label htmlFor="registry-description">Description</label>
              <textarea
                id="registry-description"
                rows={2}
                value={editor.description}
                onChange={(event) => setEditor((current) => ({ ...current, description: event.target.value }))}
                placeholder="What project or corpus is this registry for?"
                style={{ width: "100%" }}
              />
            </div>

            <div className="field-row" style={{ gap: "1rem" }}>
              <div className="field" style={{ flex: 1 }}>
                <label htmlFor="ontology-name">Ontology name</label>
                <input
                  id="ontology-name"
                  type="text"
                  value={editor.ontologyName}
                  onChange={(event) => setEditor((current) => ({ ...current, ontologyName: event.target.value }))}
                  placeholder="e.g. Military Systems + RF"
                />
              </div>
              <div className="field" style={{ width: "180px" }}>
                <label htmlFor="ontology-version">Ontology version</label>
                <input
                  id="ontology-version"
                  type="text"
                  value={editor.ontologyVersion}
                  onChange={(event) => setEditor((current) => ({ ...current, ontologyVersion: event.target.value }))}
                  placeholder="e.g. 2026.04"
                />
              </div>
            </div>

            <label className="flex-center gap-sm" style={{ marginBottom: "1rem" }}>
              <input
                type="checkbox"
                checked={editor.isActive}
                onChange={(event) => setEditor((current) => ({ ...current, isActive: event.target.checked }))}
              />
              <span className="text-sm">Make this the active registry after saving</span>
            </label>

            <div className="tabs" style={{ marginBottom: "1rem" }}>
              <button
                type="button"
                className={`tab-btn${editorTab === "ontology" ? " active" : ""}`}
                onClick={() => setEditorTab("ontology")}
              >
                Ontology Definition
              </button>
              <button
                type="button"
                className={`tab-btn${editorTab === "profiles" ? " active" : ""}`}
                onClick={() => {
                  if (activeOntologyReady) {
                    setEditorTab("profiles");
                  }
                }}
                disabled={!activeOntologyReady}
                title={
                  activeOntologyReady
                    ? undefined
                    : "Save and activate an ontology registry before building query profiles"
                }
              >
                Query Profiles
              </button>
            </div>

            {editorTab === "ontology" && (
              <div className="field">
                <label htmlFor="ontology-json">Ontology definition (JSON)</label>
                <textarea
                  id="ontology-json"
                  rows={32}
                  value={editor.ontologyJson}
                  onChange={(event) => setEditor((current) => ({ ...current, ontologyJson: event.target.value }))}
                  style={{ width: "100%", minHeight: "42rem", fontFamily: "monospace" }}
                />
                <div className="text-sm text-muted" style={{ marginTop: "0.5rem", marginBottom: "1rem" }}>
                  Save and activate this registry before you create query profiles. The active ontology is what the
                  query-profile builder, ingest path, GraphRAG extraction context, and ontology-weighted retrieval
                  helpers will use.
                </div>
              </div>
            )}

            {editorTab === "profiles" && (
              <div style={{ marginBottom: "1rem" }}>
                {!activeOntologyReady || !selectedRegistry ? (
                  <div className="alert alert-info">
                    Save and activate an ontology registry first. Query profiles are only editable for the active
                    ontology definition.
                  </div>
                ) : (
                  <>
                    <div className="flex-center gap-sm" style={{ marginBottom: "0.75rem" }}>
                      <h3 style={{ margin: 0, fontSize: "1rem" }}>
                        {editingProfileId ? `Edit Query Profile: ${editingProfileId}` : "Add Query Profile"}
                      </h3>
                      <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => {
                          setProfileDraft(blankProfileDraft());
                          setEditingProfileId(null);
                        }}
                        style={{ marginLeft: "auto" }}
                      >
                        New Profile
                      </button>
                    </div>

                    {starterProfiles.length > 0 && (
                      <div className="card" style={{ padding: "1rem", marginBottom: "1rem" }}>
                        <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Starter Profiles</div>
                        <div className="text-sm text-muted" style={{ marginBottom: "0.75rem" }}>
                          These are seeded from the repository’s current ontology and can be loaded into the form as
                          starting points.
                        </div>
                        <div className="results">
                          {starterProfiles.map((profile) => (
                            <div key={profile.id} className="result-card">
                              <div className="result-card-header">
                                <strong>{profile.label}</strong>
                                <span className={`badge ${profile.kind === "dossier" ? "badge-info" : "badge-success"}`}>
                                  {profile.kind}
                                </span>
                              </div>
                              {profile.description && (
                                <p className="text-sm" style={{ margin: "0.5rem 0" }}>{profile.description}</p>
                              )}
                              <button
                                type="button"
                                className="btn btn-ghost btn-sm"
                                onClick={() => handleLoadStarterProfile(profile)}
                              >
                                Load Into Form
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="text-sm text-muted" style={{ marginBottom: "1rem" }}>
                      Query profiles are deterministic search endpoints. Each saved profile appears as a query mode on
                      the Search Documents page.
                    </div>

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
                              if (!editingProfileId && (!current.id || current.id === slugify(current.label))) {
                                nextDraft.id = slugify(nextLabel);
                              }
                              return nextDraft;
                            })
                          }
                          placeholder="e.g. System RF Parameters"
                        />
                      </div>
                      <div className="field" style={{ width: "280px" }}>
                        <label htmlFor="profile-id">Profile id</label>
                        <input
                          id="profile-id"
                          type="text"
                          value={profileDraft.id}
                          onChange={(event) => setProfileDraft((current) => ({ ...current, id: slugify(event.target.value) }))}
                          placeholder="e.g. system_rf_parameters"
                          disabled={Boolean(editingProfileId)}
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
                          onChange={(event) =>
                            setProfileDraft((current) => ({
                              ...current,
                              kind: event.target.value as "section" | "dossier",
                              targetEntityTypes:
                                event.target.value === "section" ? current.targetEntityTypes : [],
                              sectionProfileIds:
                                event.target.value === "dossier" ? current.sectionProfileIds : [],
                            }))
                          }
                        >
                          <option value="section">Section</option>
                          <option value="dossier">Dossier</option>
                        </select>
                      </div>
                      <div className="field" style={{ flex: 1 }}>
                        <label htmlFor="profile-placeholder">Search placeholder</label>
                        <input
                          id="profile-placeholder"
                          type="text"
                          value={profileDraft.placeholderQuery}
                          onChange={(event) => setProfileDraft((current) => ({ ...current, placeholderQuery: event.target.value }))}
                          placeholder="e.g. AN/MPQ-65"
                        />
                      </div>
                    </div>

                    <label className="flex-center gap-sm" style={{ marginBottom: "1rem" }}>
                      <input
                        type="checkbox"
                        checked={profileDraft.exposed}
                        onChange={(event) => setProfileDraft((current) => ({ ...current, exposed: event.target.checked }))}
                      />
                      <span className="text-sm">Expose this profile as a Search Documents query mode</span>
                    </label>

                    <div className="field-row" style={{ gap: "1rem", alignItems: "stretch", flexWrap: "wrap" }}>
                      <MultiSelectField
                        id="profile-root-types"
                        label="Root entity types"
                        options={ontologyEntityTypes}
                        value={profileDraft.rootEntityTypes}
                        onChange={(value) => setProfileDraft((current) => ({ ...current, rootEntityTypes: value }))}
                        helperText="Optional. Restricts which entity types can be resolved as the search root."
                      />

                      {profileDraft.kind === "section" ? (
                        <MultiSelectField
                          id="profile-target-types"
                          label="Target entity types"
                          options={ontologyEntityTypes}
                          value={profileDraft.targetEntityTypes}
                          onChange={(value) => setProfileDraft((current) => ({ ...current, targetEntityTypes: value }))}
                          helperText="Optional. Restricts which entity types can be returned by the traversal."
                        />
                      ) : (
                        <MultiSelectField
                          id="profile-section-ids"
                          label="Section profiles"
                          options={sectionProfiles.map((profile) => profile.id)}
                          value={profileDraft.sectionProfileIds}
                          onChange={(value) => setProfileDraft((current) => ({ ...current, sectionProfileIds: value }))}
                          helperText="Dossier profiles bundle existing section profiles into one exact search mode."
                        />
                      )}
                    </div>

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
                                  {
                                    steps: [
                                      {
                                        direction: "out",
                                        rel_types: [],
                                        min_hops: 1,
                                        max_hops: 1,
                                      },
                                    ],
                                  },
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
                                    options={ontologyRelationshipTypes}
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
                                          {
                                            direction: "out",
                                            rel_types: [],
                                            min_hops: 1,
                                            max_hops: 1,
                                          },
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
                        disabled={profileSaving}
                        onClick={() => void handleSaveProfile()}
                      >
                        {profileSaving
                          ? "Saving..."
                          : editingProfileId
                            ? "Save Query Profile"
                            : "Add Query Profile"}
                      </button>
                      <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={() => {
                          setProfileDraft(blankProfileDraft());
                          setEditingProfileId(null);
                        }}
                      >
                        Clear Profile Form
                      </button>
                    </div>

                    <div style={{ marginTop: "1.5rem" }}>
                      <h4 style={{ marginBottom: "0.75rem" }}>Saved Profiles</h4>
                      {(selectedRegistry.profiles ?? []).length === 0 ? (
                        <div className="empty-state">
                          <div className="empty-state-title">No query profiles yet</div>
                          <div className="text-muted text-sm">
                            Add section and dossier profiles one at a time. Exposed profiles will appear in Search
                            Documents automatically.
                          </div>
                        </div>
                      ) : (
                        <div className="results">
                          {selectedRegistry.profiles.map((profile) => (
                            <div key={profile.id} className="result-card">
                              <div className="result-card-header">
                                <strong>{profile.label}</strong>
                                <div className="flex-center gap-sm">
                                  <span className={`badge ${profile.kind === "dossier" ? "badge-info" : "badge-success"}`}>
                                    {profile.kind}
                                  </span>
                                  {profile.exposed && <span className="badge badge-success">Exposed</span>}
                                </div>
                              </div>
                              <div className="text-xs text-muted" style={{ marginBottom: "0.5rem" }}>
                                {profile.id}
                              </div>
                              {profile.description && (
                                <p className="text-sm" style={{ margin: "0.5rem 0" }}>{profile.description}</p>
                              )}
                              <div className="text-xs text-muted" style={{ marginBottom: "0.75rem" }}>
                                {profile.kind === "section"
                                  ? `${profile.traversals.length} traversal path(s)`
                                  : `${profile.section_profile_ids.length} section profile reference(s)`}
                              </div>
                              <div className="flex-center gap-sm">
                                <button
                                  type="button"
                                  className="btn btn-ghost btn-sm"
                                  onClick={() => handleEditProfile(profile)}
                                >
                                  Edit
                                </button>
                                <button
                                  type="button"
                                  className="btn btn-ghost btn-sm"
                                  onClick={() => void handleDeleteProfile(profile.id)}
                                  disabled={deletingProfileId === profile.id}
                                >
                                  {deletingProfileId === profile.id ? "Deleting..." : "Delete"}
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}

            <div className="flex-center gap-sm">
              <button type="submit" className="btn btn-primary" disabled={saving}>
                {saving ? "Saving..." : selectedId ? "Save Registry" : "Create Registry"}
              </button>
              <button type="button" className="btn btn-ghost" onClick={resetEditor}>
                Clear Form
              </button>
            </div>
          </form>

          {error && <div className="alert alert-error mt-md">{error}</div>}
          {success && <div className="alert alert-success mt-md">{success}</div>}
        </section>
      )}

      {pageTab === "registries" && (
        <section className="card card-body">
          <div className="flex-center gap-sm" style={{ marginBottom: "0.75rem" }}>
            <h2 style={{ margin: 0, fontSize: "1rem" }}>Registries</h2>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => void loadData(selectedId)}
              disabled={loading}
              style={{ marginLeft: "auto" }}
            >
              {loading ? "Refreshing..." : "Refresh"}
            </button>
          </div>

          <div className="flex-center gap-sm" style={{ marginBottom: "1rem" }}>
            <button type="button" className="btn btn-primary" onClick={resetEditor}>
              New Registry
            </button>
            <span className="text-sm text-muted">
              Save and activate an ontology registry before you build query profiles for it.
            </span>
          </div>

          <div className="results">
            {registries.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-title">No registries yet</div>
                <div className="text-muted text-sm">
                  Create one to define the ontology and exact graph query modes for a project.
                </div>
              </div>
            ) : (
              registries.map((registry) => (
                <div
                  key={registry.id}
                  className="result-card"
                  style={{
                    borderColor: selectedRegistry?.id === registry.id ? "var(--color-primary)" : undefined,
                  }}
                >
                  <div className="result-card-header">
                    <strong>{registry.name}</strong>
                    {registry.is_active && <span className="badge badge-success">Active</span>}
                  </div>
                  {registry.description && (
                    <p className="text-sm" style={{ margin: "0.5rem 0" }}>{registry.description}</p>
                  )}
                  <div className="text-xs text-muted" style={{ marginBottom: "0.5rem" }}>
                    {registry.ontology_name || "Untitled ontology"}
                    {registry.ontology_version ? ` · v${registry.ontology_version}` : ""}
                    {registry.source_id ? " · linked to source" : ""}
                    {registry.profiles.length > 0 ? ` · ${registry.profiles.length} profile(s)` : ""}
                  </div>
                  <div className="flex-center gap-sm">
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={() => loadRegistryIntoEditor(registry)}
                    >
                      Edit
                    </button>
                    {!registry.is_active && (
                      <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        disabled={activatingId === registry.id}
                        onClick={() => void handleActivate(registry.id)}
                      >
                        {activatingId === registry.id ? "Activating..." : "Activate"}
                      </button>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>
      )}
    </div>
  );
}
