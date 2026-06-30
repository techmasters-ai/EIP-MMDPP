import React, { useState, useCallback, useEffect, useMemo } from "react";
import {
  getActiveQueryProfiles,
  unifiedQuery,
  getGraphNeighborhood,
  getRetrievalSettings,
  searchQueryProfileDossier,
  searchQueryProfileSection,
  type QueryProfileDefinition,
  type QueryProfileSectionResponse,
  type QueryProfileDossierResponse,
  type QueryStrategy,
  type ModalityFilter,
  type QueryResultItem,
} from "../api/client";
import { GraphView, toGraphElements } from "./GraphView";
import { FieldGroupTable } from "./FieldGroupTable";
import { DossierSectionList } from "./DossierSectionList";
import { ProvenanceChips } from "./ProvenanceChips";
import type cytoscape from "cytoscape";

interface RetrievalModePreset {
  kind: "retrieval";
  key: string;
  strategy: QueryStrategy;
  label: string;
  description: string;
}

interface GraphProfileModePreset {
  kind: "graph_profile";
  key: string;
  profileId: string;
  profileKind: "section" | "section_properties" | "dossier";
  label: string;
  description: string;
  placeholder?: string | null;
}

type ModePreset = RetrievalModePreset | GraphProfileModePreset;

const BASE_MODES: RetrievalModePreset[] = [
  {
    kind: "retrieval",
    key: "basic",
    strategy: "basic",
    label: "Text Basic",
    description: "Simple BGE vector RAG search on text chunks",
  },
  {
    kind: "retrieval",
    key: "hybrid",
    strategy: "hybrid",
    label: "Multi-Modal",
    description: "Hybrid pipeline (vectors + graph expansion)",
  },
  {
    kind: "retrieval",
    key: "global",
    strategy: "global",
    label: "Global Research",
    description: "Cross-community synthesis for broad research questions",
  },
];

const MODALITY_OPTIONS: { value: ModalityFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "text", label: "Text Only" },
  { value: "image", label: "Images Only" },
];

function isRetrievalMode(mode: ModePreset): mode is RetrievalModePreset {
  return mode.kind === "retrieval";
}

function isGraphProfileMode(mode: ModePreset): mode is GraphProfileModePreset {
  return mode.kind === "graph_profile";
}

function toGraphProfileMode(profile: QueryProfileDefinition): GraphProfileModePreset {
  return {
    kind: "graph_profile",
    key: `profile:${profile.id}`,
    profileId: profile.id,
    profileKind: profile.kind,
    label: profile.label,
    description: profile.description || "Deterministic graph traversal from the active query profile registry",
    placeholder: profile.placeholder_query,
  };
}

function scoreColor(score: number): string {
  if (score >= 0.85) return "var(--color-success)";
  if (score >= 0.65) return "var(--color-primary)";
  return "var(--color-text-muted)";
}

function ImageLightbox({ src, alt, onClose }: { src: string; alt: string; onClose: () => void }) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "rgba(0,0,0,0.8)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
      }}
      onClick={onClose}
    >
      <img
        src={src}
        alt={alt}
        style={{
          maxWidth: "90vw",
          maxHeight: "90vh",
          objectFit: "contain",
          borderRadius: "6px",
        }}
        onClick={(e) => e.stopPropagation()}
      />
      <button
        onClick={onClose}
        style={{
          position: "absolute",
          top: "1rem",
          right: "1rem",
          background: "rgba(255,255,255,0.15)",
          border: "none",
          color: "#fff",
          fontSize: "1.5rem",
          cursor: "pointer",
          borderRadius: "50%",
          width: "2.5rem",
          height: "2.5rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        &times;
      </button>
    </div>
  );
}

/* ---------- Global query: synthesized answer from community reports ---------- */
function GlobalQueryDetail({ result }: { result: QueryResultItem }) {
  const ctx = result.context as Record<string, unknown> | undefined;
  const reports = ctx?.reports as Array<Record<string, unknown>> | undefined;
  const sources = result.sources;

  return (
    <div className="space-y-4">
      <div className="prose max-w-none" style={{ whiteSpace: "pre-line" }}>
        {result.content_text || "No results"}
      </div>

      {/* Source documents with page numbers */}
      {sources && sources.length > 0 && (
        <details className="mt-4" open>
          <summary className="cursor-pointer text-sm font-medium" style={{ color: "var(--color-text-muted)" }}>
            Sources ({sources.length} document{sources.length !== 1 ? "s" : ""})
          </summary>
          <div style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
            {sources.map((src, i) => (
              <div key={i} style={{ marginBottom: "0.25rem", color: "var(--color-text-muted)" }}>
                <strong>{String(src.document_id).slice(0, 8)}...</strong>
                {src.page_number != null && <> p.{String(src.page_number)}</>}
                {src.classification && src.classification !== "UNCLASSIFIED" && (
                  <span style={{ marginLeft: "0.5rem", color: "var(--color-warning)" }}>[{String(src.classification)}]</span>
                )}
                {src.chunk_text_preview && (
                  <span style={{ marginLeft: "0.5rem", opacity: 0.7 }}>
                    {String(src.chunk_text_preview).slice(0, 100)}...
                  </span>
                )}
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Community reports used for synthesis */}
      {reports && reports.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm" style={{ color: "var(--color-text-muted)" }}>
            Community Reports ({reports.length})
          </summary>
          <div style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
            {reports.map((r, i) => (
              <div key={i} style={{
                marginBottom: "0.5rem", padding: "0.5rem",
                border: "1px solid var(--color-border)", borderRadius: "var(--radius)",
              }}>
                <strong>Community {String(r.community_id)}: {String(r.title)}</strong>
                <span style={{ marginLeft: "0.5rem", color: "var(--color-text-muted)", fontSize: "0.8rem" }}>
                  score: {Number(r.score || 0).toFixed(2)}
                </span>
                <div style={{ marginTop: "0.25rem", color: "var(--color-text-muted)" }}>
                  {String(r.summary || "").slice(0, 200)}
                  {String(r.summary || "").length > 200 ? "..." : ""}
                </div>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

/* ---------- Generic metadata detail section ---------- */
function MetadataDetail({ item }: { item: QueryResultItem }) {
  const ctx = item.context as Record<string, unknown> | undefined;

  return (
    <div style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <tbody>
          {item.chunk_id && (
            <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
              <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>chunk_id</td>
              <td style={{ padding: "0.2rem 0.5rem", fontFamily: "monospace" }}>{item.chunk_id}</td>
            </tr>
          )}
          {item.artifact_id && (
            <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
              <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>artifact_id</td>
              <td style={{ padding: "0.2rem 0.5rem", fontFamily: "monospace" }}>{item.artifact_id}</td>
            </tr>
          )}
          {item.document_id && (
            <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
              <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>document_id</td>
              <td style={{ padding: "0.2rem 0.5rem", fontFamily: "monospace" }}>{item.document_id}</td>
            </tr>
          )}
          {item.document_name && (
            <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
              <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>document_name</td>
              <td style={{ padding: "0.2rem 0.5rem" }}>{item.document_name}</td>
            </tr>
          )}
          <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
            <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>score</td>
            <td style={{ padding: "0.2rem 0.5rem" }}>{item.score.toFixed(4)}</td>
          </tr>
          <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
            <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>modality</td>
            <td style={{ padding: "0.2rem 0.5rem" }}>{item.modality}</td>
          </tr>
          {item.page_number != null && (
            <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
              <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>page</td>
              <td style={{ padding: "0.2rem 0.5rem" }}>{item.page_number}</td>
            </tr>
          )}
          <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
            <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500 }}>classification</td>
            <td style={{ padding: "0.2rem 0.5rem" }}>{item.classification}</td>
          </tr>
          {ctx && Object.keys(ctx).length > 0 && (
            <tr style={{ borderBottom: "1px solid var(--color-border, #e0e0e0)" }}>
              <td style={{ padding: "0.2rem 0.5rem", fontWeight: 500, verticalAlign: "top" }}>context</td>
              <td style={{ padding: "0.2rem 0.5rem" }}>
                <pre style={{ margin: 0, fontSize: "0.8rem", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                  {JSON.stringify(ctx, null, 2)}
                </pre>
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

/** Extract entity names from graph profile context if present. */
function extractGraphEntities(ctx: Record<string, unknown> | undefined): string[] {
  if (!ctx) return [];
  if (ctx.source === "graph_profile" && typeof ctx.entity_name === "string") {
    return [ctx.entity_name];
  }
  return [];
}

/* ---------- Provenance types and components ---------- */

interface ProvenanceSourceDoc {
  document_id: string;
  document_title: string;
}

interface ProvenanceEntity {
  id: number;
  title: string;
  type: string;
  description: string;
  source_documents: ProvenanceSourceDoc[];
}

interface ProvenanceRelationship {
  id: number;
  source: string;
  target: string;
  description: string;
  source_documents: ProvenanceSourceDoc[];
}

interface ProvenanceTextUnit {
  id: number;
  text: string;
  source_documents: ProvenanceSourceDoc[];
}

interface ProvenanceCovariate {
  id: number;
  description: string;
  source_documents: ProvenanceSourceDoc[];
}

interface ProvenanceEntry {
  report_id: string | null;
  report_title: string | null;
  report_content: string | null;
  entities: ProvenanceEntity[];
  relationships: ProvenanceRelationship[];
  text_units: ProvenanceTextUnit[];
  covariates: ProvenanceCovariate[];
}

function DocBadges({ docs }: { docs: ProvenanceSourceDoc[] }) {
  if (!docs || docs.length === 0) return null;
  return (
    <div className="prov-doc-badges">
      {docs.map((d) => (
        <span key={d.document_id} className="prov-doc-badge" title={d.document_id}>
          {d.document_title || d.document_id}
        </span>
      ))}
    </div>
  );
}

function CollapsibleSection({ title, count, children }: {
  title: string;
  count: number;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  if (count === 0) return null;
  return (
    <div className="prov-section">
      <button className="prov-section-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? "\u25BC" : "\u25B6"} {title} ({count})
      </button>
      {open && <div className="prov-section-content">{children}</div>}
    </div>
  );
}

function ProvenancePanel({ provenance }: { provenance: ProvenanceEntry[] }) {
  const [open, setOpen] = useState(false);

  if (!provenance || provenance.length === 0) return null;

  const reportCount = provenance.filter((p) => p.report_id).length;
  const label = reportCount > 0
    ? `Provenance (${reportCount} community report${reportCount !== 1 ? "s" : ""})`
    : "Provenance";

  return (
    <div className="provenance-panel">
      <button
        className="prov-toggle"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "\u25BC" : "\u25B6"} {label}
      </button>
      {open && provenance.map((entry, i) => (
        <ProvenanceReportEntry key={entry.report_id || i} entry={entry} />
      ))}
    </div>
  );
}

function ProvenanceReportEntry({ entry }: { entry: ProvenanceEntry }) {
  const [open, setOpen] = useState(false);
  const [contentOpen, setContentOpen] = useState(false);

  return (
    <div className="prov-report">
      <button className="prov-report-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? "\u25BC" : "\u25B6"} {entry.report_title || "Source Texts"}
      </button>
      {open && (
        <div className="prov-report-content">
          {entry.report_content && (
            <div className="prov-section">
              <button className="prov-section-toggle" onClick={() => setContentOpen((v) => !v)}>
                {contentOpen ? "\u25BC" : "\u25B6"} Report Content
              </button>
              {contentOpen && (
                <pre className="prov-report-text">{entry.report_content}</pre>
              )}
            </div>
          )}

          <CollapsibleSection title="Entities" count={entry.entities.length}>
            {entry.entities.map((e) => (
              <div key={e.id} className="prov-item">
                <div>
                  <span className="badge badge-info" style={{ marginRight: "0.25rem" }}>{e.type}</span>
                  <strong>{e.title}</strong>
                  {e.description && <span className="text-muted"> &mdash; {e.description.slice(0, 150)}</span>}
                </div>
                <DocBadges docs={e.source_documents} />
              </div>
            ))}
          </CollapsibleSection>

          <CollapsibleSection title="Relationships" count={entry.relationships.length}>
            {entry.relationships.map((r) => (
              <div key={r.id} className="prov-item">
                <div className="text-sm">
                  {r.source} &rarr; {r.target}: {r.description.slice(0, 150)}
                </div>
                <DocBadges docs={r.source_documents} />
              </div>
            ))}
          </CollapsibleSection>

          <CollapsibleSection title="Source Texts" count={entry.text_units.length}>
            {entry.text_units.map((t) => (
              <div key={t.id} className="prov-item">
                <pre className="prov-text-chunk">{t.text}</pre>
                <DocBadges docs={t.source_documents} />
              </div>
            ))}
          </CollapsibleSection>

          <CollapsibleSection title="Covariates" count={entry.covariates.length}>
            {entry.covariates.map((c) => (
              <div key={c.id} className="prov-item">
                <div className="text-sm">{c.description}</div>
                <DocBadges docs={c.source_documents} />
              </div>
            ))}
          </CollapsibleSection>
        </div>
      )}
    </div>
  );
}

function buildGraphProfileContentText(
  name: string,
  entityType: string,
  relationshipTypes: string[],
  properties: Record<string, unknown>,
): string {
  const summaryKeys = ["description", "summary", "value", "units", "designation", "frequency", "range"];
  const summaryParts = summaryKeys
    .map((key) => {
      const value = properties[key];
      if (value === undefined || value === null || value === "") return null;
      return `${key}: ${typeof value === "object" ? JSON.stringify(value) : String(value)}`;
    })
    .filter(Boolean);

  const lines = [
    `${name} (${entityType})`,
    relationshipTypes.length > 0 ? `Via: ${relationshipTypes.join(" -> ")}` : null,
    ...summaryParts,
  ].filter(Boolean);

  return lines.join("\n");
}

function mapGraphProfileItemToResult(
  item: {
    name: string;
    entity_type: string;
    canonical_name?: string | null;
    score?: number | null;
    hop_count?: number | null;
    relationship_types: string[];
    properties: Record<string, unknown>;
    aliases: string[];
    evidence: Array<{
      chunk_id?: string | null;
      artifact_id?: string | null;
      document_id?: string | null;
      document_name?: string | null;
      modality: string;
      page_number?: number | null;
      classification: string;
      content_text?: string | null;
    }>;
  },
  context: Record<string, unknown>,
): QueryResultItem {
  const primaryEvidence = item.evidence[0];
  const derivedScore =
    item.score ?? (item.hop_count != null ? Math.max(0.2, 1 / (item.hop_count + 1)) : 0.75);

  return {
    chunk_id: primaryEvidence?.chunk_id ?? undefined,
    artifact_id: primaryEvidence?.artifact_id ?? undefined,
    document_id: primaryEvidence?.document_id ?? undefined,
    document_name: primaryEvidence?.document_name ?? undefined,
    score: derivedScore,
    modality: "graph_profile",
    content_text: buildGraphProfileContentText(
      item.name,
      item.entity_type,
      item.relationship_types,
      item.properties,
    ),
    page_number: primaryEvidence?.page_number ?? undefined,
    classification: primaryEvidence?.classification ?? "UNCLASSIFIED",
    context: {
      source: "graph_profile",
      entity_name: item.name,
      entity_type: item.entity_type,
      canonical_name: item.canonical_name,
      relationship_types: item.relationship_types,
      hop_count: item.hop_count,
      properties: item.properties,
      aliases: item.aliases,
      evidence: item.evidence,
      ...context,
    },
  };
}

/* ---------- Result card ---------- */
function ResultCard({ item, index }: { item: QueryResultItem; index: number }) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [graphOpen, setGraphOpen] = useState(false);
  const [graphElements, setGraphElements] = useState<cytoscape.ElementDefinition[] | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);

  const displayText = item.content_text;
  const ctx = item.context as Record<string, unknown> | undefined;
  const isGlobal = ctx?.source === "global";
  const isGraphProfile = ctx?.source === "graph_profile";

  const graphEntities = isGraphProfile ? extractGraphEntities(ctx) : [];
  const hasGraphData = graphEntities.length > 0;
  const provenance = (ctx?.provenance as ProvenanceEntry[] | undefined) || [];

  const handleToggleGraph = async () => {
    if (graphOpen) {
      setGraphOpen(false);
      setGraphElements(null);
      setGraphError(null);
      return;
    }
    setGraphOpen(true);
    setGraphLoading(true);
    setGraphError(null);
    try {
      // Fetch neighborhood for the first entity in the context
      const entityName = graphEntities[0];
      const resp = await getGraphNeighborhood({ entity_name: entityName, hop_count: 2 });
      const elements = toGraphElements(resp, entityName);
      if (elements.length === 0) {
        setGraphElements(null);
        setGraphError(`No graph data found for "${entityName}".`);
      } else {
        setGraphElements(elements);
      }
    } catch (err) {
      console.error("Graph neighborhood fetch failed:", err);
      setGraphElements(null);
      setGraphError(err instanceof Error ? err.message : "Failed to load graph data");
    } finally {
      setGraphLoading(false);
    }
  };

  let provenanceLabel = "";
  if (ctx?.source === "ontology_relation") {
    const rel = ctx.rel_type as string | undefined;
    const related = ctx.related_entity as string | undefined;
    const reserved = ctx.reserved ? " (reserved)" : "";
    provenanceLabel = `Via ontology: ${rel || "relation"}${related ? ` → ${related}` : ""}${reserved}`;
  } else if (ctx?.source === "ontology") {
    const entity = ctx.entity_name as string | undefined;
    const entityType = ctx.entity_type as string | undefined;
    if (entity) {
      provenanceLabel = entityType
        ? `Via shared entity: ${entity} (${entityType})`
        : `Via shared entity: ${entity}`;
    }
  } else if (ctx?.source === "doc_structure") {
    const linkType = ctx.link_type as string | undefined;
    const hops = (ctx.hops as number) || 1;
    provenanceLabel = `Via document structure: ${linkType || "link"}${hops > 1 ? ` (${hops} hops)` : ""}`;
  } else if (ctx?.source === "cross_modal") {
    const edge = ctx.edge_type as string | undefined;
    if (edge) provenanceLabel = `Via graph bridge: ${edge}`;
  } else if (isGlobal) {
    provenanceLabel = "Global Research: cross-community synthesis";
  } else if (isGraphProfile) {
    const profileLabel = ctx?.profile_label as string | undefined;
    const sectionLabel = ctx?.section_label as string | undefined;
    provenanceLabel = sectionLabel
      ? `Query Profile: ${profileLabel || "Graph Profile"} / ${sectionLabel}`
      : `Query Profile: ${profileLabel || "Graph Profile"}`;
  }

  // Preview: first 300 chars always visible
  const previewLen = 300;
  const preview = displayText
    ? displayText.length > previewLen
      ? displayText.slice(0, previewLen) + "\u2026"
      : displayText
    : null;

  return (
    <div className="result-card">
      <div className="result-card-header">
        <span className="text-xs text-muted">#{index + 1}</span>
        <span className="result-score" style={{ color: scoreColor(item.score) }}>
          {(item.score * 100).toFixed(0)}%
        </span>
        <span className="badge badge-info">{item.modality}</span>
        {item.classification && item.classification !== "UNCLASSIFIED" && (
          <span className="badge badge-error">{item.classification}</span>
        )}
        {item.document_name && (
          <span className="text-xs text-muted" title={item.document_name}>{item.document_name}</span>
        )}
        {item.page_number != null && (
          <span className="text-xs text-muted">p.{item.page_number}</span>
        )}
        {/* Multi-page indicator — only when page_numbers carries extra pages beyond the primary */}
        {item.page_numbers && item.page_numbers.length > 1 && (() => {
          const extra = item.page_numbers.filter((n) => n !== item.page_number);
          return extra.length > 0 ? (
            <span
              className="badge badge-info text-xs"
              title={`Spans pages: ${item.page_numbers.join(", ")}`}
              style={{ fontSize: "0.72rem" }}
            >
              pp.&nbsp;{item.page_numbers[0]}&ndash;{item.page_numbers[item.page_numbers.length - 1]}
            </span>
          ) : null;
        })()}
        {hasGraphData && (
          <button
            className={`btn btn-ghost btn-sm graph-toggle-btn${graphOpen ? " active" : ""}`}
            onClick={() => void handleToggleGraph()}
            title="Toggle graph view"
          >
            ◉
          </button>
        )}
      </div>

      {provenanceLabel && (
        <div className="text-sm text-muted" style={{ marginBottom: "0.25rem" }}>
          {provenanceLabel}
        </div>
      )}

      {/* Inline image thumbnail */}
      {item.image_url && (
        <div style={{ margin: "0.5rem 0" }}>
          <img
            src={item.image_url}
            alt={item.content_text || "Retrieved image"}
            loading="lazy"
            style={{
              maxHeight: "200px",
              maxWidth: "100%",
              objectFit: "contain",
              borderRadius: "4px",
              border: "1px solid var(--color-border, #e0e0e0)",
              cursor: "pointer",
            }}
            title="Click for full size"
            onClick={() => setModalOpen(true)}
          />
          {modalOpen && (
            <ImageLightbox
              src={item.image_url}
              alt={item.content_text || "Retrieved image"}
              onClose={() => setModalOpen(false)}
            />
          )}
        </div>
      )}

      {/* Text preview — always visible */}
      {preview && (
        <p className="result-text">{preview}</p>
      )}

      {/* Graph view (when toggled and entity data present) */}
      {graphOpen && (
        graphLoading ? (
          <div className="graph-view-container" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span className="spinner" />
          </div>
        ) : graphElements && graphElements.length > 0 ? (
          <GraphView
            elements={graphElements}
            onNodeClick={() => {}}
            onClose={() => { setGraphOpen(false); setGraphElements(null); setGraphError(null); }}
          />
        ) : (
          <div className="alert alert-error mt-sm">
            {graphError || "No graph data available."}
          </div>
        )
      )}

      {/* Global query: synthesized answer */}
      {isGlobal && <GlobalQueryDetail result={item} />}

      {/* Show details toggle — full text + all metadata */}
      <button
        className="btn btn-ghost btn-sm mt-sm"
        onClick={() => setDetailsOpen((v) => !v)}
      >
        {detailsOpen ? "Hide details" : "Show details"}
      </button>

      {detailsOpen && (
        <div style={{ marginTop: "0.5rem", borderTop: "1px solid var(--color-border, #e0e0e0)", paddingTop: "0.5rem" }}>
          {/* Full text if longer than preview */}
          {displayText && displayText.length > previewLen && (
            <div style={{ marginBottom: "0.5rem" }}>
              <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>Full Text</div>
              <p className="text-sm" style={{ whiteSpace: "pre-wrap" }}>{displayText}</p>
            </div>
          )}
          {provenance.length > 0 && <ProvenancePanel provenance={provenance} />}
          {/* Chunk-level evidence IDs from backend provenance fields */}
          <ProvenanceChips
            evidenceIds={[...(item.evidence_ids ?? []), ...(item.self_refs ?? []).filter((r) => !(item.evidence_ids ?? []).includes(r))]}
            pageNumbers={item.page_numbers}
          />
          <MetadataDetail item={item} />
        </div>
      )}
    </div>
  );
}

export function QueryPage() {
  const [queryText, setQueryText] = useState("");
  const [queryImage, setQueryImage] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [profileModes, setProfileModes] = useState<GraphProfileModePreset[]>([]);
  const [activeRegistryName, setActiveRegistryName] = useState<string | null>(null);
  const [modalityFilter, setModalityFilter] = useState<ModalityFilter>("all");
  const [topK, setTopK] = useState(10);
  const [rerankerTopN, setRerankerTopN] = useState(20);
  const [reservedSlots, setReservedSlots] = useState<number>(3);
  const [minConfidence, setMinConfidence] = useState(0.1);
  const [results, setResults] = useState<QueryResultItem[] | null>(null);
  const [totalResults, setTotalResults] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [sectionResponse, setSectionResponse] = useState<QueryProfileSectionResponse | null>(null);
  const [dossierResponse, setDossierResponse] = useState<QueryProfileDossierResponse | null>(null);

  const allModes = useMemo(() => [...BASE_MODES, ...profileModes], [profileModes]);

  // Fetch server defaults on mount
  useEffect(() => {
    getRetrievalSettings().then((s) => {
      setTopK(s.top_k);
      setRerankerTopN(s.reranker_top_n);
      setMinConfidence(s.min_confidence);
      setReservedSlots((s as any).ontology_reserved_slots ?? 3);
    }).catch(() => {});  // keep hardcoded defaults on failure
  }, []);

  useEffect(() => {
    getActiveQueryProfiles()
      .then((payload) => {
        setActiveRegistryName(
          payload.registry?.name
            ?? (payload.exposed_profiles.length > 0 ? "Current ontology template" : null),
        );
        setProfileModes(payload.exposed_profiles.map(toGraphProfileMode));
      })
      .catch(() => {
        setActiveRegistryName(null);
        setProfileModes([]);
      });
  }, []);

  useEffect(() => {
    if (selectedIdx >= allModes.length) {
      setSelectedIdx(0);
    }
  }, [allModes.length, selectedIdx]);

  const selected = allModes[selectedIdx] ?? BASE_MODES[0];
  const retrievalSelected = isRetrievalMode(selected) ? selected : null;
  const selectedIsRetrieval = retrievalSelected !== null;
  const selectedIsGraphProfile = isGraphProfileMode(selected);
  const showImageInput = retrievalSelected?.strategy === "hybrid";
  const queryPlaceholder = selectedIsGraphProfile
    ? (selected.placeholder || "e.g. exact entity name")
    : "e.g. Patriot PAC-3 guidance computer specifications";

  const handleImageFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setImagePreview(dataUrl);
      setQueryImage(dataUrl.split(",")[1]);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) handleImageFile(file);
    },
    [handleImageFile],
  );

  const clearImage = () => {
    setQueryImage(null);
    setImagePreview(null);
  };

  const hasQuery = selectedIsGraphProfile ? queryText.trim() : (queryText.trim() || queryImage);

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!hasQuery) return;

    setLoading(true);
    setError(null);
    setResults(null);
    setTotalResults(0);
    setElapsed(null);
    setSectionResponse(null);
    setDossierResponse(null);
    const t0 = performance.now();

    try {
      if (selectedIsGraphProfile) {
        if (selected.profileKind === "dossier") {
          const res = await searchQueryProfileDossier({
            profile_id: selected.profileId,
            query_text: queryText.trim(),
            include_aliases: true,
            include_evidence: true,
            evidence_top_k: 3,
            top_k: topK,
          });
          setDossierResponse(res);
          setTotalResults(res.total ?? 0);
          setElapsed(Math.round(performance.now() - t0));
        } else if (selected.profileKind === "section_properties") {
          const res = await searchQueryProfileSection({
            profile_id: selected.profileId,
            query_text: queryText.trim(),
            include_aliases: true,
            include_evidence: true,
            evidence_top_k: 3,
            top_k: topK,
          });
          setSectionResponse(res);
          setTotalResults(res.total ?? 0);
          setElapsed(Math.round(performance.now() - t0));
        } else {
          const res = await searchQueryProfileSection({
            profile_id: selected.profileId,
            query_text: queryText.trim(),
            include_aliases: true,
            include_evidence: true,
            evidence_top_k: 3,
            top_k: topK,
          });
          const flattened = res.items.map((item) =>
            mapGraphProfileItemToResult(item, {
              profile_id: res.profile_id,
              profile_label: res.profile_label,
              registry_id: res.registry_id,
              resolved_root: res.resolved_root,
            }),
          );
          setResults(flattened);
          setTotalResults(flattened.length);
          setElapsed(Math.round(performance.now() - t0));
        }
      } else if (retrievalSelected) {
        // Sync path: basic, hybrid, and global
        const res = await unifiedQuery({
          query_text: queryText.trim() || undefined,
          query_image: queryImage || undefined,
          strategy: retrievalSelected.strategy,
          modality_filter: retrievalSelected.strategy === "hybrid" ? modalityFilter : "all",
          top_k: topK,
          reranker_top_n: rerankerTopN,
          min_confidence: minConfidence,
          include_context: true,
          ontology_reserved_slots: retrievalSelected.strategy === "hybrid" ? reservedSlots : undefined,
        });
        setResults(res.results);
        setTotalResults(res.total);
        setElapsed(Math.round(performance.now() - t0));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="card card-body">
        <form onSubmit={(e) => void handleQuery(e)}>
          <div className="field">
            <label htmlFor="query-mode-select">Query mode</label>
            <select
              id="query-mode-select"
              value={selectedIdx}
              onChange={(e) => {
                setSelectedIdx(parseInt(e.target.value, 10));
                setModalityFilter("all");
              }}
              style={{ marginBottom: "0.5rem" }}
            >
              <optgroup label="Retrieval">
                {BASE_MODES.map((m, i) => (
                  <option key={m.key} value={i}>
                    {m.label} — {m.description}
                  </option>
                ))}
              </optgroup>
              {profileModes.length > 0 && (
                <optgroup label={activeRegistryName ? `Query Profiles (${activeRegistryName})` : "Query Profiles"}>
                  {profileModes.map((m, i) => (
                    <option key={m.key} value={BASE_MODES.length + i}>
                      {m.label} — {m.description}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            <div className="text-xs text-muted" style={{ marginBottom: "1rem" }}>
              {activeRegistryName
                ? `Active query profile registry: ${activeRegistryName}`
                : "No active query profile registry. Configure one to expose ontology-specific exact graph query modes."}
            </div>

            {/* Modality sub-filter for Multi-Modal */}
            {retrievalSelected?.strategy === "hybrid" && (
              <div className="mode-selector" style={{ marginBottom: "1rem" }}>
                {MODALITY_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    className={`mode-btn${modalityFilter === opt.value ? " active" : ""}`}
                    onClick={() => setModalityFilter(opt.value)}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          {showImageInput && (
            <div className="field" style={{ marginBottom: "1rem" }}>
              <label>Query image (optional &mdash; for image-based search)</label>
              {imagePreview ? (
                <div style={{ display: "flex", alignItems: "flex-start", gap: "1rem" }}>
                  <img
                    src={imagePreview}
                    alt="Query preview"
                    style={{ maxWidth: "160px", maxHeight: "120px", borderRadius: "4px", border: "1px solid var(--color-border)" }}
                  />
                  <button type="button" className="btn btn-ghost btn-sm" onClick={clearImage}>
                    Remove
                  </button>
                </div>
              ) : (
                <div
                  className="drop-zone"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={handleDrop}
                  onClick={() => document.getElementById("query-image-input")?.click()}
                  style={{
                    cursor: "pointer",
                    textAlign: "center",
                    padding: "1.5rem",
                    border: "2px dashed var(--color-border)",
                    borderRadius: "6px",
                  }}
                >
                  <div className="text-muted text-sm">
                    Drop an image here or click to select &mdash; text-only queries also work for text-to-image search
                  </div>
                  <input
                    id="query-image-input"
                    type="file"
                    accept="image/*"
                    style={{ display: "none" }}
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleImageFile(file);
                    }}
                  />
                </div>
              )}
            </div>
          )}

          <div className="field-row" style={{ alignItems: "flex-end" }}>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="query-input">
                {selectedIsGraphProfile ? "Root entity" : "Search query"}
              </label>
              <input
                id="query-input"
                type="search"
                placeholder={queryPlaceholder}
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                autoFocus
              />
            </div>
            <div className="field" style={{ width: "80px", flexShrink: 0 }}>
              <label htmlFor="top-k">Top K</label>
              <input
                id="top-k"
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value, 10) || 10)}
              />
            </div>
            {selectedIsRetrieval && (
              <div className="field" style={{ width: "110px", flexShrink: 0 }}>
                <label htmlFor="reranker-top-n">Reranker Top N</label>
                <input
                  id="reranker-top-n"
                  type="number"
                  min={1}
                  max={200}
                  value={rerankerTopN}
                  onChange={(e) => setRerankerTopN(parseInt(e.target.value, 10) || 20)}
                />
              </div>
            )}
            {selectedIsRetrieval && (
              <div className="field" style={{ width: "120px", flexShrink: 0 }}>
                <label htmlFor="min-confidence">Min Confidence</label>
                <input
                  id="min-confidence"
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={minConfidence}
                  onChange={(e) => setMinConfidence(parseFloat(e.target.value) || 0)}
                />
              </div>
            )}
            {retrievalSelected?.strategy === "hybrid" && (
              <div className="field" style={{ width: "140px", flexShrink: 0 }}>
                <label htmlFor="ontology-reserved-slots">Ontology reserved slots</label>
                <input
                  id="ontology-reserved-slots"
                  type="number"
                  min={0}
                  max={topK}
                  value={reservedSlots}
                  onChange={(e) =>
                    setReservedSlots(Math.max(0, Math.min(topK, Number(e.target.value) || 0)))
                  }
                />
              </div>
            )}
            <div style={{ paddingBottom: "0" }}>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading || !hasQuery}
              >
                {loading ? <><span className="spinner" /> Searching&hellip;</> : "Search"}
              </button>
            </div>
          </div>
        </form>
      </div>

      {error && <div className="alert alert-error" style={{ marginTop: "1rem" }}>{error}</div>}

      {sectionResponse && (
        <div className="results">
          <div className="flex-center gap-sm" style={{ marginBottom: "0.5rem" }}>
            <span className="text-sm text-muted">
              {sectionResponse.profile_label}
              {" \u2014 "}
              {sectionResponse.resolved_root.name}
              {totalResults > 0
                ? ` \u2014 ${totalResults} field${totalResults !== 1 ? "s" : ""}`
                : ""}
              {elapsed != null && ` \u2014 ${elapsed}ms`}
            </span>
          </div>
          <FieldGroupTable groups={sectionResponse.field_groups} />
          {sectionResponse.related_systems.length > 0 && (
            <div className="related-systems mt-md">
              <strong>Related systems:</strong>{" "}
              {sectionResponse.related_systems.map((s) => (
                <span key={s.node_id ?? s.name} className="chip">
                  {s.entity_type} / {s.name}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {dossierResponse && (
        <div className="results">
          <div className="flex-center gap-sm" style={{ marginBottom: "0.5rem" }}>
            <span className="text-sm text-muted">
              {dossierResponse.profile_label}
              {" \u2014 "}
              {dossierResponse.resolved_root.name}
              {elapsed != null && ` \u2014 ${elapsed}ms`}
            </span>
          </div>
          <DossierSectionList sections={dossierResponse.sections} />
        </div>
      )}

      {results !== null && (
        <div className="results">
          <div className="flex-center gap-sm" style={{ marginBottom: "0.5rem" }}>
            <span className="text-sm text-muted">
              {totalResults === 0
                ? "No results found."
                : `${totalResults} result${totalResults !== 1 ? "s" : ""}`}
              {elapsed != null && ` \u2014 ${elapsed}ms`}
            </span>
          </div>

          {totalResults === 0 ? (
            <div className="empty-state">
              <div className="empty-state-title">No matching documents</div>
              <div className="text-muted text-sm">
                Try a different query or upload documents first.
              </div>
            </div>
          ) : (
            results.map((item: QueryResultItem, i: number) => (
              <ResultCard key={`result-${i}`} item={item} index={i} />
            ))
          )}
        </div>
      )}

      <div className="api-info mt-md">
        <strong>LangGraph tool endpoint:</strong>{" "}
        <code>GET /v1/agent/context?query=&hellip;&amp;strategy=basic&amp;top_k=10</code>
        <br />
        Returns a pre-formatted markdown context string ready for LLM prompt injection.
      </div>
    </div>
  );
}
