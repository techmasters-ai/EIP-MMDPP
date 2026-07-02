/**
 * Typed API client for EIP-MMDPP backend.
 *
 * All functions return typed data or throw an Error with a message
 * suitable for display in the UI.
 */

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export interface Source {
  id: string;
  name: string;
  description?: string;
  created_at: string;
}

export interface Document {
  id: string;
  source_id: string;
  filename: string;
  pipeline_status: string;
  pipeline_stage?: string;
  error_message?: string;
  failed_stages?: string[];
  created_at: string;
}

export interface WatchDir {
  id: string;
  source_id: string;
  path: string;
  poll_interval_seconds: number;
  file_patterns: string[];
  enabled: boolean;
  created_at: string;
}

export interface QueryResultItem {
  chunk_id?: string;
  artifact_id?: string;
  document_id?: string;
  document_name?: string;
  score: number;
  modality: string;
  content_text?: string;
  page_number?: number;
  classification: string;
  // Data lineage
  source_characterization?: string;
  date_of_information?: string;
  extraction_confidence?: number;
  sources?: Array<{
    document_id: string;
    page_number?: number;
    classification?: string;
    chunk_text_preview?: string;
  }>;
  context?: Record<string, unknown>;
  image_url?: string;
  // Chunk-level provenance (populated by /v1/retrieval/query and /v1/graph/query)
  self_refs?: string[];
  evidence_ids?: string[];
  page_numbers?: number[];
}

export type QueryStrategy = "basic" | "hybrid" | "global";
export type ModalityFilter = "all" | "text" | "image";

export interface UnifiedQueryResponse {
  query_text?: string;
  query_image?: string;
  strategy: string;
  modality_filter: string;
  results: QueryResultItem[];
  total: number;
}

export interface AgentSource {
  chunk_id?: string;
  score: number;
  modality: string;
  classification: string;
}

export interface AgentContextResponse {
  query: string;
  strategy: string;
  modality_filter: string;
  total_results: number;
  context: string;
  sources: AgentSource[];
}

export interface GraphIngestResponse {
  status: string;
  node_id?: string;
}

export interface TrustedDataSubmission {
  id: string;
  content: string;
  source_context?: Record<string, unknown>;
  confidence: number;
  status: string;
  proposed_by?: string;
  reviewed_by?: string;
  review_notes?: string;
  created_at: string;
  index_status?: string;
  index_error?: string;
  embedding_model?: string;
  embedded_at?: string;
}

export interface TrustedDataQueryResult {
  content_text: string;
  score: number;
  submission_id?: string;
  confidence?: number;
  classification?: string;
}

export interface TrustedDataQueryResponse {
  query: string;
  results: TrustedDataQueryResult[];
  total: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Turn a FastAPI error `detail` into a human-readable message.
 *
 * `detail` arrives as: a pre-formatted string (409s, dossier-ref errors, the
 * backend's `_assert_valid_shape` message), OR an array of
 * `{loc, msg, type}` request-schema validation errors. For the array form we
 * join each `msg` (prefixed with the last, non-`body` `loc` segment) into
 * readable lines rather than dumping raw JSON.
 */
export function formatApiErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const lines = detail
      .map((item) => {
        if (!item || typeof item !== "object") return null;
        const record = item as { loc?: unknown; msg?: unknown };
        const msg = typeof record.msg === "string" ? record.msg : null;
        if (!msg) return null;
        const loc = Array.isArray(record.loc)
          ? record.loc.filter((seg) => seg !== "body").map(String)
          : [];
        const field = loc.length > 0 ? loc[loc.length - 1] : null;
        return field ? `${field}: ${msg}` : msg;
      })
      .filter((line): line is string => Boolean(line));
    if (lines.length > 0) return lines.join("; ");
  }
  return fallback;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail: unknown = null;
    try {
      const body = await res.json();
      detail = body?.detail ?? null;
    } catch {
      // ignore parse error
    }
    throw new Error(formatApiErrorDetail(detail, `HTTP ${res.status}`));
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

export async function listSources(): Promise<Source[]> {
  const res = await fetch("/v1/sources");
  return handleResponse<Source[]>(res);
}

export async function createSource(name: string, description?: string): Promise<Source> {
  const res = await fetch("/v1/sources", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description }),
  });
  return handleResponse<Source>(res);
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

export async function listDocumentsBySource(sourceId: string): Promise<Document[]> {
  const res = await fetch(`/v1/sources/${sourceId}/documents`);
  return handleResponse<Document[]>(res);
}

export async function getDocumentStatus(documentId: string): Promise<Document> {
  const res = await fetch(`/v1/documents/${documentId}/status`);
  return handleResponse<Document>(res);
}

export async function batchDocumentStatus(ids: string[]): Promise<Document[]> {
  const res = await fetch("/v1/documents/batch-status", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_ids: ids }),
  });
  return handleResponse<Document[]>(res);
}

export function uploadFile(
  sourceId: string,
  file: File,
  onProgress?: (pct: number) => void,
): Promise<Document> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as Document);
        } catch {
          reject(new Error("Invalid JSON response from server"));
        }
      } else {
        let detail = `HTTP ${xhr.status}`;
        try {
          const body = JSON.parse(xhr.responseText);
          detail = body?.detail ?? detail;
        } catch {
          // ignore
        }
        reject(new Error(typeof detail === "string" ? detail : JSON.stringify(detail)));
      }
    };

    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.onabort = () => reject(new Error("Upload aborted"));

    xhr.open("POST", `/v1/sources/${sourceId}/documents`);
    xhr.send(form);
  });
}

// ---------------------------------------------------------------------------
// Watch Directories
// ---------------------------------------------------------------------------

export async function listWatchDirs(): Promise<WatchDir[]> {
  const res = await fetch("/v1/watch-dirs");
  return handleResponse<WatchDir[]>(res);
}

export async function createWatchDir(params: {
  source_id: string;
  path: string;
  poll_interval_seconds?: number;
  file_patterns?: string[];
}): Promise<WatchDir> {
  const res = await fetch("/v1/watch-dirs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<WatchDir>(res);
}

export async function deleteWatchDir(id: string): Promise<void> {
  const res = await fetch(`/v1/watch-dirs/${id}`, { method: "DELETE" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.detail ?? `HTTP ${res.status}`);
  }
}

// ---------------------------------------------------------------------------
// Unified Retrieval
// ---------------------------------------------------------------------------

export async function unifiedQuery(params: {
  query_text?: string;
  query_image?: string;
  strategy: QueryStrategy;
  modality_filter: ModalityFilter;
  top_k?: number;
  reranker_top_n?: number;
  min_confidence?: number;
  include_context?: boolean;
  ontology_reserved_slots?: number;
}): Promise<UnifiedQueryResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 min timeout
  try {
    const res = await fetch("/v1/retrieval/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ top_k: 10, include_context: true, ...params }),
      signal: controller.signal,
    });
    return await handleResponse<UnifiedQueryResponse>(res);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Search timed out. The first query may be slow while models load — please try again.");
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

// ---------------------------------------------------------------------------
// Graph Store
// ---------------------------------------------------------------------------

export async function ingestGraphEntity(params: {
  entity_type: string;
  name: string;
  properties?: Record<string, unknown>;
}): Promise<GraphIngestResponse> {
  const res = await fetch("/v1/graph/ingest/entity", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<GraphIngestResponse>(res);
}

export async function ingestGraphRelationship(params: {
  from_entity: string;
  from_type: string;
  to_entity: string;
  to_type: string;
  relationship_type: string;
}): Promise<GraphIngestResponse> {
  const res = await fetch("/v1/graph/ingest/relationship", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<GraphIngestResponse>(res);
}

export async function queryGraph(params: {
  query: string;
  top_k?: number;
}): Promise<{ results: QueryResultItem[] }> {
  const res = await fetch("/v1/graph/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: params.query, top_k: params.top_k ?? 20, hop_count: 2 }),
  });
  const data = await handleResponse<QueryResultItem[]>(res);
  return { results: data };
}

// ---------------------------------------------------------------------------
// Trusted Data
// ---------------------------------------------------------------------------

export async function proposeTrustedData(params: {
  content: string;
  source_context?: Record<string, unknown>;
  confidence?: number;
}): Promise<TrustedDataSubmission> {
  const res = await fetch("/v1/trusted-data/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<TrustedDataSubmission>(res);
}

export async function listTrustedDataSubmissions(status?: string): Promise<TrustedDataSubmission[]> {
  const url = status ? `/v1/trusted-data/proposals?status=${status}` : "/v1/trusted-data/proposals";
  const res = await fetch(url);
  return handleResponse<TrustedDataSubmission[]>(res);
}

export async function approveTrustedData(id: string, notes?: string): Promise<TrustedDataSubmission> {
  const res = await fetch(`/v1/trusted-data/proposals/${id}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ notes }),
  });
  return handleResponse<TrustedDataSubmission>(res);
}

export async function rejectTrustedData(id: string, notes?: string): Promise<TrustedDataSubmission> {
  const res = await fetch(`/v1/trusted-data/proposals/${id}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ notes }),
  });
  return handleResponse<TrustedDataSubmission>(res);
}

export async function reindexTrustedData(id: string): Promise<TrustedDataSubmission> {
  const res = await fetch(`/v1/trusted-data/proposals/${id}/reindex`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  return handleResponse<TrustedDataSubmission>(res);
}

export async function queryTrustedData(params: {
  query: string;
  top_k?: number;
}): Promise<TrustedDataQueryResponse> {
  const res = await fetch("/v1/trusted-data/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: params.query, top_k: params.top_k ?? 10 }),
  });
  return handleResponse<TrustedDataQueryResponse>(res);
}

// ---------------------------------------------------------------------------
// Document Reingest
// ---------------------------------------------------------------------------

export async function reingestDocument(
  documentId: string,
  mode: "full" | "embeddings_only" | "graph_only" = "full",
): Promise<{ document_id: string; mode: string; task_id: string }> {
  const res = await fetch(`/v1/documents/${documentId}/reingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  return handleResponse<{ document_id: string; mode: string; task_id: string }>(res);
}

export async function cancelDocument(documentId: string): Promise<{ document_id: string; status: string }> {
  const res = await fetch(`/v1/documents/${documentId}/cancel`, { method: "POST" });
  return handleResponse<{ document_id: string; status: string }>(res);
}

export async function deleteDocument(documentId: string): Promise<void> {
  const res = await fetch(`/v1/documents/${documentId}`, { method: "DELETE" });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // ignore
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

export async function deleteAllSourceDocuments(sourceId: string): Promise<{ deleted: number }> {
  const res = await fetch(`/v1/sources/${sourceId}/documents`, { method: "DELETE" });
  return handleResponse<{ deleted: number }>(res);
}

// ---------------------------------------------------------------------------
// LangGraph agent context
// ---------------------------------------------------------------------------

export async function getAgentContext(params: {
  query: string;
  strategy?: QueryStrategy;
  modality_filter?: ModalityFilter;
  top_k?: number;
}): Promise<AgentContextResponse> {
  const search = new URLSearchParams({
    query: params.query,
    strategy: params.strategy ?? "basic",
    modality_filter: params.modality_filter ?? "all",
    top_k: String(params.top_k ?? 10),
  });
  const res = await fetch(`/v1/agent/context?${search}`);
  return handleResponse<AgentContextResponse>(res);
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export interface RetrievalSettings {
  top_k: number;
  reranker_top_n: number;
  min_confidence: number;
}

export async function getRetrievalSettings(): Promise<RetrievalSettings> {
  const res = await fetch("/v1/settings/retrieval");
  return handleResponse<RetrievalSettings>(res);
}

// ---------------------------------------------------------------------------
// Graph Neighborhood
// ---------------------------------------------------------------------------

export interface GraphNeighborhoodResponse {
  center: Record<string, unknown> | null;
  nodes: Record<string, unknown>[];
  edges: Array<{
    source: string;
    target: string;
    rel_type: string;
    provenance?: {
      evidence_ids?: string[];
      self_refs?: string[];
      page_numbers?: number[];
    } | null;
    [key: string]: unknown;
  }>;
}

export async function getGraphNeighborhood(params: {
  entity_name: string;
  hop_count?: number;
}): Promise<GraphNeighborhoodResponse> {
  const res = await fetch("/v1/graph/neighborhood", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      entity_name: params.entity_name,
      hop_count: params.hop_count ?? 2,
    }),
  });
  return handleResponse<GraphNeighborhoodResponse>(res);
}

// ---------------------------------------------------------------------------
// Query Profiles
// ---------------------------------------------------------------------------

export interface QueryProfileStep {
  direction: "out" | "in";
  rel_types: string[];
  min_hops: number;
  max_hops: number;
}

export interface QueryProfileTraversal {
  steps: QueryProfileStep[];
}

export type QueryProfileKind = "section" | "section_properties" | "dossier";

/** The nested ``definition`` dict carried by a flat query profile row. */
export interface QueryProfileDefinitionBody {
  target_entity_types?: string[];
  traversals?: QueryProfileTraversal[];
  section_profile_ids?: string[];
  profile_sections?: string[];
  profile_subgroup?: string | null;
  include_associated_systems?: boolean;
  placeholder_query?: string | null;
}

/** A query profile row as returned by the flat CRUD + list endpoints. */
export interface QueryProfileResponse {
  id?: string | null;
  profile_key: string;
  label: string;
  description?: string | null;
  kind: QueryProfileKind;
  root_entity_types: string[];
  definition: QueryProfileDefinitionBody;
  source_id?: string | null;
  enabled: boolean;
  created_at?: string | null;
  updated_at?: string | null;
}

/** Create payload for POST /v1/query-profiles. */
export interface QueryProfileCreate {
  profile_key: string;
  label: string;
  description?: string | null;
  kind: QueryProfileKind;
  root_entity_types: string[];
  definition: QueryProfileDefinitionBody;
  source_id?: string | null;
  enabled: boolean;
}

/** Partial update payload for PUT /v1/query-profiles/{profile_key}. Only
 * fields present are applied; an explicit ``source_id: null`` clears to Global. */
export interface QueryProfileUpdate {
  label?: string;
  description?: string | null;
  kind?: QueryProfileKind;
  root_entity_types?: string[];
  definition?: QueryProfileDefinitionBody;
  source_id?: string | null;
  enabled?: boolean;
}

// Live ontology (GET /v1/ontology) — served from the air_defense_v3 SSoT.
export interface OntologyEntityType {
  name: string;
  label: string;
}

export interface OntologyRelationshipType {
  name: string;
}

export interface OntologySection {
  name: string;
  description: string;
}

export interface OntologyResponse {
  version: string;
  entity_types: OntologyEntityType[];
  relationship_types: OntologyRelationshipType[];
  profile_sections: OntologySection[];
}

export interface QueryProfileFieldEvidence {
  chunk_id?: string | null;
  chunk_type?: string | null;
  artifact_id?: string | null;
  document_id?: string | null;
  document_name?: string | null;
  modality?: string | null;
  page_number?: number | null;
  classification: string;
  content_text?: string | null;
  source_characterization?: string | null;
  date_of_information?: string | null;
  extraction_confidence?: number | null;
  supporting_snippet: string;
  element_uid?: string | null;
}

export interface QueryProfileFieldEntry {
  name: string;
  label: string;
  value: unknown;
  description?: string | null;
  examples?: unknown[] | null;
  enum?: string[] | null;
  evidence: QueryProfileFieldEvidence[];
}

export interface QueryProfileFieldGroup {
  subgroup?: string | null;
  subgroup_label?: string | null;
  fields: QueryProfileFieldEntry[];
}

export interface GraphProfileEntityResult {
  node_id?: string | null;
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
    chunk_type: string;
    artifact_id?: string | null;
    document_id?: string | null;
    document_name?: string | null;
    modality: string;
    page_number?: number | null;
    classification: string;
    content_text?: string | null;
  }>;
}

export interface QueryProfileSectionResponse {
  profile_id: string;
  profile_label: string;
  resolved_root: GraphProfileEntityResult;
  field_groups: QueryProfileFieldGroup[];
  related_systems: GraphProfileEntityResult[];
  items: GraphProfileEntityResult[];
  total: number;
}

export interface QueryProfileDossierSection {
  profile_id: string;
  profile_label: string;
  kind: "section" | "section_properties";
  field_groups: QueryProfileFieldGroup[];
  related_systems: GraphProfileEntityResult[];
  items: GraphProfileEntityResult[];
  total: number;
}

export interface QueryProfileDossierResponse {
  profile_id: string;
  profile_label: string;
  resolved_root: GraphProfileEntityResult;
  aliases: string[];
  sections: QueryProfileDossierSection[];
  total?: number;
}

export async function getOntology(): Promise<OntologyResponse> {
  return handleResponse<OntologyResponse>(await fetch("/v1/ontology"));
}

export async function listQueryProfiles(enabledOnly = false): Promise<QueryProfileResponse[]> {
  const url = enabledOnly ? "/v1/query-profiles?enabled_only=true" : "/v1/query-profiles";
  return handleResponse<QueryProfileResponse[]>(await fetch(url));
}

export async function createQueryProfile(body: QueryProfileCreate): Promise<QueryProfileResponse> {
  const res = await fetch("/v1/query-profiles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleResponse<QueryProfileResponse>(res);
}

export async function updateQueryProfile(
  profileKey: string,
  body: QueryProfileUpdate,
): Promise<QueryProfileResponse> {
  const res = await fetch(`/v1/query-profiles/${encodeURIComponent(profileKey)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleResponse<QueryProfileResponse>(res);
}

export async function deleteQueryProfile(profileKey: string): Promise<void> {
  const res = await fetch(`/v1/query-profiles/${encodeURIComponent(profileKey)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    let detail: unknown = null;
    try {
      const body = await res.json();
      detail = body?.detail ?? null;
    } catch {
      // ignore parse error
    }
    throw new Error(formatApiErrorDetail(detail, `HTTP ${res.status}`));
  }
}

export async function searchQueryProfileSection(params: {
  profile_id: string;
  query_text: string;
  include_aliases?: boolean;
  include_evidence?: boolean;
  evidence_top_k?: number;
  top_k?: number;
}): Promise<QueryProfileSectionResponse> {
  const res = await fetch("/v1/query-profiles/search/section", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<QueryProfileSectionResponse>(res);
}

export async function searchQueryProfileDossier(params: {
  profile_id: string;
  query_text: string;
  include_aliases?: boolean;
  include_evidence?: boolean;
  evidence_top_k?: number;
  top_k?: number;
}): Promise<QueryProfileDossierResponse> {
  const res = await fetch("/v1/query-profiles/search/dossier", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<QueryProfileDossierResponse>(res);
}

// ---------------------------------------------------------------------------
// Docling Document
// ---------------------------------------------------------------------------

export interface DoclingImageRef {
  element_uid: string;
  url: string;
}

export interface DoclingDocumentResponse {
  document_id: string;
  filename: string;
  markdown: string;
  document_json: Record<string, unknown>;
  images: DoclingImageRef[];
}

export async function getDoclingDocument(documentId: string): Promise<DoclingDocumentResponse> {
  const res = await fetch(`/v1/documents/${documentId}/docling`);
  return handleResponse<DoclingDocumentResponse>(res);
}

export async function getDoclingRawJson(documentId: string): Promise<Record<string, unknown>> {
  const res = await fetch(`/v1/documents/${documentId}/docling-raw`);
  return handleResponse<Record<string, unknown>>(res);
}

export interface ImageDescription {
  element_uid: string;
  content_text: string;
  page_number: number | null;
  artifact_id: string | null;
}

export async function getDocumentImageDescriptions(
  documentId: string,
): Promise<ImageDescription[]> {
  try {
    const res = await fetch(`/v1/documents/${documentId}/image-descriptions`);
    if (res.status === 404) return [];
    return handleResponse<ImageDescription[]>(res);
  } catch {
    return [];
  }
}

export async function getDocumentMetadata(
  documentId: string,
): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`/v1/documents/${documentId}/metadata`);
    if (res.status === 404) return null;
    return handleResponse<Record<string, unknown>>(res);
  } catch {
    return null;
  }
}

export interface DocumentTranslation {
  document_id: string;
  detected_language: string;
  translated_markdown: string;
}

export async function getDocumentTranslation(documentId: string): Promise<DocumentTranslation | null> {
  try {
    const res = await fetch(`/v1/documents/${documentId}/translation`);
    if (res.status === 404) return null;
    return handleResponse<DocumentTranslation>(res);
  } catch {
    return null;
  }
}

export interface ElementTranslation {
  element_uid: string;
  original_text: string;
  translated_text: string;
}

export async function getElementTranslations(documentId: string): Promise<ElementTranslation[]> {
  try {
    const res = await fetch(`/v1/documents/${documentId}/element-translations`);
    if (res.status === 404) return [];
    return handleResponse<ElementTranslation[]>(res);
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Community / Global Search Indexing
// ---------------------------------------------------------------------------

export interface IndexingSettings {
  indexing_enabled: boolean;
  indexing_interval_minutes: number;
  post_ingest_enabled: boolean;
  post_ingest_threshold: number;
  algorithm: string;
  last_run: {
    status: string;
    started_at: string | null;
    completed_at: string | null;
    total_communities: number;
    reports_generated: number;
    reports_reused: number;
  } | null;
  last_indexing_at: string | null;
}

export interface CommunityReport {
  community_id: number;
  title: string;
  summary: string;
  member_count: number;
  generated_at?: string;
}

export async function getIndexingSettings(): Promise<IndexingSettings> {
  return handleResponse<IndexingSettings>(await fetch("/v1/community/settings"));
}

export async function triggerIndexing(mode: "incremental" | "full"): Promise<{ run_id: string }> {
  return handleResponse(
    await fetch("/v1/community/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    })
  );
}

export async function getIndexingStatus(): Promise<Record<string, unknown>> {
  return handleResponse(await fetch("/v1/community/status"));
}

export async function getCommunityReports(): Promise<{ reports: CommunityReport[]; total: number }> {
  return handleResponse(await fetch("/v1/community/reports"));
}
