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
  context?: Record<string, unknown>;
  image_url?: string;
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

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // ignore parse error
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
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
// GraphRAG Settings & Indexing
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

export interface QueryProfileDefinition {
  id: string;
  label: string;
  description?: string | null;
  kind: "section" | "dossier";
  exposed: boolean;
  root_entity_types: string[];
  target_entity_types: string[];
  traversals: QueryProfileTraversal[];
  section_profile_ids: string[];
  placeholder_query?: string | null;
}

export interface QueryProfileRegistry {
  id: string;
  name: string;
  description?: string | null;
  source_id?: string | null;
  ontology_name?: string | null;
  ontology_version?: string | null;
  ontology_definition?: Record<string, unknown> | null;
  profiles: QueryProfileDefinition[];
  is_active: boolean;
  created_by: string;
  created_at: string;
  updated_at: string;
}

export interface QueryProfileRegistryTemplate {
  name: string;
  description?: string | null;
  source_id?: string | null;
  ontology_name?: string | null;
  ontology_version?: string | null;
  ontology_definition?: Record<string, unknown> | null;
  profiles: QueryProfileDefinition[];
  is_active: boolean;
}

export interface ActiveQueryProfilesResponse {
  registry: QueryProfileRegistry | null;
  exposed_profiles: QueryProfileDefinition[];
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
  registry_id?: string | null;
  profile_id: string;
  profile_label: string;
  resolved_root: GraphProfileEntityResult;
  items: GraphProfileEntityResult[];
  total: number;
}

export interface QueryProfileDossierSection {
  profile_id: string;
  profile_label: string;
  items: GraphProfileEntityResult[];
  total: number;
}

export interface QueryProfileDossierResponse {
  registry_id?: string | null;
  profile_id: string;
  profile_label: string;
  resolved_root: GraphProfileEntityResult;
  aliases: string[];
  sections: QueryProfileDossierSection[];
}

export async function listQueryProfileRegistries(): Promise<QueryProfileRegistry[]> {
  const res = await fetch("/v1/query-profiles/registries");
  return handleResponse<QueryProfileRegistry[]>(res);
}

export async function createQueryProfileRegistry(params: {
  name: string;
  description?: string;
  source_id?: string | null;
  ontology_name?: string;
  ontology_version?: string;
  ontology_definition?: Record<string, unknown> | null;
  profiles: QueryProfileDefinition[];
  is_active?: boolean;
}): Promise<QueryProfileRegistry> {
  const res = await fetch("/v1/query-profiles/registries", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<QueryProfileRegistry>(res);
}

export async function updateQueryProfileRegistry(
  registryId: string,
  params: {
    name?: string;
    description?: string;
    source_id?: string | null;
    ontology_name?: string;
    ontology_version?: string;
    ontology_definition?: Record<string, unknown> | null;
    profiles?: QueryProfileDefinition[];
    is_active?: boolean;
  },
): Promise<QueryProfileRegistry> {
  const res = await fetch(`/v1/query-profiles/registries/${registryId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<QueryProfileRegistry>(res);
}

export async function activateQueryProfileRegistry(
  registryId: string,
): Promise<QueryProfileRegistry> {
  const res = await fetch(`/v1/query-profiles/registries/${registryId}/activate`, {
    method: "POST",
  });
  return handleResponse<QueryProfileRegistry>(res);
}

export async function createRegistryQueryProfile(
  registryId: string,
  profile: QueryProfileDefinition,
): Promise<QueryProfileRegistry> {
  const res = await fetch(`/v1/query-profiles/registries/${registryId}/profiles`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
  return handleResponse<QueryProfileRegistry>(res);
}

export async function updateRegistryQueryProfile(
  registryId: string,
  profileId: string,
  profile: QueryProfileDefinition,
): Promise<QueryProfileRegistry> {
  const res = await fetch(
    `/v1/query-profiles/registries/${registryId}/profiles/${profileId}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile),
    },
  );
  return handleResponse<QueryProfileRegistry>(res);
}

export async function deleteRegistryQueryProfile(
  registryId: string,
  profileId: string,
): Promise<QueryProfileRegistry> {
  const res = await fetch(`/v1/query-profiles/registries/${registryId}/profiles/${profileId}`, {
    method: "DELETE",
  });
  return handleResponse<QueryProfileRegistry>(res);
}

export async function getActiveQueryProfiles(): Promise<ActiveQueryProfilesResponse> {
  const res = await fetch("/v1/query-profiles");
  return handleResponse<ActiveQueryProfilesResponse>(res);
}

export async function getDefaultQueryProfileTemplate(): Promise<QueryProfileRegistryTemplate> {
  const res = await fetch("/v1/query-profiles/default-template");
  return handleResponse<QueryProfileRegistryTemplate>(res);
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
