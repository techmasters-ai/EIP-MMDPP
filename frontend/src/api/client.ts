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
  score: number;
  modality: string;
  content_text?: string;
  page_number?: number;
  classification: string;
  context?: Record<string, unknown>;
  image_url?: string;
}

export interface UnifiedQueryResponse {
  query_text?: string;
  query_image?: string;
  mode: string;
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
  mode: string;
  total_results: number;
  context: string;
  sources: AgentSource[];
}

export type QueryMode = "text_basic" | "text_only" | "images_only" | "multi_modal" | "memory";

export interface TextIngestResponse {
  chunk_ids: string[];
  chunks_created: number;
}

export interface ImageIngestResponse {
  chunk_id: string;
}

export interface GraphIngestResponse {
  status: string;
  node_id?: string;
}

export interface MemoryProposal {
  id: string;
  content: string;
  source_context?: Record<string, unknown>;
  confidence: number;
  status: string;
  proposed_by?: string;
  reviewed_by?: string;
  review_notes?: string;
  created_at: string;
}

export interface MemoryQueryResponse {
  query: string;
  results: QueryResultItem[];
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

export async function getDocumentStatus(documentId: string): Promise<Document> {
  const res = await fetch(`/v1/documents/${documentId}/status`);
  return handleResponse<Document>(res);
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
  mode: QueryMode;
  top_k?: number;
  include_context?: boolean;
}): Promise<UnifiedQueryResponse> {
  const res = await fetch("/v1/retrieval/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k: 10, include_context: true, ...params }),
  });
  return handleResponse<UnifiedQueryResponse>(res);
}

// ---------------------------------------------------------------------------
// Text Store
// ---------------------------------------------------------------------------

export async function ingestText(params: {
  text: string;
  modality?: string;
  classification?: string;
}): Promise<TextIngestResponse> {
  const res = await fetch("/v1/text/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<TextIngestResponse>(res);
}

// ---------------------------------------------------------------------------
// Image Store
// ---------------------------------------------------------------------------

export async function ingestImage(params: {
  image: string;
  alt_text?: string;
  classification?: string;
}): Promise<ImageIngestResponse> {
  const res = await fetch("/v1/images/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<ImageIngestResponse>(res);
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
// Memory
// ---------------------------------------------------------------------------

export async function proposeMemory(params: {
  content: string;
  source_context?: Record<string, unknown>;
  confidence?: number;
}): Promise<MemoryProposal> {
  const res = await fetch("/v1/memory/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<MemoryProposal>(res);
}

export async function listMemoryProposals(status?: string): Promise<MemoryProposal[]> {
  const url = status ? `/v1/memory/proposals?status=${status}` : "/v1/memory/proposals";
  const res = await fetch(url);
  return handleResponse<MemoryProposal[]>(res);
}

export async function approveMemory(id: string, notes?: string): Promise<MemoryProposal> {
  const res = await fetch(`/v1/memory/proposals/${id}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ notes }),
  });
  return handleResponse<MemoryProposal>(res);
}

export async function rejectMemory(id: string, notes?: string): Promise<MemoryProposal> {
  const res = await fetch(`/v1/memory/proposals/${id}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ notes }),
  });
  return handleResponse<MemoryProposal>(res);
}

// ---------------------------------------------------------------------------
// LangGraph agent context
// ---------------------------------------------------------------------------

export async function getAgentContext(params: {
  query: string;
  mode?: QueryMode;
  top_k?: number;
}): Promise<AgentContextResponse> {
  const search = new URLSearchParams({
    query: params.query,
    mode: params.mode ?? "text_basic",
    top_k: String(params.top_k ?? 10),
  });
  const res = await fetch(`/v1/agent/context?${search}`);
  return handleResponse<AgentContextResponse>(res);
}
