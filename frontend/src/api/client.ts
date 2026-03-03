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
  score: number;
  modality: string;
  content_text?: string;
  page_number?: number;
  classification: string;
  context?: Record<string, unknown>;
}

export interface QueryResponse {
  query: string;
  mode: string;
  results: QueryResultItem[];
  total_results: number;
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

export type QueryMode = "semantic" | "graph" | "hybrid" | "cross_modal" | "cognee_graph";

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

/**
 * Upload a single file with XHR so we can track upload progress.
 *
 * @param sourceId  The source to attach this document to.
 * @param file      The File object from the input / drop zone.
 * @param onProgress  Called with upload progress 0–100.
 * @returns  The created Document record.
 */
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
// Retrieval
// ---------------------------------------------------------------------------

export async function query(params: {
  query: string;
  mode: QueryMode;
  top_k?: number;
  include_context?: boolean;
}): Promise<QueryResponse> {
  const res = await fetch("/v1/retrieval/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k: 10, include_context: true, ...params }),
  });
  return handleResponse<QueryResponse>(res);
}

// ---------------------------------------------------------------------------
// LangGraph agent context (for demonstration / testing)
// ---------------------------------------------------------------------------

export async function getAgentContext(params: {
  query: string;
  mode?: QueryMode;
  top_k?: number;
}): Promise<AgentContextResponse> {
  const search = new URLSearchParams({
    query: params.query,
    mode: params.mode ?? "hybrid",
    top_k: String(params.top_k ?? 10),
  });
  const res = await fetch(`/v1/agent/context?${search}`);
  return handleResponse<AgentContextResponse>(res);
}
