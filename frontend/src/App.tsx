import { useState } from "react";
import { Nav, type Page } from "./components/Nav";
import { IngestPage } from "./components/IngestPage";
import { QueryPage } from "./components/QueryPage";
import { GraphExplorer } from "./components/GraphExplorer";
import { MemoryPanel } from "./components/MemoryPanel";

export function App() {
  const [page, setPage] = useState<Page>("ingest");

  return (
    <>
      <Nav page={page} onNavigate={setPage} />
      <main>
        {page === "ingest" && <IngestPage />}
        {page === "query" && (
          <div className="page">
            <h1 className="page-title">Search Documents</h1>
            <p className="page-subtitle">
              Query across all knowledge layers: text vectors, image vectors,
              ontology graph, cross-modal bridging, and approved trusted data.
            </p>
            <QueryPage />
          </div>
        )}
        {page === "graph" && (
          <div className="page">
            <h1 className="page-title">Graph Explorer</h1>
            <p className="page-subtitle">
              Browse and add entities and relationships in the Neo4j knowledge graph.
            </p>
            <GraphExplorer />
          </div>
        )}
        {page === "memory" && (
          <div className="page">
            <h1 className="page-title">Trusted Data</h1>
            <p className="page-subtitle">
              Propose, review, and search governed knowledge in the trusted data layer.
            </p>
            <MemoryPanel />
          </div>
        )}
      </main>
    </>
  );
}
