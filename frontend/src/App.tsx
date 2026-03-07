import { useState } from "react";
import { Nav, type Page } from "./components/Nav";
import { IngestPage } from "./components/IngestPage";
import { type FileEntry } from "./components/FileUpload";
import { QueryPage } from "./components/QueryPage";
import { GraphExplorer } from "./components/GraphExplorer";
import { TrustedDataPanel } from "./components/TrustedDataPanel";

export function App() {
  const [page, setPage] = useState<Page>("ingest");

  // Lifted from FileUpload so upload state survives page navigation
  const [uploadEntries, setUploadEntries] = useState<FileEntry[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string>("");

  return (
    <>
      <Nav page={page} onNavigate={setPage} />
      <main>
        {page === "ingest" && (
          <IngestPage
            entries={uploadEntries}
            setEntries={setUploadEntries}
            selectedSourceId={selectedSourceId}
            setSelectedSourceId={setSelectedSourceId}
          />
        )}
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
            <TrustedDataPanel />
          </div>
        )}
      </main>
    </>
  );
}
