import React, { useState } from "react";
import { Nav } from "./components/Nav";
import { IngestPage } from "./components/IngestPage";
import { QueryPage } from "./components/QueryPage";

type Page = "ingest" | "query";

export function App() {
  const [page, setPage] = useState<Page>("ingest");

  return (
    <>
      <Nav page={page} onNavigate={setPage} />
      <main>
        {page === "ingest" && <IngestPage />}
        {page === "query" && (
          <div className="page">
            <h1 className="page-title">Query Documents</h1>
            <p className="page-subtitle">
              Search ingested documents using semantic vector search, ontology graph
              traversal, or a hybrid of both.
            </p>
            <QueryPage />
          </div>
        )}
      </main>
    </>
  );
}
