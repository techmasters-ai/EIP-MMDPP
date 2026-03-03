import React, { useState } from "react";
import { DirectoryMonitor } from "./DirectoryMonitor";
import { FileUpload } from "./FileUpload";

type Tab = "upload" | "monitor";

const TABS: { value: Tab; label: string }[] = [
  { value: "upload", label: "Upload Files" },
  { value: "monitor", label: "Monitor Directory" },
];

export function IngestPage() {
  const [tab, setTab] = useState<Tab>("upload");

  return (
    <div className="page">
      <h1 className="page-title">Ingest Documents</h1>
      <p className="page-subtitle">
        Upload individual files or entire directories, or configure automatic directory
        monitoring for continuous ingest.
      </p>

      <div className="tabs">
        {TABS.map((t) => (
          <button
            key={t.value}
            className={`tab-btn${tab === t.value ? " active" : ""}`}
            onClick={() => setTab(t.value)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "upload" && <FileUpload />}
      {tab === "monitor" && <DirectoryMonitor />}
    </div>
  );
}
