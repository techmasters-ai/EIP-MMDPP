import { useState } from "react";
import { DirectoryMonitor } from "./DirectoryMonitor";
import { FileUpload, type FileEntry } from "./FileUpload";

type Tab = "upload" | "monitor";

const TABS: { value: Tab; label: string }[] = [
  { value: "upload", label: "Upload Files" },
  { value: "monitor", label: "Monitor Directory" },
];

interface IngestPageProps {
  entries: FileEntry[];
  setEntries: React.Dispatch<React.SetStateAction<FileEntry[]>>;
  selectedSourceId: string;
  setSelectedSourceId: React.Dispatch<React.SetStateAction<string>>;
}

export function IngestPage({ entries, setEntries, selectedSourceId, setSelectedSourceId }: IngestPageProps) {
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

      {tab === "upload" && (
        <FileUpload
          entries={entries}
          setEntries={setEntries}
          selectedSourceId={selectedSourceId}
          setSelectedSourceId={setSelectedSourceId}
        />
      )}
      {tab === "monitor" && <DirectoryMonitor />}
    </div>
  );
}
