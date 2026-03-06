export type Page = "ingest" | "query" | "text" | "images" | "graph" | "memory";

interface NavProps {
  page: Page;
  onNavigate: (page: Page) => void;
}

const NAV_ITEMS: { value: Page; label: string }[] = [
  { value: "ingest", label: "Upload" },
  { value: "query", label: "Search" },
  { value: "text", label: "Text Store" },
  { value: "images", label: "Image Store" },
  { value: "graph", label: "Ontology" },
  { value: "memory", label: "Trusted Data" },
];

export function Nav({ page, onNavigate }: NavProps) {
  return (
    <nav className="nav">
      <div className="nav-brand">
        <svg
          className="nav-brand-icon"
          width="36"
          height="36"
          viewBox="0 0 40 40"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          aria-label="Tec-Masters Inc. logo"
        >
          <circle cx="20" cy="20" r="18" />
          <ellipse cx="20" cy="20" rx="8" ry="18" />
          <line x1="2" y1="14" x2="38" y2="14" />
          <line x1="2" y1="26" x2="38" y2="26" />
          <line x1="20" y1="2" x2="20" y2="38" />
        </svg>
        <span>
          Tec-Masters Inc.
          <span className="nav-brand-sub">EIP Multi-Modal Data Platform</span>
        </span>
      </div>

      <div className="nav-links">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.value}
            className={`nav-link${page === item.value ? " active" : ""}`}
            onClick={() => onNavigate(item.value)}
          >
            {item.label}
          </button>
        ))}
      </div>
    </nav>
  );
}
