interface NavProps {
  page: "ingest" | "query";
  onNavigate: (page: "ingest" | "query") => void;
}

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
        <button
          className={`nav-link${page === "ingest" ? " active" : ""}`}
          onClick={() => onNavigate("ingest")}
        >
          Ingest
        </button>
        <button
          className={`nav-link${page === "query" ? " active" : ""}`}
          onClick={() => onNavigate("query")}
        >
          Query
        </button>
      </div>
    </nav>
  );
}
