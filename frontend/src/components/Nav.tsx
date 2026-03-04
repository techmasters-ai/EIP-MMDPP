interface NavProps {
  page: "ingest" | "query";
  onNavigate: (page: "ingest" | "query") => void;
}

export function Nav({ page, onNavigate }: NavProps) {
  return (
    <nav className="nav">
      <div className="nav-brand">
        {/* TecMasters wordmark — inline SVG so no asset path needed */}
        <svg
          width="32"
          height="32"
          viewBox="0 0 32 32"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-label="TecMasters logo"
        >
          <rect width="32" height="32" rx="6" fill="#7B6B52" />
          <text
            x="16"
            y="22"
            textAnchor="middle"
            fontFamily="system-ui, sans-serif"
            fontSize="16"
            fontWeight="700"
            fill="#F5F0E8"
          >
            T
          </text>
        </svg>
        <span>
          TecMasters
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
