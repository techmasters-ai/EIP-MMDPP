interface NavProps {
  page: "ingest" | "query";
  onNavigate: (page: "ingest" | "query") => void;
}

export function Nav({ page, onNavigate }: NavProps) {
  return (
    <nav className="nav">
      <div className="nav-brand">
        {/* Tec-Masters globe icon — inline SVG */}
        <svg
          width="36"
          height="36"
          viewBox="0 0 36 36"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-label="Tec-Masters Inc. logo"
        >
          {/* Globe outline */}
          <circle cx="18" cy="18" r="15" stroke="#3D3426" strokeWidth="1.5" fill="none" />
          {/* Vertical meridian (ellipse) */}
          <ellipse cx="18" cy="18" rx="7" ry="15" stroke="#3D3426" strokeWidth="1.2" fill="none" />
          {/* Horizontal parallels */}
          <ellipse cx="18" cy="10" rx="13.5" ry="2" stroke="#3D3426" strokeWidth="1" fill="none" />
          <line x1="3" y1="18" x2="33" y2="18" stroke="#3D3426" strokeWidth="1.2" />
          <ellipse cx="18" cy="26" rx="13.5" ry="2" stroke="#3D3426" strokeWidth="1" fill="none" />
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
