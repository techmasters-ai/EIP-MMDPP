# Graph Visualization + DoclingDocument Viewer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Two features: (1) Interactive Cytoscape.js graph overlay for Graph Explorer search results, (2) Full-page DoclingDocument viewer for completed uploads.

**Architecture:** Feature 1 adds a `/v1/graph/neighborhood` endpoint and frontend Cytoscape components. Feature 2 persists DoclingDocument JSON/markdown to MinIO during pipeline processing and serves via a new endpoint with a react-markdown frontend viewer.

**Tech Stack:** Cytoscape.js, react-cytoscapejs, react-markdown, remark-gfm, MinIO

---

## Feature 1: Graph Search Result Visualization

### Task 1: Backend — Graph neighborhood Cypher query

**Files:**
- Modify: `app/services/neo4j_graph.py`

**Step 1: Write the async neighborhood graph function**

Add `get_neighborhood_graph_async()` after the existing `get_neighborhood_async()` function (after line 488):

```python
async def get_neighborhood_graph_async(
    driver,
    entity_name: str,
    hop_count: int = 2,
    limit: int = 100,
) -> dict[str, Any]:
    """Get a node's neighborhood as separate nodes and edges for graph visualization.

    Unlike get_neighborhood_async(), this unwinds variable-length paths into
    individual edges with full properties on both endpoints.
    """
    hop_count = max(1, min(hop_count, 4))

    query = f"""
        MATCH (start:Entity {{name: $name}})
        OPTIONAL MATCH path = (start)-[*1..{hop_count}]-(neighbor:Entity)
        WITH start, relationships(path) AS rels, nodes(path) AS path_nodes
        UNWIND range(0, size(rels)-1) AS idx
        WITH start,
             path_nodes[idx] AS from_node,
             rels[idx] AS rel,
             path_nodes[idx+1] AS to_node
        RETURN DISTINCT
            properties(start) AS center_props,
            start.entity_type AS center_type,
            from_node.name AS source,
            from_node.entity_type AS source_type,
            properties(from_node) AS source_props,
            type(rel) AS rel_type,
            properties(rel) AS rel_props,
            to_node.name AS target,
            to_node.entity_type AS target_type,
            properties(to_node) AS target_props
        LIMIT $limit
    """

    center: dict[str, Any] | None = None
    nodes_map: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    try:
        async with driver.session() as session:
            result = await session.run(query, name=entity_name, limit=limit)
            records = await result.data()

            for r in records:
                # Capture center node from first record
                if center is None and r.get("center_props"):
                    center = dict(r["center_props"])
                    center["entity_type"] = r["center_type"]
                    nodes_map[entity_name] = center

                source_name = r.get("source")
                target_name = r.get("target")
                if not source_name or not target_name:
                    continue

                # Collect unique nodes
                if source_name not in nodes_map and r.get("source_props"):
                    node = dict(r["source_props"])
                    node["entity_type"] = r["source_type"]
                    nodes_map[source_name] = node
                if target_name not in nodes_map and r.get("target_props"):
                    node = dict(r["target_props"])
                    node["entity_type"] = r["target_type"]
                    nodes_map[target_name] = node

                # Collect edges
                edge: dict[str, Any] = {
                    "source": source_name,
                    "target": target_name,
                    "rel_type": r.get("rel_type", "UNKNOWN"),
                }
                if r.get("rel_props"):
                    edge.update(r["rel_props"])
                edges.append(edge)

        # If no paths found but entity exists, return just the center node
        if center is None:
            # Try to find the entity directly
            q2 = """
                MATCH (n:Entity {name: $name})
                RETURN properties(n) AS props, n.entity_type AS entity_type
                LIMIT 1
            """
            async with driver.session() as session:
                result = await session.run(q2, name=entity_name)
                records = await result.data()
                if records:
                    center = dict(records[0]["props"])
                    center["entity_type"] = records[0]["entity_type"]

    except Exception as e:
        logger.warning("get_neighborhood_graph_async failed for '%s': %s", entity_name, e)

    return {
        "center": center,
        "nodes": list(nodes_map.values()),
        "edges": edges,
    }
```

**Step 2: Run tests to verify no regressions**

Run: `cd /home/josh/development/EIP-MMDPP && python -c "from app.services.neo4j_graph import get_neighborhood_graph_async; print('import ok')"`
Expected: PASS (import succeeds)

**Step 3: Commit**

```bash
git add app/services/neo4j_graph.py
git commit -m "feat: add get_neighborhood_graph_async for graph visualization"
```

---

### Task 2: Backend — Neighborhood schemas and endpoint

**Files:**
- Modify: `app/schemas/graph_store.py`
- Modify: `app/api/v1/graph_store.py`

**Step 1: Add schemas to `app/schemas/graph_store.py`**

Add after the existing `GraphQueryRequest` class (after line 37):

```python
class GraphNeighborhoodRequest(APIModel):
    entity_name: str = Field(..., min_length=1, max_length=4096)
    hop_count: int = Field(default=2, ge=1, le=4)


class GraphNeighborhoodNode(APIModel):
    name: str
    entity_type: str
    properties: dict[str, Any] = {}


class GraphNeighborhoodEdge(APIModel):
    source: str
    target: str
    rel_type: str
    properties: dict[str, Any] = {}


class GraphNeighborhoodResponse(APIModel):
    center: Optional[dict[str, Any]] = None
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
```

**Step 2: Add endpoint to `app/api/v1/graph_store.py`**

Add after the existing `query_graph` function (after line 129):

```python
@router.post("/graph/neighborhood", response_model=GraphNeighborhoodResponse)
async def get_neighborhood(
    body: GraphNeighborhoodRequest,
) -> GraphNeighborhoodResponse:
    """Get an entity's full neighborhood graph for visualization."""
    from app.services.neo4j_graph import get_neighborhood_graph_async

    driver = get_neo4j_async_driver()
    result = await get_neighborhood_graph_async(
        driver, body.entity_name, hop_count=body.hop_count
    )

    return GraphNeighborhoodResponse(
        center=result["center"],
        nodes=result["nodes"],
        edges=result["edges"],
    )
```

Update the imports at the top of `graph_store.py` to include the new schemas:

```python
from app.schemas.graph_store import (
    GraphEntityIngest,
    GraphIngestResponse,
    GraphNeighborhoodRequest,
    GraphNeighborhoodResponse,
    GraphQueryRequest,
    GraphRelationshipIngest,
)
```

**Step 3: Verify imports**

Run: `cd /home/josh/development/EIP-MMDPP && python -c "from app.api.v1.graph_store import router; print('import ok')"`
Expected: PASS

**Step 4: Commit**

```bash
git add app/schemas/graph_store.py app/api/v1/graph_store.py
git commit -m "feat: add POST /v1/graph/neighborhood endpoint for graph visualization"
```

---

### Task 3: Frontend — Install npm dependencies for Feature 1

**Files:**
- Modify: `frontend/package.json`

**Step 1: Install cytoscape and react-cytoscapejs**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npm install cytoscape react-cytoscapejs && npm install -D @types/cytoscape`

**Step 2: Verify install**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npm ls cytoscape react-cytoscapejs`
Expected: Shows both packages installed

**Step 3: Commit**

```bash
git add frontend/package.json frontend/package-lock.json
git commit -m "chore: add cytoscape and react-cytoscapejs dependencies"
```

---

### Task 4: Frontend — API client function for neighborhood

**Files:**
- Modify: `frontend/src/api/client.ts`

**Step 1: Add types and function**

Add after the existing `queryGraph` function (after line 313):

```typescript
export interface GraphNeighborhoodResponse {
  center: Record<string, unknown> | null;
  nodes: Record<string, unknown>[];
  edges: Array<{
    source: string;
    target: string;
    rel_type: string;
    [key: string]: unknown;
  }>;
}

export async function getGraphNeighborhood(params: {
  entity_name: string;
  hop_count?: number;
}): Promise<GraphNeighborhoodResponse> {
  const res = await fetch("/v1/graph/neighborhood", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      entity_name: params.entity_name,
      hop_count: params.hop_count ?? 2,
    }),
  });
  return handleResponse<GraphNeighborhoodResponse>(res);
}
```

**Step 2: Verify build**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npx tsc --noEmit`
Expected: PASS (no type errors)

**Step 3: Commit**

```bash
git add frontend/src/api/client.ts
git commit -m "feat: add getGraphNeighborhood API client function"
```

---

### Task 5: Frontend — GraphTooltip component

**Files:**
- Create: `frontend/src/components/GraphTooltip.tsx`

**Step 1: Create the component**

```typescript
import React from "react";

interface GraphTooltipProps {
  visible: boolean;
  x: number;
  y: number;
  data: Record<string, unknown>;
  containerRect?: DOMRect;
}

/** Keys to exclude from tooltip display. */
const HIDDEN_KEYS = new Set(["id", "label", "classes"]);

export function GraphTooltip({ visible, x, y, data, containerRect }: GraphTooltipProps) {
  if (!visible || !data || Object.keys(data).length === 0) return null;

  const entries = Object.entries(data).filter(
    ([key, value]) =>
      !HIDDEN_KEYS.has(key) &&
      value !== null &&
      value !== undefined &&
      value !== "",
  );

  if (entries.length === 0) return null;

  // Clamp position to stay within container
  let left = x + 12;
  let top = y + 12;
  if (containerRect) {
    const tooltipWidth = 280;
    const tooltipHeight = entries.length * 28 + 24;
    if (left + tooltipWidth > containerRect.right) left = x - tooltipWidth - 12;
    if (top + tooltipHeight > containerRect.bottom) top = y - tooltipHeight - 12;
    if (left < containerRect.left) left = containerRect.left + 4;
    if (top < containerRect.top) top = containerRect.top + 4;
  }

  return (
    <div
      className="graph-tooltip"
      style={{ left, top }}
    >
      <table>
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key}>
              <td className="graph-tooltip-key">{key}</td>
              <td className="graph-tooltip-value">
                {typeof value === "object" ? JSON.stringify(value) : String(value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

**Step 2: Verify build**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npx tsc --noEmit`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/components/GraphTooltip.tsx
git commit -m "feat: add GraphTooltip component for graph visualization hover"
```

---

### Task 6: Frontend — GraphView component

**Files:**
- Create: `frontend/src/components/GraphView.tsx`

**Step 1: Create the component**

```typescript
import React, { useRef, useState, useCallback } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import type cytoscape from "cytoscape";
import { GraphTooltip } from "./GraphTooltip";
import type { GraphNeighborhoodResponse } from "../api/client";

// Entity type → CSS category class
const ENTITY_CATEGORY: Record<string, string> = {};
const MILITARY = [
  "RadarSystem", "MissileSystem", "AirDefenseArtillerySystem",
  "IntegratedAirDefenseSystem", "ElectronicWarfareSystem", "FireControlSystem",
  "LauncherSystem", "WeaponSystem", "Platform", "Subsystem", "Component",
];
const EMRF = [
  "FrequencyBand", "Waveform", "Modulation", "RFEmission", "RFSignature",
  "Antenna", "Transmitter", "Receiver", "SignalProcessingChain", "ScanPattern",
];
const WEAPON = ["Seeker", "GuidanceMethod", "MissilePerformance", "PropulsionStack"];
const OPERATIONAL = ["Capability", "EngagementTimeline", "RadarPerformance"];
const REFERENCE = ["Organization", "Document", "Assertion"];

MILITARY.forEach((t) => (ENTITY_CATEGORY[t] = "military"));
EMRF.forEach((t) => (ENTITY_CATEGORY[t] = "emrf"));
WEAPON.forEach((t) => (ENTITY_CATEGORY[t] = "weapon"));
OPERATIONAL.forEach((t) => (ENTITY_CATEGORY[t] = "operational"));
REFERENCE.forEach((t) => (ENTITY_CATEGORY[t] = "reference"));

function getCategory(entityType: string): string {
  return ENTITY_CATEGORY[entityType] || "reference";
}

/** Convert API response to Cytoscape elements. */
export function toGraphElements(
  response: GraphNeighborhoodResponse,
  centerName: string,
): cytoscape.ElementDefinition[] {
  const elements: cytoscape.ElementDefinition[] = [];
  const nodeIds = new Set<string>();

  // Collect all nodes
  for (const node of response.nodes) {
    const name = node.name as string;
    if (!name || nodeIds.has(name)) continue;
    nodeIds.add(name);
    const entityType = (node.entity_type as string) || "UNKNOWN";
    const label = name.length > 20 ? name.slice(0, 18) + "…" : name;
    elements.push({
      data: { id: name, label, ...node },
      classes: `${getCategory(entityType)}${name === centerName ? " center" : ""}`,
    });
  }

  // Collect edges (dedup by source->target::rel_type)
  const edgeIds = new Set<string>();
  for (const edge of response.edges) {
    const edgeId = `${edge.source}->${edge.target}::${edge.rel_type}`;
    if (edgeIds.has(edgeId)) continue;
    edgeIds.add(edgeId);
    // Only add edge if both endpoints exist
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) continue;
    const { source, target, rel_type, ...rest } = edge;
    elements.push({
      data: {
        id: edgeId,
        source,
        target,
        label: rel_type,
        rel_type,
        ...rest,
      },
    });
  }

  return elements;
}

const LAYOUT = {
  name: "cose",
  animate: false,
  nodeDimensionsIncludeLabels: true,
  nodeRepulsion: () => 8000,
  idealEdgeLength: () => 120,
  edgeElasticity: () => 100,
  gravity: 0.25,
  padding: 40,
};

const STYLESHEET: cytoscape.Stylesheet[] = [
  {
    selector: "node",
    style: {
      label: "data(label)",
      "text-valign": "bottom",
      "text-halign": "center",
      "font-size": "11px",
      color: "#1C1C1C",
      "text-margin-y": 6,
      width: 32,
      height: 32,
      "border-width": 2,
      "border-color": "#DDD5C8",
      "background-color": "#6B6560",
    },
  },
  {
    selector: "node.military",
    style: { "background-color": "#7B6B52" },
  },
  {
    selector: "node.emrf",
    style: { "background-color": "#2A5A8A" },
  },
  {
    selector: "node.weapon",
    style: { "background-color": "#B94040" },
  },
  {
    selector: "node.operational",
    style: { "background-color": "#7A6020" },
  },
  {
    selector: "node.reference",
    style: { "background-color": "#6B6560" },
  },
  {
    selector: "node.center",
    style: {
      "border-width": 3,
      "border-color": "#fff",
      width: 40,
      height: 40,
    },
  },
  {
    selector: "edge",
    style: {
      label: "data(label)",
      "font-size": "9px",
      color: "#6B6560",
      "text-rotation": "autorotate",
      "text-margin-y": -8,
      "curve-style": "bezier",
      "target-arrow-shape": "triangle",
      "target-arrow-color": "#DDD5C8",
      "line-color": "#DDD5C8",
      width: 1.5,
    },
  },
];

interface GraphViewProps {
  elements: cytoscape.ElementDefinition[];
  onNodeClick: (entityName: string) => void;
  onClose: () => void;
}

export function GraphView({ elements, onNodeClick, onClose }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    data: Record<string, unknown>;
  }>({ visible: false, x: 0, y: 0, data: {} });

  const handleCy = useCallback(
    (cy: cytoscape.Core) => {
      cyRef.current = cy;

      cy.on("mouseover", "node, edge", (evt) => {
        const el = evt.target;
        const data = { ...el.data() };
        // Remove Cytoscape internals
        delete data.id;
        delete data.source;
        delete data.target;
        const pos = evt.renderedPosition || evt.position;
        const rect = containerRef.current?.getBoundingClientRect();
        setTooltip({
          visible: true,
          x: (rect?.left || 0) + (pos?.x || 0),
          y: (rect?.top || 0) + (pos?.y || 0),
          data,
        });
      });

      cy.on("mouseout", "node, edge", () => {
        setTooltip((prev) => ({ ...prev, visible: false }));
      });

      cy.on("tap", "node", (evt) => {
        const name = evt.target.data("id");
        if (name) onNodeClick(name);
      });

      cy.on("tap", (evt) => {
        if (evt.target === cy) {
          setTooltip((prev) => ({ ...prev, visible: false }));
        }
      });
    },
    [onNodeClick],
  );

  return (
    <div className="graph-view-container" ref={containerRef}>
      <button className="graph-view-close btn btn-ghost btn-sm" onClick={onClose}>
        ✕
      </button>
      <CytoscapeComponent
        elements={elements}
        layout={LAYOUT}
        stylesheet={STYLESHEET}
        style={{ width: "100%", height: "100%" }}
        cy={handleCy}
      />
      <GraphTooltip
        visible={tooltip.visible}
        x={tooltip.x}
        y={tooltip.y}
        data={tooltip.data}
        containerRect={containerRef.current?.getBoundingClientRect()}
      />
    </div>
  );
}
```

**Step 2: Verify build**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npx tsc --noEmit`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/components/GraphView.tsx
git commit -m "feat: add GraphView component with Cytoscape rendering and interactions"
```

---

### Task 7: Frontend — Integrate GraphView into GraphExplorer

**Files:**
- Modify: `frontend/src/components/GraphExplorer.tsx`

**Step 1: Update the GraphSearch component**

Add imports at the top:

```typescript
import { ingestGraphEntity, ingestGraphRelationship, queryGraph, getGraphNeighborhood, type QueryResultItem, type GraphNeighborhoodResponse } from "../api/client";
import { GraphView, toGraphElements } from "./GraphView";
```

In the `GraphSearch` function, add state for graph view:

```typescript
const [graphViewIndex, setGraphViewIndex] = useState<number | null>(null);
const [graphElements, setGraphElements] = useState<cytoscape.ElementDefinition[] | null>(null);
const [graphLoading, setGraphLoading] = useState(false);
```

Add a handler for toggling graph view:

```typescript
const handleToggleGraph = async (index: number, entityName: string) => {
  if (graphViewIndex === index) {
    setGraphViewIndex(null);
    setGraphElements(null);
    return;
  }
  setGraphViewIndex(index);
  setGraphLoading(true);
  try {
    const resp = await getGraphNeighborhood({ entity_name: entityName });
    setGraphElements(toGraphElements(resp, entityName));
  } catch {
    setGraphElements(null);
  } finally {
    setGraphLoading(false);
  }
};
```

Add a handler for node click (triggers new search):

```typescript
const handleNodeClick = async (entityName: string) => {
  setGraphViewIndex(null);
  setGraphElements(null);
  setQuery(entityName);
  setLoading(true);
  setError(null);
  try {
    const res = await queryGraph({ query: entityName, top_k: 20 });
    setResults(res.results);
  } catch (err) {
    setError(err instanceof Error ? err.message : "Search failed");
  } finally {
    setLoading(false);
  }
};
```

Modify the result card rendering (lines 96–109) to include the toggle button and graph overlay:

```typescript
results.map((item, i) => (
  <div key={i} className="result-card">
    <div className="result-card-header">
      <span className="text-xs text-muted">#{i + 1}</span>
      <span className="badge badge-info">{item.modality}</span>
      {item.content_text && (
        <button
          className={`btn btn-ghost btn-sm graph-toggle-btn${graphViewIndex === i ? " active" : ""}`}
          onClick={() => void handleToggleGraph(i, item.content_text!)}
          title="Toggle graph view"
        >
          ◉
        </button>
      )}
    </div>
    {graphViewIndex === i ? (
      graphLoading ? (
        <div className="graph-view-container" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span className="spinner" />
        </div>
      ) : graphElements ? (
        <GraphView
          elements={graphElements}
          onNodeClick={(name) => void handleNodeClick(name)}
          onClose={() => { setGraphViewIndex(null); setGraphElements(null); }}
        />
      ) : (
        <p className="text-muted text-sm" style={{ padding: "1rem" }}>
          Could not load graph data.
        </p>
      )
    ) : (
      <>
        {item.content_text && <p>{item.content_text}</p>}
        {item.context && (
          <pre className="text-xs" style={{ whiteSpace: "pre-wrap", margin: "0.5rem 0" }}>
            {JSON.stringify(item.context, null, 2)}
          </pre>
        )}
      </>
    )}
  </div>
))
```

**Step 2: Verify build**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npx tsc --noEmit`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/components/GraphExplorer.tsx
git commit -m "feat: integrate GraphView toggle into Graph Explorer search results"
```

---

### Task 8: Frontend — CSS styles for graph visualization

**Files:**
- Modify: `frontend/src/styles.css`

**Step 1: Add graph visualization styles**

Append to the end of `styles.css`:

```css
/* ---- Graph visualization ---- */
.graph-view-container {
  position: relative;
  width: 100%;
  height: 400px;
  background: var(--color-surface-2);
  border-radius: var(--radius);
  border: 1px solid var(--color-border);
  margin-top: 0.5rem;
  overflow: hidden;
}

.graph-view-close {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  z-index: 10;
}

.graph-toggle-btn {
  margin-left: auto;
  padding: 0.2rem 0.5rem;
  font-size: 0.85rem;
}

.graph-toggle-btn.active {
  color: var(--color-primary);
  background: var(--color-primary-l);
}

.graph-tooltip {
  position: fixed;
  z-index: 200;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-md);
  padding: 0.5rem 0.75rem;
  max-width: 280px;
  pointer-events: none;
  font-size: 0.775rem;
}

.graph-tooltip table {
  border-collapse: collapse;
  width: 100%;
}

.graph-tooltip-key {
  font-weight: 600;
  color: var(--color-text-muted);
  padding: 0.1rem 0.5rem 0.1rem 0;
  white-space: nowrap;
  vertical-align: top;
}

.graph-tooltip-value {
  color: var(--color-text);
  padding: 0.1rem 0;
  word-break: break-word;
}
```

**Step 2: Verify build**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npm run build`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/styles.css
git commit -m "feat: add CSS styles for graph visualization and tooltip"
```

---

### Task 9: Run tests and verify Feature 1

**Step 1: Run the test suite**

Run: `cd /home/josh/development/EIP-MMDPP && ./scripts/run_tests.sh`
Expected: All existing tests pass, no regressions

**Step 2: Verify frontend builds**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npm run build`
Expected: PASS

**Step 3: Update README if needed**

Per project convention, update README.md with any relevant changes.

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test: verify Feature 1 graph visualization"
```

---

## Feature 2: DoclingDocument Viewer

### Task 10: Docling service — Add document_json to response

**Files:**
- Modify: `docker/docling/app/schemas.py`
- Modify: `docker/docling/app/converter.py`

**Step 1: Add `document_json` field to `ConvertResponse`**

In `docker/docling/app/schemas.py`, add to the `ConvertResponse` class (after line 33):

```python
    document_json: dict | None = None  # Full DoclingDocument dict
```

**Step 2: Populate `document_json` in converter**

In `docker/docling/app/converter.py`, modify the `convert_document` function (around line 107-119). After `markdown = doc.export_to_markdown()` (line 108), add:

```python
        document_json = doc.export_to_dict()
```

Then update the `ConvertResponse` constructor to include it:

```python
        return ConvertResponse(
            status="ok",
            filename=filename,
            num_pages=num_pages,
            elements=elements,
            markdown=markdown,
            document_json=document_json,
            processing_time_ms=round(elapsed_ms, 1),
        )
```

**Step 3: Verify import**

Run: `cd /home/josh/development/EIP-MMDPP && python -c "from docker.docling.app.schemas import ConvertResponse; print(ConvertResponse.model_fields.keys())"`
Expected: Shows `document_json` in the fields

**Step 4: Commit**

```bash
git add docker/docling/app/schemas.py docker/docling/app/converter.py
git commit -m "feat: include document_json in Docling ConvertResponse"
```

---

### Task 11: Pipeline client — Propagate document_json

**Files:**
- Modify: `app/services/docling_client.py`

**Step 1: Add `document_json` to `DoclingConversionResult`**

In `app/services/docling_client.py`, update the dataclass (line 32-38):

```python
@dataclass
class DoclingConversionResult:
    """Result from the Docling service."""

    elements: list[ExtractedChunk]
    markdown: str
    num_pages: int
    processing_time_ms: float
    document_json: dict | None = None
```

**Step 2: Populate it in `convert_document_sync`**

In the `convert_document_sync` function (around line 71-76), update the return:

```python
    return DoclingConversionResult(
        elements=chunks,
        markdown=data.get("markdown", ""),
        num_pages=data.get("num_pages", 0),
        processing_time_ms=data.get("processing_time_ms", 0),
        document_json=data.get("document_json"),
    )
```

**Step 3: Verify import**

Run: `cd /home/josh/development/EIP-MMDPP && python -c "from app.services.docling_client import DoclingConversionResult; print('ok')"`
Expected: PASS

**Step 4: Commit**

```bash
git add app/services/docling_client.py
git commit -m "feat: propagate document_json through DoclingConversionResult"
```

---

### Task 12: Pipeline — Persist DoclingDocument to MinIO

**Files:**
- Modify: `app/workers/pipeline.py`

**Step 1: Add MinIO uploads after element extraction**

In `prepare_document`, after the stale element cleanup and before `_update_stage_run` (around line 757, before the metrics update), add:

```python
        # Persist DoclingDocument markdown and JSON to MinIO for the viewer
        try:
            from app.services.storage import upload_bytes_sync
            _docling_base = f"artifacts/{document_id}"
            if result.markdown:
                upload_bytes_sync(
                    result.markdown.encode("utf-8"),
                    settings.minio_bucket_derived,
                    f"{_docling_base}/docling_document.md",
                    content_type="text/markdown; charset=utf-8",
                )
            if getattr(result, "document_json", None):
                import json as _json
                upload_bytes_sync(
                    _json.dumps(result.document_json, ensure_ascii=False, default=str).encode("utf-8"),
                    settings.minio_bucket_derived,
                    f"{_docling_base}/docling_document.json",
                    content_type="application/json; charset=utf-8",
                )
                logger.info("prepare_document: persisted DoclingDocument md+json for %s", document_id)
        except Exception as _doc_err:
            logger.warning("prepare_document: failed to persist DoclingDocument for %s: %s", document_id, _doc_err)
```

**Step 2: Verify import**

Run: `cd /home/josh/development/EIP-MMDPP && python -c "from app.workers.pipeline import prepare_document; print('ok')"`
Expected: PASS

**Step 3: Commit**

```bash
git add app/workers/pipeline.py
git commit -m "feat: persist DoclingDocument markdown and JSON to MinIO during prepare_document"
```

---

### Task 13: Backend — DoclingDocument API endpoint

**Files:**
- Modify: `app/schemas/retrieval.py` (or create separate schema)
- Modify: `app/api/v1/sources.py`

**Step 1: Add response schema**

In `app/schemas/retrieval.py`, add at the end:

```python
class DoclingImageRef(APIModel):
    element_uid: str
    url: str


class DoclingDocumentResponse(APIModel):
    document_id: str
    filename: str
    markdown: str
    document_json: dict[str, Any]
    images: list[DoclingImageRef] = []
```

**Step 2: Add endpoint to `app/api/v1/sources.py`**

Add the endpoint (at the end of the file, before any utility functions):

```python
@router.get("/documents/{document_id}/docling", response_model=DoclingDocumentResponse)
async def get_docling_document(
    document_id: str,
    db: AsyncSession = Depends(get_async_db),
):
    """Retrieve the persisted DoclingDocument (markdown + JSON) for a processed document."""
    import json

    from app.services.storage import download_bytes_async
    from app.config import get_settings

    settings = get_settings()
    doc_uuid = uuid.UUID(document_id)

    # Verify document exists
    doc = await db.get(Document, doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    base_key = f"artifacts/{document_id}"
    bucket = settings.minio_bucket_derived

    # Fetch markdown and JSON from MinIO
    try:
        md_bytes = await download_bytes_async(bucket, f"{base_key}/docling_document.md")
        json_bytes = await download_bytes_async(bucket, f"{base_key}/docling_document.json")
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="DoclingDocument not available for this document. Re-ingest to generate.",
        )

    markdown_text = md_bytes.decode("utf-8")
    document_json = json.loads(json_bytes.decode("utf-8"))

    # Build image URL list from artifacts
    from sqlalchemy import select
    from app.models.ingest import Artifact

    stmt = select(Artifact).where(
        Artifact.document_id == doc_uuid,
        Artifact.artifact_type.in_(["image", "schematic"]),
        Artifact.storage_key.isnot(None),
    )
    result = await db.execute(stmt)
    artifacts = result.scalars().all()

    images = []
    for art in artifacts:
        # Use the API proxy pattern (same as retrieval image endpoint)
        url = f"/v1/documents/{document_id}/artifacts/{art.id}/image"
        # Find matching element_uid from DocumentElement
        from app.models.ingest import DocumentElement
        elem_stmt = select(DocumentElement.element_uid).where(
            DocumentElement.artifact_id == art.id
        )
        elem_result = await db.execute(elem_stmt)
        elem_uid = elem_result.scalar_one_or_none()
        if elem_uid:
            images.append({"element_uid": elem_uid, "url": url})

    # Replace image references in markdown with proxy URLs
    # Docling markdown images look like: ![caption](image_path)
    # We replace with our API proxy URLs keyed by element order
    for img in images:
        # Simple approach: append images at matching positions
        pass  # Images will be mapped by the frontend using the images array

    return DoclingDocumentResponse(
        document_id=document_id,
        filename=doc.filename or "",
        markdown=markdown_text,
        document_json=document_json,
        images=images,
    )
```

Also add an image proxy endpoint for serving artifact images:

```python
@router.get("/documents/{document_id}/artifacts/{artifact_id}/image")
async def get_artifact_image(
    document_id: str,
    artifact_id: str,
    db: AsyncSession = Depends(get_async_db),
):
    """Stream an artifact image from MinIO."""
    from app.services.storage import download_bytes_async
    from fastapi.responses import Response

    doc_uuid = uuid.UUID(document_id)
    art_uuid = uuid.UUID(artifact_id)

    art = await db.get(Artifact, art_uuid)
    if not art or str(art.document_id) != document_id or not art.storage_key:
        raise HTTPException(status_code=404, detail="Artifact not found")

    image_bytes = await download_bytes_async(art.storage_bucket, art.storage_key)
    # Determine content type from storage key extension
    ext = art.storage_key.rsplit(".", 1)[-1].lower() if "." in art.storage_key else "png"
    content_type = {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "tiff": "image/tiff", "tif": "image/tiff", "gif": "image/gif",
        "bmp": "image/bmp",
    }.get(ext, "image/png")

    return Response(content=image_bytes, media_type=content_type)
```

Ensure the required imports are at the top of `sources.py`:

```python
import uuid
from app.models.ingest import Document, Artifact
```

**Step 3: Check if `download_bytes_async` exists**

If `download_bytes_async` doesn't exist in `app/services/storage.py`, add it:

```python
async def download_bytes_async(bucket: str, key: str) -> bytes:
    """Download bytes asynchronously (FastAPI context)."""
    async with get_async_s3_client() as client:
        response = await client.get_object(Bucket=bucket, Key=key)
        async with response["Body"] as stream:
            return await stream.read()
```

**Step 4: Verify import**

Run: `cd /home/josh/development/EIP-MMDPP && python -c "from app.api.v1.sources import router; print('ok')"`
Expected: PASS

**Step 5: Commit**

```bash
git add app/schemas/retrieval.py app/api/v1/sources.py app/services/storage.py
git commit -m "feat: add GET /v1/documents/{document_id}/docling endpoint and artifact image proxy"
```

---

### Task 14: Frontend — Install npm dependencies for Feature 2

**Files:**
- Modify: `frontend/package.json`

**Step 1: Install react-markdown and remark-gfm**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npm install react-markdown remark-gfm`

**Step 2: Verify install**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npm ls react-markdown remark-gfm`
Expected: Shows both packages installed

**Step 3: Commit**

```bash
git add frontend/package.json frontend/package-lock.json
git commit -m "chore: add react-markdown and remark-gfm dependencies"
```

---

### Task 15: Frontend — API client function for DoclingDocument

**Files:**
- Modify: `frontend/src/api/client.ts`

**Step 1: Add types and function**

Add after the graph neighborhood function:

```typescript
export interface DoclingImageRef {
  element_uid: string;
  url: string;
}

export interface DoclingDocumentResponse {
  document_id: string;
  filename: string;
  markdown: string;
  document_json: Record<string, unknown>;
  images: DoclingImageRef[];
}

export async function getDoclingDocument(documentId: string): Promise<DoclingDocumentResponse> {
  const res = await fetch(`/v1/documents/${documentId}/docling`);
  return handleResponse<DoclingDocumentResponse>(res);
}
```

**Step 2: Verify build**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npx tsc --noEmit`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/api/client.ts
git commit -m "feat: add getDoclingDocument API client function"
```

---

### Task 16: Frontend — DoclingViewer component

**Files:**
- Create: `frontend/src/components/DoclingViewer.tsx`

**Step 1: Create the component**

```typescript
import React, { useEffect, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getDoclingDocument, type DoclingDocumentResponse } from "../api/client";

interface DoclingViewerProps {
  documentId: string;
  filename: string;
  onClose: () => void;
}

type ViewMode = "markdown" | "json";

export function DoclingViewer({ documentId, filename, onClose }: DoclingViewerProps) {
  const [data, setData] = useState<DoclingDocumentResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("markdown");

  useEffect(() => {
    setLoading(true);
    setError(null);
    getDoclingDocument(documentId)
      .then(setData)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Failed to load document"),
      )
      .finally(() => setLoading(false));
  }, [documentId]);

  // Close on Escape key
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="docling-overlay" onClick={onClose}>
      <div className="docling-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="docling-modal-header">
          <h3 className="docling-modal-title" title={filename}>
            {filename}
          </h3>
          <div className="docling-mode-toggle">
            <button
              className={`mode-btn${mode === "markdown" ? " active" : ""}`}
              onClick={() => setMode("markdown")}
            >
              Document
            </button>
            <button
              className={`mode-btn${mode === "json" ? " active" : ""}`}
              onClick={() => setMode("json")}
            >
              JSON
            </button>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>
            ✕
          </button>
        </div>

        {/* Body */}
        <div className="docling-modal-body">
          {loading && (
            <div className="empty-state">
              <span className="spinner" />
              <p className="mt-sm">Loading document...</p>
            </div>
          )}

          {error && (
            <div className="alert alert-error">{error}</div>
          )}

          {data && mode === "markdown" && (
            <div className="docling-markdown-content">
              <Markdown
                remarkPlugins={[remarkGfm]}
                components={{
                  img: ({ src, alt, ...props }) => {
                    // Map Docling image references to API proxy URLs
                    const matchedImage = data.images.find(
                      (img) => src?.includes(img.element_uid),
                    );
                    const resolvedSrc = matchedImage ? matchedImage.url : src;
                    return (
                      <img
                        src={resolvedSrc}
                        alt={alt || ""}
                        className="docling-inline-image"
                        {...props}
                      />
                    );
                  },
                }}
              >
                {data.markdown}
              </Markdown>
              {/* Render any images not embedded in markdown */}
              {data.images.length > 0 && (
                <div className="docling-image-gallery">
                  {data.images.map((img) => (
                    <img
                      key={img.element_uid}
                      src={img.url}
                      alt={img.element_uid}
                      className="docling-inline-image"
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {data && mode === "json" && (
            <pre className="docling-json-content">
              {JSON.stringify(data.document_json, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Verify build**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npx tsc --noEmit`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/components/DoclingViewer.tsx
git commit -m "feat: add DoclingViewer modal component with markdown/JSON toggle"
```

---

### Task 17: Frontend — Integrate DoclingViewer into FileUpload

**Files:**
- Modify: `frontend/src/components/FileUpload.tsx`

**Step 1: Add import and state**

Add import at the top:

```typescript
import { DoclingViewer } from "./DoclingViewer";
```

Inside the `FileUpload` component, add state:

```typescript
const [viewingDoc, setViewingDoc] = useState<{ id: string; filename: string } | null>(null);
```

**Step 2: Add "View" button to upload entries**

In the entries list (around line 353, after the `StatusBadge`), add the View button for COMPLETE entries:

```typescript
{entry.status === "COMPLETE" && entry.documentId && (
  <button
    className="btn btn-ghost btn-sm"
    onClick={() => setViewingDoc({ id: entry.documentId!, filename: entry.fileName })}
    title="View document"
  >
    View
  </button>
)}
```

**Step 3: Add "View" button to existing source documents**

In the existing docs list (around line 427, after the `StatusBadge`), add:

```typescript
{doc.pipeline_status === "COMPLETE" && (
  <button
    className="btn btn-ghost btn-sm"
    onClick={() => setViewingDoc({ id: doc.id, filename: doc.filename })}
    title="View document"
  >
    View
  </button>
)}
```

**Step 4: Render the modal**

At the end of the component's return, just before the closing `</div>`, add:

```typescript
{viewingDoc && (
  <DoclingViewer
    documentId={viewingDoc.id}
    filename={viewingDoc.filename}
    onClose={() => setViewingDoc(null)}
  />
)}
```

**Step 5: Verify build**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npx tsc --noEmit`
Expected: PASS

**Step 6: Commit**

```bash
git add frontend/src/components/FileUpload.tsx
git commit -m "feat: add View button to completed documents opening DoclingViewer modal"
```

---

### Task 18: Frontend — CSS styles for DoclingViewer

**Files:**
- Modify: `frontend/src/styles.css`

**Step 1: Add DoclingViewer styles**

Append to the end of `styles.css`:

```css
/* ---- DoclingViewer modal ---- */
.docling-overlay {
  position: fixed;
  inset: 0;
  z-index: 500;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
}

.docling-modal {
  background: var(--color-surface);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  width: 90vw;
  height: 90vh;
  max-width: 1200px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.docling-modal-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.docling-modal-title {
  font-size: 1rem;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.docling-mode-toggle {
  display: flex;
  gap: 0.25rem;
}

.docling-modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem;
}

.docling-markdown-content {
  max-width: 800px;
  margin: 0 auto;
  line-height: 1.7;
}

.docling-markdown-content h1 { font-size: 1.75rem; font-weight: 700; margin: 1.5rem 0 0.75rem; }
.docling-markdown-content h2 { font-size: 1.4rem; font-weight: 700; margin: 1.25rem 0 0.5rem; }
.docling-markdown-content h3 { font-size: 1.15rem; font-weight: 600; margin: 1rem 0 0.5rem; }
.docling-markdown-content h4 { font-size: 1rem; font-weight: 600; margin: 0.75rem 0 0.35rem; }
.docling-markdown-content p { margin: 0.5rem 0; }

.docling-markdown-content table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  font-size: 0.875rem;
}

.docling-markdown-content th,
.docling-markdown-content td {
  border: 1px solid var(--color-border);
  padding: 0.5rem 0.75rem;
  text-align: left;
}

.docling-markdown-content th {
  background: var(--color-surface-2);
  font-weight: 600;
}

.docling-inline-image {
  max-width: 100%;
  height: auto;
  border-radius: var(--radius);
  margin: 0.75rem 0;
  border: 1px solid var(--color-border);
}

.docling-image-gallery {
  margin-top: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.docling-json-content {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--color-surface-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: 1rem;
  margin: 0;
  overflow-x: auto;
}
```

**Step 2: Verify build**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npm run build`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/styles.css
git commit -m "feat: add CSS styles for DoclingViewer modal and markdown rendering"
```

---

### Task 19: Run tests and verify Feature 2

**Step 1: Run the test suite**

Run: `cd /home/josh/development/EIP-MMDPP && ./scripts/run_tests.sh`
Expected: All existing tests pass, no regressions

**Step 2: Verify frontend builds**

Run: `cd /home/josh/development/EIP-MMDPP/frontend && npm run build`
Expected: PASS

**Step 3: Update README**

Per project convention, update README.md with the new features.

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test: verify Feature 2 DoclingDocument viewer"
```
