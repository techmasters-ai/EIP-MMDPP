# Graph Search Result Visualization — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Add an interactive graph visualization overlay to each search result in the Graph Explorer's Search tab, allowing users to toggle between the current text/JSON view and a Cytoscape.js-powered graph view showing the full entity neighborhood.

**Architecture:** Each search result card gets a toggle button. Clicking it swaps the card content to a Cytoscape graph showing the searched entity and its full 2-hop neighborhood. Nodes and edges display all non-empty metadata on hover. Clicking a node triggers a new search for that entity.

**Tech Stack:** Cytoscape.js + react-cytoscapejs (React wrapper)

---

## 1. Component Architecture

### New Components

**`GraphView.tsx`** — Cytoscape wrapper component.
- Props: `nodes` and `edges` arrays (Cytoscape element format), `onNodeClick` callback
- Renders a Cytoscape canvas with cose (force-directed) layout
- Manages hover tooltip state and positioning
- Emits node click events to parent for triggering new searches

**`GraphTooltip.tsx`** — Positioned HTML tooltip overlay.
- Props: `visible`, `position` (x/y), `data` (key-value pairs)
- Renders a styled `div` positioned near the cursor, clamped to container bounds
- Displays all non-null/non-empty properties as key-value pairs
- Works for both nodes and edges

### Modified Components

**`GraphExplorer.tsx`** — The `GraphSearch` sub-component gets:
- A small graph icon toggle button in each result card header
- State tracking which card (if any) is in graph view (only one at a time)
- When graph view is active: fetches full neighborhood data via new API call, transforms to Cytoscape format, renders `GraphView` as overlay replacing the card content
- When a node is clicked in the graph: calls `queryGraph()` with that entity name, replaces search results

**`styles.css`** — New styles for:
- Graph container (fixed height, dark-themed background)
- Graph toggle button
- Tooltip appearance (dark card matching existing theme)
- Entity-type color classes for node styling

**`api/client.ts`** — New function:
- `getGraphNeighborhood(name, hopCount?)` → `POST /v1/graph/neighborhood`

### No Changes To

- `QueryPage.tsx`, `Nav.tsx`, `App.tsx` — unaffected
- Backend graph ingest endpoints — unaffected

## 2. Backend: Neighborhood Endpoint

The existing `/v1/graph/query` endpoint is insufficient for visualization because:
1. It truncates neighbors to 5 per result (`neighbors[:5]`)
2. The Cypher query uses `type(head(r))` which only captures the first relationship in a multi-hop path, losing intermediate nodes and edges

**New endpoint:** `POST /v1/graph/neighborhood`

**Request:**
```json
{
  "entity_name": "Patriot PAC-3",
  "hop_count": 2
}
```

**Response:**
```json
{
  "center": { "name": "Patriot PAC-3", "entity_type": "MissileSystem", "confidence": 0.95, ... },
  "nodes": [
    { "name": "...", "entity_type": "...", "confidence": ..., ...custom_props }
  ],
  "edges": [
    { "source": "Patriot PAC-3", "target": "Fire Control Radar", "rel_type": "contains", "confidence": 0.9, "artifact_id": "..." }
  ]
}
```

**New Cypher query** in `neo4j_graph.py` — properly unwinds variable-length paths into individual edges:
```cypher
MATCH (start:Entity {name: $name})
OPTIONAL MATCH path = (start)-[*1..N]-(neighbor:Entity)
WITH start, relationships(path) AS rels, nodes(path) AS path_nodes
UNWIND range(0, size(rels)-1) AS idx
WITH start,
     path_nodes[idx] AS from_node,
     rels[idx] AS rel,
     path_nodes[idx+1] AS to_node
RETURN DISTINCT
    from_node.name AS source,
    from_node.entity_type AS source_type,
    properties(from_node) AS source_props,
    type(rel) AS rel_type,
    properties(rel) AS rel_props,
    to_node.name AS target,
    to_node.entity_type AS target_type,
    properties(to_node) AS target_props
LIMIT $limit
```

This returns individual edges with full properties on both endpoints, which maps directly to Cytoscape's `{ nodes, edges }` format.

**Files:**
- Modify: `app/services/neo4j_graph.py` — add `get_neighborhood_graph_async()`
- Modify: `app/api/v1/graph_store.py` — add `POST /v1/graph/neighborhood` route
- Create: `app/schemas/graph_store.py` — add `GraphNeighborhoodRequest` and `GraphNeighborhoodResponse` schemas

## 3. Graph Rendering & Styling

**Layout:** Cose (Compound Spring Embedder) — Cytoscape's built-in force-directed layout. Deterministic, keeps related nodes clustered, works well for 20-50 node graphs.

**Node styling by entity type category:**
- Military systems (RadarSystem, MissileSystem, etc.) — `--color-primary` (#7B6B52) family
- EM/RF (FrequencyBand, Waveform, Antenna, etc.) — `--color-info` (#2A5A8A) family
- Weapon (Seeker, GuidanceMethod, etc.) — `--color-error` (#B94040) family
- Operational (Capability, EngagementTimeline, etc.) — `--color-warning` (#7A6020) family
- Reference (Organization, Document, Assertion) — `--color-text-muted` (#6B6560) family
- Center/searched entity gets a highlighted border (2px solid white or accent)

**Node labels:** Entity `name`, truncated with ellipsis if > 20 characters.

**Edge styling:**
- Labeled with relationship type (e.g., `contains`)
- Curve style: `bezier` (allows multiple edges between same pair)
- Subtle color: `--color-border` (#DDD5C8) with readable label text
- Arrow on target end to show direction

**Interactions:**
- Pan: mouse drag on background
- Zoom: scroll wheel
- Hover node: tooltip with all non-empty properties
- Hover edge: tooltip with rel_type + edge properties
- Click node: triggers new search
- Click background: dismisses tooltip

## 4. Data Transformation

Utility function `toGraphElements(response: GraphNeighborhoodResponse)` converts the API response to Cytoscape format:

```typescript
{
  nodes: [
    { data: { id: "Patriot PAC-3", label: "Patriot PAC-3", entity_type: "MissileSystem", ...props }, classes: "military" },
    ...
  ],
  edges: [
    { data: { id: "Patriot PAC-3->Fire Control Radar::contains", source: "Patriot PAC-3", target: "Fire Control Radar", label: "contains", ...props } },
    ...
  ]
}
```

- Node `id` = entity name (unique within a neighborhood)
- Edge `id` = `source->target::rel_type` (deduplicates parallel edges of same type)
- `classes` field maps entity_type to a category for styling

## 5. Tooltip Rendering

`GraphTooltip` is a positioned HTML `div` overlay (not a Cytoscape extension) for full CSS control.

**For nodes:**
| Key | Value |
|-----|-------|
| name | Patriot PAC-3 |
| entity_type | MissileSystem |
| confidence | 0.95 |
| artifact_id | abc123 |
| ...custom | ...values |

**For edges:**
| Key | Value |
|-----|-------|
| relationship | contains |
| confidence | 0.9 |
| artifact_id | abc123 |

- Skips null, undefined, empty-string, and internal ID values
- Positioned near cursor, clamped to stay within graph container
- Styled to match existing dark card aesthetic (`--color-surface`, `--color-border`)

## 6. Overlay Behavior

- Toggle button: small graph icon in result card header (next to rank badge)
- Only one graph open at a time — toggling one closes any other
- Graph container replaces the result card content area
- Container has a fixed height (~400px) with the card expanding to fit
- Close button (X) in the top-right corner of the graph area returns to text view

## 7. Click-to-Search Flow

1. User clicks a node in the graph
2. `onNodeClick` callback fires with the entity name
3. `GraphSearch` component calls `queryGraph({ query: entityName })`
4. Search results update, graph view closes
5. User can toggle graph view on any new result

## 8. Dependencies & Files Summary

**New npm dependencies:**
- `cytoscape` — graph visualization engine
- `react-cytoscapejs` — React wrapper

**Files to create:**
- `frontend/src/components/GraphView.tsx`
- `frontend/src/components/GraphTooltip.tsx`

**Files to modify:**
- `frontend/src/components/GraphExplorer.tsx` — toggle button, graph state, search-on-click
- `frontend/src/styles.css` — graph container, tooltip, toggle button, entity-type colors
- `frontend/src/api/client.ts` — `getGraphNeighborhood()` function
- `frontend/package.json` — add cytoscape + react-cytoscapejs
- `app/services/neo4j_graph.py` — `get_neighborhood_graph_async()` function
- `app/api/v1/graph_store.py` — `POST /v1/graph/neighborhood` route
- `app/schemas/graph_store.py` — request/response schemas
