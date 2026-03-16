import { useRef, useState, useCallback } from "react";
// @ts-expect-error — react-cytoscapejs has no type declarations
import CytoscapeComponent from "react-cytoscapejs";
import type cytoscape from "cytoscape";
import { GraphTooltip } from "./GraphTooltip";
import type { GraphNeighborhoodResponse } from "../api/client";
import { getEntityCategory } from "../constants/entityTypes";

/** Convert API response to Cytoscape elements. */
export function toGraphElements(
  response: GraphNeighborhoodResponse,
  centerName: string,
): cytoscape.ElementDefinition[] {
  const elements: cytoscape.ElementDefinition[] = [];
  const nodeIds = new Set<string>();

  for (const node of response.nodes) {
    const name = node.name as string;
    if (!name || nodeIds.has(name)) continue;
    nodeIds.add(name);
    const entityType = (node.entity_type as string) || "UNKNOWN";
    const label = name.length > 20 ? name.slice(0, 18) + "\u2026" : name;
    elements.push({
      data: { id: name, label, ...node },
      classes: `${getEntityCategory(entityType)}${name === centerName ? " center" : ""}`,
    });
  }

  const edgeIds = new Set<string>();
  for (const edge of response.edges) {
    const edgeId = `${edge.source}->${edge.target}::${edge.rel_type}`;
    if (edgeIds.has(edgeId)) continue;
    edgeIds.add(edgeId);
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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const STYLESHEET: any[] = [
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
  { selector: "node.military", style: { "background-color": "#7B6B52" } },
  { selector: "node.emrf", style: { "background-color": "#2A5A8A" } },
  { selector: "node.weapon", style: { "background-color": "#B94040" } },
  { selector: "node.operational", style: { "background-color": "#7A6020" } },
  { selector: "node.reference", style: { "background-color": "#6B6560" } },
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
