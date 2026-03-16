import React from "react";

interface GraphTooltipProps {
  visible: boolean;
  x: number;
  y: number;
  data: Record<string, unknown>;
  containerRect?: DOMRect;
}

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
    <div className="graph-tooltip" style={{ left, top }}>
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
