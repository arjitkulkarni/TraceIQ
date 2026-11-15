import cytoscape from "cytoscape";
import { useEffect, useRef } from "react";

const SHAPE = { illicit: "triangle", licit: "rectangle", unknown: "ellipse" };

function readToken(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Real directed neighborhood. Shape shows the dataset label, border shows the model tier. */
export default function GraphView({ graph, onSelect }) {
  const container = useRef(null);

  useEffect(() => {
    if (!container.current || !graph) return undefined;
    const tierColor = {
      high: readToken("--critical"),
      medium: readToken("--serious"),
      low: readToken("--line-strong"),
    };
    const labelFill = {
      illicit: readToken("--series-2"),
      licit: readToken("--series-1"),
      unknown: readToken("--unknown"),
    };
    const drops = graph.edges.map((edge) => Math.abs(edge.logit_drop ?? 0));
    const maxDrop = Math.max(...drops, 1e-9);
    const crowded = graph.nodes.length > 20;
    const cy = cytoscape({
      container: container.current,
      elements: [
        ...graph.nodes.map((node) => ({
          data: {
            id: node.transaction_id,
            hover: `${node.transaction_id} · score ${node.model_score.toFixed(3)} · ${node.dataset_label}`,
            caption: !crowded || node.is_focal ? `${node.transaction_id} · ${node.model_score.toFixed(2)}` : "",
            shape: SHAPE[node.dataset_label],
            fill: labelFill[node.dataset_label],
            border: tierColor[node.risk_tier],
            size: node.is_focal ? 34 : 24,
            borderWidth: node.is_focal ? 5 : 3,
          },
        })),
        ...graph.edges.map((edge) => ({
          data: {
            id: `${edge.source}->${edge.target}`,
            source: edge.source,
            target: edge.target,
            width: edge.logit_drop == null ? 1.2 : 1.2 + 4 * (Math.abs(edge.logit_drop) / maxDrop),
          },
        })),
      ],
      layout: { name: "cose", animate: false, nodeRepulsion: 9000, idealEdgeLength: 70, randomize: false },
      minZoom: 0.3,
      maxZoom: 2.5,
      style: [
        {
          selector: "node",
          style: {
            shape: "data(shape)",
            width: "data(size)",
            height: "data(size)",
            "background-color": "data(fill)",
            "border-color": "data(border)",
            "border-width": "data(borderWidth)",
            label: "data(caption)",
            "font-size": 9,
            color: readToken("--ink-2"),
            "text-valign": "bottom",
            "text-margin-y": 4,
          },
        },
        {
          selector: "edge",
          style: {
            width: "data(width)",
            "line-color": readToken("--line-strong"),
            "target-arrow-color": readToken("--line-strong"),
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "arrow-scale": 0.9,
          },
        },
      ],
    });
    cy.on("tap", "node", (event) => onSelect?.(event.target.id()));
    cy.on("mouseover", "node", (event) => event.target.style("label", event.target.data("hover")));
    cy.on("mouseout", "node", (event) => event.target.style("label", event.target.data("caption")));
    return () => cy.destroy();
  }, [graph, onSelect]);

  return <div className="graph" ref={container} role="img" aria-label="Directed transaction neighborhood" />;
}
