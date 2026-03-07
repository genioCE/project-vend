import { useRef, useEffect, useState } from "react";

interface GraphNode {
  id: number;
  type: string;
  name: string;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

interface GraphEdge {
  source: number;
  target: number;
  relationship: string;
}

interface GraphPanelProps {
  visible: boolean;
  onClose: () => void;
  center?: string;
}

const NODE_COLORS: Record<string, string> = {
  Entry: "#4a9eff",
  Person: "#ff6b6b",
  Place: "#51cf66",
  Concept: "#ffd43b",
  Emotion: "#cc5de8",
  Decision: "#ff922b",
  Archetype: "#20c997",
};

export default function GraphPanel({ visible, onClose, center }: GraphPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [loading, setLoading] = useState(false);
  const animRef = useRef<number>(0);

  useEffect(() => {
    if (!visible || !center) return;

    setLoading(true);
    fetch(`/api/graph/subgraph?center=${encodeURIComponent(center)}&limit=40`)
      .then((r) => r.json())
      .then((data) => {
        const n = (data.nodes || []).map((node: GraphNode) => ({
          ...node,
          x: 200 + Math.random() * 200,
          y: 150 + Math.random() * 200,
          vx: 0,
          vy: 0,
        }));
        setNodes(n);
        setEdges(data.edges || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [visible, center]);

  // Simple force simulation drawn on canvas
  useEffect(() => {
    if (!visible || nodes.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));

    function tick() {
      // Repulsion between all nodes
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          let dx = (b.x || 0) - (a.x || 0);
          let dy = (b.y || 0) - (a.y || 0);
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = 800 / (dist * dist);
          dx = (dx / dist) * force;
          dy = (dy / dist) * force;
          a.vx = (a.vx || 0) - dx;
          a.vy = (a.vy || 0) - dy;
          b.vx = (b.vx || 0) + dx;
          b.vy = (b.vy || 0) + dy;
        }
      }

      // Attraction along edges
      for (const e of edges) {
        const a = nodeMap.get(e.source);
        const b = nodeMap.get(e.target);
        if (!a || !b) continue;
        let dx = (b.x || 0) - (a.x || 0);
        let dy = (b.y || 0) - (a.y || 0);
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (dist - 80) * 0.02;
        dx = (dx / dist) * force;
        dy = (dy / dist) * force;
        a.vx = (a.vx || 0) + dx;
        a.vy = (a.vy || 0) + dy;
        b.vx = (b.vx || 0) - dx;
        b.vy = (b.vy || 0) - dy;
      }

      // Center gravity + damping
      for (const n of nodes) {
        n.vx = ((n.vx || 0) + (W / 2 - (n.x || 0)) * 0.001) * 0.9;
        n.vy = ((n.vy || 0) + (H / 2 - (n.y || 0)) * 0.001) * 0.9;
        n.x = Math.max(20, Math.min(W - 20, (n.x || 0) + (n.vx || 0)));
        n.y = Math.max(20, Math.min(H - 20, (n.y || 0) + (n.vy || 0)));
      }

      // Draw
      ctx.clearRect(0, 0, W, H);

      // Edges
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 1;
      for (const e of edges) {
        const a = nodeMap.get(e.source);
        const b = nodeMap.get(e.target);
        if (!a || !b) continue;
        ctx.beginPath();
        ctx.moveTo(a.x || 0, a.y || 0);
        ctx.lineTo(b.x || 0, b.y || 0);
        ctx.stroke();
      }

      // Nodes
      for (const n of nodes) {
        const color = NODE_COLORS[n.type] || "#888";
        const r = n.name.toLowerCase() === center?.toLowerCase() ? 10 : 6;
        ctx.beginPath();
        ctx.arc(n.x || 0, n.y || 0, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        ctx.fillStyle = "#ccc";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "center";
        const label = n.name.length > 18 ? n.name.slice(0, 16) + ".." : n.name;
        ctx.fillText(label, n.x || 0, (n.y || 0) + r + 12);
      }

      animRef.current = requestAnimationFrame(tick);
    }

    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [visible, nodes, edges, center]);

  if (!visible) return null;

  return (
    <div className="graph-panel">
      <div className="graph-panel-header">
        <span>Knowledge Graph: {center}</span>
        <button className="graph-panel-close" onClick={onClose}>
          {"\u2715"}
        </button>
      </div>
      {loading ? (
        <div className="graph-panel-loading">Loading graph...</div>
      ) : nodes.length === 0 ? (
        <div className="graph-panel-loading">No graph data found</div>
      ) : (
        <canvas
          ref={canvasRef}
          width={380}
          height={500}
          className="graph-canvas"
        />
      )}
      <div className="graph-panel-legend">
        {Object.entries(NODE_COLORS).map(([type, color]) => (
          <span key={type} className="graph-legend-item">
            <span className="graph-legend-dot" style={{ background: color }} />
            {type}
          </span>
        ))}
      </div>
    </div>
  );
}
