import { useRef, useEffect } from "react";

interface ConceptNode {
  id: number;
  type: string;
  name: string;
}

interface ConceptEdge {
  source: number;
  target: number;
  relationship: string;
}

interface ConceptMapProps {
  nodes: ConceptNode[];
  edges: ConceptEdge[];
  onNodeClick?: (name: string) => void;
}

const COLORS: Record<string, string> = {
  Entry: "#4a9eff",
  Person: "#ff6b6b",
  Place: "#51cf66",
  Concept: "#ffd43b",
  Emotion: "#cc5de8",
  Decision: "#ff922b",
  Archetype: "#20c997",
};

export default function ConceptMap({ nodes, edges, onNodeClick }: ConceptMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    // Layout: arrange in a circle
    const positioned = nodes.slice(0, 10).map((n, i, arr) => {
      const angle = (2 * Math.PI * i) / arr.length - Math.PI / 2;
      const rx = W * 0.35;
      const ry = H * 0.35;
      return { ...n, x: W / 2 + rx * Math.cos(angle), y: H / 2 + ry * Math.sin(angle) };
    });

    const nodeMap = new Map(positioned.map((n) => [n.id, n]));

    ctx.clearRect(0, 0, W, H);

    // Edges
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    for (const e of edges) {
      const a = nodeMap.get(e.source);
      const b = nodeMap.get(e.target);
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    // Nodes
    for (const n of positioned) {
      const color = COLORS[n.type] || "#888";
      ctx.beginPath();
      ctx.arc(n.x, n.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.fillStyle = "#bbb";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      const label = n.name.length > 14 ? n.name.slice(0, 12) + ".." : n.name;
      ctx.fillText(label, n.x, n.y + 16);
    }
  }, [nodes, edges]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!onNodeClick) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const positioned = nodes.slice(0, 10).map((n, i, arr) => {
      const angle = (2 * Math.PI * i) / arr.length - Math.PI / 2;
      return {
        ...n,
        x: canvas.width / 2 + canvas.width * 0.35 * Math.cos(angle),
        y: canvas.height / 2 + canvas.height * 0.35 * Math.sin(angle),
      };
    });

    for (const n of positioned) {
      const dx = mx - n.x;
      const dy = my - n.y;
      if (dx * dx + dy * dy < 100) {
        onNodeClick(n.name);
        break;
      }
    }
  }

  if (nodes.length === 0) return null;

  return (
    <div className="concept-map">
      <canvas
        ref={canvasRef}
        width={300}
        height={200}
        className="concept-map-canvas"
        onClick={handleClick}
      />
    </div>
  );
}
