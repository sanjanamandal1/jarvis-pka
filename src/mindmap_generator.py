"""
Mind Map Generator — extracts concepts and relationships from documents,
renders as an interactive D3.js force-directed graph via Streamlit HTML component.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import List, Dict
from langchain_core.messages import HumanMessage
from src.llm_provider import get_llm


MINDMAP_PROMPT = """Analyze the document context and extract a knowledge graph for a mind map.

Document context:
{context}

Return ONLY valid JSON, no markdown:
{{
  "central": "Main Topic",
  "nodes": [
    {{"id": "1", "label": "Concept Name", "type": "concept", "description": "brief description"}},
    {{"id": "2", "label": "Another Concept", "type": "subtopic", "description": "brief description"}}
  ],
  "links": [
    {{"source": "central", "target": "1", "label": "includes"}},
    {{"source": "1", "target": "2", "label": "relates to"}}
  ]
}}

Rules:
- central: the main topic (1 node)
- nodes: 8-14 key concepts from the document
- links: connections between concepts
- types: "concept", "subtopic", "example", "definition"
- Keep labels SHORT (2-4 words max)"""


@dataclass
class MindMapData:
    central: str
    nodes: List[Dict]
    links: List[Dict]
    doc_name: str = ""


class MindMapGenerator:
    def __init__(self, kb, model: str = None):
        self.kb = kb
        self.llm = get_llm(model=model, temperature=0.3)

    def generate(self, topic: str = "", doc_ids: list = None) -> MindMapData:
        query = topic if topic else "main concepts key ideas overview"
        results = self.kb.search(query, k=10, doc_ids=doc_ids)

        docs = []
        for item in results:
            if isinstance(item, tuple):
                docs.append(item[0])
            elif hasattr(item, "page_content"):
                docs.append(item)

        if not docs:
            return MindMapData(central="No Data", nodes=[], links=[], doc_name="")

        context = "\n\n".join(
            f"[{d.metadata.get('filename','?')}]\n{d.page_content[:800]}"
            for d in docs[:8]
        )
        doc_name = docs[0].metadata.get("filename", "Documents") if docs else ""

        prompt = MINDMAP_PROMPT.format(context=context)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        raw = re.sub(r"```json|```", "", raw).strip()

        try:
            data = json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return MindMapData(central="Parse Error", nodes=[], links=[], doc_name=doc_name)
            else:
                return MindMapData(central="Parse Error", nodes=[], links=[], doc_name=doc_name)

        return MindMapData(
            central=data.get("central", "Main Topic"),
            nodes=data.get("nodes", []),
            links=data.get("links", []),
            doc_name=doc_name,
        )


def render_mindmap_html(mindmap: MindMapData) -> str:
    """Generate a self-contained D3.js force-directed mind map HTML."""

    # Build node/link data for D3
    all_nodes = [{"id": "central", "label": mindmap.central, "type": "central", "description": ""}]
    for n in mindmap.nodes:
        all_nodes.append({
            "id": n.get("id", n.get("label", "")),
            "label": n.get("label", ""),
            "type": n.get("type", "concept"),
            "description": n.get("description", ""),
        })

    all_links = []
    for l in mindmap.links:
        all_links.append({
            "source": l.get("source", ""),
            "target": l.get("target", ""),
            "label": l.get("label", ""),
        })

    nodes_json = json.dumps(all_nodes)
    links_json = json.dumps(all_links)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #000; font-family: 'Raleway', sans-serif; overflow: hidden; }}
  @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700;900&display=swap');

  svg {{ width: 100%; height: 100vh; }}

  .link {{ stroke: rgba(255,26,26,0.4); stroke-width: 1.5; fill: none; }}
  .link-label {{ fill: #6b6870; font-size: 9px; font-family: Raleway; letter-spacing: 0.05em; }}

  .node circle {{ stroke-width: 2; cursor: pointer; transition: all 0.3s; }}
  .node circle:hover {{ filter: brightness(1.5); }}

  .node.central circle {{ fill: #000; stroke: #ff1a1a; r: 40; filter: drop-shadow(0 0 15px rgba(255,26,26,0.8)); }}
  .node.concept circle {{ fill: #0a0a0f; stroke: #ff1a1a; }}
  .node.subtopic circle {{ fill: #0a0a0f; stroke: #00e5ff; }}
  .node.example circle {{ fill: #0a0a0f; stroke: #ffb700; }}
  .node.definition circle {{ fill: #0a0a0f; stroke: #cc44ff; }}

  .node text {{ font-family: Raleway; fill: #e8e0d0; text-anchor: middle; pointer-events: none; }}
  .node.central text {{ font-weight: 900; font-size: 11px; fill: #ff1a1a; }}
  .node.concept text {{ font-size: 9px; font-weight: 700; }}
  .node.subtopic text {{ font-size: 9px; fill: #00e5ff; }}
  .node.example text {{ font-size: 9px; fill: #ffb700; }}
  .node.definition text {{ font-size: 9px; fill: #cc44ff; }}

  .tooltip {{
    position: absolute; background: #0a0a0f; border: 1px solid #ff1a1a;
    color: #e8e0d0; padding: 8px 12px; font-size: 11px; font-family: Raleway;
    pointer-events: none; max-width: 200px; display: none;
    clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 0 100%);
  }}

  .legend {{ position: absolute; bottom: 20px; left: 20px; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 10px; font-family: Raleway; color: #6b6870; letter-spacing: 0.1em; text-transform: uppercase; }}
  .legend-dot {{ width: 8px; height: 8px; border-radius: 50%; border: 1.5px solid; }}

  /* pulse animation for central node */
  @keyframes pulse-ring {{
    0% {{ r: 42; opacity: 0.8; }}
    50% {{ r: 50; opacity: 0.2; }}
    100% {{ r: 42; opacity: 0.8; }}
  }}
  .pulse-ring {{ fill: none; stroke: #ff1a1a; stroke-width: 1; animation: pulse-ring 2s ease-in-out infinite; }}
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
<svg id="graph">
  <defs>
    <marker id="arrow" markerWidth="6" markerHeight="6" refX="20" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="rgba(255,26,26,0.4)"/>
    </marker>
    <!-- Scanline filter -->
    <filter id="scanlines">
      <feTurbulence type="fractalNoise" baseFrequency="0 0.5" numOctaves="1" result="noise"/>
      <feColorMatrix type="saturate" values="0" in="noise" result="grayNoise"/>
      <feBlend in="SourceGraphic" in2="grayNoise" mode="overlay" result="blend"/>
      <feComponentTransfer in="blend">
        <feFuncA type="linear" slope="0.97"/>
      </feComponentTransfer>
    </filter>
  </defs>
  <g id="links-group"></g>
  <g id="nodes-group"></g>
</svg>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="border-color:#ff1a1a"></div>Core Concept</div>
  <div class="legend-item"><div class="legend-dot" style="border-color:#00e5ff"></div>Subtopic</div>
  <div class="legend-item"><div class="legend-dot" style="border-color:#ffb700"></div>Example</div>
  <div class="legend-item"><div class="legend-dot" style="border-color:#cc44ff"></div>Definition</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const nodes = {nodes_json};
const links = {links_json};

const w = window.innerWidth, h = window.innerHeight;
const svg = d3.select("#graph").attr("viewBox", `0 0 ${{w}} ${{h}}`);
const tooltip = document.getElementById("tooltip");

const colorMap = {{
  central: "#ff1a1a", concept: "#ff1a1a",
  subtopic: "#00e5ff", example: "#ffb700", definition: "#cc44ff"
}};

const radiusMap = {{
  central: 42, concept: 28, subtopic: 24, example: 22, definition: 22
}};

const sim = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(d => d.source.type === "central" ? 160 : 90).strength(0.8))
  .force("charge", d3.forceManyBody().strength(-400))
  .force("center", d3.forceCenter(w/2, h/2))
  .force("collision", d3.forceCollide().radius(d => radiusMap[d.type] + 18));

// Links
const linkG = svg.select("#links-group").selectAll("g").data(links).enter().append("g");
const linkLine = linkG.append("line").attr("class", "link").attr("marker-end", "url(#arrow)");
const linkLabel = linkG.append("text").attr("class", "link-label").attr("text-anchor","middle").text(d => d.label);

// Nodes
const nodeG = svg.select("#nodes-group").selectAll("g").data(nodes).enter()
  .append("g").attr("class", d => `node ${{d.type}}`)
  .call(d3.drag()
    .on("start", (e,d) => {{ if (!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
    .on("drag",  (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
    .on("end",   (e,d) => {{ if (!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }})
  );

// Pulse ring for central
nodeG.filter(d => d.type === "central").append("circle").attr("class","pulse-ring").attr("r", 44);

// Glow circles (background)
nodeG.append("circle")
  .attr("r", d => radiusMap[d.type] + 6)
  .attr("fill", "none")
  .attr("stroke", d => colorMap[d.type] || "#ff1a1a")
  .attr("stroke-width", 0.5)
  .attr("opacity", 0.3);

// Main circles
nodeG.append("circle")
  .attr("r", d => radiusMap[d.type])
  .attr("fill", d => d.type === "central" ? "#050508" : "#0a0a0f")
  .attr("stroke", d => colorMap[d.type] || "#ff1a1a")
  .attr("stroke-width", d => d.type === "central" ? 2.5 : 1.5)
  .style("filter", d => `drop-shadow(0 0 ${{d.type==="central"?12:6}}px ${{colorMap[d.type]||"#ff1a1a"}}66)`);

// Labels (word wrap)
nodeG.each(function(d) {{
  const g = d3.select(this);
  const words = d.label.split(" ");
  const r = radiusMap[d.type];
  if (words.length <= 2) {{
    g.append("text").attr("dy", "0.35em").attr("font-size", d.type==="central"?"12px":"9px").text(d.label);
  }} else {{
    const line1 = words.slice(0, Math.ceil(words.length/2)).join(" ");
    const line2 = words.slice(Math.ceil(words.length/2)).join(" ");
    g.append("text").attr("dy", "-0.3em").attr("font-size","9px").text(line1);
    g.append("text").attr("dy", "0.9em").attr("font-size","9px").text(line2);
  }}
}});

// Tooltip
nodeG.on("mouseover", (e, d) => {{
  if (d.description) {{
    tooltip.style.display = "block";
    tooltip.style.left = (e.pageX + 12) + "px";
    tooltip.style.top = (e.pageY - 10) + "px";
    tooltip.innerHTML = `<strong style="color:#ff1a1a">${{d.label}}</strong><br/>${{d.description}}`;
  }}
}}).on("mouseout", () => {{ tooltip.style.display = "none"; }});

sim.on("tick", () => {{
  linkLine
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  linkLabel
    .attr("x", d => (d.source.x + d.target.x)/2)
    .attr("y", d => (d.source.y + d.target.y)/2 - 5);
  nodeG.attr("transform", d => `translate(${{d.x}},${{d.y)}}`);
}});
</script>
</body>
</html>"""