"""
Mind Map Generator — uses Canvas API (no external dependencies).
Pure HTML/CSS/JS with no CDN imports — works in Streamlit sandbox.
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

Return ONLY valid JSON, no markdown, no explanation:
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
- nodes: 8-12 key concepts from the document
- links: connections between them
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
    """Canvas-based force-directed mind map — no external CDN needed."""

    all_nodes = [{"id": "central", "label": mindmap.central, "type": "central", "description": ""}]
    for n in mindmap.nodes:
        all_nodes.append({
            "id": n.get("id", n.get("label", "")),
            "label": n.get("label", ""),
            "type": n.get("type", "concept"),
            "description": n.get("description", ""),
        })

    all_links = []
    for lnk in mindmap.links:
        all_links.append({
            "source": lnk.get("source", ""),
            "target": lnk.get("target", ""),
            "label": lnk.get("label", ""),
        })

    nodes_json = json.dumps(all_nodes)
    links_json = json.dumps(all_links)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#000; overflow:hidden; font-family:'Raleway',sans-serif; }}
  canvas {{ display:block; }}
  #tooltip {{
    position:absolute; background:#0a0a0f; border:1px solid #ff1a1a;
    color:#e8e0d0; padding:8px 12px; font-size:11px; font-family:sans-serif;
    pointer-events:none; max-width:200px; display:none; border-radius:2px;
  }}
  #legend {{
    position:absolute; bottom:16px; left:16px;
    font-family:sans-serif; font-size:10px;
  }}
  .li {{ display:flex; align-items:center; gap:6px; margin-bottom:4px; color:#6b6870; text-transform:uppercase; letter-spacing:0.1em; }}
  .dot {{ width:8px; height:8px; border-radius:50%; border:1.5px solid; }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="tooltip"></div>
<div id="legend">
  <div class="li"><div class="dot" style="border-color:#ff1a1a"></div>Core Concept</div>
  <div class="li"><div class="dot" style="border-color:#00e5ff"></div>Subtopic</div>
  <div class="li"><div class="dot" style="border-color:#ffb700"></div>Example</div>
  <div class="li"><div class="dot" style="border-color:#cc44ff"></div>Definition</div>
</div>
<script>
const NODES = {nodes_json};
const LINKS = {links_json};

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

let W = window.innerWidth, H = window.innerHeight;
canvas.width = W; canvas.height = H;

const COLOR = {{ central:'#ff1a1a', concept:'#ff1a1a', subtopic:'#00e5ff', example:'#ffb700', definition:'#cc44ff' }};
const RADIUS = {{ central:50, concept:34, subtopic:28, example:26, definition:26 }};

// Init node positions in a circle
NODES.forEach((n, i) => {{
  if (n.type === 'central') {{ n.x = W/2; n.y = H/2; }}
  else {{
    const angle = (2 * Math.PI * i) / (NODES.length - 1);
    const r = Math.min(W, H) * 0.32;
    n.x = W/2 + r * Math.cos(angle);
    n.y = H/2 + r * Math.sin(angle);
  }}
  n.vx = 0; n.vy = 0;
  n.r = RADIUS[n.type] || 28;
}});

// Build id→node map
const nodeMap = {{}};
NODES.forEach(n => nodeMap[n.id] = n);

// Resolve links
const resolvedLinks = LINKS.map(l => ({{
  source: nodeMap[l.source] || NODES[0],
  target: nodeMap[l.target] || NODES[1],
  label: l.label
}})).filter(l => l.source && l.target);

// Force simulation
let dragging = null, dragOffX = 0, dragOffY = 0;
let pulse = 0;

function simulate() {{
  const K = 0.04, REP = 3500, DAMP = 0.85;

  // Spring forces (links)
  resolvedLinks.forEach(l => {{
    const dx = l.target.x - l.source.x;
    const dy = l.target.y - l.source.y;
    const dist = Math.sqrt(dx*dx + dy*dy) || 1;
    const ideal = l.source.type==='central' ? 180 : 110;
    const force = (dist - ideal) * K;
    const fx = (dx/dist) * force;
    const fy = (dy/dist) * force;
    if (l.source.type !== 'central') {{ l.source.vx += fx; l.source.vy += fy; }}
    if (l.target.type !== 'central') {{ l.target.vx -= fx; l.target.vy -= fy; }}
  }});

  // Repulsion
  for (let i = 0; i < NODES.length; i++) {{
    for (let j = i+1; j < NODES.length; j++) {{
      const a = NODES[i], b = NODES[j];
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist2 = dx*dx + dy*dy || 1;
      const force = REP / dist2;
      const fx = (dx / Math.sqrt(dist2)) * force;
      const fy = (dy / Math.sqrt(dist2)) * force;
      if (a.type !== 'central') {{ a.vx -= fx; a.vy -= fy; }}
      if (b.type !== 'central') {{ b.vx += fx; b.vy += fy; }}
    }}
  }}

  // Center attraction
  NODES.forEach(n => {{
    if (n.type === 'central') return;
    n.vx += (W/2 - n.x) * 0.002;
    n.vy += (H/2 - n.y) * 0.002;
  }});

  // Integrate
  NODES.forEach(n => {{
    if (n === dragging || n.type === 'central') return;
    n.vx *= DAMP; n.vy *= DAMP;
    n.x += n.vx; n.y += n.vy;
    // Bounds
    n.x = Math.max(n.r+10, Math.min(W-n.r-10, n.x));
    n.y = Math.max(n.r+10, Math.min(H-n.r-10, n.y));
  }});
}}

function drawGlow(x, y, r, color, blur) {{
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI*2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();
}}

function drawNode(n) {{
  const color = COLOR[n.type] || '#ff1a1a';

  // Outer glow ring
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = n.type==='central' ? 25 : 12;
  ctx.beginPath();
  ctx.arc(n.x, n.y, n.r, 0, Math.PI*2);
  ctx.strokeStyle = color + '44';
  ctx.lineWidth = n.type==='central' ? 10 : 6;
  ctx.stroke();
  ctx.restore();

  // Pulse ring for central
  if (n.type === 'central') {{
    const pr = n.r + 8 + Math.sin(pulse) * 6;
    ctx.beginPath();
    ctx.arc(n.x, n.y, pr, 0, Math.PI*2);
    ctx.strokeStyle = `rgba(255,26,26,${{0.3 + 0.2 * Math.sin(pulse)}})`;
    ctx.lineWidth = 1;
    ctx.stroke();
  }}

  // Fill
  ctx.beginPath();
  ctx.arc(n.x, n.y, n.r, 0, Math.PI*2);
  ctx.fillStyle = n.type==='central' ? '#050508' : '#0a0a0f';
  ctx.fill();

  // Border
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = n.type==='central' ? 20 : 8;
  ctx.beginPath();
  ctx.arc(n.x, n.y, n.r, 0, Math.PI*2);
  ctx.strokeStyle = color;
  ctx.lineWidth = n.type==='central' ? 2.5 : 1.5;
  ctx.stroke();
  ctx.restore();

  // Label
  ctx.fillStyle = COLOR[n.type] || '#e8e0d0';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const fontSize = n.type==='central' ? 12 : 9;
  ctx.font = `${{n.type==='central'?'900':'700'}} ${{fontSize}}px sans-serif`;

  const words = n.label.split(' ');
  if (words.length <= 2) {{
    ctx.fillText(n.label, n.x, n.y);
  }} else {{
    const half = Math.ceil(words.length / 2);
    ctx.fillText(words.slice(0, half).join(' '), n.x, n.y - fontSize*0.6);
    ctx.fillText(words.slice(half).join(' '), n.x, n.y + fontSize*0.6);
  }}
}}

function drawLink(l) {{
  const dx = l.target.x - l.source.x;
  const dy = l.target.y - l.source.y;
  const dist = Math.sqrt(dx*dx + dy*dy) || 1;
  const ux = dx/dist, uy = dy/dist;

  const sx = l.source.x + ux * l.source.r;
  const sy = l.source.y + uy * l.source.r;
  const ex = l.target.x - ux * (l.target.r + 6);
  const ey = l.target.y - uy * (l.target.r + 6);

  // Line
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.strokeStyle = 'rgba(255,26,26,0.25)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Arrow
  const angle = Math.atan2(ey-sy, ex-sx);
  ctx.save();
  ctx.translate(ex, ey);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.moveTo(0,0); ctx.lineTo(-8,-4); ctx.lineTo(-8,4); ctx.closePath();
  ctx.fillStyle = 'rgba(255,26,26,0.4)';
  ctx.fill();
  ctx.restore();

  // Link label
  if (l.label) {{
    ctx.fillStyle = '#444';
    ctx.font = '8px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(l.label, (sx+ex)/2, (sy+ey)/2 - 5);
  }}
}}

function draw() {{
  ctx.clearRect(0,0,W,H);

  // Scanline effect
  for (let y=0; y<H; y+=4) {{
    ctx.fillStyle = 'rgba(255,26,26,0.012)';
    ctx.fillRect(0, y, W, 1);
  }}

  simulate();
  pulse += 0.04;

  resolvedLinks.forEach(drawLink);
  NODES.forEach(drawNode);

  requestAnimationFrame(draw);
}}

// Mouse interaction
function getNode(mx, my) {{
  return NODES.find(n => Math.hypot(n.x-mx, n.y-my) < n.r + 8);
}}

canvas.addEventListener('mousedown', e => {{
  const n = getNode(e.offsetX, e.offsetY);
  if (n) {{ dragging = n; dragOffX = n.x - e.offsetX; dragOffY = n.y - e.offsetY; }}
}});
canvas.addEventListener('mousemove', e => {{
  if (dragging) {{
    dragging.x = e.offsetX + dragOffX;
    dragging.y = e.offsetY + dragOffY;
    dragging.vx = 0; dragging.vy = 0;
  }}
  const n = getNode(e.offsetX, e.offsetY);
  if (n && n.description) {{
    tooltip.style.display = 'block';
    tooltip.style.left = (e.offsetX + 14) + 'px';
    tooltip.style.top = (e.offsetY - 10) + 'px';
    tooltip.innerHTML = `<strong style="color:#ff1a1a">${{n.label}}</strong><br>${{n.description}}`;
  }} else {{
    tooltip.style.display = 'none';
  }}
}});
canvas.addEventListener('mouseup', () => {{ dragging = null; }});

window.addEventListener('resize', () => {{
  W = window.innerWidth; H = window.innerHeight;
  canvas.width = W; canvas.height = H;
  NODES[0].x = W/2; NODES[0].y = H/2;
}});

draw();
</script>
</body>
</html>"""