"""
Export Center — convert JARVIS-generated content to downloadable formats.

All methods return strings (Markdown or plain text) suitable for
Streamlit's st.download_button — no server-side file creation needed.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from .logger import get_logger

log = get_logger("exporter")


class Exporter:
    """Static export helpers. All methods return a string ready for download."""

    # ── Chat exports ──────────────────────────────────────────────────────────

    @staticmethod
    def chat_to_markdown(chat_history: List[Dict[str, Any]], workspace_name: str = "JARVIS") -> str:
        """Export full chat history as a Markdown document."""
        lines = [
            f"# {workspace_name} — Chat Log",
            f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
            "",
            "---",
            "",
        ]
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"**▶ You:** {content}")
                lines.append("")
            elif role == "assistant":
                intent = msg.get("intent", "RESPONSE").upper()
                icon = msg.get("intent_icon", "◈")
                lines.append(f"**{icon} JARVIS [{intent}]:**")
                lines.append(content)
                sources = msg.get("sources", [])
                if sources:
                    lines.append("")
                    lines.append("_Sources: " + ", ".join(
                        f"{s['filename']} v{s['version']}" for s in sources
                    ) + "_")
                lines.append("")
                lines.append("---")
                lines.append("")
        return "\n".join(lines)

    @staticmethod
    def chat_to_txt(chat_history: List[Dict[str, Any]]) -> str:
        """Export full chat history as plain text."""
        lines = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"YOU: {content}")
            elif role == "assistant":
                lines.append(f"JARVIS: {content}")
            lines.append("")
        return "\n".join(lines)

    # ── Summary exports ───────────────────────────────────────────────────────

    @staticmethod
    def summaries_to_markdown(doc_summaries: Dict[str, Any]) -> str:
        """Export hierarchical summaries as Markdown."""
        lines = [
            "# JARVIS — Document Summaries",
            f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
            "",
        ]
        for doc_id, summary in doc_summaries.items():
            lines.append(f"## {summary.filename}")
            lines.append("")
            lines.append(f"> {summary.document_summary}")
            lines.append("")
            lines.append(f"_{summary.total_chunks} chunks · {summary.total_words:,} words · {len(summary.section_summaries)} sections_")
            lines.append("")
            for i, sec in enumerate(summary.section_summaries, 1):
                lines.append(f"### Section {i}")
                lines.append(sec.summary)
                lines.append("")
            lines.append("---")
            lines.append("")
        return "\n".join(lines)

    # ── Quiz exports ──────────────────────────────────────────────────────────

    @staticmethod
    def quiz_to_markdown(quiz: Any) -> str:
        """Export quiz as a formatted Markdown worksheet."""
        lines = [
            "# JARVIS — Quiz Sheet",
            f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
            "",
            "---",
            "",
        ]
        for i, q in enumerate(quiz.questions, 1):
            lines.append(f"**Q{i}.** {q.question}")
            lines.append("")
            for opt in q.options:
                lines.append(f"- {opt}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## Answer Key")
        lines.append("")
        for i, q in enumerate(quiz.questions, 1):
            lines.append(f"**Q{i}.** {q.answer}" + (f" — {q.explanation}" if q.explanation else ""))
        return "\n".join(lines)

    # ── Mind map exports ──────────────────────────────────────────────────────

    @staticmethod
    def mindmap_to_json(mindmap: Any) -> str:
        """Export mind map data as JSON (nodes + links)."""
        data = {
            "central": mindmap.central,
            "nodes": [
                {"id": n.id, "label": n.label, "group": n.group, "size": n.size}
                for n in mindmap.nodes
            ],
            "links": [
                {"source": lk.source, "target": lk.target, "label": lk.label}
                for lk in mindmap.links
            ],
            "exported_at": datetime.now().isoformat(),
        }
        return json.dumps(data, indent=2)
