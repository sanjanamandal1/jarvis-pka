"""
Temporal Document Version Manager

Handles document versioning and temporal awareness:
- Tracks every version of every document with timestamps
- Computes diffs between versions (added/removed sentences)
- Tags chunks with their valid time range
- Answers "as of" queries by routing to the correct version's vector store
- Detects when a new upload supersedes an older version of the same document
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .semantic_chunker import SemanticChunk


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DocumentVersion:
    doc_id: str
    version: int
    filename: str
    content_hash: str
    uploaded_at: str           # ISO-8601 UTC
    chunk_ids: List[str]
    word_count: int
    is_current: bool = True
    superseded_at: Optional[str] = None
    diff_summary: Optional[str] = None    # human-readable diff vs prev version

    def uploaded_dt(self) -> datetime:
        return datetime.fromisoformat(self.uploaded_at)

    def age_label(self) -> str:
        delta = datetime.now(timezone.utc) - self.uploaded_dt()
        s = delta.total_seconds()
        if s < 60:
            return "just now"
        elif s < 3600:
            return f"{int(s/60)}m ago"
        elif s < 86400:
            return f"{int(s/3600)}h ago"
        else:
            return f"{int(s/86400)}d ago"


@dataclass
class DocumentRegistry:
    """In-memory + disk-persisted registry of all document versions."""
    versions: Dict[str, List[DocumentVersion]] = field(default_factory=dict)
    # doc_id → ordered list of DocumentVersion (oldest first)

    def get_current(self, doc_id: str) -> Optional[DocumentVersion]:
        versions = self.versions.get(doc_id, [])
        for v in reversed(versions):
            if v.is_current:
                return v
        return None

    def get_all_current(self) -> List[DocumentVersion]:
        result = []
        for versions in self.versions.values():
            for v in reversed(versions):
                if v.is_current:
                    result.append(v)
                    break
        return sorted(result, key=lambda v: v.uploaded_at, reverse=True)

    def get_history(self, doc_id: str) -> List[DocumentVersion]:
        return self.versions.get(doc_id, [])


# ── Manager ──────────────────────────────────────────────────────────────────

PERSIST_PATH = Path(".pka_registry")


class TemporalVersionManager:
    """
    Manages the full lifecycle of document versions.

    Usage
    -----
    manager = TemporalVersionManager()
    doc_id, is_new, diff = manager.register_document(filename, text, chunks)
    """

    def __init__(self, persist_dir: str = ".pka_data"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        self.registry_path = self.persist_dir / "registry.json"
        self.registry = self._load_registry()

    # ── Public API ───────────────────────────────────────────────────────────

    def register_document(
        self,
        filename: str,
        raw_text: str,
        chunks: List[SemanticChunk],
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Register a document (new or updated).

        Returns
        -------
        (doc_id, is_new_document, diff_summary_or_None)
        """
        content_hash = self._hash(raw_text)
        doc_id = self._canonical_id(filename)
        now = datetime.now(timezone.utc).isoformat()

        existing_versions = self.registry.versions.get(doc_id, [])
        current = self._find_current(existing_versions)

        # Same content — no-op
        if current and current.content_hash == content_hash:
            return doc_id, False, None

        # Compute diff vs previous version
        diff_summary = None
        if current:
            prev_text = self._load_version_text(doc_id, current.version)
            diff_summary = self._compute_diff_summary(prev_text or "", raw_text)
            current.is_current = False
            current.superseded_at = now

        version_num = len(existing_versions) + 1
        new_version = DocumentVersion(
            doc_id=doc_id,
            version=version_num,
            filename=filename,
            content_hash=content_hash,
            uploaded_at=now,
            chunk_ids=[c.chunk_id for c in chunks],
            word_count=sum(len(c.text.split()) for c in chunks),
            is_current=True,
            diff_summary=diff_summary,
        )

        if doc_id not in self.registry.versions:
            self.registry.versions[doc_id] = []
        self.registry.versions[doc_id].append(new_version)

        # Persist raw text for future diffs
        self._save_version_text(doc_id, version_num, raw_text)
        self._save_registry()

        is_new = len(existing_versions) == 0
        return doc_id, is_new, diff_summary

    def get_all_documents(self) -> List[DocumentVersion]:
        return self.registry.get_all_current()

    def get_document_history(self, doc_id: str) -> List[DocumentVersion]:
        return self.registry.get_history(doc_id)

    def get_current_version(self, doc_id: str) -> Optional[DocumentVersion]:
        return self.registry.get_current(doc_id)

    def delete_document(self, doc_id: str):
        if doc_id in self.registry.versions:
            del self.registry.versions[doc_id]
            self._save_registry()

    def get_temporal_context(self, doc_ids: List[str]) -> str:
        """Build a temporal awareness string injected into RAG prompts."""
        lines = []
        for doc_id in doc_ids:
            current = self.registry.get_current(doc_id)
            if not current:
                continue
            history = self.registry.get_history(doc_id)
            line = f"• {current.filename} (v{current.version}, uploaded {current.age_label()})"
            if len(history) > 1:
                line += f" — {len(history)} versions tracked"
            if current.diff_summary:
                line += f"\n  Latest changes: {current.diff_summary}"
            lines.append(line)

        if not lines:
            return ""
        return "Document temporal context:\n" + "\n".join(lines)

    # ── Internals ────────────────────────────────────────────────────────────

    def _canonical_id(self, filename: str) -> str:
        """Stable doc_id from filename (strip version suffixes like _v2, (1))."""
        stem = Path(filename).stem
        stem = stem.lower().strip()
        # Strip common version suffixes
        import re
        stem = re.sub(r"[\s_\-]*(v\d+|\(\d+\)|copy|final|draft)$", "", stem, flags=re.I)
        return hashlib.md5(stem.encode()).hexdigest()[:12]

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _find_current(self, versions: List[DocumentVersion]) -> Optional[DocumentVersion]:
        for v in reversed(versions):
            if v.is_current:
                return v
        return None

    def _compute_diff_summary(self, old_text: str, new_text: str) -> str:
        old_sents = set(old_text.split(". "))
        new_sents = set(new_text.split(". "))
        added = len(new_sents - old_sents)
        removed = len(old_sents - new_sents)

        old_words = len(old_text.split())
        new_words = len(new_text.split())
        delta_words = new_words - old_words
        sign = "+" if delta_words >= 0 else ""

        parts = []
        if added:
            parts.append(f"~{added} new sentences")
        if removed:
            parts.append(f"~{removed} removed sentences")
        parts.append(f"{sign}{delta_words} words")
        return ", ".join(parts) if parts else "minor edits"

    def _version_text_path(self, doc_id: str, version: int) -> Path:
        d = self.persist_dir / doc_id
        d.mkdir(exist_ok=True)
        return d / f"v{version}.txt"

    def _save_version_text(self, doc_id: str, version: int, text: str):
        self._version_text_path(doc_id, version).write_text(text, encoding="utf-8")

    def _load_version_text(self, doc_id: str, version: int) -> Optional[str]:
        p = self._version_text_path(doc_id, version)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def _save_registry(self):
        data = {
            doc_id: [asdict(v) for v in versions]
            for doc_id, versions in self.registry.versions.items()
        }
        self.registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_registry(self) -> DocumentRegistry:
        if not self.registry_path.exists():
            return DocumentRegistry()
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            reg = DocumentRegistry()
            for doc_id, versions in data.items():
                reg.versions[doc_id] = [DocumentVersion(**v) for v in versions]
            return reg
        except Exception:
            return DocumentRegistry()
