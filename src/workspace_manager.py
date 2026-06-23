"""
Workspace Manager — named, persistent project workspaces for JARVIS.

Each workspace gets its own isolated directory under .pka_data/workspaces/<slug>/
containing its own FAISS index and version manager state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .logger import get_logger

log = get_logger("workspace_manager")

WORKSPACES_ROOT = Path(".pka_data/workspaces")
DEFAULT_WORKSPACE = "default"


@dataclass
class WorkspaceMeta:
    name: str
    slug: str
    created_at: str
    doc_count: int = 0
    last_used: str = ""

    def age_label(self) -> str:
        try:
            dt = datetime.fromisoformat(self.created_at)
            delta = datetime.now(timezone.utc) - dt.replace(tzinfo=timezone.utc)
            days = delta.days
            if days == 0:
                return "today"
            elif days == 1:
                return "yesterday"
            else:
                return f"{days}d ago"
        except Exception:
            return ""


def _slugify(name: str) -> str:
    """Convert a workspace name to a safe directory slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")[:40]
    return slug or "workspace"


class WorkspaceManager:
    """
    Manages named project workspaces.
    Each workspace = isolated dir with its own KB and version history.
    """

    def __init__(self, root: Path = WORKSPACES_ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        # Ensure default workspace always exists
        if not self._meta_path(DEFAULT_WORKSPACE).exists():
            self._write_meta(WorkspaceMeta(
                name="Default",
                slug=DEFAULT_WORKSPACE,
                created_at=datetime.now(timezone.utc).isoformat(),
            ))

    # ── Public API ────────────────────────────────────────────────────────────

    def list_workspaces(self) -> List[WorkspaceMeta]:
        """Return all workspaces sorted by last_used desc."""
        results = []
        for meta_file in self.root.glob("*/meta.json"):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                results.append(WorkspaceMeta(**data))
            except Exception:
                pass
        results.sort(key=lambda w: w.last_used or w.created_at, reverse=True)
        return results

    def create_workspace(self, name: str) -> WorkspaceMeta:
        """Create a new workspace. Returns existing one if slug already exists."""
        slug = _slugify(name)
        path = self.root / slug
        if path.exists():
            log.info(f"Workspace '{slug}' already exists, returning existing.")
            return self._read_meta(slug)
        path.mkdir(parents=True, exist_ok=True)
        meta = WorkspaceMeta(
            name=name,
            slug=slug,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._write_meta(meta)
        log.info(f"Created workspace: {name} ({slug})")
        return meta

    def delete_workspace(self, slug: str) -> bool:
        """Delete a workspace and all its data. Cannot delete 'default'."""
        if slug == DEFAULT_WORKSPACE:
            log.warning("Cannot delete the default workspace.")
            return False
        path = self.root / slug
        if not path.exists():
            return False
        import shutil
        shutil.rmtree(path)
        log.info(f"Deleted workspace: {slug}")
        return True

    def get_workspace_path(self, slug: str) -> Path:
        """Return the root path for a workspace, creating it if needed."""
        path = self.root / slug
        path.mkdir(parents=True, exist_ok=True)
        return path

    def touch_workspace(self, slug: str, doc_count: int = 0):
        """Update last_used timestamp and doc_count for a workspace."""
        try:
            meta = self._read_meta(slug)
            meta.last_used = datetime.now(timezone.utc).isoformat()
            meta.doc_count = doc_count
            self._write_meta(meta)
        except Exception:
            pass

    def workspace_exists(self, slug: str) -> bool:
        return self._meta_path(slug).exists()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _meta_path(self, slug: str) -> Path:
        return self.root / slug / "meta.json"

    def _write_meta(self, meta: WorkspaceMeta):
        path = self.root / meta.slug
        path.mkdir(parents=True, exist_ok=True)
        self._meta_path(meta.slug).write_text(
            json.dumps(asdict(meta), indent=2), encoding="utf-8"
        )

    def _read_meta(self, slug: str) -> WorkspaceMeta:
        data = json.loads(self._meta_path(slug).read_text(encoding="utf-8"))
        return WorkspaceMeta(**data)
