"""Tests for WorkspaceManager."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.workspace_manager import WorkspaceManager


@pytest.fixture
def wm(tmp_path):
    return WorkspaceManager(root=tmp_path / "workspaces")


def test_default_workspace_created(wm):
    workspaces = wm.list_workspaces()
    slugs = [w.slug for w in workspaces]
    assert "default" in slugs


def test_create_workspace(wm):
    meta = wm.create_workspace("ML Research")
    assert meta.slug == "ml_research"
    assert meta.name == "ML Research"
    assert wm.workspace_exists("ml_research")


def test_list_workspaces_includes_new(wm):
    wm.create_workspace("Legal Docs")
    names = [w.name for w in wm.list_workspaces()]
    assert "Legal Docs" in names


def test_delete_workspace(wm):
    wm.create_workspace("Temp")
    result = wm.delete_workspace("temp")
    assert result is True
    assert not wm.workspace_exists("temp")


def test_cannot_delete_default(wm):
    result = wm.delete_workspace("default")
    assert result is False
    assert wm.workspace_exists("default")


def test_get_workspace_path(wm):
    path = wm.get_workspace_path("default")
    assert path.exists()
