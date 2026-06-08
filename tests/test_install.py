"""Tests for graphify install (Claude Code only)."""
import os
from pathlib import Path
import sys
from unittest.mock import patch
import pytest


def _install(tmp_path, platform):
    from graphify.__main__ import install

    old_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        with patch("graphify.__main__.Path.home", return_value=tmp_path):
            install(platform=platform)
    finally:
        os.chdir(old_cwd)


def test_install_default_claude(tmp_path):
    _install(tmp_path, "claude")
    assert (tmp_path / ".claude" / "skills" / "graphify" / "SKILL.md").exists()


def test_install_windows_alias_maps_to_claude(tmp_path):
    _install(tmp_path, "windows")
    assert (tmp_path / ".claude" / "skills" / "graphify" / "SKILL.md").exists()


def test_install_project_claude_writes_project_scope(tmp_path, monkeypatch, capsys):
    from graphify.__main__ import main
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(sys, "argv", ["graphify", "install", "--project"])
    with patch("graphify.__main__.Path.home", return_value=home):
        main()
    assert (project / ".claude" / "skills" / "graphify" / "SKILL.md").exists()
    assert (project / ".claude" / "CLAUDE.md").exists()
    assert not (home / ".claude" / "skills" / "graphify" / "SKILL.md").exists()
    assert ".claude/skills/graphify/SKILL.md" in (project / ".claude" / "CLAUDE.md").read_text()
    assert "~/.claude/skills/graphify/SKILL.md" not in (project / ".claude" / "CLAUDE.md").read_text()
    assert "git add .claude/" in capsys.readouterr().out


def test_claude_subcommand_project_install_and_uninstall_are_project_scoped(tmp_path, monkeypatch):
    from graphify.__main__ import main
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    user_skill = home / ".claude" / "skills" / "graphify" / "SKILL.md"
    user_skill.parent.mkdir(parents=True)
    user_skill.write_text("user skill")
    monkeypatch.chdir(project)
    with patch("graphify.__main__.Path.home", return_value=home):
        monkeypatch.setattr(sys, "argv", ["graphify", "claude", "install", "--project"])
        main()
        assert (project / ".claude" / "skills" / "graphify" / "SKILL.md").exists()
        assert (project / ".claude" / "CLAUDE.md").exists()
        assert (project / "CLAUDE.md").exists()
        assert user_skill.exists()

        monkeypatch.setattr(sys, "argv", ["graphify", "claude", "uninstall", "--project"])
        main()

    assert user_skill.exists()
    assert not (project / ".claude" / "skills" / "graphify" / "SKILL.md").exists()
    assert not (project / ".claude" / "CLAUDE.md").exists()
    assert not (project / "CLAUDE.md").exists()


def test_install_unknown_platform_exits(tmp_path):
    with pytest.raises(SystemExit):
        _install(tmp_path, "unknown")


def test_claude_install_registers_claude_md(tmp_path):
    """Claude platform install writes CLAUDE.md."""
    _install(tmp_path, "claude")
    assert (tmp_path / ".claude" / "CLAUDE.md").exists()


def test_uninstall_project_without_platform_removes_project_installs(tmp_path, monkeypatch):
    from graphify.__main__ import main
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    user_skill = home / ".claude" / "skills" / "graphify" / "SKILL.md"
    user_skill.parent.mkdir(parents=True)
    user_skill.write_text("user skill")
    monkeypatch.chdir(project)
    with patch("graphify.__main__.Path.home", return_value=home):
        monkeypatch.setattr(sys, "argv", ["graphify", "install", "--project"])
        main()
        monkeypatch.setattr(sys, "argv", ["graphify", "uninstall", "--project"])
        main()
    assert user_skill.exists()
    assert not (project / ".claude" / "skills" / "graphify" / "SKILL.md").exists()
    assert not (project / ".claude" / "CLAUDE.md").exists()
