"""Tests for graphify install --platform routing."""
import os
from pathlib import Path
import sys
from unittest.mock import patch
import pytest


PLATFORMS = {
    "claude": (".claude/skills/graphify/SKILL.md",),
    "codebuddy": (".codebuddy/skills/graphify/SKILL.md",),
    "codex": (".codex/skills/graphify/SKILL.md",),
    "opencode": (".config/opencode/skills/graphify/SKILL.md",),
    "kilo": (
        ".config/kilo/skills/graphify/SKILL.md",
        ".config/kilo/command/graphify.md",
    ),
    "claw": (".openclaw/skills/graphify/SKILL.md",),
    "droid": (".factory/skills/graphify/SKILL.md",),
    "trae": (".trae/skills/graphify/SKILL.md",),
    "trae-cn": (".trae-cn/skills/graphify/SKILL.md",),
    "windows": (".claude/skills/graphify/SKILL.md",),
}


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
    """Claude platform install writes CLAUDE.md; others do not."""
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


def _agents_install(tmp_path, platform):
    from graphify.__main__ import _agents_install as _install_fn

    _install_fn(tmp_path, platform)


def _agents_uninstall(tmp_path, platform=""):
    from graphify.__main__ import _agents_uninstall as _uninstall_fn

    _uninstall_fn(tmp_path, platform=platform)


def _kilo_install(project_dir, home_dir):
    from graphify.__main__ import _kilo_install as _install_fn

    with patch("graphify.__main__.Path.home", return_value=home_dir):
        _install_fn(project_dir)


def _kilo_uninstall(project_dir, home_dir):
    from graphify.__main__ import _kilo_uninstall as _uninstall_fn

    with patch("graphify.__main__.Path.home", return_value=home_dir):
        _uninstall_fn(project_dir)


def test_agents_uninstall_no_op_when_not_installed(tmp_path, capsys):
    _agents_uninstall(tmp_path)
    out = capsys.readouterr().out
    assert "nothing to do" in out


# --- OpenCode plugin tests ---


def test_cursor_install_writes_rule(tmp_path):
    """cursor install writes .cursor/rules/graphify.mdc."""
    from graphify.__main__ import _cursor_install

    _cursor_install(tmp_path)
    rule = tmp_path / ".cursor" / "rules" / "graphify.mdc"
    assert rule.exists()
    content = rule.read_text()
    assert "alwaysApply: true" in content
    assert "graphify-out/GRAPH_REPORT.md" in content


def test_cursor_install_idempotent(tmp_path):
    """cursor install does not overwrite an existing rule file."""
    from graphify.__main__ import _cursor_install

    _cursor_install(tmp_path)
    rule = tmp_path / ".cursor" / "rules" / "graphify.mdc"
    original = rule.read_text()
    _cursor_install(tmp_path)
    assert rule.read_text() == original


def test_cursor_uninstall_removes_rule(tmp_path):
    """cursor uninstall removes the rule file."""
    from graphify.__main__ import _cursor_install, _cursor_uninstall

    _cursor_install(tmp_path)
    _cursor_uninstall(tmp_path)
    rule = tmp_path / ".cursor" / "rules" / "graphify.mdc"
    assert not rule.exists()


def test_cursor_uninstall_noop_if_not_installed(tmp_path):
    """cursor uninstall does nothing if rule was never written."""
    from graphify.__main__ import _cursor_uninstall

    _cursor_uninstall(tmp_path)  # should not raise


# ── Gemini CLI ────────────────────────────────────────────────────────────────


def test_gemini_uninstall_noop_if_not_installed(tmp_path):
    from graphify.__main__ import gemini_uninstall

    gemini_uninstall(tmp_path)  # should not raise


