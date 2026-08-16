"""Sincronia da matriz docs/agent-coverage.md com rules, skills e docs."""

from __future__ import annotations

import re

from aether_paths import repo_path


_RULE_RE = re.compile(r"`([a-z0-9_-]+\.mdc)`")
_SKILL_RE = re.compile(r"`([a-z0-9_-]+)`")
_DOC_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+\.md)\)")


def _matrix_rows() -> list[str]:
    text = repo_path("docs", "agent-coverage.md").read_text(encoding="utf-8")
    rows: list[str] = []
    in_matrix = False
    for line in text.splitlines():
        if line.startswith("## Matriz"):
            in_matrix = True
            continue
        if in_matrix and line.startswith("## "):
            break
        if in_matrix and line.startswith("|") and "Superficie" not in line and not line.startswith("|---"):
            rows.append(line)
    return rows


def test_agents_md_exists():
    assert repo_path("AGENTS.md").is_file()


def test_agent_coverage_matrix_rules_and_skills_exist():
    rows = _matrix_rows()
    assert len(rows) >= 10
    rules_dir = repo_path(".cursor", "rules")
    skills_dir = repo_path(".cursor", "skills")
    for row in rows:
        cols = [c.strip() for c in row.strip("|").split("|")]
        assert len(cols) >= 4
        rule_cell, skill_cell = cols[2], cols[3]
        for name in _RULE_RE.findall(rule_cell):
            assert (rules_dir / name).is_file(), f"rule ausente: {name}"
        if "—" in skill_cell or skill_cell in {"-", ""}:
            continue
        for name in _SKILL_RE.findall(skill_cell):
            if name.endswith(".mdc"):
                continue
            skill_path = skills_dir / name / "SKILL.md"
            assert skill_path.is_file(), f"skill ausente: {name}"


def test_agent_coverage_matrix_docs_exist():
    rows = _matrix_rows()
    docs_root = repo_path("docs")
    for row in rows:
        cols = [c.strip() for c in row.strip("|").split("|")]
        doc_cell = cols[1]
        for _label, rel in _DOC_LINK_RE.findall(doc_cell):
            path = docs_root / rel
            assert path.is_file(), f"doc ausente: {rel}"


def test_all_rules_are_always_apply():
    rules_dir = repo_path(".cursor", "rules")
    files = sorted(rules_dir.glob("*.mdc"))
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "alwaysApply: true" in text, f"{path.name} sem alwaysApply: true"
        assert "alwaysApply: false" not in text, f"{path.name} ainda false"
