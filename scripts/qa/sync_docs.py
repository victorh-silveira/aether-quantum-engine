"""Script de sincronizacao de superficie de governanca."""

from __future__ import annotations

import re
from pathlib import Path


_REPLACEMENTS = (
    (r"indicator/app/src/", "src/indicator/"),
    (r"app/src/", "src/"),
    (r"app/scripts/", "scripts/"),
    (r"app/settings_io\.py", "src/shared/settings_io.py"),
    (r"\.\./settings_io\.py", "../src/shared/settings_io.py"),
    (r"(?<![/\w])settings_io\.py", "src/shared/settings_io.py"),
    (r"(?<![/\w])aether_paths\.py", "src/shared/paths.py"),
    (r"(?<![/\w])aether_asyncio\.py", "src/shared/asyncio_runner.py"),
    (r"(?<![/\w])run_indicator\.py", "src/indicator/__main__.py"),
    (r"(?<![/\w])train\.py", "src/application/services/orchestrator/train_cli.py"),
)


def main() -> None:
    """Atualiza referencias a pastas antigas em documentacoes e regras."""
    targets = [Path(".agents"), Path("README.md"), Path("docs")]
    for target in targets:
        files = [target] if target.is_file() else list(target.rglob("*.md*"))
        for p in files:
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8")
            new_text = text
            for pattern, repl in _REPLACEMENTS:
                new_text = re.sub(pattern, repl, new_text)
            if text != new_text:
                p.write_text(new_text, encoding="utf-8")
                print(f"Updated {p}")


if __name__ == "__main__":
    main()
