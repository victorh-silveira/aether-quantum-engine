"""Script de sincronizacao de superficie de governanca."""

from pathlib import Path


def main() -> None:
    """Atualiza referencias a pastas antigas em documentacoes e regras."""
    targets = [Path(".agents"), Path(".cursor"), Path("README.md"), Path("docs")]
    for target in targets:
        files = [target] if target.is_file() else list(target.rglob("*.md*"))
        for p in files:
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8")
            new_text = text.replace("indicator/app/src/", "src/indicator/")
            new_text = new_text.replace("app/src/", "src/")
            new_text = new_text.replace("app/scripts/", "scripts/")
            new_text = new_text.replace("app/settings_io.py", "settings_io.py")
            if text != new_text:
                p.write_text(new_text, encoding="utf-8")
                print(f"Updated {p}")


if __name__ == "__main__":
    main()
