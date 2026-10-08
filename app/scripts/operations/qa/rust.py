"""Gates de qualidade para o crate Rust nativo aether_core_rs."""

import os
from pathlib import Path

from scripts.operations.qa.common import require_tool, run_cmd, skip, which


def _manifest_path(root: Path) -> Path:
    """Localiza o arquivo de manifesto Cargo.toml do crate Rust nativo."""
    return root / "native" / "aether_core" / "Cargo.toml"


def _rust_env() -> dict[str, str]:
    """Configura ambiente com suporte estavel ao Python 3.13 via CPython Stable ABI."""
    env = os.environ.copy()
    env.setdefault("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")
    return env


def run_rust(stage: str, root: Path) -> None:
    """Executa os gates de qualidade do Rust: lint, validate, security, test e build."""
    manifest = _manifest_path(root)
    if not manifest.is_file():
        skip("rust", "manifesto Cargo.toml ausente")
        return

    cargo = require_tool("cargo", area="rust")
    if cargo is None:
        return

    rust_env = _rust_env()

    if stage == "lint":
        rustfmt = which("rustfmt")
        if rustfmt:
            run_cmd(
                [cargo, "fmt", "--manifest-path", str(manifest), "--", "--check"],
                cwd=root,
                description="cargo fmt --check",
                env=rust_env,
            )
        else:
            skip("rust", "rustfmt ausente")

        clippy = which("cargo-clippy")
        if clippy or which("clippy-driver"):
            run_cmd(
                [cargo, "clippy", "--manifest-path", str(manifest), "--", "-D", "warnings"],
                cwd=root,
                description="cargo clippy -D warnings",
                env=rust_env,
            )
        else:
            skip("rust", "clippy ausente")
        return

    if stage == "validate":
        run_cmd(
            [cargo, "check", "--manifest-path", str(manifest)],
            cwd=root,
            description="cargo check",
            env=rust_env,
        )
        return

    if stage == "security":
        audit = which("cargo-audit")
        if audit:
            run_cmd(
                [cargo, "audit", "--file", str(manifest.parent / "Cargo.lock")],
                cwd=root,
                description="cargo audit",
                env=rust_env,
            )
        else:
            skip("rust", "cargo-audit ausente")
        return

    if stage == "test":
        run_cmd(
            [cargo, "test", "--manifest-path", str(manifest), "--no-default-features"],
            cwd=root,
            description="cargo test --no-default-features",
            env=rust_env,
        )
        return

    if stage == "build":
        run_cmd(
            [cargo, "build", "--release", "--manifest-path", str(manifest)],
            cwd=root,
            description="cargo build --release",
            env=rust_env,
        )
        return

    skip("rust", f"estagio {stage} nao aplicavel")
