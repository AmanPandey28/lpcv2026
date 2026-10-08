#!/usr/bin/env python3
"""Reject sensitive/unnecessary files and broken local Markdown links before publishing.

This is a deterministic hygiene check, not a substitute for a full secret scanner.
It scans tracked files even when .gitignore would otherwise hide them.
"""

import argparse
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_NAMES = {
    ".agents",
    ".codex",
    ".aws",
    ".qai_hub",
    "agents.md",
    "claude.md",
    "gemini.md",
    "client.ini",
    "credentials",
    ".env",
}
ASSET_DIRS = {"data", "models", "artifacts", "external", "logs", "runs", "cache"}
BLOCKED_SUFFIXES = {
    ".pdf",
    ".tex",
    ".docx",
    ".pptx",
    ".ipynb",
    ".onnx",
    ".dlc",
    ".pt",
    ".pth",
    ".safetensors",
    ".npy",
    ".npz",
    ".pkl",
    ".zip",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".log",
    ".encodings",
}
SENSITIVE_CONTENT = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"api[_-]?token\s*[:=]\s*['\"][A-Za-z0-9]{20,}['\"]", re.I),
    re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
]


def public_files(include_untracked=False):
    args = ["git", "ls-files", "-z", "--cached"]
    if include_untracked:
        args += ["--others", "--exclude-standard"]
    output = subprocess.check_output(args, cwd=ROOT)
    return sorted(set(Path(p.decode()) for p in output.split(b"\0") if p))


def audit(paths):
    errors = []
    for relative in paths:
        path = ROOT / relative
        if not path.exists():  # A tracked deletion in a local pre-commit run.
            continue
        parts = [part.lower() for part in relative.parts]
        if (
            set(parts) & (PRIVATE_NAMES | ASSET_DIRS)
            or path.suffix.lower() in BLOCKED_SUFFIXES
            or any(
                any(
                    word in part
                    for word in ("interview", "resume", "book", "engineering_record")
                )
                for part in parts
            )
            or any(part.startswith(".env.") for part in parts)
        ):
            errors.append(f"Forbidden public file: {relative}")
            continue
        if path.is_symlink():
            errors.append(f"Public symlink requires manual review: {relative}")
            continue
        if path.stat().st_size > 1024 * 1024:
            errors.append(f"Oversized public file: {relative}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"Unexpected binary: {relative}")
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if any(pattern.search(line) for pattern in SENSITIVE_CONTENT):
                # Report only location: never echo possible credentials.
                errors.append(f"Sensitive content: {relative}:{number}")
        if path.suffix.lower() == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", content):
                if target.startswith(("http:", "https:", "mailto:", "#")):
                    continue
                link = target.split("#", 1)[0]
                if link and not (path.parent / link).exists():
                    errors.append(f"Broken local link: {relative} -> {target}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-untracked", action="store_true")
    args = parser.parse_args()
    paths = public_files(args.include_untracked)
    errors = audit(paths)
    for error in errors:
        print(error)
    if errors:
        return 1
    print(f"Public-file audit passed: {len(paths)} indexed/candidate paths checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
