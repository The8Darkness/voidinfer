"""Fail when a repository-relative Markdown link points at a missing path."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def markdown_files(root: Path, inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        path = root / value
        if path.is_dir():
            files.extend(path.rglob("*.md"))
        elif path.suffix.lower() == ".md" and path.is_file():
            files.append(path)
    return sorted(set(files))


def main() -> int:
    root = Path.cwd().resolve()
    failures: list[str] = []
    for source in markdown_files(root, sys.argv[1:] or ["README.md", "docs"]):
        text = source.read_text(encoding="utf-8")
        for raw in LINK.findall(text):
            target = raw.strip().split()[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            # Mathematical prose such as ``[x](f(y))`` can resemble a link to the
            # deliberately small parser. Repository links in this project contain
            # a path separator or a filename extension.
            if "/" not in target and "\\" not in target and "." not in target:
                continue
            resolved = (source.parent / target).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                failures.append(f"{source.relative_to(root)}: link escapes repository: {raw}")
                continue
            if not resolved.exists():
                failures.append(f"{source.relative_to(root)}: missing: {raw}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Markdown links OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
