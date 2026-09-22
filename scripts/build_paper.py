"""Build the paper and enforce its page count, references, and pinned style.

Requires pdflatex. This performs layout checks, not proof verification.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STYLE_GIT_BLOB = "f61ad7efce0855557694078c0945e6c33feb8236"


def git_blob_sha1(data: bytes) -> str:
    """Hash as a Git blob, without needing a repository or a network call."""
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main(expected_main_pages: int = 9) -> None:
    if expected_main_pages < 1:
        raise ValueError("expected_main_pages must be positive")
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("Install pdflatex and the LaTeX packages in paper/main.tex.")
    paper = ROOT / "paper"
    actual_style = git_blob_sha1((paper / "iclr2027_conference.sty").read_bytes())
    if actual_style != STYLE_GIT_BLOB:
        raise RuntimeError(
            "Style differs from the pinned official ICLR 2027 source. "
            "Check provenance before deliberately updating the pin."
        )
    with (paper / "build.log").open("w", encoding="utf-8") as log:
        for _ in range(3):
            subprocess.run(
                [executable, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=paper, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
    text = (paper / "main.log").read_text(errors="replace")
    if re.search(r"undefined references|(?:Citation|Reference).*?undefined", text, re.S):
        raise RuntimeError("Unresolved citation/reference: inspect paper/main.log.")
    if re.search(r"Overfull \\[hv]box", text):
        raise RuntimeError("An overflowing box remains: inspect paper/main.log.")
    aux = (paper / "main.aux").read_text(errors="replace")
    marker = re.search(r"\\newlabel\{main:end\}\{\{[^}]*\}\{(\d+)\}", aux)
    if marker is None:
        raise RuntimeError("Missing main:end label; cannot audit main-text length.")
    pages = int(marker.group(1))
    if pages != expected_main_pages:
        raise RuntimeError(f"Main text ends on page {pages}; expected {expected_main_pages}.")
    total = re.search(r"Output written on main.pdf \((\d+) pages", text)
    print(paper / "main.pdf")
    print(f"Main text: {pages} pages. Total: {total.group(1) if total else 'see PDF'} pages.")
    print("Pinned official style verified; no unresolved references or overflowing boxes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-main-pages", type=int, default=9)
    main(parser.parse_args().expected_main_pages)
