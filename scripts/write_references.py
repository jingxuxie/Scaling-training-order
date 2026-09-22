"""Render the checked bibliography from docs/references.json.

Edit that JSON file to update a record. Sources are in docs/reference_audit.json.
"""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def sentence(value):
    return value if value.endswith((".", "?", "!")) else value + "."


def main():
    refs = json.loads((ROOT / "docs/references.json").read_text(encoding="utf-8"))
    keys = [ref["key"] for ref in refs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate bibliography key")
    cited = set()
    for source in (ROOT / "paper").glob("*.tex"):
        if source.name == "references.tex":
            continue
        for group in re.findall(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}",
                                source.read_text(encoding="utf-8")):
            cited.update(key.strip() for key in group.split(","))
    if cited != set(keys):
        raise ValueError(f"Missing entries: {cited-set(keys)}; uncited entries: {set(keys)-cited}")
    lines = [rf"\begin{{thebibliography}}{{{len(refs)}}}\label{{sec:references}}"]
    for ref in sorted(refs, key=lambda item: item["key"]):
        if str(ref["year"]) not in ref["label"]:
            raise ValueError(f"Citation year disagrees with record: {ref['key']}")
        lines.append(
            rf"\bibitem[{ref['label']}]{{{ref['key']}}}" + "\n"
            + f"{sentence(ref['authors'])} {sentence(ref['title'])} "
            + rf"\emph{{{ref['venue']}}}, {ref['year']}. \url{{{ref['url']}}}." + "\n"
        )
    lines.append(r"\end{thebibliography}")
    (ROOT / "paper/references.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(refs)} references; every citation resolves and every entry is cited.")


if __name__ == "__main__":
    main()
