"""Link check of the documentation (exit code 1 on any problem).

Checks, over every Markdown file and the Markdown cells of every notebook:
  - relative links and images resolve to an existing file or folder;
  - ``#anchors`` exist in the target Markdown file (GitHub slug rules);
  - repository paths written as inline code (e.g. ``results/metrics.csv``) exist, so a renamed file cannot
    survive in prose; globs must match at least one file;
  - every PNG in figures/ and screenshots/ is embedded in at least one Markdown file.

Usage: ``python tools/check_links.py``
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".scratch", ".demo", ".reproduce", ".venv", "__pycache__", ".pytest_cache",
             ".ipynb_checkpoints"}
LINK_RE = re.compile(r"!?\[(?:[^\]\[]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_RE = re.compile(r"""<(?:img|a)\s[^>]*(?:src|href)=["']([^"']+)["']""")
REFDEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.M)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.M)
ANCHOR_TAG_RE = re.compile(r"""<a\s+(?:name|id)=["']([^"']+)["']""")
FENCE_RE = re.compile(r"```.*?```", re.S)
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
TOP_DIRS = ("results", "src", "docs", "figures", "screenshots", "media", "weights", "protocols", "tools", "tests",
            "examples", "notebooks", ".github")
REPO_PATH_RE = re.compile(r"^(?:\.\./)*(?:" + "|".join(re.escape(d) for d in TOP_DIRS) + r")/[\w.*/{}-]*$")
IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def slug(text: str) -> str:
    """GitHub heading anchor: lower case, drop punctuation except '-' and '_', spaces to '-'."""
    text = re.sub(r"`", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    out = []
    for ch in text.strip().lower():
        cat = unicodedata.category(ch)
        if ch == " ":
            out.append("-")
        elif ch in "-_" or cat[0] in "LN" or cat == "Mn":
            out.append(ch)
    return "".join(out)


_anchor_cache: dict[Path, set[str]] = {}


def anchors(path: Path) -> set[str]:
    if path not in _anchor_cache:
        text = path.read_text(encoding="utf-8")
        seen: dict[str, int] = {}
        out = set(ANCHOR_TAG_RE.findall(text))
        for _, title in HEADING_RE.findall(FENCE_RE.sub("", text)):
            s = slug(title)
            n = seen.get(s, 0)
            out.add(s if n == 0 else f"{s}-{n}")
            seen[s] = n + 1
        _anchor_cache[path] = out
    return _anchor_cache[path]


def doc_sources() -> list[tuple[Path, str]]:
    """(file, markdown text) for every .md file and every notebook's Markdown cells."""
    out = []
    for p in sorted(REPO.rglob("*")):
        if SKIP_DIRS.intersection(p.relative_to(REPO).parts) or not p.is_file():
            continue
        if p.suffix == ".md":
            out.append((p, p.read_text(encoding="utf-8")))
        elif p.suffix == ".ipynb":
            nb = json.loads(p.read_text(encoding="utf-8"))
            md = [("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
                  for c in nb.get("cells", []) if c.get("cell_type") == "markdown"]
            out.append((p, "\n\n".join(md)))
    return out


def resolve(base: Path, rel: str) -> list[Path]:
    """Existing paths `rel` can mean from `base` (the document's folder) or from the repository root."""
    hits = []
    for root in (base, REPO):
        if any(ch in rel for ch in "*{"):
            pattern = re.sub(r"\{[^}]*\}", "*", rel.lstrip("./") if root is REPO else rel)
            try:
                hits += list(root.glob(pattern))
            except (ValueError, NotImplementedError):
                pass
        elif (root / rel).exists():
            hits.append((root / rel).resolve())
    return hits


def main() -> int:
    broken: list[str] = []
    embedded: set[Path] = set()
    n_links = n_paths = 0
    for doc, text in doc_sources():
        rel_doc = doc.relative_to(REPO).as_posix()
        prose = FENCE_RE.sub("", text)
        no_code = CODE_SPAN_RE.sub("", prose)
        for t in LINK_RE.findall(no_code) + HTML_RE.findall(no_code) + REFDEF_RE.findall(no_code):
            if re.match(r"^[a-z][a-z0-9+.-]*:", t) or t.startswith("//"):
                continue
            n_links += 1
            path_part, _, frag = t.partition("#")
            target = doc if not path_part else (doc.parent / path_part).resolve()
            if not target.exists():
                broken.append(f"{rel_doc}: {t} (missing)")
                continue
            if target.suffix.lower() in IMG_EXT:
                embedded.add(target)
            if frag and target.suffix == ".md" and frag not in anchors(target):
                broken.append(f"{rel_doc}: {t} (no anchor)")
        for span in CODE_SPAN_RE.findall(prose):
            cand = span.strip().split()[0].rstrip(".,;:)")
            if not REPO_PATH_RE.match(cand):
                continue
            n_paths += 1
            if not resolve(doc.parent, cand):
                broken.append(f"{rel_doc}: `{cand}` (path in code span does not exist)")
    orphans = sorted(p.relative_to(REPO).as_posix()
                     for d in ("figures", "screenshots") for p in (REPO / d).glob("*.png")
                     if p.resolve() not in embedded)
    broken += [f"{o}: not embedded in any Markdown file" for o in orphans]
    print(f"links checked: {n_links}; code-span paths checked: {n_paths}; problems: {len(broken)}")
    for b in broken:
        print("  ", b)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
