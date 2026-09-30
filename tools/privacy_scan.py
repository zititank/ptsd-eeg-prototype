"""Privacy scan of the public repository.

Fails (exit code 1) on any *hit*; *warnings* are printed for manual review but do not fail.

Hits:
  - subject-code patterns of the confidential dataset (one or two capitals, three digits, an optional
    lower-case letter and a two-digit age suffix; or four capitals and one digit);
  - any token whose salted hash is in the private ID list (``PTSD_ID_HASHES`` -> JSON with ``salt`` and
    ``hashes``; the check is skipped with a notice when the variable is unset, e.g. in CI);
  - any term of the private confidentiality denylist (``PTSD_CONFIDENTIAL_TERMS`` -> text file, one regex per
    line: organiser, event, data provider, device brand, dataset folder names, dates, participant descriptions;
    skipped with a notice when unset; the file must live outside the repository);
  - any Cyrillic letter (the repository is English-only; Cyrillic text would be a copied folder or file name);
  - participant numbers of the public Mendeley dataset (``EEG_sb<N>``, ``sb<N>``, ``NF <N>``, ``MI <N>``,
    ``P<NN>``);
  - recording dates ``dd.mm.yy[yy]`` other than the allowlisted constants;
  - EDF headers, ``.edf/.bdf/.pkl/.mat/...`` files, undecodable binaries, files larger than 5 MB;
  - absolute home-directory paths (user names) anywhere, including notebook outputs;
  - CSV group-size columns (``n``, ``n_folders``, ``n_scorable``, ...) with a value of 1-4.

Warnings:
  - upper-case tokens of 4-5 letters (optionally one digit) that are not in the acronym whitelist.

Usage: ``python tools/privacy_scan.py [--root DIR] [--hashes FILE] [--terms FILE] [--quiet-warnings]``
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
SKIP_DIRS = {".git", ".scratch", ".reproduce", ".demo", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv",
             "per_subject_DO_NOT_COMMIT"}
FORBIDDEN_EXT = {".edf", ".bdf", ".rec", ".pkl", ".pickle", ".joblib", ".mat", ".npy", ".h5", ".hdf5",
                 ".parquet", ".feather", ".xlsx", ".xls", ".sav", ".db", ".sqlite"}
BINARY_OK_EXT = {".npz", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".svg", ".mp4"}
EDF_MAGIC = b"0       "

ID_PATTERNS = [re.compile(r"(?<![\w-])[A-Z\u0410-\u042f\u0401]{1,2}\d{3}[a-z\u0430-\u044f\u0451]?_\d{2}(?!\w)"),
               re.compile(r"(?<![\w-])[A-Z]{4}\d(?![\w-])")]
MENDELEY_PATTERNS = [re.compile(r"EEG_sb\d+"), re.compile(r"(?<![\w-])sb\d{1,3}(?!\w)"),
                     re.compile(r"(?<![\w-])(?:NF|MI)[ _-]?\d{1,2}(?!\d)"),
                     re.compile(r"(?<![\w-])P\d{2}(?!\w)")]
ALLOWED_CONTEXT = [re.compile(r"expected n is \d+ \((?:NF|MI) \d+, (?:NF|MI) \d+\)")]
"""Spans where Mendeley-like patterns are group sizes, not participant numbers."""

DATE_RE = re.compile(r"(?<![\d.])(\d{2})\.(\d{2})\.(\d{2}|\d{4})(?![\d.])")
ALLOWED_DATES = {"01.01.00", "02.01.00"}
"""Synthetic EDF start dates (default and the generic control-batch-B marker of `ptsd_eeg.layout`)."""
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]+")

HOME_RE = re.compile(r"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9_.-]+")
ACRONYM_RE = re.compile(r"(?<![\w-])[A-Z]{4,5}\d?(?![\w-])")
ACRONYMS = {
    # study design and cohort labels
    "PTSD", "CTRL", "SOMA", "LOSO", "NOTE", "TODO", "ONLY", "NEVER", "WARN", "PASS", "FAIL", "TRUE", "FALSE",
    "NONE", "NULL", "OOF", "PROXY", "LEAK", "BLOCK", "FINAL", "MAJOR", "MINOR", "REST",
    # merge markers in the docs
    "STORY", "MEDIA", "TEAM",
    # methods / statistics
    "CLR", "LDA", "IAF", "AUC", "HOLM", "BCA", "SHAP", "ROC", "IQR", "BOOT", "SMOTE",
    # EEG / file formats / devices
    "EDF", "BDF", "ICA", "EEG", "GTEC", "FLEX", "MNE",
    # plotting constants in tools/make_animations.py
    "MUTED", "STEP", "PANEL", "SPEED", "MOVE", "NAME",
    # software
    "JSON", "YAML", "HTML", "HTTP", "HTTPS", "UTF", "ASCII", "NUMPY", "SCIPY", "PANDAS", "MAKE", "PATH",
    "HOME", "PYTHON", "SHA", "CUDA", "BLAS", "LAPACK", "POSIX", "LOKY", "README", "MIT", "API", "CLI",
    "DATA", "UNIX", "NFKC",
    # bibliography
    "PMID", "JMLR", "PMLR", "PNAS", "CORE",
    # module constants of this repository
    "REPO", "ROOT", "FILE", "SEED", "SPEC", "GRID", "FLOOR", "JOBS", "PHONY",
}
CAPS_EXEMPT = {"LICENSE"}
"""Standard license text written in capitals; the ID, hash and date checks still apply."""


@dataclass
class Finding:
    level: str
    path: str
    line: int
    check: str
    detail: str

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"{self.level:4s} {loc}  [{self.check}] {self.detail}"


def mask(tok: str) -> str:
    """Show a token only as its first character and length, so the report itself leaks nothing."""
    return f"{tok[:1]}***({len(tok)})"


# Homoglyphs folded before hashing, matching tools/export_public.py of the private repository.
_FOLD = str.maketrans("\u0410\u0412\u0421\u0415\u041d\u041a\u041c\u041e\u0420\u0422\u0425\u0423"
                      "\u0430\u0432\u0441\u0435\u043e\u043a\u043c\u0440\u0445\u0443",
                      "ABCEHKMOPTXYabceokmpxy")


def normalise_token(tok: str) -> str:
    return unicodedata.normalize("NFKC", tok).translate(_FOLD).upper()


class HashList:
    """Salted SHA-256 hashes of the real subject codes; the file lives outside the repository."""

    def __init__(self, path: Path | None):
        self.enabled = bool(path and path.is_file())
        self.path = path
        self.salt, self.hashes = "", frozenset()
        if self.enabled:
            d = json.loads(path.read_text(encoding="utf-8"))
            self.salt, self.hashes = d["salt"], frozenset(d["hashes"])

    def candidates(self, token: str) -> set[str]:
        parts = token.split("_")
        out = {token, *parts}
        out.update("_".join(parts[i:i + 2]) for i in range(len(parts) - 1))
        return {c for c in out if len(c) >= 3}

    def hits(self, text: str) -> list[str]:
        if not self.enabled:
            return []
        found = []
        for tok in set(re.findall(r"\w+", text)):
            for c in self.candidates(tok):
                if hashlib.sha256((self.salt + normalise_token(c)).encode()).hexdigest() in self.hashes:
                    found.append(c)
        return found


class TermList:
    """Confidentiality denylist (one regex per line, `#` comments); the file lives outside the repository."""

    def __init__(self, path: Path | None):
        self.enabled = bool(path and path.is_file())
        self.path = path
        self.patterns: list[re.Pattern] = []
        if self.enabled:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    self.patterns.append(re.compile(line.strip(), re.IGNORECASE))

    def hits(self, text: str) -> list[tuple[int, str]]:
        """(term index, matched text) for every denylisted term in `text` (raw and homoglyph-folded)."""
        if not self.enabled:
            return []
        norm = unicodedata.normalize("NFKC", text)
        variants = {norm, norm.translate(_FOLD)}
        return [(k, m.group(0)) for k, pat in enumerate(self.patterns, 1) for v in variants
                for m in pat.finditer(v)]


NO_TERMS = TermList(None)


def allowed_spans(line: str) -> list[tuple[int, int]]:
    return [m.span() for pat in ALLOWED_CONTEXT for m in pat.finditer(line)]


def inside(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(a <= span[0] and span[1] <= b for a, b in spans)


def valid_date(m: re.Match) -> bool:
    d, mo = int(m.group(1)), int(m.group(2))
    return 1 <= d <= 31 and 1 <= mo <= 12


def scan_text(rel: str, text: str, hashes: HashList, out: list[Finding], where: str = "",
              terms: TermList = NO_TERMS) -> None:
    """Line-level checks shared by source files, CSV cells, file names and notebook outputs."""
    tag = f" ({where})" if where else ""
    for i, line in enumerate(text.splitlines(), 1):
        spans = allowed_spans(line)
        for k, tok in sorted(set(terms.hits(line))):
            out.append(Finding("HIT", rel, i, "confidential-term", f"term #{k}: {mask(tok)}" + tag))
        for m in CYRILLIC_RE.finditer(line):
            out.append(Finding("HIT", rel, i, "cyrillic", mask(m.group(0)) + tag))
        for pat in ID_PATTERNS:
            for m in pat.finditer(line):
                out.append(Finding("HIT", rel, i, "id-pattern", mask(m.group(0)) + tag))
        for tok in hashes.hits(line):
            out.append(Finding("HIT", rel, i, "id-hash", mask(tok) + tag))
        for pat in MENDELEY_PATTERNS:
            for m in pat.finditer(line):
                if not inside(m.span(), spans):
                    out.append(Finding("HIT", rel, i, "mendeley-participant", m.group(0) + tag))
        for m in DATE_RE.finditer(line):
            if valid_date(m) and m.group(0) not in ALLOWED_DATES:
                out.append(Finding("HIT", rel, i, "date", m.group(0) + tag))
        for m in HOME_RE.finditer(line):
            out.append(Finding("HIT", rel, i, "home-path", mask(m.group(0)) + tag))
        if "Startdate" in line and re.search(r"Startdate \d{2}-[A-Z]{3}-\d{4}", line):
            out.append(Finding("HIT", rel, i, "edf-header", "EDF+ Startdate field" + tag))
        if rel in CAPS_EXEMPT:
            continue
        for m in ACRONYM_RE.finditer(line):
            tok = m.group(0)
            if tok.rstrip("0123456789") not in ACRONYMS and tok not in ACRONYMS:
                out.append(Finding("WARN", rel, i, "caps-token", tok + tag))


GROUP_SIZE_COL = re.compile(r"^(n|n_(folders|scorable|all|subjects|ptsd|control|primary|train|training|"
                            r"training_subjects|soma|soma_elderly|ctrl_c|O1_scorable|gtec|flex|members|bags))$")


def scan_csv(rel: str, text: str, out: list[Finding]) -> None:
    """Group sizes of 1-4 in aggregate tables (the n >= 5 rule)."""
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return
    cols = [j for j, c in enumerate(rows[0]) if GROUP_SIZE_COL.match(c.strip())]
    for i, row in enumerate(rows[1:], 2):
        for j in cols:
            if j < len(row):
                try:
                    v = float(row[j])
                except ValueError:
                    continue
                if 1 <= v <= 4:
                    out.append(Finding("HIT", rel, i, "small-cell", f"{rows[0][j]}={row[j]}"))


def scan_notebook(rel: str, text: str, hashes: HashList, out: list[Finding], terms: TermList = NO_TERMS) -> None:
    """Notebook sources and text outputs (images are skipped; base64 is not text)."""
    nb = json.loads(text)
    for k, cell in enumerate(nb.get("cells", [])):
        scan_text(rel, "".join(cell.get("source", [])), hashes, out, where=f"cell {k} source", terms=terms)
        for o in cell.get("outputs", []):
            chunks = []
            if "text" in o:
                chunks.append("".join(o["text"]))
            for mime, val in o.get("data", {}).items():
                if mime.startswith("text/") or mime == "application/json":
                    chunks.append("".join(val) if isinstance(val, list) else json.dumps(val))
            if o.get("output_type") == "error":
                chunks.append("\n".join(o.get("traceback", [])))
                out.append(Finding("HIT", rel, 0, "notebook-error", f"cell {k} has an error output"))
            for c in chunks:
                scan_text(rel, c, hashes, out, where=f"cell {k} output", terms=terms)
    scan_text(rel, json.dumps(nb.get("metadata", {}), ensure_ascii=False), hashes, out, where="metadata", terms=terms)


def scan_npz(rel: str, path: Path, hashes: HashList, out: list[Finding], terms: TermList = NO_TERMS) -> None:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    except zipfile.BadZipFile:
        out.append(Finding("HIT", rel, 0, "binary", "unreadable .npz"))
        return
    scan_text(rel, "\n".join(names), hashes, out, where="npz member names", terms=terms)
    if not rel.startswith("weights/"):
        out.append(Finding("WARN", rel, 0, "npz", "array file outside weights/"))


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("per_subject"))
        for f in sorted(filenames):
            yield Path(dirpath) / f


def scan(root: Path, hashes: HashList, terms: TermList = NO_TERMS) -> list[Finding]:
    out: list[Finding] = []
    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        scan_text(rel, rel, hashes, out, where="file name", terms=terms)
        ext = path.suffix.lower()
        size = path.stat().st_size
        if size > MAX_BYTES:
            out.append(Finding("HIT", rel, 0, "size", f"{size / 1e6:.1f} MB > 5 MB"))
        if ext in FORBIDDEN_EXT:
            out.append(Finding("HIT", rel, 0, "file-type", f"{ext} files are not allowed"))
            continue
        raw = path.read_bytes()
        if raw.startswith(EDF_MAGIC) and len(raw) >= 256:
            out.append(Finding("HIT", rel, 0, "edf-header", "file starts with an EDF header"))
            continue
        if ext == ".npz":
            scan_npz(rel, path, hashes, out, terms)
            continue
        if ext in BINARY_OK_EXT - {".svg"}:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            out.append(Finding("HIT", rel, 0, "binary", "undecodable binary file"))
            continue
        if ext == ".ipynb":
            scan_notebook(rel, text, hashes, out, terms)
            continue
        scan_text(rel, text, hashes, out, terms=terms)
        if ext == ".csv":
            scan_csv(rel, text, out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--hashes", type=Path, default=None,
                    help="salted ID-hash JSON (default: $PTSD_ID_HASHES; skipped when absent)")
    ap.add_argument("--terms", type=Path, default=None,
                    help="confidentiality denylist (default: $PTSD_CONFIDENTIAL_TERMS; skipped when absent)")
    ap.add_argument("--quiet-warnings", action="store_true")
    a = ap.parse_args(argv)
    hpath = a.hashes or (Path(os.environ["PTSD_ID_HASHES"]) if os.environ.get("PTSD_ID_HASHES") else None)
    tpath = a.terms or (Path(os.environ["PTSD_CONFIDENTIAL_TERMS"]) if os.environ.get("PTSD_CONFIDENTIAL_TERMS")
                        else None)
    for what, pth in [("ID-hash list", hpath), ("confidentiality denylist", tpath)]:
        if pth is not None and pth.resolve().is_relative_to(a.root.resolve()):
            print(f"the {what} must live outside the repository", file=sys.stderr)
            return 2
    hashes = HashList(hpath)
    terms = TermList(tpath)
    findings = scan(a.root.resolve(), hashes, terms)
    hits = [f for f in findings if f.level == "HIT"]
    warns = [f for f in findings if f.level == "WARN"]
    for f in hits + ([] if a.quiet_warnings else warns):
        print(f)
    state = f"on ({len(hashes.hashes)} hashes)" if hashes.enabled else "skipped (PTSD_ID_HASHES not set)"
    tstate = (f"on ({len(terms.patterns)} terms)" if terms.enabled
              else "skipped (PTSD_CONFIDENTIAL_TERMS not set)")
    print(f"privacy scan: {len(hits)} hits, {len(warns)} warnings; salted-hash check {state}; "
          f"confidentiality denylist {tstate}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
