"""Regenerate screenshots/*.png from real local runs.

    python tools/make_screenshots.py [--data DATA_ROOT] [--layout FILE] [--id-hashes FILE] [--terms FILE]
                                     [--only name ...]
    python tools/make_screenshots.py gate-record        print the recorded reproduction gates (used below)

Each screenshot runs its commands in the repository, writes the raw log to `.scratch/screenshots/`,
replaces machine paths (repository, data root, interpreter, home) with neutral names, and privacy-scans the
sanitised log with `tools/privacy_scan.py` (including the private confidentiality denylist when given) plus
extra patterns (Cyrillic subject codes, upper-case codes, dates). Only a clean log is rendered: dark terminal HTML, then `chromium --headless --screenshot`.

`pytest_full` needs the confidential data (`--data` or `PTSD_DATA`, plus its private layout file `--layout` or
`PTSD_LAYOUT`) and is skipped otherwise, keeping the committed PNG. `clean_venv_gate` does not rebuild a clean virtual environment: that needs the pinned wheels,
which are not available offline here. It re-renders the gate recorded at delivery
(`results/final_gates_summary.csv`) and runs the public weights checks live.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / ".scratch" / "screenshots"
OUT = REPO / "screenshots"
PAGE_WIDTH = 1200
LINE_PX = 21
TERM_COLUMNS = 124
"""Characters per rendered line at the chosen font size and width (used to estimate the page height)."""

ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m")
ANSI_COLOURS = {30: "#4d4d4d", 31: "#ff6b6b", 32: "#7ee787", 33: "#f2cc60", 34: "#79c0ff", 35: "#d2a8ff",
                36: "#76e3ea", 37: "#e6edf3", 90: "#8b949e", 91: "#ff8182", 92: "#a5f3a0", 93: "#f8e3a1",
                94: "#a5d6ff", 95: "#e2c5ff", 96: "#b3f0ff", 97: "#ffffff"}
EXTRA_PATTERNS = {
    "cyrillic-code": re.compile(r"[\u0410-\u042f\u0401]{1,3}[-_ ]?\d{2,4}"),
    "latin-code": re.compile(r"(?<![\w-])[A-Z]{4,5}\d{0,2}(?![\w-])"),
    "date": re.compile(r"(?<![\d.])(?:(?:0[1-9]|[12]\d|3[01])[./-](?:0[1-9]|1[0-2])[./-](?:\d{4}|\d{2})"
                       r"|(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))(?![\d.])"),
}
LATIN_OK = {"PTSD", "CTRL", "SOMA", "PASS", "FAIL", "LOSO", "NOTE", "GTEC", "FLEX", "TRUE", "FALSE", "NONE"}


# ------------------------------------------------------------------ gate record (printed inside a screenshot)
def gate_record() -> int:
    import csv
    rows = list(csv.DictReader((REPO / "results" / "final_gates_summary.csv").open(encoding="utf-8")))
    eq = json.loads((REPO / "results" / "weights_equivalence.json").read_text(encoding="utf-8"))
    print("Recorded reproduction gates of the delivered model (results/final_gates_summary.csv)")
    print(f"  model sha256 {rows[0]['model_sha256'][:16]}...   tolerance {float(rows[0]['tolerance']):g}")
    print(f"  {'gate':<16}{'result':<8}{'n':>5}  {'max |diff|':<12}description")
    for r in rows:
        diff = f"{float(r['max_abs_diff']):.1e}" if r["max_abs_diff"] else "-"
        mark = "PASS" if r["passed"] == "True" else "FAIL"
        print(f"  {r['gate']:<16}{mark:<8}{int(float(r['n'])):>5}  {diff:<12}{r['description']}")
    d = next(r for r in rows if r["gate"] == "d_clean_venv")
    print()
    print(f"Gate d (clean venv): fresh venv with the pinned runtime requirements only, package copied to a temp")
    print(f"dir, other cwd, no PYTHONPATH; somatoform folders scored -> max |diff| vs gate c = "
          f"{float(d['max_abs_diff']):.1f} over n = {int(float(d['n']))}")
    print()
    print("Public weights vs delivered model (results/weights_equivalence.json)")
    print(f"  every real subject folder: max |dScore| {eq['max_abs_diff']:.1e}, "
          f"decisions changed at 0.5: {eq['n_decision_changed_at_0_5']}")
    return 0


# ------------------------------------------------------------------ steps
def steps(data: Path | None, id_hashes: Path | None, terms: Path | None = None,
          layout: Path | None = None) -> dict[str, dict]:
    hashes_env = {"PTSD_ID_HASHES": str(id_hashes)} if id_hashes and id_hashes.is_file() else {}
    if terms and terms.is_file():
        hashes_env["PTSD_CONFIDENTIAL_TERMS"] = str(terms)
    return {
        "pytest_fast": dict(
            title="make test  (synthetic suite, as in CI)",
            cmds=[("make test PYTHON=python", "make test", {})]),
        "pytest_full": dict(
            title="Full test suite on the confidential data (data-gated tests enabled)",
            needs_data=True,
            cmds=[("PTSD_DATA=$PTSD_DATA python -m pytest -v", "python -m pytest -v -p no:cacheprovider",
                   {"PTSD_DATA": str(data) if data else "", "PTSD_LAYOUT": str(layout) if layout else ""})]),
        "make_demo": dict(
            title="make demo  (synthetic EDF -> shipped weights -> predictions; no data needed)",
            cmds=[("make demo", "make demo PYTHON=python", {})]),
        "cli_predict": dict(
            title="CLI predict on the synthetic folder  (python -m ptsd_eeg.cli = ptsd-eeg after pip install -e .)",
            cmds=[("python -m ptsd_eeg.cli predict --help", "python -m ptsd_eeg.cli predict --help", {}),
                  ("python -m ptsd_eeg.cli predict .demo/edf .demo/predictions.csv",
                   "python -m ptsd_eeg.cli predict .demo/edf .demo/predictions.csv", {}),
                  ("column -s, -t .demo/predictions.csv", "column -s, -t .demo/predictions.csv", {})]),
        "privacy_scan": dict(
            title="make privacy  (repository scan incl. salted-hash ID check and confidentiality denylist; 0 hits)",
            cmds=[("make privacy", "make privacy PYTHON=python", hashes_env)]),
        "clean_venv_gate": dict(
            title="Reproduction gates: recorded clean-venv gate + live public weights checks",
            note=("Gate d is the recorded delivery run (no offline wheels here to rebuild a clean venv); "
                  "the weights checks below ran live."),
            cmds=[("python tools/make_screenshots.py gate-record", "python tools/make_screenshots.py gate-record", {}),
                  ("python -m ptsd_eeg.cli verify-weights", "python -m ptsd_eeg.cli verify-weights", {}),
                  ("python -m pytest tests/test_weights.py -v", "python -m pytest tests/test_weights.py -v "
                   "-p no:cacheprovider", {})]),
    }


def run_env(extra: dict) -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    env.update(PYTHONPATH=f"src{os.pathsep}examples", OMP_NUM_THREADS="1", PY_COLORS="1", COLUMNS=str(TERM_COLUMNS))
    env.pop("PTSD_DATA", None)
    env.pop("PTSD_LAYOUT", None)
    env.update({k: v for k, v in extra.items() if v})
    return env


def sanitise(text: str, data: Path | None) -> str:
    subs = []
    if data:
        subs += [(str(data.resolve()), "$PTSD_DATA"), (str(data), "$PTSD_DATA")]
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    exe_dirs = dict.fromkeys([str(Path(sys.executable).parent), exe_dir, str(Path(sys.executable).resolve().parent),
                              os.path.relpath(exe_dir, REPO)])
    subs += [(str(REPO), "ptsd-eeg-prototype"), *((d, "") for d in exe_dirs),
             (str(REPO.parent / ".private_release"), "<outside-repo>"), ("../.private_release", "<outside-repo>"),
             (str(REPO.parent), "<workspace>"), (str(Path.home()), "~")]
    for a, b in subs:
        text = text.replace(a + "/python", "python") if b == "" else text.replace(a, b)
    text = re.sub(r"/tmp/[\w./-]+", "<tmp>", text)
    return text.replace("\r", "")


def extra_hits(text: str) -> tuple[list[str], list[str]]:
    """(blocking hits, review-only tokens). Upper-case codes are review-only: acronyms look the same."""
    plain = ANSI_RE.sub("", text)
    hits, review = [], set()
    for name, pat in EXTRA_PATTERNS.items():
        for m in pat.finditer(plain):
            tok = m.group(0)
            if name == "latin-code":
                if tok.rstrip("0123456789") not in LATIN_OK:
                    review.add(tok)
                continue
            hits.append(f"{name}: {tok[:1]}***({len(tok)})")
    return hits, sorted(review)


def privacy_check(log: Path, id_hashes: Path | None, terms: Path | None = None) -> tuple[bool, str]:
    """Scan one sanitised log in an isolated folder with the repository scanner plus extra patterns."""
    box = SCRATCH / "scan" / log.stem
    if box.exists():
        assert box.resolve().is_relative_to(SCRATCH.resolve())
        shutil.rmtree(box)
    box.mkdir(parents=True)
    shutil.copy(log, box / log.name)
    cmd = [sys.executable, str(REPO / "tools" / "privacy_scan.py"), "--root", str(box)]
    if id_hashes and id_hashes.is_file():
        cmd += ["--hashes", str(id_hashes)]
    if terms and terms.is_file():
        cmd += ["--terms", str(terms)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    extra, review = extra_hits(log.read_text(encoding="utf-8"))
    summary = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()
    ok = r.returncode == 0 and not extra and "0 hits" in summary
    summary += f"; extra patterns: {extra or 0}"
    if review:
        summary += f"; upper-case tokens to review: {' '.join(review)}"
    return ok, summary


# ------------------------------------------------------------------ rendering
def ansi_to_html(text: str) -> str:
    out, open_span = [], False
    pos = 0
    for m in ANSI_RE.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        pos = m.end()
        codes = [int(c) for c in m.group(1).split(";") if c.isdigit()] or [0]
        styles = []
        for c in codes:
            if c in ANSI_COLOURS:
                styles.append(f"color:{ANSI_COLOURS[c]}")
            elif c == 1:
                styles.append("font-weight:bold")
        if open_span:
            out.append("</span>")
            open_span = False
        if styles:
            out.append(f'<span style="{";".join(styles)}">')
            open_span = True
    out.append(html.escape(text[pos:]))
    if open_span:
        out.append("</span>")
    return "".join(out)


def render_html(title: str, blocks: list[tuple[str, str]], note: str | None) -> tuple[str, int]:
    body, n_lines = [], 0
    for shown, output in blocks:
        body.append(f'<div class="cmd"><span class="prompt">$</span> {html.escape(shown)}</div>')
        body.append(f'<div class="out">{ansi_to_html(output.rstrip())}</div>')
        plain = ANSI_RE.sub("", output.rstrip()).splitlines() or [""]
        n_lines += 1 + sum(max(1, math.ceil(len(s) / TERM_COLUMNS)) for s in plain) + 0.6
    note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
    height = int(96 + (46 if note else 0) + n_lines * LINE_PX + 36)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:#0d1117;}}
.win{{margin:0;background:#161b22;min-height:100vh;box-sizing:border-box;}}
.bar{{height:40px;background:#21262d;display:flex;align-items:center;padding:0 16px;border-bottom:1px solid #30363d;}}
.dot{{width:13px;height:13px;border-radius:50%;margin-right:8px;display:inline-block;}}
.title{{color:#c9d1d9;font:600 15px 'DejaVu Sans',sans-serif;margin-left:14px;white-space:nowrap;overflow:hidden;}}
.term{{padding:14px 22px 18px 22px;font:15px/{LINE_PX}px 'DejaVu Sans Mono',monospace;color:#e6edf3;}}
.cmd{{color:#e6edf3;font-weight:bold;margin-top:{int(LINE_PX * 0.6)}px;white-space:pre-wrap;word-break:break-all;}}
.cmd:first-child{{margin-top:0;}}
.prompt{{color:#7ee787;}}
.out{{white-space:pre-wrap;word-break:break-all;color:#c9d1d9;}}
.note{{margin:10px 22px 0 22px;padding:8px 12px;border-left:3px solid #f2cc60;color:#f2cc60;
      font:14px 'DejaVu Sans',sans-serif;background:#1f1d14;}}
</style></head><body><div class="win">
<div class="bar"><span class="dot" style="background:#ff5f56"></span><span class="dot" style="background:#ffbd2e">
</span><span class="dot" style="background:#27c93f"></span><span class="title">{html.escape(title)}</span></div>
{note_html}<div class="term">{"".join(body)}</div></div></body></html>"""
    return page, height


def chromium() -> str:
    for name in ("chromium", "chromium-browser", "google-chrome"):
        if shutil.which(name):
            return shutil.which(name)
    raise SystemExit("chromium not found (needed for headless screenshots)")


def screenshot(html_path: Path, png: Path, height: int) -> None:
    profile = SCRATCH / "chromium-profile"
    cmd = [chromium(), "--headless", "--disable-gpu", "--disable-gpu-compositing", "--disable-crash-reporter",
           "--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=1", f"--user-data-dir={profile}",
           "--no-first-run", "--disable-extensions", "--virtual-time-budget=5000", f"--screenshot={png}",
           f"--window-size={PAGE_WIDTH},{height}", html_path.resolve().as_uri()]
    for _ in range(2):
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if png.is_file() and png.stat().st_mtime >= t0 - 1:
            crop_bottom(png)
            return
    raise RuntimeError(f"chromium failed: {r.stderr[-800:]}")


def crop_bottom(png: Path, pad: int = 24) -> None:
    """Drop trailing rows that only contain the page background (the height is estimated generously)."""
    import numpy as np
    from PIL import Image
    im = Image.open(png).convert("RGB")
    a = np.asarray(im).astype(int)
    bg = a[-1, -1]
    rows = np.flatnonzero(np.abs(a - bg).sum(axis=2).max(axis=1) > 30)
    if rows.size and rows[-1] + pad < a.shape[0] - 1:
        im.crop((0, 0, a.shape[1], rows[-1] + pad)).save(png, optimize=True)


# ------------------------------------------------------------------ notebook page
def notebook_01(id_hashes: Path | None, terms: Path | None = None) -> None:
    """Title cell plus the scoring section of notebook 01 (already executed), rendered by nbconvert."""
    import nbformat
    from nbconvert import HTMLExporter
    nb = nbformat.read(REPO / "notebooks" / "01_quickstart_synthetic.ipynb", as_version=4)
    cells = nb.cells
    start = next(i for i, c in enumerate(cells) if c.cell_type == "markdown" and c.source.startswith("## 5."))
    end = next(i for i, c in enumerate(cells) if i > start and c.cell_type == "markdown")
    nb.cells = [cells[0], *cells[start:end]]
    exporter = HTMLExporter(template_name="lab")
    body, _ = exporter.from_notebook_node(nb)
    body = sanitise(body, None)
    banner = ('<div style="font:13px \'DejaVu Sans\',sans-serif;color:#57606a;background:#f6f8fa;'
              'border-bottom:1px solid #d0d7de;padding:8px 24px">notebooks/01_quickstart_synthetic.ipynb '
              '(executed; excerpt: title cell and section 5, synthetic data only)</div>')
    body = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + banner, body, count=1)
    page = SCRATCH / "notebook_01.html"
    page.write_text(body, encoding="utf-8")
    text_only = SCRATCH / "notebook_01.log"
    text_only.write_text(re.sub(r"data:image/[^\"']+", "", body), encoding="utf-8")
    ok, summary = privacy_check(text_only, id_hashes, terms)
    print(f"  privacy: {summary}")
    if not ok:
        raise SystemExit("notebook_01: privacy check failed, not rendered")
    screenshot(page, OUT / "notebook_01.png", 1660)


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["gate-record"]:
        return gate_record()
    names = [*steps(None, None), "notebook_01"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=os.environ.get("PTSD_DATA") or None)
    ap.add_argument("--id-hashes", type=Path, default=os.environ.get("PTSD_ID_HASHES") or None)
    ap.add_argument("--terms", type=Path, default=os.environ.get("PTSD_CONFIDENTIAL_TERMS") or None)
    ap.add_argument("--layout", type=Path, default=os.environ.get("PTSD_LAYOUT") or None)
    ap.add_argument("--only", nargs="+", choices=names)
    a = ap.parse_args(argv)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    todo = a.only or names
    failed = []
    for name, spec in steps(a.data, a.id_hashes, a.terms, a.layout).items():
        if name not in todo:
            continue
        if spec.get("needs_data") and not (a.data and a.layout and a.layout.is_file()):
            print(f"skipped {name}: confidential data or its layout not given (--data, --layout); "
                  f"keeping screenshots/{name}.png")
            continue
        print(f"{name}:")
        blocks, raw_all, step_ok = [], [], True
        for shown, cmd, extra in spec["cmds"]:
            t0 = time.time()
            r = subprocess.run(cmd, shell=True, cwd=REPO, env=run_env(extra), capture_output=True, text=True)
            raw = r.stdout + r.stderr
            raw_all.append(f"$ {cmd}\n{raw}\n[exit {r.returncode}, {time.time() - t0:.1f} s]\n")
            print(f"  $ {shown}  -> exit {r.returncode} ({time.time() - t0:.1f} s)")
            if r.returncode != 0:
                failed.append(f"{name}: '{shown}' exit {r.returncode}, not rendered")
                step_ok = False
            blocks.append((shown, sanitise(raw, Path(a.data) if a.data else None)))
        (SCRATCH / f"{name}.raw.log").write_text("\n".join(raw_all), encoding="utf-8")
        log = SCRATCH / f"{name}.log"
        log.write_text("\n".join(f"$ {s}\n{o}" for s, o in blocks), encoding="utf-8")
        ok, summary = privacy_check(log, a.id_hashes, a.terms)
        print(f"  privacy: {summary}")
        if not ok:
            failed.append(f"{name}: privacy check failed, not rendered")
            continue
        if not step_ok:
            continue
        page, height = render_html(spec["title"], blocks, spec.get("note"))
        html_path = SCRATCH / f"{name}.html"
        html_path.write_text(page, encoding="utf-8")
        screenshot(html_path, OUT / f"{name}.png", height)
        print(f"  wrote screenshots/{name}.png ({(OUT / f'{name}.png').stat().st_size / 1024:.0f} KB)")
    if "notebook_01" in todo:
        print("notebook_01:")
        notebook_01(a.id_hashes, a.terms)
        print(f"  wrote screenshots/notebook_01.png ({(OUT / 'notebook_01.png').stat().st_size / 1024:.0f} KB)")
    if failed:
        print("problems:\n  " + "\n  ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
