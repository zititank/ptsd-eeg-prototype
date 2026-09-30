"""The privacy scanner catches each class of leak (fake codes are assembled at run time)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import privacy_scan as ps  # noqa: E402

FAKE_CODE = "QW" + "123_45"
FAKE_WORD = "ZQX" + "VK"
FAKE_BRAND = "Blor" + "pwave"
CYRILLIC = "\u0442\u0435\u0441\u0442"


def _hashes(tmp_path: Path, codes: list[str]) -> Path:
    salt = "00ff"
    p = tmp_path / "private" / "ids.json"
    p.parent.mkdir()
    p.write_text(json.dumps(dict(salt=salt, hashes=[
        hashlib.sha256((salt + ps.normalise_token(c)).encode()).hexdigest() for c in codes])))
    return p


def _terms(tmp_path: Path, terms: list[str]) -> Path:
    p = tmp_path / "private" / "terms.txt"
    p.parent.mkdir(exist_ok=True)
    p.write_text("# comment\n" + "\n".join(terms) + "\n", encoding="utf-8")
    return p


def _checks(root: Path, hashes: Path | None = None, terms: Path | None = None) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in ps.scan(root, ps.HashList(hashes), ps.TermList(terms)):
        out[f"{f.level}:{f.check}"] = out.get(f"{f.level}:{f.check}", 0) + 1
    return out


def test_clean_tree_has_no_hits(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "a.md").write_text("PTSD vs CTRL, LOSO, batch 02.01.00, alpha 8-12 Hz, P3 P4 O1 O2\n")
    assert _checks(tmp_path / "repo") == {}


def test_each_leak_class_is_caught(tmp_path):
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    (root / "notes.md").write_text(
        f"subject {FAKE_CODE} and {FAKE_WORD} recorded {'14.03'}.24\n"
        f"EEG_sb{10}_ss1 and NF {22}, participant P{'08'}\n"
        "path /home/" + "someone/data\n")
    (root / "results" / "t.csv").write_text("batch,n,n_positive\nA,3,1\nB,20,2\n")
    (root / "x.edf").write_bytes(b"0" + b" " * 300)
    (root / "blob.txt").write_bytes(b"0" + b" " * 7 + b"x" * 300)
    (root / "big.json").write_bytes(b" " * (ps.MAX_BYTES + 1))
    nb = dict(cells=[dict(cell_type="code", source=["print(1)"],
                          outputs=[dict(output_type="stream", text=[f"{FAKE_CODE}\n"])])], metadata={})
    (root / "n.ipynb").write_text(json.dumps(nb))
    c = _checks(root, _hashes(tmp_path, [FAKE_WORD]))
    assert c["HIT:id-pattern"] >= 2          # notes.md + notebook output
    assert c["HIT:id-hash"] == 1             # FAKE_WORD (no pattern would catch it)
    assert c["HIT:date"] == 1
    assert c["HIT:mendeley-participant"] == 3
    assert c["HIT:home-path"] == 1
    assert c["HIT:small-cell"] == 1          # n=3 only; n_positive is an outcome count
    assert c["HIT:file-type"] == 1
    assert c["HIT:edf-header"] == 1
    assert c["HIT:size"] == 1
    assert c["WARN:caps-token"] >= 1         # FAKE_WORD is also an unknown capitals token


def test_confidential_terms_and_cyrillic_are_caught(tmp_path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "a.md").write_text(f"recorded with a {FAKE_BRAND.upper()} headset\n", encoding="utf-8")
    (root / "b.csv").write_text(f"batch,auc\n{CYRILLIC},0.5\n", encoding="utf-8")
    nb = dict(cells=[dict(cell_type="markdown", source=[f"the {FAKE_BRAND}-2 device"])], metadata={})
    (root / "n.ipynb").write_text(json.dumps(nb))
    c = _checks(root, terms=_terms(tmp_path, [r"blor\s?pwave"]))
    assert c["HIT:confidential-term"] == 2
    assert c["HIT:cyrillic"] == 1
    assert _checks(root) == {"HIT:cyrillic": 1}
    assert ps.TermList(None).enabled is False


def test_private_lists_must_live_outside_the_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "terms.txt").write_text("x\n")
    assert ps.main(["--root", str(root), "--terms", str(root / "terms.txt")]) == 2


def test_hash_check_is_skipped_without_list(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.md").write_text(f"{FAKE_WORD.lower()}\n")
    assert _checks(root) == {}
    assert ps.HashList(None).enabled is False


def test_count_line_is_not_a_participant_list(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.md").write_text("The expected n is 15 (NF 8, MI 7).\n")
    assert _checks(root) == {}


def test_shipped_repo_is_clean():
    hits = [f for f in ps.scan(REPO, ps.HashList(None)) if f.level == "HIT"]
    assert hits == [], "\n".join(map(str, hits))
