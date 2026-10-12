#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_validator.py — Validate Anki CSV files cho cheatsheet-anki-vn repo.

Cach dung:
    python _tools/csv_validator.py <file1.csv> [<file2.csv> ...]
    python _tools/csv_validator.py 01-math/anki-csv/*.csv
    python _tools/csv_validator.py --all      # validate tat ca CSV trong repo

Kiem tra:
1. Parse CSV OK (delimiter ',', quote '"', UTF-8, auto strip BOM)
2. Dung 8 cot header giong SCHEMA.md
3. Field bat buoc khong rong: Front, Back, Tags, Deck, CardType, Difficulty
4. CardType thuoc enum: Basic, Cloze, TypeIn, Reversed
5. Difficulty thuoc enum: easy, medium, hard
6. Deck format: AI::XX-Name::Topic (regex)
7. Tags co it nhat 1 tag hierarchical (match <topic>::<subtopic>)
8. Dedupe Front trong cung deck (warning neu trung)
9. MathJax balance: dem \( \) va $$ (warning neu le)
10. Bao cao: tong the / deck / card type mix / difficulty mix

Exit code:
    0 = pass (co warning cung pass)
    1 = co error (parse fail, sai schema, field bat buoc rong, enum sai)
"""

import sys
import re
import csv
import io
import glob
import os
from pathlib import Path
from collections import Counter, defaultdict

# Force stdout UTF-8 (Windows cp1252 khong encode duoc tieng Viet)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# === Quy uong SCHEMA ===
EXPECTED_HEADERS = [
    "Front", "Back", "Tags", "Deck",
    "CardType", "Source", "Difficulty", "Linked",
]
CARD_TYPES = {"Basic", "Cloze", "TypeIn", "Reversed"}
DIFFICULTIES = {"easy", "medium", "hard"}
DECK_RE = re.compile(r"^AI::\d{2}-[A-Za-z][\w-]*::[\w][\w-]*$")
TAG_HIER_RE = re.compile(r"^\w[\w-]*(?:::[\w-]+)+$")
MATHJAX_INLINE_RE = re.compile(r"\\\(")
MATHJAX_INLINE_CLOSE_RE = re.compile(r"\\\)")
MATHJAX_BLOCK_RE = re.compile(r"\$\$")

REQUIRED_FIELDS = ["Front", "Back", "Tags", "Deck", "CardType", "Difficulty"]


def strip_bom(path: Path) -> str:
    """Doc file, tu bo BOM UTF-8 neu co."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return raw.decode("utf-8", errors="replace")


def validate_one(path: Path) -> dict:
    """Validate 1 file CSV. Tra ve dict bao cao."""
    report = {
        "file": str(path),
        "errors": [],
        "warnings": [],
        "rows": 0,
        "deck_mix": Counter(),
        "cardtype_mix": Counter(),
        "difficulty_mix": Counter(),
        "front_seen_by_deck": defaultdict(list),
    }

    text = strip_bom(path)
    try:
        reader = csv.DictReader(io.StringIO(text))
        headers = reader.fieldnames or []
    except csv.Error as e:
        report["errors"].append(f"CSV parse fail: {e}")
        return report

    # Kiem tra headers
    if headers != EXPECTED_HEADERS:
        report["errors"].append(
            f"Sai header. Mong doi {EXPECTED_HEADERS}, thay {headers}"
        )
        return report

    for i, row in enumerate(reader, start=2):
        report["rows"] += 1

        # 1. Field bat buoc khong rong
        for f in REQUIRED_FIELDS:
            val = (row.get(f) or "").strip()
            if not val:
                report["errors"].append(
                    f"dong {i}: field '{f}' rong"
                )

        # 2. CardType enum
        ct = (row.get("CardType") or "").strip()
        if ct and ct not in CARD_TYPES:
            report["errors"].append(
                f"dong {i}: CardType='{ct}' khong thuoc {CARD_TYPES}"
            )

        # 3. Difficulty enum
        diff = (row.get("Difficulty") or "").strip()
        if diff and diff not in DIFFICULTIES:
            report["errors"].append(
                f"dong {i}: Difficulty='{diff}' khong thuoc {DIFFICULTIES}"
            )

        # 4. Deck format
        deck = (row.get("Deck") or "").strip()
        if deck and not DECK_RE.match(deck):
            report["warnings"].append(
                f"dong {i}: Deck='{deck}' khong match regex AI::XX-Name::Topic"
            )

        # 5. Tags hierarchical
        tags = (row.get("Tags") or "").strip()
        if tags:
            tag_list = tags.split()
            has_hier = any(TAG_HIER_RE.match(t) for t in tag_list)
            if not has_hier:
                report["warnings"].append(
                    f"dong {i}: Tags='{tags}' khong co tag hierarchical nao "
                    f"(<topic>::<subtopic>)"
                )

        # 6. MathJax balance (tren Back)
        back = row.get("Back") or ""
        if back:
            n_open = len(MATHJAX_INLINE_RE.findall(back))
            n_close = len(MATHJAX_INLINE_CLOSE_RE.findall(back))
            if n_open != n_close:
                report["warnings"].append(
                    f"dong {i}: MathJax inline \\( ({n_open}) != \\) ({n_close})"
                )
            n_block = len(MATHJAX_BLOCK_RE.findall(back))
            if n_block % 2 != 0:
                report["warnings"].append(
                    f"dong {i}: MathJax block $$ le ({n_block})"
                )

        # 7. Dedupe Front trong cung deck
        front = (row.get("Front") or "").strip()
        if front and deck:
            report["front_seen_by_deck"][deck].append((i, front))

        # 8. Mix thong ke
        if deck:
            report["deck_mix"][deck] += 1
        if ct:
            report["cardtype_mix"][ct] += 1
        if diff:
            report["difficulty_mix"][diff] += 1

    # 9. Dedupe check (post-parse)
    for deck, items in report["front_seen_by_deck"].items():
        front_count = Counter(f for _, f in items)
        for front, n in front_count.items():
            if n > 1:
                dups = [i for i, f in items if f == front]
                report["warnings"].append(
                    f"Deck '{deck}': Front trung {n} lan tai dong {dups}: "
                    f"'{front[:80]}...'"
                )

    return report


def print_report(rep: dict) -> bool:
    """In bao cao 1 file. Tra True neu pass (no errors), False neu co errors."""
    print(f"\n{'=' * 70}")
    print(f"  FILE: {rep['file']}")
    print(f"{'=' * 70}")
    print(f"  Rows        : {rep['rows']}")
    print(f"  Errors      : {len(rep['errors'])}")
    print(f"  Warnings    : {len(rep['warnings'])}")

    if rep["deck_mix"]:
        print(f"  Deck mix    :")
        for d, n in sorted(rep["deck_mix"].items()):
            print(f"    - {d}: {n}")
    if rep["cardtype_mix"]:
        print(f"  CardType mix:")
        for c, n in sorted(rep["cardtype_mix"].items()):
            print(f"    - {c}: {n}")
    if rep["difficulty_mix"]:
        print(f"  Difficulty  :")
        for d, n in sorted(rep["difficulty_mix"].items()):
            print(f"    - {d}: {n}")

    if rep["errors"]:
        print(f"\n  [X] ERRORS ({len(rep['errors'])}):")
        for e in rep["errors"]:
            print(f"    - {e}")
    if rep["warnings"]:
        print(f"\n  [!] WARNINGS ({len(rep['warnings'])}):")
        for w in rep["warnings"]:
            print(f"    - {w}")

    if not rep["errors"] and not rep["warnings"]:
        print("\n  [OK] ALL CLEAN")
    elif not rep["errors"]:
        print("\n  [OK] PASS (with warnings)")
    else:
        print("\n  [X] FAIL")

    return len(rep["errors"]) == 0


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)

    if args[0] in ("--all", "-a"):
        root = Path(__file__).resolve().parent.parent
        files = sorted(root.rglob("*.csv"))
        files = [f for f in files if "_tools" not in str(f)]
    else:
        files = []
        for a in args:
            p = Path(a)
            if p.is_file():
                files.append(p)
            elif "*" in a or "?" in a:
                files.extend(sorted(Path().glob(a)))
            else:
                print(f"Khong tim thay file: {a}", file=sys.stderr)

    if not files:
        print("Khong co CSV nao de validate.", file=sys.stderr)
        sys.exit(2)

    print(f"Validating {len(files)} CSV file(s)...")
    all_pass = True
    total_rows = 0
    for f in files:
        rep = validate_one(f)
        ok = print_report(rep)
        if not ok:
            all_pass = False
        total_rows += rep["rows"]

    print(f"\n{'#' * 70}")
    print(f"  SUMMARY: {len(files)} file(s), {total_rows} the tong so")
    print(f"  Status: {'[OK] ALL PASS' if all_pass else '[X] SOME FAIL'}")
    print(f"{'#' * 70}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
