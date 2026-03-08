#!/usr/bin/env python3
"""Normalize corpus filenames to M-D-YYYY [Title].md convention.

Run on the host before Docker ingest (corpus is mounted read-only in containers).

Usage:
    python scripts/normalize_filenames.py --dry-run
    python scripts/normalize_filenames.py --corpus-path /path/to/corpus
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Date patterns (mirrored from corpus_utils/file_discovery.py)
# ---------------------------------------------------------------------------

# Target convention: M-D-YYYY (no zero-padding)
CONFORMING_PATTERN = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})")

# Other formats we can extract dates from
DATE_PATTERN_YMD = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
DATE_PATTERN_COMPACT = re.compile(r"^(\d{4})(\d{2})(\d{2})")

# M-D without year (e.g. "4-22 dream") — needs year from parent dir
DATE_PATTERN_MD_ONLY = re.compile(r"^(\d{1,2})-(\d{1,2})\b")

# Shorthand MDDYYYY or MDDYYYY (e.g. "52025" = 5/20/25, "52925" = 5/29/25)
# These are 5-6 digit dates without separators: M + DD + YY or YYYY
DATE_PATTERN_SHORT = re.compile(r"^(\d{1})(\d{2})(\d{2,4})\b")

# MM/DD/YYYY in file content header
HEADER_DATE_PATTERN = re.compile(r"(\d{2})/(\d{2})/(\d{4})")

# Year directory name
YEAR_DIR_PATTERN = re.compile(r"^(20\d{2})$")


def parse_date_from_content(text: str) -> datetime | None:
    """Extract a date from the first 300 chars of file content (e.g. # 08/18/2025)."""
    head = text[:300]
    m = HEADER_DATE_PATTERN.search(head)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            pass
    return None


def get_birthtime(path: Path) -> datetime:
    """Get file creation time (macOS st_birthtime, falls back to mtime)."""
    st = os.stat(path)
    ts = getattr(st, "st_birthtime", None) or st.st_mtime
    return datetime.fromtimestamp(ts)


def format_date_mdy(dt: datetime) -> str:
    """Format datetime as M-D-YYYY (no zero-padding)."""
    return f"{dt.month}-{dt.day}-{dt.year}"


def get_parent_year(path: Path) -> int | None:
    """If the parent directory is a year (e.g. 2025/), return it."""
    m = YEAR_DIR_PATTERN.match(path.parent.name)
    return int(m.group(1)) if m else None


def classify_file(path: Path) -> tuple[str, datetime | None, str | None, str]:
    """Classify a file and extract its date and title.

    Returns (status, date, title, source) where status is one of:
        "conforming" - already matches M-D-YYYY convention
        "rename"     - needs renaming, date extracted
    source indicates how the date was determined.
    """
    stem = path.stem

    # Check if already conforming (M-D-YYYY with optional title)
    m = CONFORMING_PATTERN.match(stem)
    if m:
        return "conforming", None, None, ""

    # Try YYYY-MM-DD
    m = DATE_PATTERN_YMD.match(stem)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = datetime(year, month, day)
            rest = stem[m.end():].strip(" -_")
            return "rename", dt, rest or None, "filename"
        except ValueError:
            pass

    # Try YYYYMMDD (8 digits)
    m = DATE_PATTERN_COMPACT.match(stem)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = datetime(year, month, day)
            rest = stem[m.end():].strip(" -_")
            return "rename", dt, rest or None, "filename"
        except ValueError:
            pass

    # Try shorthand like 52025 = 5/20/25 (M+DD+YY) — infer century from parent dir
    m = DATE_PATTERN_SHORT.match(stem)
    if m:
        month = int(m.group(1))
        day = int(m.group(2))
        year_raw = int(m.group(3))
        # 2-digit year: infer century (20xx)
        if year_raw < 100:
            year = 2000 + year_raw
        else:
            year = year_raw
        try:
            dt = datetime(year, month, day)
            rest = stem[m.end():].strip(" -_")
            return "rename", dt, rest or None, "filename-short"
        except ValueError:
            pass

    # Try M-D without year — infer year from parent directory
    parent_year = get_parent_year(path)
    if parent_year:
        m = DATE_PATTERN_MD_ONLY.match(stem)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            try:
                dt = datetime(parent_year, month, day)
                rest = stem[m.end():].strip(" -_")
                return "rename", dt, rest or None, "filename+dir"
            except ValueError:
                pass

    # No date in filename — try content header
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        dt = parse_date_from_content(text)
        if dt:
            return "rename", dt, stem, "content"
    except OSError:
        pass

    # Last resort: file birthtime
    dt = get_birthtime(path)
    return "rename", dt, stem, "birthtime"


def build_target_name(dt: datetime, title: str | None) -> str:
    """Build the target filename (without extension)."""
    date_str = format_date_mdy(dt)
    if title:
        return f"{date_str} {title}"
    return date_str


def resolve_collision(target: Path) -> Path:
    """If target already exists, append (2), (3), etc."""
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    counter = 2
    while True:
        candidate = parent / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def normalize_corpus(corpus_path: str, dry_run: bool = True) -> None:
    root = Path(corpus_path)
    if not root.is_dir():
        print(f"Error: {corpus_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = sorted(root.rglob("*.md"))
    # Filter out macOS resource forks
    files = [f for f in files if not f.name.startswith("._")]

    skipped = 0
    renamed = 0
    errors = 0

    for path in files:
        status, dt, title, source = classify_file(path)

        if status == "conforming":
            skipped += 1
            continue

        target_stem = build_target_name(dt, title)
        target = path.parent / f"{target_stem}.md"
        target = resolve_collision(target)

        if target == path:
            skipped += 1
            continue

        rel_old = path.relative_to(root)
        rel_new = target.relative_to(root)
        tag = f"  [{source}]" if dry_run else ""

        if dry_run:
            print(f"  {rel_old}  →  {rel_new}{tag}")
        else:
            try:
                path.rename(target)
                print(f"  {rel_old}  →  {rel_new}")
                renamed += 1
            except OSError as e:
                print(f"  ERROR: {rel_old}: {e}", file=sys.stderr)
                errors += 1

    print()
    if dry_run:
        total_to_rename = len(files) - skipped
        print(f"Dry run: {total_to_rename} file(s) would be renamed, {skipped} already conforming")
    else:
        print(f"Done: {renamed} renamed, {skipped} skipped, {errors} errors")


def load_corpus_path_from_env() -> str | None:
    """Try to load CORPUS_PATH from .env file."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("CORPUS_PATH=") and not line.startswith("#"):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Normalize corpus filenames to M-D-YYYY [Title].md"
    )
    parser.add_argument(
        "--corpus-path",
        default=None,
        help="Path to corpus directory (default: from .env or ./data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without doing it",
    )
    args = parser.parse_args()

    corpus_path = args.corpus_path or load_corpus_path_from_env() or "./data"
    corpus_path = os.path.expanduser(corpus_path)

    print(f"Corpus: {corpus_path}")
    print(f"Mode:   {'dry run' if args.dry_run else 'LIVE'}")
    print()

    normalize_corpus(corpus_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
