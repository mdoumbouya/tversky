#!/usr/bin/env python
"""Add or replace copyright headers in all Python files under the repo.

The header is read from copyright.txt at the repo root. It is wrapped in
delimiter comments so future runs can locate and replace it:

    # <<<SU:HDR:BEGIN>>>
    # ...
    # <<<SU:HDR:END>>>

Run from the repo root:
    python scripts/add_copyright.py [--check]

Options:
    --check   Report files that are missing or have an outdated header; exit 1 if any.
    --dry-run Show what would change without writing files.
"""
import argparse
import pathlib
import sys

_BEGIN = "# <<<SU:HDR:BEGIN>>>"
_END   = "# <<<SU:HDR:END>>>"

_REPO_ROOT  = pathlib.Path(__file__).parent.parent
_COPYRIGHT  = _REPO_ROOT / "copyright.txt"
_SKIP_SELF  = pathlib.Path(__file__).resolve()


def _build_header(copyright_text: str) -> str:
    lines = [_BEGIN]
    for line in copyright_text.rstrip("\n").splitlines():
        lines.append(f"# {line}" if line.strip() else "#")
    lines.append(_END)
    return "\n".join(lines) + "\n"


def _apply(src: str, header: str) -> str:
    """Return src with the copyright header inserted or replaced."""
    lines = src.splitlines(keepends=True)

    # Locate existing block
    begin_i = end_i = None
    for i, line in enumerate(lines):
        if line.rstrip() == _BEGIN:
            begin_i = i
        if line.rstrip() == _END and begin_i is not None:
            end_i = i
            break

    if begin_i is not None and end_i is not None:
        # Replace existing block (keep surrounding blank lines as-is)
        return "".join(lines[:begin_i]) + header + "".join(lines[end_i + 1:])

    # Insert after shebang (and its trailing blank line, if present)
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
        if insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1

    return "".join(lines[:insert_at]) + header + "\n" + "".join(lines[insert_at:])


def _current_header(src: str) -> str | None:
    """Extract the existing copyright block from src, or None if absent."""
    lines = src.splitlines(keepends=True)
    begin_i = end_i = None
    for i, line in enumerate(lines):
        if line.rstrip() == _BEGIN:
            begin_i = i
        if line.rstrip() == _END and begin_i is not None:
            end_i = i
            break
    if begin_i is None or end_i is None:
        return None
    return "".join(lines[begin_i:end_i + 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check",   action="store_true",
                        help="Exit 1 if any file is missing or has an outdated header.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing files.")
    args = parser.parse_args()

    if not _COPYRIGHT.exists():
        sys.exit(f"copyright.txt not found at {_COPYRIGHT}")

    header = _build_header(_COPYRIGHT.read_text(encoding="utf-8"))

    py_files = sorted(
        p for p in _REPO_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and p.resolve() != _SKIP_SELF
    )

    stale, missing = [], []
    for path in py_files:
        src = path.read_text(encoding="utf-8")
        existing = _current_header(src)

        if existing is None:
            missing.append(path)
            tag = "ADD"
        elif existing.rstrip("\n") != header.rstrip("\n"):
            stale.append(path)
            tag = "UPDATE"
        else:
            continue  # already up to date

        rel = path.relative_to(_REPO_ROOT)
        if args.dry_run or args.check:
            print(f"  [{tag}] {rel}")
        else:
            new_src = _apply(src, header)
            path.write_text(new_src, encoding="utf-8")
            print(f"  [{tag}] {rel}")

    total = len(missing) + len(stale)
    if total == 0:
        print("All files up to date.")
    elif not args.dry_run and not args.check:
        print(f"\nUpdated {total} file(s).")

    if args.check and total > 0:
        sys.exit(f"\n{total} file(s) need a copyright header update.")


if __name__ == "__main__":
    main()
