#!/usr/bin/env python3
"""Check a Conventional Commit title and reject Chinese text in the full message."""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ALLOWED_TYPES = "feat|fix|refactor|test|docs|build|ci|chore"
TITLE = re.compile(
    rf"^(?:{ALLOWED_TYPES})\([a-z0-9][a-z0-9-]*\)(?:!)?:\s+\S(?:.*\S)?$"
)


def validate(message: str) -> tuple[bool, str]:
    title = message.splitlines()[0].strip() if message.splitlines() else ""
    if not title:
        return False, "commit title is empty"
    if any(
        char == "\u3007"
        or unicodedata.name(char, "").startswith(
            ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
        )
        for char in message
    ):
        return False, "commit messages must be written in English; Chinese text is not allowed"
    if not TITLE.fullmatch(title):
        return False, "expected type(scope): description with an allowed type"
    if len(title) > 100:
        return False, "commit title must be at most 100 characters"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        print("usage: check_commit_message.py [MESSAGE|COMMIT_MSG_FILE]", file=sys.stderr)
        return 2
    if not args:
        message = sys.stdin.read()
    else:
        # Treat an argument as a message by default; commit-msg hooks pass a
        # short file path, while long titles must never be sent to stat(2).
        candidate = Path(args[0])
        try:
            is_file = len(args[0]) < 4096 and candidate.is_file()
        except OSError:
            is_file = False
        message = candidate.read_text(encoding="utf-8") if is_file else args[0]
    valid, reason = validate(message)
    if not valid:
        print(f"invalid commit message: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
