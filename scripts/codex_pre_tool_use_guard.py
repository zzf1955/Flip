#!/usr/bin/env python3
"""PreToolUse guard for Codex Bash commands in FLIP.

This project may run Codex with danger-full-access for GPU work. This hook is a
best-effort guardrail for obviously dangerous shell commands; it is not a
security sandbox.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/disk_n/zzf/flip")
ALLOWED_WRITE_ROOTS = (
    PROJECT_ROOT,
    Path("/tmp"),
    Path("/disk_n/zzf/.cache/huggingface"),
    Path("/disk_n/zzf/.pip_cache"),
)
PRIVILEGE_COMMANDS = {"sudo", "su", "doas", "pkexec"}
DESTRUCTIVE_GIT = [
    ("git", "reset", "--hard"),
    ("git", "clean"),
    ("git", "push", "--force"),
    ("git", "push", "-f"),
]
DANGEROUS_ROOT_TARGETS = {
    "/",
    "~",
    "$HOME",
    "/home/leadtek",
    "/disk_n",
    "/disk_n/zzf",
    "/home/leadtek/.codex",
}


def load_event() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def find_command(event: dict[str, Any]) -> str:
    candidates: list[Any] = [
        event.get("tool_input", {}).get("cmd"),
        event.get("tool_input", {}).get("command"),
        event.get("toolInput", {}).get("cmd"),
        event.get("toolInput", {}).get("command"),
        event.get("input", {}).get("cmd"),
        event.get("input", {}).get("command"),
        event.get("cmd"),
        event.get("command"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        if isinstance(candidate, list) and candidate:
            return " ".join(shlex.quote(str(part)) for part in candidate)
    return ""


def split_segments(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;()<>")
    lexer.whitespace_split = True
    lexer.commenters = ""

    segments: list[list[str]] = []
    current: list[str] = []
    for token in lexer:
        if token in {"|", "&", ";", "&&", "||", "(", ")", "<", ">", ">>", "2>", "2>>"}:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def command_name(segment: list[str]) -> str:
    idx = 0
    while idx < len(segment):
        token = segment[idx]
        if "=" in token and not token.startswith("-") and token.split("=", 1)[0].isidentifier():
            idx += 1
            continue
        return Path(token).name
    return ""


def normalize_path(token: str) -> Path | None:
    if not token or token.startswith("-"):
        return None
    if token in {".", ".."} or token.startswith("./") or token.startswith("../"):
        return (PROJECT_ROOT / token).resolve()
    if token.startswith("/"):
        return Path(token).resolve()
    return None


def is_allowed_path(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in ALLOWED_WRITE_ROOTS)


def deny(reason: str) -> None:
    print(f"Codex PreToolUse guard blocked command: {reason}", file=sys.stderr)
    sys.exit(2)


def check_raw_patterns(command: str) -> None:
    if re.search(r"\bchmod\s+[^;&|]*[ugoa]*\+s\b", command):
        deny("setuid/setgid chmod is not allowed")
    if re.search(r"\bchmod\s+[^;&|]*[0-7]*[2367][0-7]{3}\b", command):
        deny("setuid/setgid chmod mode is not allowed")
    if re.search(r"\bchown\s+[^;&|]*(^|\s)(root|0)(:|\s|$)", command):
        deny("chown to root is not allowed")
    for target in DANGEROUS_ROOT_TARGETS:
        escaped = re.escape(target)
        if re.search(rf"\brm\s+[^;&|]*(-r|-R|--recursive)[^;&|]*\s+{escaped}(/)?(\s|$)", command):
            deny(f"recursive deletion of {target} is not allowed")


def check_segments(segments: list[list[str]]) -> None:
    for segment in segments:
        name = command_name(segment)
        if not name:
            continue
        if name in PRIVILEGE_COMMANDS:
            deny(f"privilege command `{name}` is not allowed")

        for git_pattern in DESTRUCTIVE_GIT:
            if len(segment) >= len(git_pattern) and tuple(segment[: len(git_pattern)]) == git_pattern:
                deny(f"dangerous git command is not allowed: {' '.join(git_pattern)}")

        if name == "rm":
            recursive = any(
                arg in {"-r", "-R", "-rf", "-fr"} or "r" in arg.lstrip("-")
                for arg in segment[1:]
                if arg.startswith("-")
            )
            if recursive:
                for arg in segment[1:]:
                    path = normalize_path(arg)
                    if path is not None and not is_allowed_path(path):
                        deny(f"recursive rm outside allowed roots is not allowed: {arg}")

        if name in {"mv", "cp", "rsync"}:
            paths = [normalize_path(arg) for arg in segment[1:] if not arg.startswith("-")]
            concrete_paths = [path for path in paths if path is not None]
            if concrete_paths:
                destination = concrete_paths[-1]
                if not is_allowed_path(destination):
                    deny(f"writing outside allowed roots via {name} is not allowed: {destination}")


def main() -> int:
    event = load_event()
    command = find_command(event)
    if not command:
        return 0
    check_raw_patterns(command)
    check_segments(split_segments(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
