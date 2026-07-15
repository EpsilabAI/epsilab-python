"""Lightweight interactive prompts for the Epsilab CLI.

Provides Vercel-style step-by-step prompts when stdin is a TTY,
falls through silently when running non-interactively (CI, pipes).
No external dependencies — uses only the Python standard library.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

T = TypeVar("T")

# ANSI helpers
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


def is_interactive() -> bool:
    """True when stdin is a TTY and prompts should be shown."""
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


def _label(text: str) -> str:
    return f"{_CYAN}?{_RESET} {_BOLD}{text}{_RESET}"


def confirm(message: str, *, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        answer = input(f"{_label(message)} {_DIM}({hint}){_RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def text(
    message: str,
    *,
    default: str = "",
    required: bool = False,
    validator: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """Prompt for a text value with optional default and validation."""
    suffix = f" {_DIM}({default}){_RESET}" if default else ""
    while True:
        try:
            answer = input(f"{_label(message)}{suffix} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            if default:
                return default
            sys.exit(1)
        if not answer:
            answer = default
        if required and not answer:
            print(f"  {_RED}This field is required.{_RESET}")
            continue
        if validator:
            err = validator(answer)
            if err:
                print(f"  {_RED}{err}{_RESET}")
                continue
        return answer


def select(
    message: str,
    choices: Sequence[Dict[str, str]],
    *,
    default: Optional[str] = None,
) -> str:
    """Numbered selection list. Each choice has 'value' and 'label' keys.

    Returns the selected 'value'.
    """
    print(f"{_label(message)}")
    for i, choice in enumerate(choices, 1):
        marker = f"{_GREEN}>{_RESET}" if choice["value"] == default else " "
        print(f"  {marker} {i}. {choice['label']}")

    while True:
        try:
            raw = input(f"  {_DIM}Enter number (1-{len(choices)}){_RESET}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            if default:
                return default
            sys.exit(1)
        if not raw and default:
            return default
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                selected = choices[idx]
                print(f"  {_GREEN}✓{_RESET} {selected['label']}")
                return selected["value"]
        except ValueError:
            pass
        print(f"  {_RED}Please enter a number between 1 and {len(choices)}.{_RESET}")


def select_or_create(
    message: str,
    choices: Sequence[Dict[str, str]],
    *,
    create_label: str = "Create new",
    create_fn: Optional[Callable[[], Optional[str]]] = None,
) -> str:
    """Selection list with a 'create new' option at the end."""
    full_choices = list(choices) + [{"value": "__create__", "label": f"{_YELLOW}+ {create_label}{_RESET}"}]
    result = select(message, full_choices)
    if result == "__create__" and create_fn:
        created = create_fn()
        if created is None:
            sys.exit(1)
        return created
    return result


def status(message: str, *, ok: bool = True) -> None:
    """Print a status line (checkmark or cross)."""
    icon = f"{_GREEN}✓{_RESET}" if ok else f"{_RED}✗{_RESET}"
    print(f"  {icon} {message}")


def step(message: str) -> None:
    """Print a step header."""
    print(f"\n{_BOLD}{message}{_RESET}")


def info(message: str) -> None:
    """Print an informational message."""
    print(f"  {_DIM}{message}{_RESET}")
