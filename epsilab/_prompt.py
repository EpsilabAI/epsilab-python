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


def _read_key() -> str:
    """Read a single keypress, returning arrow keys as 'up'/'down' and
    enter as 'enter'.  Falls back to simple input if termios is unavailable."""
    import tty
    import termios

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "escape"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _render_choices(choices: Sequence[Dict[str, str]], cursor: int) -> str:
    lines = []
    for i, choice in enumerate(choices):
        if i == cursor:
            lines.append(f"  {_GREEN}❯{_RESET} {_BOLD}{choice['label']}{_RESET}")
        else:
            lines.append(f"    {_DIM}{choice['label']}{_RESET}")
    return "\n".join(lines)


def select(
    message: str,
    choices: Sequence[Dict[str, str]],
    *,
    default: Optional[str] = None,
) -> str:
    """Arrow-key selection list. Each choice has 'value' and 'label' keys.

    Falls back to numbered input if the terminal doesn't support raw mode.
    Returns the selected 'value'.
    """
    cursor = 0
    if default:
        for i, c in enumerate(choices):
            if c["value"] == default:
                cursor = i
                break

    # Try arrow-key mode; fall back to numbered input
    try:
        import termios as _termios  # noqa: F401
    except ImportError:
        return _select_fallback(message, choices, default=default)

    if not sys.stdin.isatty():
        return _select_fallback(message, choices, default=default)

    print(f"{_label(message)}")
    n = len(choices)
    sys.stdout.write(_render_choices(choices, cursor) + "\n")
    sys.stdout.write(f"  {_DIM}↑/↓ to navigate, enter to select{_RESET}")
    sys.stdout.flush()

    # +1 for the hint line
    total_lines = n + 1

    try:
        while True:
            key = _read_key()
            if key == "up":
                cursor = (cursor - 1) % n
            elif key == "down":
                cursor = (cursor + 1) % n
            elif key == "enter":
                break
            else:
                continue

            # Move cursor up to the start of the choices block, clear, and redraw
            sys.stdout.write(f"\033[{total_lines}A")
            for _ in range(total_lines):
                sys.stdout.write("\033[2K\n")
            sys.stdout.write(f"\033[{total_lines}A")
            sys.stdout.write(_render_choices(choices, cursor) + "\n")
            sys.stdout.write(f"  {_DIM}↑/↓ to navigate, enter to select{_RESET}")
            sys.stdout.flush()
    except (KeyboardInterrupt, EOFError):
        print()
        if default:
            return default
        sys.exit(1)

    # Clear the choices block and print the selection
    sys.stdout.write(f"\033[{total_lines}A")
    for _ in range(total_lines):
        sys.stdout.write("\033[2K\n")
    sys.stdout.write(f"\033[{total_lines}A")

    selected = choices[cursor]
    print(f"  {_GREEN}✓{_RESET} {selected['label']}")
    return selected["value"]


def _select_fallback(
    message: str,
    choices: Sequence[Dict[str, str]],
    *,
    default: Optional[str] = None,
) -> str:
    """Numbered input fallback for non-TTY environments."""
    print(f"{_label(message)}")
    for i, choice in enumerate(choices, 1):
        print(f"    {i}. {choice['label']}")

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
