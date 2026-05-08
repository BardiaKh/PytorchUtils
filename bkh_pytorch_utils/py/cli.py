"""Inline CLI parameters for scripts that double as notebooks.

Use `param()` to declare a flag where you'd otherwise hardcode a value:

    from bkh_pytorch_utils import param as P

    batch_size = P("batch_size", 32)
    lr         = P("lr", 1e-4, help="learning rate")
    use_amp    = P("use_amp", True)               # --use-amp / --no-use-amp
    augs       = P("augs", ["flip", "rot"])       # --augs flip rot
    seed       = P("seed", None, type=int)        # type= when default is None

In a notebook (or under pytest) `param` returns the default unchanged. As a
script, it overrides from sys.argv. Passing `--help` / `-h` prints all
declared params (discovered via AST scan of the caller) and exits.
"""
from __future__ import annotations

import ast
import builtins
import inspect
import os
import sys
from typing import Any, Callable, Optional, TypeVar

from .utils import is_notebook_running

T = TypeVar("T")

_REGISTRY: dict[str, dict] = {}
_HELP_DONE = False


def _under_pytest() -> bool:
    return "pytest" in os.path.basename(sys.argv[0]) or "PYTEST_CURRENT_TEST" in os.environ


def _passive_mode() -> bool:
    return is_notebook_running() or _under_pytest()


def _scan(argv: list[str], flag: str) -> tuple[Optional[int], Optional[str]]:
    for i, a in enumerate(argv):
        if a == flag:
            return (i, None)
        if a.startswith(flag + "="):
            return (i, a[len(flag) + 1:])
    return (None, None)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _ast_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return "..."


def _ast_scan_caller(filename: str, lineno: int) -> list[tuple[str, Any, str]]:
    """Parse `filename`, locate the call at `lineno`, and return every
    sibling call with the same function name as (param_name, default, help)."""
    try:
        with open(filename) as f:
            src = f.read()
        tree = ast.parse(src)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    target_name: Optional[str] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node, "lineno", None) == lineno:
            func = node.func
            if isinstance(func, ast.Name):
                target_name = func.id
                break
            if isinstance(func, ast.Attribute):
                target_name = func.attr
                break
    if target_name is None:
        return []

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        fname = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else None
        )
        if fname != target_name:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        name = first.value
        default = _ast_literal(node.args[1]) if len(node.args) >= 2 else None
        help_text = ""
        for kw in node.keywords:
            if kw.arg == "help":
                v = _ast_literal(kw.value)
                if isinstance(v, str):
                    help_text = v
        found.append((name, default, help_text))
    return found


def _print_help_and_exit() -> None:
    print(f"usage: {os.path.basename(sys.argv[0])} [options]\n")
    print("options:")
    print(f"  {'-h, --help':30s} show this help message and exit")
    for name, info in _REGISTRY.items():
        d = info["default"]
        flag = info["flag"]
        h = info["help"]
        line = f"{flag} / --no-{name.replace('_', '-')}" if isinstance(d, bool) else flag
        suffix = f"(default: {d!r})"
        print(f"  {line:30s} {h + '  ' if h else ''}{suffix}")
    sys.exit(0)


def param(
    name: str,
    default: T = None,
    *,
    type: Optional[Callable[[str], Any]] = None,
    help: str = "",
) -> T:
    """Declare a CLI parameter. Returns the default in notebooks/pytest;
    otherwise reads from sys.argv. `--help` / `-h` prints usage and exits."""
    flag_name = name.replace("_", "-")
    flag = f"--{flag_name}"
    _REGISTRY[name] = {"default": default, "type": type, "help": help, "flag": flag}

    if _passive_mode():
        return default

    argv = sys.argv[1:]

    global _HELP_DONE
    if not _HELP_DONE and any(a in ("--help", "-h") for a in argv):
        _HELP_DONE = True
        try:
            frame = inspect.currentframe().f_back
            for nm, dflt, hlp in _ast_scan_caller(frame.f_code.co_filename, frame.f_lineno):
                if nm not in _REGISTRY:
                    fn = nm.replace("_", "-")
                    _REGISTRY[nm] = {"default": dflt, "type": None, "help": hlp, "flag": f"--{fn}"}
        except Exception:
            pass
        _print_help_and_exit()

    is_bool = isinstance(default, bool)
    is_seq = isinstance(default, (list, tuple))

    if type is not None:
        cast = type
    elif is_bool:
        cast = lambda x: str(x).lower() in ("1", "true", "yes", "y")
    elif is_seq and default:
        cast = builtins.type(default[0])
    elif default is not None and not is_seq:
        cast = builtins.type(default)
    else:
        cast = str

    if is_bool:
        idx, val = _scan(argv, flag)
        if idx is not None:
            return cast(val) if val is not None else True
        if _scan(argv, f"--no-{flag_name}")[0] is not None:
            return False
        return default

    if is_seq:
        idx, val = _scan(argv, flag)
        if idx is None:
            return default
        if val is not None:
            return [cast(x) for x in val.split(",")]
        out = []
        for a in argv[idx + 1:]:
            if a.startswith("-") and not _is_number(a):
                break
            out.append(cast(a))
        return out

    idx, val = _scan(argv, flag)
    if idx is None:
        return default
    if val is not None:
        return cast(val)
    if idx + 1 >= len(argv):
        raise ValueError(f"{flag} requires a value")
    return cast(argv[idx + 1])
