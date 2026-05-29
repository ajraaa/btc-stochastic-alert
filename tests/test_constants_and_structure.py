"""Structural / AST tests for the BTC Stochastic DCA Monitor source file.

Task 13.2 / Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 9.1, 9.2,
9.3, 9.4, 9.5, 10.3, 11.6.

These tests treat ``btc_stochastic_monitor.py`` as data: they parse the source
with :mod:`ast` and assert structural invariants that the rest of the suite
cannot easily verify behaviourally. Specifically they enforce:

* **Constants-first layout (REQ-1.1, REQ-1.2..REQ-1.8):** every required
  module-level constant is a plain ``ast.Assign`` whose line number precedes the
  first function/class definition, so an operator edits configuration in one
  place at the top of the file.
* **Function inventory (REQ-9.1..REQ-9.3):** the module defines every named
  function the design and requirements call for.
* **Documentation (REQ-9.4):** each of ``send_telegram_notification``,
  ``process_data``, ``on_open``, ``on_message``, and ``on_close`` opens with a
  docstring. (``ast`` does not preserve ``#`` comments, so the comment-based
  half of REQ-9.4 is verified via the leading docstring, which all five carry.)
* **No direct stdout/stderr writes (REQ-11.6):** the source contains zero
  ``print(...)`` calls and zero ``sys.stdout.write`` / ``sys.stderr.write``
  calls. Passing ``sys.stdout`` as an argument (e.g.
  ``logging.StreamHandler(sys.stdout)``) is explicitly allowed -- only ``.write``
  *calls* on those streams are forbidden.
* **Dependency manifest (REQ-10.3):** ``requirements.txt`` lists the runtime
  dependencies ``pandas``, ``requests``, and ``websocket-client``.

  NOTE: ``pandas-ta`` has been deprecated from the project. Its transitive build
  dependency (``numba``) cannot build on Python 3.14+, so the Stochastic
  Oscillator is now computed with native vanilla pandas rolling windows. These
  tests therefore assert ``pandas-ta`` is *absent* from ``requirements.txt``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import btc_stochastic_monitor as bsm


# ---------------------------------------------------------------------------
# Fixtures / shared parse artifacts
# ---------------------------------------------------------------------------

# Resolve the module source robustly from the imported module's ``__file__``.
MODULE_PATH = pathlib.Path(bsm.__file__).resolve()
SOURCE = MODULE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

# Repository root is the parent of the tests/ directory; requirements.txt lives
# alongside the module at the repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"


# Required module-level constants that must appear (as ``ast.Assign``) before any
# function or class definition (REQ-1.1..REQ-1.8).
REQUIRED_CONSTANTS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "SYMBOL",
    "INTERVAL",
    "STOCH_K_LENGTH",
    "STOCH_K_SMOOTH",
    "STOCH_D_SMOOTH",
    "STOCH_K_COL",
    "STOCH_D_COL",
    "OVERSOLD_THRESHOLD",
    "HISTORY_LIMIT",
    "RECONNECT_DELAY_S",
    "BACKUP_LOG_FILE",
    "STALE_BUFFER_THRESHOLD_MS",
    "REST_TIMEOUT_S",
    "TELEGRAM_TIMEOUT_S",
    "BINANCE_REST_URL",
    "BINANCE_WS_URL",
]

# Functions the module must define (REQ-9.1..REQ-9.3 and design Function
# Signatures).
REQUIRED_FUNCTIONS = [
    "validate_config",
    "init_logger",
    "bootstrap_history",
    "update_buffer",
    "compute_stochastic",
    "evaluate_trigger",
    "format_alert",
    "send_telegram_notification",
    "on_open",
    "on_message",
    "on_close",
    "process_data",
    "check_buffer_freshness",
    "run_forever",
    "main",
]

# Functions that must open with a docstring (REQ-9.4). ``ast`` does not preserve
# ``#`` comments, so the comment-based requirement is verified through the
# leading docstring, which all five of these functions carry.
DOCSTRING_FUNCTIONS = [
    "send_telegram_notification",
    "process_data",
    "on_open",
    "on_message",
    "on_close",
]

_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _first_def_lineno(tree: ast.Module) -> int:
    """Return the smallest line number among top-level def/class statements."""
    linenos = [node.lineno for node in tree.body if isinstance(node, _DEF_TYPES)]
    assert linenos, "module defines no functions or classes -- unexpected layout"
    return min(linenos)


def _top_level_assign_linenos() -> dict[str, list[int]]:
    """Map each top-level assigned Name to the line numbers it is assigned on."""
    assignments: dict[str, list[int]] = {}
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(node.lineno)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            # Plain Assign is expected in this module, but tolerate an annotated
            # assignment as well so the test tracks the structural intent.
            assignments.setdefault(node.target.id, []).append(node.lineno)
    return assignments


def _top_level_functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Map each top-level function name to its def node."""
    return {
        node.name: node
        for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


# ---------------------------------------------------------------------------
# Constants-first layout (REQ-1.1..REQ-1.8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("const_name", REQUIRED_CONSTANTS)
def test_required_constant_assigned_before_any_def(const_name: str) -> None:
    """Each required constant is a top-level Assign before the first def/class."""
    first_def = _first_def_lineno(TREE)
    assignments = _top_level_assign_linenos()

    assert const_name in assignments, (
        f"required constant {const_name!r} is not assigned at module top level"
    )
    # At least one assignment of this constant must precede every def/class so an
    # operator finds all configuration above the logic (REQ-1.1).
    earliest = min(assignments[const_name])
    assert earliest < first_def, (
        f"constant {const_name!r} assigned on line {earliest} is not placed "
        f"before the first function/class definition on line {first_def}"
    )


def test_all_required_constants_present() -> None:
    """Every required constant name is assigned at module top level (REQ-1.1)."""
    assignments = _top_level_assign_linenos()
    missing = [name for name in REQUIRED_CONSTANTS if name not in assignments]
    assert not missing, f"missing required module-level constants: {missing}"


# ---------------------------------------------------------------------------
# Function inventory (REQ-9.1..REQ-9.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func_name", REQUIRED_FUNCTIONS)
def test_module_defines_required_function(func_name: str) -> None:
    """The module defines each required function by name (REQ-9.1..REQ-9.3)."""
    functions = _top_level_functions()
    assert func_name in functions, (
        f"module does not define a top-level function named {func_name!r}"
    )


# ---------------------------------------------------------------------------
# Docstrings on the documented functions (REQ-9.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func_name", DOCSTRING_FUNCTIONS)
def test_documented_function_has_docstring(func_name: str) -> None:
    """Each documented function opens with a docstring (REQ-9.4)."""
    functions = _top_level_functions()
    assert func_name in functions, f"function {func_name!r} not defined"
    node = functions[func_name]
    assert ast.get_docstring(node) is not None, (
        f"function {func_name!r} is missing a leading docstring (REQ-9.4)"
    )


# ---------------------------------------------------------------------------
# No print() calls and no sys.stdout/sys.stderr .write() calls (REQ-11.6)
# ---------------------------------------------------------------------------


def test_source_has_no_print_calls() -> None:
    """No ``print(...)`` call appears anywhere in the AST (REQ-11.6)."""
    print_calls = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert not print_calls, (
        f"found {len(print_calls)} print(...) call(s) at lines "
        f"{[n.lineno for n in print_calls]}; components must log instead (REQ-11.6)"
    )


def test_source_text_has_no_print_token() -> None:
    """Complementary text scan: the literal ``print(`` never appears (REQ-11.6)."""
    assert "print(" not in SOURCE, "source text contains a 'print(' token (REQ-11.6)"


def test_source_has_no_stdout_stderr_write_calls() -> None:
    """No ``.write`` call on ``sys.stdout`` / ``sys.stderr`` exists (REQ-11.6).

    Passing ``sys.stdout`` as an argument (e.g. ``StreamHandler(sys.stdout)``) is
    allowed -- only a ``.write`` *call* whose receiver resolves to
    ``sys.stdout`` / ``sys.stderr`` is forbidden.
    """
    offending: list[int] = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Looking for ``<receiver>.write(...)`` where <receiver> is
        # ``sys.stdout`` or ``sys.stderr`` (an Attribute whose attr is
        # ``stdout`` / ``stderr``).
        if isinstance(func, ast.Attribute) and func.attr == "write":
            receiver = func.value
            if (
                isinstance(receiver, ast.Attribute)
                and receiver.attr in {"stdout", "stderr"}
            ):
                offending.append(node.lineno)
    assert not offending, (
        f"found sys.stdout/sys.stderr .write() call(s) at lines {offending}; "
        f"components must log instead (REQ-11.6)"
    )


# ---------------------------------------------------------------------------
# Dependency manifest (REQ-10.3)
# ---------------------------------------------------------------------------


def _requirement_package_names() -> list[str]:
    """Return the lowercased package names listed in requirements.txt.

    Comment lines (``#...``) and blank lines are ignored. For each remaining
    line the package name is the leading token before any version specifier
    (``>=``, ``==``, ``<``, ``~=``, ``!=``, ``>``, ``<=``) or inline comment.
    """
    names: list[str] = []
    for raw_line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip an inline comment if present.
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # Split off the version specifier; the package name is the leading run of
        # characters not part of a specifier operator.
        name = line
        for op in ("===", "==", ">=", "<=", "~=", "!=", ">", "<"):
            idx = name.find(op)
            if idx != -1:
                name = name[:idx]
        names.append(name.strip().lower())
    return names


@pytest.mark.parametrize("package", ["pandas", "requests", "websocket-client"])
def test_requirements_lists_runtime_dependency(package: str) -> None:
    """requirements.txt lists each required runtime dependency (REQ-10.3)."""
    names = _requirement_package_names()
    assert package in names, (
        f"requirements.txt does not list runtime dependency {package!r}; "
        f"found packages: {names}"
    )


def test_requirements_does_not_list_pandas_ta() -> None:
    """requirements.txt must NOT list the deprecated ``pandas-ta`` (REQ-10.3).

    pandas-ta was removed because its build dependency (numba) cannot build on
    Python 3.14+. The Stochastic is computed with native pandas instead.
    """
    names = _requirement_package_names()
    assert "pandas-ta" not in names, (
        "requirements.txt lists 'pandas-ta', which was deprecated from the "
        "project; the Stochastic is now computed with native pandas"
    )
    # Belt-and-suspenders text scan over non-comment lines: ensure no stray
    # reference to pandas-ta survives in the manifest.
    for raw_line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert "pandas-ta" not in line.lower(), (
            f"non-comment requirements.txt line references pandas-ta: {raw_line!r}"
        )
