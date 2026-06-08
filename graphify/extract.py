"""Deterministic structural extraction from source code using tree-sitter. Outputs nodes+edges dicts."""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .cache import load_cached, save_cached

_RECURSION_LIMIT = 10_000

# Language built-in globals that AST may classify as call targets when used as
# constructors or coercion functions (e.g. String(x), Number(x), Boolean(x)).
# Without this filter they become god-nodes accumulating spurious edges from
# every call site. Filter applied at same-file and cross-file resolution.
# See issue #726.
_LANGUAGE_BUILTIN_GLOBALS: frozenset[str] = frozenset({
    # JavaScript / TypeScript ECMAScript built-ins
    "String", "Number", "Boolean", "Object", "Array", "Symbol", "BigInt",
    "Date", "RegExp", "Error", "TypeError", "RangeError", "SyntaxError",
    "ReferenceError", "EvalError", "URIError",
    "Promise", "Map", "Set", "WeakMap", "WeakSet", "JSON", "Math",
    "Reflect", "Proxy", "Intl",
    "parseInt", "parseFloat", "isNaN", "isFinite",
    "encodeURIComponent", "decodeURIComponent", "encodeURI", "decodeURI",
    # Browser / Node common globals
    "URL", "URLSearchParams", "FormData", "Blob", "File",
    "Headers", "Request", "Response", "AbortController", "AbortSignal",
    "TextEncoder", "TextDecoder", "console",
    # Python built-in callables
    "str", "int", "float", "bool", "list", "dict", "set", "tuple", "bytes",
    "len", "range", "enumerate", "zip", "map", "filter", "sum", "min", "max",
    "print", "open", "isinstance", "type", "super", "sorted", "reversed",
    "any", "all", "abs", "round", "next", "iter", "hash", "id", "repr",
    "callable", "getattr", "setattr", "hasattr", "delattr", "vars", "dir",
})


def _raise_recursion_limit() -> None:
    if sys.getrecursionlimit() < _RECURSION_LIMIT:
        sys.setrecursionlimit(_RECURSION_LIMIT)


def _safe_extract(extractor: Callable, path: Path) -> dict:
    try:
        return extractor(path)
    except RecursionError:
        print(f"  warning: skipped {path} (recursion limit exceeded)", file=sys.stderr, flush=True)
        return {"nodes": [], "edges": [], "error": "recursion_limit_exceeded"}
    except Exception as e:
        if os.environ.get("GRAPHIFY_DEBUG"):
            import traceback
            traceback.print_exc(file=sys.stderr)
        print(f"  warning: skipped {path} ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return {"nodes": [], "edges": [], "error": f"{type(e).__name__}: {e}"}


def _make_id(*parts: str) -> str:
    r"""Build a stable node ID from one or more name parts.

    Preserves Unicode letters/digits (CJK, Cyrillic, Arabic, accented Latin,
    etc.) so non-ASCII identifiers produce distinct IDs and don't collapse to
    a single per-file node (#811). NFKC normalization ensures composed and
    decomposed forms of the same character (e.g. é vs e+combining-acute)
    produce the same ID. Must stay in sync with build._normalize_id.
    """
    combined = "_".join(p.strip("_.") for p in parts if p)
    combined = unicodedata.normalize("NFKC", combined)
    cleaned = re.sub(r"[^\w]+", "_", combined, flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_").casefold()


def _file_stem(path: Path) -> str:
    """Return a stem qualified with the parent directory name to avoid ID collisions
    when multiple files share the same filename in different directories (#550)."""
    parent = path.parent.name
    if parent and parent not in (".", ""):
        return f"{parent}.{path.stem}"
    return path.stem


def _file_node_id(rel_path: Path) -> str:
    """File-level node ID matching the skill.md spec: ``{parent_dir}_{stem}`` —
    one parent directory level, no extension. ``rel_path`` MUST be relative to
    the project root so top-level files collapse to a bare stem (``setup.py`` ->
    ``setup``) instead of picking up the root directory name. This must equal the
    ID semantic subagents generate, or AST and semantic extraction split a file
    into two disconnected ghost nodes (#1033)."""
    return _make_id(_file_stem(rel_path))


_TSCONFIG_ALIAS_CACHE: dict[str, dict[str, str]] = {}
_WORKSPACE_PACKAGE_CACHE: dict[str, dict[str, Path]] = {}
_JS_CACHE_BYPASS_SUFFIXES = {".js", ".jsx", ".mjs", ".ts", ".tsx", ".vue", ".svelte"}
_JS_RESOLVE_EXTS = (".ts", ".tsx", ".svelte", ".js", ".jsx", ".mjs")
_JS_INDEX_FILES = ("index.ts", "index.tsx", "index.svelte", "index.js", "index.jsx", "index.mjs")


REFERENCE_CONTEXTS = frozenset({
    "field", "parameter_type", "return_type", "generic_arg", "attribute", "value", "type",
})


def _source_location(line: int | str | None) -> str | None:
    if line is None:
        return None
    if isinstance(line, str):
        return line if line.startswith("L") else f"L{line}"
    return f"L{line}"


def _semantic_reference_edge(
    source: str,
    target: str,
    context: str,
    source_file: str,
    line: int | str | None,
) -> dict:
    if context not in REFERENCE_CONTEXTS:
        raise ValueError(f"unknown reference context: {context}")
    return {
        "source": source,
        "target": target,
        "relation": "references",
        "context": context,
        "confidence": "EXTRACTED",
        "source_file": source_file,
        "source_location": _source_location(line),
        "weight": 1.0,
    }


def _resolve_js_import_path(candidate: Path) -> Path:
    """Resolve a JS/TS/Svelte import target to a local file when it exists."""
    candidate = Path(os.path.normpath(candidate))
    if candidate.is_file():
        return candidate

    # TS ESM convention: imports often spell .js/.jsx while source is .ts/.tsx.
    if candidate.suffix == ".js":
        ts_candidate = candidate.with_suffix(".ts")
        if ts_candidate.is_file():
            return ts_candidate
    elif candidate.suffix == ".jsx":
        tsx_candidate = candidate.with_suffix(".tsx")
        if tsx_candidate.is_file():
            return tsx_candidate

    # Append extensions to the full filename, which covers extensionless imports,
    # multi-dot helpers, and Svelte 5 rune files like Foo.svelte.ts.
    for ext in _JS_RESOLVE_EXTS:
        with_ext = candidate.parent / f"{candidate.name}{ext}"
        if with_ext.is_file():
            return with_ext

    # Only fall back to directory indexes after file candidates lose.
    if candidate.is_dir():
        for index_name in _JS_INDEX_FILES:
            index_candidate = candidate / index_name
            if index_candidate.is_file():
                return index_candidate

    return candidate


def _strip_jsonc(text: str) -> str:
    """Strip // line comments, /* */ block comments, and trailing commas from JSONC.

    Preserves string contents (including // and /* inside strings) by skipping over
    quoted spans first. Required for tsconfig.json files generated by SvelteKit,
    NestJS, Vite, T3, Astro, etc., which use JSONC by default (#700).
    """
    # Remove block and line comments while leaving string literals untouched.
    pattern = re.compile(
        r'"(?:\\.|[^"\\])*"'    # double-quoted string (with escapes)
        r"|/\*.*?\*/"           # /* block comment */
        r"|//[^\n]*",           # // line comment
        re.DOTALL,
    )

    def _replace(match: re.Match) -> str:
        token = match.group(0)
        if token.startswith('"'):
            return token
        return ""

    stripped = pattern.sub(_replace, text)
    # Remove trailing commas before } or ] (allowing whitespace between).
    stripped = re.sub(r",(\s*[}\]])", r"\1", stripped)
    return stripped


def _read_tsconfig_aliases(tsconfig: Path, base_dir: Path, seen: set) -> dict[str, str]:
    """Recursively read path aliases from a tsconfig, following extends chains.

    Child config paths override parent. Circular extends are detected via seen set.
    npm package configs (e.g. @tsconfig/svelte) are skipped since they're not on disk.
    Handles JSONC (comments + trailing commas) which is the default tsconfig format
    for SvelteKit, NestJS, Vite, T3, Astro, etc. (#700).
    """
    if str(tsconfig) in seen:
        return {}
    seen.add(str(tsconfig))
    try:
        raw = tsconfig.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  warning: could not read {tsconfig} ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(_strip_jsonc(raw))
        except json.JSONDecodeError as e:
            print(f"  warning: failed to parse {tsconfig} as JSON/JSONC ({e.msg} at line {e.lineno} col {e.colno})", file=sys.stderr, flush=True)
            return {}
    except Exception as e:
        print(f"  warning: failed to parse {tsconfig} ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return {}

    aliases: dict[str, str] = {}
    # `extends` may be a string or, since TypeScript 5.0, an array of paths.
    # For an array, parents are processed in order with later entries
    # overriding earlier ones; the extending config (paths below) overrides
    # all parents. Without the list branch, an array `extends` raised
    # `AttributeError: 'list' object has no attribute 'startswith'`, which
    # _safe_extract turned into a skip of the whole file.
    extends = data.get("extends")
    if isinstance(extends, str):
        extends_list = [extends]
    elif isinstance(extends, list):
        extends_list = [e for e in extends if isinstance(e, str)]
    else:
        extends_list = []
    for ext in extends_list:
        # Skip scoped npm package configs (e.g. @tsconfig/svelte) — not on disk.
        if not ext or ext.startswith("@"):
            continue
        extended_path = (base_dir / ext).resolve()
        if not extended_path.suffix:
            extended_path = extended_path.with_suffix(".json")
        if extended_path.exists():
            aliases.update(_read_tsconfig_aliases(extended_path, extended_path.parent, seen))

    paths = data.get("compilerOptions", {}).get("paths", {})
    for alias, targets in paths.items():
        if not targets:
            continue
        alias_prefix = alias.rstrip("/*")
        target_base = targets[0].rstrip("/*")
        aliases[alias_prefix] = str(base_dir / target_base)

    return aliases


def _load_tsconfig_aliases(start_dir: Path) -> dict[str, str]:
    """Walk up from start_dir to find tsconfig.json and return compilerOptions.paths aliases.

    Follows extends chains so SvelteKit/Nuxt/NestJS inherited aliases are included.
    Returns a dict mapping alias prefix (e.g. "@/") to resolved base dir (e.g. "src/").
    Result is cached by tsconfig path string.
    """
    current = start_dir.resolve()
    for candidate in [current, *current.parents]:
        tsconfig = candidate / "tsconfig.json"
        if tsconfig.exists():
            key = str(tsconfig)
            if key not in _TSCONFIG_ALIAS_CACHE:
                _TSCONFIG_ALIAS_CACHE[key] = _read_tsconfig_aliases(tsconfig, candidate, seen=set())
            return _TSCONFIG_ALIAS_CACHE[key]
    return {}


def _find_workspace_root(start_dir: Path) -> Path | None:
    current = start_dir.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pnpm-workspace.yaml").exists():
            return candidate
    return None


def _workspace_globs(workspace_file: Path) -> list[str]:
    globs: list[str] = []
    in_packages = False
    for raw_line in workspace_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("packages:"):
            in_packages = True
            continue
        if in_packages and line.startswith("-"):
            value = line[1:].strip().strip("'\"")
            if value and not value.startswith("!"):
                globs.append(value)
            continue
        if in_packages and not raw_line.startswith((" ", "\t")):
            break
    return globs


def _load_workspace_packages(start_dir: Path) -> dict[str, Path]:
    root = _find_workspace_root(start_dir)
    if root is None:
        return {}
    key = str(root)
    if key in _WORKSPACE_PACKAGE_CACHE:
        return _WORKSPACE_PACKAGE_CACHE[key]

    packages: dict[str, Path] = {}
    for pattern in _workspace_globs(root / "pnpm-workspace.yaml"):
        package_dirs: list[Path] = [root] if pattern in (".", "./") else list(root.glob(pattern))
        for package_dir in package_dirs:
            manifest = package_dir / "package.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                continue
            name = data.get("name")
            if isinstance(name, str) and name:
                packages[name] = package_dir
    _WORKSPACE_PACKAGE_CACHE[key] = packages
    return packages


def _package_entry_candidates(package_dir: Path, subpath: str) -> list[Path]:
    manifest = package_dir / "package.json"
    manifest_data: dict[str, Any] = {}
    try:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        pass

    if subpath:
        return [package_dir / subpath]

    exports = manifest_data.get("exports")
    if isinstance(exports, str):
        return [package_dir / exports]
    if isinstance(exports, dict):
        dot_export = exports.get(".")
        if isinstance(dot_export, str):
            return [package_dir / dot_export]
        if isinstance(dot_export, dict):
            for key in ("types", "import", "default", "svelte"):
                value = dot_export.get(key)
                if isinstance(value, str):
                    return [package_dir / value]

    candidates: list[Path] = []
    for key in ("svelte", "module", "main", "types"):
        value = manifest_data.get(key)
        if isinstance(value, str):
            candidates.append(package_dir / value)
    candidates.append(package_dir / "src/index")
    candidates.append(package_dir / "index")
    return candidates


def _resolve_workspace_import(raw: str, start_dir: Path) -> Path | None:
    packages = _load_workspace_packages(start_dir)
    for package_name, package_dir in packages.items():
        if raw == package_name:
            subpath = ""
        elif raw.startswith(package_name + "/"):
            subpath = raw[len(package_name) + 1:]
        else:
            continue
        for candidate in _package_entry_candidates(package_dir, subpath):
            resolved = _resolve_js_import_path(candidate)
            if resolved.is_file():
                return resolved
    return None


def _resolve_js_module_path(raw: str | Path, start_dir: Path | None = None) -> Path | None:
    """Resolve a JS/TS module path or specifier to a local source file.

    With a Path argument this preserves the path-based helper API used by
    import-extension tests. With a string plus start_dir it resolves JS/TS
    module specifiers including relative paths, tsconfig aliases, and workspace
    packages.
    """
    if isinstance(raw, Path):
        return _resolve_js_import_path(raw)
    if start_dir is None:
        return _resolve_js_import_path(Path(raw))
    if raw.startswith("."):
        return _resolve_js_import_path(start_dir / raw)

    aliases = _load_tsconfig_aliases(start_dir)
    for alias_prefix, alias_base in aliases.items():
        if raw == alias_prefix or raw.startswith(alias_prefix + "/"):
            rest = raw[len(alias_prefix):].lstrip("/")
            return _resolve_js_import_path(Path(os.path.normpath(Path(alias_base) / rest)))

    return _resolve_workspace_import(raw, start_dir)


# ── LanguageConfig dataclass ─────────────────────────────────────────────────

@dataclass
class LanguageConfig:
    ts_module: str                                   # e.g. "tree_sitter_python"
    ts_language_fn: str = "language"                 # attr to call: e.g. tslang.language()

    class_types: frozenset = frozenset()
    function_types: frozenset = frozenset()
    import_types: frozenset = frozenset()
    call_types: frozenset = frozenset()
    static_prop_types: frozenset = frozenset()
    helper_fn_names: frozenset = frozenset()
    container_bind_methods: frozenset = frozenset()
    event_listener_properties: frozenset = frozenset()

    # Name extraction
    name_field: str = "name"
    name_fallback_child_types: tuple = ()

    # Body detection
    body_field: str = "body"
    body_fallback_child_types: tuple = ()   # e.g. ("declaration_list", "compound_statement")

    # Call name extraction
    call_function_field: str = "function"           # field on call node for callee
    call_accessor_node_types: frozenset = frozenset()  # member/attribute nodes
    call_accessor_field: str = "attribute"          # field on accessor for method name

    # Stop recursion at these types in walk_calls
    function_boundary_types: frozenset = frozenset()

    # Import handler: called for import nodes instead of generic handling
    import_handler: Callable | None = None

    # Optional custom name resolver for functions (C, C++ declarator unwrapping)
    resolve_function_name_fn: Callable | None = None

    # Extra label formatting for functions: if True, functions get "name()" label
    function_label_parens: bool = True

    # Extra walk hook called after generic dispatch (for JS arrow functions, C# namespaces, etc.)
    extra_walk_fn: Callable | None = None


# ── Generic helpers ───────────────────────────────────────────────────────────

def _read_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


_PYTHON_TYPE_CONTAINERS = frozenset({
    "list", "dict", "set", "tuple", "frozenset", "type",
    "List", "Dict", "Set", "Tuple", "FrozenSet", "Type",
    "Optional", "Union", "Sequence", "Iterable", "Mapping", "MutableMapping",
    "Iterator", "Callable", "Awaitable", "AsyncIterable", "AsyncIterator", "Coroutine",
    "Generator", "AsyncGenerator", "ContextManager", "AsyncContextManager",
    "Annotated", "ClassVar", "Final", "Literal", "Concatenate", "ParamSpec", "TypeVar",
    "None", "Ellipsis",
})

# Scalar builtins and test-mock names that appear as type annotations but carry
# no useful semantic meaning as graph nodes (#1147). Suppressed at the annotation
# walker level so they are never created as nodes or emitted as edges.
_PYTHON_ANNOTATION_NOISE = frozenset({
    # scalar builtins
    "str", "int", "float", "bool", "bytes", "bytearray", "complex", "object",
    "True", "False",
    # unittest.mock
    "MagicMock", "Mock", "AsyncMock", "NonCallableMock",
    "NonCallableMagicMock", "PropertyMock", "patch", "sentinel",
})


def _python_collect_type_refs(node, source: bytes, generic: bool, out: list[tuple[str, str]]) -> None:
    """Walk a Python type annotation; append (name, role) where role is 'type' or 'generic_arg'.

    Builtin/typing containers (list, dict, Optional, Union, …) are not emitted as refs themselves,
    but their nested type arguments still count as generic_arg.
    """
    if node is None:
        return
    t = node.type
    if t == "type":
        for c in node.children:
            if c.is_named:
                _python_collect_type_refs(c, source, generic, out)
        return
    if t == "identifier":
        name = _read_text(node, source)
        if name and name not in _PYTHON_TYPE_CONTAINERS and name not in _PYTHON_ANNOTATION_NOISE:
            out.append((name, "generic_arg" if generic else "type"))
        return
    if t == "attribute":
        tail = _read_text(node, source).rsplit(".", 1)[-1]
        if tail and tail not in _PYTHON_TYPE_CONTAINERS and tail not in _PYTHON_ANNOTATION_NOISE:
            out.append((tail, "generic_arg" if generic else "type"))
        return
    if t == "generic_type":
        for c in node.children:
            if c.type == "identifier":
                container = _read_text(c, source)
                if container and container not in _PYTHON_TYPE_CONTAINERS and container not in _PYTHON_ANNOTATION_NOISE:
                    out.append((container, "generic_arg" if generic else "type"))
            elif c.type == "type_parameter":
                for sub in c.children:
                    if sub.is_named:
                        _python_collect_type_refs(sub, source, True, out)
        return
    if t == "subscript":
        value = node.child_by_field_name("value")
        if value is not None:
            _python_collect_type_refs(value, source, generic, out)
        for c in node.children:
            if c is value or not c.is_named:
                continue
            _python_collect_type_refs(c, source, True, out)
        return
    if node.is_named:
        for c in node.children:
            if c.is_named:
                _python_collect_type_refs(c, source, generic, out)


def _csharp_pre_scan_interfaces(root_node, source: bytes) -> set[str]:
    """Return names declared as `interface` in this C# compilation unit."""
    out: set[str] = set()
    stack = [root_node]
    while stack:
        n = stack.pop()
        if n.type == "interface_declaration":
            name_node = n.child_by_field_name("name")
            if name_node is not None:
                text = _read_text(name_node, source)
                if text:
                    out.add(text)
        stack.extend(n.children)
    return out


def _csharp_classify_base(name: str, interface_names: set[str]) -> str:
    """`implements` if the base name is an interface (declared or by I-prefix convention), else `inherits`."""
    if name in interface_names:
        return "implements"
    if len(name) >= 2 and name[0] == "I" and name[1].isupper():
        return "implements"
    return "inherits"


def _csharp_collect_type_refs(node, source: bytes, generic: bool, out: list[tuple[str, str]]) -> None:
    """Walk a C# type expression; append (name, role) tuples (role is 'type' or 'generic_arg')."""
    if node is None:
        return
    t = node.type
    if t == "predefined_type":
        return
    if t == "identifier":
        name = _read_text(node, source)
        if name:
            out.append((name, "generic_arg" if generic else "type"))
        return
    if t == "qualified_name":
        text = _read_text(node, source).rsplit(".", 1)[-1]
        if text:
            out.append((text, "generic_arg" if generic else "type"))
        return
    if t == "generic_name":
        name_child = node.child_by_field_name("name")
        if name_child is None:
            for sub in node.children:
                if sub.type == "identifier":
                    name_child = sub
                    break
        if name_child is not None:
            name = _read_text(name_child, source)
            if name:
                out.append((name, "generic_arg" if generic else "type"))
        for sub in node.children:
            if sub.type == "type_argument_list":
                for arg in sub.children:
                    if arg.is_named:
                        _csharp_collect_type_refs(arg, source, True, out)
        return
    if t in ("nullable_type", "array_type", "pointer_type", "ref_type"):
        for c in node.children:
            if c.is_named:
                _csharp_collect_type_refs(c, source, generic, out)
        return
    if node.is_named:
        for c in node.children:
            if c.is_named:
                _csharp_collect_type_refs(c, source, generic, out)


def _csharp_attribute_names(method_node, source: bytes) -> list[str]:
    """Collect attribute names from a C# method/declaration's attribute_list children."""
    names: list[str] = []
    for child in method_node.children:
        if child.type != "attribute_list":
            continue
        for attr in child.children:
            if attr.type != "attribute":
                continue
            name_node = attr.child_by_field_name("name")
            if name_node is None:
                for sub in attr.children:
                    if sub.type in ("identifier", "qualified_name"):
                        name_node = sub
                        break
            if name_node is not None:
                text = _read_text(name_node, source).rsplit(".", 1)[-1]
                if text:
                    names.append(text)
    return names


def _java_collect_type_refs(node, source: bytes, generic: bool, out: list[tuple[str, str]]) -> None:
    """Walk a Java type expression; append (name, role) tuples."""
    if node is None:
        return
    t = node.type
    if t in ("integral_type", "floating_point_type", "boolean_type", "void_type"):
        return
    if t == "type_identifier":
        name = _read_text(node, source)
        if name:
            out.append((name, "generic_arg" if generic else "type"))
        return
    if t == "scoped_type_identifier":
        text = _read_text(node, source).rsplit(".", 1)[-1]
        if text:
            out.append((text, "generic_arg" if generic else "type"))
        return
    if t == "generic_type":
        for c in node.children:
            if c.type in ("type_identifier", "scoped_type_identifier"):
                text = _read_text(c, source).rsplit(".", 1)[-1]
                if text:
                    out.append((text, "generic_arg" if generic else "type"))
                break
        for c in node.children:
            if c.type == "type_arguments":
                for arg in c.children:
                    if arg.is_named:
                        _java_collect_type_refs(arg, source, True, out)
        return
    if t == "array_type":
        for c in node.children:
            if c.is_named:
                _java_collect_type_refs(c, source, generic, out)
        return
    if node.is_named:
        for c in node.children:
            if c.is_named:
                _java_collect_type_refs(c, source, generic, out)


def _java_method_annotation_names(method_node, source: bytes) -> list[str]:
    """Collect annotation names from a Java method's `modifiers` child."""
    names: list[str] = []
    modifiers = None
    for child in method_node.children:
        if child.type == "modifiers":
            modifiers = child
            break
    if modifiers is None:
        return names
    for anno in modifiers.children:
        if anno.type not in ("marker_annotation", "annotation"):
            continue
        name_node = anno.child_by_field_name("name")
        if name_node is None:
            for sub in anno.children:
                if sub.type in ("identifier", "scoped_identifier", "type_identifier"):
                    name_node = sub
                    break
        if name_node is not None:
            text = _read_text(name_node, source).rsplit(".", 1)[-1]
            if text:
                names.append(text)
    return names


_GO_PREDECLARED_TYPES = frozenset({
    "bool", "byte", "complex64", "complex128", "error", "float32", "float64",
    "int", "int8", "int16", "int32", "int64", "rune", "string",
    "uint", "uint8", "uint16", "uint32", "uint64", "uintptr", "any", "comparable",
})


def _go_collect_type_refs(node, source: bytes, generic: bool, out: list[tuple[str, str]]) -> None:
    """Walk a Go type expression; append (name, role) tuples."""
    if node is None:
        return
    t = node.type
    if t == "type_identifier":
        text = _read_text(node, source)
        if text and text not in _GO_PREDECLARED_TYPES:
            out.append((text, "generic_arg" if generic else "type"))
        return
    if t == "qualified_type":
        text = _read_text(node, source).rsplit(".", 1)[-1]
        if text and text not in _GO_PREDECLARED_TYPES:
            out.append((text, "generic_arg" if generic else "type"))
        return
    if t == "generic_type":
        type_field = node.child_by_field_name("type")
        if type_field is not None:
            sub: list[tuple[str, str]] = []
            _go_collect_type_refs(type_field, source, generic, sub)
            out.extend(sub)
        for c in node.children:
            if c.type == "type_arguments":
                for arg in c.children:
                    if arg.is_named:
                        _go_collect_type_refs(arg, source, True, out)
        return
    if t in ("pointer_type", "slice_type", "array_type", "map_type",
             "channel_type", "parenthesized_type"):
        for c in node.children:
            if c.is_named:
                _go_collect_type_refs(c, source, generic, out)
        return
    if node.is_named:
        for c in node.children:
            if c.is_named:
                _go_collect_type_refs(c, source, generic, out)


def _php_name_text(node, source: bytes) -> str | None:
    """Return the unqualified name text from a PHP `name`/`qualified_name` node."""
    if node is None:
        return None
    return _read_text(node, source).rsplit("\\", 1)[-1] or None


def _php_collect_type_refs(node, source: bytes, generic: bool, out: list[tuple[str, str]]) -> None:
    """Walk a PHP type expression; append (name, role) tuples."""
    if node is None:
        return
    t = node.type
    if t == "primitive_type":
        return
    if t == "named_type":
        for c in node.children:
            if c.type in ("name", "qualified_name"):
                text = _php_name_text(c, source)
                if text:
                    out.append((text, "generic_arg" if generic else "type"))
                return
        return
    if t in ("name", "qualified_name"):
        text = _php_name_text(node, source)
        if text:
            out.append((text, "generic_arg" if generic else "type"))
        return
    if t in ("nullable_type", "union_type", "intersection_type", "optional_type"):
        for c in node.children:
            if c.is_named:
                _php_collect_type_refs(c, source, generic, out)
        return
    if node.is_named:
        for c in node.children:
            if c.is_named:
                _php_collect_type_refs(c, source, generic, out)


def _php_method_return_type_node(method_node):
    """Return the named_type/primitive_type node sitting after formal_parameters."""
    saw_params = False
    for c in method_node.children:
        if c.type == "formal_parameters":
            saw_params = True
            continue
        if saw_params and c.is_named and c.type not in ("compound_statement",):
            if c.type in ("named_type", "primitive_type", "nullable_type",
                          "union_type", "intersection_type", "optional_type"):
                return c
    return None


def _kotlin_user_type_name(user_type_node, source: bytes) -> str | None:
    """Return the head identifier text from a Kotlin user_type node (without generics)."""
    if user_type_node is None:
        return None
    for c in user_type_node.children:
        if c.type == "type_identifier":
            text = _read_text(c, source)
            return text or None
        if c.type == "identifier":
            text = _read_text(c, source)
            return text or None
        if c.type == "simple_user_type":
            for sub in c.children:
                if sub.type in ("identifier", "type_identifier"):
                    text = _read_text(sub, source)
                    return text or None
    return None


def _kotlin_collect_type_refs(node, source: bytes, generic: bool, out: list[tuple[str, str]]) -> None:
    """Walk a Kotlin type expression; append (name, role) tuples."""
    if node is None:
        return
    t = node.type
    if t in ("integral_literal", "boolean_literal"):
        return
    if t == "user_type":
        for c in node.children:
            if c.type in ("identifier", "type_identifier"):
                text = _read_text(c, source)
                if text:
                    out.append((text, "generic_arg" if generic else "type"))
                break
            if c.type == "simple_user_type":
                for sub in c.children:
                    if sub.type in ("identifier", "type_identifier"):
                        text = _read_text(sub, source)
                        if text:
                            out.append((text, "generic_arg" if generic else "type"))
                        break
                break
        for c in node.children:
            if c.type == "type_arguments":
                for arg in c.children:
                    if arg.type == "type_projection":
                        for sub in arg.children:
                            if sub.is_named:
                                _kotlin_collect_type_refs(sub, source, True, out)
                    elif arg.is_named:
                        _kotlin_collect_type_refs(arg, source, True, out)
        return
    if t in ("identifier", "type_identifier"):
        text = _read_text(node, source)
        if text:
            out.append((text, "generic_arg" if generic else "type"))
        return
    if t in ("nullable_type", "parenthesized_type", "type_reference"):
        for c in node.children:
            if c.is_named:
                _kotlin_collect_type_refs(c, source, generic, out)
        return
    if node.is_named:
        for c in node.children:
            if c.is_named:
                _kotlin_collect_type_refs(c, source, generic, out)


def _kotlin_property_type_node(property_node):
    """Find the user_type node within a Kotlin property_declaration."""
    for c in property_node.children:
        if c.type == "variable_declaration":
            for sub in c.children:
                if sub.type in ("user_type", "nullable_type", "type_reference"):
                    return sub
        if c.type in ("user_type", "nullable_type", "type_reference"):
            return c
    return None


def _kotlin_function_return_type_node(func_node):
    """Find the return-type node of a Kotlin function_declaration (the type after `: ` post-params)."""
    saw_params = False
    saw_colon = False
    for c in func_node.children:
        if c.type == "function_value_parameters":
            saw_params = True
            continue
        if saw_params and c.type == ":":
            saw_colon = True
            continue
        if saw_colon:
            if c.is_named:
                return c
    return None


def _swift_declaration_keyword(node) -> str | None:
    """Return the leading kind token for a Swift class_declaration: class/struct/enum/extension/actor."""
    for c in node.children:
        if not c.is_named and c.type in ("class", "struct", "enum", "extension", "actor"):
            return c.type
    return None


def _swift_pre_scan(root_node, source: bytes) -> tuple[set[str], set[str]]:
    """Pre-scan a Swift compilation unit and return (protocol_names, class_like_names)."""
    protocols: set[str] = set()
    classes: set[str] = set()
    stack = [root_node]
    while stack:
        n = stack.pop()
        if n.type == "protocol_declaration":
            name_node = n.child_by_field_name("name")
            if name_node is None:
                for c in n.children:
                    if c.type == "type_identifier":
                        name_node = c
                        break
            if name_node is not None:
                text = _read_text(name_node, source)
                if text:
                    protocols.add(text)
        elif n.type == "class_declaration":
            kw = _swift_declaration_keyword(n)
            if kw in ("class", "struct", "enum", "actor"):
                name_node = n.child_by_field_name("name")
                if name_node is not None:
                    text = _read_text(name_node, source)
                    if text:
                        classes.add(text)
        stack.extend(n.children)
    return protocols, classes


def _swift_classify_base(name: str, kind: str | None, is_first: bool,
                          protocols: set[str], classes: set[str]) -> str:
    """Classify a Swift inheritance_specifier entry as `inherits` or `implements`."""
    if name in protocols:
        return "implements"
    if name in classes:
        return "inherits"
    # struct/enum/extension/actor cannot inherit a class — all conformances are protocols.
    if kind in ("struct", "enum", "extension", "actor"):
        return "implements"
    # `class`: first entry is conventionally the base class; subsequent are protocols.
    return "inherits" if is_first else "implements"


def _swift_user_type_name(user_type_node, source: bytes) -> str | None:
    """Return the head type_identifier text from a Swift user_type node (without generics)."""
    if user_type_node is None:
        return None
    for c in user_type_node.children:
        if c.type == "type_identifier":
            text = _read_text(c, source)
            return text or None
    return None


def _swift_collect_type_refs(node, source: bytes, generic: bool, out: list[tuple[str, str]]) -> None:
    """Walk a Swift type expression; append (name, role) tuples (role 'type' or 'generic_arg')."""
    if node is None:
        return
    t = node.type
    if t == "type_annotation":
        for c in node.children:
            if c.is_named:
                _swift_collect_type_refs(c, source, generic, out)
        return
    if t == "user_type":
        for c in node.children:
            if c.type == "type_identifier":
                text = _read_text(c, source)
                if text:
                    out.append((text, "generic_arg" if generic else "type"))
                break
        for c in node.children:
            if c.type == "type_arguments":
                for arg in c.children:
                    if arg.is_named:
                        _swift_collect_type_refs(arg, source, True, out)
        return
    if t == "type_identifier":
        text = _read_text(node, source)
        if text:
            out.append((text, "generic_arg" if generic else "type"))
        return
    if t in ("optional_type", "implicitly_unwrapped_optional_type", "array_type",
             "dictionary_type", "tuple_type"):
        for c in node.children:
            if c.is_named:
                _swift_collect_type_refs(c, source, generic, out)
        return
    if node.is_named:
        for c in node.children:
            if c.is_named:
                _swift_collect_type_refs(c, source, generic, out)


def _swift_property_type_node(property_node):
    """Return the type_annotation child of a Swift property_declaration, if any."""
    for c in property_node.children:
        if c.type == "type_annotation":
            return c
    return None


# ── C / C++ type-ref helpers ─────────────────────────────────────────────────

_C_PRIMITIVE_TYPE_NODES = frozenset({
    "primitive_type", "sized_type_specifier", "auto", "placeholder_type_specifier",
})


def _c_collect_type_refs(node, source: bytes, generic: bool, out: list[tuple[str, str]]) -> None:
    """Walk a C type expression; append (name, role) tuples for user-defined types.
    Skips primitive types and qualifiers; recognises type_identifier."""
    if node is None or node.type in _C_PRIMITIVE_TYPE_NODES:
        return
    t = node.type
    if t == "type_identifier":
        text = _read_text(node, source)
        if text:
            out.append((text, "generic_arg" if generic else "type"))
        return
    if t in ("pointer_declarator", "reference_declarator", "array_declarator",
             "type_qualifier", "type_descriptor", "abstract_pointer_declarator",
             "abstract_reference_declarator", "abstract_array_declarator"):
        for c in node.children:
            if c.is_named:
                _c_collect_type_refs(c, source, generic, out)


def _cpp_collect_type_refs(node, source: bytes, generic: bool, out: list[tuple[str, str]]) -> None:
    """Walk a C++ type expression; append (name, role) tuples.
    Resolves qualified_identifier tails (std::string → string) and template_type
    base + arguments (std::vector<HttpClient> → vector + HttpClient as generic_arg)."""
    if node is None or node.type in _C_PRIMITIVE_TYPE_NODES:
        return
    t = node.type
    if t == "type_identifier":
        text = _read_text(node, source)
        if text:
            out.append((text, "generic_arg" if generic else "type"))
        return
    if t == "qualified_identifier":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            _cpp_collect_type_refs(name_node, source, generic, out)
        return
    if t == "template_type":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            text = _read_text(name_node, source)
            if text:
                out.append((text, "generic_arg" if generic else "type"))
        args_node = node.child_by_field_name("arguments")
        if args_node is not None:
            for c in args_node.children:
                if c.is_named:
                    _cpp_collect_type_refs(c, source, True, out)
        return
    if t in ("type_descriptor", "pointer_declarator", "reference_declarator",
             "array_declarator", "type_qualifier", "abstract_pointer_declarator",
             "abstract_reference_declarator", "abstract_array_declarator"):
        for c in node.children:
            if c.is_named:
                _cpp_collect_type_refs(c, source, generic, out)


# ── Scala type-ref helpers ───────────────────────────────────────────────────

def _scala_collect_type_refs(node, source: bytes, generic: bool, out: list[tuple[str, str]]) -> None:
    """Walk a Scala type expression; append (name, role) tuples.
    Handles type_identifier, generic_type (List[T]), and common type wrappers."""
    if node is None:
        return
    t = node.type
    if t == "type_identifier":
        text = _read_text(node, source)
        if text:
            out.append((text, "generic_arg" if generic else "type"))
        return
    if t == "generic_type":
        base = node.child_by_field_name("type")
        if base is None:
            for c in node.children:
                if c.type == "type_identifier":
                    base = c
                    break
        if base is not None and base.type == "type_identifier":
            text = _read_text(base, source)
            if text:
                out.append((text, "generic_arg" if generic else "type"))
        for c in node.children:
            if c.type == "type_arguments":
                for arg in c.children:
                    if arg.is_named:
                        _scala_collect_type_refs(arg, source, True, out)
        return
    if t in ("compound_type", "infix_type", "function_type", "tuple_type",
             "annotated_type", "projected_type"):
        for c in node.children:
            if c.is_named:
                _scala_collect_type_refs(c, source, generic, out)


def _python_collect_param_refs(params_node, source: bytes) -> list[tuple[str, str]]:
    """Collect type refs from each typed parameter under a `parameters` node."""
    out: list[tuple[str, str]] = []
    if params_node is None:
        return out
    for child in params_node.children:
        if child.type in ("typed_parameter", "typed_default_parameter"):
            type_node = child.child_by_field_name("type")
            _python_collect_type_refs(type_node, source, False, out)
    return out


def _find_body(node, config: LanguageConfig):
    """Find the body node using config.body_field, falling back to child types."""
    b = node.child_by_field_name(config.body_field)
    if b:
        return b
    for child in node.children:
        if child.type in config.body_fallback_child_types:
            return child
    return None


# ── Import handlers ───────────────────────────────────────────────────────────

def _import_python(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> None:
    t = node.type
    if t == "import_statement":
        for child in node.children:
            if child.type in ("dotted_name", "aliased_import"):
                raw = _read_text(child, source)
                module_name = raw.split(" as ")[0].strip().lstrip(".")
                tgt_nid = _make_id(module_name)
                edges.append({
                    "source": file_nid,
                    "target": tgt_nid,
                    "relation": "imports",
                    "context": "import",
                    "confidence": "EXTRACTED",
                    "source_file": str_path,
                    "source_location": f"L{node.start_point[0] + 1}",
                    "weight": 1.0,
                })
    elif t == "import_from_statement":
        module_node = node.child_by_field_name("module_name")
        if module_node:
            raw = _read_text(module_node, source)
            if raw.startswith("."):
                # Relative import - resolve to full path so IDs match file node IDs
                dots = len(raw) - len(raw.lstrip("."))
                module_name = raw.lstrip(".")
                base = Path(str_path).parent
                for _ in range(dots - 1):
                    base = base.parent
                rel = (module_name.replace(".", "/") + ".py") if module_name else "__init__.py"
                tgt_nid = _make_id(str(base / rel))
            else:
                tgt_nid = _make_id(raw)
            edges.append({
                "source": file_nid,
                "target": tgt_nid,
                "relation": "imports_from",
                "context": "import",
                "confidence": "EXTRACTED",
                "source_file": str_path,
                "source_location": f"L{node.start_point[0] + 1}",
                "weight": 1.0,
            })


def _resolve_js_import_target(raw: str, str_path: str) -> "tuple[str, Path | None] | None":
    """Resolve a JS/TS import path string to (target_nid, resolved_path).

    Handles relative paths, tsconfig path aliases, workspace packages, and
    bare/scoped imports.
    Returns None if `raw` is empty.
    """
    if not raw:
        return None
    resolved_path = _resolve_js_module_path(raw, Path(str_path).parent)
    if resolved_path is not None:
        return _make_id(str(resolved_path)), resolved_path
    module_name = raw.split("/")[-1]
    if not module_name:
        return None
    return _make_id(module_name), None


def _dynamic_import_js(node, source: bytes, caller_nid: str, str_path: str, edges: list,
                       seen_dyn_pairs: set) -> bool:
    """Detect dynamic import() calls in JS/TS and emit imports_from edges.

    Handles patterns like:
      await import('./foo.js')
      import('./foo.js').then(...)
      const m = await import(`./foo`)

    Returns True if the node was a dynamic import (caller should skip normal call handling).
    """
    # Dynamic import is a call_expression whose function child is the keyword "import".
    # tree-sitter-typescript parses `import('...')` as call_expression with first child
    # being an "import" token (type="import").
    func_node = node.child_by_field_name("function")
    if func_node is None:
        # Fallback: check first child directly (some TS versions)
        if node.children and _read_text(node.children[0], source) == "import":
            func_node = node.children[0]
        else:
            return False
    if _read_text(func_node, source) != "import":
        return False

    # Extract the module path from the arguments
    args = node.child_by_field_name("arguments")
    if args is None:
        return True  # It's an import() but no args — skip
    for arg in args.children:
        if arg.type == "template_string":
            # Skip dynamic template literals — path can't be statically resolved
            if any(c.type == "template_substitution" for c in arg.children):
                break
            raw = _read_text(arg, source).strip("`")
        elif arg.type == "string":
            raw = _read_text(arg, source).strip("'\" ")
        else:
            continue
        if not raw:
            break
        # Resolve path using the same logic as static imports.
        resolved = _resolve_js_import_target(raw, str_path)
        if resolved is None:
            break
        tgt_nid, _ = resolved
        pair = (caller_nid, tgt_nid)
        if pair not in seen_dyn_pairs:
            seen_dyn_pairs.add(pair)
            edges.append({
                "source": caller_nid,
                "target": tgt_nid,
                "relation": "imports_from",
                "context": "import",
                "confidence": "EXTRACTED",
                "source_file": str_path,
                "source_location": f"L{node.start_point[0] + 1}",
                "weight": 1.0,
            })
        break
    return True


# ── C/C++ function name helpers ───────────────────────────────────────────────


def _get_cpp_func_name(node, source: bytes) -> str | None:
    """Recursively unwrap declarator to find the innermost identifier (C++)."""
    if node.type == "identifier":
        return _read_text(node, source)
    if node.type in ("field_identifier", "destructor_name", "operator_name"):
        return _read_text(node, source)
    if node.type == "qualified_identifier":
        name_node = node.child_by_field_name("name")
        if name_node:
            return _read_text(name_node, source)
    decl = node.child_by_field_name("declarator")
    if decl:
        return _get_cpp_func_name(decl, source)
    for child in node.children:
        if child.type == "identifier":
            return _read_text(child, source)
    return None


# ── JS/TS extra walk for arrow functions ──────────────────────────────────────

def _find_require_call(value_node):
    """Return the call_expression node if `value_node` is a `require(...)` call
    or `require(...).x` member access. Otherwise None."""
    if value_node is None:
        return None
    if value_node.type == "call_expression":
        fn = value_node.child_by_field_name("function")
        if fn is not None and fn.type == "identifier":
            return value_node
    if value_node.type == "member_expression":
        obj = value_node.child_by_field_name("object")
        return _find_require_call(obj)
    return None


def _require_imports_js(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> bool:
    """Detect CommonJS require imports inside lexical_declaration / variable_declaration.

    Handles three patterns:
      const { foo, bar } = require('./mod')   → file → mod (imports_from), file → foo, file → bar
      const mod         = require('./mod')   → file → mod (imports_from)
      const x           = require('./mod').y → file → mod (imports_from), file → y

    Returns True if any require import was found.
    """
    if node.type not in ("lexical_declaration", "variable_declaration"):
        return False
    found = False
    for child in node.children:
        if child.type != "variable_declarator":
            continue
        value = child.child_by_field_name("value")
        call = _find_require_call(value)
        if call is None:
            continue
        fn = call.child_by_field_name("function")
        if fn is None or _read_text(fn, source) != "require":
            continue
        args = call.child_by_field_name("arguments")
        if args is None:
            continue
        raw = None
        for arg in args.children:
            if arg.type == "string":
                raw = _read_text(arg, source).strip("'\"` ")
                break
        if not raw:
            continue
        resolved = _resolve_js_import_target(raw, str_path)
        if resolved is None:
            continue
        tgt_nid, resolved_path = resolved
        line = node.start_point[0] + 1
        edges.append({
            "source": file_nid,
            "target": tgt_nid,
            "relation": "imports_from",
            "context": "import",
            "confidence": "EXTRACTED",
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": 1.0,
        })
        found = True

        # Symbol-level edges for destructured / accessor binders.
        target_stem = _file_stem(resolved_path) if resolved_path is not None else None
        name_node = child.child_by_field_name("name")
        sym_names: list[str] = []
        if name_node is not None and name_node.type == "object_pattern":
            # `const { a, b: alias } = require('./m')` — emit edges for each property key
            for prop in name_node.children:
                if prop.type == "shorthand_property_identifier_pattern":
                    sym_names.append(_read_text(prop, source))
                elif prop.type == "pair_pattern":
                    key = prop.child_by_field_name("key")
                    if key is not None:
                        sym_names.append(_read_text(key, source))
        elif value is not None and value.type == "member_expression":
            # `const x = require('./m').y` — symbol is the property accessed
            prop = value.child_by_field_name("property")
            if prop is not None:
                sym_names.append(_read_text(prop, source))
        if target_stem is not None:
            for sym in sym_names:
                edges.append({
                    "source": file_nid,
                    "target": _make_id(target_stem, sym),
                    "relation": "imports",
                    "context": "import",
                    "confidence": "EXTRACTED",
                    "source_file": str_path,
                    "source_location": f"L{line}",
                    "weight": 1.0,
                })
    return found


def _js_extra_walk(node, source: bytes, file_nid: str, stem: str, str_path: str,
                   nodes: list, edges: list, seen_ids: set, function_bodies: list,
                   parent_class_nid: str | None, add_node_fn, add_edge_fn) -> bool:
    """Handle lexical_declaration (arrow functions, CJS requires, module-level const literals) for JS/TS. Returns True if handled."""
    if node.type in ("lexical_declaration", "variable_declaration"):
        # CJS require imports — emit edges, do not block other lexical_declaration handling
        require_found = _require_imports_js(node, source, file_nid, stem, edges, str_path)

        # Scope guard (#1077): only emit nodes for module-level declarations.
        # Without this, `const x = ...` inside an arrow callback (e.g. inside
        # `describe(() => { const set = new Set(...) })`) emits a bare-named
        # node, and the same name collides across unrelated files producing
        # phantom god-nodes. Bodies of arrow functions are walked separately
        # via function_bodies, so we never need to emit nodes for locals here.
        parent = node.parent
        is_module_level = parent is not None and (
            parent.type == "program"
            or (parent.type == "export_statement"
                and parent.parent is not None
                and parent.parent.type == "program")
        )

        # Arrow function declarations and module-level const literals (lexical_declaration only)
        arrow_found = False
        const_found = False
        if node.type == "lexical_declaration" and is_module_level:
            for child in node.children:
                if child.type == "variable_declarator":
                    value = child.child_by_field_name("value")
                    if value and value.type == "arrow_function":
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            func_name = _read_text(name_node, source)
                            line = child.start_point[0] + 1
                            func_nid = _make_id(stem, func_name)
                            add_node_fn(func_nid, f"{func_name}()", line)
                            add_edge_fn(file_nid, func_nid, "contains", line)
                            body = value.child_by_field_name("body")
                            if body:
                                function_bodies.append((func_nid, body))
                            arrow_found = True
                    elif value and value.type in (
                        "object", "array", "as_expression", "call_expression", "new_expression",
                    ):
                        # Module-level const with literal/object/array/factory value
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            const_name = _read_text(name_node, source)
                            line = child.start_point[0] + 1
                            const_nid = _make_id(stem, const_name)
                            add_node_fn(const_nid, const_name, line)
                            add_edge_fn(file_nid, const_nid, "contains", line)
                            const_found = True
        if arrow_found:
            return True
        if const_found:
            return True
        if require_found:
            return True
    return False


# ── C# extra walk for namespace declarations ──────────────────────────────────

def _csharp_extra_walk(node, source: bytes, file_nid: str, stem: str, str_path: str,
                       nodes: list, edges: list, seen_ids: set, function_bodies: list,
                       parent_class_nid: str | None, add_node_fn, add_edge_fn,
                       walk_fn) -> bool:
    """Handle namespace_declaration for C#. Returns True if handled."""
    if node.type == "namespace_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            ns_name = _read_text(name_node, source)
            ns_nid = _make_id(stem, ns_name)
            line = node.start_point[0] + 1
            add_node_fn(ns_nid, ns_name, line)
            add_edge_fn(file_nid, ns_nid, "contains", line)
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                walk_fn(child, parent_class_nid)
        return True
    return False


# ── Swift extra walk for enum cases ──────────────────────────────────────────

def _swift_extra_walk(node, source: bytes, file_nid: str, stem: str, str_path: str,
                      nodes: list, edges: list, seen_ids: set, function_bodies: list,
                      parent_class_nid: str | None, add_node_fn, add_edge_fn) -> bool:
    """Handle enum_entry for Swift. Returns True if handled."""
    if node.type == "enum_entry" and parent_class_nid:
        for child in node.children:
            if child.type == "simple_identifier":
                case_name = _read_text(child, source)
                case_nid = _make_id(parent_class_nid, case_name)
                line = node.start_point[0] + 1
                add_node_fn(case_nid, case_name, line)
                add_edge_fn(parent_class_nid, case_nid, "case_of", line)
        return True
    return False


# ── Language configs ──────────────────────────────────────────────────────────

_PYTHON_CONFIG = LanguageConfig(
    ts_module="tree_sitter_python",
    class_types=frozenset({"class_definition"}),
    function_types=frozenset({"function_definition"}),
    import_types=frozenset({"import_statement", "import_from_statement"}),
    call_types=frozenset({"call"}),
    call_function_field="function",
    call_accessor_node_types=frozenset({"attribute"}),
    call_accessor_field="attribute",
    function_boundary_types=frozenset({"function_definition"}),
    import_handler=_import_python,
)


# .tsx files must use the TSX grammar (JSX-aware), not the plain TypeScript grammar.
# tree-sitter-typescript ships two languages: language_typescript (for .ts) and
# language_tsx (for .tsx). Parsing .tsx with language_typescript silently fails on
# JSX expressions, dropping any call_expression nested inside JSX (e.g. {fmtDate(x)}).


def _read_csharp_type_name(node, source: bytes) -> str | None:
    """Resolve a readable C# type name from a field/type node."""
    if node is None:
        return None
    if node.type in ("identifier", "predefined_type"):
        return _read_text(node, source)
    if node.type == "qualified_name":
        return _read_text(node, source).split(".")[-1]
    if node.type == "generic_name":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return _read_text(name_node, source)
    for child in node.children:
        if not child.is_named:
            continue
        name = _read_csharp_type_name(child, source)
        if name:
            return name
    return None


# ── Generic extractor ─────────────────────────────────────────────────────────

def _extract_generic(path: Path, config: LanguageConfig) -> dict:
    """Generic AST extractor driven by LanguageConfig."""
    try:
        mod = importlib.import_module(config.ts_module)
        from tree_sitter import Language, Parser
        lang_fn = getattr(mod, config.ts_language_fn, None)
        if lang_fn is None:
            # Fallback for PHP: try "language_php" then "language"
            lang_fn = getattr(mod, "language", None)
        if lang_fn is None:
            return {"nodes": [], "edges": [], "error": f"No language function in {config.ts_module}"}
        language = Language(lang_fn())
    except ImportError:
        return {"nodes": [], "edges": [], "error": f"{config.ts_module} not installed"}
    except TypeError as e:
        # tree-sitter version mismatch: old Language() expects (lib_path),
        # new Language() expects (language_capsule, name). Surface a hint
        # so users see the upgrade path instead of a bare TypeError.
        hint = (
            f"tree-sitter version mismatch for {config.ts_module}: {e}. "
            "Try: pip install --upgrade tree-sitter tree-sitter-languages"
        )
        return {"nodes": [], "edges": [], "error": hint}
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    try:
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    function_bodies: list[tuple[str, object]] = []
    pending_listen_edges: list[tuple[str, str, int]] = []
    # tree-sitter-swift parses both `class Foo` and `extension Foo` as
    # `class_declaration`. Same-file pairs collapse via seen_ids, but cross-file
    # extensions don't (file stem is part of the id), so they're collected here
    # for a corpus-level merge after every file has been parsed.
    swift_extensions: list[dict] = []

    csharp_interface_names: set[str] = set()
    if config.ts_module == "tree_sitter_c_sharp":
        csharp_interface_names = _csharp_pre_scan_interfaces(root, source)

    swift_protocol_names: set[str] = set()
    swift_class_names: set[str] = set()
    if config.ts_module == "tree_sitter_swift":
        swift_protocol_names, swift_class_names = _swift_pre_scan(root, source)

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({
                "id": nid,
                "label": label,
                "file_type": "code",
                "source_file": str_path,
                "source_location": f"L{line}",
            })

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0,
                 context: str | None = None) -> None:
        edge = {
            "source": src,
            "target": tgt,
            "relation": relation,
            "confidence": confidence,
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": weight,
        }
        if context:
            edge["context"] = context
        edges.append(edge)

    def ensure_named_node(name: str, line: int) -> str:
        nid = _make_id(stem, name)
        if nid in seen_ids:
            return nid
        nid = _make_id(name)
        if nid not in seen_ids:
            add_node(nid, name, line)
        return nid

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    def walk(node, parent_class_nid: str | None = None) -> None:
        t = node.type

        # Import types
        if t in config.import_types:
            if config.import_handler:
                config.import_handler(node, source, file_nid, stem, edges, str_path)
            # For export_statement: only return (skip children) if it's a re-export
            # (has a `from` source). Otherwise fall through to walk children which may
            # contain function_declaration, class_declaration, etc.
            if t == "export_statement":
                has_source = any(c.type == "string" for c in node.children)
                if not has_source:
                    for child in node.children:
                        walk(child, parent_class_nid)
            return

        # Class types
        if t in config.class_types:
            # Resolve class name
            name_node = node.child_by_field_name(config.name_field)
            if name_node is None:
                for child in node.children:
                    if child.type in config.name_fallback_child_types:
                        name_node = child
                        break
            if not name_node:
                return
            class_name = _read_text(name_node, source)
            class_nid = _make_id(stem, class_name)
            line = node.start_point[0] + 1
            add_node(class_nid, class_name, line)
            add_edge(file_nid, class_nid, "contains", line)

            if config.ts_module == "tree_sitter_swift" and any(
                c.type == "extension" for c in node.children
            ):
                swift_extensions.append({"nid": class_nid, "label": class_name})

            # Python-specific: inheritance
            if config.ts_module == "tree_sitter_python":
                args = node.child_by_field_name("superclasses")
                if args:
                    for arg in args.children:
                        if arg.type == "identifier":
                            base = _read_text(arg, source)
                            base_nid = _make_id(stem, base)
                            if base_nid not in seen_ids:
                                base_nid = _make_id(base)
                                if base_nid not in seen_ids:
                                    nodes.append({
                                        "id": base_nid,
                                        "label": base,
                                        "file_type": "code",
                                        "source_file": "",
                                        "source_location": "",
                                    })
                                    seen_ids.add(base_nid)
                            add_edge(class_nid, base_nid, "inherits", line)

            # Swift-specific: conformance / inheritance
            if config.ts_module == "tree_sitter_swift":
                swift_kind = _swift_declaration_keyword(node) if t == "class_declaration" else "protocol"
                seen_swift_base = False
                for child in node.children:
                    if child.type != "inheritance_specifier":
                        continue
                    base_name: str | None = None
                    user_type_node = None
                    for sub in child.children:
                        if sub.type == "user_type":
                            user_type_node = sub
                            base_name = _swift_user_type_name(sub, source)
                            break
                        if sub.type == "type_identifier":
                            base_name = _read_text(sub, source) or None
                            break
                    if not base_name:
                        continue
                    base_nid = _make_id(stem, base_name)
                    if base_nid not in seen_ids:
                        base_nid = _make_id(base_name)
                        if base_nid not in seen_ids:
                            nodes.append({
                                "id": base_nid,
                                "label": base_name,
                                "file_type": "code",
                                "source_file": "",
                                "source_location": "",
                            })
                            seen_ids.add(base_nid)
                    if t == "protocol_declaration":
                        relation = "inherits"
                    else:
                        relation = _swift_classify_base(
                            base_name, swift_kind, not seen_swift_base,
                            swift_protocol_names, swift_class_names,
                        )
                    seen_swift_base = True
                    add_edge(class_nid, base_nid, relation, line)
                    if user_type_node is not None:
                        for arg_child in user_type_node.children:
                            if arg_child.type != "type_arguments":
                                continue
                            for arg in arg_child.children:
                                if not arg.is_named:
                                    continue
                                refs: list[tuple[str, str]] = []
                                _swift_collect_type_refs(arg, source, True, refs)
                                for ref_name, _role in refs:
                                    target = ensure_named_node(ref_name, line)
                                    add_edge(class_nid, target, "references", line,
                                             context="generic_arg")

            # PHP-specific: extends → inherits, implements → implements, use → mixes_in
            if config.ts_module == "tree_sitter_php":
                def _php_emit_base(base_name: str, rel: str, at_line: int) -> None:
                    if not base_name:
                        return
                    base_nid = _make_id(stem, base_name)
                    if base_nid not in seen_ids:
                        base_nid = _make_id(base_name)
                        if base_nid not in seen_ids:
                            nodes.append({
                                "id": base_nid,
                                "label": base_name,
                                "file_type": "code",
                                "source_file": "",
                                "source_location": "",
                            })
                            seen_ids.add(base_nid)
                    add_edge(class_nid, base_nid, rel, at_line)

                for child in node.children:
                    if child.type == "base_clause":
                        for sub in child.children:
                            if sub.type in ("name", "qualified_name"):
                                _php_emit_base(_php_name_text(sub, source) or "",
                                                "inherits", child.start_point[0] + 1)
                    elif child.type == "class_interface_clause":
                        for sub in child.children:
                            if sub.type in ("name", "qualified_name"):
                                _php_emit_base(_php_name_text(sub, source) or "",
                                                "implements", child.start_point[0] + 1)
                body = node.child_by_field_name("body")
                if body is None:
                    for c in node.children:
                        if c.type == "declaration_list":
                            body = c
                            break
                if body is not None:
                    for member in body.children:
                        if member.type != "use_declaration":
                            continue
                        for sub in member.children:
                            if sub.type in ("name", "qualified_name"):
                                _php_emit_base(_php_name_text(sub, source) or "",
                                                "mixes_in", member.start_point[0] + 1)

            # Kotlin-specific: delegation_specifiers → inherits (constructor_invocation) / implements (user_type)
            if config.ts_module == "tree_sitter_kotlin":
                for child in node.children:
                    if child.type != "delegation_specifiers":
                        continue
                    for spec in child.children:
                        if spec.type != "delegation_specifier":
                            continue
                        relation = "implements"
                        user_type_node = None
                        for sub in spec.children:
                            if sub.type == "constructor_invocation":
                                relation = "inherits"
                                for inner in sub.children:
                                    if inner.type == "user_type":
                                        user_type_node = inner
                                        break
                                break
                            if sub.type == "user_type":
                                user_type_node = sub
                                break
                        if user_type_node is None:
                            continue
                        base = _kotlin_user_type_name(user_type_node, source)
                        if not base:
                            continue
                        base_nid = _make_id(stem, base)
                        if base_nid not in seen_ids:
                            base_nid = _make_id(base)
                            if base_nid not in seen_ids:
                                nodes.append({
                                    "id": base_nid,
                                    "label": base,
                                    "file_type": "code",
                                    "source_file": "",
                                    "source_location": "",
                                })
                                seen_ids.add(base_nid)
                        add_edge(class_nid, base_nid, relation, line)
                        for arg_child in user_type_node.children:
                            if arg_child.type != "type_arguments":
                                continue
                            for arg in arg_child.children:
                                if arg.type == "type_projection":
                                    for inner in arg.children:
                                        if not inner.is_named:
                                            continue
                                        refs: list[tuple[str, str]] = []
                                        _kotlin_collect_type_refs(inner, source, True, refs)
                                        for ref_name, _role in refs:
                                            target = ensure_named_node(ref_name, line)
                                            add_edge(class_nid, target, "references", line,
                                                     context="generic_arg")

            # C#-specific: inheritance / interface implementation via base_list
            if config.ts_module == "tree_sitter_c_sharp":
                for child in node.children:
                    if child.type != "base_list":
                        continue
                    for sub in child.children:
                        if sub.type not in ("identifier", "generic_name", "qualified_name"):
                            continue
                        if sub.type == "generic_name":
                            name_child = sub.child_by_field_name("name")
                            base = (
                                _read_text(name_child, source) if name_child
                                else _read_text(sub.children[0], source)
                            )
                        elif sub.type == "qualified_name":
                            base = _read_text(sub, source).rsplit(".", 1)[-1]
                        else:
                            base = _read_text(sub, source)
                        if not base:
                            continue
                        base_nid = _make_id(stem, base)
                        if base_nid not in seen_ids:
                            base_nid = _make_id(base)
                            if base_nid not in seen_ids:
                                nodes.append({
                                    "id": base_nid,
                                    "label": base,
                                    "file_type": "code",
                                    "source_file": "",
                                    "source_location": "",
                                })
                                seen_ids.add(base_nid)
                        relation = _csharp_classify_base(base, csharp_interface_names)
                        add_edge(class_nid, base_nid, relation, line)
                        if sub.type == "generic_name":
                            for tal in sub.children:
                                if tal.type != "type_argument_list":
                                    continue
                                for arg in tal.children:
                                    if not arg.is_named:
                                        continue
                                    refs: list[tuple[str, str]] = []
                                    _csharp_collect_type_refs(arg, source, True, refs)
                                    for ref_name, _role in refs:
                                        target = ensure_named_node(ref_name, line)
                                        add_edge(class_nid, target, "references", line,
                                                 context="generic_arg")

            # Java-specific: extends (superclass) / implements (interfaces) / interface-extends
            if config.ts_module == "tree_sitter_java":
                def _emit_java_parent(base_name: str, rel: str, at_line: int) -> None:
                    if not base_name:
                        return
                    base_nid = _make_id(stem, base_name)
                    if base_nid not in seen_ids:
                        base_nid = _make_id(base_name)
                        if base_nid not in seen_ids:
                            nodes.append({
                                "id": base_nid,
                                "label": base_name,
                                "file_type": "code",
                                "source_file": "",
                                "source_location": "",
                            })
                            seen_ids.add(base_nid)
                    add_edge(class_nid, base_nid, rel, at_line)

                sup = node.child_by_field_name("superclass")
                if sup is not None:
                    for sub in sup.children:
                        if sub.type == "type_identifier":
                            _emit_java_parent(_read_text(sub, source), "inherits", line)
                            break

                ifs = node.child_by_field_name("interfaces")
                if ifs is not None:
                    for sub in ifs.children:
                        if sub.type == "type_list":
                            for tid in sub.children:
                                if tid.type == "type_identifier":
                                    _emit_java_parent(_read_text(tid, source), "implements", line)

                if t == "interface_declaration":
                    for child in node.children:
                        if child.type == "extends_interfaces":
                            for sub in child.children:
                                if sub.type == "type_list":
                                    for tid in sub.children:
                                        if tid.type == "type_identifier":
                                            _emit_java_parent(_read_text(tid, source), "inherits", line)

            # Scala: extends_clause carries `extends Base with Trait1 with Trait2`.
            # The first base after `extends` is `inherits`; each subsequent
            # type after `with` is `mixes_in`. Also walk class_parameters for
            # constructor-as-field type references.
            if config.ts_module == "tree_sitter_scala":
                extend = node.child_by_field_name("extend")
                if extend is None:
                    for c in node.children:
                        if c.type == "extends_clause":
                            extend = c
                            break
                if extend is not None:
                    bases: list[tuple[str, int]] = []
                    for c in extend.children:
                        if c.type == "type_identifier":
                            bases.append((_read_text(c, source), c.start_point[0] + 1))
                        elif c.type == "generic_type":
                            base = c.child_by_field_name("type")
                            if base is None:
                                for sc in c.children:
                                    if sc.type == "type_identifier":
                                        base = sc
                                        break
                            if base is not None:
                                bases.append((_read_text(base, source), c.start_point[0] + 1))
                    for idx, (base_name, base_line) in enumerate(bases):
                        rel = "inherits" if idx == 0 else "mixes_in"
                        base_nid = ensure_named_node(base_name, base_line)
                        if base_nid != class_nid:
                            add_edge(class_nid, base_nid, rel, base_line)
                for c in node.children:
                    if c.type != "class_parameters":
                        continue
                    for cp in c.children:
                        if cp.type != "class_parameter":
                            continue
                        ptype = cp.child_by_field_name("type")
                        if ptype is None:
                            continue
                        cp_line = cp.start_point[0] + 1
                        refs: list[tuple[str, str]] = []
                        _scala_collect_type_refs(ptype, source, False, refs)
                        for ref_name, role in refs:
                            ctx = "generic_arg" if role == "generic_arg" else "field"
                            target_nid = ensure_named_node(ref_name, cp_line)
                            if target_nid != class_nid:
                                add_edge(class_nid, target_nid, "references",
                                         cp_line, context=ctx)

            # C++-specific: inheritance via base_class_clause (class and struct).
            # tree-sitter-cpp shape:
            #   class_specifier / struct_specifier
            #     base_class_clause
            #       access_specifier? ("public"/"protected"/"private")  -- skip
            #       "virtual"?                                          -- skip
            #       type_identifier                                     -- "Base"
            #       qualified_identifier                                -- "ns::Base"
            #       template_type                                       -- "Vec<int>"
            # Multiple bases are siblings separated by ',' tokens.
            if config.ts_module == "tree_sitter_cpp":
                for child in node.children:
                    if child.type != "base_class_clause":
                        continue
                    for sub in child.children:
                        base = ""
                        if sub.type == "type_identifier":
                            base = _read_text(sub, source)
                        elif sub.type == "qualified_identifier":
                            # Use the unqualified tail so "std::vector" matches
                            # a "vector" node id if one exists in the graph;
                            # fall back to the full qualified text otherwise.
                            tail = sub.child_by_field_name("name")
                            base = _read_text(tail, source) if tail else _read_text(sub, source)
                        elif sub.type == "template_type":
                            tname = sub.child_by_field_name("name")
                            base = _read_text(tname, source) if tname else _read_text(sub, source)
                        else:
                            continue
                        if not base:
                            continue
                        base_nid = _make_id(stem, base)
                        if base_nid not in seen_ids:
                            base_nid = _make_id(base)
                            if base_nid not in seen_ids:
                                nodes.append({
                                    "id": base_nid,
                                    "label": base,
                                    "file_type": "code",
                                    "source_file": "",
                                    "source_location": "",
                                })
                                seen_ids.add(base_nid)
                        add_edge(class_nid, base_nid, "inherits", line)

            # Find body and recurse
            body = _find_body(node, config)
            if body:
                for child in body.children:
                    walk(child, parent_class_nid=class_nid)
            return

        # Event listener property arrays: $listen = [Event::class => [Listener::class]]
        if (t == "property_declaration"
                and parent_class_nid
                and config.event_listener_properties):
            handled_event_listener = False
            for element in node.children:
                if element.type != "property_element":
                    continue
                prop_name: str | None = None
                array_node = None
                for c in element.children:
                    if c.type == "variable_name":
                        for sc in c.children:
                            if sc.type == "name":
                                prop_name = _read_text(sc, source)
                                break
                    elif c.type == "array_creation_expression":
                        array_node = c
                if (prop_name is None
                        or prop_name not in config.event_listener_properties
                        or array_node is None):
                    continue
                handled_event_listener = True
                for entry in array_node.children:
                    if entry.type != "array_element_initializer":
                        continue
                    event_cls: str | None = None
                    listener_arr = None
                    for sub in entry.children:
                        if sub.type == "class_constant_access_expression" and event_cls is None:
                            for sc in sub.children:
                                if sc.is_named and sc.type in ("name", "qualified_name"):
                                    event_cls = _read_text(sc, source)
                                    break
                        elif sub.type == "array_creation_expression":
                            listener_arr = sub
                    if not event_cls or listener_arr is None:
                        continue
                    for listener_entry in listener_arr.children:
                        if listener_entry.type != "array_element_initializer":
                            continue
                        for item in listener_entry.children:
                            if item.type != "class_constant_access_expression":
                                continue
                            for sc in item.children:
                                if sc.is_named and sc.type in ("name", "qualified_name"):
                                    listener_cls = _read_text(sc, source)
                                    line_no = item.start_point[0] + 1
                                    pending_listen_edges.append((event_cls, listener_cls, line_no))
                                    break
                            break
            if handled_event_listener:
                return

        if (config.ts_module == "tree_sitter_c_sharp"
                and t == "field_declaration"
                and parent_class_nid):
            type_node = node.child_by_field_name("type")
            if type_node is None:
                for child in node.children:
                    if child.type == "variable_declaration":
                        type_node = child.child_by_field_name("type")
                        if type_node is not None:
                            break
            type_name = _read_csharp_type_name(type_node, source)
            if type_name:
                line = node.start_point[0] + 1
                add_edge(parent_class_nid, ensure_named_node(type_name, line),
                         "references", line, context="field")
            return

        if (config.ts_module == "tree_sitter_php"
                and t == "property_declaration"
                and parent_class_nid):
            for c in node.children:
                if c.type not in ("named_type", "primitive_type", "nullable_type",
                                   "union_type", "intersection_type", "optional_type"):
                    continue
                line = node.start_point[0] + 1
                refs: list[tuple[str, str]] = []
                _php_collect_type_refs(c, source, False, refs)
                for ref_name, role in refs:
                    ctx = "generic_arg" if role == "generic_arg" else "field"
                    target_nid = ensure_named_node(ref_name, line)
                    if target_nid != parent_class_nid:
                        add_edge(parent_class_nid, target_nid, "references", line, context=ctx)
                break
            return

        if (config.ts_module == "tree_sitter_kotlin"
                and t == "property_declaration"
                and parent_class_nid):
            type_node = _kotlin_property_type_node(node)
            if type_node is not None:
                line = node.start_point[0] + 1
                refs: list[tuple[str, str]] = []
                _kotlin_collect_type_refs(type_node, source, False, refs)
                for ref_name, role in refs:
                    ctx = "generic_arg" if role == "generic_arg" else "field"
                    target_nid = ensure_named_node(ref_name, line)
                    if target_nid != parent_class_nid:
                        add_edge(parent_class_nid, target_nid, "references", line, context=ctx)
            return

        if (config.ts_module == "tree_sitter_swift"
                and t == "property_declaration"
                and parent_class_nid):
            type_anno = _swift_property_type_node(node)
            if type_anno is not None:
                line = node.start_point[0] + 1
                refs: list[tuple[str, str]] = []
                _swift_collect_type_refs(type_anno, source, False, refs)
                for ref_name, role in refs:
                    ctx = "generic_arg" if role == "generic_arg" else "field"
                    target_nid = ensure_named_node(ref_name, line)
                    if target_nid != parent_class_nid:
                        add_edge(parent_class_nid, target_nid, "references", line, context=ctx)
            return

        if (config.ts_module == "tree_sitter_scala"
                and t == "val_definition"
                and parent_class_nid):
            type_node = node.child_by_field_name("type")
            if type_node is not None:
                line = node.start_point[0] + 1
                refs: list[tuple[str, str]] = []
                _scala_collect_type_refs(type_node, source, False, refs)
                for ref_name, role in refs:
                    ctx = "generic_arg" if role == "generic_arg" else "field"
                    target_nid = ensure_named_node(ref_name, line)
                    if target_nid != parent_class_nid:
                        add_edge(parent_class_nid, target_nid, "references",
                                 line, context=ctx)
            # fall through so any call expressions in the initializer get walked

        if (config.ts_module == "tree_sitter_cpp"
                and t == "field_declaration"
                and parent_class_nid):
            # Skip method prototypes (field_declaration with a function_declarator
            # is a member-function declaration, not a data member).
            decls = list(node.children_by_field_name("declarator"))
            is_method = any(
                d.type == "function_declarator"
                or (d.type in ("pointer_declarator", "reference_declarator")
                    and any(c.type == "function_declarator" for c in d.children))
                for d in decls
            )
            if not is_method:
                type_node = node.child_by_field_name("type")
                if type_node is not None:
                    line = node.start_point[0] + 1
                    refs: list[tuple[str, str]] = []
                    _cpp_collect_type_refs(type_node, source, False, refs)
                    for ref_name, role in refs:
                        ctx = "generic_arg" if role == "generic_arg" else "field"
                        target_nid = ensure_named_node(ref_name, line)
                        if target_nid != parent_class_nid:
                            add_edge(parent_class_nid, target_nid, "references",
                                     line, context=ctx)
            # Emit a node for each data member. Use children_by_field_name so we
            # only visit declarator children, not the type node (which would give
            # us the type name, not the field name). Handles int x, y; via
            # multiple declarator fields and static const int MAX = 100; via the
            # init_declarator → field_identifier recursion in _get_cpp_func_name.
            for decl in decls:
                name = _get_cpp_func_name(decl, source)
                if name:
                    line = decl.start_point[0] + 1
                    field_nid = _make_id(parent_class_nid, name)
                    add_node(field_nid, name, line)
                    add_edge(parent_class_nid, field_nid, "defines", line, context="field")
            return

        # Function types
        if t in config.function_types:
            # Swift deinit/subscript have no name field — resolve before generic fallback
            if t == "deinit_declaration":
                func_name: str | None = "deinit"
            elif t == "subscript_declaration":
                func_name = "subscript"
            elif config.resolve_function_name_fn is not None:
                # C/C++ style: use declarator
                declarator = node.child_by_field_name("declarator")
                func_name = None
                if declarator:
                    func_name = config.resolve_function_name_fn(declarator, source)
            else:
                name_node = node.child_by_field_name(config.name_field)
                if name_node is None:
                    for child in node.children:
                        if child.type in config.name_fallback_child_types:
                            name_node = child
                            break
                func_name = _read_text(name_node, source) if name_node else None

            if not func_name:
                return

            line = node.start_point[0] + 1
            if parent_class_nid:
                func_nid = _make_id(parent_class_nid, func_name)
                add_node(func_nid, f".{func_name}()", line)
                add_edge(parent_class_nid, func_nid, "method", line)
            else:
                func_nid = _make_id(stem, func_name)
                add_node(func_nid, f"{func_name}()", line)
                add_edge(file_nid, func_nid, "contains", line)

            if config.ts_module == "tree_sitter_python":
                params_node = node.child_by_field_name("parameters")
                for ref_name, role in _python_collect_param_refs(params_node, source):
                    ctx = "generic_arg" if role == "generic_arg" else "parameter_type"
                    target_nid = ensure_named_node(ref_name, line)
                    if target_nid != func_nid:
                        edges.append(
                            _semantic_reference_edge(func_nid, target_nid, ctx, str_path, line)
                        )
                return_type_node = node.child_by_field_name("return_type")
                if return_type_node is not None:
                    return_refs: list[tuple[str, str]] = []
                    _python_collect_type_refs(return_type_node, source, False, return_refs)
                    for ref_name, role in return_refs:
                        ctx = "generic_arg" if role == "generic_arg" else "return_type"
                        target_nid = ensure_named_node(ref_name, line)
                        if target_nid != func_nid:
                            edges.append(
                                _semantic_reference_edge(func_nid, target_nid, ctx, str_path, line)
                            )

            if config.ts_module == "tree_sitter_c_sharp":
                params_node = node.child_by_field_name("parameters")
                if params_node is not None:
                    for p in params_node.children:
                        if p.type != "parameter":
                            continue
                        type_node = p.child_by_field_name("type")
                        refs: list[tuple[str, str]] = []
                        _csharp_collect_type_refs(type_node, source, False, refs)
                        for ref_name, role in refs:
                            ctx = "generic_arg" if role == "generic_arg" else "parameter_type"
                            target_nid = ensure_named_node(ref_name, line)
                            if target_nid != func_nid:
                                add_edge(func_nid, target_nid, "references", line, context=ctx)
                return_node = node.child_by_field_name("returns")
                if return_node is not None:
                    refs = []
                    _csharp_collect_type_refs(return_node, source, False, refs)
                    for ref_name, role in refs:
                        ctx = "generic_arg" if role == "generic_arg" else "return_type"
                        target_nid = ensure_named_node(ref_name, line)
                        if target_nid != func_nid:
                            add_edge(func_nid, target_nid, "references", line, context=ctx)
                for attr_name in _csharp_attribute_names(node, source):
                    target_nid = ensure_named_node(attr_name, line)
                    if target_nid != func_nid:
                        add_edge(func_nid, target_nid, "references", line, context="attribute")

            if config.ts_module == "tree_sitter_java":
                params_node = node.child_by_field_name("parameters")
                if params_node is not None:
                    for p in params_node.children:
                        if p.type != "formal_parameter":
                            continue
                        type_node = p.child_by_field_name("type")
                        refs = []
                        _java_collect_type_refs(type_node, source, False, refs)
                        for ref_name, role in refs:
                            ctx = "generic_arg" if role == "generic_arg" else "parameter_type"
                            target_nid = ensure_named_node(ref_name, line)
                            if target_nid != func_nid:
                                add_edge(func_nid, target_nid, "references", line, context=ctx)
                return_node = node.child_by_field_name("type")
                if return_node is not None:
                    refs = []
                    _java_collect_type_refs(return_node, source, False, refs)
                    for ref_name, role in refs:
                        ctx = "generic_arg" if role == "generic_arg" else "return_type"
                        target_nid = ensure_named_node(ref_name, line)
                        if target_nid != func_nid:
                            add_edge(func_nid, target_nid, "references", line, context=ctx)
                for anno_name in _java_method_annotation_names(node, source):
                    target_nid = ensure_named_node(anno_name, line)
                    if target_nid != func_nid:
                        add_edge(func_nid, target_nid, "references", line, context="attribute")

            if config.ts_module == "tree_sitter_php":
                params_container = None
                for c in node.children:
                    if c.type == "formal_parameters":
                        params_container = c
                        break
                if params_container is not None:
                    for p in params_container.children:
                        if p.type != "simple_parameter":
                            continue
                        type_node = None
                        for sub in p.children:
                            if sub.type in ("named_type", "primitive_type", "nullable_type",
                                             "union_type", "intersection_type", "optional_type"):
                                type_node = sub
                                break
                        refs: list[tuple[str, str]] = []
                        _php_collect_type_refs(type_node, source, False, refs)
                        for ref_name, role in refs:
                            ctx = "generic_arg" if role == "generic_arg" else "parameter_type"
                            target_nid = ensure_named_node(ref_name, line)
                            if target_nid != func_nid:
                                add_edge(func_nid, target_nid, "references", line, context=ctx)
                return_node = _php_method_return_type_node(node)
                if return_node is not None:
                    refs = []
                    _php_collect_type_refs(return_node, source, False, refs)
                    for ref_name, role in refs:
                        ctx = "generic_arg" if role == "generic_arg" else "return_type"
                        target_nid = ensure_named_node(ref_name, line)
                        if target_nid != func_nid:
                            add_edge(func_nid, target_nid, "references", line, context=ctx)

            if config.ts_module == "tree_sitter_kotlin":
                params_container = None
                for c in node.children:
                    if c.type == "function_value_parameters":
                        params_container = c
                        break
                if params_container is not None:
                    for p in params_container.children:
                        if p.type != "parameter":
                            continue
                        param_type_node = None
                        for sub in p.children:
                            if sub.type in ("user_type", "nullable_type", "type_reference"):
                                param_type_node = sub
                                break
                        refs: list[tuple[str, str]] = []
                        _kotlin_collect_type_refs(param_type_node, source, False, refs)
                        for ref_name, role in refs:
                            ctx = "generic_arg" if role == "generic_arg" else "parameter_type"
                            target_nid = ensure_named_node(ref_name, line)
                            if target_nid != func_nid:
                                add_edge(func_nid, target_nid, "references", line, context=ctx)
                return_type_node = _kotlin_function_return_type_node(node)
                if return_type_node is not None:
                    refs = []
                    _kotlin_collect_type_refs(return_type_node, source, False, refs)
                    for ref_name, role in refs:
                        ctx = "generic_arg" if role == "generic_arg" else "return_type"
                        target_nid = ensure_named_node(ref_name, line)
                        if target_nid != func_nid:
                            add_edge(func_nid, target_nid, "references", line, context=ctx)

            if config.ts_module == "tree_sitter_swift":
                for p in node.children:
                    if p.type != "parameter":
                        continue
                    type_node = p.child_by_field_name("type")
                    refs: list[tuple[str, str]] = []
                    _swift_collect_type_refs(type_node, source, False, refs)
                    for ref_name, role in refs:
                        ctx = "generic_arg" if role == "generic_arg" else "parameter_type"
                        target_nid = ensure_named_node(ref_name, line)
                        if target_nid != func_nid:
                            add_edge(func_nid, target_nid, "references", line, context=ctx)
                return_node = node.child_by_field_name("return_type")
                if return_node is not None:
                    refs = []
                    _swift_collect_type_refs(return_node, source, False, refs)
                    for ref_name, role in refs:
                        ctx = "generic_arg" if role == "generic_arg" else "return_type"
                        target_nid = ensure_named_node(ref_name, line)
                        if target_nid != func_nid:
                            add_edge(func_nid, target_nid, "references", line, context=ctx)

            if config.ts_module in ("tree_sitter_c", "tree_sitter_cpp"):
                collect = (_cpp_collect_type_refs if config.ts_module == "tree_sitter_cpp"
                           else _c_collect_type_refs)
                return_node = node.child_by_field_name("type")
                if return_node is not None:
                    refs: list[tuple[str, str]] = []
                    collect(return_node, source, False, refs)
                    for ref_name, role in refs:
                        ctx = "generic_arg" if role == "generic_arg" else "return_type"
                        target_nid = ensure_named_node(ref_name, line)
                        if target_nid != func_nid:
                            add_edge(func_nid, target_nid, "references", line, context=ctx)
                # function_declarator may be wrapped in pointer/reference declarators
                decl = node.child_by_field_name("declarator")
                while decl is not None and decl.type in (
                        "pointer_declarator", "reference_declarator"):
                    decl = decl.child_by_field_name("declarator")
                if decl is not None and decl.type == "function_declarator":
                    params_node = decl.child_by_field_name("parameters")
                    if params_node is not None:
                        for p in params_node.children:
                            if p.type != "parameter_declaration":
                                continue
                            ptype = p.child_by_field_name("type")
                            if ptype is None:
                                continue
                            refs = []
                            collect(ptype, source, False, refs)
                            for ref_name, role in refs:
                                ctx = "generic_arg" if role == "generic_arg" else "parameter_type"
                                target_nid = ensure_named_node(ref_name, line)
                                if target_nid != func_nid:
                                    add_edge(func_nid, target_nid, "references",
                                             line, context=ctx)

            if config.ts_module == "tree_sitter_scala":
                params_node = None
                for c in node.children:
                    if c.type == "parameters":
                        params_node = c
                        break
                if params_node is not None:
                    for p in params_node.children:
                        if p.type != "parameter":
                            continue
                        ptype = p.child_by_field_name("type")
                        if ptype is None:
                            continue
                        refs: list[tuple[str, str]] = []
                        _scala_collect_type_refs(ptype, source, False, refs)
                        for ref_name, role in refs:
                            ctx = "generic_arg" if role == "generic_arg" else "parameter_type"
                            target_nid = ensure_named_node(ref_name, line)
                            if target_nid != func_nid:
                                add_edge(func_nid, target_nid, "references",
                                         line, context=ctx)
                return_node = node.child_by_field_name("return_type")
                if return_node is not None:
                    refs = []
                    _scala_collect_type_refs(return_node, source, False, refs)
                    for ref_name, role in refs:
                        ctx = "generic_arg" if role == "generic_arg" else "return_type"
                        target_nid = ensure_named_node(ref_name, line)
                        if target_nid != func_nid:
                            add_edge(func_nid, target_nid, "references",
                                     line, context=ctx)

            body = _find_body(node, config)
            if body:
                function_bodies.append((func_nid, body))
            return

        # JS/TS arrow functions and C# namespaces — language-specific extra handling
        if config.ts_module in ("tree_sitter_javascript", "tree_sitter_typescript"):
            if _js_extra_walk(node, source, file_nid, stem, str_path,
                              nodes, edges, seen_ids, function_bodies,
                              parent_class_nid, add_node, add_edge):
                return

        if config.ts_module == "tree_sitter_c_sharp":
            if _csharp_extra_walk(node, source, file_nid, stem, str_path,
                                   nodes, edges, seen_ids, function_bodies,
                                   parent_class_nid, add_node, add_edge, walk):
                return

        if config.ts_module == "tree_sitter_swift":
            if _swift_extra_walk(node, source, file_nid, stem, str_path,
                                  nodes, edges, seen_ids, function_bodies,
                                  parent_class_nid, add_node, add_edge):
                return

        # Python's `@property` / `@staticmethod` / `@classmethod` wrap the
        # inner function_definition in a `decorated_definition` node. The
        # default recurse below clears parent_class_nid, which would cause the
        # inner method to be emitted with a class-unqualified node id (e.g.
        # `file_baz` instead of `file_bar_baz`). That diverges from the
        # class-qualified id the rationale walker uses for the same method's
        # docstring, leaving the rationale edge dangling and the docstring
        # node orphaned (#1050). Treat decorated_definition as a transparent
        # wrapper so parent_class_nid propagates to the real function node.
        if t == "decorated_definition":
            for child in node.children:
                walk(child, parent_class_nid=parent_class_nid)
            return

        # Default: recurse
        for child in node.children:
            walk(child, parent_class_nid=None)

    walk(root)

    # ── Call-graph pass ───────────────────────────────────────────────────────
    label_to_nid: dict[str, str] = {}     # case-sensitive (Ruby, C#, Java, Kotlin, etc.)
    label_to_nid_ci: dict[str, str] = {}  # case-insensitive (PHP functions/classes)
    for n in nodes:
        raw = n["label"]
        normalised = raw.strip("()").lstrip(".")
        label_to_nid[normalised] = n["id"]
        label_to_nid_ci[normalised.lower()] = n["id"]

    seen_call_pairs: set[tuple[str, str]] = set()
    seen_dyn_import_pairs: set[tuple[str, str]] = set()
    seen_static_ref_pairs: set[tuple[str, str, str]] = set()
    seen_helper_ref_pairs: set[tuple[str, str, str]] = set()
    seen_bind_pairs: set[tuple[str, str, str]] = set()
    raw_calls: list[dict] = []  # unresolved calls for cross-file resolution in extract()

    def _php_class_const_scope(n) -> str | None:
        scope = n.child_by_field_name("scope")
        if scope is None:
            for c in n.children:
                if c.is_named and c.type in ("name", "qualified_name", "identifier"):
                    scope = c
                    break
        if scope is None:
            return None
        return _read_text(scope, source)

    def walk_calls(node, caller_nid: str) -> None:
        if node.type in config.function_boundary_types:
            return

        if node.type in config.call_types:
            # JS/TS dynamic imports: await import('./foo.js')
            if config.ts_module in ("tree_sitter_javascript", "tree_sitter_typescript"):
                if _dynamic_import_js(node, source, caller_nid, str_path,
                                      edges, seen_dyn_import_pairs):
                    # Still recurse into children (import().then(...) may have calls)
                    for child in node.children:
                        walk_calls(child, caller_nid)
                    return

            callee_name: str | None = None
            is_member_call: bool = False

            # Special handling per language
            if config.ts_module == "tree_sitter_swift":
                # Swift: first child may be simple_identifier or navigation_expression
                first = node.children[0] if node.children else None
                if first:
                    if first.type == "simple_identifier":
                        callee_name = _read_text(first, source)
                    elif first.type == "navigation_expression":
                        is_member_call = True
                        for child in first.children:
                            if child.type == "navigation_suffix":
                                for sc in child.children:
                                    if sc.type == "simple_identifier":
                                        callee_name = _read_text(sc, source)
            elif config.ts_module == "tree_sitter_kotlin":
                # Kotlin: first child may be simple_identifier/identifier or
                # navigation_expression. PyPI's `tree_sitter_kotlin` produces
                # `identifier` for plain identifier nodes; older grammar
                # versions (including the JVM `io.github.bonede:tree-sitter-kotlin`
                # binding) produce `simple_identifier`. Accept both.
                first = node.children[0] if node.children else None
                if first:
                    if first.type in ("simple_identifier", "identifier"):
                        callee_name = _read_text(first, source)
                    elif first.type == "navigation_expression":
                        is_member_call = True
                        for child in reversed(first.children):
                            if child.type in ("simple_identifier", "identifier"):
                                callee_name = _read_text(child, source)
                                break
            elif config.ts_module == "tree_sitter_scala":
                # Scala: first child
                first = node.children[0] if node.children else None
                if first:
                    if first.type == "identifier":
                        callee_name = _read_text(first, source)
                    elif first.type == "field_expression":
                        is_member_call = True
                        field = first.child_by_field_name("field")
                        if field:
                            callee_name = _read_text(field, source)
                        else:
                            for child in reversed(first.children):
                                if child.type == "identifier":
                                    callee_name = _read_text(child, source)
                                    break
            elif config.ts_module == "tree_sitter_c_sharp" and node.type == "invocation_expression":
                # C#: try name field, then first named child
                name_node = node.child_by_field_name("name")
                if name_node:
                    callee_name = _read_text(name_node, source)
                else:
                    for child in node.children:
                        if child.is_named:
                            raw = _read_text(child, source)
                            if "." in raw:
                                callee_name = raw.split(".")[-1]
                                is_member_call = True
                            else:
                                callee_name = raw
                            break
            elif config.ts_module == "tree_sitter_php":
                # PHP: distinguish call expression subtypes
                if node.type == "function_call_expression":
                    func_node = node.child_by_field_name("function")
                    if func_node:
                        callee_name = _read_text(func_node, source)
                elif node.type == "scoped_call_expression":
                    # Static method call: Helper::format() → callee = "Helper"
                    scope_node = node.child_by_field_name("scope")
                    if scope_node:
                        callee_name = _read_text(scope_node, source)
                else:
                    # member_call_expression: $obj->method()
                    is_member_call = True
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        callee_name = _read_text(name_node, source)
            elif config.ts_module == "tree_sitter_cpp":
                # C++: function field, then field_expression/qualified_identifier
                func_node = node.child_by_field_name(config.call_function_field) if config.call_function_field else None
                if func_node:
                    if func_node.type == "identifier":
                        callee_name = _read_text(func_node, source)
                    elif func_node.type in ("field_expression", "qualified_identifier"):
                        is_member_call = True
                        name = func_node.child_by_field_name("field") or func_node.child_by_field_name("name")
                        if name:
                            callee_name = _read_text(name, source)
            else:
                # Generic: get callee from call_function_field
                func_node = node.child_by_field_name(config.call_function_field) if config.call_function_field else None
                if func_node:
                    if func_node.type == "identifier":
                        callee_name = _read_text(func_node, source)
                    elif func_node.type in config.call_accessor_node_types:
                        is_member_call = True
                        if config.call_accessor_field:
                            attr = func_node.child_by_field_name(config.call_accessor_field)
                            if attr:
                                callee_name = _read_text(attr, source)
                    else:
                        # Try reading the node directly (e.g. Java name field is the callee)
                        callee_name = _read_text(func_node, source)

            if callee_name and callee_name not in _LANGUAGE_BUILTIN_GLOBALS:
                tgt_nid = label_to_nid.get(callee_name)
                if tgt_nid and tgt_nid != caller_nid:
                    pair = (caller_nid, tgt_nid)
                    if pair not in seen_call_pairs:
                        seen_call_pairs.add(pair)
                        line = node.start_point[0] + 1
                        edges.append({
                            "source": caller_nid,
                            "target": tgt_nid,
                            "relation": "calls",
                            "context": "call",
                            "confidence": "EXTRACTED",
                            "source_file": str_path,
                            "source_location": f"L{line}",
                            "weight": 1.0,
                        })
                elif callee_name and not tgt_nid:
                    # Callee not in this file — save for cross-file resolution in extract()
                    raw_calls.append({
                        "caller_nid": caller_nid,
                        "callee": callee_name,
                        "is_member_call": is_member_call,
                        "source_file": str_path,
                        "source_location": f"L{node.start_point[0] + 1}",
                    })

            # Helper function calls: config('foo.bar') → uses_config edge to "foo"
            if (callee_name and callee_name in config.helper_fn_names):
                args_node = node.child_by_field_name("arguments")
                first_key: str | None = None
                if args_node:
                    for arg in args_node.children:
                        if arg.type != "argument":
                            continue
                        for inner in arg.children:
                            if inner.type == "string":
                                for sc in inner.children:
                                    if sc.type == "string_content":
                                        first_key = _read_text(sc, source)
                                        break
                                break
                        if first_key:
                            break
                if first_key:
                    segment = first_key.split(".")[0]
                    tgt_nid = (label_to_nid_ci.get(segment.lower())
                               or label_to_nid_ci.get(f"{segment}.php".lower()))
                    if tgt_nid and tgt_nid != caller_nid:
                        relation = f"uses_{callee_name}"
                        pair3 = (caller_nid, tgt_nid, relation)
                        if pair3 not in seen_helper_ref_pairs:
                            seen_helper_ref_pairs.add(pair3)
                            line = node.start_point[0] + 1
                            edges.append({
                                "source": caller_nid,
                                "target": tgt_nid,
                                "relation": relation,
                                "confidence": "EXTRACTED",
                                "confidence_score": 1.0,
                                "source_file": str_path,
                                "source_location": f"L{line}",
                                "weight": 1.0,
                            })

            # Service container bindings: $this->app->bind(Foo::class, Bar::class)
            if (node.type == "member_call_expression"
                    and callee_name
                    and callee_name in config.container_bind_methods):
                args_node = node.child_by_field_name("arguments")
                class_args: list[str] = []
                if args_node:
                    for arg in args_node.children:
                        if arg.type != "argument":
                            continue
                        for inner in arg.children:
                            if inner.type == "class_constant_access_expression":
                                cls = _php_class_const_scope(inner)
                                if cls:
                                    class_args.append(cls)
                                break
                        if len(class_args) >= 2:
                            break
                if len(class_args) == 2:
                    contract_name, impl_name = class_args
                    contract_nid = label_to_nid_ci.get(contract_name.lower())
                    impl_nid = label_to_nid_ci.get(impl_name.lower())
                    if contract_nid and impl_nid and contract_nid != impl_nid:
                        pair3 = (contract_nid, impl_nid, "bound_to")
                        if pair3 not in seen_bind_pairs:
                            seen_bind_pairs.add(pair3)
                            line = node.start_point[0] + 1
                            edges.append({
                                "source": contract_nid,
                                "target": impl_nid,
                                "relation": "bound_to",
                                "confidence": "EXTRACTED",
                                "confidence_score": 1.0,
                                "source_file": str_path,
                                "source_location": f"L{line}",
                                "weight": 1.0,
                            })

        # Static property access: Foo::$bar → uses_static_prop edge
        if node.type in config.static_prop_types:
            scope_node = node.child_by_field_name("scope")
            if scope_node is None:
                for child in node.children:
                    if child.is_named and child.type in ("name", "qualified_name", "identifier"):
                        scope_node = child
                        break
            if scope_node is not None:
                class_name = _read_text(scope_node, source)
                tgt_nid = label_to_nid_ci.get(class_name.lower())
                if tgt_nid and tgt_nid != caller_nid:
                    pair3 = (caller_nid, tgt_nid, "uses_static_prop")
                    if pair3 not in seen_static_ref_pairs:
                        seen_static_ref_pairs.add(pair3)
                        line = node.start_point[0] + 1
                        edges.append({
                            "source": caller_nid,
                            "target": tgt_nid,
                            "relation": "uses_static_prop",
                            "confidence": "EXTRACTED",
                            "confidence_score": 1.0,
                            "source_file": str_path,
                            "source_location": f"L{line}",
                            "weight": 1.0,
                        })

        # PHP class constant access: Foo::BAR → references_constant edge
        if config.ts_module == "tree_sitter_php" and node.type == "class_constant_access_expression":
            class_name = _php_class_const_scope(node)
            if class_name:
                tgt_nid = label_to_nid_ci.get(class_name.lower())
                if tgt_nid and tgt_nid != caller_nid:
                    pair3 = (caller_nid, tgt_nid, "references_constant")
                    if pair3 not in seen_static_ref_pairs:
                        seen_static_ref_pairs.add(pair3)
                        line = node.start_point[0] + 1
                        edges.append({
                            "source": caller_nid,
                            "target": tgt_nid,
                            "relation": "references_constant",
                            "confidence": "EXTRACTED",
                            "confidence_score": 1.0,
                            "source_file": str_path,
                            "source_location": f"L{line}",
                            "weight": 1.0,
                        })

        for child in node.children:
            walk_calls(child, caller_nid)

    for caller_nid, body_node in function_bodies:
        walk_calls(body_node, caller_nid)

    # ── Event listener pass ───────────────────────────────────────────────────
    seen_listen_pairs: set[tuple[str, str]] = set()
    for event_name, listener_name, line in pending_listen_edges:
        event_nid = label_to_nid_ci.get(event_name.lower())
        listener_nid = label_to_nid_ci.get(listener_name.lower())
        if not event_nid or not listener_nid or event_nid == listener_nid:
            continue
        pair2 = (event_nid, listener_nid)
        if pair2 in seen_listen_pairs:
            continue
        seen_listen_pairs.add(pair2)
        edges.append({
            "source": event_nid,
            "target": listener_nid,
            "relation": "listened_by",
            "confidence": "EXTRACTED",
            "confidence_score": 1.0,
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": 1.0,
        })

    # ── Clean edges ───────────────────────────────────────────────────────────
    valid_ids = seen_ids
    clean_edges = []
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in valid_ids and (tgt in valid_ids or edge["relation"] in ("imports", "imports_from", "re_exports")):
            clean_edges.append(edge)

    result = {"nodes": nodes, "edges": clean_edges, "raw_calls": raw_calls}
    if swift_extensions:
        result["swift_extensions"] = swift_extensions
    return result


# ── Python rationale extraction ───────────────────────────────────────────────

_RATIONALE_PREFIXES = ("# NOTE:", "# IMPORTANT:", "# HACK:", "# WHY:", "# RATIONALE:", "# TODO:", "# FIXME:")


def _is_autogenerated_python(source: bytes) -> bool:
    """Return True if this Python file is auto-generated and its module docstring is noise.

    Covers: Alembic/Flask-Migrate revisions, Django migrations, protobuf/gRPC/OpenAPI stubs.
    Module docstrings in these files are change annotations or boilerplate, not rationale.
    """
    head = source[:2048].decode("utf-8", errors="replace")
    # Generic generated-file markers (protobuf, gRPC, OpenAPI codegen, etc.)
    if any(m in head for m in ("DO NOT EDIT", "@generated", "Generated by the protocol buffer")):
        return True
    # Alembic / Flask-Migrate revision files
    if (re.search(r"^revision\s*[:=]", head, re.MULTILINE)
            and "def upgrade(" in head
            and "down_revision" in head):
        return True
    # Django migrations
    if "class Migration(migrations.Migration)" in head and "operations" in head:
        return True
    return False


def _extract_python_rationale(path: Path, result: dict) -> None:
    """Post-pass: extract docstrings and rationale comments from Python source.
    Mutates result in-place by appending to result['nodes'] and result['edges'].
    """
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        language = Language(tspython.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception:
        return

    stem = _file_stem(path)
    str_path = str(path)
    nodes = result["nodes"]
    edges = result["edges"]
    seen_ids = {n["id"] for n in nodes}
    file_nid = _make_id(str(path))

    def _get_docstring(body_node) -> tuple[str, int] | None:
        if not body_node:
            return None
        for child in body_node.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type in ("string", "concatenated_string"):
                        text = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                        text = text.strip("\"'").strip('"""').strip("'''").strip()
                        if len(text) > 20:
                            return text, child.start_point[0] + 1
            break
        return None

    def _add_rationale(text: str, line: int, parent_nid: str) -> None:
        label = text[:80].replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
        rid = _make_id(stem, "rationale", str(line))
        if rid not in seen_ids:
            seen_ids.add(rid)
            nodes.append({
                "id": rid,
                "label": label,
                "file_type": "rationale",
                "source_file": str_path,
                "source_location": f"L{line}",
            })
        edges.append({
            "source": rid,
            "target": parent_nid,
            "relation": "rationale_for",
            "confidence": "EXTRACTED",
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": 1.0,
        })

    # Module-level docstring — skip for auto-generated files (Alembic, Django
    # migrations, protobuf stubs, etc.) whose module docstrings are revision
    # annotations, not architectural rationale.
    if not _is_autogenerated_python(source):
        ds = _get_docstring(root)
        if ds:
            _add_rationale(ds[0], ds[1], file_nid)

    # Class and function docstrings
    def walk_docstrings(node, parent_nid: str) -> None:
        t = node.type
        if t == "class_definition":
            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name_node and body:
                class_name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                nid = _make_id(stem, class_name)
                ds = _get_docstring(body)
                if ds:
                    _add_rationale(ds[0], ds[1], nid)
                for child in body.children:
                    walk_docstrings(child, nid)
            return
        if t == "function_definition":
            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name_node and body:
                func_name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                nid = _make_id(parent_nid, func_name) if parent_nid != file_nid else _make_id(stem, func_name)
                ds = _get_docstring(body)
                if ds:
                    _add_rationale(ds[0], ds[1], nid)
            return
        for child in node.children:
            walk_docstrings(child, parent_nid)

    walk_docstrings(root, file_nid)

    # Rationale comments (# NOTE:, # IMPORTANT:, etc.)
    source_text = source.decode("utf-8", errors="replace")
    for lineno, line_text in enumerate(source_text.splitlines(), start=1):
        stripped = line_text.strip()
        if any(stripped.startswith(p) for p in _RATIONALE_PREFIXES):
            _add_rationale(stripped, lineno, file_nid)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_python(path: Path) -> dict:
    """Extract classes, functions, and imports from a .py file via tree-sitter AST."""
    result = _extract_generic(path, _PYTHON_CONFIG)
    if "error" not in result:
        _extract_python_rationale(path, result)
    return result


def extract_sql(path: Path, content: str | bytes | None = None) -> dict:
    """Extract tables, views, functions, and relationships from .sql files via tree-sitter."""
    try:
        import tree_sitter_sql as tssql
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_sql not installed. Run: pip install tree-sitter-sql"}

    try:
        language = Language(tssql.language())
        parser = Parser(language)
        source = (
            content.encode("utf-8") if isinstance(content, str)
            else content if content is not None
            else path.read_bytes()
        )
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}


    stem = _file_stem(path)
    str_path = str(path)
    file_nid = _make_id(str_path)
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                           "source_file": str_path, "source_location": None}]
    edges: list[dict] = []
    seen_ids: set[str] = {file_nid}
    table_nids: dict[str, str] = {}  # name → nid for reference resolution

    def _read(n) -> str:
        return source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    def _obj_name(n) -> str | None:
        for c in n.children:
            if c.type == "object_reference":
                return _read(c)
        return None

    def _add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                           "source_file": str_path, "source_location": f"L{line}"})
            edges.append({"source": file_nid, "target": nid, "relation": "contains",
                           "confidence": "EXTRACTED", "source_file": str_path,
                           "source_location": f"L{line}", "weight": 1.0})

    def _add_edge(src: str, tgt: str, relation: str, line: int) -> None:
        edges.append({"source": src, "target": tgt, "relation": relation,
                       "confidence": "EXTRACTED", "source_file": str_path,
                       "source_location": f"L{line}", "weight": 1.0})

    def walk(node) -> None:
        t = node.type
        line = node.start_point[0] + 1

        if t == "create_table":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, name, line)
                table_nids[name.lower()] = nid
                # Foreign key REFERENCES
                for col in node.children:
                    if col.type == "column_definitions":
                        has_error = any(cd.type == "ERROR" for cd in col.children)
                        seen_refs: set[str] = set()
                        for cd in col.children:
                            if cd.type == "column_definition":
                                # Inline column-level REFERENCES
                                ref_name: str | None = None
                                found_ref = False
                                for cc in cd.children:
                                    if cc.type == "keyword_references":
                                        found_ref = True
                                    elif found_ref and cc.type == "object_reference":
                                        ref_name = _read(cc)
                                        break
                                if ref_name:
                                    ref_nid = table_nids.get(ref_name.lower()) or _make_id(stem, ref_name)
                                    _add_edge(nid, ref_nid, "references", line)
                                    seen_refs.add(ref_name.lower())
                            elif cd.type == "constraints":
                                # Table-level FOREIGN KEY ... REFERENCES ... constraints
                                for constraint in cd.children:
                                    if constraint.type != "constraint":
                                        continue
                                    ref_name = None
                                    found_ref = False
                                    for cc in constraint.children:
                                        if cc.type == "keyword_references":
                                            found_ref = True
                                        elif found_ref and cc.type == "object_reference":
                                            ref_name = _read(cc)
                                            break
                                    if ref_name:
                                        ref_nid = table_nids.get(ref_name.lower()) or _make_id(stem, ref_name)
                                        _add_edge(nid, ref_nid, "references", line)
                                        seen_refs.add(ref_name.lower())
                        if has_error:
                            # Dialect-specific syntax (e.g. Firebird COMPUTED BY) causes ERROR
                            # nodes that make the parser drop the trailing constraints block.
                            # Regex-scan the raw column_definitions text as fallback.
                            col_text = _read(col)
                            for rm in re.finditer(r"\bREFERENCES\s+([\w$]+)", col_text, re.IGNORECASE):
                                ref_name = rm.group(1)
                                if ref_name.lower() not in seen_refs:
                                    ref_nid = table_nids.get(ref_name.lower()) or _make_id(stem, ref_name)
                                    _add_edge(nid, ref_nid, "references", line)
                                    seen_refs.add(ref_name.lower())

        elif t == "create_view":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, name, line)
                table_nids[name.lower()] = nid
                # FROM/JOIN table references inside view body
                _walk_from_refs(node, nid, line)

        elif t == "create_function":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, f"{name}()", line)
                _walk_from_refs(node, nid, line)

        elif t == "create_procedure":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, f"{name}()", line)
                _walk_from_refs(node, nid, line)

        elif t == "alter_table":
            name = _obj_name(node)
            if name:
                src_nid = table_nids.get(name.lower())
                if not src_nid:
                    src_nid = _make_id(stem, name)
                    _add_node(src_nid, name, line)
                    table_nids[name.lower()] = src_nid
                for child in node.children:
                    if child.type == "add_constraint":
                        for cc in child.children:
                            if cc.type != "constraint":
                                continue
                            found_ref = False
                            ref_name: str | None = None
                            for ccc in cc.children:
                                if ccc.type == "keyword_references":
                                    found_ref = True
                                elif found_ref and ccc.type == "object_reference":
                                    ref_name = _read(ccc)
                                    break
                            if ref_name:
                                ref_nid = table_nids.get(ref_name.lower())
                                if not ref_nid:
                                    ref_nid = _make_id(stem, ref_name)
                                _add_edge(src_nid, ref_nid, "references", line)

        elif t == "create_trigger":
            trig_name: str | None = None
            tbl_name: str | None = None
            after_trigger = False
            after_for = False
            for c in node.children:
                if c.type == "keyword_trigger":
                    after_trigger = True
                elif after_trigger and not trig_name and c.type == "object_reference":
                    trig_name = _read(c)
                elif c.type == "keyword_for":
                    after_for = True
                elif after_for and not tbl_name and c.type == "object_reference":
                    tbl_name = _read(c)
            if trig_name:
                trig_nid = _make_id(stem, trig_name)
                _add_node(trig_nid, trig_name, line)
                if tbl_name:
                    tbl_nid = table_nids.get(tbl_name.lower()) or _make_id(stem, tbl_name)
                    _add_edge(trig_nid, tbl_nid, "triggers", line)

        elif t == "fb_proc_or_trigger":
            text = _read(node)
            m = re.match(
                r"CREATE\s+(?:OR\s+(?:REPLACE|ALTER)\s+)?"
                r"(PROCEDURE|TRIGGER|FUNCTION)\s+([\w$]+)",
                text, re.IGNORECASE,
            )
            if m:
                obj_type = m.group(1).upper()
                obj_name = m.group(2)
                obj_nid = _make_id(stem, obj_name)
                label = obj_name if obj_type == "TRIGGER" else f"{obj_name}()"
                _add_node(obj_nid, label, line)
                if obj_type == "TRIGGER":
                    fm = re.search(r"\bFOR\s+([\w$]+)", text, re.IGNORECASE)
                    if fm:
                        tbl = fm.group(1)
                        tbl_nid = table_nids.get(tbl.lower()) or _make_id(stem, tbl)
                        _add_edge(obj_nid, tbl_nid, "triggers", line)
                _NON_TABLES = {
                    "select", "where", "set", "dual", "null", "true", "false",
                    "first", "skip", "rows", "next", "only", "lateral",
                }
                seen_tbls: set[str] = set()
                for rm in re.finditer(r"\b(?:FROM|JOIN|INTO)\s+([\w$]+)", text, re.IGNORECASE):
                    tbl = rm.group(1)
                    if tbl.lower() not in _NON_TABLES and tbl.lower() not in seen_tbls:
                        seen_tbls.add(tbl.lower())
                        tbl_nid = table_nids.get(tbl.lower()) or _make_id(stem, tbl)
                        _add_edge(obj_nid, tbl_nid, "reads_from", line)
                for rm in re.finditer(r"\bUPDATE\s+([\w$]+)", text, re.IGNORECASE):
                    tbl = rm.group(1)
                    if tbl.lower() not in _NON_TABLES and tbl.lower() not in seen_tbls:
                        seen_tbls.add(tbl.lower())
                        tbl_nid = table_nids.get(tbl.lower()) or _make_id(stem, tbl)
                        _add_edge(obj_nid, tbl_nid, "reads_from", line)

        for child in node.children:
            walk(child)

    def _walk_from_refs(node, caller_nid: str, line: int) -> None:
        """Recursively find FROM/JOIN table references inside a node."""
        if node.type in ("from", "join"):
            for c in node.children:
                if c.type == "relation":
                    for cc in c.children:
                        if cc.type == "object_reference":
                            tbl = _read(cc)
                            tbl_nid = _make_id(stem, tbl)
                            _add_edge(caller_nid, tbl_nid, "reads_from",
                                      c.start_point[0] + 1)
        for child in node.children:
            _walk_from_refs(child, caller_nid, line)

    for stmt in root.children:
        if stmt.type == "statement":
            for child in stmt.children:
                walk(child)
        elif stmt.type in ("fb_proc_or_trigger", "set_term", "declare_external_function"):
            walk(stmt)

    # Global regex fallback: catch any REFERENCES missed due to ERROR nodes in the parse tree
    # (e.g. Firebird COMPUTED BY columns push constraints out of the tree entirely).
    # Snapshot after tree walk so we don't re-emit edges already captured above.
    emitted = {(e["source"], e["target"]) for e in edges if e["relation"] == "references"}
    src_text = source.decode("utf-8", errors="replace")
    for m in re.finditer(r"CREATE\s+TABLE\s+([\w$]+)\s*\(", src_text, re.IGNORECASE):
        tbl_name = m.group(1)
        tbl_nid = table_nids.get(tbl_name.lower())
        if tbl_nid is None:
            continue
        tbl_line = src_text[: m.start()].count("\n") + 1
        tail = src_text[m.start():]
        end = re.search(r"(?:^|\n)(?:CREATE|SET\s+TERM|ALTER)\s", tail[1:], re.IGNORECASE)
        block = tail[: end.start() + 1] if end else tail
        for rm in re.finditer(r"\bREFERENCES\s+([\w$]+)", block, re.IGNORECASE):
            ref_name = rm.group(1)
            ref_nid = table_nids.get(ref_name.lower()) or _make_id(stem, ref_name)
            if (tbl_nid, ref_nid) not in emitted:
                _add_edge(tbl_nid, ref_nid, "references", tbl_line)
                emitted.add((tbl_nid, ref_nid))

    return {"nodes": nodes, "edges": edges}


# ── Julia extractor (custom walk) ────────────────────────────────────────────


# ── Go extractor (custom walk) ────────────────────────────────────────────────

def extract_go(path: Path) -> dict:
    """Extract functions, methods, type declarations, and imports from a .go file."""
    try:
        import tree_sitter_go as tsgo
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree-sitter-go not installed"}

    try:
        language = Language(tsgo.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    # Use directory name as package scope so methods on the same type across
    # multiple files in a package share one canonical type node.
    pkg_scope = path.parent.name or stem
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    function_bodies: list[tuple[str, object]] = []
    go_imported_pkgs: set[str] = set()  # local names of imported packages

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({
                "id": nid,
                "label": label,
                "file_type": "code",
                "source_file": str_path,
                "source_location": f"L{line}",
            })

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0,
                 context: str | None = None) -> None:
        edge = {
            "source": src,
            "target": tgt,
            "relation": relation,
            "confidence": confidence,
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": weight,
        }
        if context:
            edge["context"] = context
        edges.append(edge)

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    def ensure_named_node(name: str, line: int) -> str:
        nid = _make_id(pkg_scope, name)
        if nid in seen_ids:
            return nid
        nid = _make_id(name)
        if nid not in seen_ids:
            add_node(nid, name, line)
        return nid

    def emit_go_method_refs(func_node, func_nid: str, line: int) -> None:
        params = func_node.child_by_field_name("parameters")
        if params is not None:
            for p in params.children:
                if p.type != "parameter_declaration":
                    continue
                type_node = p.child_by_field_name("type")
                refs: list[tuple[str, str]] = []
                _go_collect_type_refs(type_node, source, False, refs)
                for ref_name, role in refs:
                    ctx = "generic_arg" if role == "generic_arg" else "parameter_type"
                    tgt = ensure_named_node(ref_name, line)
                    if tgt != func_nid:
                        add_edge(func_nid, tgt, "references", line, context=ctx)
        result = func_node.child_by_field_name("result")
        if result is not None:
            if result.type == "parameter_list":
                for p in result.children:
                    if p.type != "parameter_declaration":
                        continue
                    type_node = p.child_by_field_name("type")
                    if type_node is None:
                        for c in p.children:
                            if c.is_named:
                                type_node = c
                                break
                    refs = []
                    _go_collect_type_refs(type_node, source, False, refs)
                    for ref_name, role in refs:
                        ctx = "generic_arg" if role == "generic_arg" else "return_type"
                        tgt = ensure_named_node(ref_name, line)
                        if tgt != func_nid:
                            add_edge(func_nid, tgt, "references", line, context=ctx)
            else:
                refs = []
                _go_collect_type_refs(result, source, False, refs)
                for ref_name, role in refs:
                    ctx = "generic_arg" if role == "generic_arg" else "return_type"
                    tgt = ensure_named_node(ref_name, line)
                    if tgt != func_nid:
                        add_edge(func_nid, tgt, "references", line, context=ctx)

    def walk(node) -> None:
        t = node.type

        if t == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                func_nid = _make_id(stem, func_name)
                add_node(func_nid, f"{func_name}()", line)
                add_edge(file_nid, func_nid, "contains", line)
                emit_go_method_refs(node, func_nid, line)
                body = node.child_by_field_name("body")
                if body:
                    function_bodies.append((func_nid, body))
            return

        if t == "method_declaration":
            receiver = node.child_by_field_name("receiver")
            receiver_type: str | None = None
            if receiver:
                for param in receiver.children:
                    if param.type == "parameter_declaration":
                        type_node = param.child_by_field_name("type")
                        if type_node:
                            receiver_type = _read_text(type_node, source).lstrip("*").strip()
                        break
            name_node = node.child_by_field_name("name")
            if not name_node:
                return
            method_name = _read_text(name_node, source)
            line = node.start_point[0] + 1

            if receiver_type:
                parent_nid = _make_id(pkg_scope, receiver_type)
                add_node(parent_nid, receiver_type, line)
                method_nid = _make_id(parent_nid, method_name)
                add_node(method_nid, f".{method_name}()", line)
                add_edge(parent_nid, method_nid, "method", line)
            else:
                method_nid = _make_id(stem, method_name)
                add_node(method_nid, f"{method_name}()", line)
                add_edge(file_nid, method_nid, "contains", line)

            emit_go_method_refs(node, method_nid, line)
            body = node.child_by_field_name("body")
            if body:
                function_bodies.append((method_nid, body))
            return

        if t == "type_declaration":
            for child in node.children:
                if child.type != "type_spec":
                    continue
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                type_name = _read_text(name_node, source)
                line = child.start_point[0] + 1
                type_nid = _make_id(pkg_scope, type_name)
                add_node(type_nid, type_name, line)
                add_edge(file_nid, type_nid, "contains", line)
                # Type body: struct fields (with embeds) or interface embedding.
                type_body = None
                for tc in child.children:
                    if tc.type in ("struct_type", "interface_type"):
                        type_body = tc
                        break
                if type_body is None:
                    continue
                if type_body.type == "struct_type":
                    for fdl in type_body.children:
                        if fdl.type != "field_declaration_list":
                            continue
                        for field in fdl.children:
                            if field.type != "field_declaration":
                                continue
                            has_name = any(
                                fc.type == "field_identifier" for fc in field.children
                            )
                            type_node = field.child_by_field_name("type")
                            if type_node is None:
                                for fc in field.children:
                                    if fc.is_named and fc.type != "field_identifier":
                                        type_node = fc
                                        break
                            refs: list[tuple[str, str]] = []
                            _go_collect_type_refs(type_node, source, False, refs)
                            for ref_name, role in refs:
                                tgt = ensure_named_node(ref_name, field.start_point[0] + 1)
                                if tgt == type_nid:
                                    continue
                                if not has_name and role == "type":
                                    add_edge(type_nid, tgt, "embeds",
                                             field.start_point[0] + 1)
                                else:
                                    ctx = "generic_arg" if role == "generic_arg" else "field"
                                    add_edge(type_nid, tgt, "references",
                                             field.start_point[0] + 1, context=ctx)
                elif type_body.type == "interface_type":
                    for elem in type_body.children:
                        if elem.type != "type_elem":
                            continue
                        refs = []
                        for sub in elem.children:
                            if sub.is_named:
                                _go_collect_type_refs(sub, source, False, refs)
                        for ref_name, role in refs:
                            tgt = ensure_named_node(ref_name, elem.start_point[0] + 1)
                            if tgt == type_nid:
                                continue
                            if role == "type":
                                add_edge(type_nid, tgt, "embeds",
                                         elem.start_point[0] + 1)
                            else:
                                add_edge(type_nid, tgt, "references",
                                         elem.start_point[0] + 1, context="generic_arg")
            return

        if t == "import_declaration":
            for child in node.children:
                if child.type == "import_spec_list":
                    for spec in child.children:
                        if spec.type == "import_spec":
                            path_node = spec.child_by_field_name("path")
                            if path_node:
                                raw = _read_text(path_node, source).strip('"')
                                # Prefix with go_pkg_ so stdlib names (e.g. "context")
                                # don't collide with local files of the same basename.
                                tgt_nid = _make_id("go", "pkg", raw)
                                add_edge(file_nid, tgt_nid, "imports_from", spec.start_point[0] + 1, context="import")
                                # Track local name (alias or last path segment)
                                alias = spec.child_by_field_name("name")
                                local_name = _read_text(alias, source) if alias else raw.split("/")[-1]
                                if local_name and local_name != "_" and local_name != ".":
                                    go_imported_pkgs.add(local_name)
                elif child.type == "import_spec":
                    path_node = child.child_by_field_name("path")
                    if path_node:
                        raw = _read_text(path_node, source).strip('"')
                        tgt_nid = _make_id("go", "pkg", raw)
                        add_edge(file_nid, tgt_nid, "imports_from", child.start_point[0] + 1, context="import")
                        alias = child.child_by_field_name("name")
                        local_name = _read_text(alias, source) if alias else raw.split("/")[-1]
                        if local_name and local_name != "_" and local_name != ".":
                            go_imported_pkgs.add(local_name)
            return

        for child in node.children:
            walk(child)

    walk(root)

    label_to_nid: dict[str, str] = {}
    for n in nodes:
        raw = n["label"]
        normalised = raw.strip("()").lstrip(".")
        label_to_nid[normalised] = n["id"]

    seen_call_pairs: set[tuple[str, str]] = set()
    raw_calls: list[dict] = []

    def walk_calls(node, caller_nid: str) -> None:
        if node.type in ("function_declaration", "method_declaration"):
            return
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            callee_name: str | None = None
            is_member_call: bool = False
            if func_node:
                if func_node.type == "identifier":
                    callee_name = _read_text(func_node, source)
                elif func_node.type == "selector_expression":
                    field = func_node.child_by_field_name("field")
                    operand = func_node.child_by_field_name("operand")
                    receiver_name = _read_text(operand, source) if operand else ""
                    # Package-qualified call (e.g. fmt.Println) → allow cross-file resolution.
                    # Receiver method call (e.g. s.logger.Log) → skip, no import evidence.
                    is_member_call = receiver_name not in go_imported_pkgs
                    if field:
                        callee_name = _read_text(field, source)
            if callee_name and callee_name not in _LANGUAGE_BUILTIN_GLOBALS:
                tgt_nid = label_to_nid.get(callee_name)
                if tgt_nid and tgt_nid != caller_nid:
                    pair = (caller_nid, tgt_nid)
                    if pair not in seen_call_pairs:
                        seen_call_pairs.add(pair)
                        line = node.start_point[0] + 1
                        edges.append({
                            "source": caller_nid,
                            "target": tgt_nid,
                            "relation": "calls",
                            "context": "call",
                            "confidence": "EXTRACTED",
                            "source_file": str_path,
                            "source_location": f"L{line}",
                            "weight": 1.0,
                        })
                elif callee_name:
                    raw_calls.append({
                        "caller_nid": caller_nid,
                        "callee": callee_name,
                        "is_member_call": is_member_call,
                        "source_file": str_path,
                        "source_location": f"L{node.start_point[0] + 1}",
                    })
        for child in node.children:
            walk_calls(child, caller_nid)

    for caller_nid, body_node in function_bodies:
        walk_calls(body_node, caller_nid)

    valid_ids = seen_ids
    clean_edges = []
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in valid_ids and (tgt in valid_ids or edge["relation"] in ("imports", "imports_from")):
            clean_edges.append(edge)

    return {"nodes": nodes, "edges": clean_edges, "raw_calls": raw_calls}


# ── Rust extractor (custom walk) ──────────────────────────────────────────────

# Common Rust trait/stdlib method names that appear in virtually every codebase.
# Resolving these cross-file produces spurious INFERRED edges across crate
# boundaries (issue #908) — skip them from the unresolved-call queue entirely.


# ── Zig ───────────────────────────────────────────────────────────────────────


# ── PowerShell ────────────────────────────────────────────────────────────────


# ── Cross-file import resolution ──────────────────────────────────────────────

def _source_key(source_file: str, root: Path) -> str:
    if not source_file:
        return ""
    source_path = Path(source_file)
    try:
        return str(source_path.resolve().relative_to(root))
    except Exception:
        return str(source_path)


def _disambiguate_colliding_node_ids(
    nodes: list[dict],
    edges: list[dict],
    raw_calls: list[dict],
    root: Path,
) -> None:
    """Rewrite only colliding node IDs, using source path as the disambiguator."""
    by_id: dict[str, list[dict]] = {}
    for node in nodes:
        nid = node.get("id")
        if isinstance(nid, str) and nid:
            by_id.setdefault(nid, []).append(node)

    remap: dict[tuple[str, str], str] = {}
    ambiguous_ids: set[str] = set()
    for old_id, group in by_id.items():
        source_keys = {_source_key(str(node.get("source_file", "")), root) for node in group}
        if len(group) < 2 or len(source_keys) < 2:
            continue
        ambiguous_ids.add(old_id)
        for node in group:
            source_key = _source_key(str(node.get("source_file", "")), root)
            if not source_key:
                continue
            new_id = _make_id(source_key, old_id)
            remap[(old_id, source_key)] = new_id
            if new_id != old_id:
                node["id"] = new_id

    if not remap:
        return

    unambiguous_remaps: dict[str, str] = {}
    for old_id, group in by_id.items():
        if old_id in ambiguous_ids:
            continue
        candidates = {
            node["id"] for node in group
            if isinstance(node.get("id"), str) and node["id"] != old_id
        }
        if len(candidates) == 1:
            unambiguous_remaps[old_id] = next(iter(candidates))

    for edge in edges:
        edge_source_key = _source_key(str(edge.get("source_file", "")), root)
        source_key = (edge.get("source", ""), edge_source_key)
        target_key = (edge.get("target", ""), edge_source_key)
        if source_key in remap:
            edge["source"] = remap[source_key]
        elif edge.get("source") in unambiguous_remaps:
            edge["source"] = unambiguous_remaps[str(edge["source"])]
        if target_key in remap:
            edge["target"] = remap[target_key]
        elif edge.get("target") in unambiguous_remaps:
            edge["target"] = unambiguous_remaps[str(edge["target"])]

    for raw_call in raw_calls:
        call_source_key = _source_key(str(raw_call.get("source_file", "")), root)
        caller_key = (raw_call.get("caller_nid", ""), call_source_key)
        if caller_key in remap:
            raw_call["caller_nid"] = remap[caller_key]
        elif raw_call.get("caller_nid") in unambiguous_remaps:
            raw_call["caller_nid"] = unambiguous_remaps[str(raw_call["caller_nid"])]


def _node_label_key(node: dict) -> str:
    label = str(node.get("label", "")).strip()
    return re.sub(r"[^a-zA-Z0-9]+", "", label).lower()


def _is_type_like_definition(node: dict) -> bool:
    label = str(node.get("label", "")).strip()
    if not label:
        return False
    if label.endswith(")") or label.startswith("."):
        return False
    if "." in label:
        return False
    return node.get("file_type") == "code"


def _rewire_unique_stub_nodes(nodes: list[dict], edges: list[dict]) -> None:
    """Map unresolved no-source stubs to a unique real definition with the same label."""
    real_by_label: dict[str, list[dict]] = {}
    stubs: list[dict] = []

    for node in nodes:
        key = _node_label_key(node)
        if not key:
            continue
        if node.get("source_file"):
            if _is_type_like_definition(node):
                real_by_label.setdefault(key, []).append(node)
            continue
        stubs.append(node)

    remap: dict[str, str] = {}
    drop_ids: set[str] = set()
    for stub in stubs:
        stub_id = str(stub.get("id", ""))
        if not stub_id:
            continue
        candidates = real_by_label.get(_node_label_key(stub), [])
        if len(candidates) != 1:
            continue
        target_id = candidates[0].get("id")
        if isinstance(target_id, str) and target_id and target_id != stub_id:
            remap[stub_id] = target_id
            drop_ids.add(stub_id)

    if not remap:
        return

    for edge in edges:
        if edge.get("source") in remap:
            edge["source"] = remap[str(edge["source"])]
        if edge.get("target") in remap:
            edge["target"] = remap[str(edge["target"])]

    nodes[:] = [node for node in nodes if node.get("id") not in drop_ids]


def _js_source_path(source_file: str, root: Path) -> Path | None:
    if not source_file:
        return None
    path = Path(source_file)
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve()
    except Exception:
        return path


@dataclass(frozen=True)
class _SymbolDeclarationFact:
    file_path: Path
    name: str
    line: int


@dataclass(frozen=True)
class _SymbolImportFact:
    file_path: Path
    local_name: str
    target_path: Path
    imported_name: str
    line: int


@dataclass(frozen=True)
class _SymbolAliasFact:
    file_path: Path
    alias: str
    target_name: str
    line: int


@dataclass(frozen=True)
class _SymbolExportFact:
    file_path: Path
    exported_name: str
    line: int
    local_name: str | None = None
    target_path: Path | None = None
    target_name: str | None = None


@dataclass(frozen=True)
class _StarExportFact:
    file_path: Path
    target_path: Path
    line: int


@dataclass(frozen=True)
class _SymbolUseFact:
    file_path: Path
    source_id: str
    local_name: str
    relation: str
    context: str
    line: int


@dataclass
class _SymbolResolutionFacts:
    declarations: list[_SymbolDeclarationFact] = field(default_factory=list)
    imports: list[_SymbolImportFact] = field(default_factory=list)
    aliases: list[_SymbolAliasFact] = field(default_factory=list)
    exports: list[_SymbolExportFact] = field(default_factory=list)
    star_exports: list[_StarExportFact] = field(default_factory=list)
    uses: list[_SymbolUseFact] = field(default_factory=list)
    # File-to-file submodule imports from `from pkg import submod` (#1146).
    # Each entry is (importing_file, submodule_file, line).
    module_imports: list[tuple[Path, Path, int]] = field(default_factory=list)


def _apply_symbol_resolution_facts(
    paths: list[Path],
    nodes: list[dict],
    edges: list[dict],
    root: Path,
    facts: _SymbolResolutionFacts,
) -> None:
    """Apply language-provided import/export/use facts to graph edges."""
    if not (
        facts.declarations
        or facts.imports
        or facts.aliases
        or facts.exports
        or facts.star_exports
        or facts.uses
        or facts.module_imports
    ):
        return

    path_by_resolved = {path.resolve(): path for path in paths}
    source_file_id = {path.resolve(): _make_id(str(path)) for path in paths}
    symbol_nodes: dict[tuple[Path, str], str] = {}
    for node in nodes:
        source_path = _js_source_path(str(node.get("source_file", "")), root)
        if source_path is None:
            continue
        label = str(node.get("label", "")).strip().strip("()").lstrip(".")
        if label and node.get("id"):
            symbol_nodes[(source_path, label)] = str(node["id"])

    def ensure_symbol_node(path: Path, name: str, line: int) -> str:
        resolved_path = path.resolve()
        existing = symbol_nodes.get((resolved_path, name))
        if existing is not None:
            return existing
        node_id = _make_id(_file_stem(path), name)
        symbol_nodes[(resolved_path, name)] = node_id
        nodes.append({
            "id": node_id,
            "label": name,
            "file_type": "code",
            "source_file": str(path),
            "source_location": f"L{line}",
        })
        return node_id

    existing_edges = {
        (
            str(edge.get("source")),
            str(edge.get("target")),
            str(edge.get("relation")),
            str(edge.get("context") or ""),
        )
        for edge in edges
    }

    def add_edge(source: str, target: str, relation: str, context: str, line: int, source_path: Path) -> None:
        key = (source, target, relation, context or "")
        if key in existing_edges:
            return
        existing_edges.add(key)
        edges.append({
            "source": source,
            "target": target,
            "relation": relation,
            "context": context,
            "confidence": "EXTRACTED",
            "source_file": str(source_path),
            "source_location": f"L{line}",
            "weight": 1.0,
        })

    for declaration in facts.declarations:
        ensure_symbol_node(declaration.file_path, declaration.name, declaration.line)

    local_aliases_by_file: dict[Path, dict[str, tuple[Path, str]]] = {}
    for import_fact in facts.imports:
        file_path = import_fact.file_path.resolve()
        local_aliases_by_file.setdefault(file_path, {})[import_fact.local_name] = (
            import_fact.target_path.resolve(),
            import_fact.imported_name,
        )

    pending_aliases_by_file: dict[Path, list[_SymbolAliasFact]] = {}
    for alias_fact in facts.aliases:
        pending_aliases_by_file.setdefault(alias_fact.file_path.resolve(), []).append(alias_fact)

    for file_path, aliases in pending_aliases_by_file.items():
        local_aliases = local_aliases_by_file.setdefault(file_path, {})
        changed = True
        while changed:
            changed = False
            for alias_fact in aliases:
                if alias_fact.alias in local_aliases:
                    continue
                origin = local_aliases.get(alias_fact.target_name)
                if origin is not None:
                    local_aliases[alias_fact.alias] = origin
                    changed = True

    named_exports_by_file: dict[Path, dict[str, tuple[Path, str]]] = {}
    star_exports_by_file: dict[Path, list[Path]] = {}

    for star_fact in facts.star_exports:
        source_path = star_fact.file_path.resolve()
        target_path = star_fact.target_path.resolve()
        star_exports_by_file.setdefault(source_path, []).append(target_path)
        source_id = source_file_id.get(source_path)
        if source_id is not None:
            add_edge(
                source_id,
                _make_id(str(path_by_resolved.get(target_path, target_path))),
                "re_exports",
                "export",
                star_fact.line,
                star_fact.file_path,
            )

    for export_fact in facts.exports:
        file_path = export_fact.file_path.resolve()
        origin: tuple[Path, str] | None = None
        if export_fact.target_path is not None and export_fact.target_name is not None:
            origin = (export_fact.target_path.resolve(), export_fact.target_name)
        elif export_fact.local_name is not None:
            origin = local_aliases_by_file.get(file_path, {}).get(export_fact.local_name)
            if origin is None and (file_path, export_fact.local_name) in symbol_nodes:
                origin = (file_path, export_fact.local_name)
        if origin is None:
            continue
        named_exports_by_file.setdefault(file_path, {})[export_fact.exported_name] = origin
        if origin[0] != file_path:
            source_id = source_file_id.get(file_path)
            if source_id is not None:
                add_edge(
                    source_id,
                    _make_id(str(path_by_resolved.get(origin[0], origin[0]))),
                    "re_exports",
                    "export",
                    export_fact.line,
                    export_fact.file_path,
                )

    def resolve_exported_origin(target_path: Path, imported_name: str, seen: set[tuple[Path, str]] | None = None) -> tuple[Path, str]:
        target_path = target_path.resolve()
        key = (target_path, imported_name)
        if seen is None:
            seen = set()
        if key in seen:
            return key
        seen.add(key)
        origin = named_exports_by_file.get(target_path, {}).get(imported_name)
        if origin is not None:
            return resolve_exported_origin(origin[0], origin[1], seen)
        for star_target in star_exports_by_file.get(target_path, []):
            star_key = (star_target, imported_name)
            if star_key in symbol_nodes:
                return star_key
            resolved = resolve_exported_origin(star_target, imported_name, seen)
            if resolved in symbol_nodes:
                return resolved
        return key

    for import_fact in facts.imports:
        source_id = source_file_id.get(import_fact.file_path.resolve())
        if source_id is None:
            continue
        origin_path, origin_symbol = resolve_exported_origin(
            import_fact.target_path,
            import_fact.imported_name,
        )
        target_id = symbol_nodes.get((origin_path, origin_symbol))
        if target_id is None:
            continue
        add_edge(
            source_id,
            target_id,
            "imports",
            "import",
            import_fact.line,
            import_fact.file_path,
        )

    # #1146: emit file-to-file imports_from edges for package-form submodule imports.
    for from_path, to_path, line in facts.module_imports:
        try:
            from_rel = from_path.relative_to(root)
            to_rel = to_path.relative_to(root)
        except ValueError:
            continue
        source_id = _make_id(_file_stem(from_rel))
        target_id = _make_id(_file_stem(to_rel))
        add_edge(source_id, target_id, "imports_from", "submodule_import", line, from_path)

    for use_fact in facts.uses:
        file_path = use_fact.file_path.resolve()
        target_id = None
        unresolved_origin = local_aliases_by_file.get(file_path, {}).get(use_fact.local_name)
        if unresolved_origin is not None:
            origin_path, origin_symbol = resolve_exported_origin(*unresolved_origin)
            target_id = symbol_nodes.get((origin_path, origin_symbol))
        if target_id is None and use_fact.relation in ("inherits", "implements"):
            # Same-file fallback for HERITAGE only: a base declared in the same
            # file (`class X extends Y`, `interface A extends B`) has no import
            # alias, so resolve it directly against the file's own symbol nodes.
            # Scoped to heritage because same-file calls/uses already resolve via
            # the dedicated call-graph pass; widening this would duplicate those
            # edges. Import resolution still takes precedence (#1095).
            target_id = symbol_nodes.get((file_path, use_fact.local_name))
        if target_id is None:
            continue
        add_edge(
            use_fact.source_id,
            target_id,
            use_fact.relation,
            use_fact.context,
            use_fact.line,
            use_fact.file_path,
        )


def _parse_js_tree(path: Path):
    try:
        from tree_sitter import Language, Parser
        if path.suffix in (".ts", ".tsx"):
            import tree_sitter_typescript as tstypescript
            language = Language(tstypescript.language_typescript())
        else:
            import tree_sitter_javascript as tsjavascript
            language = Language(tsjavascript.language())
        source = path.read_bytes()
        parser = Parser(language)
        return source, parser.parse(source).root_node
    except Exception:
        return None


def _walk_js_tree(node):
    yield node
    for child in node.children:
        yield from _walk_js_tree(child)


def _js_module_specifier(node, source: bytes) -> str | None:
    source_node = node.child_by_field_name("source")
    if source_node is None:
        for child in node.children:
            if child.type == "string":
                source_node = child
                break
    if source_node is None:
        return None
    raw = _read_text(source_node, source).strip()
    return raw.strip("'\"`") or None


def _js_named_specifiers(node, source: bytes, specifier_type: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for child in _walk_js_tree(node):
        if child.type != specifier_type:
            continue
        name_node = child.child_by_field_name("name")
        if name_node is None:
            continue
        alias_node = child.child_by_field_name("alias")
        name = _read_text(name_node, source)
        exposed = _read_text(alias_node, source) if alias_node is not None else name
        if name and exposed:
            pairs.append((name, exposed))
    return pairs


def _js_export_clause(node):
    for child in node.children:
        if child.type == "export_clause":
            return child
    return None


def _js_export_statement_is_star(node) -> bool:
    return any(child.type == "*" for child in node.children)


def _js_lexical_aliases(node, source: bytes) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    if node.type != "lexical_declaration":
        return aliases
    for child in node.children:
        if child.type != "variable_declarator":
            continue
        name_node = child.child_by_field_name("name")
        value_node = child.child_by_field_name("value")
        if (
            name_node is not None
            and value_node is not None
            and value_node.type in ("identifier", "type_identifier")
        ):
            aliases.append((_read_text(name_node, source), _read_text(value_node, source)))
    return aliases


def _js_exported_declaration_names(node, source: bytes) -> list[str]:
    names: list[str] = []
    declaration = node.child_by_field_name("declaration")
    if declaration is None:
        return names

    if declaration.type == "lexical_declaration":
        names.extend(alias for alias, _target in _js_lexical_aliases(declaration, source))
        return names

    if declaration.type in (
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "function_declaration",
    ):
        name_node = declaration.child_by_field_name("name")
        if name_node is not None:
            names.append(_read_text(name_node, source))
    return names


def _js_top_level_function_bodies(path: Path, root_node, source: bytes) -> list[tuple[str, object]]:
    bodies: list[tuple[str, object]] = []
    stem = _file_stem(path)
    for node in root_node.children:
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name_node is not None and body is not None:
                bodies.append((_make_id(stem, _read_text(name_node, source)), body))
            continue
        if node.type != "lexical_declaration":
            continue
        for child in node.children:
            if child.type != "variable_declarator":
                continue
            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if (
                name_node is not None
                and value_node is not None
                and value_node.type == "arrow_function"
            ):
                bodies.append((_make_id(stem, _read_text(name_node, source)), value_node))
    return bodies


def _js_call_identifier(node, source: bytes) -> str | None:
    if node.type != "call_expression":
        return None
    function_node = node.child_by_field_name("function")
    if function_node is None:
        for child in node.children:
            if child.is_named:
                function_node = child
                break
    if function_node is not None and function_node.type in ("identifier", "type_identifier"):
        return _read_text(function_node, source)
    return None


_JS_PRIMITIVE_TYPES = frozenset({
    "string", "number", "boolean", "any", "unknown", "void", "never",
    "object", "null", "undefined", "bigint", "symbol", "this",
})


def _ts_heritage_clause_entries(clause_node, source: bytes) -> list[str]:
    """Return base/interface type names from an extends_clause or implements_clause."""
    out: list[str] = []
    for child in clause_node.children:
        if not child.is_named:
            continue
        if child.type in ("identifier", "type_identifier"):
            name = _read_text(child, source)
            if name:
                out.append(name)
        elif child.type == "generic_type":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                for sub in child.children:
                    if sub.type in ("type_identifier", "nested_type_identifier", "identifier"):
                        name_node = sub
                        break
            if name_node is not None:
                text = _read_text(name_node, source).rsplit(".", 1)[-1]
                if text:
                    out.append(text)
        elif child.type == "nested_type_identifier":
            text = _read_text(child, source).rsplit(".", 1)[-1]
            if text:
                out.append(text)
    return out


def _ts_collect_type_refs(node, source: bytes, generic: bool, out: list[tuple[str, str]]) -> None:
    """Walk a TS type annotation tree; append (name, role) tuples.

    role is 'type' for the outermost type position and 'generic_arg' for entries
    that appear inside `type_arguments`.
    """
    if node is None:
        return
    t = node.type
    if t == "type_annotation":
        for c in node.children:
            if c.is_named:
                _ts_collect_type_refs(c, source, generic, out)
        return
    if t in ("type_identifier", "identifier"):
        name = _read_text(node, source)
        if name and name not in _JS_PRIMITIVE_TYPES:
            out.append((name, "generic_arg" if generic else "type"))
        return
    if t == "nested_type_identifier":
        tail = _read_text(node, source).rsplit(".", 1)[-1]
        if tail and tail not in _JS_PRIMITIVE_TYPES:
            out.append((tail, "generic_arg" if generic else "type"))
        return
    if t == "generic_type":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            text = _read_text(name_node, source).rsplit(".", 1)[-1]
            if text and text not in _JS_PRIMITIVE_TYPES:
                out.append((text, "generic_arg" if generic else "type"))
        else:
            for c in node.children:
                if c.type in ("type_identifier", "nested_type_identifier"):
                    text = _read_text(c, source).rsplit(".", 1)[-1]
                    if text and text not in _JS_PRIMITIVE_TYPES:
                        out.append((text, "generic_arg" if generic else "type"))
                    break
        for c in node.children:
            if c.type == "type_arguments":
                for sub in c.children:
                    if sub.is_named:
                        _ts_collect_type_refs(sub, source, True, out)
        return
    if node.is_named:
        for c in node.children:
            if c.is_named:
                _ts_collect_type_refs(c, source, generic, out)


def _ts_walk_class_members(class_node, source: bytes, path: Path, class_nid: str,
                            facts: _SymbolResolutionFacts) -> None:
    """Emit type-relation and type-reference use facts for a class declaration node."""
    line = class_node.start_point[0] + 1
    for child in class_node.children:
        if child.type == "class_heritage":
            for clause in child.children:
                if clause.type == "extends_clause":
                    for name in _ts_heritage_clause_entries(clause, source):
                        facts.uses.append(
                            _SymbolUseFact(path, class_nid, name, "inherits", "type",
                                           clause.start_point[0] + 1)
                        )
                elif clause.type == "implements_clause":
                    for name in _ts_heritage_clause_entries(clause, source):
                        facts.uses.append(
                            _SymbolUseFact(path, class_nid, name, "implements", "type",
                                           clause.start_point[0] + 1)
                        )
        elif child.type == "extends_type_clause":
            # Interface heritage (`interface A extends B, C`) is an
            # extends_type_clause node, NOT a class_heritage. Its base entries
            # are the same node types extends_clause holds, so the helper is
            # reusable. Without this branch interface inheritance is dropped (#1095).
            for name in _ts_heritage_clause_entries(child, source):
                facts.uses.append(
                    _SymbolUseFact(path, class_nid, name, "inherits", "type",
                                   child.start_point[0] + 1)
                )

    body = class_node.child_by_field_name("body")
    if body is None:
        return

    for member in body.children:
        m_line = member.start_point[0] + 1
        if member.type in ("method_definition", "method_signature", "abstract_method_signature"):
            name_node = member.child_by_field_name("name")
            if name_node is None:
                continue
            method_name = _read_text(name_node, source)
            method_nid = _make_id(class_nid, method_name)
            params = member.child_by_field_name("parameters")
            if params is not None:
                for p in params.children:
                    if p.type not in ("required_parameter", "optional_parameter"):
                        continue
                    type_anno = p.child_by_field_name("type")
                    if type_anno is None:
                        continue
                    refs: list[tuple[str, str]] = []
                    _ts_collect_type_refs(type_anno, source, False, refs)
                    for name, role in refs:
                        ctx = "generic_arg" if role == "generic_arg" else "parameter_type"
                        facts.uses.append(
                            _SymbolUseFact(path, method_nid, name, "references", ctx, m_line)
                        )
            return_type = member.child_by_field_name("return_type")
            if return_type is not None:
                refs = []
                _ts_collect_type_refs(return_type, source, False, refs)
                for name, role in refs:
                    ctx = "generic_arg" if role == "generic_arg" else "return_type"
                    facts.uses.append(
                        _SymbolUseFact(path, method_nid, name, "references", ctx, m_line)
                    )
        elif member.type in ("public_field_definition", "property_signature"):
            type_anno = None
            for c in member.children:
                if c.type == "type_annotation":
                    type_anno = c
                    break
            if type_anno is None:
                continue
            refs = []
            _ts_collect_type_refs(type_anno, source, False, refs)
            for name, role in refs:
                ctx = "generic_arg" if role == "generic_arg" else "field"
                facts.uses.append(
                    _SymbolUseFact(path, class_nid, name, "references", ctx, m_line)
                )


def _collect_js_symbol_resolution_facts(paths: list[Path], facts: _SymbolResolutionFacts) -> None:
    js_paths = [
        path for path in paths
        if path.suffix in _JS_CACHE_BYPASS_SUFFIXES and path.suffix != ".vue"
    ]
    if not js_paths:
        return

    trees: dict[Path, tuple[bytes, object]] = {}

    for path in js_paths:
        resolved_path = path.resolve()
        parsed = _parse_js_tree(path)
        if parsed is None:
            continue
        source, root_node = parsed
        trees[resolved_path] = parsed

        for node in _walk_js_tree(root_node):
            if node.type == "export_statement":
                for name in _js_exported_declaration_names(node, source):
                    facts.declarations.append(
                        _SymbolDeclarationFact(path, name, node.start_point[0] + 1)
                    )

            if node.type != "import_statement":
                continue
            raw_module = _js_module_specifier(node, source)
            if raw_module is None:
                continue
            target_path = _resolve_js_module_path(raw_module, path.parent)
            if target_path is None:
                continue
            target_path = target_path.resolve()
            for imported_name, local_name in _js_named_specifiers(node, source, "import_specifier"):
                facts.imports.append(
                    _SymbolImportFact(
                        path,
                        local_name,
                        target_path,
                        imported_name,
                        node.start_point[0] + 1,
                    )
                )

        for node in _walk_js_tree(root_node):
            for alias, target in _js_lexical_aliases(node, source):
                facts.aliases.append(
                    _SymbolAliasFact(path, alias, target, node.start_point[0] + 1)
                )

    for path in js_paths:
        resolved_path = path.resolve()
        parsed = trees.get(resolved_path)
        if parsed is None:
            continue
        source, root_node = parsed

        for node in _walk_js_tree(root_node):
            if node.type != "export_statement":
                continue

            raw_module = _js_module_specifier(node, source)
            export_clause = _js_export_clause(node)
            if raw_module is not None:
                target_path = _resolve_js_module_path(raw_module, path.parent)
                if target_path is None:
                    continue
                target_path = target_path.resolve()
                if _js_export_statement_is_star(node):
                    facts.star_exports.append(
                        _StarExportFact(path, target_path, node.start_point[0] + 1)
                    )
                if export_clause is not None:
                    for original_name, exported_name in _js_named_specifiers(
                        export_clause, source, "export_specifier"
                    ):
                        facts.exports.append(
                            _SymbolExportFact(
                                path,
                                exported_name,
                                node.start_point[0] + 1,
                                target_path=target_path,
                                target_name=original_name,
                            )
                        )
                continue

            if export_clause is not None:
                for local_name, exported_name in _js_named_specifiers(
                    export_clause, source, "export_specifier"
                ):
                    facts.exports.append(
                        _SymbolExportFact(
                            path,
                            exported_name,
                            node.start_point[0] + 1,
                            local_name=local_name,
                        )
                    )
                continue

            for exported_name in _js_exported_declaration_names(node, source):
                facts.exports.append(
                    _SymbolExportFact(
                        path,
                        exported_name,
                        node.start_point[0] + 1,
                        local_name=exported_name,
                    )
                )

    for path in js_paths:
        resolved_path = path.resolve()
        parsed = trees.get(resolved_path)
        if parsed is None:
            continue
        source, root_node = parsed
        for source_id, body in _js_top_level_function_bodies(path, root_node, source):
            for node in _walk_js_tree(body):
                imported_name = _js_call_identifier(node, source)
                if imported_name is None:
                    continue
                facts.uses.append(
                    _SymbolUseFact(
                        path,
                        source_id,
                        imported_name,
                        "calls",
                        "call",
                        node.start_point[0] + 1,
                    )
                )

    for path in js_paths:
        resolved_path = path.resolve()
        parsed = trees.get(resolved_path)
        if parsed is None:
            continue
        source, root_node = parsed
        stem = _file_stem(path)
        for node in _walk_js_tree(root_node):
            if node.type not in (
                "class_declaration",
                "abstract_class_declaration",
                "interface_declaration",
            ):
                continue
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            class_name = _read_text(name_node, source)
            if not class_name:
                continue
            class_nid = _make_id(stem, class_name)
            _ts_walk_class_members(node, source, path, class_nid, facts)


def _parse_python_tree(path: Path):
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        source = path.read_bytes()
        parser = Parser(Language(tspython.language()))
        return source, parser.parse(source).root_node
    except Exception:
        return None


def _walk_python_tree(node):
    yield node
    for child in node.children:
        yield from _walk_python_tree(child)


def _python_import_from_module(node, source: bytes) -> tuple[int, str] | None:
    level = 0
    module_name = ""
    for child in node.children:
        if child.type == "import":
            break
        if child.type == "relative_import":
            raw = _read_text(child, source)
            level = len(raw) - len(raw.lstrip("."))
            remainder = raw.lstrip(".")
            if remainder:
                module_name = remainder
            for sub in child.children:
                if sub.type == "dotted_name":
                    module_name = _read_text(sub, source)
        elif child.type == "dotted_name":
            module_name = _read_text(child, source)
    if level == 0 and not module_name:
        return None
    return level, module_name


def _python_imported_names(node, source: bytes) -> list[tuple[str, str]]:
    names: list[tuple[str, str]] = []
    past_import = False
    for child in node.children:
        if child.type == "import":
            past_import = True
            continue
        if not past_import:
            continue
        if child.type == "dotted_name":
            name = _read_text(child, source)
            names.append((name, name.split(".")[-1]))
        elif child.type == "aliased_import":
            name_node = child.child_by_field_name("name")
            alias_node = child.child_by_field_name("alias")
            if name_node is None:
                continue
            name = _read_text(name_node, source)
            local = _read_text(alias_node, source) if alias_node is not None else name.split(".")[-1]
            names.append((name, local))
    return names


def _resolve_python_module_path(module_name: str, current_path: Path, root: Path, level: int) -> Path | None:
    if level > 0:
        base = current_path.parent
        for _ in range(level - 1):
            base = base.parent
        candidate = base / module_name.replace(".", "/") if module_name else base
    else:
        candidate = root / module_name.replace(".", "/")

    if candidate.is_dir():
        init_path = candidate / "__init__.py"
        if init_path.is_file():
            return init_path
    if candidate.is_file():
        return candidate
    py_candidate = candidate.with_suffix(".py")
    if py_candidate.is_file():
        return py_candidate
    return None


def _python_top_level_function_bodies(path: Path, root_node, source: bytes) -> list[tuple[str, object]]:
    bodies: list[tuple[str, object]] = []
    stem = _file_stem(path)
    for node in root_node.children:
        if node.type != "function_definition":
            continue
        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if name_node is not None and body is not None:
            bodies.append((_make_id(stem, _read_text(name_node, source)), body))
    return bodies


def _python_call_identifier(node, source: bytes) -> str | None:
    if node.type != "call":
        return None
    function_node = node.child_by_field_name("function")
    if function_node is not None and function_node.type == "identifier":
        return _read_text(function_node, source)
    return None


def _collect_python_symbol_resolution_facts(
    paths: list[Path],
    root: Path,
    facts: _SymbolResolutionFacts,
) -> None:
    py_paths = [path for path in paths if path.suffix == ".py"]
    if not py_paths:
        return

    trees: dict[Path, tuple[bytes, object]] = {}
    for path in py_paths:
        parsed = _parse_python_tree(path)
        if parsed is None:
            continue
        source, root_node = parsed
        trees[path.resolve()] = parsed

        for node in _walk_python_tree(root_node):
            if node.type != "import_from_statement":
                continue
            module = _python_import_from_module(node, source)
            if module is None:
                continue
            level, module_name = module
            target_path = _resolve_python_module_path(module_name, path, root, level)
            if target_path is None:
                continue
            # #1146: `from pkg import submod` — if the target is a package
            # (__init__.py) and an imported name matches a submodule file on
            # disk, emit a file-level import edge to that submodule rather
            # than only to the package.
            pkg_dir = target_path.parent if target_path.name == "__init__.py" else None
            for imported_name, local_name in _python_imported_names(node, source):
                line = node.start_point[0] + 1
                if pkg_dir is not None:
                    sub_py = pkg_dir / f"{imported_name}.py"
                    sub_pkg = pkg_dir / imported_name / "__init__.py"
                    submodule = sub_py if sub_py.is_file() else (sub_pkg if sub_pkg.is_file() else None)
                    if submodule is not None:
                        facts.module_imports.append((path, submodule, line))
                        continue
                facts.imports.append(
                    _SymbolImportFact(path, local_name, target_path, imported_name, line)
                )
                if path.name == "__init__.py":
                    facts.exports.append(
                        _SymbolExportFact(
                            path,
                            local_name,
                            line,
                            target_path=target_path,
                            target_name=imported_name,
                        )
                    )

    for path in py_paths:
        parsed = trees.get(path.resolve())
        if parsed is None:
            continue
        source, root_node = parsed
        for source_id, body in _python_top_level_function_bodies(path, root_node, source):
            for node in _walk_python_tree(body):
                imported_name = _python_call_identifier(node, source)
                if imported_name is None:
                    continue
                facts.uses.append(
                    _SymbolUseFact(
                        path,
                        source_id,
                        imported_name,
                        "calls",
                        "call",
                        node.start_point[0] + 1,
                    )
                )


def _augment_symbol_resolution_edges(
    paths: list[Path],
    nodes: list[dict],
    edges: list[dict],
    root: Path,
) -> None:
    facts = _SymbolResolutionFacts()
    _collect_js_symbol_resolution_facts(paths, facts)
    _collect_python_symbol_resolution_facts(paths, root, facts)
    _apply_symbol_resolution_facts(paths, nodes, edges, root, facts)


def _resolve_cross_file_imports(
    per_file: list[dict],
    paths: list[Path],
) -> list[dict]:
    """
    Two-pass import resolution: turn file-level imports into class-level edges.

    Pass 1 - build a global map: class/function name → node_id, per stem.
    Pass 2 - for each `from .module import Name`, look up Name in the global
              map and add a direct INFERRED edge from each class in the
              importing file to the imported entity.

    This turns:
        auth.py --imports_from--> models.py          (obvious, filtered out)
    Into:
        DigestAuth --uses--> Response  [INFERRED]    (cross-file, interesting!)
        BasicAuth  --uses--> Request   [INFERRED]
    """
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError:
        return []

    language = Language(tspython.language())
    parser = Parser(language)

    # Pass 1: _file_stem(path) → {ClassName: node_id}
    # Keyed by directory-qualified stem (e.g. "auth_models") to avoid collisions
    # when multiple files share the same filename in different directories.
    # A secondary bare-stem index handles absolute imports where only the module
    # name is known — first writer wins when names collide (inherently ambiguous).
    stem_to_entities: dict[str, dict[str, str]] = {}
    bare_to_qualified: dict[str, str] = {}
    for file_result in per_file:
        for node in file_result.get("nodes", []):
            src = node.get("source_file", "")
            if not src:
                continue
            src_path = Path(src)
            fq_stem = _file_stem(src_path)
            label = node.get("label", "")
            nid = node.get("id", "")
            # Index class-level entities only. Function/method labels end in "()"
            # so are excluded by the `endswith(")")` filter; file nodes end in ".py";
            # private/internal labels start with "_"; rationale nodes carry
            # file_type=="rationale" and must never participate in cross-file
            # import resolution (#563).
            if (
                label
                and not label.endswith((")", ".py"))
                and "_" not in label[:1]
                and node.get("file_type") != "rationale"
            ):
                stem_to_entities.setdefault(fq_stem, {})[label] = nid
                if src_path.stem not in bare_to_qualified:
                    bare_to_qualified[src_path.stem] = fq_stem

    # Pass 2: for each file, find `from .X import A, B, C` and resolve
    new_edges: list[dict] = []
    stem_to_path: dict[str, Path] = {_file_stem(p): p for p in paths}

    for file_result, path in zip(per_file, paths):
        stem = _file_stem(path)
        str_path = str(path)

        # Find all classes defined in this file (the importers).
        # Excludes rationale nodes whose labels happen not to end in ")" or ".py"
        # but which must never be treated as importing entities (#563).
        local_classes = [
            n["id"] for n in file_result.get("nodes", [])
            if n.get("source_file") == str_path
            and not n["label"].endswith((")", ".py"))
            and n["id"] != _make_id(stem)  # exclude file-level node
            and n.get("file_type") != "rationale"
        ]
        if not local_classes:
            continue

        # Parse imports from this file
        try:
            source = path.read_bytes()
            tree = parser.parse(source)
        except Exception:
            continue

        def walk_imports(node) -> None:
            if node.type == "import_from_statement":
                # Find the module name - handles both absolute and relative imports.
                # Relative: `from .models import X` → relative_import → dotted_name
                # Absolute: `from models import X`  → module_name field
                # target_fq is the directory-qualified stem used as the key in
                # stem_to_entities. Relative imports are resolved exactly via the
                # importing file's directory; absolute imports fall back to the
                # bare-stem secondary index (first-writer-wins when names collide).
                target_fq: str | None = None
                for child in node.children:
                    if child.type == "relative_import":
                        for sub in child.children:
                            if sub.type == "dotted_name":
                                raw = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                                bare = raw.split(".")[-1]
                                # Resolve relative import to exact qualified stem.
                                candidate = path.parent / f"{bare}.py"
                                target_fq = _file_stem(candidate)
                                break
                        break
                    if child.type == "dotted_name" and target_fq is None:
                        raw = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                        bare = raw.split(".")[-1]
                        target_fq = bare_to_qualified.get(bare)

                if not target_fq or target_fq not in stem_to_entities:
                    return

                # Collect imported names: dotted_name children of import_from_statement
                # that come AFTER the 'import' keyword token.
                imported_names: list[str] = []
                past_import_kw = False
                for child in node.children:
                    if child.type == "import":
                        past_import_kw = True
                        continue
                    if not past_import_kw:
                        continue
                    if child.type == "dotted_name":
                        imported_names.append(
                            source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                        )
                    elif child.type == "aliased_import":
                        # `import X as Y` - take the original name
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            imported_names.append(
                                source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                            )

                line = node.start_point[0] + 1
                for name in imported_names:
                    tgt_nid = stem_to_entities[target_fq].get(name)
                    if tgt_nid:
                        for src_class_nid in local_classes:
                            new_edges.append({
                                "source": src_class_nid,
                                "target": tgt_nid,
                                "relation": "uses",
                                "confidence": "INFERRED",
                                "source_file": str_path,
                                "source_location": f"L{line}",
                                "weight": 0.8,
                            })
            for child in node.children:
                walk_imports(child)

        walk_imports(tree.root_node)

    return new_edges


def _merge_swift_extensions(
    per_file: list[dict],
    all_nodes: list[dict],
    all_edges: list[dict],
) -> None:
    """Collapse cross-file Swift `extension Foo` nodes into the canonical `Foo`.

    tree-sitter-swift reuses `class_declaration` for both `class Foo` and
    `extension Foo`, and node ids carry the file stem, so each file that
    extends `Foo` produces its own `Foo` node. The match is done by label:
    when exactly one non-extension declaration shares the label, extension
    nodes redirect onto it. Extensions of types outside the corpus (no match)
    and ambiguous labels (more than one match) are left untouched — picking
    arbitrarily would invent edges.
    """
    extension_nids: set[str] = set()
    extension_labels: dict[str, str] = {}
    for result in per_file:
        for ext in result.get("swift_extensions", []) or []:
            extension_nids.add(ext["nid"])
            extension_labels[ext["nid"]] = ext["label"]

    if not extension_nids:
        return

    label_to_canonical: dict[str, list[str]] = {}
    for n in all_nodes:
        if n.get("id") in extension_nids:
            continue
        label = n.get("label")
        if not label:
            continue
        label_to_canonical.setdefault(label, []).append(n["id"])

    remap: dict[str, str] = {}
    for ext_nid in extension_nids:
        candidates = label_to_canonical.get(extension_labels[ext_nid], [])
        if len(candidates) != 1:
            continue
        canonical_nid = candidates[0]
        if canonical_nid != ext_nid:
            remap[ext_nid] = canonical_nid

    if not remap:
        return

    all_nodes[:] = [n for n in all_nodes if n.get("id") not in remap]

    # Each extension file's `contains` edge ends up pointing at the canonical
    # type — multiple files containing the same node is the intended shape:
    # the type owns the methods, the files own their slice. Self-loops are
    # dropped (e.g. an in-file extension method whose call already pointed at
    # the canonical type).
    rewritten: list[dict] = []
    seen_keys: set[tuple] = set()
    for e in all_edges:
        src = remap.get(e.get("source"), e.get("source"))
        tgt = remap.get(e.get("target"), e.get("target"))
        if src == tgt:
            continue
        e["source"] = src
        e["target"] = tgt
        key = (src, tgt, e.get("relation"), e.get("source_file"), e.get("source_location"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rewritten.append(e)
    all_edges[:] = rewritten


def _resolve_cross_file_java_imports(
    per_file: list[dict],
    paths: list[Path],
) -> list[dict]:
    """Two-pass Java import resolution.

    Pass 1: build a global index {ClassName: [node_id, ...]} across all Java nodes.
    Pass 2: re-parse each Java file; for every `import a.b.C;`, resolve C against
    the index. Wildcard and stdlib imports produce no edge.
    """
    try:
        import tree_sitter_java as tsjava
        from tree_sitter import Language, Parser
    except ImportError:
        return []

    language = Language(tsjava.language())
    parser = Parser(language)

    # Pass 1: class-name → node_id index (only internal, uppercase-starting names)
    name_to_ids: dict[str, list[str]] = {}
    for file_result in per_file:
        for node in file_result.get("nodes", []):
            label = node.get("label", "")
            nid = node.get("id", "")
            src = node.get("source_file", "")
            if not label or not nid or not src:
                continue
            if label.endswith(")") or label.endswith(".java"):
                continue
            if not label[0].isalpha() or not label[0].isupper():
                continue
            name_to_ids.setdefault(label, []).append(nid)

    # Pass 2: resolve imports to real node IDs
    new_edges: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for path in paths:
        file_nid = _make_id(str(path))
        try:
            source = path.read_bytes()
            tree = parser.parse(source)
        except Exception:
            continue

        def walk(n) -> None:
            if n.type == "import_declaration":
                raw = _read_text(n, source).strip()
                body = raw[len("import"):].strip().rstrip(";").strip()
                if body.startswith("static "):
                    body = body[len("static "):].strip()
                if body.endswith(".*"):
                    return
                parts = body.split(".")
                if not parts:
                    return
                last = parts[-1]
                if last and last[0].islower() and len(parts) >= 2:
                    last = parts[-2]
                at_line = n.start_point[0] + 1
                for tgt_nid in name_to_ids.get(last, []):
                    if tgt_nid == file_nid:
                        continue
                    key = (file_nid, tgt_nid)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    new_edges.append({
                        "source": file_nid,
                        "target": tgt_nid,
                        "relation": "imports",
                        "confidence": "EXTRACTED",
                        "confidence_score": 1.0,
                        "source_file": str(path),
                        "source_location": f"L{at_line}",
                        "weight": 1.0,
                    })
            for child in n.children:
                walk(child)

        walk(tree.root_node)

    return new_edges


def extract_markdown(path: Path) -> dict:
    """Extract structural nodes and edges from a Markdown file.

    Produces nodes for:
    - The file itself
    - Each heading (# / ## / ### etc.)

    Produces edges for:
    - file --contains--> heading
    - parent heading --contains--> child heading (nesting by level)
    - heading --references--> other node (when backtick `Name` matches a known pattern)

    Fenced code blocks (``` ... ```) are skipped during parsing so their
    contents don't get treated as headings, but no node is emitted for
    them — they were always orphans (only a single contains edge to the
    parent doc) and inflated the disconnected-component count (#1077).

    No tree-sitter dependency — pure line-by-line parsing.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()

    def add_node(nid: str, label: str, line: int, file_type: str = "document") -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": file_type,
                          "source_file": str_path, "source_location": f"L{line}"})

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0) -> None:
        edges.append({"source": src, "target": tgt, "relation": relation,
                      "confidence": confidence, "source_file": str_path,
                      "source_location": f"L{line}", "weight": weight})

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    # Track heading stack for nesting: [(level, nid), ...]
    heading_stack: list[tuple[int, str]] = []
    in_code_block = False

    lines = source.splitlines()
    for line_num_0, line_text in enumerate(lines):
        line_num = line_num_0 + 1

        # Skip over fenced code blocks so their contents are not parsed as
        # headings, but do not emit nodes/edges for them (#1077).
        stripped = line_text.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        # Detect headings: # Heading, ## Heading, etc.
        heading_match = re.match(r'^(#{1,6})\s+(.+)', line_text)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            h_nid = _make_id(stem, title)
            # Avoid duplicate heading IDs by appending line number
            if h_nid in seen_ids:
                h_nid = _make_id(stem, title, str(line_num))
            add_node(h_nid, title, line_num)

            # Pop headings at same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            # Connect to parent heading or file
            parent = heading_stack[-1][1] if heading_stack else file_nid
            add_edge(parent, h_nid, "contains", line_num)

            heading_stack.append((level, h_nid))
            continue

    return {"nodes": nodes, "edges": edges, "input_tokens": 0, "output_tokens": 0}


# ── Pascal / Delphi extractor ─────────────────────────────────────────────────


# Size cap for project XML files we parse with stdlib ElementTree.
# Real .csproj/.fsproj/.vbproj/.lpk files are well under 2 MiB; anything
# larger is either malformed or hostile.


# ── Main extract and collect_files ────────────────────────────────────────────


def _check_tree_sitter_version() -> None:
    """Raise a clear error if tree-sitter is too old for the new Language API."""
    try:
        from tree_sitter import LANGUAGE_VERSION
    except ImportError:
        raise ImportError(
            "tree-sitter is not installed. Run: pip install 'tree-sitter>=0.23.0'"
        )
    # Language API v2 starts at LANGUAGE_VERSION 14
    if LANGUAGE_VERSION < 14:
        import tree_sitter as _ts
        raise RuntimeError(
            f"tree-sitter {getattr(_ts, '__version__', 'unknown')} is too old. "
            f"graphify requires tree-sitter >= 0.23.0 (Language API v2). "
            f"Run: pip install --upgrade tree-sitter"
        )


# ── .NET project files (.sln, .csproj, .razor) ──────────────────────────────


# ── DM (BYOND DreamMaker) extractor ──────────────────────────────────────────
# DM identity is path-based (`/datum/object/proc/New()`), not block-based, so
# the generic class-body walker doesn't fit well.


# ── DMI (BYOND icon files) ────────────────────────────────────────────────────
# .dmi is a PNG with a tEXt/zTXt "Description" chunk containing BYOND state
# metadata. We want the icon state names (icon_state = "X" in DM code
# references them).


# ── DMM (BYOND map files) ─────────────────────────────────────────────────────
# A .dmm starts with a tile dictionary — each "key" = (type, type{var=val}, ...)
# names one or more types that compose a tile — then a grid. We only need the
# dictionary section: every type path referenced is a `uses` edge.


# ── DMF (BYOND interface forms) ───────────────────────────────────────────────


# Head tokens in an HCL traversal that are meta/builtins, not references to a
# block defined in the corpus (count.index, each.key, self.*, path.module, ...).


# Dispatch is gated to the languages this fork supports: Python, Go, SQL,
# and Markdown. The other extractor functions remain defined in this module
# (a follow-up will prune them) but are never routed to.
_DISPATCH: dict[str, Any] = {
    ".py": extract_python,
    ".go": extract_go,
    ".sql": extract_sql,
    ".md": extract_markdown,
    ".mdx": extract_markdown,
    ".qmd": extract_markdown,
}


def _get_extractor(path: Path) -> Any | None:
    """Return the correct extractor function for a file, or None if unsupported."""
    return _DISPATCH.get(path.suffix)


def _extract_single_file(args: tuple) -> tuple[int, dict]:
    """Worker function for parallel extraction. Runs in a subprocess.

    Must be at module level (not a closure) so it can be pickled by
    ProcessPoolExecutor.

    Args:
        args: (index, path_str, cache_root_str) tuple

    Returns:
        (index, result_dict) so results can be placed back in order.
    """
    idx, path_str, cache_root_str = args
    path = Path(path_str)
    cache_root = Path(cache_root_str)
    _raise_recursion_limit()
    bypass_cache = path.suffix in _JS_CACHE_BYPASS_SUFFIXES

    # Check cache first (avoid re-extraction)
    if not bypass_cache:
        cached = load_cached(path, cache_root)
        if cached is not None:
            return idx, cached

    extractor = _get_extractor(path)
    if extractor is None:
        return idx, {"nodes": [], "edges": []}

    result = _safe_extract(extractor, path)
    if not bypass_cache and "error" not in result:
        save_cached(path, result, cache_root)
    return idx, result


def _extract_parallel(
    uncached_work: list[tuple[int, Path]],
    per_file: list[dict | None],
    effective_root: Path,
    max_workers: int | None,
    total_files: int,
) -> bool:
    """Extract uncached files in parallel using ProcessPoolExecutor.

    Returns True if the pool ran to completion. Returns False if the pool
    failed in a recoverable way (typically Windows-spawn without an
    ``if __name__ == "__main__"`` guard in the calling script, which causes
    BrokenProcessPool); the caller should fall back to sequential extraction.
    """
    import concurrent.futures

    if max_workers is None:
        # Honour GRAPHIFY_MAX_WORKERS env override; otherwise scale to the
        # full CPU. The historical `, 8)` cap was a safety bound for laptops
        # in 2023 — on a 32-thread workstation it costs a 4x slowdown
        # (issue #792). Capping at len(uncached_work) keeps small jobs
        # from spawning useless idle workers.
        env_raw = os.environ.get("GRAPHIFY_MAX_WORKERS", "").strip()
        env_cap = None
        if env_raw:
            try:
                v = int(env_raw)
                if v > 0:
                    env_cap = v
            except ValueError:
                pass
        cpu_cap = env_cap if env_cap is not None else (os.cpu_count() or 4)
        max_workers = min(cpu_cap, len(uncached_work))

    root_str = str(effective_root)
    work_items = [(idx, str(path), root_str) for idx, path in uncached_work]

    done_count = 0
    _PROGRESS_INTERVAL = 100
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_extract_single_file, item): item[0] for item in work_items
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, result = future.result()
                    per_file[idx] = result
                except Exception as exc:
                    idx = futures[future]
                    print(
                        f"  warning: worker failed for {work_items[idx][1]}: {exc}",
                        file=sys.stderr, flush=True,
                    )
                done_count += 1
                if (
                    total_files >= _PROGRESS_INTERVAL
                    and done_count % _PROGRESS_INTERVAL == 0
                ):
                    print(
                        f"  AST extraction: {done_count}/{len(uncached_work)} uncached files "
                        f"({done_count * 100 // len(uncached_work)}%) [{max_workers} workers]",
                        flush=True,
                    )
    except concurrent.futures.process.BrokenProcessPool:
        # On Windows (spawn start method) the worker subprocesses re-import the
        # caller's __main__. Inline invocations like `python -c "..."` have no
        # __main__ guard, so worker bootstrap raises and the pool dies before
        # any work completes. Fall back to in-process sequential extraction —
        # slower but correct.
        print(
            "  warning: parallel extraction failed (BrokenProcessPool); "
            "falling back to sequential. On Windows this usually means the "
            'caller is missing an `if __name__ == "__main__":` guard. Pass '
            "parallel=False to extract() to skip the pool entirely.",
            flush=True,
        )
        return False
    if total_files >= _PROGRESS_INTERVAL:
        print(
            f"  AST extraction: {total_files}/{total_files} files (100%) [{max_workers} workers]",
            flush=True,
        )
    return True


def _extract_sequential(
    uncached_work: list[tuple[int, Path]],
    per_file: list[dict | None],
    effective_root: Path,
    total_files: int,
) -> None:
    """Extract uncached files sequentially (fallback for small batches)."""
    _PROGRESS_INTERVAL = 100
    for work_idx, (idx, path) in enumerate(uncached_work):
        if (
            total_files >= _PROGRESS_INTERVAL
            and work_idx % _PROGRESS_INTERVAL == 0
            and work_idx > 0
        ):
            print(
                f"  AST extraction: {work_idx}/{len(uncached_work)} uncached files ({work_idx * 100 // len(uncached_work)}%)",
                flush=True,
            )
        extractor = _get_extractor(path)
        if extractor is None:
            per_file[idx] = {"nodes": [], "edges": []}
            continue
        bypass_cache = path.suffix in _JS_CACHE_BYPASS_SUFFIXES
        result = _safe_extract(extractor, path)
        if not bypass_cache and "error" not in result:
            save_cached(path, result, effective_root)
        per_file[idx] = result
    if total_files >= _PROGRESS_INTERVAL:
        print(f"  AST extraction: {total_files}/{total_files} files (100%)", flush=True)


_PARALLEL_THRESHOLD = 20


def extract(
    paths: list[Path],
    cache_root: Path | None = None,
    *,
    parallel: bool = True,
    max_workers: int | None = None,
) -> dict:
    """Extract AST nodes and edges from a list of code files.

    Two-pass process:
    1. Per-file structural extraction (classes, functions, imports)
    2. Cross-file import resolution: turns file-level imports into
       class-level INFERRED edges (DigestAuth --uses--> Response)

    Args:
        paths: files to extract from
        cache_root: explicit root for graphify-out/cache/ (overrides the
            inferred common path prefix). Pass Path('.') when running on a
            subdirectory so the cache stays at ./graphify-out/cache/.
        parallel: if True and there are >= _PARALLEL_THRESHOLD uncached files,
            use ProcessPoolExecutor for multi-core extraction.
        max_workers: max subprocess count. Defaults to cpu_count (or the
            value of GRAPHIFY_MAX_WORKERS if set), bounded by len(uncached_work).
    """
    paths = [Path(p) for p in paths]
    _check_tree_sitter_version()
    _raise_recursion_limit()
    # Workspace package manifests/globs can change during watch or repeated extraction.
    _WORKSPACE_PACKAGE_CACHE.clear()

    # Infer a common root for cache keys (use first diverging segment, not sum of all matches)
    try:
        if not paths:
            root = Path(".")
        elif len(paths) == 1:
            root = paths[0].parent
        else:
            min_parts = min(len(p.parts) for p in paths)
            common_len = 0
            for i in range(min_parts):
                if len({p.parts[i] for p in paths}) == 1:
                    common_len += 1
                else:
                    break
            root = Path(*paths[0].parts[:common_len]) if common_len else Path(".")
    except Exception:
        root = Path(".")
    if cache_root is not None:
        root = cache_root
    root = root.resolve()

    effective_root = cache_root or root
    total = len(paths)

    # Phase 1: separate cached hits from uncached work
    per_file: list[dict | None] = [None] * total
    uncached_work: list[tuple[int, Path]] = []

    for i, path in enumerate(paths):
        if _get_extractor(path) is None:
            per_file[i] = {"nodes": [], "edges": []}
            continue
        bypass_cache = path.suffix in _JS_CACHE_BYPASS_SUFFIXES
        if not bypass_cache:
            cached = load_cached(path, effective_root)
            if cached is not None:
                per_file[i] = cached
                continue
        uncached_work.append((i, path))

    # Phase 2: extract uncached files (parallel or sequential)
    if uncached_work:
        ran_parallel = False
        if parallel and len(uncached_work) >= _PARALLEL_THRESHOLD:
            ran_parallel = _extract_parallel(
                uncached_work, per_file, effective_root, max_workers, total
            )
        if not ran_parallel:
            _extract_sequential(uncached_work, per_file, effective_root, total)

    # Fill any remaining None slots (shouldn't happen, but defensive)
    for i in range(total):
        if per_file[i] is None:
            per_file[i] = {"nodes": [], "edges": []}

    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    all_raw_calls: list[dict] = []
    for result in per_file:
        all_nodes.extend(result.get("nodes", []))
        all_edges.extend(result.get("edges", []))
        all_raw_calls.extend(result.get("raw_calls", []))

    _augment_symbol_resolution_edges(paths, all_nodes, all_edges, root)

    # Remap file node IDs from absolute-path-derived to the canonical
    # {parent_dir}_{stem} spec form so (a) graph.json edge endpoints are stable
    # across machines (#502) and (b) AST file nodes match the IDs semantic
    # subagents generate (#1033). Resolve before relativizing so paths passed in
    # relative form still anchor to the (resolved) root.
    id_remap: dict[str, str] = {}
    # Symbol node IDs embed the file stem as a prefix (_file_node_id of the path
    # the extractor saw). For a root-level file that stem picks up the absolute
    # parent directory name, so a symbol becomes <rootdir>_main_run while the
    # file node is correctly relativized to main and the skill.md spec wants
    # main_run -- splitting the symbol into AST/semantic ghosts (#1096). Relativize
    # the symbol prefix the same way, gated by source_file so two files sharing a
    # prefix can't cross-contaminate. Keyed by resolved path -> (old_pref, new_pref).
    prefix_remap: dict[Path, tuple[str, str]] = {}
    for path in paths:
        old_id = _make_id(str(path))
        try:
            rel = path.relative_to(root)
        except ValueError:
            try:
                rel = path.resolve().relative_to(root)
            except ValueError:
                continue
        new_id = _file_node_id(rel)
        if old_id != new_id:
            id_remap[old_id] = new_id
        old_pref = _file_node_id(path)
        if old_pref != new_id:
            prefix_remap[path.resolve()] = (old_pref, new_id)
    if id_remap:
        for n in all_nodes:
            if n.get("id") in id_remap:
                n["id"] = id_remap[n["id"]]
        for e in all_edges:
            if e.get("source") in id_remap:
                e["source"] = id_remap[e["source"]]
            if e.get("target") in id_remap:
                e["target"] = id_remap[e["target"]]
    if prefix_remap:
        sym_remap: dict[str, str] = {}
        for n in all_nodes:
            sf = n.get("source_file")
            if not sf:
                continue
            try:
                entry = prefix_remap.get(Path(sf).resolve())
            except Exception:
                continue
            if entry is None:
                continue
            old_pref, new_pref = entry
            nid = n.get("id", "")
            if nid.startswith(old_pref + "_"):
                new_nid = new_pref + nid[len(old_pref):]
                if new_nid != nid:
                    sym_remap[nid] = new_nid
        if sym_remap:
            for n in all_nodes:
                if n.get("id") in sym_remap:
                    n["id"] = sym_remap[n["id"]]
            for e in all_edges:
                if e.get("source") in sym_remap:
                    e["source"] = sym_remap[e["source"]]
                if e.get("target") in sym_remap:
                    e["target"] = sym_remap[e["target"]]
            # raw_calls carry caller_nid (a symbol id) consumed by the cross-file
            # call pass below, after this remap — rewrite it too or those edges
            # would dangle on their (stale) source.
            for rc in all_raw_calls:
                cn = rc.get("caller_nid")
                if cn in sym_remap:
                    rc["caller_nid"] = sym_remap[cn]

    _merge_swift_extensions(per_file, all_nodes, all_edges)
    _disambiguate_colliding_node_ids(all_nodes, all_edges, all_raw_calls, root)
    _rewire_unique_stub_nodes(all_nodes, all_edges)

    # Add cross-file class-level edges (Python only - uses Python parser internally)
    py_paths = [p for p in paths if p.suffix == ".py"]
    if py_paths:
        py_results = [r for r, p in zip(per_file, paths) if p.suffix == ".py"]
        try:
            cross_file_edges = _resolve_cross_file_imports(py_results, py_paths)
            all_edges.extend(cross_file_edges)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Cross-file import resolution failed, skipping: %s", exc)

    # Cross-file Java import resolution
    java_paths = [p for p in paths if p.suffix == ".java"]
    if java_paths:
        java_results = [r for r, p in zip(per_file, paths) if p.suffix == ".java"]
        try:
            all_edges.extend(_resolve_cross_file_java_imports(java_results, java_paths))
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Java cross-file import resolution failed, skipping: %s", exc)

    # Cross-file call resolution for all languages
    # Each extractor saved unresolved calls in raw_calls. Now that we have all
    # nodes from all files, resolve any callee that exists in another file.
    # Build name → ALL matching node IDs so we can skip ambiguous common names
    # (e.g. "log", "execute", "find") that appear in multiple files — resolving
    # those inflates god_nodes ranking with spurious cross-file edges.
    # Build label -> node_id index for cross-file call resolution.
    # Skip rationale nodes (their labels are docstring text, not callable
    # identifiers, and they were polluting matches for short names — #563).
    global_label_to_nids: dict[str, list[str]] = {}
    for n in all_nodes:
        if n.get("file_type") == "rationale":
            continue
        raw = n.get("label", "")
        normalised = raw.strip("()").lstrip(".")
        if normalised:
            key = normalised.lower()
            global_label_to_nids.setdefault(key, []).append(n["id"])

    # Build evidence index from import edges so cross-file calls backed by an
    # explicit import statement can be promoted from INFERRED to EXTRACTED.
    # Direct symbol imports (`import { foo }` / `const { foo } = require()`) are
    # the strongest evidence — caller's file_id has an `imports` edge directly to
    # the callee's symbol id. Module imports (`imports_from`) are weaker but still
    # confirm the caller pulled in the callee's source file.
    file_to_symbol_imports: dict[str, set[str]] = {}
    file_to_module_imports: dict[str, set[str]] = {}
    for e in all_edges:
        if e.get("relation") == "imports":
            file_to_symbol_imports.setdefault(e["source"], set()).add(e["target"])
        elif e.get("relation") == "imports_from":
            file_to_module_imports.setdefault(e["source"], set()).add(e["target"])

    # Map each node back to its containing file_id so we can ask
    # "did the caller's file import the callee's file?"
    # Use relativized paths to match how file node IDs were remapped above (#502).
    nid_to_file_nid: dict[str, str] = {}
    for n in all_nodes:
        sf = n.get("source_file")
        if not sf:
            continue
        sf_path = Path(sf)
        try:
            sf_rel = sf_path.relative_to(root) if sf_path.is_absolute() else sf_path
        except ValueError:
            sf_rel = sf_path
        nid_to_file_nid[n["id"]] = _file_node_id(sf_rel)

    existing_pairs = {(e["source"], e["target"]) for e in all_edges}
    for rc in all_raw_calls:
        callee = rc.get("callee", "")
        if not callee:
            continue
        if callee in _LANGUAGE_BUILTIN_GLOBALS:
            continue
        # Skip member-call callees: obj.log() → "log" has no import evidence
        # and collides with any top-level function named "log" in the corpus.
        if rc.get("is_member_call"):
            continue
        candidates = global_label_to_nids.get(callee.lower(), [])
        # Skip ambiguous names that resolve to multiple nodes — these are
        # common short names (log, execute, find) with no import evidence
        # to pick the right target; emitting all edges inflates god_nodes.
        if len(candidates) != 1:
            continue
        tgt = candidates[0]
        caller = rc["caller_nid"]
        if tgt != caller and (caller, tgt) not in existing_pairs:
            existing_pairs.add((caller, tgt))
            # Promote to EXTRACTED when there's a direct import edge from the
            # caller's file pointing at either the callee symbol itself or the
            # file the callee lives in.
            caller_file_nid = nid_to_file_nid.get(caller)
            callee_file_nid = nid_to_file_nid.get(tgt)
            imported_symbols = file_to_symbol_imports.get(caller_file_nid, set())
            imported_modules = file_to_module_imports.get(caller_file_nid, set())
            has_import_evidence = (
                tgt in imported_symbols
                or (callee_file_nid is not None and callee_file_nid in imported_modules)
            )
            if has_import_evidence:
                confidence = "EXTRACTED"
                confidence_score = 1.0
            else:
                confidence = "INFERRED"
                confidence_score = 0.8
            all_edges.append({
                "source": caller,
                "target": tgt,
                "relation": "calls",
                "context": "call",
                "confidence": confidence,
                "confidence_score": confidence_score,
                "source_file": rc.get("source_file", ""),
                "source_location": rc.get("source_location"),
                "weight": 1.0,
            })

    # Relativize source_file fields so paths are portable across machines (#555)
    for item in all_nodes + all_edges:
        sf = item.get("source_file")
        if not sf:
            continue
        sf_path = Path(sf)
        if not sf_path.is_absolute():
            continue
        try:
            item["source_file"] = sf_path.relative_to(root).as_posix()
        except ValueError:
            pass

    # Tag AST provenance so the incremental watch rebuild can distinguish
    # AST-extracted nodes from semantic/LLM nodes. On a full re-extraction
    # the watcher drops any AST-marked node missing from the fresh output
    # even when its source file still exists (#1116).
    for n in all_nodes:
        n["_origin"] = "ast"

    return {
        "nodes": all_nodes,
        "edges": all_edges,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def collect_files(target: Path, *, follow_symlinks: bool = False, root: Path | None = None) -> list[Path]:
    if target.is_file():
        return [target]
    _EXTENSIONS = set(_DISPATCH.keys())
    from graphify.detect import _is_ignored, _is_noise_dir, _load_graphifyignore
    ignore_root = root if root is not None else target
    patterns = _load_graphifyignore(ignore_root)

    def _ignored(p: Path) -> bool:
        return bool(patterns and _is_ignored(p, ignore_root, patterns))

    if not follow_symlinks:
        results: list[Path] = []
        for ext in sorted(_EXTENSIONS):
            results.extend(
                p for p in target.rglob(f"*{ext}")
                if not any(_is_noise_dir(part) for part in p.parts)
                and not _ignored(p)
            )
        return sorted(results)
    # Walk with symlink following + cycle detection
    results = []
    for dirpath, dirnames, filenames in os.walk(target, followlinks=True):
        if os.path.islink(dirpath):
            real = os.path.realpath(dirpath)
            parent_real = os.path.realpath(os.path.dirname(dirpath))
            if parent_real == real or parent_real.startswith(real + os.sep):
                dirnames.clear()
                continue
        dp = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not _is_noise_dir(d)]
        for fname in filenames:
            p = dp / fname
            if p.suffix in _EXTENSIONS and not _ignored(p):
                results.append(p)
    return sorted(results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m graphify.extract <file_or_dir> ...", file=sys.stderr)
        sys.exit(1)

    paths: list[Path] = []
    for arg in sys.argv[1:]:
        paths.extend(collect_files(Path(arg)))

    result = extract(paths)
    print(json.dumps(result, indent=2))
