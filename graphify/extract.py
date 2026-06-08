"""graphify extract - structural extraction for Markdown and SQL.

Pipeline stage 2:  detect() -> extract() -> build_graph() -> cluster() -> ...

Only two file kinds are supported:

* Markdown (``.md`` / ``.mdx`` / ``.qmd``) - headings and their nesting, pure
  line-by-line parsing, no third-party dependency.
* SQL (``.sql``) - tables, views, functions, triggers and the foreign-key /
  reads-from relationships between them, parsed with ``tree-sitter-sql``.

Every extractor returns ``{"nodes": [...], "edges": [...]}``. ``extract()``
merges the per-file results, canonicalises file-node IDs to the
``{parent_dir}_{stem}`` spec form, and relativises ``source_file`` paths so the
output is portable across machines.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable

from .cache import load_cached, save_cached

_RECURSION_LIMIT = 10_000


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
    decomposed forms of the same character (e.g. e+combining-acute vs precomposed)
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
    """File-level node ID matching the skill.md spec: ``{parent_dir}_{stem}`` -
    one parent directory level, no extension. ``rel_path`` MUST be relative to
    the project root so top-level files collapse to a bare stem (``setup.py`` ->
    ``setup``) instead of picking up the root directory name."""
    return _make_id(_file_stem(rel_path))


def _source_location(line: int | str | None) -> str | None:
    if line is None:
        return None
    if isinstance(line, str):
        return line if line.startswith("L") else f"L{line}"
    return f"L{line}"


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



# ---------------------------------------------------------------------------
# Dispatch + orchestration
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Callable[[Path], dict]] = {
    ".sql": extract_sql,
    ".md": extract_markdown,
    ".mdx": extract_markdown,
    ".qmd": extract_markdown,
}


def _get_extractor(path: Path) -> Callable[[Path], dict] | None:
    """Return the extractor for a file, or None if the suffix is unsupported."""
    return _DISPATCH.get(path.suffix)


def _extract_single_file(args: tuple) -> tuple[int, dict]:
    """Worker for parallel extraction. Runs in a subprocess, so it must live at
    module level to be picklable by ProcessPoolExecutor.

    Args: ``(index, path_str, cache_root_str)``. Returns ``(index, result)`` so
    results can be placed back in order.
    """
    idx, path_str, cache_root_str = args
    path = Path(path_str)
    cache_root = Path(cache_root_str)
    _raise_recursion_limit()

    cached = load_cached(path, cache_root)
    if cached is not None:
        return idx, cached

    extractor = _get_extractor(path)
    if extractor is None:
        return idx, {"nodes": [], "edges": []}

    result = _safe_extract(extractor, path)
    if "error" not in result:
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

    Returns True if the pool ran to completion, False if it failed in a
    recoverable way (e.g. Windows-spawn BrokenProcessPool) so the caller can
    fall back to sequential extraction.
    """
    import concurrent.futures

    if max_workers is None:
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
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_extract_single_file, item): item[0] for item in work_items}
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, result = future.result()
                    per_file[idx] = result
                except Exception as exc:
                    idx = futures[future]
                    print(f"  warning: worker failed for {work_items[idx][1]}: {exc}",
                          file=sys.stderr, flush=True)
    except concurrent.futures.process.BrokenProcessPool:
        print("  warning: parallel extraction failed (BrokenProcessPool); "
              "falling back to sequential.", flush=True)
        return False
    return True


def _extract_sequential(
    uncached_work: list[tuple[int, Path]],
    per_file: list[dict | None],
    effective_root: Path,
    total_files: int,
) -> None:
    """Extract uncached files sequentially (fallback / small batches)."""
    for idx, path in uncached_work:
        extractor = _get_extractor(path)
        if extractor is None:
            per_file[idx] = {"nodes": [], "edges": []}
            continue
        result = _safe_extract(extractor, path)
        if "error" not in result:
            save_cached(path, result, effective_root)
        per_file[idx] = result


_PARALLEL_THRESHOLD = 50


def extract(
    paths: list[Path],
    cache_root: Path | None = None,
    *,
    parallel: bool = True,
    max_workers: int | None = None,
) -> dict:
    """Extract structural nodes and edges from a list of Markdown / SQL files.

    Args:
        paths: files to extract from.
        cache_root: explicit root for ``graphify-out/cache/`` (overrides the
            inferred common path prefix). Pass ``Path('.')`` when running on a
            subdirectory so the cache stays at ``./graphify-out/cache/``.
        parallel: if True and there are >= ``_PARALLEL_THRESHOLD`` uncached
            files, use a process pool for multi-core extraction.
        max_workers: max subprocess count (defaults to cpu_count or
            ``GRAPHIFY_MAX_WORKERS``), bounded by the amount of uncached work.

    Returns ``{"nodes", "edges", "input_tokens", "output_tokens"}``. File node
    IDs are remapped to the canonical ``{parent_dir}_{stem}`` form and
    ``source_file`` paths are relativised to the inferred root.
    """
    paths = [Path(p) for p in paths]
    _check_tree_sitter_version()
    _raise_recursion_limit()

    # Infer a common root for cache keys (first diverging path segment).
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

    # Phase 1: separate cached hits from uncached work.
    per_file: list[dict | None] = [None] * total
    uncached_work: list[tuple[int, Path]] = []
    for i, path in enumerate(paths):
        if _get_extractor(path) is None:
            per_file[i] = {"nodes": [], "edges": []}
            continue
        cached = load_cached(path, effective_root)
        if cached is not None:
            per_file[i] = cached
            continue
        uncached_work.append((i, path))

    # Phase 2: extract uncached files (parallel or sequential).
    if uncached_work:
        ran_parallel = False
        if parallel and len(uncached_work) >= _PARALLEL_THRESHOLD:
            ran_parallel = _extract_parallel(
                uncached_work, per_file, effective_root, max_workers, total
            )
        if not ran_parallel:
            _extract_sequential(uncached_work, per_file, effective_root, total)

    for i in range(total):
        if per_file[i] is None:
            per_file[i] = {"nodes": [], "edges": []}

    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    for result in per_file:
        all_nodes.extend(result.get("nodes", []))
        all_edges.extend(result.get("edges", []))

    # Remap file node IDs from absolute-path-derived to the canonical
    # {parent_dir}_{stem} spec form so graph.json edge endpoints stay stable
    # across machines (#502) and AST file nodes match semantic-subagent IDs.
    id_remap: dict[str, str] = {}
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
    if id_remap:
        for n in all_nodes:
            if n.get("id") in id_remap:
                n["id"] = id_remap[n["id"]]
        for e in all_edges:
            if e.get("source") in id_remap:
                e["source"] = id_remap[e["source"]]
            if e.get("target") in id_remap:
                e["target"] = id_remap[e["target"]]

    # Relativise source_file fields so paths are portable across machines (#555).
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

    # Tag AST provenance so the incremental watch rebuild can tell AST-extracted
    # nodes apart from semantic/LLM nodes (#1116).
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
    _paths: list[Path] = []
    for arg in sys.argv[1:]:
        _paths.extend(collect_files(Path(arg)))
    print(json.dumps(extract(_paths), indent=2))
