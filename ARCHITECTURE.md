# Architecture

graphify is a Claude Code skill backed by a Python library. The skill
orchestrates the library; the library can be used standalone. This fork is
scoped to **Markdown / knowledge-base files and SQL**.

## Pipeline

```
detect()  →  extract()  →  build_graph()  →  cluster()  →  analyze()  →  report()  →  export()
```

Each stage is a single function in its own module. They communicate through
plain Python dicts and NetworkX graphs - no shared state, no side effects
outside `graphify-out/`.

## Module responsibilities

| Module | Function | Input → Output |
|--------|----------|----------------|
| `detect.py` | `detect(root)` | directory → files classified as `code` (SQL) / `document` (Markdown) |
| `extract.py` | `extract(paths)` | file paths → `{nodes, edges}` (Markdown headings, SQL schema) |
| `build.py` | `build_from_json(...)` | extraction dicts → `nx.Graph` |
| `cluster.py` | `cluster(G)` | graph → graph with `community` attr on each node |
| `analyze.py` | `analyze(G)` | graph → analysis dict (god nodes, surprises, questions) |
| `report.py` | `generate(...)` | graph + analysis → GRAPH_REPORT.md string |
| `export.py` | `to_json / to_html / to_svg / to_canvas` | graph → graph.json, graph.html, graph.svg, Obsidian canvas |
| `wiki.py` | `to_wiki(...)` | graph → Obsidian-style wiki vault |
| `callflow_html.py` / `tree_html.py` | HTML views | graphify-out files → HTML visualisations |
| `serve.py` | `start_server(graph_path)` | graph file path → MCP stdio query server |
| `watch.py` | `watch(root, ...)` | directory → incremental graph rebuilds on change |
| `cache.py` | `load_cached / save_cached` | per-file extraction cache |
| `security.py` | validation helpers | path / label → validated or raises |
| `validate.py` | `validate_extraction(data)` | extraction dict → raises on schema errors |
| `llm.py` | semantic extraction | document corpus → concept nodes/edges (optional, needs an LLM key) |

## Extractors

Only two file kinds are extracted (`graphify/extract.py`):

- **Markdown** (`.md` / `.mdx` / `.qmd`) — pure line-by-line parsing of headings
  and their nesting. No third-party dependency.
- **SQL** (`.sql`) — `tree-sitter-sql` parse of tables, views, functions,
  procedures and triggers, plus the foreign-key / `reads_from` / `triggers`
  relationships between them.

`extract()` merges per-file results, canonicalises file-node IDs to the
`{parent_dir}_{stem}` spec form, and relativises `source_file` paths.

## Extraction output schema

Every extractor returns:

```json
{
  "nodes": [
    {"id": "unique_string", "label": "human name", "source_file": "path", "source_location": "L42"}
  ],
  "edges": [
    {"source": "id_a", "target": "id_b", "relation": "contains|references|reads_from|triggers", "confidence": "EXTRACTED|INFERRED|AMBIGUOUS"}
  ]
}
```

`validate.py` enforces this schema before `build_graph()` consumes it.

## Confidence labels

| Label | Meaning |
|-------|---------|
| `EXTRACTED` | Relationship is explicitly stated in the source (e.g. a foreign key, a heading nesting) |
| `INFERRED` | Relationship is a reasonable deduction (e.g. semantic co-occurrence) |
| `AMBIGUOUS` | Relationship is uncertain; flagged for human review in GRAPH_REPORT.md |

## Testing

One test file per module under `tests/`. Run with:

```bash
uv run pytest tests/ -q
```
