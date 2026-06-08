# graphify (Python · Go · SQL · Markdown edition)

Turn a folder of **Python**, **Go**, **SQL**, and **Markdown / knowledge-base
notes** into a queryable **relationship graph**.

This is a slimmed fork of [graphify](https://github.com/safishamsi/graphify),
trimmed down to graph the relationships inside Python and Go code, a SQL schema,
and a knowledge base. The other ~35 language extractors, source ingesters
(URLs, GitHub PRs, Google Workspace, video/audio, PDF/Office), and the
multi-host installers have been removed. It installs as a single
[Claude Code](https://claude.com/claude-code) skill.

## What it extracts

| Files | Nodes | Edges |
|-------|-------|-------|
| `.py` | modules, classes, functions, methods | `contains`, `method`, `calls`, `inherits`, `imports` (incl. cross-file resolution) |
| `.go` | packages, structs, interfaces, functions, methods | `contains`, `method`, `calls`, receiver-type links |
| `.sql` | tables, views, functions, procedures, triggers | `references` (foreign keys), `reads_from` (FROM/JOIN), `triggers` |
| `.md` / `.mdx` / `.qmd` | the document, each heading | `contains` (file → heading, heading → sub-heading) |

Code (Python, Go, SQL) and Markdown headings are parsed structurally via
tree-sitter and need no API key. Richer, *semantic* relationships across your
notes are produced by the optional LLM extraction pass (the Claude Code skill
drives this) — see **Semantic extraction** below.

## Install

```bash
uv tool install graphifyy            # or: pip install graphifyy
graphify install                     # copies the skill into .claude/skills/
```

Then, in Claude Code, ask it to graph a folder, or run the pipeline directly.

## Use it directly (no AI assistant)

```bash
# Build a graph from the current folder (SQL is AST-only; Markdown headings too)
graphify update .

# Ask questions of the graph
graphify query "which tables reference users?"
graphify explain "orders"
graphify path "orders" "users"
graphify affected "users"            # what depends on the users table

# Keep the graph fresh as files change
graphify watch .
```

Outputs land in `graphify-out/`:

- `graph.json` — the graph (nodes + edges)
- `GRAPH_REPORT.md` — god nodes, communities, surprising connections
- `graph.html` / `graph.svg` — visualisations
- an Obsidian-style wiki/canvas export

## Semantic extraction (optional, needs an LLM key)

Markdown notes carry meaning that headings alone don't capture. The semantic
pass reads the docs and extracts concept-level relationships into the same
graph. Set one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
… and run `graphify extract <path>` (or let the Claude Code skill do it).
A SQL-only corpus never needs a key.

## Query server (MCP)

```bash
pip install 'graphifyy[mcp]'
graphify serve graphify-out/graph.json
```

Exposes graph queries (`query_graph`, `get_node`, `get_neighbors`,
`get_community`, `god_nodes`, `graph_stats`, `shortest_path`) to any MCP client.

## Development

```bash
uv sync --all-extras
uv run pytest tests/ -q
```

## License

MIT — see [LICENSE](LICENSE). Original project © its authors; this fork keeps
the same license.
