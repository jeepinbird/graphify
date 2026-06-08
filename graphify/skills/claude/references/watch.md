# graphify reference: watch a folder

Load this when the user passed `--watch`. It is not part of the default build.

Start a background watcher that monitors a folder and auto-updates the graph when files change.

```bash
python3 -m graphify.watch INPUT_PATH --debounce 3
```

Replace INPUT_PATH with the folder to watch. Behavior depends on what changed:

- **Code files (.py, .go, .sql):** re-runs AST extraction + rebuild + cluster immediately, no LLM needed. `graph.json` and `GRAPH_REPORT.md` are updated automatically.
- **Markdown docs (.md, .mdx, .qmd):** writes a `graphify-out/needs_update` flag and prints a notification to run `/graphify --update` (LLM semantic re-extraction required).

Debounce (default 3s): waits until file activity stops before triggering, so a wave of parallel agent writes doesn't trigger a rebuild per file.

Press Ctrl+C to stop.

For agentic workflows: run `--watch` in a background terminal. Code changes from agent waves are picked up automatically between waves. If agents are also writing Markdown notes, you'll need a manual `/graphify --update` after those waves.
