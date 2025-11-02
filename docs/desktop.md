# ReduceLang Desktop (Phase 1)

A Windows-native desktop shell built with PySide6.

Tabs:

- Alphabet / Entropy: paste text or load a file, choose an alphabet, compute entropy and view character frequencies.
- Huffman: paste text or load a file, train Huffman, see average code length (training distribution), code lengths chart, and the code table.
- Jobs: monitor queued/running jobs, progress, and cancel selected jobs.
- PPM: select order and (optionally) update exclusion, train in background, view bits/char and top contexts; small trie preview with Cytoscape.
- Proofs: select language, corpus, and snapshot, generate Markdown proof from your saved results, preview inline with KaTeX; figures are generated locally.
- Settings: choose theme (System/Light/Dark), set max workers, and override artifacts/cache directories. Dark mode defaults to the OS theme.

## Run locally

1. Install optional desktop dependencies:

   - With uv: `uv sync --extra desktop`
   - Or pip: `pip install -e .[desktop]`

2. Start the app:

```bash
uv run -q python -m app.main
```

## Notes

- Runs fully offline; ECharts is vendored under `app/assets/`.
- Jobs run in separate processes; you can continue using the UI while jobs execute.
- More screens (PPM, Proofs, Corpus) will be added in later phases.
- Proofs expect results under `results/entropy/<lang>/<corpus>/<snapshot>/`. Use the "Browse Snapshot..." button to point to that folder.

- Theme: default is 'System' â€” the app detects the palette and applies a dark or light theme. Switch in Settings to force Light/Dark; embedded charts and previews update live.
