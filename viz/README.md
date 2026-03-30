# SFT Pipeline — Visualization App

Streamlit dashboard for exploring pipeline outputs: prompt browser, cluster scatter plot, stats, and answers viewer.

## Quickstart

```bash
# Install dependencies (separate from the pipeline)
pip install -r viz/requirements.txt

# Export a snapshot from a completed run
python viz/export.py --run-dir /path/to/run

# Launch the app
streamlit run viz/app.py
```

## Sharing the URL

```bash
# Expose publicly via Cloudflare Tunnel (free, no account needed)
cloudflared tunnel --url http://localhost:8501
```

This generates a `*.trycloudflare.com` URL you can share with anyone. The tunnel is temporary — it dies when the process stops. For a permanent URL, run it in a `screen`/`tmux` session or use `nohup`.

## Export Script

```
python viz/export.py [options]

Options:
  --run-dir   Path to the pipeline run directory (the base_path in config). Required.
  --out       Output path for snapshot.parquet. Default: viz/data/snapshot.parquet
  --sample    Max prompts to include. Default: 50000
  --seed      Random seed for sampling. Default: 42
```

The script auto-detects which stages are complete:

| Stage | What it adds |
|-------|-------------|
| Stage 1 | prompts, source, domain |
| Stage 3 | difficulty, cluster_id |
| Stage 3 embeddings | umap_x, umap_y (UMAP computed here, ~1 min for 50k) |
| Stage 5 | reasoning, answer |
| Stage 6 | passed_filters |

Re-run export any time after a stage completes. The app picks up the new snapshot automatically on next page load (cache is busted by file mtime).

## Pages

| Page | Description |
|------|-------------|
| Home (`app.py`) | Stage status badges, domain breakdown, snapshot summary |
| Stats | Domain/source/difficulty charts, dedup rate, full source table |
| Prompts | Searchable filterable table; click a row to expand the full prompt |
| Clusters | UMAP scatter coloured by domain/cluster/difficulty; click a point to see the prompt |
| Answers | Prompt + reasoning + answer, paginated; "passed filters only" toggle after Stage 6 |

## File Structure

```
viz/
├── export.py              # Snapshot export CLI
├── app.py                 # Streamlit entry point (home page)
├── pages/
│   ├── 1_Stats.py
│   ├── 2_Prompts.py
│   ├── 3_Clusters.py
│   └── 4_Answers.py
├── components/
│   ├── data_loader.py     # st.cache_data loader
│   └── filters.py         # Shared sidebar filter widgets
├── data/                  # snapshot.parquet and meta.json go here
└── requirements.txt
```

## Notes

- The `data/` directory is in `.gitignore` — snapshots are not committed to the repo.
- UMAP is computed once during export, not on app load. For 50k points it takes ~60s.
- The app runs against a static snapshot; it does not read live pipeline output. Re-run export to refresh.
- Pages that depend on stages not yet complete show an informational message instead of erroring.
