from __future__ import annotations

import json
import math
from collections import Counter
from multiprocessing import Event
from pathlib import Path
from queue import SimpleQueue
from typing import Any

from reducelang.alphabet import get_alphabet_by_name, Alphabet
from reducelang.huffman import HuffmanModel
from reducelang.models.ppm import PPMModel
from reducelang.proofs.generator import ProofGenerator
from reducelang.proofs.renderer import ProofRenderer
from reducelang.config import Config


def _normalize_counts_from_file(path: Path, alphabet: Alphabet, progress, cancel: Event) -> tuple[Counter, int]:
    total_chars = 0
    counts: Counter[str] = Counter()
    size = path.stat().st_size or 1
    read_bytes = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(1024 * 1024)  # 1 MiB text chunk
            if not chunk:
                break
            if cancel.is_set():
                return Counter(), 0
            norm = alphabet.normalize(chunk)
            counts.update(norm)
            total_chars += len(norm)
            read_bytes += len(chunk.encode("utf-8", errors="ignore"))
            progress(min(0.99, read_bytes / size))
    progress(1.0)
    return counts, total_chars


def _normalize_counts_from_text(text: str, alphabet: Alphabet, progress, cancel: Event) -> tuple[Counter, int]:
    if cancel.is_set():
        return Counter(), 0
    norm = alphabet.normalize(text)
    counts = Counter(norm)
    progress(1.0)
    return counts, len(norm)


def job_entropy(spec: dict[str, Any], q: SimpleQueue, cancel: Event) -> dict[str, Any]:
    # spec: {"alphabet_name", "source": "file"|"text", "path"?, "text"?}
    def report(p: float, msg: str = ""):
        try:
            q.put({"type": "progress", "progress": float(p), "message": msg})
        except Exception:
            pass

    alphabet = get_alphabet_by_name(spec["alphabet_name"])
    source = spec.get("source", "text")
    if source == "file":
        counts, total = _normalize_counts_from_file(Path(spec["path"]), alphabet, lambda p: report(p), cancel)
    else:
        counts, total = _normalize_counts_from_text(spec.get("text", ""), alphabet, lambda p: report(p), cancel)

    if cancel.is_set():
        return {"canceled": True}

    # Compute entropy
    if total <= 0:
        H = 0.0
    else:
        H = 0.0
        for cnt in counts.values():
            p = cnt / total
            H -= p * math.log2(p) if p > 0 else 0.0

    # Sort counts by frequency desc
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    symbols = [k for k, _ in items]
    freqs = [int(v) for _, v in items]
    report(1.0)
    return {
        "alphabet": alphabet.name,
        "entropy": float(H),
        "total": int(total),
        "x": symbols,
        "y": freqs,
    }


def job_huffman(spec: dict[str, Any], q: SimpleQueue, cancel: Event) -> dict[str, Any]:
    def report(p: float, msg: str = ""):
        try:
            q.put({"type": "progress", "progress": float(p), "message": msg})
        except Exception:
            pass

    alphabet = get_alphabet_by_name(spec["alphabet_name"])
    source = spec.get("source", "text")
    if source == "file":
        counts, total = _normalize_counts_from_file(Path(spec["path"]), alphabet, lambda p: report(p), cancel)
    else:
        counts, total = _normalize_counts_from_text(spec.get("text", ""), alphabet, lambda p: report(p), cancel)

    if cancel.is_set():
        return {"canceled": True}

    # Build Huffman from counts without materializing full text
    model = HuffmanModel(alphabet)
    model._char_frequencies = dict(counts)
    tree = model._build_huffman_tree()
    model._huffman_tree = tree
    model._code_table = {}
    if tree is not None:
        if tree.symbol is not None and tree.left is None and tree.right is None:
            model._code_table[tree.symbol] = "0"
        else:
            model._generate_codes(tree, prefix="")
    avg_len = model._compute_avg_code_length()

    # Chart and table
    items = sorted(model._code_table.items(), key=lambda kv: (len(kv[1]), kv[0]))
    symbols = [k for k, _ in items]
    lengths = [len(v) for _, v in items]
    codes = [v for _, v in items]
    report(1.0)
    return {
        "alphabet": alphabet.name,
        "avg_bits": float(avg_len),
        "unique": int(len(counts)),
        "x": symbols,
        "y": lengths,
        "codes": codes,
    }


def job_ppm(spec: dict[str, Any], q: SimpleQueue, cancel: Event) -> dict[str, Any]:
    def report(p: float, msg: str = ""):
        try:
            q.put({"type": "progress", "progress": float(p), "message": msg})
        except Exception:
            pass

    alphabet = get_alphabet_by_name(spec["alphabet_name"])
    depth = int(spec.get("depth", 5))
    update_exclusion = bool(spec.get("update_exclusion", False))

    source = spec.get("source", "text")
    if source == "file":
        # Normalize file into memory for now (Phase 3 baseline)
        path = Path(spec["path"])
        size = path.stat().st_size or 1
        buf: list[str] = []
        read_bytes = 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                if cancel.is_set():
                    return {"canceled": True}
                norm = alphabet.normalize(chunk)
                buf.append(norm)
                read_bytes += len(chunk.encode("utf-8", errors="ignore"))
                report(min(0.9, read_bytes / size), "normalizing")
        text = "".join(buf)
    else:
        text = alphabet.normalize(spec.get("text", ""))

    if cancel.is_set():
        return {"canceled": True}

    # Train
    model = PPMModel(alphabet=alphabet, depth=depth, escape_method=str(spec.get("escape", "A")), update_exclusion=update_exclusion)
    model.fit(text)
    report(0.95, "trained")
    # Evaluate on train text for baseline bits/char (per Phase 1/2 pattern)
    bpc = model.evaluate(text) if text else 0.0
    meta = model.to_dict()

    # Top contexts by total count
    tree = getattr(model, "_context_tree", {})
    items = []
    for ctx, stats in tree.items():
        tot = int(stats.get("__total__", 0))
        if tot > 0 and ctx != "":
            items.append((ctx, tot))
    items.sort(key=lambda kv: (-kv[1], len(kv[0])))
    top = items[:100]

    # Graph: nodes for top contexts; edges parent->child when both present
    node_set = {ctx for ctx, _ in top}
    nodes = [{"id": ctx, "label": ctx} for ctx in node_set]
    edges = []
    for ctx in node_set:
        parent = ctx[1:] if len(ctx) > 0 else ""
        if parent and parent in node_set:
            edges.append({"source": parent, "target": ctx})

    # Bar chart data
    labels = [ctx for ctx, _ in top[:30]]
    totals = [int(t) for _, t in top[:30]]

    report(1.0, "done")
    return {
        "alphabet": alphabet.name,
        "depth": depth,
        "update_exclusion": update_exclusion,
        "bits_per_char": float(bpc),
        "context_tree_size": int(meta.get("context_tree_size", 0)),
        "total_contexts": int(meta.get("total_contexts", 0)),
        "top_labels": labels,
        "top_totals": totals,
        "nodes": nodes,
        "edges": edges,
    }


def job_proof(spec: dict[str, Any], q: SimpleQueue, cancel: Event) -> dict[str, Any]:
    def report(p: float, msg: str = ""):
        try:
            q.put({"type": "progress", "progress": float(p), "message": msg})
        except Exception:
            pass

    lang = str(spec.get("lang", "en"))
    corpus = str(spec.get("corpus", "text8"))
    snapshot = str(spec.get("snapshot", str(Config.DEFAULT_SNAPSHOT_DATE)))

    # Build context from results dir
    report(0.05, "loading context")
    ctx = ProofGenerator(lang=lang, corpus=corpus, snapshot=snapshot).generate_context()

    # Output directory for artifacts
    from app.core.paths import artifacts_dir

    out_dir = artifacts_dir() / "proofs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures to local dir and set relative paths
    report(0.3, "figures")
    figs_dir = out_dir / f"figs_{lang}_{corpus}_{snapshot}"
    renderer = ProofRenderer()
    figs = renderer.generate_figures(ctx, figs_dir)
    # Relative path from markdown file to figures
    ctx["figs_rel_path_md"] = figs_dir.name
    ctx["figs_rel_path_tex"] = figs_dir.name

    # Render markdown
    report(0.6, "rendering markdown")
    md_path = out_dir / f"proof_{lang}_{corpus}_{snapshot}.md"
    renderer.render_markdown(ctx, md_path)
    markdown = md_path.read_text(encoding="utf-8")

    report(1.0, "done")
    return {
        "lang": lang,
        "corpus": corpus,
        "snapshot": snapshot,
        "markdown_path": str(md_path),
        "markdown": markdown,
    }
