#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score
from tqdm import tqdm
from umap import UMAP


SYSTEM_EXTRACTION_PROMPT = (
    "You extract concise issue statements from coroners' report text. "
    "Use British English spelling and terminology. "
    "Return JSON only. Do not include reasoning, notes, or explanations."
)

USER_EXTRACTION_PROMPT_TEMPLATE = """Extract distinct issue statements from the text below.

Rules:
1. Each issue must be one simple sentence.
2. Maximum 24 words.
3. Use direct, plain wording; avoid narrative filler.
4. Focus on systemic concern/failure, not biography or chronology.
5. Avoid duplicates and near-duplicates.
6. If no clear issue exists, return an empty list.
7. Use British English spelling and wording.

Output schema:
{{
  "issues": ["string"]
}}

Text:
{source_text}
"""

SYSTEM_LABEL_PROMPT = (
    "You name clusters of similar safety issues. Return JSON only. "
    "Use British English spelling and terminology. "
    "Do not include reasoning."
)

USER_LABEL_PROMPT_TEMPLATE = """Given these issue statements, return:
1) label: 2-6 words
2) description: one short sentence

Rules:
1. Plain English
2. No filler words like 'misc', 'other', 'general'
3. Keep label specific and practical
4. Use British English spelling and wording.

Output schema:
{{
  "label": "string",
  "description": "string"
}}

Issues:
{issues_json}
"""

SYSTEM_CANONICAL_PROMPT = (
    "You standardise one safety issue statement into a canonical issue concept. "
    "Use British English spelling and terminology. "
    "Return JSON only. Do not include reasoning."
)

USER_CANONICAL_PROMPT_TEMPLATE = """Rewrite the issue as a canonical concept and assign a short family label.

Rules:
1. Preserve meaning exactly; do not invent details.
2. Canonical issue: one concise sentence, maximum {canonical_max_words} words.
3. Issue family: 2-{issue_family_max_words} words, high-level grouping term.
4. Use neutral, reusable wording that should match semantically equivalent issues.
5. Use British English spelling and wording.

Output schema:
{{
  "canonical_issue": "string",
  "issue_family": "string"
}}

Issue:
{issue_sentence}
"""

VAGUE_PREFIXES = (
    "there was",
    "it was noted",
    "it appears",
    "the coroner",
    "concerns were raised",
    "it was found",
)


@dataclass
class RunConfig:
    input_csv: str
    output_dir: str
    subset_size: int
    random_seed: int
    ollama_host: str
    gemma_model: str
    embedding_model: str
    extraction_retries: int
    extraction_max_issues: int
    extraction_max_words: int
    extraction_max_source_chars: int
    normalise_issues: bool
    normalise_retries: int
    canonical_max_words: int
    issue_family_max_words: int
    embedding_source: str
    umap_n_neighbors: int
    umap_min_dist: float
    umap_n_components: int
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int
    hdbscan_cluster_selection_method: str
    hdbscan_cluster_selection_epsilon: float
    min_assignment_similarity: float
    min_cluster_median_similarity: float
    min_cluster_size_final: int
    analysis_mode: str
    similarity_threshold: float
    similarity_top_k: int
    similarity_min_group_size: int
    similarity_require_mutual_neighbors: bool
    similarity_centroid_merge_threshold: float
    label_clusters: bool
    label_sample_size: int
    issues_csv_input: str | None
    keep_runs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build issue extraction + clustering snapshot (Gemma + BERTopic-style)."
    )
    parser.add_argument("--input-csv", default="all_reports.csv")
    parser.add_argument("--output-dir", default="artifacts/issue_tracker_mvp")
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--gemma-model", default="gemma4:26b")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")

    parser.add_argument("--extraction-retries", type=int, default=2)
    parser.add_argument("--extraction-max-issues", type=int, default=8)
    parser.add_argument("--extraction-max-words", type=int, default=24)
    parser.add_argument("--extraction-max-source-chars", type=int, default=9000)
    parser.add_argument(
        "--normalise-issues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate canonical issue and issue family fields with Gemma.",
    )
    parser.add_argument(
        "--normalise-retries",
        type=int,
        default=1,
        help="Retries for canonical issue normalisation calls.",
    )
    parser.add_argument(
        "--canonical-max-words",
        type=int,
        default=14,
        help="Maximum words in canonical issue sentence.",
    )
    parser.add_argument(
        "--issue-family-max-words",
        type=int,
        default=4,
        help="Maximum words in issue_family label.",
    )
    parser.add_argument(
        "--embedding-source",
        choices=["canonical", "raw"],
        default="canonical",
        help="Use canonical_issue (default) or issue_sentence text for embeddings.",
    )

    parser.add_argument("--umap-n-neighbors", type=int, default=10)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--umap-n-components", type=int, default=10)

    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=12)
    parser.add_argument("--hdbscan-min-samples", type=int, default=10)
    parser.add_argument(
        "--hdbscan-cluster-selection-method",
        choices=["eom", "leaf"],
        default="leaf",
    )
    parser.add_argument("--hdbscan-cluster-selection-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--min-assignment-similarity",
        type=float,
        default=0.72,
        help="Issues below this cosine similarity to their cluster centroid are reassigned to unclustered.",
    )
    parser.add_argument(
        "--min-cluster-median-similarity",
        type=float,
        default=0.76,
        help="Clusters with median centroid similarity below this threshold are dropped to unclustered.",
    )
    parser.add_argument(
        "--min-cluster-size-final",
        type=int,
        default=8,
        help="After strict filtering, clusters below this size are moved to unclustered.",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["similarity", "cluster"],
        default="similarity",
        help="Analysis mode: cosine similarity grouping (default) or density clustering.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.9,
        help="Cosine threshold for linking two issue sentences into the same similarity group.",
    )
    parser.add_argument(
        "--similarity-top-k",
        type=int,
        default=6,
        help="Number of nearest neighbors to keep per issue in similarity outputs.",
    )
    parser.add_argument(
        "--similarity-min-group-size",
        type=int,
        default=4,
        help="Minimum connected-component size to keep as a similarity group.",
    )
    parser.add_argument(
        "--similarity-require-mutual-neighbors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require two issues to be in each other's top-k neighbors before linking groups.",
    )
    parser.add_argument(
        "--similarity-centroid-merge-threshold",
        type=float,
        default=0.95,
        help=(
            "Post-pass merge threshold for group centroids. "
            "Groups with centroid cosine similarity >= this value are merged. "
            "Set above 1.0 to disable."
        ),
    )

    parser.add_argument("--label-sample-size", type=int, default=12)
    parser.add_argument("--no-label-clusters", action="store_true")
    parser.add_argument(
        "--keep-runs",
        type=int,
        default=0,
        help="How many previous run_* directories to retain in output-dir (default: 0).",
    )

    parser.add_argument(
        "--issues-csv-input",
        default=None,
        help="Skip extraction and load issues from an existing 01_report_issues.csv file.",
    )

    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalise_issue_sentence(text: str, max_words: int) -> str | None:
    candidate = re.sub(r"\s+", " ", clean_text(text))
    candidate = re.sub(r"^[\-\*\d\.)\s]+", "", candidate)
    candidate = candidate.strip(' "\'')
    if not candidate:
        return None

    lower = candidate.lower()
    if any(lower.startswith(prefix) for prefix in VAGUE_PREFIXES):
        return None

    word_count = len(candidate.split())
    if word_count == 0 or word_count > max_words:
        return None

    if len(candidate) < 12:
        return None

    if not candidate.endswith(('.', '!', '?')):
        candidate = f"{candidate}."

    return candidate


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = clean_text(value).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(clean_text(value))
    return out


def normalise_issue_family(text: str, max_words: int) -> str:
    value = re.sub(r"\s+", " ", clean_text(text)).strip(' "\'.')
    if not value:
        return "General safety issue"
    words = value.split()
    if len(words) > max_words:
        value = " ".join(words[:max_words])
    return value


def normalise_issue_semantics(
    *,
    issue_sentence: str,
    host: str,
    model: str,
    retries: int,
    canonical_max_words: int,
    issue_family_max_words: int,
) -> tuple[str, str, str]:
    prompt = USER_CANONICAL_PROMPT_TEMPLATE.format(
        canonical_max_words=canonical_max_words,
        issue_family_max_words=issue_family_max_words,
        issue_sentence=issue_sentence,
    )

    last_error = ""
    for _ in range(retries + 1):
        try:
            parsed = ollama_chat_json(
                host=host,
                model=model,
                system_prompt=SYSTEM_CANONICAL_PROMPT,
                user_prompt=prompt,
                temperature=0.0,
                num_predict=120,
            )
            canonical_raw = clean_text(parsed.get("canonical_issue"))
            family_raw = clean_text(parsed.get("issue_family"))

            canonical = normalise_issue_sentence(canonical_raw, max_words=canonical_max_words)
            if not canonical:
                canonical = normalise_issue_sentence(issue_sentence, max_words=canonical_max_words)
            if not canonical:
                canonical = issue_sentence

            family = normalise_issue_family(family_raw, max_words=issue_family_max_words)
            return canonical, family, ""
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

    fallback = normalise_issue_sentence(issue_sentence, max_words=canonical_max_words) or issue_sentence
    return fallback, "General safety issue", last_error


def apply_issue_normalisation(issue_df: pd.DataFrame, *, args: argparse.Namespace) -> pd.DataFrame:
    working = issue_df.copy().reset_index(drop=True)
    unique_issues = unique_preserve_order([clean_text(value) for value in working["issue_sentence"].tolist()])

    cache: dict[str, tuple[str, str, str]] = {}
    for issue in tqdm(unique_issues, desc="Normalising issues", unit="issue"):
        if not issue:
            continue
        cache[issue] = normalise_issue_semantics(
            issue_sentence=issue,
            host=args.ollama_host,
            model=args.gemma_model,
            retries=args.normalise_retries,
            canonical_max_words=args.canonical_max_words,
            issue_family_max_words=args.issue_family_max_words,
        )

    canonical_values: list[str] = []
    family_values: list[str] = []
    error_values: list[str] = []
    for issue in working["issue_sentence"].map(clean_text).tolist():
        canonical, family, error = cache.get(issue, ("", "", ""))
        canonical_values.append(canonical)
        family_values.append(family)
        error_values.append(error)

    working["canonical_issue"] = canonical_values
    working["issue_family"] = family_values
    working["normalise_error"] = error_values
    return working


def strip_code_fences(text: str) -> str:
    value = clean_text(text)
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
        value = re.sub(r"```$", "", value).strip()
    return value


def parse_json_payload(raw: str) -> dict[str, Any]:
    value = strip_code_fences(raw)
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", value, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {}


def ollama_chat_json(
    *,
    host: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_predict: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": 1.0,
            "num_predict": num_predict,
        },
    }

    response = requests.post(
        f"{host.rstrip('/')}/api/chat",
        json=payload,
        timeout=240,
    )
    response.raise_for_status()
    body = response.json()
    message = (body.get("message") or {}).get("content", "")
    return parse_json_payload(message)


def extract_issues(
    *,
    source_text: str,
    host: str,
    model: str,
    retries: int,
    max_issues: int,
    max_words: int,
) -> tuple[list[str], str | None]:
    prompt = USER_EXTRACTION_PROMPT_TEMPLATE.format(source_text=source_text)

    last_error: str | None = None
    for _ in range(retries + 1):
        try:
            parsed = ollama_chat_json(
                host=host,
                model=model,
                system_prompt=SYSTEM_EXTRACTION_PROMPT,
                user_prompt=prompt,
                temperature=0.0,
                num_predict=220,
            )
            issues_raw = parsed.get("issues", [])
            if not isinstance(issues_raw, list):
                issues_raw = []

            cleaned: list[str] = []
            seen: set[str] = set()
            for value in issues_raw:
                issue = normalise_issue_sentence(clean_text(value), max_words=max_words)
                if not issue:
                    continue
                key = issue.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(issue)
                if len(cleaned) >= max_issues:
                    break
            return cleaned, None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

    return [], last_error


def sample_reports(df: pd.DataFrame, subset_size: int, seed: int) -> pd.DataFrame:
    if subset_size <= 0 or subset_size >= len(df):
        return df.reset_index(drop=True)
    sampled = df.sample(n=subset_size, random_state=seed)
    return sampled.reset_index(drop=True)


def load_reports(path: Path) -> pd.DataFrame:
    required = ["id", "url", "date", "circumstances", "concerns", "coroner", "area"]
    df = pd.read_csv(path)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    for column in required:
        df[column] = df[column].map(clean_text)

    mask = (df["circumstances"].str.len() > 0) | (df["concerns"].str.len() > 0)
    filtered = df.loc[mask].copy()
    return filtered


def build_source_text(row: pd.Series, max_chars: int) -> str:
    source = (
        f"Circumstances:\n{clean_text(row.get('circumstances'))}\n\n"
        f"Concerns:\n{clean_text(row.get('concerns'))}"
    )
    if len(source) <= max_chars:
        return source
    return source[:max_chars]


def write_issue_outputs(issues_df: pd.DataFrame, out_dir: Path) -> None:
    issues_path = out_dir / "01_report_issues.csv"
    issues_df.to_csv(issues_path, index=False)

    grouped = (
        issues_df.groupby(["report_id", "report_url", "report_date"], dropna=False)["issue_sentence"]
        .apply(lambda items: " | ".join(items))
        .reset_index(name="issue_sentences")
    )
    if "canonical_issue" in issues_df.columns:
        grouped_canonical = (
            issues_df.groupby(["report_id", "report_url", "report_date"], dropna=False)["canonical_issue"]
            .apply(lambda items: " | ".join(unique_preserve_order([clean_text(value) for value in items if clean_text(value)])))
            .reset_index(name="canonical_issue_sentences")
        )
        grouped = grouped.merge(grouped_canonical, on=["report_id", "report_url", "report_date"], how="left")
    if "issue_family" in issues_df.columns:
        grouped_family = (
            issues_df.groupby(["report_id", "report_url", "report_date"], dropna=False)["issue_family"]
            .apply(lambda items: " | ".join(unique_preserve_order([clean_text(value) for value in items if clean_text(value)])))
            .reset_index(name="issue_families")
        )
        grouped = grouped.merge(grouped_family, on=["report_id", "report_url", "report_date"], how="left")
    grouped.to_csv(out_dir / "01_report_issue_lists.csv", index=False)


def safe_date_bounds(values: pd.Series) -> tuple[str, str]:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().sum() == 0:
        return "", ""
    return parsed.min().strftime("%Y-%m-%d"), parsed.max().strftime("%Y-%m-%d")


def prune_old_runs(output_root: Path, keep_runs: int) -> list[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    keep = max(0, int(keep_runs))
    run_dirs = sorted(
        [path for path in output_root.iterdir() if path.is_dir() and path.name.startswith("run_")],
        key=lambda path: path.name,
    )
    if keep == 0:
        to_delete = run_dirs
    else:
        to_delete = run_dirs[:-keep]
    deleted: list[str] = []
    for path in to_delete:
        shutil.rmtree(path, ignore_errors=True)
        deleted.append(str(path))
    return deleted


def prune_old_cluster_artifacts(output_root: Path) -> dict[str, int]:
    """
    In clustering-only mode, keep extracted issue sentence artifacts (01_*)
    and remove older clustering/tuning outputs.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    run_dirs = [path for path in output_root.iterdir() if path.is_dir() and path.name.startswith("run_")]
    files_removed = 0
    dirs_removed = 0

    for run_dir in run_dirs:
        for child in run_dir.iterdir():
            if child.is_file() and not child.name.startswith("01_"):
                child.unlink(missing_ok=True)
                files_removed += 1
        if not any(run_dir.iterdir()):
            shutil.rmtree(run_dir, ignore_errors=True)
            dirs_removed += 1

    return {"files_removed": files_removed, "dirs_removed": dirs_removed}


def encode_embeddings(
    *,
    sentences: list[str],
    model_name: str,
    batch_size: int = 64,
) -> np.ndarray:
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load embedding model. If this is first run, ensure network access "
            f"for downloading '{model_name}', or pass a local/cached model name."
        ) from exc
    vectors: list[np.ndarray] = []

    for start in tqdm(range(0, len(sentences), batch_size), desc="Embedding issues", unit="batch"):
        batch = sentences[start : start + batch_size]
        emb = model.encode(
            batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors.append(emb)

    return np.vstack(vectors)


def reduce_dimensions(
    embeddings: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    seed: int,
) -> np.ndarray:
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def cluster_embeddings(
    reduced: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
) -> np.ndarray:
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
        store_centers="medoid",
        n_jobs=-1,
    )
    labels = clusterer.fit_predict(reduced)
    return labels


def apply_precision_filters(
    *,
    labels: np.ndarray,
    embeddings: np.ndarray,
    min_assignment_similarity: float,
    min_cluster_median_similarity: float,
    min_cluster_size_final: int,
) -> tuple[np.ndarray, np.ndarray]:
    filtered = labels.astype(int).copy()
    assignment_similarity = np.full(shape=filtered.shape, fill_value=np.nan, dtype=float)

    # First pass: reject weakly assigned issues within each cluster.
    for cluster_id in sorted({int(value) for value in filtered.tolist() if int(value) >= 0}):
        idx = np.where(filtered == cluster_id)[0]
        if len(idx) == 0:
            continue
        cluster_emb = embeddings[idx]
        centroid = cluster_emb.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            filtered[idx] = -1
            continue
        centroid = centroid / centroid_norm
        sims = cluster_emb @ centroid
        assignment_similarity[idx] = sims
        weak_idx = idx[sims < min_assignment_similarity]
        filtered[weak_idx] = -1

    # Second pass: reject tiny or low-cohesion clusters after first pass.
    for cluster_id in sorted({int(value) for value in filtered.tolist() if int(value) >= 0}):
        idx = np.where(filtered == cluster_id)[0]
        if len(idx) < min_cluster_size_final:
            filtered[idx] = -1
            continue

        cluster_emb = embeddings[idx]
        centroid = cluster_emb.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            filtered[idx] = -1
            continue
        centroid = centroid / centroid_norm
        sims = cluster_emb @ centroid
        assignment_similarity[idx] = sims
        if float(np.median(sims)) < min_cluster_median_similarity:
            filtered[idx] = -1

    # Compress cluster IDs to stable 0..N-1 after filtering.
    valid_ids = sorted({int(value) for value in filtered.tolist() if int(value) >= 0})
    remap = {old: new for new, old in enumerate(valid_ids)}
    for old, new in remap.items():
        filtered[filtered == old] = new

    return filtered, assignment_similarity


def _uf_find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = int(parent[x])
    return x


def _uf_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def _safe_similarity_threshold(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def build_similarity_neighbors(
    *,
    issue_df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int,
    embedding_text_column: str,
) -> pd.DataFrame:
    n = embeddings.shape[0]
    if n == 0:
        return pd.DataFrame(
            columns=[
                "issue_id",
                "report_url",
                "issue_sentence",
                "embedding_text",
                "neighbor_rank",
                "neighbor_issue_id",
                "neighbor_report_url",
                "neighbor_issue_sentence",
                "neighbor_embedding_text",
                "cosine_similarity",
            ]
        )

    if n == 1:
        return pd.DataFrame(
            columns=[
                "issue_id",
                "report_url",
                "issue_sentence",
                "embedding_text",
                "neighbor_rank",
                "neighbor_issue_id",
                "neighbor_report_url",
                "neighbor_issue_sentence",
                "neighbor_embedding_text",
                "cosine_similarity",
            ]
        )

    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -1.0)

    rows: list[dict[str, Any]] = []
    k = max(1, min(int(top_k), n - 1))
    for i in range(n):
        row = sims[i]
        idx = np.argpartition(-row, k - 1)[:k]
        idx = idx[np.argsort(-row[idx])]
        for rank, j in enumerate(idx.tolist(), start=1):
            rows.append(
                {
                    "issue_id": issue_df.iloc[i]["issue_id"],
                    "report_url": issue_df.iloc[i]["report_url"],
                    "issue_sentence": issue_df.iloc[i]["issue_sentence"],
                    "embedding_text": issue_df.iloc[i][embedding_text_column],
                    "neighbor_rank": rank,
                    "neighbor_issue_id": issue_df.iloc[j]["issue_id"],
                    "neighbor_report_url": issue_df.iloc[j]["report_url"],
                    "neighbor_issue_sentence": issue_df.iloc[j]["issue_sentence"],
                    "neighbor_embedding_text": issue_df.iloc[j][embedding_text_column],
                    "cosine_similarity": float(row[j]),
                }
            )
    return pd.DataFrame(rows)


def similarity_group_assignments(
    *,
    embeddings: np.ndarray,
    threshold: float,
    min_group_size: int,
    top_k: int,
    require_mutual_neighbors: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n = embeddings.shape[0]
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -1.0)

    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    t = _safe_similarity_threshold(threshold)
    k = max(1, min(int(top_k), n - 1))

    neighbor_sets: list[set[int]] = []
    for i in range(n):
        row = sims[i]
        idx = np.argpartition(-row, k - 1)[:k]
        idx = idx[row[idx] >= t]
        neighbor_sets.append({int(j) for j in idx.tolist()})

    for i, neighbors in enumerate(neighbor_sets):
        for j in neighbors:
            if require_mutual_neighbors and i not in neighbor_sets[j]:
                continue
            _uf_union(parent, rank, i, j)

    root_to_members: dict[int, list[int]] = {}
    for i in range(n):
        root = _uf_find(parent, i)
        root_to_members.setdefault(root, []).append(i)

    assignments = np.full(shape=n, fill_value=-1, dtype=int)
    group_id = 0
    for members in sorted(root_to_members.values(), key=len, reverse=True):
        if len(members) < int(min_group_size):
            continue
        for idx in members:
            assignments[idx] = group_id
        group_id += 1

    return assignments, sims


def merge_similarity_groups_by_centroid(
    *,
    assignments: np.ndarray,
    embeddings: np.ndarray,
    threshold: float,
) -> np.ndarray:
    merged = assignments.astype(int).copy()
    group_ids = sorted({int(value) for value in merged.tolist() if int(value) >= 0})
    if len(group_ids) <= 1:
        return merged

    if float(threshold) > 1.0:
        return merged

    t = _safe_similarity_threshold(threshold)

    centroids: list[np.ndarray] = []
    for group_id in group_ids:
        idx = np.where(merged == group_id)[0]
        group_emb = embeddings[idx]
        centroid = group_emb.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)

    C = np.vstack(centroids)
    S = C @ C.T
    np.fill_diagonal(S, -1.0)

    parent = np.arange(len(group_ids), dtype=int)
    rank = np.zeros(len(group_ids), dtype=int)
    tri_i, tri_j = np.where(np.triu(S, 1) >= t)
    for a, b in zip(tri_i.tolist(), tri_j.tolist(), strict=False):
        _uf_union(parent, rank, a, b)

    root_to_new: dict[int, int] = {}
    next_id = 0
    old_to_new: dict[int, int] = {}
    for local_idx, old_group_id in enumerate(group_ids):
        root = _uf_find(parent, local_idx)
        if root not in root_to_new:
            root_to_new[root] = next_id
            next_id += 1
        old_to_new[old_group_id] = root_to_new[root]

    for old_group_id, new_group_id in old_to_new.items():
        merged[merged == old_group_id] = new_group_id

    return merged


def run_similarity_analysis(
    *,
    run_output: Path,
    issue_df: pd.DataFrame,
    embeddings: np.ndarray,
    args: argparse.Namespace,
    embedding_text_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assignments, sims = similarity_group_assignments(
        embeddings=embeddings,
        threshold=args.similarity_threshold,
        min_group_size=args.similarity_min_group_size,
        top_k=args.similarity_top_k,
        require_mutual_neighbors=args.similarity_require_mutual_neighbors,
    )
    assignments = merge_similarity_groups_by_centroid(
        assignments=assignments,
        embeddings=embeddings,
        threshold=args.similarity_centroid_merge_threshold,
    )

    working = issue_df.copy().reset_index(drop=True)
    working["similarity_group_id"] = assignments

    neighbors = build_similarity_neighbors(
        issue_df=working,
        embeddings=embeddings,
        top_k=args.similarity_top_k,
        embedding_text_column=embedding_text_column,
    )
    neighbors = neighbors[neighbors["cosine_similarity"] >= _safe_similarity_threshold(args.similarity_threshold)]
    neighbors.to_csv(run_output / "02_similarity_neighbors.csv", index=False)

    unique_groups = sorted([int(value) for value in np.unique(assignments).tolist() if int(value) >= 0])
    summary_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    label_map: dict[int, tuple[str, str]] = {}

    if unique_groups and not args.no_label_clusters:
        ensure_ollama_ready(host=args.ollama_host, model=args.gemma_model)

    group_strength: list[float] = []
    label_text_column = embedding_text_column if embedding_text_column in working.columns else "issue_sentence"
    for group_id in tqdm(unique_groups, desc="Summarising similarity groups", unit="group"):
        member_idx = np.where(assignments == group_id)[0]
        group_df = working.iloc[member_idx]
        group_emb = embeddings[member_idx]
        centroid = group_emb.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        group_sims = group_emb @ centroid if centroid_norm > 0 else np.zeros(group_emb.shape[0], dtype=float)
        order = np.argsort(-group_sims)
        group_strength.extend(group_sims.tolist())

        top_n = min(len(order), max(8, int(args.label_sample_size)))
        top_idx = member_idx[order[:top_n]]
        top_sims = group_sims[order[:top_n]]

        for local_rank, (idx, sim) in enumerate(zip(top_idx.tolist(), top_sims.tolist(), strict=False), start=1):
            row = working.iloc[idx]
            example_rows.append(
                {
                    "group_id": group_id,
                    "issue_id": row["issue_id"],
                    "report_url": row["report_url"],
                    "report_date": row["report_date"],
                    "issue_sentence": row["issue_sentence"],
                    "canonical_issue": row.get("canonical_issue", ""),
                    "issue_family": row.get("issue_family", ""),
                    "similarity_to_group_centroid": float(sim),
                    "rank_in_group": local_rank,
                }
            )

        label = f"Group {group_id}"
        description = ""
        ranked_sentences = working.iloc[top_idx][label_text_column].tolist()
        unique_ranked_sentences = unique_preserve_order(ranked_sentences)
        if not args.no_label_clusters:
            issue_examples_for_label = unique_ranked_sentences[: max(1, int(args.label_sample_size))]
            try:
                label, description = label_cluster(
                    host=args.ollama_host,
                    model=args.gemma_model,
                    example_issues=issue_examples_for_label,
                )
            except Exception as exc:  # noqa: BLE001
                label = f"Group {group_id}"
                description = f"Label generation failed: {exc}"
        label_map[group_id] = (label, description)

        first_date, last_date = safe_date_bounds(group_df["report_date"])
        summary_rows.append(
            {
                "group_id": group_id,
                "group_label": label,
                "group_description": description,
                "issue_count": int(group_df.shape[0]),
                "report_count": int(group_df["report_url"].nunique()),
                "first_date": first_date,
                "last_date": last_date,
                "median_centroid_similarity": float(np.median(group_sims)) if len(group_sims) else None,
                "sample_issues": " | ".join(unique_ranked_sentences[:3]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "group_id",
                "group_label",
                "group_description",
                "issue_count",
                "report_count",
                "first_date",
                "last_date",
                "median_centroid_similarity",
                "sample_issues",
            ]
        )
    summary_df.to_csv(run_output / "03_similarity_group_summary.csv", index=False)

    examples_df = pd.DataFrame(example_rows)
    if examples_df.empty:
        examples_df = pd.DataFrame(
            columns=[
                "group_id",
                "issue_id",
                "report_url",
                "report_date",
                "issue_sentence",
                "canonical_issue",
                "issue_family",
                "similarity_to_group_centroid",
                "rank_in_group",
            ]
        )
    examples_df.to_csv(run_output / "03_similarity_group_examples.csv", index=False)

    working["group_label"] = working["similarity_group_id"].map(lambda x: label_map.get(int(x), ("", ""))[0] if int(x) >= 0 else "")
    working["group_description"] = working["similarity_group_id"].map(
        lambda x: label_map.get(int(x), ("", ""))[1] if int(x) >= 0 else ""
    )
    working.to_csv(run_output / "02_similarity_grouped_issues.csv", index=False)

    ungrouped = working[working["similarity_group_id"] == -1].copy()
    ungrouped.to_csv(run_output / "03_similarity_ungrouped_issues.csv", index=False)

    group_sizes = summary_df["issue_count"] if not summary_df.empty else pd.Series(dtype=float)
    grouped_issues = int((assignments >= 0).sum())
    metrics: dict[str, Any] = {
        "analysis_mode": "similarity",
        "total_reports": int(working["report_url"].nunique()),
        "total_issues": int(len(working)),
        "issues_per_report_mean": float(working.groupby("report_url")["issue_id"].count().mean()),
        "issues_per_report_median": float(working.groupby("report_url")["issue_id"].count().median()),
        "num_groups": int(len(unique_groups)),
        "grouped_issues": grouped_issues,
        "grouped_ratio": float(grouped_issues / len(working)) if len(working) else 0.0,
        "noise_issues": int((assignments < 0).sum()),
        "noise_ratio": float((assignments < 0).mean()) if len(assignments) else 0.0,
        "largest_group_size": int(group_sizes.max()) if len(group_sizes) else 0,
        "median_group_size": float(group_sizes.median()) if len(group_sizes) else 0.0,
        "mean_group_centroid_similarity": float(np.mean(group_strength)) if group_strength else None,
        "similarity_threshold": float(args.similarity_threshold),
        "similarity_top_k": int(args.similarity_top_k),
        "similarity_min_group_size": int(args.similarity_min_group_size),
        "similarity_require_mutual_neighbors": bool(args.similarity_require_mutual_neighbors),
        "similarity_centroid_merge_threshold": float(args.similarity_centroid_merge_threshold),
        "embedding_source": clean_text(args.embedding_source),
        "normalise_issues": bool(args.normalise_issues),
    }

    return working, metrics


def representative_examples(
    issue_df: pd.DataFrame, embeddings: np.ndarray, cluster_id: int, top_k: int = 8) -> pd.DataFrame:
    idx = issue_df.index[issue_df["cluster_id"] == cluster_id].to_numpy()
    cluster_emb = embeddings[idx]
    centroid = cluster_emb.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    sims = cluster_emb @ centroid
    order = np.argsort(-sims)
    selected = idx[order[:top_k]]
    examples = issue_df.loc[selected, ["issue_id", "report_id", "report_url", "report_date", "issue_sentence"]].copy()
    examples["similarity_to_centroid"] = sims[order[:top_k]]
    return examples


def label_cluster(
    *,
    host: str,
    model: str,
    example_issues: list[str],
) -> tuple[str, str]:
    payload_text = json.dumps(example_issues, ensure_ascii=True, indent=2)
    prompt = USER_LABEL_PROMPT_TEMPLATE.format(issues_json=payload_text)

    parsed = ollama_chat_json(
        host=host,
        model=model,
        system_prompt=SYSTEM_LABEL_PROMPT,
        user_prompt=prompt,
        temperature=0.1,
        num_predict=140,
    )
    label = clean_text(parsed.get("label")) or "Unlabelled cluster"
    description = clean_text(parsed.get("description")) or ""

    if len(label.split()) > 6:
        label = " ".join(label.split()[:6])

    return label, description


def compute_metrics(issue_df: pd.DataFrame, reduced: np.ndarray, embeddings: np.ndarray) -> dict[str, Any]:
    total_issues = len(issue_df)
    noise_mask = issue_df["cluster_id"] == -1
    clustered_mask = ~noise_mask

    cluster_sizes = issue_df.loc[clustered_mask, "cluster_id"].value_counts()
    num_clusters = int(cluster_sizes.shape[0])

    metrics: dict[str, Any] = {
        "total_reports": int(issue_df["report_url"].nunique()) if "report_url" in issue_df.columns else int(issue_df["report_id"].nunique()),
        "total_issues": int(total_issues),
        "issues_per_report_mean": float(issue_df.groupby("report_url")["issue_id"].count().mean()) if "report_url" in issue_df.columns else float(issue_df.groupby("report_id")["issue_id"].count().mean()),
        "issues_per_report_median": float(issue_df.groupby("report_url")["issue_id"].count().median()) if "report_url" in issue_df.columns else float(issue_df.groupby("report_id")["issue_id"].count().median()),
        "num_clusters": num_clusters,
        "noise_issues": int(noise_mask.sum()),
        "noise_ratio": float(noise_mask.mean()),
        "largest_cluster_size": int(cluster_sizes.max()) if num_clusters else 0,
        "median_cluster_size": float(cluster_sizes.median()) if num_clusters else 0.0,
    }

    clustered_indices = np.where(clustered_mask.to_numpy())[0]
    clustered_labels = issue_df.loc[clustered_mask, "cluster_id"].to_numpy()

    if num_clusters >= 2 and len(clustered_indices) >= 10:
        reduced_clustered = reduced[clustered_indices]
        metrics["silhouette_score"] = float(silhouette_score(reduced_clustered, clustered_labels))
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(reduced_clustered, clustered_labels))
    else:
        metrics["silhouette_score"] = None
        metrics["davies_bouldin_score"] = None

    if num_clusters:
        cohesion_values: list[float] = []
        for cluster_id in cluster_sizes.index.tolist():
            idx = issue_df.index[issue_df["cluster_id"] == cluster_id].to_numpy()
            cluster_emb = embeddings[idx]
            center = cluster_emb.mean(axis=0)
            center_norm = np.linalg.norm(center)
            if center_norm == 0:
                continue
            center = center / center_norm
            sims = cluster_emb @ center
            cohesion_values.append(float(np.mean(sims)))
        metrics["mean_cluster_cohesion_cosine"] = float(np.mean(cohesion_values)) if cohesion_values else None
    else:
        metrics["mean_cluster_cohesion_cosine"] = None

    return metrics


def heuristic_recommendation(metrics: dict[str, Any]) -> str:
    noise = metrics.get("noise_ratio")
    silhouette = metrics.get("silhouette_score")

    lines: list[str] = []
    lines.append("Quick tuning recommendation")

    if isinstance(noise, float):
        if noise > 0.75:
            lines.append("- Noise is high (>75%): consider broader clusters (lower min_samples or min_cluster_size).")
        elif noise < 0.25:
            lines.append("- Noise is low (<25%): check for over-merging; try stricter settings.")
        else:
            lines.append("- Noise ratio is in a balanced range (25%-75%).")

    if isinstance(silhouette, float):
        if silhouette < 0.10:
            lines.append("- Silhouette is low: increase embedding quality or try stricter clustering.")
        elif silhouette > 0.35:
            lines.append("- Silhouette is strong: current separation looks healthy.")
        else:
            lines.append("- Silhouette is moderate: inspect cluster examples before tuning.")
    else:
        lines.append("- Not enough multi-cluster assignments for silhouette; inspect cluster examples manually.")

    lines.append("- Prioritize avoiding over-merged clusters over forcing every issue into a cluster.")
    return "\n".join(lines)


def heuristic_recommendation_similarity(metrics: dict[str, Any]) -> str:
    grouped_ratio = metrics.get("grouped_ratio")
    num_groups = metrics.get("num_groups")
    lines: list[str] = []
    lines.append("Quick similarity recommendation")

    if isinstance(grouped_ratio, float):
        if grouped_ratio < 0.20:
            lines.append("- Very strict grouping (<20% grouped): reduce similarity threshold or min group size.")
        elif grouped_ratio > 0.70:
            lines.append("- Broad grouping (>70% grouped): raise similarity threshold for issue-level precision.")
        else:
            lines.append("- Grouped ratio is in a useful issue-tracking range.")

    if isinstance(num_groups, int):
        if num_groups < 10:
            lines.append("- Few groups detected: lower min group size to increase granularity.")
        elif num_groups > 120:
            lines.append("- Very many groups: increase threshold or min group size to reduce fragmentation.")
        else:
            lines.append("- Group count looks operationally manageable.")

    lines.append("- For issue tracking, prefer higher precision and accept many ungrouped sentences.")
    return "\n".join(lines)


def ensure_ollama_ready(host: str, model: str) -> None:
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=20)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Cannot reach Ollama at {host}. Start Ollama first. Original error: {exc}"
        ) from exc

    tags = response.json().get("models", [])
    names = {clean_text(item.get("name")) for item in tags}
    if model not in names:
        raise RuntimeError(
            f"Model '{model}' not found in Ollama tags. Available examples: {sorted(list(names))[:8]}"
        )


def run_extraction(
    reports_df: pd.DataFrame,
    *,
    host: str,
    model: str,
    retries: int,
    max_issues: int,
    max_words: int,
    max_source_chars: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    issue_counter = 0

    for _, report in tqdm(reports_df.iterrows(), total=len(reports_df), desc="Extracting issues", unit="report"):
        source_text = build_source_text(report, max_chars=max_source_chars)
        issues, error_message = extract_issues(
            source_text=source_text,
            host=host,
            model=model,
            retries=retries,
            max_issues=max_issues,
            max_words=max_words,
        )

        if not issues:
            rows.append(
                {
                    "issue_id": f"iss_{issue_counter:07d}",
                    "report_id": report["id"],
                    "report_url": report["url"],
                    "report_date": report["date"],
                    "coroner": report["coroner"],
                    "area": report["area"],
                    "issue_sentence": "",
                    "extract_error": clean_text(error_message),
                }
            )
            issue_counter += 1
            continue

        for issue_sentence in issues:
            rows.append(
                {
                    "issue_id": f"iss_{issue_counter:07d}",
                    "report_id": report["id"],
                    "report_url": report["url"],
                    "report_date": report["date"],
                    "coroner": report["coroner"],
                    "area": report["area"],
                    "issue_sentence": issue_sentence,
                    "extract_error": "",
                }
            )
            issue_counter += 1

    issue_df = pd.DataFrame(rows)
    return issue_df


def main() -> None:
    args = parse_args()

    root_output = Path(args.output_dir)
    issues_csv_input_path: Path | None = None
    if args.issues_csv_input:
        issues_csv_input_path = Path(args.issues_csv_input).expanduser()
        if not issues_csv_input_path.is_absolute():
            issues_csv_input_path = Path.cwd() / issues_csv_input_path
        issues_csv_input_path = issues_csv_input_path.resolve()
        if not issues_csv_input_path.exists():
            raise FileNotFoundError(f"Issues CSV input not found: {issues_csv_input_path}")

    if args.issues_csv_input:
        cleanup = prune_old_cluster_artifacts(root_output)
        if cleanup["files_removed"] or cleanup["dirs_removed"]:
            print(
                "Pruned old clustering artifacts from "
                f"{root_output} (files: {cleanup['files_removed']}, dirs: {cleanup['dirs_removed']})."
            )
    else:
        deleted_runs = prune_old_runs(root_output, keep_runs=args.keep_runs)
        if deleted_runs:
            print(f"Pruned {len(deleted_runs)} older run directories from {root_output}.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_output = root_output / f"run_{timestamp}"
    run_output.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        input_csv=args.input_csv,
        output_dir=str(run_output),
        subset_size=args.subset_size,
        random_seed=args.random_seed,
        ollama_host=args.ollama_host,
        gemma_model=args.gemma_model,
        embedding_model=args.embedding_model,
        extraction_retries=args.extraction_retries,
        extraction_max_issues=args.extraction_max_issues,
        extraction_max_words=args.extraction_max_words,
        extraction_max_source_chars=args.extraction_max_source_chars,
        normalise_issues=args.normalise_issues,
        normalise_retries=args.normalise_retries,
        canonical_max_words=args.canonical_max_words,
        issue_family_max_words=args.issue_family_max_words,
        embedding_source=args.embedding_source,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_n_components=args.umap_n_components,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_cluster_selection_method=args.hdbscan_cluster_selection_method,
        hdbscan_cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        min_assignment_similarity=args.min_assignment_similarity,
        min_cluster_median_similarity=args.min_cluster_median_similarity,
        min_cluster_size_final=args.min_cluster_size_final,
        analysis_mode=args.analysis_mode,
        similarity_threshold=args.similarity_threshold,
        similarity_top_k=args.similarity_top_k,
        similarity_min_group_size=args.similarity_min_group_size,
        similarity_require_mutual_neighbors=args.similarity_require_mutual_neighbors,
        similarity_centroid_merge_threshold=args.similarity_centroid_merge_threshold,
        label_clusters=not args.no_label_clusters,
        label_sample_size=args.label_sample_size,
        issues_csv_input=args.issues_csv_input,
        keep_runs=args.keep_runs,
    )

    (run_output / "05_run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    if args.issues_csv_input:
        issue_df = pd.read_csv(issues_csv_input_path)
        required_cols = {"issue_id", "report_id", "report_url", "report_date", "issue_sentence"}
        missing = required_cols - set(issue_df.columns)
        if missing:
            raise ValueError(f"Input issues CSV missing columns: {sorted(missing)}")
        issue_df["report_id"] = issue_df["report_id"].map(clean_text)
        issue_df["report_url"] = issue_df["report_url"].map(clean_text)
        issue_df["report_date"] = issue_df["report_date"].map(clean_text)
        issue_df["issue_sentence"] = issue_df["issue_sentence"].map(clean_text)
        if "canonical_issue" in issue_df.columns:
            issue_df["canonical_issue"] = issue_df["canonical_issue"].map(clean_text)
        if "issue_family" in issue_df.columns:
            issue_df["issue_family"] = issue_df["issue_family"].map(clean_text)
        issue_df = issue_df[issue_df["issue_sentence"].str.len() > 0].reset_index(drop=True)
    else:
        ensure_ollama_ready(host=args.ollama_host, model=args.gemma_model)

        reports_df = load_reports(Path(args.input_csv))
        reports_df = sample_reports(reports_df, subset_size=args.subset_size, seed=args.random_seed)

        issue_df_raw = run_extraction(
            reports_df,
            host=args.ollama_host,
            model=args.gemma_model,
            retries=args.extraction_retries,
            max_issues=args.extraction_max_issues,
            max_words=args.extraction_max_words,
            max_source_chars=args.extraction_max_source_chars,
        )

        issue_df_raw.to_csv(run_output / "01_report_issues_raw.csv", index=False)
        issue_df = issue_df_raw[issue_df_raw["issue_sentence"].str.len() > 0].reset_index(drop=True)

        failed_reports = issue_df_raw[issue_df_raw["extract_error"].str.len() > 0]["report_id"].nunique()
        print(f"Extraction complete. Reports with extraction errors: {failed_reports}")

    if issue_df.empty:
        raise RuntimeError("No extracted issue sentences were produced. Clustering cannot proceed.")

    if args.normalise_issues:
        ensure_ollama_ready(host=args.ollama_host, model=args.gemma_model)
        issue_df = apply_issue_normalisation(issue_df, args=args)
    else:
        if "canonical_issue" not in issue_df.columns:
            issue_df["canonical_issue"] = issue_df["issue_sentence"]
        if "issue_family" not in issue_df.columns:
            issue_df["issue_family"] = ""
        if "normalise_error" not in issue_df.columns:
            issue_df["normalise_error"] = ""

    write_issue_outputs(issue_df, run_output)

    if args.embedding_source == "canonical":
        canonical_series = issue_df["canonical_issue"].map(clean_text) if "canonical_issue" in issue_df.columns else pd.Series([""] * len(issue_df))
        issue_df["embedding_text"] = canonical_series.where(canonical_series.str.len() > 0, issue_df["issue_sentence"])
        embedding_text_column = "embedding_text"
    else:
        issue_df["embedding_text"] = issue_df["issue_sentence"]
        embedding_text_column = "embedding_text"

    sentences = issue_df[embedding_text_column].tolist()
    embeddings = encode_embeddings(sentences=sentences, model_name=args.embedding_model)
    np.save(run_output / "02_issue_embeddings.npy", embeddings)

    reduced = reduce_dimensions(
        embeddings,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        n_components=args.umap_n_components,
        seed=args.random_seed,
    )
    np.save(run_output / "02_issue_reduced.npy", reduced)

    if args.analysis_mode == "similarity":
        _, metrics = run_similarity_analysis(
            run_output=run_output,
            issue_df=issue_df,
            embeddings=embeddings,
            args=args,
            embedding_text_column=embedding_text_column,
        )
        metrics["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        metrics["output_dir"] = str(run_output)

        all_metrics = {
            "config": asdict(config),
            "metrics": metrics,
        }
        (run_output / "04_similarity_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
        pd.DataFrame([metrics]).to_csv(run_output / "04_similarity_metrics.csv", index=False)

        recommendation = heuristic_recommendation_similarity(metrics)
        (run_output / "05_similarity_recommendations.txt").write_text(recommendation + "\n", encoding="utf-8")

        print("\nRun complete.")
        print(f"Output directory: {run_output}")
        print("Key files:")
        print(f"- {run_output / '01_report_issues.csv'}")
        print(f"- {run_output / '02_similarity_grouped_issues.csv'}")
        print(f"- {run_output / '03_similarity_group_summary.csv'}")
        print(f"- {run_output / '03_similarity_group_examples.csv'}")
        print(f"- {run_output / '04_similarity_metrics.csv'}")
        return

    labels = cluster_embeddings(
        reduced,
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        cluster_selection_method=args.hdbscan_cluster_selection_method,
        cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
    )
    labels, assignment_similarity = apply_precision_filters(
        labels=labels,
        embeddings=embeddings,
        min_assignment_similarity=args.min_assignment_similarity,
        min_cluster_median_similarity=args.min_cluster_median_similarity,
        min_cluster_size_final=args.min_cluster_size_final,
    )

    issue_df = issue_df.copy()
    issue_df["cluster_id"] = labels
    issue_df["assignment_similarity"] = assignment_similarity
    issue_df["umap_x"] = reduced[:, 0]
    if reduced.shape[1] >= 2:
        issue_df["umap_y"] = reduced[:, 1]
    else:
        issue_df["umap_y"] = np.nan
    issue_df.to_csv(run_output / "02_clustered_issues.csv", index=False)

    unique_clusters = sorted([int(value) for value in issue_df["cluster_id"].unique().tolist() if int(value) >= 0])

    cluster_summary_rows: list[dict[str, Any]] = []
    cluster_examples_rows: list[dict[str, Any]] = []

    if unique_clusters:
        if not args.no_label_clusters:
            ensure_ollama_ready(host=args.ollama_host, model=args.gemma_model)

        for cluster_id in tqdm(unique_clusters, desc="Summarising clusters", unit="cluster"):
            cluster_items = issue_df[issue_df["cluster_id"] == cluster_id]
            first_date, last_date = safe_date_bounds(cluster_items["report_date"])
            examples = representative_examples(
                issue_df=issue_df,
                embeddings=embeddings,
                cluster_id=cluster_id,
                top_k=max(8, args.label_sample_size),
            )

            for _, example in examples.iterrows():
                cluster_examples_rows.append(
                    {
                        "cluster_id": cluster_id,
                        "issue_id": example["issue_id"],
                        "report_id": example["report_id"],
                        "report_url": example["report_url"],
                        "report_date": example["report_date"],
                        "issue_sentence": example["issue_sentence"],
                        "similarity_to_centroid": example["similarity_to_centroid"],
                    }
                )

            label = f"Cluster {cluster_id}"
            description = ""
            if not args.no_label_clusters:
                issue_examples_for_label = examples["issue_sentence"].head(args.label_sample_size).tolist()
                try:
                    label, description = label_cluster(
                        host=args.ollama_host,
                        model=args.gemma_model,
                        example_issues=issue_examples_for_label,
                    )
                except Exception as exc:  # noqa: BLE001
                    label = f"Cluster {cluster_id}"
                    description = f"Label generation failed: {exc}"
            cluster_summary_rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": label,
                    "cluster_description": description,
                    "issue_count": int(cluster_items.shape[0]),
                    "report_count": int(cluster_items["report_url"].nunique()),
                    "first_date": first_date,
                    "last_date": last_date,
                    "sample_issues": " | ".join(examples["issue_sentence"].head(3).tolist()),
                }
            )

        label_frame = pd.DataFrame(cluster_summary_rows)[["cluster_id", "cluster_label", "cluster_description"]]
        issue_df = issue_df.merge(label_frame, on="cluster_id", how="left")
    else:
        issue_df["cluster_label"] = ""
        issue_df["cluster_description"] = ""

    issue_df.to_csv(run_output / "02_clustered_issues.csv", index=False)

    cluster_summary_df = pd.DataFrame(cluster_summary_rows)
    if cluster_summary_df.empty:
        cluster_summary_df = pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_label",
                "cluster_description",
                "issue_count",
                "report_count",
                "first_date",
                "last_date",
                "sample_issues",
            ]
        )
    cluster_summary_df.to_csv(run_output / "03_cluster_summary.csv", index=False)

    cluster_examples_df = pd.DataFrame(cluster_examples_rows)
    if cluster_examples_df.empty:
        cluster_examples_df = pd.DataFrame(
            columns=[
                "cluster_id",
                "issue_id",
                "report_id",
                "report_url",
                "report_date",
                "issue_sentence",
                "similarity_to_centroid",
            ]
        )
    cluster_examples_df.to_csv(run_output / "03_cluster_examples.csv", index=False)

    unclustered_df = issue_df[issue_df["cluster_id"] == -1].copy()
    unclustered_df.to_csv(run_output / "03_unclustered_issues.csv", index=False)

    metrics = compute_metrics(issue_df=issue_df, reduced=reduced, embeddings=embeddings)
    metrics["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["output_dir"] = str(run_output)

    all_metrics = {
        "config": asdict(config),
        "metrics": metrics,
    }

    (run_output / "04_tuning_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(run_output / "04_tuning_metrics.csv", index=False)

    recommendation = heuristic_recommendation(metrics)
    (run_output / "05_tuning_recommendations.txt").write_text(recommendation + "\n", encoding="utf-8")

    print("\nRun complete.")
    print(f"Output directory: {run_output}")
    print("Key files:")
    print(f"- {run_output / '01_report_issues.csv'}")
    print(f"- {run_output / '02_clustered_issues.csv'}")
    print(f"- {run_output / '03_cluster_summary.csv'}")
    print(f"- {run_output / '03_cluster_examples.csv'}")
    print(f"- {run_output / '04_tuning_metrics.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
