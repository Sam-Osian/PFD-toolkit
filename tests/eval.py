import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer 

###############################################################################
# Configuration – columns we care about (others are ignored)
###############################################################################

EVAL_COLS: List[str] = [
    "Date",
    "CoronerName",
    "Area",
    "Receiver",
    "InvestigationAndInquest",
    "CircumstancesOfDeath",
    "MattersOfConcern",
]

###############################################################################
# Similarity helpers
###############################################################################

def _ensure_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    if pd.isna(val):
        return ""
    return str(val)


def _normalise_simple(text: str) -> str:
    import re, unicodedata

    text = _ensure_text(text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text.lower()).strip(" ,.;:\n\t")
    return text


def _jaccard_5gram(a: str, b: str) -> float:
    a, b = _ensure_text(a), _ensure_text(b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    A = {a[i : i + 5] for i in range(len(a) - 4)}
    B = {b[i : i + 5] for i in range(len(b) - 4)}
    return len(A & B) / len(A | B)


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)

###############################################################################
# Comparator registry limited to EVAL_COLS
###############################################################################

@dataclass
class Comparator:
    func: Callable[[str, str], float]
    default_thresh: float
    def __call__(self, gt, pred):
        return self.func(_ensure_text(gt), _ensure_text(pred))

class _Comparators:
    def __init__(self):
        self._embedder: Optional[SentenceTransformer] = None
        self.table: Dict[str, Comparator] = {
            "Date": Comparator(lambda g, p: 1.0 if g == p else 0.0, 1.0),
            "CoronerName": Comparator(lambda g, p: fuzz.token_sort_ratio(g, p) / 100.0, 0.90),
            "Area": Comparator(lambda g, p: fuzz.token_sort_ratio(g, p) / 100.0, 0.90),
            "Receiver": Comparator(lambda g, p: fuzz.token_set_ratio(g, p) / 100.0, 0.85),
            "InvestigationAndInquest": Comparator(self._compare_long, 0.80),
            "CircumstancesOfDeath": Comparator(self._compare_long, 0.80),
            "MattersOfConcern": Comparator(self._compare_long, 0.80),
        }

    def _compare_long(self, gt: str, pred: str) -> float:
        gt_n, pr_n = _normalise_simple(gt), _normalise_simple(pred)
        fast = _jaccard_5gram(gt_n, pr_n)
        if fast >= 0.60:
            return fast
        if self._embedder is None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        vec_gt, vec_pr = self._embedder.encode([gt_n, pr_n])
        return _cosine(vec_gt, vec_pr)

###############################################################################
# Evaluation core
###############################################################################

@st.cache_data(show_spinner=False)
def evaluate_extractions(scraped: pd.DataFrame, labelled: pd.DataFrame, thresh: float = 0.85):
    # Keep only columns we care about and present in both frames
    cols_present = [c for c in EVAL_COLS if c in scraped.columns and c in labelled.columns]
    missing_cols = [c for c in EVAL_COLS if c not in cols_present]

    scraped = scraped[cols_present].copy()
    labelled = labelled[cols_present].copy()

    # align index length
    if not scraped.index.equals(labelled.index):
        scraped = scraped.reset_index(drop=True)
        labelled = labelled.reset_index(drop=True)
        if len(scraped) != len(labelled):
            raise ValueError("DataFrames differ in length even after aligning index.")

    registry = _Comparators()

    sim = pd.DataFrame(index=scraped.index, columns=cols_present, dtype=float)
    ok  = pd.DataFrame(index=scraped.index, columns=cols_present, dtype=bool)

    for col in cols_present:
        comp = registry.table[col]
        s_vals = scraped[col].tolist()
        l_vals = labelled[col].tolist()
        sim[col] = [comp(g, p) for g, p in zip(l_vals, s_vals)]
        ok[col] = sim[col] >= thresh

    summary = pd.DataFrame({
        "cell_accuracy": ok.to_numpy().mean(),
        "row_accuracy": ok.all(axis=1).mean(),
        **{f"col_{c}_accuracy": ok[c].mean() for c in cols_present},
    }, index=[0])

    detail = pd.concat([
        sim.add_suffix("|similarity"),
        ok.add_suffix("|success"),
    ], axis=1)

    return {
        "detail": detail,
        "summary": summary,
        "missing_cols": missing_cols,
    }

###############################################################################
# Streamlit UI
###############################################################################

st.set_page_config(page_title="LLM Extraction Evaluation", page_icon="📊", layout="wide")

st.sidebar.header("📂 Upload your data")

scraped_file = st.sidebar.file_uploader("Scraped reports (CSV/parquet)", type=["csv", "parquet", "pq"])
labelled_file = st.sidebar.file_uploader("Ground‑truth reports (CSV/parquet)", type=["csv", "parquet", "pq"])

t_thresh = st.sidebar.slider("Success threshold", 0.0, 1.0, 0.85, 0.01)

with st.sidebar.expander("ℹ️ Eval columns"):
    st.markdown("**Columns evaluated:** " + ", ".join(EVAL_COLS))

st.title("LLM Extraction Evaluation Dashboard")
#st.markdown("This dashboard visualises")

@st.cache_data(show_spinner=False)
def _load(upload) -> pd.DataFrame:
    if upload.name.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(upload)
    return pd.read_csv(upload)

if scraped_file and labelled_file:
    scraped_df = _load(scraped_file)
    labelled_df = _load(labelled_file)
    try:
        with st.spinner("Evaluating …"):
            report = evaluate_extractions(scraped_df, labelled_df, thresh=t_thresh)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    detail = report["detail"]
    summary = report["summary"]

    if report["missing_cols"]:
        st.warning("Missing columns in uploaded files: " + ", ".join(report["missing_cols"]))

    col1, col2, _ = st.columns([1,1,2])
    col1.metric("Cell accuracy", f"{summary.loc[0,'cell_accuracy']:.2%}")
    col2.metric("Row accuracy",  f"{summary.loc[0,'row_accuracy']:.2%}")

    st.subheader("Per‑column accuracy")
    st.dataframe(summary.T.iloc[2:], use_container_width=True, height=200)

    st.subheader("Per‑cell detail")
    st.dataframe(detail, use_container_width=True, height=600)

    st.download_button("⬇️ Detail CSV", detail.to_csv().encode(), "detail.csv", "text/csv")
    st.download_button("⬇️ Summary CSV", summary.to_csv(index=False).encode(), "summary.csv", "text/csv")
else:
    st.info("Upload scraped and labelled files to start.")
