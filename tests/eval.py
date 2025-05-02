import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer  # type: ignore

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
# Evaluation core – returns filtered frames for cell inspector
###############################################################################

@st.cache_data(show_spinner=False)
def evaluate_extractions(scraped: pd.DataFrame, labelled: pd.DataFrame, thresh: float = 0.85):
    cols_present = [c for c in EVAL_COLS if c in scraped.columns and c in labelled.columns]
    missing_cols = [c for c in EVAL_COLS if c not in cols_present]

    scraped_f = scraped[cols_present].copy()
    labelled_f = labelled[cols_present].copy()

    if not scraped_f.index.equals(labelled_f.index):
        scraped_f = scraped_f.reset_index(drop=True)
        labelled_f = labelled_f.reset_index(drop=True)
        if len(scraped_f) != len(labelled_f):
            raise ValueError("DataFrames differ in length after index alignment.")

    registry = _Comparators()
    sim = pd.DataFrame(index=scraped_f.index, columns=cols_present, dtype=float)
    ok  = pd.DataFrame(index=scraped_f.index, columns=cols_present, dtype=bool)

    for col in cols_present:
        comp = registry.table[col]
        sim[col] = [comp(g, p) for g, p in zip(labelled_f[col], scraped_f[col])]
        ok[col]  = sim[col] >= thresh

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
        "scraped_f": scraped_f,
        "labelled_f": labelled_f,
    }

###############################################################################
# Streamlit UI with Cell Inspector showing URL
###############################################################################

st.set_page_config(page_title="LLM Extraction Evaluation", page_icon="📊", layout="wide")

st.sidebar.header("📂 Upload your data")

scraped_file = st.sidebar.file_uploader("Scraped reports (CSV/parquet)", type=["csv", "parquet", "pq"])
labelled_file = st.sidebar.file_uploader("Ground‑truth reports (CSV/parquet)", type=["csv", "parquet", "pq"])

t_thresh = st.sidebar.slider("Success threshold", 0.0, 1.0, 0.85, 0.01)

with st.sidebar.expander("ℹ️ Evaluated columns"):
    st.markdown(", ".join(EVAL_COLS))

@st.cache_data(show_spinner=False)
def _load(upload) -> pd.DataFrame:
    if upload.name.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(upload)
    return pd.read_csv(upload)

st.title("📊 LLM Extraction Evaluation Dashboard")

if scraped_file and labelled_file:
    scraped_df_full = _load(scraped_file)
    labelled_df_full = _load(labelled_file)

    try:
        with st.spinner("Evaluating …"):
            report = evaluate_extractions(scraped_df_full, labelled_df_full, thresh=t_thresh)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    detail   = report["detail"]
    summary  = report["summary"]
    scraped_f = report["scraped_f"]
    labelled_f = report["labelled_f"]

    tabs = st.tabs(["📈 Dashboard", "🔍 Cell Inspector"])

    # ---------------- Dashboard tab ----------------
    with tabs[0]:
        if report["missing_cols"]:
            st.warning("Missing columns: " + ", ".join(report["missing_cols"]))

        col1, col2, _ = st.columns([1,1,2])
        col1.metric("Cell accuracy", f"{summary.loc[0,'cell_accuracy']:.2%}")
        col2.metric("Row accuracy",  f"{summary.loc[0,'row_accuracy']:.2%}")

        st.subheader("Per‑column accuracy")
        st.dataframe(summary.T.iloc[2:], use_container_width=True, height=200)

        st.subheader("Per‑cell detail (similarity | success)")
        st.dataframe(detail, use_container_width=True, height=600)

        st.download_button("⬇️ Detail CSV", detail.to_csv().encode(), "detail.csv", "text/csv")
        st.download_button("⬇️ Summary CSV", summary.to_csv(index=False).encode(), "summary.csv", "text/csv")

    # ---------------- Cell Inspector tab ----------------
    with tabs[1]:
        st.markdown("### Inspect a specific cell")
        max_row = len(scraped_f) - 1
        row_num = st.number_input("Row index", min_value=0, max_value=max_row, value=0, step=1, format="%d")
        col_choice = st.selectbox("Column", options=EVAL_COLS, index=0)

        if col_choice not in scraped_f.columns or col_choice not in labelled_f.columns:
            st.error("Selected column not present in both datasets.")
        else:
            gt_val = labelled_f.at[row_num, col_choice]
            sc_val = scraped_f.at[row_num, col_choice]
            sim_val = detail.at[row_num, f"{col_choice}|similarity"]
            success  = detail.at[row_num, f"{col_choice}|success"]

            # URL retrieval (prefer labelled, then scraped)
            url_val = None
            if "URL" in labelled_df_full.columns:
                url_val = labelled_df_full.at[row_num, "URL"]
            elif "URL" in scraped_df_full.columns:
                url_val = scraped_df_full.at[row_num, "URL"]

            st.write(f"**Similarity:** {sim_val:.3f}  |  **Success:** {'✅' if success else '❌'}")
            if url_val and isinstance(url_val, str) and url_val.strip():
                st.markdown(f"**Report URL:** [{url_val}]({url_val})")
            c1, c2 = st.columns(2)
            with c1:
                st.text_area("Ground truth", value=_ensure_text(gt_val), height=200)
            with c2:
                st.text_area("Scraped", value=_ensure_text(sc_val), height=200)
