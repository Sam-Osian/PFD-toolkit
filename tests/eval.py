import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer  # type: ignore

###############################################################################
# Constants (it's constant!)
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
GT_COLS = ["URL", *EVAL_COLS]  # ground‑truth CSV header order


###############################################################################
# Small helpers
###############################################################################

def _ensure_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    if pd.isna(val):
        return ""
    return str(val)


###############################################################################
# ---------------------------  MODE 1 : Annotating   ------------------------ #
###############################################################################

def init_gt_state(base_df: pd.DataFrame):
    """Create the editable ground‑truth DataFrame on first load."""
    if "gt_df" not in st.session_state:
        # copy any populated columns from the uploaded CSV
        gt = pd.DataFrame(index=base_df.index)
        for col in GT_COLS:
            if col in base_df.columns:
                gt[col] = base_df[col].fillna("").astype(str)
            else:
                gt[col] = ""                       # blank if absent
        st.session_state.gt_df = gt
        st.session_state.gt_idx = _first_incomplete_row(gt)



def _first_incomplete_row(df: pd.DataFrame) -> int:
    mask_complete = df[EVAL_COLS].apply(lambda r: r.ne("").all(), axis=1)
    incomplete = mask_complete.idxmin() if not mask_complete.all() else len(df) - 1
    return int(incomplete)



def gt_page(base_df: pd.DataFrame):
    """Annotating UI"""
    init_gt_state(base_df)
    gt_df: pd.DataFrame = st.session_state.gt_df

    # --- Sidebar ----
    st.sidebar.subheader("Ground‑truth progress CSV")
    up_file = st.sidebar.file_uploader("Resume progress", type="csv")
    if up_file:
        up_df = pd.read_csv(up_file)
        if set(GT_COLS).issubset(up_df.columns):
            up_df = up_df.reindex(columns=GT_COLS)
            st.session_state.gt_df = up_df
            st.session_state.gt_idx = _first_incomplete_row(up_df)
            st.rerun()
        else:
            st.sidebar.error("CSV missing required columns")

    # --- Main area ----
    idx = st.session_state.gt_idx
    total = len(gt_df)
    st.markdown(f"### Report {idx+1} / {total}")

    url_val = gt_df.at[idx, "URL"] if "URL" in gt_df.columns else ""
    if url_val:
        st.markdown(f"**URL:** [{url_val}]({url_val})")
    else:
        st.markdown("**URL:** _missing_")

    # dynamic text boxes
    cols_filled = []
    for col in EVAL_COLS:
        if col in ("Date", "CoronerName", "Area"):
            # single‑line input
            val = st.text_input(
                col,
                value=_ensure_text(gt_df.at[idx, col]),
                key=f"inp_{col}",
            )
        elif col == "Receiver":
            # small multi‑line (≈ 5 lines)
            val = st.text_area(
                col,
                value=_ensure_text(gt_df.at[idx, col]),
                height=100,
                key=f"ta_{col}",
            )
        else:
            # full‑size paragraph box
            val = st.text_area(
                col,
                value=_ensure_text(gt_df.at[idx, col]),
                height=200,
                key=f"ta_{col}",
            )

        gt_df.at[idx, col] = val
        cols_filled.append(bool(val.strip()))


    # completion progress
    complete_rows = gt_df[EVAL_COLS].apply(lambda r: r.ne("").all(), axis=1).sum()
    st.progress(complete_rows / total, text=f"Completed {complete_rows} / {total}")

    # nav buttons
    nav1, nav2, save_col = st.columns([1,1,2])
    if nav1.button("⬅ Previous", disabled=(idx == 0)):
        st.session_state.gt_idx = idx - 1
        st.rerun()
    if nav2.button("Next ➡", disabled=(idx == total - 1)):
        st.session_state.gt_idx = idx + 1
        st.rerun()

    # auto‑jump to first incomplete if we hit end
    if idx == total - 1 and all(cols_filled):
        st.toast("All rows complete! 🎉")

    # save progress
    csv_bytes = gt_df.to_csv(index=False).encode()
    save_col.download_button(
        "💾 Save progress CSV",
        data=csv_bytes,
        file_name="ground_truth_progress.csv",
        mime="text/csv",
    )



###############################################################################
# ---------------------------  MODE 2 : EVALUATION  ------------------------- #
###############################################################################

@dataclass
class Comparator:
    func: Callable[[str, str], float]
    default_thresh: float
    def __call__(self, gt, pred):
        return self.func(_ensure_text(gt), _ensure_text(pred))

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
    cols_present = [c for c in EVAL_COLS if c in scraped.columns and c in labelled.columns]
    scraped_f = scraped[cols_present].copy()
    labelled_f = labelled[cols_present].copy()

    # align index
    if not scraped_f.index.equals(labelled_f.index):
        scraped_f = scraped_f.reset_index(drop=True)
        labelled_f = labelled_f.reset_index(drop=True)

    registry = _Comparators()
    sim = pd.DataFrame(index=scraped_f.index, columns=cols_present, dtype=float)
    ok  = pd.DataFrame(index=scraped_f.index, columns=cols_present, dtype=float)

    skipped = 0
    for col in cols_present:
        comp = registry.table[col]
        sims, oks = [], []
        for gt, pred in zip(labelled_f[col], scraped_f[col]):
            if not _ensure_text(gt).strip() or not _ensure_text(pred).strip():
                sims.append(np.nan)
                oks.append(np.nan)
                skipped += 1
            else:
                s = comp(gt, pred)
                sims.append(s)
                oks.append(float(s >= thresh))
        sim[col] = sims
        ok[col]  = oks

    # summary – NaN are ignored by .mean()
    summary = pd.DataFrame({
        "cell_accuracy": ok.mean().mean(),
        "row_accuracy": ok.apply(lambda r: r.dropna().all() if r.notna().any() else np.nan, axis=1).mean(),
        **{f"col_{c}_accuracy": ok[c].mean() for c in cols_present},
    }, index=[0])

    detail = pd.concat(
        [sim.add_suffix("|similarity"), ok.add_suffix("|success")],
        axis=1
    )

    return {
        "detail": detail,
        "summary": summary,
        "skipped": skipped,        
    }


###############################################################################
# ---------------------------  STREAMLIT APP  ------------------------------- #
###############################################################################

st.set_page_config(page_title="LLM Extraction Tool", page_icon="📊", layout="wide")

mode = st.sidebar.radio("Mode", ["Manual extraction", "Evaluation"], index=1)

# shared upload for base/labelled file
base_file = st.sidebar.file_uploader(
    "Human labelled reports (CSV)", type=["csv"], key="lbl"
)

@st.cache_data(show_spinner=False)
def _load(upload) -> pd.DataFrame:
    return pd.read_csv(upload)

if base_file is None:
    st.info("First, upload the human annotations CSV. This can be finished or unfinished.")
    st.stop()

base_df = _load(base_file)

# ------------------------------------------------------------
# warn if the newly‑uploaded human file differs from the one in memory
# ------------------------------------------------------------
new_sig = hash(
    pd.util.hash_pandas_object(
        base_df.reindex(columns=GT_COLS, fill_value="").fillna("").astype(str)
    ).sum()
)

sig_in_state = st.session_state.get("gt_signature")

if sig_in_state is not None and sig_in_state != new_sig:
    st.sidebar.warning(
        "A different human‑labelled file has been uploaded. "
        "Continuing will **overwrite any unsaved progress** in the current session."
    )
    if st.sidebar.button("Overwrite my in‑memory edits"):
        st.session_state.pop("gt_df", None)
        st.session_state.pop("gt_idx", None)
        st.session_state.gt_signature = new_sig
    else:
        st.stop() # abort until the user confirms!
else:
    # first load or same file as before
    st.session_state["gt_signature"] = new_sig



if mode == "Manual extraction":
    gt_page(base_df)
else:
    # require scraped file in evaluation mode
    scraped_file = st.sidebar.file_uploader("LLM Scraped reports (CSV)", type=["csv"], key="scr")
    t_thresh = st.sidebar.slider("Success threshold", 0.0, 1.0, 0.85, 0.01)

    if scraped_file is None:
        st.info("Upload the scraped extraction file to run evaluation.")
        st.stop()

    scraped_df_full = _load(scraped_file)

    # --- evaluation UI (condensed: re‑use previous dashboard) ---
    report = evaluate_extractions(scraped_df_full, base_df, thresh=t_thresh)
    detail, summary = report["detail"], report["summary"]
    skipped = report["skipped"]

    if skipped:
        st.warning(f"{int(skipped)} cell(s) ignored because one or both values were blank.")


    # tabbed view
    tabs = st.tabs(["📊 Dashboard", "🔍 Cell Inspector"])

    with tabs[0]:
        col1, col2, _ = st.columns([1, 1, 2])
        col1.metric("Cell accuracy", f"{summary.loc[0,'cell_accuracy']:.2%}")
        col2.metric("Row accuracy", f"{summary.loc[0,'row_accuracy']:.2%}")

        st.subheader("Per‑column accuracy")
        st.dataframe(summary.T.iloc[2:], use_container_width=True, height=200)

        st.subheader("Per‑cell detail (similarity | success)")
        st.dataframe(detail, use_container_width=True, height=600)

        st.download_button("⬇️ Detail CSV", detail.to_csv().encode(), "detail.csv", "text/csv")
        st.download_button("⬇️ Summary CSV", summary.to_csv(index=False).encode(), "summary.csv", "text/csv")

    with tabs[1]:
        st.markdown("### Inspect a specific cell")
        max_row = len(base_df) - 1
        row_idx = st.number_input("Row index", 0, max_row, 0, 1, format="%d")
        col_choice = st.selectbox("Column", EVAL_COLS, 0)

        if col_choice in base_df.columns and col_choice in scraped_df_full.columns:
            gt_val = base_df.at[row_idx, col_choice]
            scr_val = scraped_df_full.at[row_idx, col_choice]
            sim_val = detail.at[row_idx, f"{col_choice}|similarity"]
            passed = detail.at[row_idx, f"{col_choice}|success"]

            url = None
            if "URL" in base_df.columns:
                url = base_df.at[row_idx, "URL"]
            elif "URL" in scraped_df_full.columns:
                url = scraped_df_full.at[row_idx, "URL"]

            st.write(f"**Similarity:** {sim_val:.3f}   |   **Success:** {'✅' if passed else '❌'}")
            if url and isinstance(url, str) and url.strip():
                st.markdown(f"**Report URL:** [{url}]({url})")

            col_a, col_b = st.columns(2)
            with col_a:
                st.text_area("Human extracted", _ensure_text(gt_val), height=200, disabled=True)
            with col_b:
                st.text_area("LLM scraped", _ensure_text(scr_val), height=200, disabled=True)
        else:
            st.error("Selected column not present in both datasets.")
