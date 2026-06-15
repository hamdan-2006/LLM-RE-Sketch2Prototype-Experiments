from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any


import textwrap
from pathlib import Path
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from matplotlib.colors import TwoSlopeNorm
from scipy import stats

try:
    from scipy import stats
except Exception as exc:  # pragma: no cover
    stats = None
    SCIPY_IMPORT_ERROR = str(exc)
else:
    SCIPY_IMPORT_ERROR = None

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols, mixedlm
except Exception as exc:  # pragma: no cover
    sm = None
    ols = None
    mixedlm = None
    STATSMODELS_IMPORT_ERROR = str(exc)
else:
    STATSMODELS_IMPORT_ERROR = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ID_COLUMNS = ["Scenario ID", "Screen ID", "Domain", "Model", "Number of RUN", "Reviewer ID"]
PAI_METRICS = [
    "Visual Fidelity (VF)",
    "Functional Alignment (FA)",
    "Structural Code Quality (SCQ)",
    "Human-Perceived Usability (HPU)",
]
EEI_METRICS = [
    "Semantic Accuracy (SA)",
    "Requirement Coverage (RC)",
    "Novelty Precision (NP)",
    "Clarity & Readability (CR)",
    "Consistency (CO)",
]
ALL_METRICS = PAI_METRICS + EEI_METRICS
COMPOSITES = ["PAI", "EEI", "Overall"]
MODEL_CANONICAL_ORDER = ["claude-sonnet-4-5", "GPT-4.1", "GPT-4o", "Gemini 2.5 Pro"]

SHORT_METRIC_LABELS = {
    "Visual Fidelity (VF)": "VF",
    "Functional Alignment (FA)": "FA",
    "Structural Code Quality (SCQ)": "SCQ",
    "Human-Perceived Usability (HPU)": "HPU",
    "Semantic Accuracy (SA)": "SA",
    "Requirement Coverage (RC)": "RC",
    "Novelty Precision (NP)": "NP",
    "Clarity & Readability (CR)": "CR",
    "Consistency (CO)": "CO",
}

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def safe_name(name: str, max_len: int = 31) -> str:
    return re.sub(r"[\\/*?:\[\]]", "_", str(name))[:max_len]


def ensure_dirs(outdir: Path) -> dict[str, Path]:
    paths = {
        "root": outdir,
        "tables": outdir / "tables",
        "plots": outdir / "plots",
        "plots_pdf": outdir / "plots_pdf",
        "logs": outdir / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def model_sort_key(model: str) -> tuple[int, str]:
    try:
        return (MODEL_CANONICAL_ORDER.index(model), model)
    except ValueError:
        return (999, model)


def sorted_models(models: Iterable[str]) -> list[str]:
    return sorted([str(m) for m in models], key=model_sort_key)


def mean_ci(series: Iterable[float], confidence: float = 0.95) -> tuple[float, float, float, int]:
    x = pd.Series(series).dropna().astype(float).to_numpy()
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(np.mean(x))
    if n < 2 or stats is None:
        return mean, np.nan, np.nan, n
    sem = stats.sem(x)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, float(mean - h), float(mean + h), n


def bootstrap_ci(values: Iterable[float], n_boot: int = 5000, seed: int = 42, confidence: float = 0.95) -> tuple[float, float, float]:
    x = pd.Series(values).dropna().astype(float).to_numpy()
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    lo = np.percentile(means, (1 - confidence) / 2 * 100)
    hi = np.percentile(means, (1 + confidence) / 2 * 100)
    return float(x.mean()), float(lo), float(hi)


def cohen_d(a: Iterable[float], b: Iterable[float]) -> float:
    a = pd.Series(a).dropna().astype(float).to_numpy()
    b = pd.Series(b).dropna().astype(float).to_numpy()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return np.nan if pooled == 0 else float((np.mean(a) - np.mean(b)) / pooled)


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    a = pd.Series(a).dropna().astype(float).to_numpy()
    b = pd.Series(b).dropna().astype(float).to_numpy()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = 0
    lt = 0
    # Efficient enough for this dataset size.
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    return float((gt - lt) / (len(a) * len(b)))


def effect_label_delta(delta: float) -> str:
    if pd.isna(delta):
        return "NA"
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx]
        adj = min(1.0, (m - rank) * raw)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def cronbach_alpha(data: pd.DataFrame) -> float:
    x = data.dropna(axis=0, how="any").to_numpy(dtype=float)
    if x.shape[0] < 2 or x.shape[1] < 2:
        return np.nan
    item_vars = x.var(axis=0, ddof=1)
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    k = x.shape[1]
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


def icc_2_1_and_2_k(ratings: pd.DataFrame) -> tuple[float, float]:
    """ICC(2,1) and ICC(2,k), two-way random-effects absolute agreement.
    Rows = targets/artifacts, columns = reviewers. Rows with missing values dropped.
    """
    r = ratings.dropna(axis=0, how="any")
    n, k = r.shape
    if n < 2 or k < 2:
        return np.nan, np.nan
    y = r.to_numpy(dtype=float)
    grand = y.mean()
    row_means = y.mean(axis=1, keepdims=True)
    col_means = y.mean(axis=0, keepdims=True)

    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((y - grand) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom_21 = ms_rows + (k - 1) * ms_error + (k * (ms_cols - ms_error) / n)
    icc21 = (ms_rows - ms_error) / denom_21 if denom_21 != 0 else np.nan
    denom_2k = ms_rows + ((ms_cols - ms_error) / n)
    icc2k = (ms_rows - ms_error) / denom_2k if denom_2k != 0 else np.nan
    return float(icc21), float(icc2k)


def weighted_kappa_quadratic(a: Iterable[Any], b: Iterable[Any]) -> float:
    """Quadratic weighted Cohen's kappa for ordinal scores."""
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    common = pd.concat([a, b], axis=1).dropna()
    if common.shape[0] < 2:
        return np.nan
    x = common.iloc[:, 0].astype(float).round().astype(int)
    y = common.iloc[:, 1].astype(float).round().astype(int)
    categories = list(range(1, 6))
    ncat = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    O = np.zeros((ncat, ncat), dtype=float)
    for xi, yi in zip(x, y):
        if xi in cat_to_idx and yi in cat_to_idx:
            O[cat_to_idx[xi], cat_to_idx[yi]] += 1
    if O.sum() == 0:
        return np.nan
    hist_x = O.sum(axis=1)
    hist_y = O.sum(axis=0)
    E = np.outer(hist_x, hist_y) / O.sum()
    W = np.zeros((ncat, ncat), dtype=float)
    for i in range(ncat):
        for j in range(ncat):
            W[i, j] = ((i - j) ** 2) / ((ncat - 1) ** 2)
    denom = (W * E).sum()
    if denom == 0:
        return np.nan
    return float(1 - (W * O).sum() / denom)


def kendalls_w(ratings: pd.DataFrame) -> float:
    """Kendall's W for complete ratings matrix: rows targets, columns raters."""
    r = ratings.dropna(axis=0, how="any")
    n, m = r.shape
    if n < 2 or m < 2:
        return np.nan
    ranks = r.rank(axis=0, method="average")
    rank_sums = ranks.sum(axis=1)
    mean_rank_sum = rank_sums.mean()
    S = ((rank_sums - mean_rank_sum) ** 2).sum()
    W = 12 * S / (m**2 * (n**3 - n))
    return float(W)


def save_fig(fig: plt.Figure, basename: str, paths: dict[str, Path]) -> None:
    fig.tight_layout()
    fig.savefig(paths["plots"] / f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(paths["plots_pdf"] / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)

# -----------------------------------------------------------------------------
# Loading and cleaning
# -----------------------------------------------------------------------------

def detect_header_row(path: Path, sheet_name: str | int = 0) -> int:
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=30)
    required = {"Scenario ID", "Screen ID", "Domain", "Model", "Reviewer ID"}
    for i, row in preview.iterrows():
        values = {str(x).strip() for x in row.values if pd.notna(x)}
        if required.issubset(values):
            return int(i)
    return 0


def load_reviews(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    header = detect_header_row(path, sheet_name)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in ID_COLUMNS + ALL_METRICS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    for col in ALL_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Model", "Reviewer ID"] + ALL_METRICS).copy()

    # Canonical text cleanup.
    for col in ["Scenario ID", "Screen ID", "Domain", "Model", "Number of RUN", "Reviewer ID"]:
        df[col] = df[col].astype(str).str.strip()
    df["Number of RUN"] = df["Number of RUN"].str.replace("Run", "", case=False, regex=False).str.strip()

    df["PAI"] = df[PAI_METRICS].mean(axis=1)
    df["EEI"] = df[EEI_METRICS].mean(axis=1)
    df["Overall"] = df[ALL_METRICS].mean(axis=1)

    df["Artifact ID"] = (
        df["Scenario ID"].astype(str)
        + "_Screen"
        + df["Screen ID"].astype(str)
        + "_Run"
        + df["Number of RUN"].astype(str)
        + "_Model"
        + df["Model"].astype(str)
    )
    return df

# -----------------------------------------------------------------------------
# Table analyses
# -----------------------------------------------------------------------------

def model_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, part in df.groupby("Model", observed=True):
        row = {"Model": model, "N": len(part)}
        for score in COMPOSITES:
            row[f"{score}_Mean"] = part[score].mean()
            row[f"{score}_SD"] = part[score].std()
            row[f"{score}_Median"] = part[score].median()
            row[f"{score}_Min"] = part[score].min()
            row[f"{score}_Max"] = part[score].max()
            m, lo, hi, _ = mean_ci(part[score])
            row[f"{score}_CI95_Low"] = lo
            row[f"{score}_CI95_High"] = hi
        rows.append(row)
    out = pd.DataFrame(rows)
    out["Overall_Rank"] = out["Overall_Mean"].rank(ascending=False, method="dense").astype(int)
    return out.sort_values(["Overall_Rank", "Overall_Mean"], ascending=[True, False])


def composite_ci_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score in COMPOSITES:
        for model, part in df.groupby("Model", observed=True):
            mean, lo, hi, n = mean_ci(part[score])
            bmean, blo, bhi = bootstrap_ci(part[score])
            rows.append({
                "Composite": score,
                "Model": model,
                "N": n,
                "Mean": mean,
                "CI95_Low_t": lo,
                "CI95_High_t": hi,
                "CI95_Low_bootstrap": blo,
                "CI95_High_bootstrap": bhi,
                "Rank": np.nan,
            })
    out = pd.DataFrame(rows)
    out["Rank"] = out.groupby("Composite")["Mean"].rank(ascending=False, method="dense").astype(int)
    return out.sort_values(["Composite", "Rank"])


def metric_by_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ALL_METRICS + COMPOSITES:
        for model, part in df.groupby("Model", observed=True):
            mean, lo, hi, n = mean_ci(part[metric])
            rows.append({
                "Metric": metric,
                "Metric_short": SHORT_METRIC_LABELS.get(metric, metric),
                "Model": model,
                "N": n,
                "Mean": mean,
                "SD": part[metric].std(),
                "Median": part[metric].median(),
                "Min": part[metric].min(),
                "Max": part[metric].max(),
                "CI95_Low": lo,
                "CI95_High": hi,
            })
    return pd.DataFrame(rows).sort_values(["Metric", "Mean"], ascending=[True, False])


def grouped_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in df.groupby(group_cols + ["Model"], observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols + ["Model"], keys))
        row["N"] = len(part)
        for score in COMPOSITES:
            row[f"{score}_Mean"] = part[score].mean()
            row[f"{score}_SD"] = part[score].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols + ["Overall_Mean"], ascending=[True]*len(group_cols) + [False])


def reliability_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    alpha_rows = []
    alpha_rows.append({"Scale": "PAI", "Items": len(PAI_METRICS), "Cronbach_Alpha": cronbach_alpha(df[PAI_METRICS])})
    alpha_rows.append({"Scale": "EEI", "Items": len(EEI_METRICS), "Cronbach_Alpha": cronbach_alpha(df[EEI_METRICS])})
    alpha_rows.append({"Scale": "Overall", "Items": len(ALL_METRICS), "Cronbach_Alpha": cronbach_alpha(df[ALL_METRICS])})
    for model, part in df.groupby("Model", observed=True):
        alpha_rows.append({"Scale": f"PAI | {model}", "Items": len(PAI_METRICS), "Cronbach_Alpha": cronbach_alpha(part[PAI_METRICS])})
        alpha_rows.append({"Scale": f"EEI | {model}", "Items": len(EEI_METRICS), "Cronbach_Alpha": cronbach_alpha(part[EEI_METRICS])})
        alpha_rows.append({"Scale": f"Overall | {model}", "Items": len(ALL_METRICS), "Cronbach_Alpha": cronbach_alpha(part[ALL_METRICS])})
    tables["internal_consistency_cronbach_alpha"] = pd.DataFrame(alpha_rows)

    measure_rows = []
    for score in COMPOSITES + ALL_METRICS:
        pivot = df.pivot_table(index=["Artifact ID"], columns="Reviewer ID", values=score, aggfunc="mean")
        icc21, icc2k = icc_2_1_and_2_k(pivot)
        W = kendalls_w(pivot)
        measure_rows.append({
            "Measure": score,
            "ICC_2_1_single": icc21,
            "ICC_2_k_average": icc2k,
            "Kendalls_W": W,
            "Targets_complete": pivot.dropna().shape[0],
            "Reviewers": pivot.shape[1],
        })
    tables["reliability_by_measure"] = pd.DataFrame(measure_rows)

    kappa_rows = []
    reviewers = sorted(df["Reviewer ID"].unique())
    for score in COMPOSITES + ALL_METRICS:
        pivot = df.pivot_table(index=["Artifact ID"], columns="Reviewer ID", values=score, aggfunc="mean")
        for r1, r2 in itertools.combinations(reviewers, 2):
            if r1 in pivot.columns and r2 in pivot.columns:
                kappa = weighted_kappa_quadratic(pivot[r1], pivot[r2])
                kappa_rows.append({"Measure": score, "Reviewer_A": r1, "Reviewer_B": r2, "Weighted_Kappa_quadratic": kappa})
    kappa_df = pd.DataFrame(kappa_rows)
    tables["pairwise_weighted_kappa"] = kappa_df
    if not kappa_df.empty:
        tables["pairwise_weighted_kappa_summary"] = (
            kappa_df.groupby("Measure", observed=True)["Weighted_Kappa_quadratic"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .rename(columns={"mean": "Mean_Kappa", "std": "SD_Kappa", "min": "Min_Kappa", "max": "Max_Kappa"})
        )
    return tables


def _analysis_block_columns(df: pd.DataFrame) -> list[str]:
    """Return the repeated-measures block columns available in the dataset.

    The main inferential comparison treats Model as the within-block condition.
    A block is normally Reviewer × Scenario × Domain × Screen × Run. This honors
    the fact that the same reviewers score matched artifacts across models.
    """
    preferred = ["Reviewer ID", "Scenario ID", "Domain", "Screen ID", "Number of RUN"]
    return [c for c in preferred if c in df.columns]


def _model_pivot_for_score(df: pd.DataFrame, score: str, models: list[str]) -> pd.DataFrame:
    """Build a complete-case matrix: rows = matched blocks, columns = models."""
    block_cols = _analysis_block_columns(df)
    pivot = df.pivot_table(
        index=block_cols,
        columns="Model",
        values=score,
        aggfunc="mean",
    )
    present = [m for m in models if m in pivot.columns]
    return pivot[present].dropna(axis=0, how="any")


def kendalls_w_from_friedman(chi_square: float, n_blocks: int, k_conditions: int) -> float:
    """Kendall's W effect size for a Friedman test."""
    if pd.isna(chi_square) or n_blocks <= 0 or k_conditions <= 1:
        return np.nan
    return float(chi_square / (n_blocks * (k_conditions - 1)))


def wilcoxon_z_approximation(w_stat: float, n_nonzero: int) -> float:
    """Normal approximation Z for the Wilcoxon signed-rank statistic.

    SciPy returns W = min(W+, W-) for two-sided tests. This approximation is used
    only to compute an interpretable r effect size. Exact p-values from SciPy are
    still retained when available.
    """
    if pd.isna(w_stat) or n_nonzero <= 0:
        return np.nan
    mean_w = n_nonzero * (n_nonzero + 1) / 4.0
    sd_w = math.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0)
    if sd_w == 0:
        return np.nan
    return float((w_stat - mean_w) / sd_w)


def rank_biserial_from_pairs(a: Iterable[float], b: Iterable[float]) -> float:
    """Matched-pairs rank-biserial correlation for paired model comparisons."""
    x = pd.Series(a, dtype="float64")
    y = pd.Series(b, dtype="float64")
    d = (x - y).dropna()
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(d)) if stats is not None else pd.Series(np.abs(d)).rank().to_numpy()
    pos = float(np.sum(ranks[d.to_numpy() > 0]))
    neg = float(np.sum(ranks[d.to_numpy() < 0]))
    denom = pos + neg
    return np.nan if denom == 0 else float((pos - neg) / denom)


def paired_direction_probability(a: Iterable[float], b: Iterable[float]) -> tuple[float, float, float]:
    """Return paired dominance proportions: A>B, A<B, A=B."""
    d = (pd.Series(a, dtype="float64") - pd.Series(b, dtype="float64")).dropna()
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    return (
        float((d > 0).mean()),
        float((d < 0).mean()),
        float((d == 0).mean()),
    )


def inferential_tests(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Publication-oriented inferential analyses for the repeated-review design.

    Primary design logic:
    - Treatment/condition: Model.
    - Repeated-measures block: Reviewer × Scenario × Domain × Screen × Run.
    - Omnibus test: Friedman test on complete matched blocks.
    - Post-hoc tests: paired Wilcoxon signed-rank tests on the same matched blocks.
    - Multiplicity correction: Holm-Bonferroni within each score family.
    - Effect sizes: Kendall's W for Friedman; Wilcoxon r, matched rank-biserial
      correlation, and Cliff's delta for pairwise differences.

    The older independent-sample tests (Kruskal-Wallis, Mann-Whitney U, one-way
    ANOVA) are intentionally omitted because reviewer/artifact observations are
    not independent in this experiment.
    """
    tables: dict[str, pd.DataFrame] = {}
    friedman_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    dominance_rows: list[dict[str, Any]] = []
    mixed_rows: list[dict[str, Any]] = []
    mixed_fixed_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []

    models = sorted_models(df["Model"].dropna().unique())
    scores = COMPOSITES + ALL_METRICS
    block_cols = _analysis_block_columns(df)

    design_rows.append({
        "Design_Element": "Primary repeated-measures block",
        "Specification": " × ".join(block_cols),
        "Rationale": "Scores are matched by reviewer and artifact context; model is compared within each block.",
    })
    design_rows.append({
        "Design_Element": "Primary omnibus test",
        "Specification": "Friedman chi-square",
        "Rationale": "Non-parametric within-subjects comparison for ordinal/repeated scores.",
    })
    design_rows.append({
        "Design_Element": "Primary post-hoc test",
        "Specification": "Wilcoxon signed-rank with Holm correction",
        "Rationale": "Pairwise model comparisons retain the matched block structure.",
    })
    design_rows.append({
        "Design_Element": "Removed from primary analysis",
        "Specification": "Kruskal-Wallis, Mann-Whitney U, one-way ANOVA",
        "Rationale": "These tests assume independent observations and are not appropriate as primary tests for this repeated-review design.",
    })

    for score in scores:
        pivot = _model_pivot_for_score(df, score, models)
        k = pivot.shape[1]
        n_blocks = pivot.shape[0]

        if stats is not None and n_blocks > 1 and k > 2:
            fr = stats.friedmanchisquare(*[pivot[m].to_numpy(dtype=float) for m in pivot.columns])
            chi_square = float(fr.statistic)
            p_value = float(fr.pvalue)
            kendall_w = kendalls_w_from_friedman(chi_square, n_blocks, k)
        else:
            chi_square = np.nan
            p_value = np.nan
            kendall_w = np.nan

        friedman_rows.append({
            "Score": score,
            "Test": "Friedman repeated-measures chi-square",
            "Block": " × ".join(block_cols),
            "Models_Compared": k,
            "Complete_Blocks": n_blocks,
            "Chi_square": chi_square,
            "df": k - 1 if k else np.nan,
            "p_value": p_value,
            "Kendalls_W_effect_size": kendall_w,
            "Interpretation": effect_label_delta(kendall_w) if not pd.isna(kendall_w) else "NA",
        })

        temp_rows: list[dict[str, Any]] = []
        raw_p_values: list[float] = []
        for a, b in itertools.combinations(pivot.columns.tolist(), 2):
            paired = pivot[[a, b]].dropna()
            va = paired[a].astype(float)
            vb = paired[b].astype(float)
            diffs = va - vb
            nonzero = diffs[diffs != 0]

            if stats is not None and len(nonzero) > 0:
                try:
                    # method='auto' uses exact calculation when feasible and asymptotic otherwise.
                    wt = stats.wilcoxon(va, vb, alternative="two-sided", zero_method="wilcox", method="auto")
                except TypeError:  # older SciPy versions do not support method=
                    wt = stats.wilcoxon(va, vb, alternative="two-sided", zero_method="wilcox")
                w_stat = float(wt.statistic)
                p_raw = float(wt.pvalue)
                z_approx = wilcoxon_z_approximation(w_stat, len(nonzero))
                r_effect = float(abs(z_approx) / math.sqrt(len(nonzero))) if len(nonzero) else np.nan
            else:
                w_stat = np.nan
                p_raw = np.nan
                z_approx = np.nan
                r_effect = np.nan

            prop_a_gt_b, prop_a_lt_b, prop_equal = paired_direction_probability(va, vb)
            rb = rank_biserial_from_pairs(va, vb)
            cd = cliffs_delta(va, vb)

            row = {
                "Score": score,
                "Model_A": a,
                "Model_B": b,
                "N_matched_blocks": int(len(paired)),
                "N_nonzero_pairs": int(len(nonzero)),
                "Mean_A": float(va.mean()) if len(va) else np.nan,
                "Mean_B": float(vb.mean()) if len(vb) else np.nan,
                "Median_A": float(va.median()) if len(va) else np.nan,
                "Median_B": float(vb.median()) if len(vb) else np.nan,
                "Mean_Difference_A_minus_B": float(diffs.mean()) if len(diffs) else np.nan,
                "Median_Difference_A_minus_B": float(diffs.median()) if len(diffs) else np.nan,
                "Wilcoxon_W": w_stat,
                "Wilcoxon_Z_approx": z_approx,
                "p_value_raw": p_raw,
                "Wilcoxon_r_effect_size": r_effect,
                "Rank_Biserial_Correlation": rb,
                "Cliffs_delta": cd,
                "Cliffs_delta_interpretation": effect_label_delta(cd),
                "Proportion_A_greater_than_B": prop_a_gt_b,
                "Proportion_A_less_than_B": prop_a_lt_b,
                "Proportion_equal": prop_equal,
                "Preferred_Model_by_mean": a if va.mean() >= vb.mean() else b,
            }
            temp_rows.append(row)
            raw_p_values.append(p_raw if not pd.isna(p_raw) else 1.0)

            dominance_rows.append({
                "Score": score,
                "Model_A": a,
                "Model_B": b,
                "A_greater_than_B_blocks_pct": prop_a_gt_b * 100 if not pd.isna(prop_a_gt_b) else np.nan,
                "B_greater_than_A_blocks_pct": prop_a_lt_b * 100 if not pd.isna(prop_a_lt_b) else np.nan,
                "Tie_blocks_pct": prop_equal * 100 if not pd.isna(prop_equal) else np.nan,
                "Mean_Difference_A_minus_B": row["Mean_Difference_A_minus_B"],
                "Preferred_Model_by_mean": row["Preferred_Model_by_mean"],
            })

        adjusted = holm_adjust(raw_p_values) if raw_p_values else []
        for row, p_adj in zip(temp_rows, adjusted):
            row["p_value_holm"] = p_adj
            row["Significant_after_Holm_0.05"] = bool(p_adj < 0.05) if not pd.isna(p_adj) else False
            pairwise_rows.append(row)

        # Mixed-effects sensitivity model. This is secondary because Likert scores
        # are ordinal; it helps assess robustness while accounting for reviewer and
        # artifact clustering.
        if mixedlm is not None:
            try:
                needed = [score, "Model", "Domain", "Number of RUN", "Reviewer ID", "Scenario ID", "Screen ID"]
                d = df[[c for c in needed if c in df.columns]].dropna().copy()
                d = d.rename(columns={score: "ScoreValue", "Number of RUN": "Run", "Screen ID": "ScreenID"})
                d["ReviewerGroup"] = d["Reviewer ID"].astype(str)
                artifact_terms = [c for c in ["Scenario ID", "Domain", "ScreenID", "Run"] if c in d.columns]
                d["ArtifactCluster"] = d[artifact_terms].astype(str).agg("|".join, axis=1)

                formula_terms = ["C(Model)"]
                if "Domain" in d.columns:
                    formula_terms.append("C(Domain)")
                if "Run" in d.columns:
                    formula_terms.append("C(Run)")
                if "ScreenID" in d.columns:
                    formula_terms.append("C(ScreenID)")
                formula = "ScoreValue ~ " + " + ".join(formula_terms)

                md = mixedlm(
                    formula,
                    data=d,
                    groups=d["ReviewerGroup"],
                    vc_formula={"ArtifactCluster": "0 + C(ArtifactCluster)"},
                )
                try:
                    mdf = md.fit(method="lbfgs", reml=False, maxiter=500, disp=False)
                except Exception:
                    mdf = md.fit(method="powell", reml=False, maxiter=500, disp=False)

                vc_var = np.nan
                try:
                    if hasattr(mdf, "vcomp") and len(mdf.vcomp):
                        vc_var = float(mdf.vcomp[0])
                except Exception:
                    vc_var = np.nan

                reviewer_var = np.nan
                try:
                    if mdf.cov_re is not None and mdf.cov_re.size:
                        reviewer_var = float(mdf.cov_re.iloc[0, 0])
                except Exception:
                    reviewer_var = np.nan

                mixed_rows.append({
                    "Score": score,
                    "Model": "Score ~ Model + Domain + Run + Screen + (1|Reviewer) + (1|ArtifactCluster)",
                    "Converged": bool(getattr(mdf, "converged", False)),
                    "N": int(d.shape[0]),
                    "AIC": float(mdf.aic) if hasattr(mdf, "aic") else np.nan,
                    "BIC": float(mdf.bic) if hasattr(mdf, "bic") else np.nan,
                    "Reviewer_random_variance": reviewer_var,
                    "Artifact_random_variance": vc_var,
                    "Residual_variance": float(mdf.scale) if hasattr(mdf, "scale") else np.nan,
                    "Error": "",
                })

                conf = mdf.conf_int()
                for term, coef in mdf.params.items():
                    if str(term).startswith("C(Model)"):
                        mixed_fixed_rows.append({
                            "Score": score,
                            "Term": term,
                            "Coefficient": float(coef),
                            "Std_Error": float(mdf.bse.get(term, np.nan)) if hasattr(mdf, "bse") else np.nan,
                            "z_or_t": float(mdf.tvalues.get(term, np.nan)) if hasattr(mdf, "tvalues") else np.nan,
                            "p_value": float(mdf.pvalues.get(term, np.nan)) if hasattr(mdf, "pvalues") else np.nan,
                            "CI95_Low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
                            "CI95_High": float(conf.loc[term, 1]) if term in conf.index else np.nan,
                        })
            except Exception as exc:
                mixed_rows.append({
                    "Score": score,
                    "Model": "Score ~ Model + Domain + Run + Screen + (1|Reviewer) + (1|ArtifactCluster)",
                    "Converged": False,
                    "N": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "Reviewer_random_variance": np.nan,
                    "Artifact_random_variance": np.nan,
                    "Residual_variance": np.nan,
                    "Error": repr(exc),
                })

    tables["inferential_design_notes"] = pd.DataFrame(design_rows)
    tables["statistical_tests_friedman"] = pd.DataFrame(friedman_rows)
    tables["pairwise_model_comparisons"] = pd.DataFrame(pairwise_rows)
    tables["model_dominance_matrix"] = pd.DataFrame(dominance_rows)
    tables["mixed_effects_models"] = pd.DataFrame(mixed_rows)
    tables["mixed_effects_model_coefficients"] = pd.DataFrame(mixed_fixed_rows)
    return tables


def correlation_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    corr_vars = ALL_METRICS + COMPOSITES
    tables["spearman_metric_correlation_matrix"] = df[corr_vars].corr(method="spearman").round(3)
    rows = []
    for a, b in itertools.combinations(corr_vars, 2):
        if stats is not None:
            r = stats.spearmanr(df[a], df[b], nan_policy="omit")
            rows.append({"Variable_A": a, "Variable_B": b, "Spearman_rho": r.correlation, "p_value": r.pvalue})
    tables["spearman_metric_correlation_long"] = pd.DataFrame(rows).sort_values("Spearman_rho", ascending=False)

    # Model-specific EEI/PAI relationship.
    ep_rows = []
    for model, part in df.groupby("Model", observed=True):
        if stats is not None:
            r = stats.spearmanr(part["EEI"], part["PAI"], nan_policy="omit")
            ep_rows.append({"Model": model, "N": len(part), "Spearman_EEI_PAI": r.correlation, "p_value": r.pvalue})
    if stats is not None:
        r = stats.spearmanr(df["EEI"], df["PAI"], nan_policy="omit")
        ep_rows.append({"Model": "ALL", "N": len(df), "Spearman_EEI_PAI": r.correlation, "p_value": r.pvalue})
    tables["eei_pai_relationship"] = pd.DataFrame(ep_rows)
    return tables


def robustness_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    tables["domain_model_descriptives"] = grouped_summary(df, ["Domain"])
    tables["run_model_means"] = grouped_summary(df, ["Number of RUN"])
    tables["screen_model_means"] = grouped_summary(df, ["Domain", "Screen ID"])
    tables["reviewer_model_means"] = grouped_summary(df, ["Reviewer ID"])

    # Rank stability across grouping factors.
    rank_rows = []
    for factor in ["Domain", "Number of RUN", "Screen ID", "Reviewer ID"]:
        for score in COMPOSITES:
            for key, part in df.groupby(factor, observed=True):
                means = part.groupby("Model", observed=True)[score].mean().sort_values(ascending=False)
                ranking = " > ".join(means.index.astype(str).tolist())
                top = means.index[0] if len(means) else None
                rank_rows.append({"Factor": factor, "Level": key, "Score": score, "Top_Model": top, "Ranking": ranking})
    tables["rank_stability"] = pd.DataFrame(rank_rows)

    # Domain sensitivity: SD of model scores across domains.
    sens_rows = []
    for model, part in df.groupby("Model", observed=True):
        for score in COMPOSITES:
            domain_means = part.groupby("Domain", observed=True)[score].mean()
            sens_rows.append({
                "Model": model,
                "Score": score,
                "Domain_Mean_SD": domain_means.std(),
                "Domain_Mean_Range": domain_means.max() - domain_means.min(),
                "Min_Domain": domain_means.idxmin(),
                "Max_Domain": domain_means.idxmax(),
            })
    tables["domain_sensitivity"] = pd.DataFrame(sens_rows)
    return tables


def reviewer_diagnostics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    grand = df["Overall"].mean()
    tables["reviewer_summary"] = (
        df.groupby("Reviewer ID", observed=True)
        .agg(
            N=("Overall", "size"),
            Overall_Mean=("Overall", "mean"),
            Overall_SD=("Overall", "std"),
            PAI_Mean=("PAI", "mean"),
            EEI_Mean=("EEI", "mean"),
        )
        .assign(Strictness_vs_Grand_Mean=lambda x: x["Overall_Mean"] - grand)
        .reset_index()
        .sort_values("Overall_Mean", ascending=False)
    )
    # Reviewer by metric mean.
    rows = []
    for reviewer, part in df.groupby("Reviewer ID", observed=True):
        for metric in ALL_METRICS + COMPOSITES:
            rows.append({"Reviewer ID": reviewer, "Metric": metric, "Mean": part[metric].mean(), "SD": part[metric].std(), "N": len(part)})
    tables["reviewer_metric_means"] = pd.DataFrame(rows)

    # Reviewer notes simple transparent keyword counts.
    if "Reviewer Notes" in df.columns:
        notes = df[["Reviewer ID", "Model", "Domain", "Reviewer Notes"]].dropna(subset=["Reviewer Notes"]).copy()
        notes["Reviewer Notes"] = notes["Reviewer Notes"].astype(str)
        keywords = [
            "missing", "incomplete", "ambiguous", "clear", "consistent", "layout", "navigation",
            "workflow", "button", "form", "validation", "accessibility", "confirmation", "field",
            "visual", "functional", "prototype", "requirement", "usability", "code", "semantic",
        ]
        counts = []
        for kw in keywords:
            counts.append({"Keyword": kw, "Mentions": int(notes["Reviewer Notes"].str.contains(kw, case=False, regex=False).sum())})
        tables["reviewer_note_keywords"] = pd.DataFrame(counts).sort_values("Mentions", ascending=False)
        tables["reviewer_notes_extracted"] = notes
    return tables


def build_all_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables["model_composite_descriptives"] = model_summary(df)
    tables["model_composite_95CI"] = composite_ci_table(df)
    tables["model_metric_descriptives"] = metric_by_model(df)
    tables.update(reliability_tables(df))
    tables.update(inferential_tests(df))
    tables.update(correlation_tables(df))
    tables.update(robustness_tables(df))
    tables.update(reviewer_diagnostics(df))
    return tables

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Palette & global style
# ---------------------------------------------------------------------------

# 10-colour qualitative palette (ColorBrewer Set2 extended)
PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC",
]

GRID_COLOR   = "#E8E8E8"
SPINE_COLOR  = "#CCCCCC"
LABEL_COLOR  = "#2C2C2C"
ANNOT_SIZE   = 8.5       # bar / cell annotation font size
TITLE_PAD    = 18

mpl.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.titlesize":      13,
    "axes.labelsize":      11,
    "xtick.labelsize":     9.5,
    "ytick.labelsize":     9.5,
    "legend.fontsize":     9,
    "legend.frameon":      False,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.left":    True,
    "axes.spines.bottom":  True,
    "axes.edgecolor":      SPINE_COLOR,
    "axes.grid":           True,
    "grid.color":          GRID_COLOR,
    "grid.linewidth":      0.8,
    "axes.axisbelow":      True,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "xtick.direction":     "out",
    "ytick.direction":     "out",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _palette(n: int) -> list[str]:
    """Cycle through PALETTE for n models."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def wrap_labels(labels, width: int = 14) -> list[str]:
    """Wrap long labels and prefix a sequential number for disambiguation."""
    return [
        f"{i + 1}. " + "\n".join(textwrap.wrap(str(lbl), width))
        for i, lbl in enumerate(labels)
    ]


def wrap_labels_plain(labels, width: int = 16) -> list[str]:
    """Wrap without numbering (used for legend / y-axis where numbers clutter)."""
    return ["\n".join(textwrap.wrap(str(lbl), width)) for lbl in labels]


def _fig_number_stamp(ax, number: int) -> None:
    """Light watermark-style figure number in top-left corner."""
    ax.text(
        0.01, 0.99, f"Fig. {number}",
        transform=ax.transAxes,
        fontsize=8, color="#AAAAAA",
        va="top", ha="left",
        fontstyle="italic",
    )


def _heatmap_cell_color(value: float, vmin: float, vmax: float) -> str:
    """Return 'black' or 'white' depending on background luminance."""
    norm = (value - vmin) / max(vmax - vmin, 1e-9)
    return "white" if norm > 0.55 else "black"


def add_bar_labels(
    ax,
    orientation: str = "vertical",
    fmt: str = "{:.2f}",
    padding: float = 0.04,
    fontsize: float = ANNOT_SIZE,
) -> None:
    """
    Add numeric labels to bar charts.
    Vertical bars: label sits above the bar top (or above error cap).
    Horizontal bars: label sits to the right of the bar end.
    Skips NaN values silently.
    """
    for container in ax.containers:
        if container is None:
            continue
        for bar in container:
            if bar is None:
                continue
            if orientation == "vertical":
                h = bar.get_height()
                if pd.isna(h):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + padding,
                    fmt.format(h),
                    ha="center", va="bottom",
                    fontsize=fontsize, color=LABEL_COLOR,
                    clip_on=True,
                )
            else:
                w = bar.get_width()
                if pd.isna(w):
                    continue
                sign = 1 if w >= 0 else -1
                ax.text(
                    w + sign * abs(padding),
                    bar.get_y() + bar.get_height() / 2,
                    fmt.format(w),
                    ha="left" if w >= 0 else "right",
                    va="center",
                    fontsize=fontsize, color=LABEL_COLOR,
                    clip_on=True,
                )


def finalize_plot(
    fig, ax,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    figure_number: int | None = None,
) -> None:
    """Apply consistent finishing touches to any axes."""
    if title:
        ax.set_title(title, fontweight="bold", pad=TITLE_PAD, color=LABEL_COLOR)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10, color=LABEL_COLOR)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=10, color=LABEL_COLOR)

    ax.tick_params(colors=LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    if legend and ax.get_legend_handles_labels()[1]:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            handlelength=1.4,
        )

    if figure_number is not None:
        _fig_number_stamp(ax, figure_number)

    fig.tight_layout()


# ---------------------------------------------------------------------------
# 01. Composite score comparisons
# ---------------------------------------------------------------------------

def plot_model_composites(df: pd.DataFrame, paths: dict[str, Path]) -> None:
    """
    Grouped bar chart: one figure per composite score.
    Bars are coloured by model, error bars show 95 % CI,
    bar labels are nudged above the upper error cap.
    """
    summary = model_summary(df)
    models  = sorted_models(summary["Model"])
    colors  = _palette(len(models))

    for idx, score in enumerate(COMPOSITES, start=1):
        rows = []
        for m in models:
            part = df[df["Model"] == m]
            mean, lo, hi, _ = mean_ci(part[score])
            rows.append((m, mean, mean - lo, hi - mean))

        labels, means, lowerr, hierr = zip(*rows)
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(max(9, len(models) * 1.4), 5.8))

        bars = ax.bar(
            x, means,
            yerr=[lowerr, hierr],
            capsize=6,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            error_kw={"elinewidth": 1.4, "ecolor": "#666666", "capthick": 1.4},
            zorder=3,
        )

        ax.set_ylim(1, 5.55)
        ax.set_xticks(x)
        ax.set_xticklabels(wrap_labels(labels, 13), rotation=0, ha="center")

        # Label above the upper CI cap, not just the bar top
        for bar, hi_err, mean_val in zip(bars, hierr, means):
            top = mean_val + hi_err
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                top + 0.09,
                f"{mean_val:.2f}",
                ha="center", va="bottom",
                fontsize=ANNOT_SIZE, fontweight="bold",
                color=LABEL_COLOR,
            )

        # Reference line at grand mean
        grand = df[score].mean()
        ax.axhline(grand, linestyle="--", linewidth=1.1, color="#999999", zorder=2)
        ax.text(
            len(models) - 0.45, grand + 0.05,
            f"Grand mean = {grand:.2f}",
            fontsize=8, color="#777777", va="bottom",
        )

        finalize_plot(
            fig, ax,
            title=f"Mean {score} Performance by Model (95 % CI)",
            ylabel=f"Mean {score} Score",
            legend=False
        )
        save_fig(fig, f"01_{score.lower()}_model_comparison", paths)


# ---------------------------------------------------------------------------
# 02. Metric heatmaps
# ---------------------------------------------------------------------------

def plot_metric_heatmaps(df: pd.DataFrame, paths: dict[str, Path]) -> None:
    """
    Heatmap per metric group.
    Cell text uses white on dark cells, black on light cells for legibility.
    Column headers are numbered to avoid overlap.
    """
    heatmap_specs = [
        (EEI_METRICS, "EEI",        "Requirements Evaluation Dimensions"),
        (PAI_METRICS, "PAI",        "Prototype Evaluation Dimensions"),
        (ALL_METRICS, "All Metrics","All Evaluation Dimensions"),
    ]

    for fig_no, (metrics, name, title_suffix) in enumerate(heatmap_specs, start=3):
        heat = (
            df.groupby("Model", observed=True)[metrics]
            .mean()
            .loc[sorted_models(df["Model"].unique())]
        )

        n_cols = len(metrics)
        n_rows = len(heat.index)
        fig_w  = max(12, n_cols * 0.9)
        fig_h  = max(5.5, n_rows * 0.95)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(heat.values, aspect="auto", vmin=1, vmax=5, cmap="RdYlGn")

        cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
        cbar.set_label("Mean Score (1–5)", rotation=270, labelpad=18)
        cbar.ax.tick_params(labelsize=8)

        # Numbered column headers
        short_labels = [SHORT_METRIC_LABELS[m] for m in metrics]
        col_labels   = [f"{i+1}. {lbl}" for i, lbl in enumerate(short_labels)]

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            ["\n".join(textwrap.wrap(lbl, 14)) for lbl in col_labels],
            rotation=0, ha="center",
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(wrap_labels_plain(heat.index, 18))

        ax.set_title(
            f"Mean {name} Metric Scores by Model: {title_suffix}",
            fontweight="bold", pad=TITLE_PAD,
        )

        # Cell annotations with contrast-aware text colour
        for i in range(n_rows):
            for j in range(n_cols):
                val = heat.iloc[i, j]
                fg  = _heatmap_cell_color(val, 1, 5)
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=ANNOT_SIZE, fontweight="bold",
                    color=fg,
                )

        fig.tight_layout()
        save_fig(fig, f"02_{name.lower().replace(' ', '_')}_metric_heatmap", paths)


# ---------------------------------------------------------------------------
# 03–04. Radar charts
# ---------------------------------------------------------------------------

def radar_plot(
    df: pd.DataFrame,
    metrics: list[str],
    basename: str,
    title: str,
    figure_number: int,
    paths: dict[str, Path],
) -> None:
    """
    Spider / radar chart.
    • Gridlines are annotated (1–5) so the scale is immediately obvious.
    • Each model gets a distinct colour + marker.
    • Labels are numbered to match a compact legend key.
    """
    models = sorted_models(df["Model"].unique())
    colors = _palette(len(models))

    short_labels = [SHORT_METRIC_LABELS[m] for m in metrics]
    # Numbered spoke labels
    spoke_labels = [f"{i+1}. {lbl}" for i, lbl in enumerate(short_labels)]

    N      = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9.2, 9.2))
    ax  = fig.add_subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for idx, (model, color) in enumerate(zip(models, colors)):
        vals = df[df["Model"] == model][metrics].mean().to_list()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.2, marker="o", markersize=5,
                label=f"{idx+1}. {model}", color=color)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(lbl, 14)) for lbl in spoke_labels],
        fontsize=9,
    )
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7.5, color="#888888")
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)

    ax.set_title(
        f"Figure {title}",
        fontweight="bold", pad=28, color=LABEL_COLOR,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=min(3, len(models)),
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    save_fig(fig, basename, paths)


def plot_radar_charts(df: pd.DataFrame, paths: dict[str, Path]) -> None:
    radar_plot(df, EEI_METRICS, "03_eei_radar_chart",
               "Requirements Quality Dimensions (EEI)", 6, paths)
    radar_plot(df, PAI_METRICS, "04_pai_radar_chart",
               "Prototype Quality Dimensions (PAI)",    7, paths)


# ---------------------------------------------------------------------------
# 05. Box plots  (+ jitter strip)
# ---------------------------------------------------------------------------

def plot_box_violin(df: pd.DataFrame, paths: dict[str, Path]) -> None:
    """
    Boxplot per composite score.
    • Jittered raw observations overlaid for sample-size transparency.
    • Numbered x-axis labels.
    • Mean diamond marker retained.
    """
    models = sorted_models(df["Model"].unique())
    colors = _palette(len(models))
    rng    = np.random.default_rng(42)

    for fig_no, score in enumerate(COMPOSITES, start=8):
        data   = [df[df["Model"] == m][score].dropna().to_numpy() for m in models]
        x_tick = wrap_labels(models, 13)

        fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 6.0))

        bp = ax.boxplot(
            data,
            labels=x_tick,
            showmeans=True,
            patch_artist=True,
            widths=0.42,
            medianprops={"linewidth": 2.2, "color": "#333333"},
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#333333",
                "markersize": 6,
            },
            whiskerprops={"linewidth": 1.3},
            capprops={"linewidth": 1.3},
            flierprops={
                "marker": "o", "markersize": 3.5,
                "markerfacecolor": "#AAAAAA", "markeredgecolor": "#AAAAAA",
            },
            zorder=3,
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)

        # Jitter strip
        for i, (arr, color) in enumerate(zip(data, colors), start=1):
            jitter = rng.uniform(-0.18, 0.18, size=len(arr))
            ax.scatter(
                np.full(len(arr), i) + jitter, arr,
                s=14, alpha=0.45,
                color=color, edgecolors="none",
                zorder=2,
            )

        ax.set_ylim(0.8, 5.4)

        # Grand-mean reference
        grand = df[score].mean()
        ax.axhline(grand, linestyle="--", linewidth=1.1, color="#999999")
        ax.text(
            len(models) + 0.35, grand + 0.05,
            f"Grand\nmean\n{grand:.2f}",
            fontsize=7.5, color="#888888", va="bottom",
        )

        finalize_plot(
            fig, ax,
            title=f"Distribution of {score} Scores by Model",
            ylabel=f"{score} Score",
            legend=False
        )
        save_fig(fig, f"05_{score.lower()}_boxplot_by_model", paths)


# ---------------------------------------------------------------------------
# 06–10. Domain, run, screen, reviewer diagnostics
# ---------------------------------------------------------------------------

def plot_domain_run_screen_reviewer(df: pd.DataFrame, paths: dict[str, Path]) -> None:
    figure_number = 10
    models  = sorted_models(df["Model"].unique())
    n_mod   = len(models)
    colors  = _palette(n_mod)
    color_map = dict(zip(models, colors))

    # --- Domain grouped bars -------------------------------------------
    for score in COMPOSITES:
        pivot = (
            df.pivot_table(index="Domain", columns="Model", values=score, aggfunc="mean")
              .reindex(columns=[m for m in models if m in df["Model"].unique()])
        )

        n_domains = len(pivot)
        fig_w = max(13, n_domains * n_mod * 0.28 + 3)
        fig, ax = plt.subplots(figsize=(fig_w, 6.5))

        pivot.plot(
            kind="bar", ax=ax,
            color=[color_map[m] for m in pivot.columns],
            edgecolor="white", linewidth=0.5,
            width=0.72,
        )

        ax.set_ylim(1, 5.55)
        ax.set_xticklabels(wrap_labels(pivot.index, 16), rotation=0)
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))

        # Value labels on bars — only when bars are wide enough
        if n_domains * n_mod <= 28:
            add_bar_labels(ax, fmt="{:.1f}", padding=0.04, fontsize=7.5)

        finalize_plot(
            fig, ax,
            title=f"{score} Performance by Domain and Model",
            xlabel="Application Domain",
            ylabel=f"Mean {score}"
        )
        save_fig(fig, f"06_{score.lower()}_domain_model_performance", paths)
        figure_number += 1

    # --- Run-level trend -----------------------------------------------
    for score in COMPOSITES:
        pivot = (
            df.pivot_table(index="Number of RUN", columns="Model", values=score, aggfunc="mean")
              .reindex(columns=[m for m in models if m in df["Model"].unique()])
        )

        fig, ax = plt.subplots(figsize=(11.5, 5.8))
        for col in pivot.columns:
            ax.plot(
                pivot.index, pivot[col],
                marker="o", linewidth=2.2, markersize=7,
                label=col, color=color_map[col],
            )
            # Endpoint annotation — use axes-fraction x so it works for any index type
            last_y = pivot[col].iloc[-1]
            if pd.notna(last_y):
                ax.annotate(
                    f"{last_y:.2f}",
                    xy=(1, last_y),
                    xycoords=("axes fraction", "data"),
                    xytext=(4, 0),
                    textcoords="offset points",
                    fontsize=7.5, va="center",
                    color=color_map[col],
                    clip_on=False,
                )

        ax.set_ylim(1, 5.25)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        finalize_plot(
            fig, ax,
            title=f"Run-Level Stability of {score} Scores",
            xlabel="Experimental Run",
            ylabel=f"Mean {score}"
        )
        save_fig(fig, f"07_{score.lower()}_run_level_stability", paths)
        figure_number += 1

    # --- Screen-level trend --------------------------------------------
    for score in COMPOSITES:
        pivot = (
            df.pivot_table(index="Screen ID", columns="Model", values=score, aggfunc="mean")
              .reindex(columns=[m for m in models if m in df["Model"].unique()])
        )

        fig, ax = plt.subplots(figsize=(12, 5.8))
        for col in pivot.columns:
            ax.plot(
                pivot.index, pivot[col],
                marker="o", linewidth=2.0, markersize=5,
                label=col, color=color_map[col], alpha=0.88,
            )

        ax.set_ylim(1, 5.25)
        finalize_plot(
            fig, ax,
            title=f"Screen-Level Stability of {score} Scores",
            xlabel="Screen ID",
            ylabel=f"Mean {score}"
        )
        save_fig(fig, f"08_{score.lower()}_screen_level_stability", paths)
        figure_number += 1

    # --- Reviewer strictness (horizontal bar) --------------------------
    rev = reviewer_diagnostics(df)["reviewer_summary"].sort_values("Strictness_vs_Grand_Mean")

    vals    = rev["Strictness_vs_Grand_Mean"].to_numpy()
    bar_col = [PALETTE[0] if v >= 0 else PALETTE[2] for v in vals]

    fig, ax = plt.subplots(figsize=(10.5, max(5.5, len(rev) * 0.52)))
    ax.barh(
        rev["Reviewer ID"], vals,
        color=bar_col, edgecolor="white", linewidth=0.6,
    )
    ax.axvline(0, linewidth=1.4, color="#444444")
    add_bar_labels(ax, orientation="horizontal", fmt="{:+.2f}", padding=0.008)

    # Legend patches
    ax.legend(
        handles=[
            mpatches.Patch(color=PALETTE[0], label="Lenient (above grand mean)"),
            mpatches.Patch(color=PALETTE[2], label="Strict (below grand mean)"),
        ],
        loc="lower right", frameon=False,
    )

    finalize_plot(
        fig, ax,
        title=f"Reviewer Strictness / Leniency Diagnostic",
        xlabel="Deviation from Grand Mean",
        ylabel="Reviewer ID",
        legend=False
    )
    save_fig(fig, "09_reviewer_strictness", paths)
    figure_number += 1

    # --- Reviewer × model heatmap (diverging) --------------------------
    pivot = (
        df.pivot_table(index="Reviewer ID", columns="Model", values="Overall", aggfunc="mean")
          .reindex(columns=[m for m in models if m in df["Model"].unique()])
    )

    grand_mean = df["Overall"].mean()
    vmin_div   = max(1, grand_mean - 1.5)
    vmax_div   = min(5, grand_mean + 1.5)
    norm       = TwoSlopeNorm(vmin=vmin_div, vcenter=grand_mean, vmax=vmax_div)

    fig, ax = plt.subplots(figsize=(max(11, len(pivot.columns)), max(5.5, len(pivot) * 0.75)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", norm=norm)

    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("Mean Overall Score", rotation=270, labelpad=18)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(wrap_labels(pivot.columns, 13), rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title(
        f"Reviewer-Specific Model Mean Overall Scores "
        f"(diverging from grand mean {grand_mean:.2f})",
        fontweight="bold", pad=TITLE_PAD,
    )

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            fg  = _heatmap_cell_color(val, 1, 5)
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=ANNOT_SIZE, fontweight="bold", color=fg)

    fig.tight_layout()
    save_fig(fig, "10_reviewer_model_heatmap", paths)


# ---------------------------------------------------------------------------
# 11–12. Correlations and EEI–PAI scatter
# ---------------------------------------------------------------------------

def plot_correlations(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    paths: dict[str, Path],
) -> None:

    corr = tables["spearman_metric_correlation_matrix"]
    n    = len(corr)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.85), max(10, n * 0.85)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")

    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("Spearman's ρ", rotation=270, labelpad=18)

    col_labels = [SHORT_METRIC_LABELS.get(c, c) for c in corr.columns]
    numbered   = [f"{i+1}. {lbl}" for i, lbl in enumerate(col_labels)]

    ax.set_xticks(range(n))
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(lbl, 14)) for lbl in numbered],
        rotation=0, ha="center",
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        ["\n".join(textwrap.wrap(lbl, 16)) for lbl in numbered],
    )

    ax.set_title(
        "Spearman Correlation Matrix Among Evaluation Dimensions",
        fontweight="bold", pad=TITLE_PAD,
    )

    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            fg  = _heatmap_cell_color(abs(val), 0, 1)
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=max(6.5, 9 - n * 0.15),
                color=fg,
            )

    fig.tight_layout()
    save_fig(fig, "11_metric_correlation_heatmap", paths)

    # --- EEI vs PAI scatter with per-model regression -----------------
    models = sorted_models(df["Model"].unique())
    colors = _palette(len(models))

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    for model, color in zip(models, colors):
        part = df[df["Model"] == model].dropna(subset=["EEI", "PAI"])
        ax.scatter(
            part["EEI"], part["PAI"],
            alpha=0.55, s=48, label=model,
            color=color, edgecolors="white", linewidth=0.5,
            zorder=3,
        )
        # OLS regression line per model
        if len(part) >= 4:
            slope, intercept, r, *_ = stats.linregress(part["EEI"], part["PAI"])
            xs = np.linspace(part["EEI"].min(), part["EEI"].max(), 60)
            ax.plot(xs, intercept + slope * xs,
                    color=color, linewidth=1.4, linestyle="--", alpha=0.6)

    # Overall Pearson r annotation
    clean = df.dropna(subset=["EEI", "PAI"])
    r_all, p_all = stats.pearsonr(clean["EEI"], clean["PAI"])
    ax.text(
        0.04, 0.96,
        f"Overall Pearson r = {r_all:.2f}  (p {'< 0.001' if p_all < 0.001 else f'= {p_all:.3f}'})",
        transform=ax.transAxes,
        fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=SPINE_COLOR, alpha=0.85),
    )

    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)

    finalize_plot(
        fig, ax,
        title="Relationship Between Requirements Quality (EEI) and Prototype Quality (PAI)",
        xlabel="Requirements Evaluation Index (EEI)",
        ylabel="Prototype Assessment Index (PAI)"
    )
    save_fig(fig, "12_eei_pai_scatter", paths)


# ---------------------------------------------------------------------------
# 13–15. Reliability plots
# ---------------------------------------------------------------------------

def plot_reliability(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    paths: dict[str, Path],
) -> None:

    # --- Cronbach's Alpha bar chart ------------------------------------
    alpha      = tables["internal_consistency_cronbach_alpha"]
    alpha_main = alpha[alpha["Scale"].isin(["PAI", "EEI", "Overall"])]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(
        alpha_main["Scale"],
        alpha_main["Cronbach_Alpha"],
        color=_palette(len(alpha_main)),
        edgecolor="white", linewidth=0.8,
    )
    ax.set_ylim(0, 1.12)
    ax.axhline(0.70, linestyle="--", linewidth=1.3, color="#777777")
    ax.text(
        len(alpha_main) - 0.5, 0.715,
        "Acceptable threshold (α = 0.70)",
        fontsize=8.5, va="bottom", ha="right", color="#666666",
    )
    add_bar_labels(ax, fmt="{:.3f}", padding=0.025)

    finalize_plot(
        fig, ax,
        title="Internal Consistency of Evaluation Scales (Cronbach's α)",
        xlabel="Evaluation Scale",
        ylabel="Cronbach's Alpha",
        legend=False
    )
    save_fig(fig, "13_cronbach_alpha", paths)

    # --- ICC grouped bar chart ----------------------------------------
    rel      = tables["reliability_by_measure"]
    rel_main = rel[rel["Measure"].isin(COMPOSITES)]

    x     = np.arange(len(rel_main))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    b1 = ax.bar(x - width / 2, rel_main["ICC_2_1_single"],   width,
                label="ICC(2,1) — Single reviewer",
                color=PALETTE[0], edgecolor="white", linewidth=0.7)
    b2 = ax.bar(x + width / 2, rel_main["ICC_2_k_average"],  width,
                label="ICC(2,k) — Average reviewers",
                color=PALETTE[1], edgecolor="white", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(rel_main["Measure"])
    ax.set_ylim(0, .3)
    add_bar_labels(ax, fmt="{:.2f}", padding=0.025)

    finalize_plot(
        fig, ax,
        title="Inter-Rater Reliability by Composite Score (ICC)",
        xlabel="Composite Measure",
        ylabel="Intraclass Correlation Coefficient"
    )
    save_fig(fig, "14_icc_composite", paths)

    # --- Weighted kappa -----------------------------------------------
    kappa = tables.get("pairwise_weighted_kappa_summary", pd.DataFrame())
    if not kappa.empty:
        kappa_main = kappa[kappa["Measure"].isin(COMPOSITES)]

        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        ax.bar(
            kappa_main["Measure"],
            kappa_main["Mean_Kappa"],
            color=_palette(len(kappa_main)),
            edgecolor="white", linewidth=0.8,
        )
        ax.set_ylim(0, 1.12)
        ax.axhline(0.61, linestyle="--", linewidth=1.2, color="#999999")
        ax.text(
            len(kappa_main) - 0.5, 0.625,
            "Substantial agreement threshold (0.61)",
            fontsize=8, color="#888888", ha="right",
        )
        add_bar_labels(ax, fmt="{:.3f}", padding=0.025)

        finalize_plot(
            fig, ax,
            title="Average Pairwise Reviewer Agreement (Weighted κ)",
            xlabel="Composite Measure",
            ylabel="Mean Weighted Kappa",
            legend=False,
        )
        save_fig(fig, "15_weighted_kappa_summary", paths)


# ---------------------------------------------------------------------------
# 16. Effect sizes (Cliff's delta)
# ---------------------------------------------------------------------------

def plot_effect_sizes(tables: dict[str, pd.DataFrame], paths: dict[str, Path]) -> None:
    pair = tables.get("pairwise_model_comparisons", pd.DataFrame())
    if pair.empty:
        return

    # Cliff's delta effect-size band thresholds
    BANDS = [
        ( 0.474, "Large",    "#D73027"),
        ( 0.330, "Medium",   "#FC8D59"),
        ( 0.147, "Small",    "#FEE090"),
        (-0.147, "Negligible","#E0F3F8"),
        (-0.330, "Small",    "#91BFDB"),
        (-0.474, "Medium",   "#4575B4"),
        (-1.001, "Large",    "#2C4A87"),
    ]

    for fig_no, score in enumerate(COMPOSITES, start=23):
        d = pair[pair["Score"] == score].copy()
        d["Comparison"] = d["Model_A"] + "  vs  " + d["Model_B"]
        d = d.sort_values("Cliffs_delta")

        fig_h = max(5.8, len(d) * 0.46)
        fig, ax = plt.subplots(figsize=(13.5, fig_h))

        bar_colors = []
        for val in d["Cliffs_delta"]:
            for threshold, _, color in BANDS:
                if val >= threshold:
                    bar_colors.append(color)
                    break
            else:
                bar_colors.append(PALETTE[0])

        ax.barh(
            wrap_labels_plain(d["Comparison"], 34),
            d["Cliffs_delta"],
            color=bar_colors,
            edgecolor="white", linewidth=0.6,
        )
        ax.axvline(0, linewidth=1.4, color="#444444")

        # Effect-size region shading
        for lo, hi, label, alpha_v in [
            ( 0.147,  0.330, "Small",  0.07),
            ( 0.330,  0.474, "Medium", 0.07),
            ( 0.474,  1.000, "Large",  0.07),
            (-0.330, -0.147, "Small",  0.07),
            (-0.474, -0.330, "Medium", 0.07),
            (-1.000, -0.474, "Large",  0.07),
        ]:
            ax.axvspan(lo, hi, alpha=alpha_v, color="#888888")

        add_bar_labels(ax, orientation="horizontal", fmt="{:+.2f}", padding=0.012)

        finalize_plot(
            fig, ax,
            title=f"Pairwise Model Effect Sizes for {score} (Cliff's δ)",
            xlabel="Cliff's Delta  (negative = Model A < Model B)",
            ylabel="Model Comparison",
            legend=False,
        )
        save_fig(fig, f"16_{score.lower()}_pairwise_cliffs_delta", paths)


# ---------------------------------------------------------------------------
# Master plotting function
# ---------------------------------------------------------------------------

def create_all_plots(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    paths: dict[str, Path],
) -> None:
    plot_model_composites(df, paths)
    plot_metric_heatmaps(df, paths)
    plot_radar_charts(df, paths)
    plot_box_violin(df, paths)
    plot_domain_run_screen_reviewer(df, paths)
    plot_correlations(df, tables, paths)
    plot_reliability(df, tables, paths)
    plot_effect_sizes(tables, paths)

# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------

def export_tables(tables: dict[str, pd.DataFrame], paths: dict[str, Path]) -> None:
    for name, table in tables.items():
        table.to_csv(paths["tables"] / f"{name}.csv", index=not isinstance(table.index, pd.RangeIndex))


def export_excel(df: pd.DataFrame, tables: dict[str, pd.DataFrame], paths: dict[str, Path]) -> Path:
    outpath = paths["root"] / "advanced_reviewer_analysis.xlsx"
    with pd.ExcelWriter(outpath, engine="xlsxwriter") as writer:
        workbook = writer.book
        title_fmt = workbook.add_format({"bold": True, "font_size": 16})
        section_fmt = workbook.add_format({"bold": True, "font_size": 12, "bg_color": "#D9EAF7", "border": 1})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#EAF2F8", "border": 1})
        num_fmt = workbook.add_format({"num_format": "0.000"})

        dash = workbook.add_worksheet("Dashboard")
        writer.sheets["Dashboard"] = dash
        dash.write("A1", "Advanced LLM Reviewer Evaluation Analysis", title_fmt)
        summary_items = [
            ("Rows analyzed", len(df)),
            ("Models", df["Model"].nunique()),
            ("Reviewers", df["Reviewer ID"].nunique()),
            ("Domains", df["Domain"].nunique()),
            ("Screens", df["Screen ID"].nunique()),
            ("Runs", df["Number of RUN"].nunique()),
            ("PAI metrics", len(PAI_METRICS)),
            ("EEI metrics", len(EEI_METRICS)),
        ]
        dash.write("A3", "Dataset Summary", section_fmt)
        for i, (k, v) in enumerate(summary_items, start=4):
            dash.write(i, 0, k)
            dash.write(i, 1, v)
        dash.write("D3", "Interpretive Note", section_fmt)
        dash.write("D4", "Higher scores indicate stronger evaluated quality on a 1--5 scale. PAI and EEI are composite means of their respective dimensions.")
        if (paths["plots"] / "01_overall_model_comparison.png").exists():
            dash.insert_image("A14", str(paths["plots"] / "01_overall_model_comparison.png"), {"x_scale": 0.65, "y_scale": 0.65})
        if (paths["plots"] / "02_all_metrics_metric_heatmap.png").exists():
            dash.insert_image("H14", str(paths["plots"] / "02_all_metrics_metric_heatmap.png"), {"x_scale": 0.55, "y_scale": 0.55})
        dash.set_column("A:A", 25)
        dash.set_column("B:B", 14)
        dash.set_column("D:K", 20)

        for name, table in tables.items():
            sheet = safe_name(name)
            table.to_excel(writer, sheet_name=sheet, index=not isinstance(table.index, pd.RangeIndex))
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 0)
            nrows, ncols = table.shape
            ws.autofilter(0, 0, max(nrows, 1), max(ncols - 1, 0))
            for col_idx, col_name in enumerate(list(table.columns)):
                ws.write(0, col_idx, col_name, header_fmt)
                ws.set_column(col_idx, col_idx, min(max(len(str(col_name)) + 3, 12), 42), num_fmt if col_idx > 0 else None)

        df.to_excel(writer, sheet_name="clean_data", index=False)
        ws = writer.sheets["clean_data"]
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(df), len(df.columns) - 1)
        ws.set_column(0, len(df.columns) - 1, 18)

        methodology = pd.DataFrame({
            "Component": [
                "Composite scores", "Internal consistency", "Inter-rater reliability", "Agreement",
                "Main statistical tests", "Pairwise tests", "Effect sizes", "Mixed-effects models",
                "Correlation", "Robustness", "Reviewer diagnostics", "Qualitative notes",
            ],
            "Method": [
                "PAI, EEI, and Overall are arithmetic means of their scoring dimensions.",
                "Cronbach's alpha for PAI, EEI, all metrics, and model-specific scales.",
                "ICC(2,1) and ICC(2,k), two-way random-effects absolute agreement.",
                "Quadratic weighted Cohen's kappa and Kendall's W.",
                "Friedman repeated-measures tests using matched Reviewer × Scenario × Domain × Screen × Run blocks.",
                "Wilcoxon signed-rank tests with Holm correction for matched pairwise model comparisons.",
                "Kendall's W, Wilcoxon r, matched rank-biserial correlation, and Cliff's delta.",
                "Score ~ Model + Domain + Run + Screen + (1|Reviewer) + (1|ArtifactCluster); failures recorded when singular.",
                "Spearman correlations because Likert-type scores are ordinal/discrete.",
                "Domain, screen, run, reviewer, rank stability, and domain sensitivity analyses.",
                "Reviewer strictness/leniency and reviewer-by-model diagnostics.",
                "Transparent keyword counts from reviewer notes; manual thematic coding can be added.",
            ],
        })
        methodology.to_excel(writer, sheet_name="methodology_notes", index=False)
        writer.sheets["methodology_notes"].set_column(0, 1, 80)
    return outpath


def write_report(df: pd.DataFrame, tables: dict[str, pd.DataFrame], paths: dict[str, Path]) -> Path:
    report_path = paths["root"] / "research_report.md"
    model_sum = tables["model_composite_descriptives"].copy()
    alpha = tables["internal_consistency_cronbach_alpha"]
    rel = tables["reliability_by_measure"]
    ep = tables["eei_pai_relationship"]
    pair = tables["pairwise_model_comparisons"]

    top_model = model_sum.sort_values("Overall_Mean", ascending=False).iloc[0]["Model"]
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Advanced LLM Reviewer Evaluation Analysis Report\n\n")
        f.write("## Dataset\n\n")
        f.write(f"- Rows analyzed: {len(df)}\n")
        f.write(f"- Models: {df['Model'].nunique()}\n")
        f.write(f"- Reviewers: {df['Reviewer ID'].nunique()}\n")
        f.write(f"- Domains: {df['Domain'].nunique()}\n")
        f.write(f"- Screens: {df['Screen ID'].nunique()}\n")
        f.write(f"- Runs: {df['Number of RUN'].nunique()}\n\n")
        f.write("## Main finding\n\n")
        f.write(f"The highest overall ranked model is **{top_model}** based on the mean Overall score.\n\n")
        f.write("## Model summary\n\n")
        f.write(model_sum.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Internal consistency\n\n")
        f.write(alpha.head(3).to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Inter-rater reliability\n\n")
        f.write(rel[rel["Measure"].isin(COMPOSITES)].to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## EEI-PAI relationship\n\n")
        f.write(ep.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Significant pairwise comparisons for composite scores\n\n")
        pcomp = pair[(pair["Score"].isin(COMPOSITES)) & (pair["p_value_holm"] < 0.05)].copy()
        f.write(pcomp[["Score", "Model_A", "Model_B", "Mean_Difference_A_minus_B", "Cliffs_delta", "Cliffs_delta_interpretation", "p_value_holm"]].to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## Recommended manuscript figures\n\n")
        recommended = [
            "01_eei_model_comparison.pdf",
            "01_pai_model_comparison.pdf",
            "03_eei_radar_chart.pdf",
            "04_pai_radar_chart.pdf",
            "11_metric_correlation_heatmap.pdf",
            "12_eei_pai_scatter.pdf",
            "13_cronbach_alpha.pdf",
            "14_icc_composite.pdf",
            "09_reviewer_strictness.pdf",
        ]
        for item in recommended:
            f.write(f"- plots_pdf/{item}\n")
        f.write("\n## Notes\n\n")
        f.write("Mixed-effects model failures, if any, are recorded in tables/mixed_effects_models.csv and should be described transparently rather than ignored.\n")
    return report_path


def write_manifest(df: pd.DataFrame, tables: dict[str, pd.DataFrame], paths: dict[str, Path], args: argparse.Namespace) -> Path:
    manifest = {
        "input_file": str(args.input),
        "outdir": str(args.outdir),
        "rows_analyzed": int(len(df)),
        "models": sorted_models(df["Model"].unique()),
        "reviewers": sorted(df["Reviewer ID"].unique().tolist()),
        "domains": sorted(df["Domain"].unique().tolist()),
        "metrics": {"PAI": PAI_METRICS, "EEI": EEI_METRICS, "All": ALL_METRICS},
        "tables": sorted([f"tables/{k}.csv" for k in tables.keys()]),
        "plots_png": sorted([f"plots/{p.name}" for p in paths["plots"].glob("*.png")]),
        "plots_pdf": sorted([f"plots_pdf/{p.name}" for p in paths["plots_pdf"].glob("*.pdf")]),
        "scipy_import_error": SCIPY_IMPORT_ERROR,
        "statsmodels_import_error": STATSMODELS_IMPORT_ERROR,
    }
    out = paths["root"] / "analysis_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def zip_outputs(paths: dict[str, Path]) -> Path:
    zip_path = paths["root"].parent / "LLM_RE_Advanced_Analysis_Outputs.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", paths["root"])
    return zip_path

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive analysis pipeline for LLM reviewer evaluations.")
    parser.add_argument("--input",type=Path,default=Path(r"C:\Users\hamda\python-projects\experiments-working\Analysis_and_excel\Aggregated_Reviews.xlsx"), help="Path to Aggregated_Reviews.xlsx")
    parser.add_argument("--sheet", default=0, help="Sheet name or index. Default: first sheet.")
    parser.add_argument("--outdir", type=Path, default=Path(r"C:\Users\hamda\python-projects\experiments-working\Analysis_and_excel\Analysis_outputs_advanced"), help="Output directory")
    args = parser.parse_args()

    # Convert numeric sheet argument if supplied as digits.
    sheet: str | int = int(args.sheet) if isinstance(args.sheet, str) and args.sheet.isdigit() else args.sheet
    args.sheet = sheet

    paths = ensure_dirs(args.outdir)
    df = load_reviews(args.input, sheet_name=sheet)
    df.to_csv(paths["root"] / "clean_data.csv", index=False)

    tables = build_all_tables(df)
    export_tables(tables, paths)
    create_all_plots(df, tables, paths)
    excel_path = export_excel(df, tables, paths)
    report_path = write_report(df, tables, paths)
    manifest_path = write_manifest(df, tables, paths, args)
    zip_path = zip_outputs(paths)

    print("Advanced analysis complete.")
    print(f"Rows analyzed: {len(df)}")
    print(f"Tables: {paths['tables']}")
    print(f"PNG plots: {paths['plots']}")
    print(f"PDF plots: {paths['plots_pdf']}")
    print(f"Excel report: {excel_path}")
    print(f"Markdown report: {report_path}")
    print(f"Manifest: {manifest_path}")
    print(f"ZIP package: {zip_path}")


if __name__ == "__main__":
    main()
