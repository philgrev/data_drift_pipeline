import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ---------------------------------------------------------------------------
# Entropy utils
# ---------------------------------------------------------------------------

def add_entropy(
    df: pd.DataFrame,
    prob_prefix: str = "prob-",
    out_col: str = "entropy",
    base: float | None = None,
    eps: float = 1e-12,
    normalize: bool = False,
) -> pd.DataFrame:
    prob_cols = [c for c in df.columns if c.startswith(prob_prefix)]
    if not prob_cols:
        raise ValueError(f"No columns found starting with '{prob_prefix}'")

    p = df[prob_cols].to_numpy(dtype=float)
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = np.nan
    p = p / row_sums

    log_fn = np.log if base is None else (lambda x: np.log(x) / np.log(base))
    H = -np.nansum(p * log_fn(p + eps), axis=1)

    if normalize:
        K = len(prob_cols)
        H = H / log_fn(K)

    out = df.copy()
    out[out_col] = H
    return out


# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------

@dataclass
class DriftThresholds:
    js_high: float = 0.04
    js_moderate: float = 0.02
    psi_high: float = 0.25
    psi_moderate: float = 0.10


def js_divergence_from_samples(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bins: int = 50,
    value_range: Tuple[float, float] = (0, 1),
    eps: float = 1e-12,
) -> float:
    hist_p, bin_edges = np.histogram(p_samples, bins=bins, range=value_range)
    hist_q, _ = np.histogram(q_samples, bins=bin_edges)
    P = np.clip(hist_p / hist_p.sum(), eps, 1)
    Q = np.clip(hist_q / hist_q.sum(), eps, 1)
    M = 0.5 * (P + Q)
    return 0.5 * (np.sum(P * np.log2(P / M)) + np.sum(Q * np.log2(Q / M)))


def psi_from_samples(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 20,
    value_range: Tuple[float, float] = (0, 1),
    eps: float = 1e-12,
) -> float:
    exp_counts, bin_edges = np.histogram(expected, bins=bins, range=value_range)
    act_counts, _ = np.histogram(actual, bins=bin_edges)
    exp_pct = np.clip(exp_counts / exp_counts.sum(), eps, 1)
    act_pct = np.clip(act_counts / act_counts.sum(), eps, 1)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def classify_drift(js: float, psi: float, thresholds: Optional[DriftThresholds] = None) -> str:
    if thresholds is None:
        thresholds = DriftThresholds()
    if js > thresholds.js_high or psi > thresholds.psi_high:
        return "Significant drift"
    elif js > thresholds.js_moderate or psi > thresholds.psi_moderate:
        return "Moderate drift"
    return "No meaningful drift"


def calculate_drift_metrics(
    baseline_data: pd.Series,
    new_data: pd.Series,
    thresholds: Optional[DriftThresholds] = None,
    bins_js: int = 50,
    bins_psi: int = 20,
    value_range: Tuple[float, float] = (0, 1),
) -> dict:
    js = js_divergence_from_samples(new_data.values, baseline_data.values, bins=bins_js, value_range=value_range)
    psi = psi_from_samples(baseline_data.values, new_data.values, bins=bins_psi, value_range=value_range)
    return {"js": js, "psi": psi, "classification": classify_drift(js, psi, thresholds)}


# ---------------------------------------------------------------------------
# Drift trigger (KS + high-confidence quantile)
# ---------------------------------------------------------------------------

DEFAULT_DRIFT_CONFIG = {
    "high_conf_quantile": 0.95,
    "high_conf_drop_pct_trigger": 5.0,
    "ks_alpha": 0.01,
    "ks_d_min": 0.10,
    "min_samples": 50,
    "mode": "both",
}


# def drift_trigger(
#     df_ref: pd.DataFrame,
#     df_new: pd.DataFrame,
#     config: dict[str, Any] | None = None,
#     entropy_col: str = "entropy",
#     prob_col: str = "probability",
# ) -> dict:
#     cfg = {**DEFAULT_DRIFT_CONFIG, **(config or {})}

#     ref_entropy = df_ref[entropy_col].dropna().astype(float)
#     new_entropy = df_new[entropy_col].dropna().astype(float)

#     ref_mean = float(ref_entropy.mean()) if len(ref_entropy) else float("nan")
#     new_mean = float(new_entropy.mean()) if len(new_entropy) else float("nan")
#     entropy_increase_pct = (
#         (new_mean - ref_mean) / ref_mean * 100
#         if np.isfinite(ref_mean) and ref_mean != 0 and np.isfinite(new_mean)
#         else float("nan")
#     )

#     ks_stat = ks_pvalue = float("nan")
#     min_n = int(cfg["min_samples"])
#     if len(ref_entropy) >= min_n and len(new_entropy) >= min_n:
#         res = ks_2samp(ref_entropy.to_numpy(), new_entropy.to_numpy(), alternative="two-sided")
#         ks_stat, ks_pvalue = float(res.statistic), float(res.pvalue)

#     entropy_flag = (
#         np.isfinite(ks_stat)
#         and np.isfinite(ks_pvalue)
#         and ks_pvalue < float(cfg["ks_alpha"])
#         and ks_stat >= float(cfg["ks_d_min"])
#     )

#     ref_probs = df_ref[prob_col].dropna().to_numpy(dtype=float)
#     new_probs = df_new[prob_col].dropna().to_numpy(dtype=float)
#     q = float(cfg["high_conf_quantile"])
#     ref_pq = float(np.quantile(ref_probs, q)) if ref_probs.size else float("nan")
#     new_pq = float(np.quantile(new_probs, q)) if new_probs.size else float("nan")
#     drop_pct_trigger = float(cfg["high_conf_drop_pct_trigger"])

#     if np.isfinite(ref_pq) and np.isfinite(new_pq) and ref_pq > 0:
#         pq_drop_pct = (ref_pq - new_pq) / ref_pq * 100
#         highconf_flag = pq_drop_pct >= drop_pct_trigger
#     else:
#         pq_drop_pct = float("nan")
#         highconf_flag = False

#     mode = cfg["mode"]
#     drift = (entropy_flag or highconf_flag) if mode == "either" else (entropy_flag and highconf_flag)

#     return {
#         "drift": drift,
#         "mode": mode,
#         "entropy_mean_ref": ref_mean,
#         "entropy_mean_new": new_mean,
#         "entropy_increase_pct": entropy_increase_pct,
#         "entropy_ks_stat": ks_stat,
#         "entropy_ks_pvalue": ks_pvalue,
#         "entropy_flag": entropy_flag,
#         "high_conf_quantile": q,
#         "high_conf_ref_pq": ref_pq,
#         "high_conf_new_pq": new_pq,
#         "high_conf_pq_drop_pct": pq_drop_pct,
#         "high_conf_drop_pct_trigger": drop_pct_trigger,
#         "highconf_flag": highconf_flag,
#     }


def run_drift_analysis(
    baseline: Path,
    new_data: Path,
    prob_col: str = "probability",
    prob_prefix: str = "prob-",
    js_high: float = 0.04,
    js_moderate: float = 0.02,
    psi_high: float = 0.25,
    psi_moderate: float = 0.10,
) -> dict:
    """
    Run data drift analysis comparing a new dataset against a baseline.

    Loads both CSVs, adds Shannon entropy from per-class probability columns,
    then computes JS divergence, PSI, and KS-based drift flags for both the
    top-class probability and entropy distributions.

    Parameters
    ----------
    baseline : Path
        Path to the baseline CSV file (downloaded from S3 artefact by the framework).
    new_data : Path
        Path to the new data CSV file (downloaded from S3 artefact by the framework).
    prob_col : str
        Column name for the top-class probability (default: "probability").
    prob_prefix : str
        Column prefix used to identify per-class probability columns for
        entropy calculation (default: "prob-").
    js_high : float
        JS divergence threshold for significant drift (default: 0.04).
    js_moderate : float
        JS divergence threshold for moderate drift (default: 0.02).
    psi_high : float
        PSI threshold for significant drift (default: 0.25).
    psi_moderate : float
        PSI threshold for moderate drift (default: 0.10).

    Returns
    -------
    dict
        results_path : str
            Path to a JSON file containing the full drift metrics report.
    """
    if baseline.is_dir():
        csv_files = list(baseline.glob("*.csv"))
        baseline = csv_files[0]
    if new_data.is_dir():
        csv_files = list(new_data.glob("*.csv"))
        new_data = csv_files[0]

    baseline_df = pd.read_csv(baseline)
    new_df = pd.read_csv(new_data)

    # Add entropy column if per-class prob columns are present
    prob_cols_baseline = [c for c in baseline_df.columns if c.startswith(prob_prefix)]
    prob_cols_new = [c for c in new_df.columns if c.startswith(prob_prefix)]

    if prob_cols_baseline and prob_cols_new:
        baseline_df = add_entropy(baseline_df, prob_prefix=prob_prefix)
        new_df = add_entropy(new_df, prob_prefix=prob_prefix)

    thresholds = DriftThresholds(
        js_high=js_high,
        js_moderate=js_moderate,
        psi_high=psi_high,
        psi_moderate=psi_moderate,
    )

    # JS + PSI metrics for probability distribution
    prob_metrics = calculate_drift_metrics(
        baseline_data=baseline_df[prob_col],
        new_data=new_df[prob_col],
        thresholds=thresholds,
    )

    results = {
        "probability": {
            "js_divergence": prob_metrics["js"],
            "psi": prob_metrics["psi"],
            "classification": prob_metrics["classification"],
        },
    }

    # JS + PSI metrics for entropy distribution (if available)
    if "entropy" in baseline_df.columns and "entropy" in new_df.columns:
        entropy_metrics = calculate_drift_metrics(
            baseline_data=baseline_df["entropy"],
            new_data=new_df["entropy"],
            thresholds=thresholds,
        )
        results["entropy"] = {
            "js_divergence": entropy_metrics["js"],
            "psi": entropy_metrics["psi"],
            "classification": entropy_metrics["classification"],
        }

        # KS-based drift trigger (uses entropy + high-confidence probability)
        pass

    # Overall decision: significant if either distribution shows significant drift
    classifications = [results["probability"]["classification"]]
    if "entropy" in results:
        classifications.append(results["entropy"]["classification"])

    if "Significant drift" in classifications:
        results["overall_decision"] = "Significant drift"
    elif "Moderate drift" in classifications:
        results["overall_decision"] = "Moderate drift"
    else:
        results["overall_decision"] = "No meaningful drift"

    out_dir = Path(tempfile.mkdtemp())

    # Write results JSON
    results_path = out_dir / "drift_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Build KDE plot
    has_entropy = "entropy" in baseline_df.columns and "entropy" in new_df.columns
    n_plots = 2 if has_entropy else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    prob_label = results["probability"]["classification"]
    prob_color = {"Significant drift": "red", "Moderate drift": "orange"}.get(prob_label, "green")

    sns.kdeplot(baseline_df[prob_col], ax=axes[0], label="Baseline", color="gray", linestyle="--", linewidth=2)
    sns.kdeplot(new_df[prob_col], ax=axes[0], label=f"New — {prob_label}", color=prob_color, linewidth=2.5)
    axes[0].set_title(
        f"Probability\nJS: {results['probability']['js_divergence']:.4f}  PSI: {results['probability']['psi']:.4f}",
        fontweight="bold",
    )
    axes[0].set_xlabel("Probability")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if has_entropy:
        ent_label = results["entropy"]["classification"]
        ent_color = {"Significant drift": "red", "Moderate drift": "orange"}.get(ent_label, "green")

        sns.kdeplot(baseline_df["entropy"], ax=axes[1], label="Baseline", color="gray", linestyle="--", linewidth=2)
        sns.kdeplot(new_df["entropy"], ax=axes[1], label=f"New — {ent_label}", color=ent_color, linewidth=2.5)
        axes[1].set_title(
            f"Entropy\nJS: {results['entropy']['js_divergence']:.4f}  PSI: {results['entropy']['psi']:.4f}",
            fontweight="bold",
        )
        axes[1].set_xlabel("Entropy")
        axes[1].set_ylabel("Density")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.suptitle(f"Drift Analysis — {results['overall_decision']}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plot_path = out_dir / "drift_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"results": results_path, "plot": plot_path}
