from __future__ import annotations

from pathlib import Path
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pymannkendall as mk


# ============================================================
# USER SETTINGS
# ============================================================

PERSISTENT_DAILY_RUN_DIRS = [
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_111_0405_132440"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_222_0405_133018"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_333_0405_133557"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_444_0405_134137"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_555_0405_134713"),
]

CUDA_SEQ_GROUPS = {
    180: [
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_111_0405_135537"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_222_0405_135706"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_333_0405_135836"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_444_0405_140005"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_555_0405_140135"),
    ],
    270: [
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_111_0405_140920"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_222_0405_141120"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_333_0405_141321"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_444_0405_141521"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_555_0405_141722"),
    ],
    365: [
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_111_0405_142432"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_222_0405_142706"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_333_0405_142941"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_444_0405_143216"),
        Path("/home/mhshah1/runs/cudalstm_daily_compare/persistent_seed_555_0405_143451"),
    ],
}

OUTDIR = Path("/mnt/disk1/to_laptop/daily_persistent_vs_cudalstm_no_warmup_paper_plots_2221")
OUTDIR.mkdir(parents=True, exist_ok=True)

BASINS = [
    "02096846",
    "04063700",
    "09065500",
    "04057510",
    "09066200",
    "08014500",
    "12082500",
    "12013500",
    "12189500",
    "14185000",
    "12035000",
    "13331500",
]

OBS_ROOT = Path("/mnt/disk1/CAMELS_US/usgs_streamflow")
CAMELS_TOPO_FILE = Path("/mnt/disk1/CAMELS_US/camels_attributes_v2.0/camels_topo.txt")

PREFERRED_DAILY_KEYS = ["1D", "D", "daily", "1d", "Daily"]
TARGET_UNIT_LABEL = "Discharge (mm/day)"

DPI = 400
FIGSIZE = (22, 8)
MIN_SEEDS_REQUIRED = 1
SAVE_CSV = True
PRINT_DEBUG_INFO = True
ANNOTATE_SCATTER = True

TEST_START_DATE = pd.Timestamp("2012-10-01 00:00:00")
TEST_END_DATE = pd.Timestamp("2013-10-30 00:00:00")

WARMUP_DAYS = 90
EVAL_START_DATE = TEST_START_DATE + pd.Timedelta(days=WARMUP_DAYS)
EVAL_END_DATE = TEST_END_DATE

FIXED_TEST_INDEX = pd.date_range(TEST_START_DATE, TEST_END_DATE, freq="D")
FIXED_EVAL_INDEX = pd.date_range(EVAL_START_DATE, EVAL_END_DATE, freq="D")


# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "axes.labelweight": "bold",
    "legend.fontsize": 17,
    "legend.title_fontsize": 17,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 1.8,
    "savefig.dpi": DPI,
})

OBSERVED_COLOR = "#9467bd"
PERSISTENT_COLOR = "#ff7f0e"

CUDA_SEQ_COLORS = {
    180: "#9ecae1",
    270: "#4292c6",
    365: "#08519c",
}

OBS_LINEWIDTH = 2.8
PERSISTENT_LINEWIDTH = 2.8
CUDA_LINEWIDTH = 2.8

OBS_LINESTYLE = "-"
PERSISTENT_LINESTYLE = "-."
CUDA_LINESTYLE = "--"


# ============================================================
# BASIC HELPERS
# ============================================================

def epoch_number_from_path(path: Path) -> int:
    m = re.search(r"model_epoch(\d+)", path.name)
    return int(m.group(1)) if m else -1


def find_test_results(run_dir: Path) -> Path:
    run_dir = Path(str(run_dir).strip())

    for folder_name in ["test", "evaluate"]:
        base = run_dir / folder_name
        if base.exists():
            epoch_dirs = [p for p in base.glob("model_epoch*") if p.is_dir()]
            if epoch_dirs:
                epoch_dirs = sorted(epoch_dirs, key=epoch_number_from_path)
                p = epoch_dirs[-1] / "test_results.p"
                if p.exists():
                    return p

    candidates = sorted(set(run_dir.glob("**/test_results.p")))
    if candidates:
        candidates = sorted(candidates, key=lambda p: epoch_number_from_path(p.parent))
        return candidates[-1]

    raise FileNotFoundError(f"No test_results.p found under: {run_dir}")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def pick_var(xr, candidates):
    vars_ = list(getattr(xr, "data_vars", []))

    for c in candidates:
        if c in vars_:
            return c

    for c in candidates:
        for v in vars_:
            if c.lower() == str(v).lower():
                return v

    for c in candidates:
        for v in vars_:
            if c.lower() in str(v).lower():
                return v

    return None


def detect_time_name(xr):
    for c in ["date", "time", "datetime", "t"]:
        if c in xr.coords:
            return c

    for d in xr.dims:
        if "time" in d.lower() or "date" in d.lower():
            return d

    return None


def choose_freq_key(d_basin: dict, preferred_keys: list[str]) -> str:
    keys = list(d_basin.keys())
    for k in preferred_keys:
        if k in keys:
            return k

    raise KeyError(f"None of preferred keys {preferred_keys} found. Available keys: {keys}")


def remove_warmup_period(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[
        (df.index >= EVAL_START_DATE) &
        (df.index <= EVAL_END_DATE)
    ].reindex(FIXED_EVAL_INDEX).copy()


def bold_ticks(ax):
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def bold_legend(legend):
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")


# ============================================================
# LOAD NEURALHYDROLOGY TEST RESULTS
# ============================================================

def extract_from_xr_daily(xr) -> pd.DataFrame:
    obs_name = pick_var(
        xr,
        ["QObs(mm/d)", "QObs", "Qobs", "qobs", "obs", "y", "target"]
    )
    pred_name = pick_var(
        xr,
        ["QSim", "Qsim", "qsim", "sim", "pred", "prediction", "y_hat"]
    )

    if obs_name is None:
        raise RuntimeError(f"Could not detect observed variable. data_vars={list(xr.data_vars)}")

    if pred_name is None:
        raise RuntimeError(f"Could not detect predicted variable. data_vars={list(xr.data_vars)}")

    tname = detect_time_name(xr)
    if tname is None:
        raise RuntimeError(
            f"Could not detect time coordinate. coords={list(xr.coords)}, dims={list(xr.dims)}"
        )

    t = pd.to_datetime(xr[tname].values)
    obs = np.squeeze(np.asarray(xr[obs_name].values))
    pred = np.squeeze(np.asarray(xr[pred_name].values))

    if obs.ndim == 2 and obs.shape[1] == 1:
        obs = obs[:, 0]

    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]

    if obs.ndim != 1 or pred.ndim != 1:
        raise RuntimeError(f"Unsupported shapes: obs={obs.shape}, pred={pred.shape}")

    n = min(len(t), len(obs), len(pred))

    df = pd.DataFrame(
        {"obs": obs[:n], "pred": pred[:n]},
        index=pd.DatetimeIndex(t[:n])
    )

    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def clip_to_test_window(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.sort_index()
    df = df.loc[(df.index >= TEST_START_DATE) & (df.index <= TEST_END_DATE)].copy()

    if PRINT_DEBUG_INFO:
        if df.empty:
            print(f"  {name}: EMPTY after clipping to fixed test window")
        else:
            print(f"  {name}: clipped to {df.index.min()} --> {df.index.max()}")

    return df


def reindex_to_fixed_test_period(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(FIXED_TEST_INDEX)


def load_single_run_for_basin(run_dir: Path, preferred_keys: list[str], basin: str) -> pd.DataFrame:
    p = find_test_results(run_dir)
    d = load_pickle(p)

    if not isinstance(d, dict):
        raise RuntimeError(f"Expected dict in {p}, got {type(d)}")

    basin = str(basin).zfill(8)

    if PRINT_DEBUG_INFO:
        print(f"  Loading basin {basin} from: {p}")

    if basin not in d:
        available = sorted(list(d.keys()))
        raise KeyError(f"Basin {basin} not found in {p}. Available basins: {available}")

    basin_block = d[basin]

    if not isinstance(basin_block, dict) or len(basin_block) == 0:
        raise RuntimeError(f"Basin block for {basin} is empty in {p}")

    freq_key = choose_freq_key(basin_block, preferred_keys)
    block = basin_block[freq_key]

    if not isinstance(block, dict) or "xr" not in block:
        raise RuntimeError(f"No xr block found for basin {basin}, freq {freq_key} in {p}")

    df = extract_from_xr_daily(block["xr"])

    if df.empty:
        raise RuntimeError(f"Extracted dataframe is empty for basin {basin} in {p}")

    return df


# ============================================================
# LOAD FULL ORIGINAL CAMELS/USGS OBSERVED DATA FOR MK TEST
# ============================================================

def load_camels_area_km2(topo_file: Path = CAMELS_TOPO_FILE) -> dict[str, float]:
    if not topo_file.exists():
        raise FileNotFoundError(f"CAMELS topo file not found: {topo_file}")

    topo = pd.read_csv(topo_file, sep=";", dtype={"gauge_id": str})
    topo["gauge_id"] = topo["gauge_id"].astype(str).str.zfill(8)

    if "area_gages2" not in topo.columns:
        raise RuntimeError(
            f"'area_gages2' column not found in {topo_file}. Columns: {list(topo.columns)}"
        )

    return dict(zip(topo["gauge_id"], topo["area_gages2"].astype(float)))


def find_camels_streamflow_file(basin: str, obs_root: Path = OBS_ROOT) -> Path:
    basin = str(basin).zfill(8)

    patterns = [
        f"**/{basin}_streamflow_qc.txt",
        f"**/{basin}*streamflow*.txt",
        f"**/{basin}*.txt",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(list(obs_root.glob(pattern)))

    matches = sorted(set(matches))

    if not matches:
        raise FileNotFoundError(
            f"No CAMELS/USGS streamflow file found for basin {basin} under {obs_root}"
        )

    return matches[0]


def load_full_observed_camels_mm_day(
    basin: str,
    area_lookup: dict[str, float],
    obs_root: Path = OBS_ROOT,
) -> pd.DataFrame:
    basin = str(basin).zfill(8)

    if basin not in area_lookup:
        raise KeyError(f"Area not found for basin {basin} in CAMELS topo file.")

    area_km2 = float(area_lookup[basin])
    sf_file = find_camels_streamflow_file(basin, obs_root)

    df = pd.read_csv(
        sf_file,
        sep=r"\s+",
        header=None,
        names=["basin", "year", "month", "day", "q_cfs", "flag"],
        dtype={"basin": str},
    )

    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]),
        errors="coerce"
    )

    df["q_cfs"] = pd.to_numeric(df["q_cfs"], errors="coerce")
    df.loc[df["q_cfs"] < 0, "q_cfs"] = np.nan

    q_m3s = df["q_cfs"] * 0.028316846592
    df["obs_mm_day"] = q_m3s * 86.4 / area_km2

    out = df[["date", "obs_mm_day"]].dropna(subset=["date"])
    out = out.set_index("date").sort_index()
    out = out[~out.index.duplicated(keep="first")]

    if PRINT_DEBUG_INFO:
        print(
            f"  Full observed CAMELS {basin}: "
            f"{out.index.min()} --> {out.index.max()}, "
            f"n={out['obs_mm_day'].notna().sum()}, file={sf_file}"
        )

    return out


# ============================================================
# METRICS
# ============================================================

def _clean_obs_sim(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


def compute_nse(obs, sim):
    obs, sim = _clean_obs_sim(obs, sim)

    if len(obs) < 2:
        return np.nan

    denom = np.sum((obs - np.mean(obs)) ** 2)

    if denom == 0:
        return np.nan

    return 1.0 - np.sum((sim - obs) ** 2) / denom


def compute_alpha_nse(obs, sim):
    obs, sim = _clean_obs_sim(obs, sim)

    if len(obs) < 2:
        return np.nan

    std_obs = np.std(obs, ddof=0)
    std_sim = np.std(sim, ddof=0)

    if std_obs == 0:
        return np.nan

    return std_sim / std_obs


def compute_beta_nse(obs, sim):
    obs, sim = _clean_obs_sim(obs, sim)

    if len(obs) < 2:
        return np.nan

    std_obs = np.std(obs, ddof=0)

    if std_obs == 0:
        return np.nan

    return (np.mean(sim) - np.mean(obs)) / std_obs


def compute_pearson_r(obs, sim):
    obs, sim = _clean_obs_sim(obs, sim)

    if len(obs) < 2:
        return np.nan

    if np.std(obs, ddof=0) == 0 or np.std(sim, ddof=0) == 0:
        return np.nan

    return float(np.corrcoef(obs, sim)[0, 1])


def compute_kge_components(obs, sim):
    obs, sim = _clean_obs_sim(obs, sim)

    if len(obs) < 2:
        return {"kge": np.nan, "kge_r": np.nan, "kge_alpha": np.nan, "kge_beta": np.nan}

    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    std_obs = np.std(obs, ddof=0)
    std_sim = np.std(sim, ddof=0)

    if mean_obs == 0 or std_obs == 0 or std_sim == 0:
        return {"kge": np.nan, "kge_r": np.nan, "kge_alpha": np.nan, "kge_beta": np.nan}

    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = std_sim / std_obs
    beta = mean_sim / mean_obs

    kge = 1.0 - np.sqrt(
        (r - 1.0) ** 2 +
        (alpha - 1.0) ** 2 +
        (beta - 1.0) ** 2
    )

    return {
        "kge": kge,
        "kge_r": r,
        "kge_alpha": alpha,
        "kge_beta": beta,
    }


def compute_all_test_metrics(obs, sim) -> dict:
    clean_obs, clean_sim = _clean_obs_sim(obs, sim)

    out = {}
    out["n_valid_days"] = len(clean_obs)
    out["nse"] = compute_nse(clean_obs, clean_sim)
    out["alpha_nse"] = compute_alpha_nse(clean_obs, clean_sim)
    out["beta_nse"] = compute_beta_nse(clean_obs, clean_sim)
    out["pearson_r"] = compute_pearson_r(clean_obs, clean_sim)

    kge_parts = compute_kge_components(clean_obs, clean_sim)
    out.update(kge_parts)

    return out


def compute_mk_tau(series):
    s = pd.Series(series).dropna()

    if len(s) < 2:
        return np.nan, np.nan

    result = mk.original_test(s.values)
    return result.Tau, result.p


def classify_tau(tau):
    if pd.isna(tau):
        return "unknown"

    a = abs(tau)

    if a >= 0.30:
        return "strong_nonstationary"
    elif a >= 0.20:
        return "moderate_high_nonstationary"
    elif a >= 0.10:
        return "weak_moderate_nonstationary"
    else:
        return "weak_or_stationary"


# ============================================================
# ENSEMBLE
# ============================================================

def ensemble_mean_from_seed_series(
    seed_series: list[pd.Series],
    min_seeds_required: int = 1,
) -> pd.Series:
    if len(seed_series) == 0:
        raise ValueError("No seed series provided.")

    df = pd.concat(seed_series, axis=1)
    df.columns = [f"seed_{i + 1}" for i in range(df.shape[1])]

    counts = df.notna().sum(axis=1)
    mean_series = df.mean(axis=1, skipna=True)
    mean_series[counts < min_seeds_required] = np.nan

    return mean_series


def load_observed_daily_from_first_run(
    run_dirs: list[Path],
    preferred_keys: list[str],
    basin: str,
    out_col: str,
) -> pd.DataFrame:
    if len(run_dirs) == 0:
        raise ValueError("Observed source run_dirs list is empty.")

    df = load_single_run_for_basin(run_dirs[0], preferred_keys, basin)

    if "obs" not in df.columns:
        raise RuntimeError(f"'obs' column missing for basin {basin}")

    df = df[["obs"]].rename(columns={"obs": out_col})
    df = clip_to_test_window(df, f"Observed test-period {basin}")
    df = reindex_to_fixed_test_period(df)

    return df


def load_ensemble_prediction_daily(
    run_dirs: list[Path],
    preferred_keys: list[str],
    basin: str,
    out_col: str,
    min_seeds_required: int = 1,
) -> pd.DataFrame:
    seed_daily_series = []

    for i, run_dir in enumerate(run_dirs, start=1):
        df = load_single_run_for_basin(run_dir, preferred_keys, basin)

        if "pred" not in df.columns:
            raise RuntimeError(f"'pred' column missing in run {run_dir} for basin {basin}")

        df = df[["pred"]].rename(columns={"pred": f"seed_{i}"})
        df = clip_to_test_window(df, f"{out_col} seed {i}")
        df = reindex_to_fixed_test_period(df)

        seed_daily_series.append(df.iloc[:, 0])

    ensemble = ensemble_mean_from_seed_series(
        seed_daily_series,
        min_seeds_required=min_seeds_required,
    )

    return pd.DataFrame({out_col: ensemble}, index=FIXED_TEST_INDEX)


# ============================================================
# PLOTS / SAVE
# ============================================================

def save_basin_outputs(df_eval_compare: pd.DataFrame, basin: str, seq_lengths: list[int]):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        df_eval_compare.index,
        df_eval_compare["obs_mm_day"],
        label="Observed",
        linewidth=OBS_LINEWIDTH,
        color=OBSERVED_COLOR,
        linestyle=OBS_LINESTYLE,
        zorder=5,
    )

    ax.plot(
        df_eval_compare.index,
        df_eval_compare["persistent_ensemble_mm_day"],
        label="Persistent LSTM",
        linewidth=PERSISTENT_LINEWIDTH,
        color=PERSISTENT_COLOR,
        linestyle=PERSISTENT_LINESTYLE,
    )

    for seq_len in seq_lengths:
        col = f"cuda_seq_{seq_len}_ensemble_mm_day"

        ax.plot(
            df_eval_compare.index,
            df_eval_compare[col],
            label=f"FSSTS LSTM (Seq-{seq_len} d)",
            linewidth=CUDA_LINEWIDTH,
            color=CUDA_SEQ_COLORS[seq_len],
            linestyle=CUDA_LINESTYLE,
        )

    ax.set_title(
        f"Basin {basin} — Daily Ensemble Comparison\n"
        f" ",
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel(TARGET_UNIT_LABEL, fontweight="bold")
    ax.grid(True, alpha=0.30)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
    ax.set_xlim(EVAL_START_DATE, EVAL_END_DATE)

    bold_ticks(ax)

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=17,
        borderpad=0.8,
        labelspacing=0.5,
    )
    bold_legend(legend)

    plt.tight_layout()

    out_png = OUTDIR / f"{basin}_daily_hydrograph_no_warmup_{WARMUP_DAYS}d_paper.png"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    if SAVE_CSV:
        out_csv = OUTDIR / f"{basin}_daily_compare_no_warmup_{WARMUP_DAYS}d.csv"
        df_eval_compare.to_csv(out_csv)

    print(f"Saved hydrograph without warmup: {out_png}")


def save_scatter_plot(df_metrics: pd.DataFrame, seq_len: int):
    ycol = f"delta_nse_{seq_len}"
    plot_df = df_metrics[["basin", "abs_tau", ycol]].dropna().copy()

    if plot_df.empty:
        print(f"No data available for scatter plot seq={seq_len}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        plot_df["abs_tau"],
        plot_df[ycol],
        s=90,
        alpha=0.90,
        edgecolor="black",
        linewidth=0.8,
    )

    if ANNOTATE_SCATTER:
        for _, row in plot_df.iterrows():
            ax.annotate(
                row["basin"],
                (row["abs_tau"], row[ycol]),
                fontsize=12,
                fontweight="bold",
                alpha=0.90,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.axhline(0.0, linestyle="--", linewidth=2.0, color="black")
    ax.set_xlabel("|Mann–Kendall τ| from Full Observed Record", fontweight="bold")
    ax.set_ylabel(f"ΔNSE = Persistent − CUDA ({seq_len} d)", fontweight="bold")
    ax.set_title(
        f"Performance Gain vs Non-Stationarity\n"
        f" ",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.30)

    bold_ticks(ax)

    plt.tight_layout()

    out_path = OUTDIR / f"tau_vs_delta_nse_seq_{seq_len}_no_warmup_{WARMUP_DAYS}d_paper.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved scatter plot: {out_path}")


def save_cdf_nse_plot(df_metrics: pd.DataFrame):
    nse_cols = {
        "Persistent LSTM": "persistent_nse",
        "CUDA LSTM 180 ": "cuda_180_nse",
        "CUDA LSTM 270 ": "cuda_270_nse",
        "CUDA LSTM 365 ": "cuda_365_nse",
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=2.2,
        color="black",
        alpha=0.75,
        label="CDF = 0.5",
    )

    for label, col in nse_cols.items():
        if col not in df_metrics.columns:
            continue

        values = pd.to_numeric(df_metrics[col], errors="coerce").dropna().values

        if len(values) == 0:
            continue

        values = np.sort(values)
        cdf = np.arange(1, len(values) + 1) / len(values)
        median_nse = np.median(values)

        ax.plot(
            values,
            cdf,
            linewidth=3.0,
            label=f"{label} median = {median_nse:.2f}",
        )

        ax.axvline(
            median_nse,
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
        )

    ax.axvline(0.0, linestyle=":", linewidth=2.2, color="black", alpha=0.85, label="NSE = 0")
    ax.axvline(0.5, linestyle=":", linewidth=2.2, color="black", alpha=0.85, label="NSE = 0.5")

    ax.set_xlabel("NSE", fontweight="bold")
    ax.set_ylabel("CDF", fontweight="bold")
    ax.set_title(
        f"NSE CDF — Test Basins",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.30)

    bold_ticks(ax)

    legend = ax.legend(
        frameon=True,
        loc="upper left",
        fontsize=15,
        borderpad=0.8,
        labelspacing=0.5,
    )
    bold_legend(legend)

    plt.tight_layout()

    out_path = OUTDIR / f"cdf_nse_persistent_vs_cudalstm_no_warmup_{WARMUP_DAYS}d_paper.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved CDF NSE plot: {out_path}")


# ============================================================
# PROCESS ONE BASIN
# ============================================================

def process_one_basin(basin: str, area_lookup: dict[str, float]):
    basin = str(basin).zfill(8)

    print(f"\n=== Processing basin {basin} ===")

    seq_lengths = sorted(CUDA_SEQ_GROUPS.keys())

    observed_full = load_full_observed_camels_mm_day(
        basin=basin,
        area_lookup=area_lookup,
        obs_root=OBS_ROOT,
    )

    tau, tau_pvalue = compute_mk_tau(observed_full["obs_mm_day"])

    observed_daily = load_observed_daily_from_first_run(
        run_dirs=PERSISTENT_DAILY_RUN_DIRS,
        preferred_keys=PREFERRED_DAILY_KEYS,
        basin=basin,
        out_col="obs_mm_day",
    )

    persistent_ensemble_daily = load_ensemble_prediction_daily(
        run_dirs=PERSISTENT_DAILY_RUN_DIRS,
        preferred_keys=PREFERRED_DAILY_KEYS,
        basin=basin,
        out_col="persistent_ensemble_mm_day",
        min_seeds_required=MIN_SEEDS_REQUIRED,
    )

    cuda_ensemble_frames = []

    for seq_len in seq_lengths:
        cuda_df = load_ensemble_prediction_daily(
            run_dirs=CUDA_SEQ_GROUPS[seq_len],
            preferred_keys=PREFERRED_DAILY_KEYS,
            basin=basin,
            out_col=f"cuda_seq_{seq_len}_ensemble_mm_day",
            min_seeds_required=MIN_SEEDS_REQUIRED,
        )
        cuda_ensemble_frames.append(cuda_df)

    daily_compare_full = pd.DataFrame(index=FIXED_TEST_INDEX)

    daily_compare_full["obs_mm_day"] = observed_daily["obs_mm_day"]
    daily_compare_full["persistent_ensemble_mm_day"] = persistent_ensemble_daily["persistent_ensemble_mm_day"]

    for seq_len, cuda_df in zip(seq_lengths, cuda_ensemble_frames):
        daily_compare_full[f"cuda_seq_{seq_len}_ensemble_mm_day"] = cuda_df[
            f"cuda_seq_{seq_len}_ensemble_mm_day"
        ]

    daily_compare_eval = remove_warmup_period(daily_compare_full)

    print("\n=== Full test-window coverage before warmup removal ===")
    print(f"Full index start: {daily_compare_full.index.min()}")
    print(f"Full index end  : {daily_compare_full.index.max()}")

    print(f"\n=== Evaluation-window coverage after removing {WARMUP_DAYS} warmup days ===")
    print(f"Eval index start: {daily_compare_eval.index.min()}")
    print(f"Eval index end  : {daily_compare_eval.index.max()}")
    print(f"Expected eval days: {len(FIXED_EVAL_INDEX)}")
    print(f"Rows with all finite values: {daily_compare_eval.dropna().shape[0]}")

    save_basin_outputs(daily_compare_eval, basin, seq_lengths)

    metrics = {
        "basin": basin,
        "full_obs_start": observed_full.index.min(),
        "full_obs_end": observed_full.index.max(),
        "full_obs_n_days": int(observed_full["obs_mm_day"].notna().sum()),
        "test_start_date": TEST_START_DATE,
        "test_end_date": TEST_END_DATE,
        "warmup_days_removed": WARMUP_DAYS,
        "eval_start_date_after_warmup": EVAL_START_DATE,
        "eval_end_date": EVAL_END_DATE,
        "eval_total_days": int(len(daily_compare_eval)),
        "eval_all_models_finite_days": int(daily_compare_eval.dropna().shape[0]),
        "tau": tau,
        "abs_tau": abs(tau) if pd.notna(tau) else np.nan,
        "tau_pvalue": tau_pvalue,
        "tau_class": classify_tau(tau),
    }

    persistent_metrics = compute_all_test_metrics(
        daily_compare_eval["obs_mm_day"],
        daily_compare_eval["persistent_ensemble_mm_day"],
    )

    for key, value in persistent_metrics.items():
        metrics[f"persistent_{key}"] = value

    print(f"\n=== Persistent metrics after removing {WARMUP_DAYS} warmup days ===")
    for key, value in persistent_metrics.items():
        print(f"persistent_{key}: {value}")

    for seq_len in seq_lengths:
        cuda_col = f"cuda_seq_{seq_len}_ensemble_mm_day"

        cuda_metrics = compute_all_test_metrics(
            daily_compare_eval["obs_mm_day"],
            daily_compare_eval[cuda_col],
        )

        for key, value in cuda_metrics.items():
            metrics[f"cuda_{seq_len}_{key}"] = value

        metrics[f"delta_nse_{seq_len}"] = (
            metrics["persistent_nse"] - metrics[f"cuda_{seq_len}_nse"]
            if pd.notna(metrics["persistent_nse"]) and pd.notna(metrics[f"cuda_{seq_len}_nse"])
            else np.nan
        )

        metrics[f"delta_kge_{seq_len}"] = (
            metrics["persistent_kge"] - metrics[f"cuda_{seq_len}_kge"]
            if pd.notna(metrics["persistent_kge"]) and pd.notna(metrics[f"cuda_{seq_len}_kge"])
            else np.nan
        )

        print(f"\n=== CUDA seq={seq_len} metrics after removing {WARMUP_DAYS} warmup days ===")
        for key, value in cuda_metrics.items():
            print(f"cuda_{seq_len}_{key}: {value}")

        print(f"delta_nse_{seq_len}: {metrics[f'delta_nse_{seq_len}']}")
        print(f"delta_kge_{seq_len}: {metrics[f'delta_kge_{seq_len}']}")

    print("\n=== Basin non-stationarity ===")
    print(f"tau        : {tau}")
    print(f"tau_pvalue : {tau_pvalue}")
    print(f"tau_class  : {classify_tau(tau)}")

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    success = []
    failed = []
    metrics_rows = []

    print("\n================ TEST / EVALUATION WINDOWS ================")
    print(f"test_start_date          : {TEST_START_DATE}")
    print(f"test_end_date            : {TEST_END_DATE}")
    print(f"warmup_days_removed      : {WARMUP_DAYS}")
    print(f"evaluation_start_date    : {EVAL_START_DATE}")
    print(f"evaluation_end_date      : {EVAL_END_DATE}")
    print("===========================================================")

    print("\nLoading CAMELS basin areas...")
    area_lookup = load_camels_area_km2(CAMELS_TOPO_FILE)
    print(f"Loaded areas for {len(area_lookup)} basins.")

    for basin in BASINS:
        try:
            metrics = process_one_basin(basin, area_lookup)
            metrics_rows.append(metrics)
            success.append(str(basin).zfill(8))

        except Exception as e:
            print(f"❌ Failed for basin {basin}: {e}")
            failed.append((str(basin).zfill(8), str(e)))

    print("\n================ SUMMARY ================")

    print(f"Successful basins: {len(success)}")
    for b in success:
        print(f"  ✅ {b}")

    print(f"\nFailed basins: {len(failed)}")
    for b, err in failed:
        print(f"  ❌ {b}: {err}")

    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows)
        df_metrics = df_metrics.sort_values("abs_tau", ascending=False)

        summary_csv = OUTDIR / f"basin_full_obs_tau_test_metrics_no_warmup_{WARMUP_DAYS}d_summary.csv"
        df_metrics.to_csv(summary_csv, index=False)

        print("\nSaved basin summary CSV:")
        print(summary_csv)

        print("\n=== Basin summary preview ===")
        print(df_metrics.to_string(index=False))

        delta_nse_cols = [
            f"delta_nse_{seq_len}" for seq_len in sorted(CUDA_SEQ_GROUPS.keys())
        ]

        delta_kge_cols = [
            f"delta_kge_{seq_len}" for seq_len in sorted(CUDA_SEQ_GROUPS.keys())
        ]

        print("\n=== Mean ΔNSE by tau class ===")
        grouped_nse = df_metrics.groupby("tau_class")[delta_nse_cols].mean(numeric_only=True)
        print(grouped_nse.to_string())

        print("\n=== Mean ΔKGE by tau class ===")
        grouped_kge = df_metrics.groupby("tau_class")[delta_kge_cols].mean(numeric_only=True)
        print(grouped_kge.to_string())

        for seq_len in sorted(CUDA_SEQ_GROUPS.keys()):
            save_scatter_plot(df_metrics, seq_len)

        save_cdf_nse_plot(df_metrics)


if __name__ == "__main__":
    main()
