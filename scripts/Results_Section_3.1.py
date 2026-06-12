from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


# ============================================================
# USER SETTINGS
# ============================================================


PERSISTENT_HOURLY_RUN_DIRS = [
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_111_0805_023351"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_222_0805_024643"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_333_0805_025935"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_444_0805_031228"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_555_0805_032520"),
]

PERSISTENT_15MIN_RUN_DIRS = [
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_111_0705_215859"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_222_0705_225049"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_333_0705_234251"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_444_0805_003455"),
    Path("/home/mhshah1/runs/persistent_4basins_seq_sweep/persistent_seed_555_0805_012708"),
]

MTS_HOURLY_RUN_DIRS = [
    Path("runs/persistent_seed_111_0805_033921"),
    Path("runs/persistent_seed_222_0805_034132"),
    Path("runs/persistent_seed_333_0805_034342"),
    Path("runs/persistent_seed_444_0805_034553"),
    Path("runs/persistent_seed_555_0805_034806"),
]

OUTDIR = Path("/mnt/disk1/to_laptop/persistent_all_24_basins_plots2_ensemble_for_paper144")
OUTDIR.mkdir(parents=True, exist_ok=True)

CAMELS_TOPO = Path("/mnt/disk1/CAMELS_US/camels_attributes_v2.0/camels_topo.txt")

TEST_START = pd.Timestamp("2012-11-01 00:00:00")
TEST_END = pd.Timestamp("2013-11-30 00:00:00")


BASINS = [
    "12010000",
    "12020000",
    "12025700",
    "12035000",
    "12040500",
    "12048000",
    "12054000",
    "12056500",
    "12082500",
    "12092000",
]

PREFERRED_HOURLY_KEYS = ["1H", "H", "hourly", "1h", "Hourly", "60min"]
PREFERRED_15MIN_KEYS = ["15T", "15min", "15Min", "15MIN", "15m", "quarter_hourly"]

PERSISTENT_HOURLY_UNIT = "mm/hour"
PERSISTENT_15MIN_UNIT = "cfs"
MTS_HOURLY_UNIT = "mm/hour"

OBSERVED_HOURLY_UNIT = "mm/hour"
OBSERVED_15MIN_UNIT = "cfs"

MIN_SEEDS_REQUIRED = 1
MIN_HOURS_PER_DAY = 20

DPI = 300
FIGSIZE = (18, 6)

TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 15
LEGEND_SIZE = 14

LINE_WIDTH_OBS = 2.0
LINE_WIDTH_MODEL = 1.5
OBS_ALPHA = 0.90
MODEL_ALPHA = 0.85


# ============================================================
# AREA LOOKUP
# ============================================================

def load_area_lookup(topo_path: Path) -> dict[str, float]:
    topo = pd.read_csv(topo_path, sep=";")
    topo["gauge_id"] = topo["gauge_id"].astype(str).str.zfill(8)
    topo["area_gages2"] = topo["area_gages2"].astype(float)
    return dict(zip(topo["gauge_id"], topo["area_gages2"]))


AREA_KM2 = load_area_lookup(CAMELS_TOPO)


# ============================================================
# UNIT CONVERSIONS
# ============================================================

def mm_per_hour_to_m3s(q_mm_hour, area_km2: float):
    return np.asarray(q_mm_hour, dtype=float) * float(area_km2) * (1000.0 / 3600.0)


def cfs_to_m3s(q_cfs):
    return np.asarray(q_cfs, dtype=float) * 0.028316846592


def convert_series_to_m3s(values, basin: str, unit: str):
    basin = basin.zfill(8)
    area_km2 = AREA_KM2.get(basin, np.nan)

    if unit == "mm/hour":
        if not np.isfinite(area_km2):
            raise RuntimeError(f"Missing basin area for basin {basin}")
        return mm_per_hour_to_m3s(values, area_km2)

    if unit == "cfs":
        return cfs_to_m3s(values)

    if unit == "m3/s":
        return np.asarray(values, dtype=float)

    raise ValueError(f"Unsupported unit: {unit}")


# ============================================================
# LOAD RESULTS
# ============================================================

def find_test_results(run_dir: Path) -> Path:
    candidates = []
    candidates += list((run_dir / "test").glob("**/test_results.p"))
    candidates += list((run_dir / "evaluate").glob("**/test_results.p"))
    candidates += list(run_dir.glob("**/test_results.p"))
    candidates = sorted(set(candidates))

    if not candidates:
        raise FileNotFoundError(f"No test_results.p found under: {run_dir}")

    return candidates[0]


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


def extract_from_xr(xr, freq_key: str) -> pd.DataFrame:
    obs_name = pick_var(xr, ["qobs", "obs", "y", "target", "QObs", "Qobs"])
    pred_name = pick_var(xr, ["qsim", "sim", "pred", "prediction", "y_hat", "QSim", "Qsim"])

    if obs_name is None:
        raise RuntimeError(f"Could not detect obs var. data_vars={list(xr.data_vars)}")

    tname = detect_time_name(xr)
    if tname is None:
        raise RuntimeError(f"Could not detect time coordinate. coords={list(xr.coords)}, dims={list(xr.dims)}")

    t = pd.to_datetime(xr[tname].values)
    obs = np.squeeze(np.asarray(xr[obs_name].values))
    pred = np.squeeze(np.asarray(xr[pred_name].values)) if pred_name is not None else None

    if obs.ndim == 1:
        n = min(len(t), len(obs)) if pred is None else min(len(t), len(obs), len(pred))
        data = {"obs": obs[:n]}
        if pred is not None:
            data["pred"] = pred[:n]

        df = pd.DataFrame(data, index=pd.DatetimeIndex(t[:n]))
        return df[~df.index.duplicated(keep="first")].sort_index()

    if pred is not None and obs.ndim == 2 and pred.ndim == 2:
        if obs.shape != pred.shape:
            raise RuntimeError(f"Obs/pred shape mismatch: obs={obs.shape}, pred={pred.shape}")

        n_rows, n_cols = obs.shape
        if len(t) != n_rows:
            raise RuntimeError(f"Time length mismatch: len(time)={len(t)}, obs.shape={obs.shape}")

        obs_flat = obs.reshape(-1)
        pred_flat = pred.reshape(-1)

        expanded_time = []
        freq_lower = str(freq_key).lower()

        if "15" in freq_lower:
            for base_time in t:
                for k in range(n_cols):
                    expanded_time.append(base_time + pd.Timedelta(minutes=15 * k))
        elif n_cols == 24:
            for base_time in t:
                for h in range(24):
                    expanded_time.append(base_time + pd.Timedelta(hours=h))
        else:
            step_hours = 24 / n_cols
            for base_time in t:
                for k in range(n_cols):
                    expanded_time.append(base_time + pd.Timedelta(hours=k * step_hours))

        expanded_time = pd.DatetimeIndex(expanded_time)
        n = min(len(expanded_time), len(obs_flat), len(pred_flat))

        df = pd.DataFrame(
            {"obs": obs_flat[:n], "pred": pred_flat[:n]},
            index=expanded_time[:n],
        )
        return df[~df.index.duplicated(keep="first")].sort_index()

    raise RuntimeError(
        f"Unsupported dimensions after squeeze: obs={obs.shape}, pred shape={None if pred is None else pred.shape}"
    )


def choose_freq_key(d_basin: dict, preferred_keys: list[str]) -> str:
    keys = list(d_basin.keys())

    for k in preferred_keys:
        if k in keys:
            return k

    raise KeyError(f"None of preferred keys {preferred_keys} found. Available keys: {keys}")


def load_single_run_for_basin(run_dir: Path, preferred_keys: list[str], basin: str) -> pd.DataFrame:
    p = find_test_results(run_dir)
    d = load_pickle(p)

    if not isinstance(d, dict):
        raise RuntimeError(f"Expected dict in {p}, got {type(d)}")

    basin = basin.zfill(8)

    if basin not in d:
        raise KeyError(f"Basin {basin} not found in {p}")

    basin_block = d[basin]
    freq_key = choose_freq_key(basin_block, preferred_keys)
    block = basin_block[freq_key]

    if not isinstance(block, dict) or "xr" not in block:
        raise RuntimeError(f"No xr block found for basin {basin}, freq {freq_key} in {p}")

    df = extract_from_xr(block["xr"], freq_key=freq_key)
    df = df.loc[TEST_START:TEST_END].copy()

    if df.empty:
        raise RuntimeError(f"Extracted dataframe is empty for basin {basin} in {p}")

    return df


# ============================================================
# CONVERSION
# ============================================================

def convert_prediction_df_to_m3s(df: pd.DataFrame, basin: str, unit: str) -> pd.DataFrame:
    out = df.copy()

    if "pred" not in out.columns:
        raise RuntimeError("Prediction dataframe does not contain pred column.")

    out["pred_m3s"] = convert_series_to_m3s(out["pred"].values, basin, unit)
    return out


def convert_observed_df_to_m3s(df: pd.DataFrame, basin: str, unit: str) -> pd.DataFrame:
    out = df.copy()

    if "obs" not in out.columns:
        raise RuntimeError("Observed dataframe does not contain obs column.")

    out["obs_m3s"] = convert_series_to_m3s(out["obs"].values, basin, unit)
    return out


# ============================================================
# GAP-SAFE RESAMPLING
# ============================================================

def make_continuous_index(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty:
        return df

    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    return df.reindex(full_idx)


def pred_to_hourly_mean_keep_gaps(df: pd.DataFrame, expected_per_hour: int) -> pd.DataFrame:
    s = df["pred_m3s"].copy()

    mean = s.resample("h").mean()
    count = s.resample("h").count()

    mean[count < expected_per_hour] = np.nan

    return pd.DataFrame({"pred_m3s": mean})


def obs_to_hourly_mean_keep_gaps(df: pd.DataFrame, expected_per_hour: int) -> pd.DataFrame:
    s = df["obs_m3s"].copy()

    mean = s.resample("h").mean()
    count = s.resample("h").count()

    mean[count < expected_per_hour] = np.nan

    return pd.DataFrame({"obs_m3s": mean})


def to_daily_mean_keep_gaps(
    df: pd.DataFrame,
    col: str,
    out_col: str,
    min_hours_per_day: int = 20,
) -> pd.DataFrame:
    s = df[col].copy()

    mean = s.resample("D").mean()
    count = s.resample("D").count()

    mean[count < min_hours_per_day] = np.nan

    return pd.DataFrame({out_col: mean})


def ensemble_mean_from_seed_dfs(seed_dfs: list[pd.DataFrame], col: str, out_col: str) -> pd.DataFrame:
    series_list = [df[col].rename(f"seed_{i+1}") for i, df in enumerate(seed_dfs)]

    joined = pd.concat(series_list, axis=1).sort_index()
    joined = make_continuous_index(joined, "h")

    counts = joined.notna().sum(axis=1)
    ens = joined.mean(axis=1, skipna=True)
    ens[counts < MIN_SEEDS_REQUIRED] = np.nan

    return pd.DataFrame({out_col: ens})


def combine_keep_gaps(*dfs, freq: str = "h") -> pd.DataFrame:
    combined = pd.concat(dfs, axis=1).sort_index()
    return make_continuous_index(combined, freq)


# ============================================================
# LOADERS
# ============================================================

def load_observed_hourly_source(basin: str) -> pd.DataFrame:
    df = load_single_run_for_basin(
        PERSISTENT_HOURLY_RUN_DIRS[0],
        PREFERRED_HOURLY_KEYS,
        basin,
    )

    df_m3s = convert_observed_df_to_m3s(df, basin, OBSERVED_HOURLY_UNIT)

    out = obs_to_hourly_mean_keep_gaps(
        df_m3s,
        expected_per_hour=1,
    )

    return out.rename(columns={"obs_m3s": "obs_hourly_m3s"})


def load_observed_15min_source_as_hourly_mean(basin: str) -> pd.DataFrame:
    df = load_single_run_for_basin(
        PERSISTENT_15MIN_RUN_DIRS[0],
        PREFERRED_15MIN_KEYS,
        basin,
    )

    df_m3s = convert_observed_df_to_m3s(df, basin, OBSERVED_15MIN_UNIT)

    out = obs_to_hourly_mean_keep_gaps(
        df_m3s,
        expected_per_hour=4,
    )

    return out.rename(columns={"obs_m3s": "obs_15min_hourly_m3s"})


def load_ensemble_prediction_as_hourly_mean(
    run_dirs: list[Path],
    preferred_keys: list[str],
    basin: str,
    unit: str,
    out_col: str,
    expected_per_hour: int,
) -> pd.DataFrame:
    seed_hourly_dfs = []

    for run_dir in run_dirs:
        df = load_single_run_for_basin(run_dir, preferred_keys, basin)
        df_m3s = convert_prediction_df_to_m3s(df, basin, unit)

        # IMPORTANT:
        # This converts prediction to hourly mean.
        # For persistent 15-min prediction, expected_per_hour must be 4.
        hourly = pred_to_hourly_mean_keep_gaps(
            df_m3s,
            expected_per_hour=expected_per_hour,
        )

        seed_hourly_dfs.append(hourly)

    return ensemble_mean_from_seed_dfs(seed_hourly_dfs, "pred_m3s", out_col)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(df: pd.DataFrame, obs_col: str, sim_col: str):
    pair = df[[obs_col, sim_col]].dropna()

    if len(pair) < 2:
        return {
            "NSE": np.nan,
            "KGE": np.nan,
            "Pearson r": np.nan,
            "Alpha-NSE": np.nan,
            "Beta-NSE": np.nan,
            "N_points": len(pair),
            "Start": None,
            "End": None,
        }

    obs = pair[obs_col].values.astype(float)
    sim = pair[sim_col].values.astype(float)

    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)

    obs_std = np.std(obs, ddof=1)
    sim_std = np.std(sim, ddof=1)

    denom = np.sum((obs - obs_mean) ** 2)
    nse_val = np.nan if denom == 0 else 1.0 - np.sum((sim - obs) ** 2) / denom

    r_val = np.nan if obs_std == 0 or sim_std == 0 else np.corrcoef(obs, sim)[0, 1]

    alpha_nse = np.nan if obs_std == 0 else sim_std / obs_std
    beta_nse = np.nan if obs_std == 0 else (sim_mean - obs_mean) / obs_std

    beta_kge = np.nan if obs_mean == 0 else sim_mean / obs_mean

    if np.isnan(r_val) or np.isnan(alpha_nse) or np.isnan(beta_kge):
        kge_val = np.nan
    else:
        kge_val = 1.0 - np.sqrt(
            (r_val - 1.0) ** 2 +
            (alpha_nse - 1.0) ** 2 +
            (beta_kge - 1.0) ** 2
        )

    return {
        "NSE": nse_val,
        "KGE": kge_val,
        "Pearson r": r_val,
        "Alpha-NSE": alpha_nse,
        "Beta-NSE": beta_nse,
        "N_points": len(pair),
        "Start": pair.index.min(),
        "End": pair.index.max(),
    }


# ============================================================
# PLOT STYLE
# ============================================================

def style_axes(ax):
    ax.set_xlabel("Date", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Discharge (m³/s)", fontsize=LABEL_SIZE, fontweight="bold")

    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.5)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(True, alpha=0.25)

    legend = ax.legend(loc="upper right", fontsize=LEGEND_SIZE, frameon=True)

    for text in legend.get_texts():
        text.set_fontweight("bold")

    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.1f}"))


# ============================================================
# PLOTTING
# ============================================================

def save_hourly_plot(df_hourly: pd.DataFrame, basin: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        df_hourly.index,
        df_hourly["obs_15min_hourly_m3s"],
        label="Observed 15-min (hourly mean)",
        linewidth=LINE_WIDTH_OBS,
        alpha=OBS_ALPHA,
    )

    ax.plot(
        df_hourly.index,
        df_hourly["obs_hourly_m3s"],
        label="Observed hourly",
        linewidth=LINE_WIDTH_OBS,
        linestyle="--",
        alpha=OBS_ALPHA,
    )

    ax.plot(
        df_hourly.index,
        df_hourly["persistent_hourly_m3s"],
        label="Persistent hourly",
        linewidth=LINE_WIDTH_MODEL,
        alpha=MODEL_ALPHA,
    )

    ax.plot(
        df_hourly.index,
        df_hourly["persistent_15min_hourly_m3s"],
        label="Persistent 15-min (hourly mean)",
        linewidth=LINE_WIDTH_MODEL,
        alpha=MODEL_ALPHA,
    )

    ax.plot(
        df_hourly.index,
        df_hourly["mts_hourly_m3s"],
        label="MTS-LSTM hourly",
        linewidth=LINE_WIDTH_MODEL,
        alpha=MODEL_ALPHA,
    )

    ax.set_title(
        f"{basin} — ensemble comparison, hourly mean, m³/s",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )

    style_axes(ax)

    plt.tight_layout()
    fig.savefig(OUTDIR / f"{basin}_ensemble_hourly_keep_gaps_m3s.png", dpi=DPI)
    plt.close(fig)


def save_daily_plot(df_daily: pd.DataFrame, basin: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        df_daily.index,
        df_daily["obs_hourly_daily_m3s"],
        label="Observed hourly daily mean",
        linewidth=LINE_WIDTH_OBS,
        alpha=OBS_ALPHA,
    )

    ax.plot(
        df_daily.index,
        df_daily["persistent_hourly_daily_m3s"],
        label="Persistent hourly daily mean",
        linewidth=LINE_WIDTH_MODEL,
        alpha=MODEL_ALPHA,
    )

    ax.plot(
        df_daily.index,
        df_daily["persistent_15min_daily_m3s"],
        label="Persistent 15-min daily mean",
        linewidth=LINE_WIDTH_MODEL,
        alpha=MODEL_ALPHA,
    )

    ax.plot(
        df_daily.index,
        df_daily["mts_hourly_daily_m3s"],
        label="MTS-LSTM hourly daily mean",
        linewidth=LINE_WIDTH_MODEL,
        alpha=MODEL_ALPHA,
    )

    ax.set_title(
        f"{basin} — ensemble comparison, daily mean, m³/s",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )

    style_axes(ax)

    plt.tight_layout()
    fig.savefig(OUTDIR / f"{basin}_ensemble_daily_keep_gaps_m3s.png", dpi=DPI)
    plt.close(fig)


def save_nse_cdf_plot(metrics_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7))

    model_order = [
        "Persistent hourly ensemble",
        "Persistent 15-min ensemble",
        "MTS hourly ensemble",
    ]

    for model in model_order:
        values = metrics_df.loc[metrics_df["Model"] == model, "NSE"].dropna().sort_values().values

        if len(values) == 0:
            continue

        cdf = np.arange(1, len(values) + 1) / len(values)
        median_val = np.nanmedian(values)

        ax.plot(values, cdf, linewidth=2.5, label=f"{model} median = {median_val:.2f}")
        ax.axvline(median_val, linestyle="--", linewidth=1.5)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.8, label="CDF = 0.5")
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.8, label="NSE = 0")
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1.8, label="NSE = 0.5")

    ax.set_title("NSE CDF — Test Basins", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("NSE", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("CDF", fontsize=LABEL_SIZE, fontweight="bold")

    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.5)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(True, alpha=0.25)

    legend = ax.legend(loc="upper left", fontsize=LEGEND_SIZE, frameon=True)

    for text in legend.get_texts():
        text.set_fontweight("bold")

    plt.tight_layout()
    fig.savefig(OUTDIR / "NSE_CDF_ensemble_keep_gaps.png", dpi=DPI)
    plt.close(fig)


# ============================================================
# PROCESS ONE BASIN
# ============================================================

def process_one_basin(basin: str):
    basin = basin.zfill(8)

    print(f"\n=== Processing basin {basin} ===")

    obs_hourly = load_observed_hourly_source(basin)
    obs_15min_hourly = load_observed_15min_source_as_hourly_mean(basin)

    persistent_hourly = load_ensemble_prediction_as_hourly_mean(
        PERSISTENT_HOURLY_RUN_DIRS,
        PREFERRED_HOURLY_KEYS,
        basin,
        PERSISTENT_HOURLY_UNIT,
        "persistent_hourly_m3s",
        expected_per_hour=1,
    )

    persistent_15min = load_ensemble_prediction_as_hourly_mean(
        PERSISTENT_15MIN_RUN_DIRS,
        PREFERRED_15MIN_KEYS,
        basin,
        PERSISTENT_15MIN_UNIT,
        "persistent_15min_hourly_m3s",
        expected_per_hour=4,
    )

    mts_hourly = load_ensemble_prediction_as_hourly_mean(
        MTS_HOURLY_RUN_DIRS,
        PREFERRED_HOURLY_KEYS,
        basin,
        MTS_HOURLY_UNIT,
        "mts_hourly_m3s",
        expected_per_hour=1,
    )

    hourly_compare = combine_keep_gaps(
        obs_15min_hourly,
        obs_hourly,
        persistent_hourly,
        persistent_15min,
        mts_hourly,
        freq="h",
    )

    save_hourly_plot(hourly_compare, basin)

    obs_hourly_daily = to_daily_mean_keep_gaps(
        hourly_compare,
        "obs_hourly_m3s",
        "obs_hourly_daily_m3s",
        MIN_HOURS_PER_DAY,
    )

    ph_daily = to_daily_mean_keep_gaps(
        hourly_compare,
        "persistent_hourly_m3s",
        "persistent_hourly_daily_m3s",
        MIN_HOURS_PER_DAY,
    )

    p15_daily = to_daily_mean_keep_gaps(
        hourly_compare,
        "persistent_15min_hourly_m3s",
        "persistent_15min_daily_m3s",
        MIN_HOURS_PER_DAY,
    )

    mts_daily = to_daily_mean_keep_gaps(
        hourly_compare,
        "mts_hourly_m3s",
        "mts_hourly_daily_m3s",
        MIN_HOURS_PER_DAY,
    )

    daily_compare = combine_keep_gaps(
        obs_hourly_daily,
        ph_daily,
        p15_daily,
        mts_daily,
        freq="D",
    )

    save_daily_plot(daily_compare, basin)

    rows = []
    obs_col_for_metrics = "obs_15min_hourly_m3s"

    model_map = {
        "Persistent hourly ensemble": "persistent_hourly_m3s",
        "Persistent 15-min ensemble": "persistent_15min_hourly_m3s",
        "MTS hourly ensemble": "mts_hourly_m3s",
    }

    for model_name, sim_col in model_map.items():
        m = compute_metrics(hourly_compare, obs_col_for_metrics, sim_col)
        m["Basin"] = basin
        m["Model"] = model_name
        rows.append(m)

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    success = []
    failed = []
    all_metrics = []

    for basin in BASINS:
        try:
            rows = process_one_basin(basin)
            all_metrics.extend(rows)
            success.append(basin)
        except Exception as e:
            print(f"❌ Failed for basin {basin}: {e}")
            failed.append((basin, str(e)))

    metrics_df = pd.DataFrame(all_metrics)

    metrics_csv = OUTDIR / "all_basin_metrics_ensemble_keep_gaps_CORRECT_KGE.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    print(f"\nSaved metrics CSV: {metrics_csv}")

    median_metrics = (
        metrics_df
        .groupby("Model")[["NSE", "KGE", "Pearson r", "Alpha-NSE", "Beta-NSE"]]
        .median()
        .round(3)
    )

    median_csv = OUTDIR / "median_metrics_ensemble_keep_gaps_CORRECT_KGE.csv"
    median_metrics.to_csv(median_csv)

    print(f"Saved median metrics table: {median_csv}")

    save_nse_cdf_plot(metrics_df)

    print(f"Saved CDF plot: {OUTDIR / 'NSE_CDF_ensemble_keep_gaps.png'}")

    print("\n================ MEDIAN METRICS ================")
    print(median_metrics)

    print("\n================ SUMMARY ================")
    print(f"Successful basins: {len(success)}")

    for b in success:
        print(f"  ✅ {b}")

    print(f"\nFailed basins: {len(failed)}")

    for b, err in failed:
        print(f"  ❌ {b}: {err}")


if __name__ == "__main__":
    main()
