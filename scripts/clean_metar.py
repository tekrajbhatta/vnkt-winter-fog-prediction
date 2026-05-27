#!/usr/bin/env python3
"""
scripts/clean_metar.py
======================

Step 1 of the VNKT Winter Fog Forecasting project.

Loads the raw IEM ASOS-AWOS-METAR CSV for station VNKT (Tribhuvan
International Airport, Kathmandu), performs unit conversion and
quality-control filtering, and writes a cleaned hourly-resolution
parquet file plus a one-page validation report.

This module is import-safe (no side effects on import). The CLI entry
point is `python -m scripts.clean_metar` or `python scripts/clean_metar.py`.

Author : Tek Raj Bhatt
Module : STW7085CEM Advanced Machine Learning (March 2026)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Nepal Standard Time offset from UTC: +5 hours 45 minutes (a famously
# awkward half-hour offset — easy to get wrong, so isolated here)
NPT_OFFSET = pd.Timedelta(hours=5, minutes=45)

# IEM uses 'M' for missing values and 'T' for trace precipitation
IEM_MISSING_TOKENS = ["M", "T"]

# Columns we keep from the raw IEM file. Everything else is dropped because:
#   mslp           -> 0.00% present (drop)
#   p01i           -> all-zero placeholder for non-US stations (drop)
#   gust           -> 0.18% present (drop)
#   ice_accretion* -> N/A in subtropical climate (drop)
#   peak_wind_*    -> sparse (drop)
#   snowdepth      -> N/A in Kathmandu (drop)
#   feel           -> derived from T, RH; no new info (drop)
#   skyc2-4, skyl2-4 -> sparse; only lowest layer matters for fog (drop)
RAW_COLUMNS_TO_KEEP = [
    "station",
    "valid",
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "alti",
    "vsby",
    "skyc1",
    "skyl1",
    "wxcodes",
    "metar",
]

# Ordinal encoding for sky-cover category. Higher = more overcast.
SKY_COVER_ORDINAL = {
    "CLR": 0,   # clear
    "SKC": 0,   # clear (legacy spelling)
    "NCD": 0,   # nil cloud detected
    "NSC": 0,   # no significant cloud
    "FEW": 1,   # few (1-2 octas)
    "SCT": 2,   # scattered (3-4 octas)
    "BKN": 3,   # broken (5-7 octas)
    "OVC": 4,   # overcast (8 octas)
    "VV":  5,   # vertical visibility (sky obscured, often fog)
}

# Physically plausible ranges for Kathmandu surface observations.
# Anything outside these ranges is flagged as a QC failure and replaced with NaN.
PLAUSIBLE_RANGES = {
    "tempc":         (-10.0,  45.0),   # °C
    "dewpointc":     (-15.0,  35.0),   # °C
    "relh":          (  0.0, 100.0),   # %
    "wind_speed_ms": (  0.0,  40.0),   # m/s
    "wind_dir_deg":  (  0.0, 360.0),   # degrees
    "pressure_hpa":  (700.0, 1050.0),  # hPa (Kathmandu sits ~1300m, MSLP-adjusted)
    "visibility_m":  (  0.0, 16100.0), # m (10 statute miles cap)
}

# Class label thresholds (metres). Operationally grounded per
# Kathmandu Post 2026-01-29 reporting:
#   - RNP-AR landing minimum: 1100 m
#   - VOR-DME landing minimum: 1600 m
#   - Departure minimum: 1200 m
# Project uses 1600 m and 800 m as the simplest two-threshold scheme.
CLASS_THRESHOLDS = {
    "diversions_likely": 800.0,
    "delays_likely":     1600.0,
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = True) -> logging.Logger:
    """Configure a console logger with a tidy format."""
    logger = logging.getLogger("clean_metar")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Unit conversion utilities (each function is independently testable)
# ---------------------------------------------------------------------------

def fahrenheit_to_celsius(temp_f: pd.Series) -> pd.Series:
    """°F -> °C."""
    return (temp_f - 32.0) * 5.0 / 9.0


def miles_to_metres(miles: pd.Series) -> pd.Series:
    """Statute miles -> metres."""
    return miles * 1609.344


def knots_to_mps(knots: pd.Series) -> pd.Series:
    """Knots -> metres per second."""
    return knots * 0.514444


def inhg_to_hpa(inhg: pd.Series) -> pd.Series:
    """Altimeter setting in inches of mercury -> hPa (millibars)."""
    return inhg * 33.8639


def wind_to_uv(speed: pd.Series, direction_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Decompose meteorological wind (direction wind is *from*, in degrees)
    into mathematical u (eastward) and v (northward) components.

    A wind FROM 270 deg blows TO the east, so u is positive.
    """
    rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


# ---------------------------------------------------------------------------
# Core cleaning pipeline
# ---------------------------------------------------------------------------

@dataclass
class CleaningStats:
    """Bookkeeping for the validation report."""
    raw_rows: int = 0
    rows_after_dedup: int = 0
    rows_after_qc: int = 0
    qc_violations: dict[str, int] = None
    fog_events: int = 0
    mist_events: int = 0
    date_min: Optional[pd.Timestamp] = None
    date_max: Optional[pd.Timestamp] = None

    def __post_init__(self) -> None:
        if self.qc_violations is None:
            self.qc_violations = {}


def load_raw(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load the IEM raw CSV, treating IEM's M / T tokens as missing."""
    logger.info(f"Loading raw file: {path}")
    df = pd.read_csv(
        path,
        low_memory=False,
        na_values=IEM_MISSING_TOKENS,
        keep_default_na=True,
    )
    logger.info(f"  raw shape: {df.shape}")
    return df


def select_columns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Drop unused columns to reduce memory footprint."""
    keep = [c for c in RAW_COLUMNS_TO_KEEP if c in df.columns]
    missing = set(RAW_COLUMNS_TO_KEEP) - set(keep)
    if missing:
        logger.warning(f"  expected columns missing from file: {missing}")
    logger.info(f"  keeping {len(keep)} columns, dropping {df.shape[1] - len(keep)}")
    return df[keep].copy()


def coerce_numeric(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Convert numeric-looking string columns to floats. Errors -> NaN."""
    numeric_cols = ["tmpf", "dwpf", "relh", "drct", "sknt",
                    "alti", "vsby", "skyl1"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_timestamps(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Parse 'valid' as UTC and add a Nepal-local time column.

    Critical step: every downstream operational decision (fog window,
    diversion threshold) is in NPT, but the raw timestamps are UTC.
    """
    df["valid_utc"] = pd.to_datetime(df["valid"], errors="coerce", utc=True)
    df = df.drop(columns=["valid"])
    df["valid_npt"] = df["valid_utc"].dt.tz_convert(None) + NPT_OFFSET

    n_bad = df["valid_utc"].isna().sum()
    if n_bad > 0:
        logger.warning(f"  {n_bad} rows had unparseable timestamps; will be dropped")
        df = df.dropna(subset=["valid_utc"]).copy()
    return df


def deduplicate(df: pd.DataFrame, logger: logging.Logger,
                stats: CleaningStats) -> pd.DataFrame:
    """Remove exact-duplicate rows by timestamp (IEM occasionally emits these)."""
    before = len(df)
    df = df.sort_values("valid_utc").drop_duplicates(subset="valid_utc", keep="first")
    removed = before - len(df)
    if removed > 0:
        logger.info(f"  removed {removed} duplicate timestamps")
    stats.rows_after_dedup = len(df)
    return df


def convert_units(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Apply all imperial-to-SI conversions in one place."""
    logger.info("Converting units (°F→°C, miles→m, knots→m/s, inHg→hPa)...")
    df["tempc"]         = fahrenheit_to_celsius(df["tmpf"])
    df["dewpointc"]     = fahrenheit_to_celsius(df["dwpf"])
    df["wind_speed_ms"] = knots_to_mps(df["sknt"])
    df["wind_dir_deg"]  = df["drct"]                          # already in degrees
    df["pressure_hpa"]  = inhg_to_hpa(df["alti"])
    df["visibility_m"]  = miles_to_metres(df["vsby"])
    df["cloud_base_m"]  = df["skyl1"] * 0.3048                # ft -> m

    # the imperial originals are no longer needed
    df = df.drop(columns=["tmpf", "dwpf", "sknt", "drct", "alti", "vsby", "skyl1"])
    return df


def encode_sky_cover(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Map skyc1 textual codes to ordinal integers; missing -> -1."""
    logger.info("Ordinal-encoding sky cover (skyc1)...")
    df["sky_cover_ord"] = (
        df["skyc1"]
        .fillna("MISSING")
        .map(SKY_COVER_ORDINAL)
        .fillna(-1)
        .astype(int)
    )
    df = df.drop(columns=["skyc1"])
    return df


def extract_weather_flags(df: pd.DataFrame, logger: logging.Logger,
                          stats: CleaningStats) -> pd.DataFrame:
    """
    Decode the wxcodes string into binary flags for fog / mist / haze.
    These provide independent validation labels alongside the visibility
    threshold rule.
    """
    logger.info("Extracting weather flags from wxcodes...")
    codes = df["wxcodes"].fillna("")

    # Order of checks matters: FZFG must be caught before FG
    df["wx_freezing_fog"] = codes.str.contains(r"\bFZFG\b").astype(int)
    df["wx_fog"]          = codes.str.contains(r"\bFG\b").astype(int)
    df["wx_mist"]         = codes.str.contains(r"\bBR\b").astype(int)
    df["wx_haze"]         = codes.str.contains(r"\bHZ\b").astype(int)

    stats.fog_events  = int(df["wx_fog"].sum())
    stats.mist_events = int(df["wx_mist"].sum())
    logger.info(f"  fog (FG)  events: {stats.fog_events:,}")
    logger.info(f"  mist (BR) events: {stats.mist_events:,}")
    return df


def decompose_wind(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add wind u/v components, plus a `calm_wind` flag.

    Wind direction is undefined when the wind is variable or near-calm
    (drct is missing); we capture this explicitly because radiation fog
    is favoured by calm conditions.
    """
    logger.info("Decomposing wind into u/v components...")
    df["wind_calm"] = (
        df["wind_speed_ms"].fillna(0).lt(1.0)
        | df["wind_dir_deg"].isna()
    ).astype(int)

    direction_filled = df["wind_dir_deg"].fillna(0.0)
    speed_filled     = df["wind_speed_ms"].fillna(0.0)
    df["wind_u_ms"], df["wind_v_ms"] = wind_to_uv(speed_filled, direction_filled)
    return df


def quality_control(df: pd.DataFrame, logger: logging.Logger,
                    stats: CleaningStats) -> pd.DataFrame:
    """
    Replace physically implausible values with NaN, and record violations.

    We DO NOT drop rows here — downstream imputation handles isolated
    missing values, and dropping would create irregular time series.
    """
    logger.info("Quality control: flagging out-of-range values...")
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n_bad = int(mask.sum())
        if n_bad > 0:
            df.loc[mask, col] = np.nan
            stats.qc_violations[col] = n_bad
            logger.info(f"  {col:15s}: nulled {n_bad:,} out-of-range values")

    # Physical consistency: dew point cannot exceed temperature
    inconsistent = df["dewpointc"] > df["tempc"] + 0.5  # allow 0.5 °C tolerance
    n_inc = int(inconsistent.sum())
    if n_inc > 0:
        # When inconsistent, trust the temperature; clip dew point to temperature
        df.loc[inconsistent, "dewpointc"] = df.loc[inconsistent, "tempc"]
        stats.qc_violations["dewpoint_gt_temp"] = n_inc
        logger.info(f"  dewpoint>temp:    capped {n_inc:,} rows at temp value")

    # Derived: dew-point depression (the single strongest fog predictor)
    df["dewpoint_depression_c"] = df["tempc"] - df["dewpointc"]

    stats.rows_after_qc = len(df)
    return df


def finalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns into a tidy schema for downstream consumption."""
    schema = [
        # identifiers / time
        "station", "valid_utc", "valid_npt",
        # primary numeric features
        "tempc", "dewpointc", "dewpoint_depression_c", "relh",
        "wind_speed_ms", "wind_dir_deg", "wind_u_ms", "wind_v_ms", "wind_calm",
        "pressure_hpa", "sky_cover_ord", "cloud_base_m",
        # target candidates
        "visibility_m",
        # weather flags (independent labels)
        "wx_fog", "wx_freezing_fog", "wx_mist", "wx_haze",
        # raw for traceability
        "wxcodes", "metar",
    ]
    keep = [c for c in schema if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def write_validation_report(df: pd.DataFrame, stats: CleaningStats,
                            out_path: Path, logger: logging.Logger) -> None:
    """Write a human-readable Markdown validation report."""
    logger.info(f"Writing validation report: {out_path}")

    completeness = (
        df.notna().mean().sort_values(ascending=False) * 100
    ).round(2)

    # winter-only class distribution at hourly level (informational only;
    # the modelling table will be at daily resolution, built in Step 2)
    winter = df[df["valid_npt"].dt.month.isin([11, 12, 1])]
    vis = winter["visibility_m"].dropna()
    n_winter = len(vis)
    n_div = int((vis < CLASS_THRESHOLDS["diversions_likely"]).sum())
    n_del = int(((vis >= CLASS_THRESHOLDS["diversions_likely"]) &
                 (vis < CLASS_THRESHOLDS["delays_likely"])).sum())
    n_nor = int((vis >= CLASS_THRESHOLDS["delays_likely"]).sum())

    lines = []
    lines.append("# Step 1 Validation Report — VNKT METAR cleaning")
    lines.append("")
    lines.append(f"- **Raw rows loaded**:        {stats.raw_rows:>10,}")
    lines.append(f"- **After deduplication**:    {stats.rows_after_dedup:>10,}")
    lines.append(f"- **After quality control**:  {stats.rows_after_qc:>10,}")
    lines.append(f"- **Date range (NPT)**:       "
                 f"{df['valid_npt'].min()} → {df['valid_npt'].max()}")
    lines.append("")
    lines.append("## Column completeness (% non-missing)")
    lines.append("")
    lines.append("| Column | % present |")
    lines.append("|---|---:|")
    for col, pct in completeness.items():
        lines.append(f"| `{col}` | {pct:.2f}% |")
    lines.append("")
    lines.append("## QC violations (out-of-range values replaced with NaN)")
    lines.append("")
    if stats.qc_violations:
        lines.append("| Field | Violations |")
        lines.append("|---|---:|")
        for k, v in stats.qc_violations.items():
            lines.append(f"| `{k}` | {v:,} |")
    else:
        lines.append("_None._")
    lines.append("")
    lines.append("## Weather code event counts")
    lines.append("")
    lines.append(f"- Fog (`FG`)  events: **{stats.fog_events:,}**")
    lines.append(f"- Mist (`BR`) events: **{stats.mist_events:,}**")
    lines.append("")
    lines.append("## Hourly-level class distribution (winter Nov/Dec/Jan only)")
    lines.append("")
    lines.append(f"_Total winter observations with valid visibility: {n_winter:,}_")
    lines.append("")
    lines.append("| Class | Threshold | Count | Share |")
    lines.append("|---|---|---:|---:|")
    lines.append(f"| Diversions-Likely | vis < 800 m       | "
                 f"{n_div:,} | {100*n_div/n_winter:.2f}% |")
    lines.append(f"| Delays-Likely     | 800 ≤ vis < 1600m | "
                 f"{n_del:,} | {100*n_del/n_winter:.2f}% |")
    lines.append(f"| Normal            | vis ≥ 1600 m      | "
                 f"{n_nor:,} | {100*n_nor/n_winter:.2f}% |")
    lines.append("")
    lines.append("_Note: the modelling table built in Step 2 will be at "
                 "**daily** resolution (next-morning 05:45–09:45 NPT minimum "
                 "visibility per day), where the Diversions-Likely class rises "
                 "to ~3.4% — an imbalance that is workable for ML._")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def clean(raw_path: Path, output_path: Path, report_path: Path,
          logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Run the full cleaning pipeline and return the cleaned dataframe.

    Side effects:
      - writes a parquet file to `output_path`
      - writes a Markdown report to `report_path`
    """
    if logger is None:
        logger = setup_logging()

    stats = CleaningStats()

    df = load_raw(raw_path, logger)
    stats.raw_rows = len(df)

    df = select_columns(df, logger)
    df = coerce_numeric(df, logger)
    df = add_timestamps(df, logger)
    df = deduplicate(df, logger, stats)
    df = convert_units(df, logger)
    df = encode_sky_cover(df, logger)
    df = extract_weather_flags(df, logger, stats)
    df = decompose_wind(df, logger)
    df = quality_control(df, logger, stats)
    df = finalise_columns(df)

    stats.date_min = df["valid_npt"].min()
    stats.date_max = df["valid_npt"].max()

    # write outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Wrote cleaned table: {output_path}  ({len(df):,} rows)")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_validation_report(df, stats, report_path, logger)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean and validate the IEM METAR file for VNKT.",
    )
    p.add_argument(
        "--raw", type=Path,
        default=Path("data/raw/VNKT.csv"),
        help="Path to the raw IEM CSV file.",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("data/interim/vnkt_clean.parquet"),
        help="Path to write the cleaned parquet file.",
    )
    p.add_argument(
        "--report", type=Path,
        default=Path("reports/step1_validation.md"),
        help="Path to write the Markdown validation report.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    clean(args.raw, args.out, args.report)
