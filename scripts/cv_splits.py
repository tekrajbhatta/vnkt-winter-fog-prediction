"""
scripts/cv_splits.py
====================
Step 3 of the VNKT Winter Fog Forecasting project.

Constructs forward-chaining time-series cross-validation folds, applies
per-fold median imputation and standardisation, and exposes a generator that
downstream training scripts (Steps 4-6) consume directly.

DESIGN DECISIONS (committed in Step 3 design memo)
--------------------------------------------------
1.  The 2025-26 winter season is reserved as a strict holdout test set and
    never enters any CV fold. Even though it contains only two Diversions
    days, temporal validity outweighs holdout size for an explicitly
    forward-in-time forecasting task.

2.  The partial 2015-16 season (Jan-Feb 2016 only; data begin 2016-01-01,
    no Oct-Dec 2015 coverage) is rolled into Fold 1's training window
    alongside 2016-17. A single partial winter with ~60 days is too small
    to serve as a standalone training fold.

3.  Forward-chaining with a growing window: Fold k trains on all seasons
    strictly earlier than the test season, tests on a single season. No
    observation from a future season ever appears in training data for an
    earlier fold. Eight folds total.

4.  `night_obs_count` is intentionally EXCLUDED from the feature set even
    though Step 2 retains it in the parquet. The mid-2017 IEM ingest
    upgrade (median ~30 -> ~45 obs/night) creates a step change exactly in
    the earliest training folds. Including it as a feature would let the
    model learn the ingest schedule as a "weather" signal.

5.  Imputation: median, fitted on training data only, then applied to
    test data. The highest missing rate is 3.84% (overnight_pressure_change_hpa)
    so median is more than adequate; no need for iterative or KNN imputation.

6.  Standardisation: StandardScaler, fitted on training data only, applied
    uniformly to all 19 features INCLUDING doy_sin/doy_cos. The cyclic
    features start in [-1, 1] so scaling barely changes them in practice,
    but uniform scaling makes the ARD length-scales (Step 5) directly
    comparable across features.

7.  The regression target is standardised separately and the fitted target
    scaler is returned with each fold so downstream code can inverse-transform
    predictions for evaluation in metres.

CONSUMER CONTRACT
-----------------
Downstream Step 4/5/6 scripts iterate folds like this:

    from cv_splits import load_modelling_table, split_holdout, iter_prepared_folds

    df = load_modelling_table()
    train_pool, holdout = split_holdout(df)

    for prepared in iter_prepared_folds(train_pool):
        # prepared.X_train, prepared.X_test  : np.ndarray, standardised
        # prepared.y_reg_train, ...          : raw metres
        # prepared.y_reg_train_scaled, ...   : standardised target
        # prepared.y_clf_train, ...          : int labels 0/1/2
        # prepared.feature_pipeline          : fitted Pipeline (impute+scale)
        # prepared.target_scaler             : fitted StandardScaler for y
        # prepared.spec                      : FoldSpec metadata
        ...

CLI
---
    python scripts/cv_splits.py
    python scripts/cv_splits.py --parquet PATH --manifest PATH

Prints a per-fold summary table and writes a JSON manifest describing the
splits to data/processed/cv_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "processed" / "vnkt_modelling_table.parquet"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "cv_manifest.json"

HOLDOUT_SEASON: str = "2025-26"
INITIAL_TRAINING_SEASONS: list[str] = ["2015-16", "2016-17"]

# 19 feature columns. `night_obs_count` deliberately omitted (see docstring).
FEATURE_COLUMNS: list[str] = [
    # Sunset snapshot
    "sunset_tempc",
    "sunset_dewpoint_depr_c",
    "sunset_pressure_hpa",
    "sunset_wind_speed_ms",
    "sunset_visibility_m",
    # Pre-dawn snapshot
    "predawn_tempc",
    "predawn_dewpoint_depr_c",
    "predawn_pressure_hpa",
    # Overnight evolution
    "overnight_temp_drop_c",
    "overnight_dewpoint_depr_drop_c",
    "overnight_pressure_change_hpa",
    # Wind regime
    "night_mean_wind_speed_ms",
    "night_calm_fraction",
    # Sky cover
    "night_mean_sky_cover",
    "night_clear_fraction",
    # Pre-existing fog/mist
    "night_mist_observed",
    "night_fog_observed",
    # Seasonality
    "doy_sin",
    "doy_cos",
]

TARGET_REGRESSION: str = "target_min_vis_m"
TARGET_CLASSIFICATION: str = "target_class"

CLASS_NAMES: dict[int, str] = {0: "Normal", 1: "Delays", 2: "Diversions"}


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure module logger with [HH:MM:SS] LEVEL message format.

    Idempotent: clears existing handlers so re-imports inside a notebook
    do not duplicate output."""
    logger = logging.getLogger("cv_splits")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


log = setup_logging()


# ----------------------------------------------------------------------------
# Season assignment
# ----------------------------------------------------------------------------

def assign_winter_season(date: pd.Timestamp) -> str:
    """Map a date to its winter-season label (Oct-Feb).

    Convention: Oct-Dec of year Y belongs to season "Y-(Y+1)"; Jan-Feb of
    year Y belongs to season "(Y-1)-Y". So 2018-11-15 -> "2018-19" and
    2019-01-05 -> "2018-19". Labels match the per-season summary plot
    produced in Step 2.

    Raises ValueError for Mar-Sep dates: these should never appear after
    the Step 2 fog-season filter, and a silent miscategorisation here
    would corrupt every downstream CV split.
    """
    year = date.year
    month = date.month
    if month >= 10:
        start_year = year
    elif month <= 2:
        start_year = year - 1
    else:
        raise ValueError(
            f"Date {date.date()} lies outside the Oct-Feb winter window; "
            "the modelling table appears not to have been season-filtered."
        )
    return f"{start_year}-{str(start_year + 1)[-2:]}"


# ----------------------------------------------------------------------------
# Loading and holdout split
# ----------------------------------------------------------------------------

def load_modelling_table(parquet_path: Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """Load the Step 2 output parquet and attach a `season` column.

    Asserts the schema we depend on: all feature columns and both target
    columns must be present. Fails loudly if any are missing -- silent
    column drift would mean the model trained on the wrong inputs.
    """
    log.info(f"Loading modelling table: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df["date_npt"] = pd.to_datetime(df["date_npt"])
    df = df.sort_values("date_npt").reset_index(drop=True)

    required = set(FEATURE_COLUMNS) | {TARGET_REGRESSION, TARGET_CLASSIFICATION, "date_npt"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Modelling table missing required columns: {sorted(missing)}"
        )

    df["season"] = df["date_npt"].apply(assign_winter_season)
    log.info(f"  rows={len(df)}  seasons={df['season'].nunique()}  "
             f"date_range={df['date_npt'].min().date()} -> {df['date_npt'].max().date()}")
    return df


def split_holdout(
    df: pd.DataFrame, holdout_season: str = HOLDOUT_SEASON
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the modelling table into (train_pool, holdout).

    The holdout is a single season reserved for final evaluation. It must
    never enter any CV fold and must never be touched until the paper's
    final results section.
    """
    if holdout_season not in df["season"].unique():
        raise ValueError(
            f"Holdout season '{holdout_season}' not present in data; "
            f"available seasons: {sorted(df['season'].unique())}"
        )
    holdout = df[df["season"] == holdout_season].copy().reset_index(drop=True)
    train_pool = df[df["season"] != holdout_season].copy().reset_index(drop=True)
    log.info(f"Holdout season {holdout_season}: {len(holdout)} rows  |  "
             f"train pool: {len(train_pool)} rows across {train_pool['season'].nunique()} seasons")
    return train_pool, holdout


# ----------------------------------------------------------------------------
# Fold specification
# ----------------------------------------------------------------------------

@dataclass
class FoldSpec:
    """Metadata describing one forward-chaining CV fold.

    `train_idx` and `test_idx` are positional indices into the train_pool
    dataframe passed to `forward_chaining_folds`. They are not original
    parquet row numbers -- they assume the train_pool has been reset_index.
    """
    fold_id: int
    train_seasons: list[str]
    test_season: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def forward_chaining_folds(train_pool: pd.DataFrame) -> list[FoldSpec]:
    """Generate forward-chaining CV folds with growing-window training.

    Seasons in INITIAL_TRAINING_SEASONS are combined into Fold 1's training
    set. Each subsequent fold extends the training window by one season.

    With 10 training-pool seasons (2015-16 .. 2024-25), this yields 8 folds:

        Fold 1: train (2015-16 + 2016-17),  test 2017-18
        Fold 2: train + 2017-18,             test 2018-19
        ...
        Fold 8: train + 2023-24,             test 2024-25

    No shuffling. Strict temporal ordering preserved.
    """
    seasons_sorted = sorted(train_pool["season"].unique())

    # Verify the assumed initial seasons are actually the earliest in the data
    for s in INITIAL_TRAINING_SEASONS:
        if s not in seasons_sorted:
            raise ValueError(
                f"Initial training season '{s}' not found in train pool. "
                f"Available seasons: {seasons_sorted}"
            )
    if seasons_sorted[: len(INITIAL_TRAINING_SEASONS)] != INITIAL_TRAINING_SEASONS:
        raise ValueError(
            f"INITIAL_TRAINING_SEASONS {INITIAL_TRAINING_SEASONS} are not "
            f"the earliest in the data {seasons_sorted[:len(INITIAL_TRAINING_SEASONS)]}. "
            "This would break temporal ordering."
        )

    test_seasons = seasons_sorted[len(INITIAL_TRAINING_SEASONS):]
    folds: list[FoldSpec] = []
    cumulative_train = list(INITIAL_TRAINING_SEASONS)

    for fold_id, test_season in enumerate(test_seasons, start=1):
        train_mask = train_pool["season"].isin(cumulative_train)
        test_mask = train_pool["season"] == test_season
        train_idx = np.flatnonzero(train_mask.values)
        test_idx = np.flatnonzero(test_mask.values)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(
                f"Fold {fold_id} has empty train ({len(train_idx)}) "
                f"or test ({len(test_idx)}) set."
            )
        folds.append(FoldSpec(
            fold_id=fold_id,
            train_seasons=list(cumulative_train),
            test_season=test_season,
            train_idx=train_idx,
            test_idx=test_idx,
            train_start=train_pool.loc[train_idx, "date_npt"].min(),
            train_end=train_pool.loc[train_idx, "date_npt"].max(),
            test_start=train_pool.loc[test_idx, "date_npt"].min(),
            test_end=train_pool.loc[test_idx, "date_npt"].max(),
        ))
        cumulative_train.append(test_season)

    log.info(f"Built {len(folds)} forward-chaining folds")
    return folds


# ----------------------------------------------------------------------------
# Feature pipeline and prepared folds
# ----------------------------------------------------------------------------

def build_feature_pipeline() -> Pipeline:
    """Return an UNFITTED sklearn Pipeline: median impute -> standardise.

    A fresh pipeline is built per fold and fitted on training data only,
    so test-fold information never leaks into the fitted parameters.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def build_target_scaler() -> StandardScaler:
    """Return an UNFITTED StandardScaler for the regression target.

    GPflow / sklearn-GP regression converges much more reliably when the
    target is standardised: the default zero-mean Gaussian prior over
    function values then sits in roughly the right range. Random Forest
    is invariant to target scaling but we standardise uniformly for a
    consistent metrics pipeline."""
    return StandardScaler()


@dataclass
class PreparedFold:
    """Output of `iter_prepared_folds` -- everything Step 4-6 need."""
    spec: FoldSpec
    X_train: np.ndarray
    X_test: np.ndarray
    y_reg_train: np.ndarray         # raw metres
    y_reg_test: np.ndarray          # raw metres
    y_reg_train_scaled: np.ndarray  # standardised
    y_reg_test_scaled: np.ndarray   # standardised with TRAIN scaler
    y_clf_train: np.ndarray         # integer labels 0/1/2
    y_clf_test: np.ndarray
    feature_pipeline: Pipeline      # fitted on this fold's training X
    target_scaler: StandardScaler   # fitted on this fold's training y_reg
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))


def iter_prepared_folds(
    train_pool: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> Iterator[PreparedFold]:
    """Yield a `PreparedFold` for each forward-chaining CV fold.

    For each fold: fit the impute+scale pipeline on training X, transform
    both X_train and X_test; fit a separate StandardScaler on training y
    (regression target), apply to both train and test targets.
    """
    folds = forward_chaining_folds(train_pool)
    for spec in folds:
        X_train_raw = train_pool.loc[spec.train_idx, feature_columns].to_numpy(dtype=float)
        X_test_raw = train_pool.loc[spec.test_idx, feature_columns].to_numpy(dtype=float)
        y_reg_train = train_pool.loc[spec.train_idx, TARGET_REGRESSION].to_numpy(dtype=float)
        y_reg_test = train_pool.loc[spec.test_idx, TARGET_REGRESSION].to_numpy(dtype=float)
        y_clf_train = train_pool.loc[spec.train_idx, TARGET_CLASSIFICATION].to_numpy(dtype=int)
        y_clf_test = train_pool.loc[spec.test_idx, TARGET_CLASSIFICATION].to_numpy(dtype=int)

        pipe = build_feature_pipeline().fit(X_train_raw)
        X_train = pipe.transform(X_train_raw)
        X_test = pipe.transform(X_test_raw)

        target_scaler = build_target_scaler().fit(y_reg_train.reshape(-1, 1))
        y_reg_train_scaled = target_scaler.transform(y_reg_train.reshape(-1, 1)).ravel()
        y_reg_test_scaled = target_scaler.transform(y_reg_test.reshape(-1, 1)).ravel()

        yield PreparedFold(
            spec=spec,
            X_train=X_train,
            X_test=X_test,
            y_reg_train=y_reg_train,
            y_reg_test=y_reg_test,
            y_reg_train_scaled=y_reg_train_scaled,
            y_reg_test_scaled=y_reg_test_scaled,
            y_clf_train=y_clf_train,
            y_clf_test=y_clf_test,
            feature_pipeline=pipe,
            target_scaler=target_scaler,
            feature_names=list(feature_columns),
        )


def find_constant_features(
    X_train: np.ndarray,
    feature_names: list[str] = FEATURE_COLUMNS,
    tol: float = 1e-6,
) -> list[str]:
    """Return the names of features whose training-fold std is below `tol`.

    A "constant" feature has no variance within a particular training fold:
    every row takes the same value. sklearn's StandardScaler safely scales
    such columns to all-zeros (it sets the divisor to 1 rather than 0), so
    no exception is raised, but the column carries no information for that
    fold's models. This helper makes the situation auditable.
    """
    std = X_train.std(axis=0)
    return [feature_names[i] for i in np.where(std < tol)[0]]


# ----------------------------------------------------------------------------
# Summary table and manifest
# ----------------------------------------------------------------------------

def summarise_folds(
    train_pool: pd.DataFrame,
    folds: list[FoldSpec],
) -> pd.DataFrame:
    """Build a tidy per-fold summary table with size, dates, class counts."""
    rows = []
    for fold in folds:
        train_classes = train_pool.loc[fold.train_idx, TARGET_CLASSIFICATION].value_counts()
        test_classes = train_pool.loc[fold.test_idx, TARGET_CLASSIFICATION].value_counts()
        rows.append({
            "fold": fold.fold_id,
            "train_seasons": " + ".join(fold.train_seasons) if len(fold.train_seasons) <= 2
                              else f"{fold.train_seasons[0]} .. {fold.train_seasons[-1]}",
            "test_season": fold.test_season,
            "train_n": len(fold.train_idx),
            "test_n": len(fold.test_idx),
            "train_Normal": int(train_classes.get(0, 0)),
            "train_Delays": int(train_classes.get(1, 0)),
            "train_Div": int(train_classes.get(2, 0)),
            "test_Normal": int(test_classes.get(0, 0)),
            "test_Delays": int(test_classes.get(1, 0)),
            "test_Div": int(test_classes.get(2, 0)),
            "train_end": fold.train_end.date().isoformat(),
            "test_start": fold.test_start.date().isoformat(),
        })
    return pd.DataFrame(rows)


def save_manifest(
    folds: list[FoldSpec],
    train_pool: pd.DataFrame,
    holdout: pd.DataFrame,
    out_path: Path = DEFAULT_MANIFEST,
) -> None:
    """Write a JSON record of the split decisions for audit/reproducibility.

    Downstream Step 4-6 scripts do NOT depend on this file -- they call
    `forward_chaining_folds` directly -- but the manifest is useful as a
    paper-appendix artefact: it documents exactly which dates landed in
    which fold without anyone needing to re-run the splitter.
    """
    holdout_classes = holdout[TARGET_CLASSIFICATION].value_counts()
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "holdout_season": HOLDOUT_SEASON,
        "initial_training_seasons": INITIAL_TRAINING_SEASONS,
        "feature_columns": FEATURE_COLUMNS,
        "target_regression": TARGET_REGRESSION,
        "target_classification": TARGET_CLASSIFICATION,
        "class_names": CLASS_NAMES,
        "n_folds": len(folds),
        "train_pool_size": int(len(train_pool)),
        "holdout": {
            "season": HOLDOUT_SEASON,
            "size": int(len(holdout)),
            "start": holdout["date_npt"].min().date().isoformat(),
            "end": holdout["date_npt"].max().date().isoformat(),
            "class_counts": {
                CLASS_NAMES[c]: int(holdout_classes.get(c, 0)) for c in [0, 1, 2]
            },
        },
        "folds": [],
    }
    for f in folds:
        train_classes = train_pool.loc[f.train_idx, TARGET_CLASSIFICATION].value_counts()
        test_classes = train_pool.loc[f.test_idx, TARGET_CLASSIFICATION].value_counts()
        manifest["folds"].append({
            "fold_id": f.fold_id,
            "train_seasons": f.train_seasons,
            "test_season": f.test_season,
            "train_size": int(len(f.train_idx)),
            "test_size": int(len(f.test_idx)),
            "train_start": f.train_start.date().isoformat(),
            "train_end": f.train_end.date().isoformat(),
            "test_start": f.test_start.date().isoformat(),
            "test_end": f.test_end.date().isoformat(),
            "train_class_counts": {
                CLASS_NAMES[c]: int(train_classes.get(c, 0)) for c in [0, 1, 2]
            },
            "test_class_counts": {
                CLASS_NAMES[c]: int(test_classes.get(c, 0)) for c in [0, 1, 2]
            },
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Wrote CV manifest: {out_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct and inspect time-series CV splits for the VNKT fog forecasting task."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                        help="Path to Step 2 modelling table (default: data/processed/vnkt_modelling_table.parquet)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="Output path for JSON manifest (default: data/processed/cv_manifest.json)")
    args = parser.parse_args()

    df = load_modelling_table(args.parquet)
    train_pool, holdout = split_holdout(df)
    folds = forward_chaining_folds(train_pool)

    summary = summarise_folds(train_pool, folds)
    log.info("Per-fold summary:")
    print()
    print(summary.to_string(index=False))
    print()

    log.info(f"Holdout {HOLDOUT_SEASON}: {len(holdout)} rows  "
             f"(Normal={int((holdout[TARGET_CLASSIFICATION]==0).sum())}, "
             f"Delays={int((holdout[TARGET_CLASSIFICATION]==1).sum())}, "
             f"Diversions={int((holdout[TARGET_CLASSIFICATION]==2).sum())})")

    save_manifest(folds, train_pool, holdout, args.manifest)

    # Sanity check: verify the prepared-fold generator runs end to end and
    # produces well-shaped arrays with no NaN remaining after imputation.
    #
    # On unit variance: a feature column can legitimately have std=0 in a small
    # training window (e.g. `night_fog_observed` may be all-zero across a fold's
    # nights if the FG code never appeared in the 18:00-05:00 window). sklearn's
    # StandardScaler safely sets scale=1 for those columns, producing an all-zero
    # scaled column. We treat that case as expected and log it as a warning
    # rather than fail -- downstream RF/GP-ARD both deweight such features
    # automatically without any damage to the experiment.
    log.info("Running end-to-end iter_prepared_folds() sanity check...")
    for prepared in iter_prepared_folds(train_pool):
        fid = prepared.spec.fold_id
        assert not np.isnan(prepared.X_train).any(), \
            f"Fold {fid}: NaN in X_train after imputation"
        assert not np.isnan(prepared.X_test).any(), \
            f"Fold {fid}: NaN in X_test after imputation"
        assert prepared.X_train.shape[1] == len(FEATURE_COLUMNS), \
            f"Fold {fid}: wrong feature count"
        assert abs(prepared.X_train.mean(axis=0)).max() < 1e-6, \
            f"Fold {fid}: X_train not zero-mean after scaling"

        # Per-feature variance check: each column must be either unit-variance
        # (normal case) or zero-variance (constant column after sklearn's
        # safe-divide). Anything else is a genuine scaling bug.
        train_std = prepared.X_train.std(axis=0)
        is_zero_var = train_std < 1e-6
        is_unit_var = np.abs(train_std - 1.0) < 1e-6
        bad = ~(is_zero_var | is_unit_var)
        assert not bad.any(), (
            f"Fold {fid}: features with unexpected std (neither 0 nor 1): "
            f"{[(FEATURE_COLUMNS[i], float(train_std[i])) for i in np.where(bad)[0]]}"
        )
        if is_zero_var.any():
            const_names = [FEATURE_COLUMNS[i] for i in np.where(is_zero_var)[0]]
            log.warning(
                f"Fold {fid}: zero-variance training features {const_names} "
                f"-- harmless (sklearn safe-divides; RF/GP-ARD will deweight them) "
                f"but these features carry no signal in this fold's training window."
            )
    log.info("  all folds passed sanity checks")


if __name__ == "__main__":
    main()
