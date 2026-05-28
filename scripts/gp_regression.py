"""
scripts/gp_regression.py
========================
Step 5 of the VNKT Winter Fog Forecasting project.

Trains a Gaussian Process regressor on every CV fold and writes out-of-fold
predictions for direct calibration comparison with the Step 4 Random Forest.

MODEL
-----
- Kernel: Matérn-5/2 with Automatic Relevance Determination (one length-scale
  per feature, 19 in total). Matérn-5/2 is the standard differentiable-twice
  kernel for smooth spatio-temporal phenomena -- smoother than Matérn-3/2,
  less aggressive than Squared Exponential -- and ARD turns the kernel into
  an interpretable feature-importance device for the paper.
- Mean function: zero (acceptable since the target is standardised in
  log-space, so the prior mean is centred).
- Likelihood: Gaussian. Standard for `gpflow.models.GPR`.
- Inference: exact conditioning (n <= 1259 across all folds, well within the
  O(n^3) regime for desktop CPU).

TARGET TRANSFORMATION
---------------------
log1p -> StandardScaler. Rationale:
  1. visibility has a hard physical floor at 0 metres; in raw-metre space a
     Gaussian posterior assigns probability mass to physically impossible
     values. Log-transform pushes the posterior onto positive metres after
     back-transformation.
  2. the metre-scale distribution is heavy-right-tailed (cluster of low-vis
     fog days, fat upper tail of clear days); log compresses the tail,
     making the Gaussian-likelihood assumption far more defensible.
  3. point estimate in metres = median of predictive distribution
     = exp(mean_log_space) - 1. This is the natural Bayesian point for an
     L1 loss; for L2 the optimum would be exp(mu + sigma^2/2) - 1 but that
     can blow up under high-variance predictions, so we accept the slight
     RMSE penalty in exchange for robustness. We document this in the
     paper.

PREDICTIVE INTERVALS
--------------------
Quantiles transform cleanly under monotonic functions. We compute 90% PI as
  z_lo, z_hi = mean_z +/- 1.6449 * std_z   (in standardised log-space)
then inverse-transform each endpoint separately. The resulting PI in metres
is correctly asymmetric (longer upper tail) for the log-Normal posterior.

OUTPUTS
-------
  data/processed/gp_oof_predictions.parquet
  data/processed/step5_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

# Silence TensorFlow's GPU-not-found notices on Ryzen CPU + AMD GPU.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import gpflow  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cv_splits import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    load_modelling_table,
    split_holdout,
    iter_prepared_folds,
    setup_logging,
)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

DEFAULT_OOF_PATH = PROJECT_ROOT / "data" / "processed" / "gp_oof_predictions.parquet"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "step5_metrics.json"

PI_Z = 1.6448536269514722  # 90% Gaussian PI z-value

# GP optimiser config. L-BFGS-B via scipy is GPflow's recommended path for
# small/medium-sized exact GPs. maxiter=500 is empirically sufficient: on the
# smallest fold (n=204) the L-BFGS-B optimiser converges by iter 268 with
# success=True; on larger folds the headroom prevents premature termination.
# Cost per fold scales as O(maxiter * n^3) where n is the training-fold size.
SCIPY_MAXITER = 500
SCIPY_OPTIONS = {"maxiter": SCIPY_MAXITER, "disp": False}

# Hyperparameter initial values (in standardised X / log-target space)
INIT_LENGTHSCALE = 1.0       # one per feature; ARD optimises them away from this
INIT_SIGNAL_VARIANCE = 1.0   # since y is standardised, signal variance ~ 1
INIT_NOISE_VARIANCE = 0.1    # 10% noise floor; far from zero to avoid Cholesky issues

# Numerical stability: a small jitter is added to the kernel diagonal via
# gpflow's default. We do NOT change `gpflow.config.default_jitter()` here.

log = setup_logging()
log.name = "gp_regression"


# ----------------------------------------------------------------------------
# Target transform: log1p + StandardScaler
# ----------------------------------------------------------------------------

@dataclass
class LogTargetTransform:
    """Pipeline: y_metres -> log1p(y) -> StandardScaler.

    The transform is fitted on training-fold targets only. `inverse_quantile`
    correctly back-transforms predictive quantiles (which is what we need
    for PIs) rather than naively inverting mean and variance separately --
    quantiles transform cleanly under monotonic functions, distributions
    do not."""
    scaler: StandardScaler

    @classmethod
    def fit(cls, y_metres: np.ndarray) -> "LogTargetTransform":
        scaler = StandardScaler().fit(np.log1p(y_metres).reshape(-1, 1))
        return cls(scaler=scaler)

    def transform(self, y_metres: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.log1p(y_metres).reshape(-1, 1)).ravel()

    def inverse_quantile(self, z: np.ndarray) -> np.ndarray:
        """Inverse-transform a standardised-log-space quantile back to metres.

        Use this for any quantile-of-the-predictive-distribution -- mean,
        median, PI lower, PI upper. expm1 is safe: it returns >= -1, and we
        clip at 0 since visibility is non-negative."""
        log_q = self.scaler.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()
        return np.maximum(np.expm1(log_q), 0.0)


# ----------------------------------------------------------------------------
# Per-fold GP fit
# ----------------------------------------------------------------------------

@dataclass
class FoldGP:
    fold_id: int
    test_season: str
    median_m: np.ndarray
    pi90_lo_m: np.ndarray
    pi90_hi_m: np.ndarray
    pred_std_logspace: np.ndarray   # standardised log-space std (for diagnostics)
    lengthscales: np.ndarray        # (n_features,) ARD lengthscales (standardised X)
    signal_variance: float
    noise_variance: float
    elbo: float                     # actually log marginal likelihood; "elbo" is loose
    n_train: int
    # Calibration curve: nominal level alpha (e.g. 0.9 = 90% PI) vs empirical
    # coverage on this fold's test set. Computed in-loop while the per-fold
    # transform is still in scope. The headline aggregate curve in the metrics
    # JSON is a weighted average across folds.
    calib_nominal: np.ndarray       # shape (n_levels,)
    calib_empirical: np.ndarray     # shape (n_levels,)


def fit_gp_one_fold(
    X_train: np.ndarray,
    y_train_z: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, gpflow.models.GPR]:
    """Fit a Matern52-ARD GPR on one fold and return predictive (mean, var).

    Returns (mean_z, pred_var_z, model) where the variance is the predictive
    variance INCLUDING noise (i.e. for a new observation y*, not the latent
    f*). This is the correct quantity for an observation-level 90% PI.
    """
    kernel = gpflow.kernels.Matern52(
        variance=INIT_SIGNAL_VARIANCE,
        lengthscales=np.full(n_features, INIT_LENGTHSCALE, dtype=np.float64),
    )
    model = gpflow.models.GPR(
        data=(
            tf.convert_to_tensor(X_train, dtype=tf.float64),
            tf.convert_to_tensor(y_train_z.reshape(-1, 1), dtype=tf.float64),
        ),
        kernel=kernel,
        mean_function=None,
        noise_variance=INIT_NOISE_VARIANCE,
    )
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss,
        model.trainable_variables,
        options=SCIPY_OPTIONS,
    )
    # predict_y returns predictive mean and variance INCLUDING Gaussian noise.
    mean_z, pred_var_z = model.predict_y(
        tf.convert_to_tensor(X_test, dtype=tf.float64)
    )
    return mean_z.numpy().ravel(), pred_var_z.numpy().ravel(), model


def train_and_evaluate(train_pool: pd.DataFrame) -> tuple[list[FoldGP], pd.DataFrame]:
    """Run GP regression on every CV fold; return per-fold artefacts + OOF frame."""
    results: list[FoldGP] = []
    oof_chunks: list[pd.DataFrame] = []
    total_t0 = time.time()

    for prepared in iter_prepared_folds(train_pool):
        spec = prepared.spec
        log.info(
            f"Fold {spec.fold_id} ({spec.test_season}): "
            f"train n={prepared.X_train.shape[0]}, test n={prepared.X_test.shape[0]} "
            f"-- fitting GP..."
        )
        fold_t0 = time.time()

        # Re-derive the target transform per fold from RAW metres (Step 3's
        # target_scaler is linear-on-metres; we want log-on-metres here).
        transform = LogTargetTransform.fit(prepared.y_reg_train)
        y_train_z = transform.transform(prepared.y_reg_train)

        mean_z, pred_var_z, model = fit_gp_one_fold(
            prepared.X_train,
            y_train_z,
            prepared.X_test,
            n_features=len(FEATURE_COLUMNS),
        )
        pred_std_z = np.sqrt(pred_var_z)

        # 90% PI quantiles in z-space -> back to metres via the safe path
        median_m = transform.inverse_quantile(mean_z)
        pi90_lo_m = transform.inverse_quantile(mean_z - PI_Z * pred_std_z)
        pi90_hi_m = transform.inverse_quantile(mean_z + PI_Z * pred_std_z)

        # Pull hyperparameters for length-scale reporting
        ls = model.kernel.lengthscales.numpy().astype(float)
        sv = float(model.kernel.variance.numpy())
        nv = float(model.likelihood.variance.numpy())
        # Negative training loss = log marginal likelihood
        lml = float(-model.training_loss().numpy())

        # Per-fold calibration curve (multiple nominal levels)
        nominal, empirical = fold_coverage_curve(
            prepared.y_reg_test, mean_z, pred_std_z, transform,
        )
        fold_elapsed = time.time() - fold_t0

        results.append(FoldGP(
            fold_id=spec.fold_id,
            test_season=spec.test_season,
            median_m=median_m,
            pi90_lo_m=pi90_lo_m,
            pi90_hi_m=pi90_hi_m,
            pred_std_logspace=pred_std_z,
            lengthscales=ls,
            signal_variance=sv,
            noise_variance=nv,
            elbo=lml,
            n_train=int(prepared.X_train.shape[0]),
            calib_nominal=nominal,
            calib_empirical=empirical,
        ))

        # OOF row slice
        test_dates = train_pool.loc[spec.test_idx, "date_npt"].values
        oof_chunks.append(pd.DataFrame({
            "date_npt": test_dates,
            "fold_id": spec.fold_id,
            "test_season": spec.test_season,
            "true_vis_m": prepared.y_reg_test,
            "true_class": prepared.y_clf_test,
            "gp_median_vis_m": median_m,
            "gp_pi90_lo_m": pi90_lo_m,
            "gp_pi90_hi_m": pi90_hi_m,
            "gp_pred_std_logspace": pred_std_z,
        }))

        log.info(
            f"  Fold {spec.fold_id}: {fold_elapsed:.1f}s, lml={lml:.1f}, "
            f"sigma_n^2={nv:.3f}, sigma_f^2={sv:.3f}, "
            f"max lengthscale={ls.max():.2f}, min lengthscale={ls.min():.2f}"
        )

    oof = pd.concat(oof_chunks, ignore_index=True)
    log.info(f"All folds complete: total GP training {time.time() - total_t0:.1f}s")
    return results, oof


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

@dataclass
class RegressionMetrics:
    mae_m: float
    rmse_m: float
    r2: float
    pi90_coverage: float
    pi90_mean_width_m: float
    pi90_median_width_m: float
    n: int


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, pi_lo: np.ndarray, pi_hi: np.ndarray,
) -> RegressionMetrics:
    inside = (y_true >= pi_lo) & (y_true <= pi_hi)
    widths = pi_hi - pi_lo
    return RegressionMetrics(
        mae_m=float(mean_absolute_error(y_true, y_pred)),
        rmse_m=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        pi90_coverage=float(inside.mean()),
        pi90_mean_width_m=float(widths.mean()),
        pi90_median_width_m=float(np.median(widths)),
        n=int(len(y_true)),
    )


CALIB_LEVELS = np.linspace(0.1, 0.9, 9)  # nominal coverage levels for calibration curves


def fold_coverage_curve(
    y_true_m: np.ndarray,
    mean_z: np.ndarray,
    std_z: np.ndarray,
    transform: LogTargetTransform,
    levels: np.ndarray = CALIB_LEVELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibration curve for one fold's predictions.

    For each nominal level alpha, compute the empirical coverage of the
    central alpha-quantile predictive interval by inverse-transforming the
    z-space quantiles back to metres and counting how many true values fall
    inside. A perfectly calibrated GP traces y = x.
    """
    from scipy.stats import norm
    nominal = np.asarray(levels, dtype=float)
    empirical = np.empty_like(nominal)
    for i, alpha in enumerate(nominal):
        z_a = norm.ppf(0.5 + alpha / 2)
        lo = transform.inverse_quantile(mean_z - z_a * std_z)
        hi = transform.inverse_quantile(mean_z + z_a * std_z)
        empirical[i] = ((y_true_m >= lo) & (y_true_m <= hi)).mean()
    return nominal, empirical


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def aggregate_metrics(
    results: list[FoldGP],
    oof: pd.DataFrame,
    train_pool: pd.DataFrame,
) -> dict:
    """Build a metrics dictionary suitable for JSON serialisation.

    Includes per-fold + aggregate-OOF regression metrics, plus per-fold ARD
    length-scales and their mean/std across folds for the ARD bar chart.
    """
    out: dict = {"per_fold": [], "aggregate": {}, "ard": {}}

    # Per-fold
    for r in results:
        # Reconstruct y_true for this fold from OOF
        mask = oof["fold_id"] == r.fold_id
        y_true = oof.loc[mask, "true_vis_m"].to_numpy()
        m = compute_regression_metrics(
            y_true, r.median_m, r.pi90_lo_m, r.pi90_hi_m,
        )
        out["per_fold"].append({
            "fold_id": r.fold_id,
            "test_season": r.test_season,
            "n_train": r.n_train,
            "log_marginal_likelihood": r.elbo,
            "signal_variance": r.signal_variance,
            "noise_variance": r.noise_variance,
            "regression": m.__dict__,
        })

    # Aggregate on concatenated OOF
    agg = compute_regression_metrics(
        oof["true_vis_m"].to_numpy(),
        oof["gp_median_vis_m"].to_numpy(),
        oof["gp_pi90_lo_m"].to_numpy(),
        oof["gp_pi90_hi_m"].to_numpy(),
    )
    out["aggregate"]["gp_regression"] = agg.__dict__

    # ARD length-scales (one row per fold, one column per feature)
    ls_stack = np.stack([r.lengthscales for r in results])  # (n_folds, n_features)
    out["ard"]["features"] = list(FEATURE_COLUMNS)
    out["ard"]["lengthscales_per_fold"] = ls_stack.tolist()
    out["ard"]["lengthscales_mean"] = ls_stack.mean(axis=0).tolist()
    out["ard"]["lengthscales_std"] = ls_stack.std(axis=0).tolist()
    # Inverse-lengthscale = "relevance" (small ls -> small input change moves
    # output a lot -> high relevance). Reported alongside for the paper's
    # interpretation paragraph.
    relevance = 1.0 / ls_stack
    out["ard"]["relevance_mean"] = relevance.mean(axis=0).tolist()
    out["ard"]["relevance_std"] = relevance.std(axis=0).tolist()

    # Calibration curve: per-fold + weighted-by-n_test aggregate
    n_per_fold = np.array([
        int((oof["fold_id"] == r.fold_id).sum()) for r in results
    ])
    calib_stack = np.stack([r.calib_empirical for r in results])  # (n_folds, n_levels)
    weighted_emp = (calib_stack * n_per_fold[:, None]).sum(axis=0) / n_per_fold.sum()
    out["calibration"] = {
        "nominal_levels": results[0].calib_nominal.tolist(),
        "per_fold_empirical": calib_stack.tolist(),
        "weighted_empirical": weighted_emp.tolist(),
        "per_fold_n_test": n_per_fold.tolist(),
    }

    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Matern-5/2 + ARD GP regressor on every CV fold."
    )
    parser.add_argument("--oof-out", type=Path, default=DEFAULT_OOF_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    df = load_modelling_table()
    train_pool, _holdout = split_holdout(df)

    log.info("Training GP regressors across all 8 forward-chaining CV folds...")
    results, oof = train_and_evaluate(train_pool)
    metrics = aggregate_metrics(results, oof, train_pool)

    args.oof_out.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.oof_out, index=False)
    log.info(f"Wrote OOF predictions ({len(oof)} rows): {args.oof_out}")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2))
    log.info(f"Wrote metrics JSON: {args.metrics_out}")

    # Headline
    agg = metrics["aggregate"]["gp_regression"]
    print()
    print("=" * 72)
    print("GP HEADLINE METRICS (out-of-fold, concatenated across 8 folds)")
    print("=" * 72)
    print(f"  MAE             : {agg['mae_m']:.0f} m")
    print(f"  RMSE            : {agg['rmse_m']:.0f} m")
    print(f"  R^2             : {agg['r2']:.3f}")
    print(f"  90% PI coverage : {agg['pi90_coverage']:.3f}   (target: 0.900)")
    print(f"  90% PI width    : mean={agg['pi90_mean_width_m']:.0f} m, "
          f"median={agg['pi90_median_width_m']:.0f} m")
    print()
    print("Per-fold log marginal likelihoods (higher = better fit):")
    for r in results:
        ls = r.lengthscales
        print(f"  Fold {r.fold_id} ({r.test_season}): lml={r.elbo:>8.1f}, "
              f"sigma_n^2={r.noise_variance:.3f}, "
              f"ARD lengthscales min/median/max = "
              f"{ls.min():.2f} / {np.median(ls):.2f} / {ls.max():.2f}")


if __name__ == "__main__":
    main()
