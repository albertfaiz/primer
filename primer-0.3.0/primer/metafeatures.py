"""Layer 4a — metafeature extraction.

Computes the dataset descriptors that the rule engine reasons over: shape,
dimensionality ratio, missingness, distributional shape (skew/kurtosis/outliers),
correlation structure / multicollinearity, and feature↔target dependence
(mutual information). Also builds the numeric design matrix consumed by the
landmark models so the work of encoding happens exactly once.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import numpyx as nx
from .types import DatasetProfile, TaskKind, TaskSpec


def build_numeric_matrix(df: "pd.DataFrame", profile: DatasetProfile
                         ) -> Tuple[np.ndarray, list]:
    """Numeric features as-is; categoricals/booleans ordinal-encoded.

    Light encoding on purpose — landmarks only need *signal*, not a production
    feature pipeline. Returns (matrix, used_feature_names).
    """
    cols, names = [], []
    for name in profile.numeric_features:
        cols.append(pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float))
        names.append(name)
    for name in profile.categorical_features:
        codes = df[name].astype("category").cat.codes.to_numpy(dtype=float)
        codes[codes < 0] = np.nan  # missing -> NaN so imputation handles it
        cols.append(codes)
        names.append(name)
    if not cols:
        return np.empty((len(df), 0)), names
    return np.column_stack(cols), names


def _encode_target(df: "pd.DataFrame", target: str, task: TaskSpec) -> np.ndarray:
    y = df[target]
    if task.kind is TaskKind.REGRESSION:
        return pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    return y.astype("category").cat.codes.to_numpy()  # integer class codes


def extract_metafeatures(df: "pd.DataFrame", profile: DatasetProfile,
                         task: TaskSpec) -> Dict[str, Any]:
    n = profile.n_rows
    p = len(profile.feature_names)
    mf: Dict[str, Any] = {
        "n_rows": n,
        "n_features": p,
        "n_numeric": len(profile.numeric_features),
        "n_categorical": len(profile.categorical_features),
        "n_datetime": len(profile.datetime_features),
        "n_text": len(profile.text_features),
        "n_to_p_ratio": float(n / p) if p else float("inf"),
        "frac_categorical": float(len(profile.categorical_features) / p) if p else 0.0,
        "frac_numeric": float(len(profile.numeric_features) / p) if p else 0.0,
        "has_datetime": len(profile.datetime_features) > 0,
        "has_text": len(profile.text_features) > 0,
    }

    feat_cols = [c for c in profile.columns if c.role.value == "feature"]
    mf["overall_missing_frac"] = float(
        np.mean([c.missing_frac for c in feat_cols])) if feat_cols else 0.0
    mf["max_missing_frac"] = float(
        np.max([c.missing_frac for c in feat_cols])) if feat_cols else 0.0
    mf["n_high_cardinality_cat"] = sum(
        1 for c in profile.columns
        if c.ctype.value == "categorical" and c.n_unique > 20)

    # ---- distributional shape over numeric features ----------------------
    if profile.numeric_features:
        skews, kurts, cvs, outs = [], [], [], []
        for name in profile.numeric_features:
            x = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            skews.append(abs(nx.skewness(x)))
            kurts.append(nx.kurtosis_excess(x))
            cv = nx.coef_variation(x)
            if np.isfinite(cv):
                cvs.append(cv)
            outs.append(nx.outlier_fraction(x))
        mf["mean_abs_skew"] = float(np.mean(skews))
        mf["max_abs_skew"] = float(np.max(skews))
        mf["mean_excess_kurtosis"] = float(np.mean(kurts))
        mf["mean_coef_variation"] = float(np.mean(cvs)) if cvs else 0.0
        mf["mean_outlier_frac"] = float(np.mean(outs))
        mf["frac_skewed_features"] = float(np.mean([s > 1.0 for s in skews]))
    else:
        mf.update(mean_abs_skew=0.0, max_abs_skew=0.0, mean_excess_kurtosis=0.0,
                  mean_coef_variation=0.0, mean_outlier_frac=0.0,
                  frac_skewed_features=0.0)

    # ---- correlation structure / multicollinearity -----------------------
    if len(profile.numeric_features) >= 2:
        Xnum = np.column_stack([
            pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            for c in profile.numeric_features])
        C = nx.correlation_matrix(Xnum)
        off = C[~np.eye(C.shape[0], dtype=bool)]
        mf["mean_abs_correlation"] = float(np.mean(np.abs(off)))
        mf["max_abs_correlation"] = float(np.max(np.abs(off)))
        mf["frac_highly_correlated_pairs"] = float(np.mean(np.abs(off) > 0.8))
        mf["condition_number"] = nx.condition_number(Xnum)
        # intrinsic (effective) dimensionality from the PCA/SVD spectrum
        k95, idr, pr = nx.intrinsic_dimensionality(Xnum)
        mf["intrinsic_dim_95"] = k95
        mf["intrinsic_dim_ratio"] = idr
        mf["participation_ratio"] = pr
    else:
        mf.update(mean_abs_correlation=0.0, max_abs_correlation=0.0,
                  frac_highly_correlated_pairs=0.0, condition_number=1.0,
                  intrinsic_dim_95=len(profile.numeric_features),
                  intrinsic_dim_ratio=1.0, participation_ratio=1.0)

    # ---- feature <-> target dependence -----------------------------------
    if profile.target is not None and task.kind in (TaskKind.REGRESSION,
                                                     TaskKind.CLASSIFICATION):
        y = _encode_target(df, profile.target, task)
        mis = []
        for name in profile.numeric_features:
            x = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            mi = nx.mutual_info_cc(x, y) if task.kind is TaskKind.REGRESSION \
                else nx.mutual_info_cd(x, y)
            mis.append(mi)
        for name in profile.categorical_features:
            x = df[name].astype("category").cat.codes.to_numpy()
            mi = nx.mutual_info_cd(y.astype(float), x) if task.kind is TaskKind.REGRESSION \
                else nx.mutual_info_dd(x, y)
            mis.append(mi)
        if mis:
            mf["mean_target_mi"] = float(np.mean(mis))
            mf["max_target_mi"] = float(np.max(mis))
            mf["n_informative_features"] = int(np.sum(np.array(mis) > 0.05))
            mf["frac_informative_features"] = float(np.mean(np.array(mis) > 0.05))

        if task.kind is TaskKind.CLASSIFICATION:
            mf["target_entropy"] = nx.entropy_discrete(y)
            if task.class_balance:
                props = np.array(list(task.class_balance.values()))
                mf["min_class_frac"] = float(props.min())
                mf["imbalance_ratio"] = float(props.max() / max(props.min(), nx.EPS))
            # geometric class overlap: can the classes even be separated?
            if profile.numeric_features:
                Xsep = np.column_stack([
                    pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                    for c in profile.numeric_features])
                mf["class_separation"] = nx.class_separation(Xsep, y)
        else:
            mf["target_skew"] = nx.skewness(y)
            mf["target_kurtosis"] = nx.kurtosis_excess(y)

        # sequential dependency in the target (random CV would leak the future)
        mf["target_autocorr"] = nx.lag1_autocorrelation(y.astype(float))

    return mf
