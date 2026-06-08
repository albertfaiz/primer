"""Model registry.

Each candidate model is described by *capabilities* (not just a name), so the
rule engine reasons about properties ("captures nonlinearity", "scales to large
n") rather than hard-coding model names everywhere. Adding a new estimator — or a
whole new family — is a one-entry change here, which is exactly the extension
point that keeps the door open for future layers.

Capability fields are in [0, 1] (graded) or bool. Values reflect broad empirical
consensus for tabular data; they are deliberately editable and, down the road,
learnable from an OpenML-scale corpus.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelSpec:
    key: str
    display: str
    family: str
    tasks: List[str]                       # "regression" / "classification" / "unsupervised"
    caps: Dict[str, float] = field(default_factory=dict)
    library_hint: str = ""                 # where the user actually gets it
    notes: str = ""

    def cap(self, name: str) -> float:
        return float(self.caps.get(name, 0.0))


# capability vocabulary (documented for rule authors)
#   nonlinearity         : models curved / threshold relationships
#   interactions         : captures feature interactions without manual crosses
#   native_categorical   : handles categoricals without one-hot blow-up
#   scales_large_n       : trains fast as rows grow
#   low_n_friendly       : works with few rows without overfitting
#   high_dim_friendly    : tolerates p >> n / wide data
#   robust_missing       : tolerates missing values natively
#   robust_outliers      : insensitive to outliers / heavy tails
#   interpretable        : easy to explain / inspect
#   fast_train           : cheap to fit overall
#   needs_scaling        : sensitive to feature scaling (cost, not benefit)
#   class_weighting      : supports class weights for imbalance
#   local_structure      : exploits local / neighbourhood structure

_REG = "regression"
_CLF = "classification"
_UNS = "unsupervised"


_MODELS: List[ModelSpec] = [
    # ---- linear -------------------------------------------------------------
    ModelSpec("ridge", "Ridge / Linear Regression", "linear", [_REG],
              caps=dict(interpretable=0.9, fast_train=0.95, scales_large_n=0.9,
                        high_dim_friendly=0.7, low_n_friendly=0.8, needs_scaling=0.7,
                        nonlinearity=0.05, interactions=0.05, robust_outliers=0.2),
              library_hint="scikit-learn (Ridge / ElasticNet)"),
    ModelSpec("elasticnet", "ElasticNet (L1+L2)", "linear", [_REG],
              caps=dict(interpretable=0.85, fast_train=0.9, scales_large_n=0.85,
                        high_dim_friendly=0.95, low_n_friendly=0.85, needs_scaling=0.7,
                        nonlinearity=0.05, interactions=0.05),
              library_hint="scikit-learn (ElasticNet)",
              notes="sparse coefficients — strong when p is large / many irrelevant features"),
    ModelSpec("logreg", "Logistic Regression", "linear", [_CLF],
              caps=dict(interpretable=0.9, fast_train=0.95, scales_large_n=0.9,
                        high_dim_friendly=0.8, low_n_friendly=0.8, needs_scaling=0.7,
                        class_weighting=1.0, nonlinearity=0.05, interactions=0.05),
              library_hint="scikit-learn (LogisticRegression)"),

    # ---- gradient-boosted trees --------------------------------------------
    ModelSpec("lightgbm", "LightGBM", "gbt", [_REG, _CLF],
              caps=dict(nonlinearity=0.95, interactions=0.95, native_categorical=0.9,
                        scales_large_n=0.98, robust_missing=0.9, robust_outliers=0.7,
                        fast_train=0.9, class_weighting=1.0, low_n_friendly=0.45,
                        interpretable=0.4),
              library_hint="lightgbm",
              notes="default first choice on large tabular data; very fast"),
    ModelSpec("xgboost", "XGBoost", "gbt", [_REG, _CLF],
              caps=dict(nonlinearity=0.95, interactions=0.95, native_categorical=0.6,
                        scales_large_n=0.9, robust_missing=0.9, robust_outliers=0.7,
                        fast_train=0.75, class_weighting=1.0, low_n_friendly=0.5,
                        interpretable=0.4),
              library_hint="xgboost",
              notes="strong accuracy, robust; slightly slower than LightGBM"),
    ModelSpec("catboost", "CatBoost", "gbt", [_REG, _CLF],
              caps=dict(nonlinearity=0.95, interactions=0.95, native_categorical=1.0,
                        scales_large_n=0.85, robust_missing=0.9, robust_outliers=0.7,
                        fast_train=0.7, class_weighting=1.0, low_n_friendly=0.6,
                        interpretable=0.4),
              library_hint="catboost",
              notes="best with many / high-cardinality categoricals; minimal tuning"),

    # ---- bagged trees -------------------------------------------------------
    ModelSpec("rf", "Random Forest", "forest", [_REG, _CLF],
          caps=dict(nonlinearity=0.85, interactions=0.85, native_categorical=0.4,
                    scales_large_n=0.6, robust_missing=0.3, robust_outliers=0.9, # <-- Elevated
                    fast_train=0.6, class_weighting=1.0, low_n_friendly=0.7,
                    interpretable=0.5),
          library_hint="scikit-learn (RandomForest)",
          notes="robust, low-tuning baseline; exceptional at resisting noise and variance"),

    # ---- instance / kernel --------------------------------------------------
    ModelSpec("knn", "k-Nearest Neighbours", "instance", [_REG, _CLF],
              caps=dict(nonlinearity=0.8, interactions=0.6, local_structure=1.0,
                        needs_scaling=1.0, scales_large_n=0.2, low_n_friendly=0.6,
                        high_dim_friendly=0.15, fast_train=0.5, interpretable=0.4),
              library_hint="scikit-learn (KNeighbors)",
              notes="good when local structure dominates and n is modest"),
    ModelSpec("svm_rbf", "SVM (RBF kernel)", "kernel", [_REG, _CLF],
              caps=dict(nonlinearity=0.9, interactions=0.7, needs_scaling=1.0,
                        scales_large_n=0.1, low_n_friendly=0.75, high_dim_friendly=0.6,
                        class_weighting=1.0, fast_train=0.3, interpretable=0.2),
              library_hint="scikit-learn (SVC / SVR)",
              notes="strong on small/medium datasets; scales poorly past ~10k rows"),

    # ---- probabilistic ------------------------------------------------------
    ModelSpec("gnb", "Gaussian Naive Bayes", "bayes", [_CLF],
              caps=dict(fast_train=0.98, scales_large_n=0.9, low_n_friendly=0.85,
                        high_dim_friendly=0.7, interpretable=0.7, nonlinearity=0.3,
                        interactions=0.0),
              library_hint="scikit-learn (GaussianNB)",
              notes="fast, strong baseline when features are roughly conditionally independent"),

    # ---- neural -------------------------------------------------------------
    ModelSpec("mlp", "MLP / Neural Net", "neural", [_REG, _CLF],
              caps=dict(nonlinearity=0.95, interactions=0.9, needs_scaling=1.0,
                        scales_large_n=0.8, low_n_friendly=0.2, high_dim_friendly=0.6,
                        fast_train=0.4, interpretable=0.15),
              library_hint="scikit-learn (MLP) / PyTorch",
              notes="rarely beats boosting on tabular; needs lots of data and tuning"),

    # ---- unsupervised -------------------------------------------------------
    ModelSpec("kmeans", "K-Means", "clustering", [_UNS],
              caps=dict(scales_large_n=0.9, needs_scaling=1.0, fast_train=0.9,
                        interpretable=0.7),
              library_hint="scikit-learn (KMeans)",
              notes="convex, similar-sized clusters; pick k via silhouette/elbow"),
    ModelSpec("gmm", "Gaussian Mixture", "clustering", [_UNS],
              caps=dict(scales_large_n=0.6, needs_scaling=1.0, fast_train=0.6,
                        interpretable=0.6),
              library_hint="scikit-learn (GaussianMixture)",
              notes="soft, elliptical clusters; gives probabilistic membership"),
    ModelSpec("hdbscan", "HDBSCAN", "clustering", [_UNS],
              caps=dict(scales_large_n=0.5, needs_scaling=0.8, fast_train=0.5,
                        local_structure=1.0, interpretable=0.5),
              library_hint="hdbscan",
              notes="density-based; finds arbitrary shapes and flags noise; no preset k"),
    ModelSpec("pca", "PCA (dimensionality reduction)", "projection", [_UNS],
              caps=dict(scales_large_n=0.9, needs_scaling=1.0, fast_train=0.9,
                        interpretable=0.6, high_dim_friendly=0.9),
              library_hint="scikit-learn (PCA)",
              notes="linear structure / compression / de-correlation; pair with clustering"),
    ModelSpec("isoforest", "Isolation Forest (anomaly)", "anomaly", [_UNS],
              caps=dict(scales_large_n=0.85, fast_train=0.8, robust_outliers=1.0,
                        interpretable=0.4),
              library_hint="scikit-learn (IsolationForest)",
              notes="use when the goal is outlier / anomaly detection rather than grouping"),
]

REGISTRY: Dict[str, ModelSpec] = {m.key: m for m in _MODELS}


def models_for(task_kind: str) -> List[ModelSpec]:
    return [m for m in _MODELS if task_kind in m.tasks]


def register(model: ModelSpec) -> None:
    """Add or override a model (extension hook)."""
    REGISTRY[model.key] = model
    if model not in _MODELS:
        _MODELS.append(model)
