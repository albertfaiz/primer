"""Smoke + behaviour tests across several synthetic regimes.

Run: python test_primer.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import primer


def make_linear_reg(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = X @ np.array([3, -2, 1.5, 0, 0, 0.0]) + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    df["target"] = y
    return df


def make_nonlinear_reg(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 5))
    y = (np.sin(X[:, 0] * 2) * 3 + (X[:, 1] > 0) * 4 * X[:, 2]
         + rng.normal(scale=0.4, size=n))
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    df["target"] = y
    return df


def make_imbalanced_clf(n=4000, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    logit = X @ np.r_[np.array([2.0, -1.5, 1.0]), np.zeros(5)] - 4.5
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["label"] = y
    return df


def make_categorical_clf(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    city = rng.choice([f"city_{i}" for i in range(40)], size=n)   # high cardinality
    plan = rng.choice(["basic", "pro", "max"], size=n)
    num = rng.normal(size=n)
    base = (plan == "pro") * 1.0 + (plan == "max") * 2.0 + num
    y = (base + rng.normal(scale=0.5, size=n) > 1.0).astype(int)
    df = pd.DataFrame({"city": city, "plan": plan, "usage": num, "label": y})
    return df


def make_leakage_reg(n=2000, seed=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
    df["target"] = y
    df["leaky"] = y * 1.0001 + rng.normal(scale=1e-3, size=n)   # near-perfect copy
    df["row_id"] = np.arange(n)                                  # id column
    return df


def make_unsupervised(n=2000, seed=5):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[0, 0], scale=0.5, size=(n // 2, 2))
    b = rng.normal(loc=[5, 5], scale=0.5, size=(n // 2, 2))
    X = np.vstack([a, b])
    return pd.DataFrame(X, columns=["dim0", "dim1"])


def show(title, df, target=None):
    print("\n\n" + "#" * 70)
    print("# " + title)
    print("#" * 70)
    rep = primer.analyze(df, target=target)
    print(rep.render())
    return rep


if __name__ == "__main__":
    r1 = show("LINEAR REGRESSION (linear should win / be near top)", make_linear_reg(), "target")
    r2 = show("NONLINEAR REGRESSION (trees/boosting should win)", make_nonlinear_reg(), "target")
    r3 = show("IMBALANCED CLASSIFICATION", make_imbalanced_clf(), "label")
    r4 = show("CATEGORICAL-HEAVY CLASSIFICATION (CatBoost/LightGBM favoured)", make_categorical_clf(), "label")
    r5 = show("REGRESSION WITH LEAKAGE + ID COLUMN", make_leakage_reg(), "target")
    r6 = show("UNSUPERVISED (no target)", make_unsupervised())

    print("\n\n" + "=" * 70)
    print("BEHAVIOUR ASSERTIONS")
    print("=" * 70)

    def family_rank(rep, family):
        for r in rep.recommendations:
            if r.family == family:
                return r.rank
        return 99

    checks = []
    # linear should rank linear family in top 3
    checks.append(("linear family top-3 on linear data", family_rank(r1, "linear") <= 3))
    # nonlinear: gbt should beat linear
    checks.append(("gbt outranks linear on nonlinear data",
                   family_rank(r2, "gbt") < family_rank(r2, "linear")))
    # categorical-heavy: a model with native categorical (gbt) should be #1 family
    checks.append(("gbt top-1 on categorical-heavy data", family_rank(r4, "gbt") == 1))
    # leakage detected
    checks.append(("leakage flagged",
                   any(d.name == "possible_target_leakage" for d in r5.diagnostics)))
    # id column dropped
    checks.append(("id column excluded", "row_id" in r5.profile.dropped))
    # imbalance flagged
    checks.append(("imbalance flagged",
                   any(d.name == "class_imbalance" for d in r3.diagnostics)))
    # unsupervised resolves correctly
    checks.append(("unsupervised task resolved",
                   r6.task.kind.value == "unsupervised"))
    # to_dict round-trips
    import json
    try:
        json.dumps(r1.to_dict())
        checks.append(("report is JSON-serialisable", True))
    except Exception as e:
        checks.append((f"report JSON-serialisable ({e})", False))

    ok = 0
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok += int(passed)
    print(f"\n  {ok}/{len(checks)} checks passed")
