# primer

**A fast, transparent model-selection brief for tabular ML.**

`primer` profiles a dataset, figures out the task, flags the data problems that
quietly wreck studies, probes the data's *structure* with a handful of cheap
"landmark" models, and hands you a **confidence-scored, fully-explained shortlist
of models to try** — in seconds, *before* you spend hours training the wrong
five.

It is self-contained: **NumPy + pandas only**. The landmark models (ridge,
decision stump, 1-NN, Gaussian NB) are hand-written in NumPy, so there is no
heavyweight dependency and nothing to install beyond the scientific-Python
basics.

```python
import primer

report = primer.analyze("data.csv", target="price")
print(report)            # one-screen brief
print(report.render())   # the full report
```

---

## What it is — and what it is not

This is the honest framing, because it determines whether the tool helps you or
misleads you.

**`primer` is a probabilistic *advisor*, not an oracle.** The "No Free Lunch"
theorem guarantees that no procedure can look at a dataset's statistics and *know*
which model will win — if that were possible, AutoML would be a closed problem.
So `primer` does not pretend to. Instead it gives you a **well-reasoned prior**:
where to *start*, with the evidence laid out so you can judge it yourself.

* It is the transparent, instant step you run **before** AutoML / heavy training.
* It **explains every recommendation** — each score decomposes into named reasons.
* It is **not** a replacement for empirical validation. The shortlist tells you
  what to try; a quick benchmark settles the winner. (That benchmark is the
  cheap-proxy validation layer on the roadmap below.)

On tabular data specifically, the recommendations lean on the now well-replicated
finding that **gradient-boosted trees (LightGBM / XGBoost / CatBoost) are the
strongest default**, with deep learning rarely pulling ahead (Grinsztajn et al.,
NeurIPS 2022; Shwartz-Ziv & Armon, 2022). `primer` encodes that prior but lets
the *data* override it — when the cheap probes say the signal is linear, it tells
you to keep it simple.

**Scope (v1): tabular data only.** Images, text, and reinforcement learning are
out of scope by design. RL in particular is a category error here — it is defined
by an environment, rewards, and sequential decisions, none of which are inferable
from a static table.

---

## Install / use

No packaging required — drop the `primer/` folder next to your code, or add it to
your path:

```python
import sys; sys.path.insert(0, "/path/to/primer_parent")
import primer
```

Requirements: `numpy`, `pandas` (both already in any DS environment).

### The three things you'll use

```python
report = primer.analyze(df_or_path, target="y")   # target=None -> unsupervised

print(report)                       # compact summary
print(report.render())              # full text report
report.recommendations              # list[Recommendation]  (ranked)
report.recommendations[0].reasons   # why the top pick was chosen
report.diagnostics                  # list[Diagnostic]      (data problems)
report.directions                   # concrete next steps
report.to_dict()                    # everything, JSON-serialisable
```

A `Recommendation` carries `.model`, `.family`, `.score` (0–1 suitability),
`.confidence` (0–1, *heuristic* — see below), and `.reasons` (signed, explained
evidence).

---

## What you get back

```
────────────────────────────────────────────────────────────────
  PRIMER — model-selection brief
────────────────────────────────────────────────────────────────
  data      : 3,000 rows x 4 cols
  task      : classification / binary  (2 classes)
  flags     : 0 critical, 0 warning
  confidence: HIGH (71%)  ██████████████······

  recommended models:
    1. CatBoost          score 1.00  conf 71%
       └ 67% of features are categorical (1 high-cardinality) — models with
         native categorical handling (CatBoost, LightGBM) avoid one-hot blow-up.
    2. LightGBM          score 0.91  conf 68%
    3. XGBoost           score 0.65  conf 57%
────────────────────────────────────────────────────────────────
```

The full report adds the task reasoning, diagnostics (leakage, imbalance,
multicollinearity, missingness, temporal traps, duplicates), the landmark probe
scores, the key metafeatures, and a **Directions** section that turns all of it
into next steps.

---

## How it works — the architecture

`primer` is built as **seven layers**, each a small module with a clean interface.
The pipeline is **injectable**: you can swap the recommender, enable the
validator, or extend the rule set without touching the layers around them.

```
  1  ingest        load + infer a semantic type & role for every column
  2  task          resolve regression / classification / unsupervised
  3  diagnostics   leakage · imbalance · collinearity · missingness · temporal
  4a metafeatures  shape · dependence (mutual information) · distribution shape
  4b landmarks     cheap NumPy models whose *scores* reveal structure
  5  recommend     the rule engine — turns evidence into an explained shortlist
  6  validate      cheap-proxy / successive-halving validation   (ROADMAP)
  7  confidence    calibrate the prior against measured curves   (ROADMAP)
```

### The clever bit: landmarks as structure detectors

Instead of guessing from summary statistics alone, `primer` fits a few
**deliberately cheap** models on a sub-sample and *compares* them:

* a **decision stump** beating a **linear model** ⇒ the signal is **nonlinear**;
* **1-NN** beating linear ⇒ **local / neighbourhood structure**;
* the linear model already near the ceiling ⇒ **keep it simple** (Occam);
* even the best probe barely beating the baseline ⇒ **weak signal** — invest in
  features, not fancier estimators.

This is the rigorous version of the practitioner's intuition that *"if the error
variance is high, a heavier model won't save you."* The probes cost milliseconds
and produce *evidence*, not guesses.

### The rule engine reasons over *capabilities*, not model names

Each model in the registry is described by **capabilities** — `nonlinearity`,
`native_categorical`, `scales_large_n`, `robust_missing`, `interpretable`, … —
graded in [0, 1]. Rules then vote on *capabilities*:

> "nonlinearity detected" → reward every model that captures nonlinearity, penalise
> the purely linear ones.

A model's final score is its family prior **plus** the sum over capabilities of
(rule vote × how much the model has that capability). Because the bookkeeping is
per-capability and signed, **every recommendation decomposes into the exact
reasons that moved it** — and adding a model or a rule never requires rewiring the
rest.

### Honesty about confidence

The `confidence` numbers are a **heuristic prior**, *not* a calibrated probability
of out-of-sample success. They reflect how decisively the evidence separates the
top choice from the rest, and how much real signal the cheap probes found — never
a promise. The report says so out loud. Turning this prior into a *calibrated*
posterior is exactly what Layer 6 is for.

---

## Extending it (the door is open)

Three extension points, all first-class:

**Add a model or a whole family** — one entry, no core changes:

```python
from primer import register, ModelSpec
register(ModelSpec(
    key="tabnet", display="TabNet", family="neural", tasks=["classification"],
    caps=dict(nonlinearity=0.9, scales_large_n=0.7, low_n_friendly=0.2),
    library_hint="pytorch-tabnet",
))
```

**Add a rule** — append to `primer.RULES`; it votes on capabilities like the rest.

**Swap the recommender** — `RuleBasedRecommender` implements the `Recommender`
interface. A future meta-learned ranker implements the same interface and drops in:

```python
primer.Primer(recommender=MyMetaLearnedRecommender()).analyze(df, target="y")
```

The per-capability rule weights are deliberately the same quantities a
meta-learner would *learn* from an OpenML-scale corpus — so v2 is a weight swap,
not a rewrite.

---

## Roadmap (the layers we left room for)

1. **Cheap-proxy validation (Layer 6) — highest value.** Take the top-k
   shortlist and actually run them on small sub-samples with tiny budgets, using
   successive-halving / Hyperband promotion. The learning-curve separation turns
   the rule engine's *prior* confidence into an *empirically calibrated* one. This
   is the real fix for "20 hours on Random Forest, then discovering XGBoost was
   better." The `ProxyValidator` interface and the intended design are already in
   `validate.py`.
2. **Meta-learning the rule weights.** Replace hand-set capability weights with
   weights learned from OpenML run histories (cf. `pymfe` for metafeatures). Same
   interface, better priors.
3. **Calibrated confidence (Layer 7).** Map evidence + measured proxy curves to
   honest, validated probabilities.
4. **Per-capability explanations.** Attribute each recommendation's reasons to the
   specific capability that drove it (rather than the rule's headline rationale).
5. **New modalities.** A `Modality` abstraction so text/image profilers can plug
   in behind the same report — still never RL, which doesn't fit the static-table
   premise.

---

## Limitations (read these)

* Recommendations are **priors, not results** — always validate empirically.
* Confidence is **heuristic**, not calibrated (until Layer 6/7 land).
* Landmark probes run on a **sub-sample** and use light encoding; they detect
  *structure*, they are not tuned models.
* Very wide (p ≫ n) or all-text datasets get thinner signal from the probes.
* For larger-than-memory data, profile a sample (the loader does this above
  50k rows) or reach for Polars/DuckDB upstream.

---

## Module map

| module | layer | responsibility |
|---|---|---|
| `ingest.py` | 1 | load + semantic column typing |
| `task.py` | 2 | regression / classification / unsupervised |
| `diagnostics.py` | 3 | data-quality and validity checks |
| `metafeatures.py` | 4a | statistical + information-theoretic descriptors |
| `landmarking.py` | 4b | cheap NumPy probe models + CV harness |
| `numpyx.py` | — | pure-NumPy kernels (skew, entropy, MI, condition number) |
| `registry.py` | — | capability-based model catalogue (extension hub) |
| `rules.py` | 5 | the rule engine + rule set |
| `recommend.py` | 5 | `Recommender` interface, scoring, confidence |
| `validate.py` | 6 | cheap-proxy validation interface (stub) |
| `report.py` | — | text rendering |
| `core.py` | — | `Primer` orchestrator + `analyze()` |
| `types.py` | — | shared, serialisable data structures |

`__version__ = "0.1.0"`
