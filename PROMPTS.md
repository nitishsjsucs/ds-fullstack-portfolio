# Prompts

The assignment was to replicate a reference portfolio of data science applications with
an AI coding assistant, and to improvise where improvising made it better. This file
records what was actually asked for, what came back, and — more usefully — the points
where the first answer was wrong and had to be pushed on.

The reference repository is
[`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples).
Its own `PROMPTS.md` is a list of one-line briefs (`/teamwork-preview Now lets do another
project - clustering using popular kaggle data set...`). This portfolio takes a different
route, and the reason is worth stating: **a one-line brief produces a plausible-looking
project, not a correct one.** The prompts below are longer because most of the work was
in the constraints, not the request.

---

## The opening brief

```text
Data science full stack with ai coding assistant.
Replicate the data science experiments in the reference repository — prompts are
provided there. Publish to GitHub, walk through it in a YouTube video, and link the
video from README.md. Use your creativity and improvise; the reference is the minimum
bar.
```

### The three decisions taken before any code

The assistant asked three questions rather than starting, and they shaped everything:

1. **Sixteen shallow apps, or fewer deep ones?** The reference has 16 projects averaging
   ~3k lines each, most of it duplicated scaffolding. Chosen: **one monorepo, eight
   projects, shared infrastructure.** The eight cover the same syllabus surface
   (regression, classification, clustering, association rules, anomaly detection,
   forecasting, AutoML, deep learning) while sharing one evaluation module, one EDA
   profiler and one design system — so a fix lands everywhere at once, and the whole
   thing is demonstrable in a single sitting.

2. **Synthetic data, or real?** Chosen: **real, with provenance.** Synthetic data makes
   every metric unfalsifiable. `scripts/fetch_data.py` downloads from the original
   sources and records a SHA-256 and a licence for every committed file.

3. **The video.** The assistant cannot record or upload one. Chosen: it produces the
   full recording kit — a timestamped narration script, a demo run-order, a one-command
   launch, and captured screenshots of every screen — and the link gets wired into the
   README once recorded.

---

## The prompts, by phase

### 1. Data

```text
Before any modelling: get real datasets. For each project, find a genuinely public
source, download it, subsample deterministically with a pinned seed, and record the
source URL, licence, curation rule and a SHA-256 for every committed file. Curation is
allowed to select columns, coerce dtypes, subsample and sort. It is NOT allowed to
impute, scale, remove outliers, or filter on anything target-aware — all of that belongs
downstream of the train/test split where the audit can see it.
```

Result: nine curated files, 838k rows — NYC TLC's own January-2024 parquet, IBM's Telco
sample, UCI Online Retail, KDD Cup 99, UCI Bike Sharing, UCI Adult, tiny-shakespeare.
The class balances match the published characteristics (Telco 26.5%, Adult 23.9%, KDD-SA
3.4%), which is the first evidence the ingestion is honest.

### 2. The shared core

```text
Build the shared machinery before the projects, and make the leakage guarantees
structural rather than aspirational:

- every fitted transform must live inside a Pipeline that CV clones per fold
- temporal splits must assert chronology, not assume it
- cross-validation over lagged features must purge the look-back window and embargo
- the hold-out is scored once, and the CV-to-hold-out gap gets published
- for imbalanced targets the headline is PR-AUC; accuracy is flagged as misleading

Write the assertions as code that raises, not comments that describe.
```

### 3. Each project

```text
For <project>: run the full CRISP-DM cycle. Every phase needs an exit gate evaluated
against the run's real numbers — a gate that cannot fail is decoration. Every finding
must carry the measured value that supports it. Every decision must name the alternative
it rejected and why.

Publish the baselines the model has to beat. If the model loses, say so.
```

### 4. The console

```text
One React app, not eight. Dark, dense, professional. Hand-drawn SVG charts rather than a
charting library so the mark specs are exact. Every project gets the same eight tabs, and
a live inference playground that is genuinely interactive — sliders that re-score on
change, not a static screenshot of a form.
```

### 5. Verification

```text
Write tests that would actually catch the mistakes. Not "the function returns a float" —
construct a frame where a leak would change the answer and assert that it doesn't. Then
write a static AST scan for the syntactic shapes that cause leakage, and run it over
every project.
```

---

## Where the first answer was wrong

This is the part worth reading. Everything below was caught after the assistant had
produced working code that looked fine.

**Segmentation deployed a two-way split.** Silhouette is maximised at k=2 on RFM data and
falls monotonically after. The first implementation took the unconstrained argmax and
shipped "high value / low value" — statistically the tightest partition and commercially
worthless, since that is what the business already does without a model. The CRISP-DM
record *said* actionability was a hard criterion while the code ignored it. Fixed by
making the constraint real: `MIN_ACTIONABLE_K = 4`, with the report stating plainly that
k=2 won on the raw index and was rejected on those grounds.

**DBSCAN "won" by discarding the hard cases.** Silhouette is computed only over assigned
points, so a detector that labels 13% of customers as noise scores higher on the easy
remainder. Fixed with an explicit coverage rule checked *before* ranking, and the
excluded entry stays on the leaderboard with its reason attached.

**Two segments got the same name.** The persona taxonomy collapsed two genuinely
different clusters — a lapsed £618 customer and a lapsed £176 one — onto one label with
one recommended action, breaking the phase-1 criterion. Fixed by crossing engagement
against four value tiers rather than three, plus a uniqueness guarantee as a backstop.

**The churn cost model argued with itself.** The threshold sweep priced a missed churner
at the full lifetime margin while the campaign simulation on the same page applied a 35%
offer-acceptance rate. The two analyses disagreed by 3x and recommended different
actions. Fixed by making both use the *recoverable* margin.

**MASE below 1 was presented as proof of skill.** It isn't: MASE divides by *in-sample*
naive error, and on this hold-out the seasonal-naive baseline also scores 0.75. Fixed by
adding a head-to-head skill score — both methods on the same data — and demoting MASE to
a supporting number.

**An 80% interval covered 67%.** Raw quantile regression is reliably over-confident. The
first version reported the gap and moved on; the correct response was to fix it. Added
conformalized quantile regression calibrated on a held-out block, which moved coverage
from 67.1% to 79.2% at the cost of a band 17% wider.

**The AutoML project was written to demonstrate search overfitting, and the measurement
disagreed.** Its whole premise was that hill climbing fits its own validation folds, so
the report was drafted around a positive CV-to-hold-out gap. On the full 39k-row training
set every family came back *negative* — the hold-out scored slightly better than CV, mean
gap −0.003. At that fold size, twenty trials do not meaningfully fit. The tempting move
was to shrink the dataset until the expected story appeared. Instead the narrative was
rewritten around the measurement, including the observation that the same search on a few
thousand rows would very likely show the opposite. The same applies to the ensemble: it
was written up as "made it worse" from the quick-mode run, and the full run put it at
+0.0001, so it became "bought nothing measurable" — which is what the number says.

**FP-Growth was described as faster before it was measured.** It isn't, at this support
floor — Apriori wins by 3.2x (7.2s against 23.2s on identical output), because
FP-Growth's advantage only pays off once the floor drops low enough for candidate
generation to explode. The narrative had been written from the textbook. Fixed to report
whichever actually won, with the reason.

**A detector returned NaN and took the project down.** Elliptic Envelope's covariance is
singular on 52 largely-one-hot columns, so it fitted without raising and then produced
NaN scores that crashed the metric. Fixed by validating the scores and reporting the
detector as a failed result — which is itself a finding about the method's assumptions.

**Every project-specific API route 404'd.** The generic `/projects/{slug}/{payload}`
route was registered before the per-project routers, so it swallowed all of them.
Caught by the API test suite, not by hand.

**SHAP silently returned nothing.** A version mismatch between XGBoost 3 and the shap
package made `TreeExplainer` fail, and the code caught the exception and returned `None`
— so the explanation panel just disappeared. Fixed by routing XGBoost through its own
native `pred_contribs`, which has no cross-package coupling, and by returning a visible
error instead of silence.

---

## The instruction that did the most work

```text
Report honestly. If the ensemble doesn't beat the single model, say so. If a phase gate
fails, leave it failed and explain why rather than moving the threshold. If an algorithm
you expected to win loses, report the measurement, not the expectation.
```

Four of the portfolio's most interesting results exist because of that instruction: the
three inverted anomaly detectors, the detector that returned NaN, the ensemble that
bought nothing measurable, and the phase gate that fails on segment cohesion. A fifth is
subtler — the AutoML project was written expecting to demonstrate search overfitting, and
the measurement came back negative. Reporting that honestly meant rewriting the
narrative around what the numbers said rather than what the chapter heading promised.
