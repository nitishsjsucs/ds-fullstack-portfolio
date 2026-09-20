# Walkthrough recording kit

Everything needed to record the YouTube walkthrough: what to run, what to click, what
to say, and where the interesting moments are.

**Target length: 14–16 minutes.** That is long enough to show all eight systems and
still dwell on the three findings that make the portfolio worth watching. If you need a
shorter cut, record the full version and use the chapter markers to publish a 6-minute
"highlights" edit from sections 1, 5, 7 and 9.

---

## Before you press record

```bash
cd ds-fullstack-portfolio

# 1. Confirm everything is trained and green. Takes about 30 seconds.
python scripts/audit.py          # expect 83/83 checks, 0 critical
pytest backend/tests -q          # expect 133 passed

# 2. Start both servers. Leave them running.
make api      # terminal 1 → :8000
make web      # terminal 2 → :5173
```

Checklist:

- [ ] Browser at **1680×1050** or larger, zoom at 100%
- [ ] Dark desktop background; hide the bookmarks bar and any notifications
- [ ] A second tab open on `http://localhost:8000/docs` for the API section
- [ ] An editor open on `backend/app/core/splitting.py` and
      `backend/app/projects/forecast/data.py` for the code moments
- [ ] Screen recorder set to capture the browser window only, at 1080p or better

---

## The script

Timings are cumulative. The narration is written to be spoken, not read — paraphrase
freely, but keep the numbers exact.

### 0:00 — Cold open (30s)

**Screen:** `#/` — the landing page.

> "This is a portfolio of eight end-to-end data science systems. Eight different
> problems — regression, classification, clustering, association mining, anomaly
> detection, forecasting, AutoML and a transformer built from scratch — on eight
> genuinely real public datasets.
>
> What I want to show you isn't that the models work. It's the parts most portfolios
> leave out: where the results are unflattering, and what the code does to stop me
> fooling myself."

Scroll to **Where to start** and let the three cards sit on screen for a beat.

> "Three anomaly detectors that rank intrusions as *more normal* than benign traffic.
> An ensemble that bought nothing. And an eighty percent prediction interval that only
> covered sixty-seven. We'll get to all three."

---

### 0:30 — The data is real (1m 15s)

**Screen:** `#/datasets`

> "Everything starts here. Nine curated files, 838,000 rows. NYC's own taxi archive for
> January 2024, IBM's Telco sample, UCI Online Retail, the KDD Cup intrusion data, UCI
> bike sharing, the census income set, and a million characters of Shakespeare."

Scroll to the NYC TLC entry and point at the checksum.

> "For every file there's the original source URL, the licence, the exact curation rule,
> and a SHA-256 of the committed bytes. That subsample is a pure function of a pinned
> seed, so two clean checkouts produce byte-identical files — and there's a test that
> recomputes every checksum and fails if they drift."

Point at the contract callout at the top.

> "And the contract is deliberately narrow. Ingestion may select columns, coerce types,
> subsample and sort. It may not impute, scale, remove outliers, or filter on anything
> target-aware — because all of that belongs downstream of the train/test split, where
> the audit can see it."

---

### 1:45 — A project end to end (3m)

**Screen:** `#/taxi/overview`

> "Let's take one project all the way through. NYC taxi: predict how long a trip will
> take and what it'll cost, at the moment someone requests a cab."

**Tab: Data.**

> "The quality scorecard grades the data as published — it doesn't repair anything.
> Note the exclusions table: eight rules, every one a physical bound. Non-positive
> duration, zero distance, negative fare. What you won't find is a rule like 'drop
> anything more than three standard deviations from the mean', because that threshold
> comes from the target, and it would delete exactly the hard cases the model gets
> scored on."

Scroll to the temporal heat map.

> "Seven days by twenty-four hours. You can read the commuting peaks straight off it —
> that's what motivated the cyclical time encoding."

**Tab: CRISP-DM.**

> "All six phases, and each has an exit gate that's actually evaluated. Every finding
> carries the number that supports it, and every decision names the alternative it
> rejected and why."

Open the **Data preparation** phase, point at the decision card.

> "Split chronologically, not randomly. Rejected: a random eighty-twenty shuffle —
> because it lets the model interpolate inside a rush hour it has already seen."

Scroll to the bottom, **Where this project looped back**.

> "And this is the section normally deleted before publication. The first evaluation
> showed airport trips with double the error of city trips, which is the only reason an
> explicit trip-type feature exists."

**Tab: Models.**

> "Six model families, each hill-climbed on cross-validated MAE. Every trial is logged —
> including the rejected ones, because a search that only reports its successes hides
> how much of the improvement was luck."

**Tab: Live inference.**

> "And it's live. Midtown to JFK, Friday at five."

Drag the **departure hour** slider from 05:00 through 18:00 and back.

> "Every change is a real call to the deployed pipeline. Watch the congestion curve at
> the bottom — same trip, every hour of the day. That gap between peak and trough is
> the congestion cost the model learned from the data alone."

Point at the SHAP waterfall.

> "And the explanation is exact: base value plus each feature's contribution equals the
> prediction. There's a test that asserts that sum."

---

### 4:45 — Finding one: three inverted detectors (2m)

**Screen:** `#/anomaly/models`

> "Now the part I actually want to talk about. Network intrusion detection, eighty
> thousand connections, a three-point-four percent attack rate. Five unsupervised
> detectors attempted, four produced usable scores, and none of them ever sees a label."

Point at the ROC-AUC column, where three rows are badged red.

> "Look at these three. ROC-AUC of 0.34, 0.28, 0.24. Those aren't weak detectors —
> below 0.5 means they're *inverted*. They rank intrusions as **more normal** than
> benign traffic."

Expand one of the diagnosis callouts.

> "The reason is genuinely interesting. The dominant attacks here are denial-of-service
> floods, and floods are enormously repetitive. So they form the *densest* region of the
> feature space, and every density-based method reads them as the most ordinary traffic
> present.
>
> Which is the lesson: 'anomalous' and 'malicious' are not the same property. A frequent
> attack is not an outlier."

Point at the Elliptic Envelope row.

> "And this one didn't produce a number at all. Its covariance goes singular on
> fifty-two mostly-one-hot columns, so it returned NaN. Earlier that crashed the whole
> project — now it's validated and reported as a result about the method's assumptions,
> because a Gaussian model on rank-deficient data isn't fitting badly, it's undefined."

**Tab: Live inference.**

> "And the operational view: an analyst reviews a fixed queue, not 'everything above
> 0.5'. Drag the budget and you're pricing recall in analyst hours."

Scroll to per-family recall.

> "This table is the one a security team actually needs. A strong average can coexist
> with an entire attack class never being surfaced."

---

### 6:45 — Finding two: ensembling bought nothing (1m 30s)

**Screen:** `#/automl/models`

> "Census income. This is the project about model selection itself, so its subject
> matter is the ways automated search deceives you."

Point at the CV vs hold-out gap column.

> "Every family is hill-climbed on cross-validated AUC, then scored once on an untouched
> hold-out, and the gap between the two is published per family. That gap is the cost of
> searching — where it's positive, quoting the CV number alone would overstate deployed
> performance by exactly that margin.
>
> And here's the thing: on this dataset it comes back *negative*. Every family scores
> slightly better on the hold-out than on CV. The search didn't overfit. With
> thirty-nine thousand training rows and five-fold CV, each fold is big enough that
> twenty trials don't meaningfully fit it.
>
> That's not a failed experiment — it's the answer. And it's on the page instead of the
> overfitting story you were probably expecting, because the whole point is that you
> measure it rather than assume it. Run the same search on two thousand rows and you'd
> get the opposite."

Scroll to the stacking panel.

> "Then the ensembling. Full Caruana greedy selection plus a logistic meta-learner on
> out-of-fold predictions. Greedy gets 0.9314. Best single model: 0.9313. The logistic
> stack actually lands below both, at 0.9309.
>
> A difference in the fourth decimal place, on a nine-thousand-row hold-out, is noise.
> Ensembling bought nothing — which is exactly what a 0.92 correlation between the base
> learners predicts. There's no disagreement left to exploit.
>
> That's reported at face value rather than dressed up as a win."

**Tab: Evaluation** → distillation panel.

> "One more: a depth-four decision tree trained to imitate the ensemble recovers
> ninety-three percent of its AUC at ninety-two percent decision fidelity. Four levels of
> if-statements. Given that the ensemble bought nothing over a single model anyway, in a
> regulated setting that tree is what you ship — and now you know exactly what it costs."

**Tab: Live inference** → scroll to fairness.

> "And sex and race are excluded from the feature set entirely — yet the disparate impact
> ratio still fails the four-fifths rule on all three attributes. Correlated proxies carry
> the same information. This is why 'fairness through unawareness' isn't a strategy."

---

### 8:15 — Finding three: the interval that lied (1m 30s)

**Screen:** `#/forecast/evaluation`

> "Bike-share demand. The evaluation protocol is the substance here."

Point at the split card, then the skill panel.

> "MAE of 28.8 rides against 64.2 for seasonal naive — a fifty-five percent skill score.
> And note the callout: MASE alone would *not* have settled this. MASE divides by
> in-sample error, so on a test period easier than the training span every method scores
> below one — including the naive baseline, at 0.75. The head-to-head skill score is the
> claim that holds up."

Scroll to interval coverage.

> "Now the interesting failure. The raw quantile band was a nominal eighty percent
> interval that covered sixty-seven point one. That isn't conservative, it's wrong —
> downstream planning treats a band as a bound.
>
> The fix is conformalized quantile regression: calibrate on a held-out block the
> quantile models never saw, and widen by the empirical conformity score. Coverage goes
> to seventy-nine point two. The price is a band about seventeen percent wider, and that
> is the correct trade."

Switch to the editor, `backend/app/projects/forecast/data.py`, the `engineer` function.

> "And underneath all of it, one line. `past = target.shift(1)` — *then* the rolling
> windows. Without that shift, every rolling mean contains the value it's used to
> predict. Nothing errors; the score just gets better. An early version of this project
> reported a MASE around 0.3, which was impossible, and that's what prompted the audit
> that found it."

Switch to `backend/tests/test_projects.py`, the shift test.

> "So there's a test that builds a series with a spike in the final hour and asserts the
> rolling feature at that spike doesn't reflect it."

---

### 9:45 — The remaining four, quickly (2m)

Thirty seconds each. Land the one distinctive thing and move on.

**`#/churn/live`** — drag the offer-cost and acceptance sliders.

> "Churn. The deliverable isn't a probability, it's a contact list. Drag the retention
> economics and the recommendation moves — that only works because the probability is
> calibrated. Isotonic calibration barely moves AUC and halves the calibration error,
> which is exactly what theory predicts, since calibration reorders nothing."

**`#/segments/live`** — change a slider until confidence flips to "borderline".

> "Segmentation. Four algorithms on identical features. DBSCAN scored the highest
> silhouette — by discarding thirteen percent of customers as noise, so it's excluded by
> an explicit coverage rule that runs *before* ranking. And note: this project's
> evaluation gate **fails**. Silhouette 0.239 against a 0.25 bar I set in phase one. The
> structure is real and stable, it just isn't crisply separated — so the gate stays
> failed rather than the threshold moving."

**`#/basket/models`** then **`live`**.

> "Market basket. Apriori and FP-Growth return identical itemsets — 1,497 of them,
> they must match, both are exact. But Apriori takes 7.2 seconds and FP-Growth 23.2.
> That's the opposite of the textbook ordering, because FP-Growth's advantage only pays
> off once the support floor drops low enough for candidate generation to explode. The
> report says what was measured, not what was expected."

Add an item in the live tab.

> "And the recommender only serves rules that survived a temporal hold-out."

**`#/nanollm/live`** — generate at temperature 0.2, then 1.4.

> "And a transformer written out from scratch — RoPE, RMSNorm, SwiGLU, weight tying.
> Two-point-one-seven bits per character against a 4.78 unigram floor. At temperature
> 0.2 it loops; at 1.4 the spelling falls apart."

Scroll to the attention heat map.

> "And that's attention read out of the trained weights. Everything above the diagonal is
> exactly zero — that's the causal mask, visible rather than asserted. There's a test that
> checks it, because a model that can see forward gets a spectacular training loss by
> copying."

---

### 11:45 — How it's kept honest (2m)

**Screen:** `#/methods`

> "So what actually stops all of this from being wishful thinking. Four guarantees, each
> enforced by shared code rather than by a claim in a README."

Read the four cards briefly.

**Terminal:**

```bash
pytest backend/tests -q
```

> "A hundred and thirty-three tests. Not 'the function returns a float' — they build
> frames where a leak would change the answer and assert that it doesn't."

```bash
python scripts/audit.py
```

> "And a static AST scan across every project for the syntactic shapes that cause
> leakage: a rolling window with no preceding shift, a shuffled temporal split, a fitted
> transform outside a pipeline, a target in a feature list. Eighty-three of eighty-three
> checks pass, zero critical."

**Screen:** any project → **Audit & card** tab.

> "Every project publishes its own audit with the evidence attached, and a model card
> that says what it must *not* be used for. The census model must not touch a real
> decision about a real person. The intrusion detector runs on a 1999 simulation. The
> language model has no knowledge and cannot answer a question."

Scroll to the audit's limits section.

> "And the audit says what it doesn't establish: a static scan matches syntax, not
> semantics. Passing every check means the known failure modes were checked for. It does
> not mean the analysis is correct."

---

### 13:45 — Close (45s)

**Screen:** back to `#/`

> "Eight systems, eight real datasets, one application. Every fitted transform inside a
> pipeline, every time series split in time order, every hold-out scored once.
>
> The results I'd point at aren't the good ones. They're the three inverted detectors,
> the ensemble that made things worse, and the phase gate that fails. A version of this
> that quietly deleted those would look better and be worth less.
>
> Everything's in the repo — the code, the generated papers, the audit, and the prompts,
> including the eleven places the first answer was wrong. Link's below."

---

## Chapter markers

Paste into the YouTube description:

```
0:00  What this is
0:30  The data is real — provenance and checksums
1:45  One project end to end: NYC taxi
4:45  Finding 1 — three anomaly detectors rank intrusions as normal
6:45  Finding 2 — ensembling bought nothing
8:15  Finding 3 — an 80% interval that covered 67%
9:45  Churn, segmentation, market basket, NanoGPT
11:45 How it's kept honest — tests and the leakage audit
13:45 Close
```

## Suggested description

```
Eight end-to-end data science systems on eight real public datasets — regression,
classification, clustering, association mining, anomaly detection, forecasting, AutoML
and a transformer built from scratch — each following CRISP-DM with evaluated phase
gates and a published leakage audit.

The walkthrough focuses on the unflattering results: three anomaly detectors that rank
intrusions as more normal than benign traffic, an ensemble that gains nothing over its
best single member, and a prediction interval that had to be conformally recalibrated
after covering 67% of outcomes instead of 80%.

Code, generated papers, audit report and prompts:
https://github.com/nitishsjsucs/ds-fullstack-portfolio

Built with FastAPI, scikit-learn, XGBoost, LightGBM, PyTorch, React and TypeScript.
```

---

## After recording

The walkthrough is recorded and already linked from the README, between the
`<!-- VIDEO:START -->` and `<!-- VIDEO:END -->` markers:

<https://youtu.be/4C5_ZU11Ues>

To swap in a re-recorded take, edit the link between those two markers and commit.

---

## If something breaks mid-take

| Symptom | Fix |
|---|---|
| A tab shows "Not trained yet" | `python scripts/train_all.py --only <slug>` |
| A page is blank | Check the browser console; the API is probably not on :8000 |
| Live inference returns 503 | The pickled model is missing — retrain that project |
| Charts render but are empty | Hard-refresh; the frontend memoises GET responses per session |
| Numbers differ from this script | Expected if you retrained — the script quotes the committed run |
