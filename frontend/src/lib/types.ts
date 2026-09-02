/** Shapes returned by the API.
 *
 * These mirror the artifact payloads written by the training scripts. They are
 * deliberately loose in the leaf positions (`unknown`, index signatures) because
 * each project extends the common envelope with its own analysis, and pretending
 * otherwise would mean either a false type or eight near-duplicate interfaces.
 * The structural parts every view relies on are typed precisely.
 */

export interface ProjectMeta {
  slug: string;
  number: number;
  title: string;
  tagline: string;
  task: string;
  domain: string;
  dataset: string;
  dataset_title: string;
  dataset_rows: number;
  primary_metric: string;
  metric_reason: string;
  techniques: string[];
  skills: string[];
  accent: string;
  icon: string;
  highlights: string[];
  trained?: boolean;
  provenance?: Provenance | null;
}

export interface Provenance {
  trained_at?: string;
  git_commit?: string | null;
  python?: string;
  platform?: string;
  dataset?: string;
  seed?: number;
  quick_mode?: boolean;
  [k: string]: unknown;
}

export interface ProjectIndex {
  portfolio: string;
  count: number;
  projects: ProjectMeta[];
  note: string;
}

export interface ProjectDetail extends ProjectMeta {
  available_payloads: string[];
  has_live_inference: boolean;
}

export interface Overview extends ProjectMeta {
  rows_modelled?: number;
  rows_raw?: number;
  champion?: string;
  headline: Record<string, number | string | boolean | null>;
  train_seconds?: number;
  _provenance?: Provenance;
}

/* ------------------------------------------------------------------- EDA -- */
export interface HistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface NumericProfile {
  column: string;
  kind: "numeric";
  count: number;
  missing: number;
  missing_pct: number;
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  q1: number;
  q3: number;
  iqr: number;
  skew: number;
  kurtosis: number;
  zeros: number;
  negatives: number;
  n_unique: number;
  outlier_fences: { lower: number; upper: number };
  outlier_count: number;
  outlier_pct: number;
  histogram: HistogramBin[];
  histogram_range: {
    lower: number;
    upper: number;
    clipped_below: number;
    clipped_above: number;
  };
  percentiles: Record<string, number>;
}

export interface CategoricalProfile {
  column: string;
  kind: "categorical";
  count: number;
  missing: number;
  missing_pct: number;
  n_unique: number;
  cardinality_ratio: number;
  mode: string | null;
  top_values: { value: string; count: number; share: number }[];
  tail_share: number;
  is_high_cardinality: boolean;
}

export interface QualityDimension {
  dimension: string;
  score: number;
  detail: string;
  rules?: { column: string; rule: string; violations: number; violation_pct: number }[];
  worst_column?: string | null;
}

export interface Eda {
  shape: { rows: number; columns: number };
  memory_mb?: number;
  dtypes?: Record<string, string>;
  numeric: NumericProfile[];
  categorical: CategoricalProfile[];
  correlation: {
    columns: string[];
    pearson: number[][];
    spearman: number[][];
    strong_pairs: {
      a: string;
      b: string;
      pearson: number;
      spearman: number;
      nonlinear_gap: number;
    }[];
  };
  missingness: {
    total_cells: number;
    missing_cells: number;
    missing_pct: number;
    complete_rows_pct: number;
    columns: { column: string; missing_pct: number; missing: number }[];
    co_missing_pairs: { a: string; b: string; jointly_missing_pct: number }[];
  };
  quality: {
    dimensions: QualityDimension[];
    overall_score: number;
    grade: string;
    note: string;
  };
  target_relationships: Record<string, unknown>[];
  target: string | null;
  temporal_heatmap?: TemporalHeatmap;
  demand_heatmap?: TemporalHeatmap;
  [k: string]: unknown;
}

export interface TemporalHeatmap {
  days: string[];
  hours: number[];
  matrix: number[][];
  aggregation: string;
  max: number;
  peak_cell: { day: string; hour: number; value: number };
}

/* -------------------------------------------------------------- CRISP-DM -- */
export interface Finding {
  statement: string;
  evidence: string;
  implication: string;
  severity: "info" | "watch" | "critical";
}

export interface Decision {
  decision: string;
  rationale: string;
  alternative_rejected: string;
  rejection_reason: string;
}

export interface Phase {
  key: string;
  title: string;
  purpose: string;
  summary: string;
  activities: string[];
  findings: Finding[];
  decisions: Decision[];
  metrics: Record<string, unknown>;
  artifacts: string[];
  gate: { criteria: string[]; passed: boolean; notes: string };
}

export interface CrispDm {
  project: string;
  business_question: string;
  success_criteria: string[];
  phases: Phase[];
  gates_passed: number;
  gates_total: number;
  all_gates_passed: boolean;
  iteration_notes: string[];
  methodology: string;
}

/* ----------------------------------------------------------- leaderboard -- */
export interface LeaderboardRow {
  rank?: number;
  model?: string;
  algorithm?: string;
  family?: string;
  cv_score?: number | null;
  seed_score?: number | null;
  improvement_from_search?: number | null;
  holdout_score?: number | null;
  generalisation_gap?: number | null;
  overfit_flag?: boolean;
  best_params?: Record<string, unknown>;
  n_trials?: number;
  fit_seconds?: number;
  hypothesis?: string;
  [k: string]: unknown;
}

export interface Trial {
  step: number;
  family: string;
  score: number | null;
  accepted: boolean;
  changed: string | null;
  note: string;
  params: Record<string, unknown>;
  elapsed_s: number;
}

export interface Leaderboard {
  scoring: string;
  leaderboard: LeaderboardRow[];
  trajectory?: Trial[];
  total_trials?: number;
  total_seconds?: number;
  ensemble?: Record<string, unknown> | null;
  protocol?: string | Record<string, unknown>;
  [k: string]: unknown;
}

/* ------------------------------------------------------------ evaluation -- */
export interface Metrics {
  [k: string]: number | boolean | null | undefined;
}

export interface Evaluation {
  metrics?: Metrics;
  roc_curve?: { fpr: number; tpr: number }[];
  pr_curve?: { recall: number; precision: number }[];
  calibration?: { predicted: number; observed: number }[];
  confusion_matrix?: { tn: number; fp: number; fn: number; tp: number };
  threshold_sweep?: {
    grid: {
      threshold: number;
      tp: number;
      fp: number;
      tn: number;
      fn: number;
      precision: number;
      recall: number;
      f1: number;
      expected_cost: number;
    }[];
    best_f1_threshold: number;
    best_cost_threshold: number;
    note: string;
  };
  gain_chart?: {
    decile: number;
    population_fraction: number;
    captured_positives: number;
    lift: number;
  }[];
  score_distribution?: {
    bin_start: number;
    bin_end: number;
    negatives: number;
    positives: number;
  }[];
  scatter?: { actual: number; predicted: number }[];
  residual_histogram?: HistogramBin[];
  residuals_vs_fitted?: { fitted: number; residual: number }[];
  quantile_calibration?: { quantile: number; predicted: number; observed: number }[];
  interval?: Record<string, unknown>;
  split?: Record<string, unknown>;
  baselines?: { name: string; description: string; [k: string]: unknown }[];
  n_test?: number;
  [k: string]: unknown;
}

/* --------------------------------------------------------------- explain -- */
export interface Explain {
  permutation?: {
    method: string;
    scoring: string;
    n_repeats: number;
    n_samples: number;
    features: {
      feature: string;
      importance: number;
      std: number;
      significant: boolean;
    }[];
    n_significant: number;
    note: string;
  };
  impurity?: {
    method: string;
    features: { feature: string; importance: number }[];
    caveat: string;
  } | null;
  shap?: {
    method: string;
    base_value: number;
    global_importance: { feature: string; mean_abs_shap: number }[];
    beeswarm: { feature: string; points: { shap: number; value_rank: number }[] }[];
    waterfalls: Record<
      string,
      {
        base_value: number;
        prediction: number;
        contributions: { feature: string; value: unknown; shap: number }[];
        residual_other_features: number;
      }
    >;
    n_explained: number;
    identity: string;
  } | null;
  partial_dependence?: {
    feature: string;
    curve: { x: number; y: number }[];
    deciles: number[];
    effect_range: number;
    monotone: boolean;
  }[];
  feature_count?: number;
  [k: string]: unknown;
}

/* ------------------------------------------------------------ model card -- */
export interface ModelCard {
  model: string;
  version: string;
  task: string;
  intended_use: string;
  out_of_scope: string[];
  training_data: Record<string, unknown>;
  evaluation_data?: Record<string, unknown>;
  features?: Record<string, unknown>;
  metrics: Record<string, unknown>;
  ethical_considerations: string[];
  limitations: string[];
  maintenance: { retrain_trigger: string; monitored_signals: string[]; owner?: string };
  [k: string]: unknown;
}

export interface AuditCheck {
  check: string;
  status: "pass" | "warn" | "fail";
  evidence: string;
}

export interface Audit {
  checks: AuditCheck[];
  passed: number;
  total: number;
  grade: string;
  scope: string;
  _provenance?: Provenance;
}

/* --------------------------------------------------------------- datasets -- */
export interface DatasetEntry {
  name: string;
  file: string;
  rows: number;
  columns: string[];
  size_mb: number;
  sha256: string;
  curation_rule: string;
  notes: string[];
  source: {
    key: string;
    title: string;
    url: string;
    licence: string;
    citation: string;
  };
}

export interface DatasetManifest {
  generated_by: string;
  seed: number;
  contract: string;
  datasets: DatasetEntry[];
}
