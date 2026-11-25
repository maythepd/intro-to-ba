/**
 * Typed client for the BaBank churn model API.
 * Base URL comes from VITE_API_BASE_URL. When it is not set, every call throws
 * ApiNotConfiguredError — callers must never substitute fabricated results.
 */

export type Country = "France" | "Germany" | "Spain";
export type Gender = "Male" | "Female";
export type PreferredChannel = "Mobile App" | "Online Web" | "ATM" | "Branch" | "No_Channel";

/** Staff-facing profile; the API resolves the complete Module 5 raw snapshot by ID. */
export interface CustomerFeatures {
  customer_id: string;
  credit_score: number;
  country: Country;
  gender: Gender;
  age: number;
  tenure: number;
  balance: number;
  products_number: number;
  credit_card: 0 | 1;
  active_member: 0 | 1;
  estimated_salary: number;
  recency_days: number;
  monthly_transactions: number;
  trans_amount_3m: number;
  trans_trend_ratio: number;
  app_logins_monthly: number;
  customer_service_complaints: number;
  has_unresolved_complaint: 0 | 1;
  preferred_channel: PreferredChannel;
  primary_device: string;
  raw_features?: Record<string, number | string | null>;
}

export type RiskLevel = "low" | "medium" | "high";

export interface PredictionSignal {
  title: string;
  description: string;
  direction: string;
}

export interface PredictionResult {
  customer_id: string;
  churn_probability: number;
  predicted_churn: boolean;
  risk_level: RiskLevel;
  model_name: string;
  signals: PredictionSignal[];
  prediction_id?: string | null;
  prediction_timestamp?: string | null;
  outcome_horizon_end?: string | null;
  monitoring_recorded?: boolean;
}

export interface RiskSummary {
  status: "scored";
  source: string;
  data_kind: string;
  model_name: string;
  total_customers: number;
  scored_customers: number;
  counts: { low: number; medium: number; high: number };
  customers_requiring_attention: number;
  risk_thresholds: { medium_from: number; high_from: number };
  scored_at: string;
  cached?: boolean;
  predictions?: Array<Pick<PredictionResult, "customer_id" | "churn_probability" | "predicted_churn" | "risk_level">>;
}

export interface HealthResponse {
  status?: string;
  [key: string]: unknown;
}

export type GenericRecord = Record<string, unknown>;

export type MonitoringSource = "live" | "snapshot_batch" | "study_replay";
export interface DistributionBin { bin: string; reference: number; current: number }
export interface DriftFeature {
  name: string; kind: string; method: string; score: number | null; status: string;
  valid_samples: number; missing_rate: number; missing_excess: number; new_category_rate: number;
  distribution: DistributionBin[];
}
export interface MonitoringAlert {
  id: string; rule: string; status: string; action: string; evidence: Record<string, unknown>;
}
export interface PerformanceReport {
  status: string; sample_count: number; mature_samples: number; labeled_samples: number;
  label_coverage: number; positive_samples: number; negative_samples: number;
  metrics: { roc_auc: number | null; average_precision: number | null; accuracy?: number; precision: number; recall: number; f1: number;
    balanced_accuracy?: number | null; log_loss?: number; brier?: number; predicted_positive_rate?: number; confusion_matrix: number[][] } | null;
  baseline: Record<string, number>; threshold: number; trigger_candidate: boolean; drops: Record<string, number>;
  label_sources?: string[]; simulated?: boolean; churn_rate?: number | null; baseline_note?: string;
  synthetic_baseline?: { batches: string[]; sample_count: number; metrics: Record<string, number> } | null;
}
export interface MonitoringRun {
  id: string; model_id: string; source: MonitoringSource; scenario: string;
  window_start: string; window_end: string; computed_at: string; status: string; sample_count: number;
  data_kind?: string; reference_sha256?: string;
  synthetic_replay?: { run_identity: string; batch_id: string; batch_index: number; batch_count: number;
    dataset_sha256: string; generation_seed: number; feature_snapshot: string; anchor: string;
    label_policy_version: string | null; label_seed: number | null; outcome_kind: string; time_kind: string };
  drift: { status: string; sample_count: number; minimum_samples: number; critical_share: number | null; trigger_candidate: boolean; features?: DriftFeature[] };
  prediction_drift?: DriftFeature | null; performance: PerformanceReport;
  quality: { request_count: number; valid_requests: number; invalid_requests: number; error_rate: number; max_missing_excess: number; trigger_candidate: boolean; note?: string };
  latency?: { p50_ms: number | null; p95_ms: number | null };
  segments?: { status: string; coverage: number; sample_count?: number; model_id?: string; distribution?: DistributionBin[]; note?: string };
  consecutive_candidates?: Record<string, number>; alerts: MonitoringAlert[];
  policy?: MonitoringSummary["policy"];
}
export interface MonitoringSummary {
  storage_ready: boolean; storage_backend: string; storage_persistent: boolean; model_id: string | null;
  reference_available: boolean; last_ingest_error: string | null; last_error_at: string | null; last_write_at: string | null;
  source: MonitoringSource; scenario: string; window_hours: number; outcome_horizon_days: number; minimum_drift_samples: number;
  latest: MonitoringRun | null; history: MonitoringRun[]; models: Array<{ id: string; name: string; created_at: string }>;
  scenarios?: string[];
  scenario_details?: Array<{ name: string; outcome_kind?: string | null; label_seed?: number | null }>;
  drift_available: boolean; performance_tracking_available: boolean; stale: boolean;
  policy: { min_labeled_samples: number; min_class_samples: number; min_label_coverage: number; consecutive_windows: number; psi_critical: number; js_critical: number; performance_drop: number; quality_error_rate: number };
}

export interface CampaignSummary {
  id: string; created_at: string; status: string; mode: string; dataset_sha256: string; label_policy_version?: string | null;
}
export interface CampaignUnit {
  campaign_id: string; algorithm: string; batch_id: string; model_id: string; task: string; cohort_sha256: string;
  report: {
    id: string; sample_count: number; status: string; role?: string; performance?: PerformanceReport;
    prediction_drift?: DriftFeature | null; assignment_method?: string; eval_N?: number; k?: number;
    silhouette?: number | null; dbi?: number | null; chi?: number | null; mean_distance?: number; p95_distance?: number;
    distortion?: number; outlier_share?: number | null; margin?: number; counts?: number[]; cluster_shares?: number[];
    baseline?: Record<string, number | null>; drops?: Record<string, number | null>; review_status?: string;
    assignment_drift?: DriftFeature; profiles?: Array<{cluster_id: number; sample_count: number; status: string; features: DriftFeature[]}>;
  };
}
export interface CampaignDetail extends CampaignSummary {
  manifest: { campaign_id: string; mode: string; anchor: string; dataset_sha256: string; label_seed: number | null;
    label_policy_version: string | null; config: {silhouette_sample_size: number; threshold_note: string};
    models: Record<string, {model_id: string; task: string; role: string; assignment_method?: string}>;
    batches: Array<{batch_id: string; index: number; N: number; cohort_sha256: string; window_start: string; window_end: string}> };
  reports: CampaignUnit[]; report_count: number;
}
export interface CampaignInput {
  campaign_id: string; batch_id: string; feature_space_id: string; reference_sha256: string;
  report: MonitoringRun["drift"] & {space: string};
}

export type ClusterAlgorithm = "kmeans" | "hierarchical";
export interface ClusterProfile {
  id: number;
  name: string;
  customers: number;
  share: number;
  avg_balance: number;
  avg_monthly_transactions: number;
  avg_app_logins_monthly: number;
  observed_churn_rate?: number;
}
export interface SegmentSummary {
  algorithm: ClusterAlgorithm;
  source: string;
  customer_count: number;
  n_clusters: number;
  silhouette_sample: number | null;
  features: string[];
  clusters: ClusterProfile[];
  assignment_method: string;
  cluster_labels_are_risk_rankings: boolean;
  study_note: string;
}
export interface SegmentCustomer {
  customer_id: string;
  kmeans: number;
  hierarchical: number;
  hierarchical_assignment_method: "fitted_ward" | "nearest_centroid_approximation";
}
export interface SegmentPrediction {
  algorithm: ClusterAlgorithm;
  cluster_id: number;
  method: string;
  warning: string | null;
}


export class ApiNotConfiguredError extends Error {
  constructor() {
    super("API is not configured (VITE_API_BASE_URL).");
    this.name = "ApiNotConfiguredError";
  }
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export class ApiTimeoutError extends Error {
  constructor() {
    super("Request timed out.");
    this.name = "ApiTimeoutError";
  }
}

const RAW_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() ?? "";
export const API_BASE_URL = RAW_BASE.replace(/\/+$/, "");
export const isApiConfigured = API_BASE_URL.length > 0;

const DEFAULT_TIMEOUT_MS = 15000;

interface RequestOptions {
  method?: "GET" | "POST";
  body?: unknown;
  signal?: AbortSignal;
  timeoutMs?: number;
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  if (!isApiConfigured) throw new ApiNotConfiguredError();

  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const onAbort = () => controller.abort();
  opts.signal?.addEventListener("abort", onAbort);

  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      method: opts.method ?? "GET",
      headers: { Accept: "application/json", ...(opts.body ? { "Content-Type": "application/json" } : {}) },
      body: opts.body ? JSON.stringify(opts.body) : undefined,
      signal: controller.signal,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const j = await res.json();
        detail = (j?.detail ?? j?.message ?? detail) as string;
      } catch {
        /* ignore non-JSON error body */
      }
      throw new ApiError(typeof detail === "string" ? detail : `HTTP ${res.status}`, res.status);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (timedOut) throw new ApiTimeoutError();
    throw err;
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener("abort", onAbort);
  }
}

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/health", { signal, timeoutMs: 60000 }),
  listCustomers: (params: { search?: string; limit?: number }, signal?: AbortSignal) => {
    const q = new URLSearchParams();
    if (params.search) q.set("search", params.search);
    q.set("limit", String(params.limit ?? 50));
    return request<CustomerFeatures[] | { items: CustomerFeatures[] }>(`/api/customers?${q}`, { signal });
  },
  getCustomer: (id: string, signal?: AbortSignal) =>
    request<CustomerFeatures>(`/api/customers/${encodeURIComponent(id)}`, { signal }),
  predict: (payload: CustomerFeatures, signal?: AbortSignal) =>
    request<PredictionResult>("/api/predict", { method: "POST", body: {...payload, customer_id: /^\d+$/.test(payload.customer_id) ? Number(payload.customer_id) : null}, signal, timeoutMs: 20000 }),
  metrics: (signal?: AbortSignal) => request<GenericRecord>("/api/metrics", { signal }),
  riskSummary: (signal?: AbortSignal) =>
    request<RiskSummary>("/api/dashboard/risk-summary", { signal }),
  batchScore: (force = false) =>
    request<RiskSummary>(`/api/predict/batch?force=${force ? "true" : "false"}`,
      { method: "POST", timeoutMs: 120000 }),
  segments: (algorithm: ClusterAlgorithm, signal?: AbortSignal) =>
    request<SegmentSummary>(`/api/segments?algorithm=${encodeURIComponent(algorithm)}`, { signal }),
  segmentCustomer: (id: string, signal?: AbortSignal) =>
    request<SegmentCustomer>(`/api/segments/customers/${encodeURIComponent(id)}`, { signal }),
  predictSegment: (payload: CustomerFeatures, algorithm: ClusterAlgorithm, signal?: AbortSignal) =>
    request<SegmentPrediction>(`/api/segments/predict?algorithm=${encodeURIComponent(algorithm)}`,
      { method: "POST", body: payload, signal }),
  monitoring: (params: { source: MonitoringSource; scenario?: string; modelId?: string; days?: number }, signal?: AbortSignal) => {
    const q = new URLSearchParams({ source: params.source, days: String(params.days ?? 30) });
    if (params.scenario) q.set("scenario", params.scenario);
    if (params.modelId) q.set("model_id", params.modelId);
    return request<MonitoringSummary>(`/api/monitoring?${q}`, { signal, timeoutMs: 90000 });
  },
  monitoringRun: (id: string, signal?: AbortSignal) => request<MonitoringRun>(`/api/monitoring/runs/${encodeURIComponent(id)}`, { signal, timeoutMs: 60000 }),
  campaigns: (signal?: AbortSignal) => request<{items: CampaignSummary[]}>("/api/monitoring/campaigns", {signal, timeoutMs: 60000}),
  campaign: (id: string, signal?: AbortSignal) => request<CampaignDetail>(`/api/monitoring/campaigns/${encodeURIComponent(id)}`, {signal, timeoutMs: 60000}),
  campaignInputs: (id: string, signal?: AbortSignal) => request<{items: CampaignInput[]}>(`/api/monitoring/campaigns/${encodeURIComponent(id)}/input-drift`, {signal, timeoutMs: 60000}),
};

export function normalizeCustomerList(data: CustomerFeatures[] | { items: CustomerFeatures[] }): CustomerFeatures[] {
  return Array.isArray(data) ? data : (data?.items ?? []);
}

export function describeError(err: unknown): string {
  if (err instanceof ApiNotConfiguredError) return err.message;
  if (err instanceof ApiTimeoutError) return "The server did not respond in time. Please try again.";
  if (err instanceof ApiError) return `Server error (${err.status}): ${err.message}`;
  if (err instanceof TypeError) return "Could not connect to the API. Check network access and CORS settings.";
  return err instanceof Error ? err.message : "Unknown error.";
}
