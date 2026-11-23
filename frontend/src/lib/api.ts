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
  decision_threshold?: number;
  risk_level: RiskLevel;
  model_name: string;
  model_id?: string;
  model_source?: string;
  signals: PredictionSignal[];
}

export interface RiskSummary {
  status: "scored";
  source: string;
  data_kind: string;
  model_name: string;
  model_id?: string;
  decision_threshold?: number;
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

export interface MonthlyTrendPoint {
  month: string;
  label: string;
  count: number;
  volume: number;
  avg: number;
}

export interface DashboardStats {
  source: string;
  total_customers: number;
  historical_churn_rate: number;
  churned_customers: number;
  retained_customers: number;
  active_members: number;
  total_balance: number;
  avg_balance: number;
  country_distribution: Record<string, number>;
  gender_distribution: Record<string, number>;
  monthly_trend: MonthlyTrendPoint[];
  risk_summary?: RiskSummary;
  champion_model: string;
}

export interface ModelCheckpoint {
  id?: string;
  name: string;
  family?: string;
  type: string;
  available: boolean;
  status: string;
  has_weights?: boolean;
  has_results?: boolean;
  path: string;
  size_kb: number;
  modified_at: string | null;
  source_location?: string;
  train_command?: string;
  tune_command?: string;
  metrics?: Record<string, number> | null;
  parameters: Record<string, unknown>;
}

export interface ModelComparisonRow {
  model_id: string;
  model: string;
  family: string;
  available: boolean;
  status: "available" | "unavailable";
  source_location: string;
  roc_auc: number | null;
  pr_auc: number | null;
  f1: number | null;
  precision: number | null;
  recall: number | null;
  accuracy: number | null;
  balanced_accuracy?: number | null;
  cv_roc_auc?: number | null;
  cv_pr_auc?: number | null;
  train_command: string;
}

export interface DecileLiftRow {
  decile: string;
  customers: number;
  churners: number;
  churn_rate: string;
  decile_lift: string;
  cum_recall: string;
  cum_lift: string;
}

export interface DecileLiftTableResponse {
  model: string;
  available: boolean;
  source: string;
  deciles: DecileLiftRow[];
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
  relative_pct: number;
}

export interface FeatureImportanceResponse {
  features?: Record<string, FeatureImportanceItem[]>;
  sources?: Record<string, string>;
  available?: Record<string, boolean>;
  [model: string]: unknown;
}

export interface TrafficTelemetry {
  time: string;
  method: string;
  path: string;
  status: number;
  latency_ms: number;
}

export interface MonitoringTelemetry {
  status: string;
  uptime_seconds: number;
  process_memory_mb: number;
  cpu_percent: number;
  process_cpu_percent?: number;
  system_cpu_percent?: number;
  total_inferences: number;
  relational_tables: Record<string, number>;
  avg_latency_ms: number;
  recent_traffic: TrafficTelemetry[];
  model_loaded?: boolean;
  segmentation_loaded?: boolean;
}

export type GenericRecord = Record<string, unknown>;

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
// In browser or unified production server, same-origin relative paths (/api/...) are always valid.
export const isApiConfigured = API_BASE_URL.length > 0 || typeof window !== "undefined";

const DEFAULT_TIMEOUT_MS = 30000;

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
  health: (signal?: AbortSignal) => request<HealthResponse>("/health", { signal, timeoutMs: 15000 }),
  listCustomers: (params: { search?: string; limit?: number }, signal?: AbortSignal) => {
    const q = new URLSearchParams();
    if (params.search) q.set("search", params.search);
    q.set("limit", String(params.limit ?? 50));
    return request<CustomerFeatures[] | { items: CustomerFeatures[] }>(`/api/customers?${q}`, { signal });
  },
  getCustomer: (id: string, signal?: AbortSignal) =>
    request<CustomerFeatures>(`/api/customers/${encodeURIComponent(id)}`, { signal }),
  predict: (
    payload: CustomerFeatures,
    options?: { model?: string; threshold?: number },
    signal?: AbortSignal
  ) => {
    const q = new URLSearchParams();
    if (options?.model) q.set("model", options.model);
    if (options?.threshold !== undefined) q.set("threshold", String(options.threshold));
    const qs = q.toString() ? `?${q.toString()}` : "";
    return request<PredictionResult>(`/api/predict${qs}`, {
      method: "POST",
      body: {
        ...payload,
        customer_id: /^\d+$/.test(payload.customer_id) ? Number(payload.customer_id) : null,
        model_name: options?.model,
        decision_threshold: options?.threshold,
      },
      signal,
      timeoutMs: 20000,
    });
  },
  metrics: (signal?: AbortSignal) => request<GenericRecord>("/api/metrics", { signal }),
  riskSummary: (options?: { model?: string; threshold?: number }, signal?: AbortSignal) => {
    const q = new URLSearchParams();
    if (options?.model) q.set("model", options.model);
    if (options?.threshold !== undefined) q.set("threshold", String(options.threshold));
    const qs = q.toString() ? `?${q.toString()}` : "";
    return request<RiskSummary>(`/api/dashboard/risk-summary${qs}`, { signal });
  },
  dashboardStats: (signal?: AbortSignal) =>
    request<DashboardStats>("/api/dashboard/stats", { signal }),
  batchScore: (force = false, model = "xgboost", threshold = 0.50) =>
    request<RiskSummary>(
      `/api/predict/batch?force=${force ? "true" : "false"}&model=${encodeURIComponent(model)}&threshold=${encodeURIComponent(String(threshold))}`,
      { method: "POST", timeoutMs: 120000 }
    ),
  modelCheckpoints: (signal?: AbortSignal) =>
    request<ModelCheckpoint[]>("/api/models/checkpoints", { signal }),
  decileLift: (model: string = "xgboost", signal?: AbortSignal) =>
    request<DecileLiftTableResponse>(`/api/models/decile-lift?model=${encodeURIComponent(model)}`, { signal }),
  featureImportance: (signal?: AbortSignal) =>
    request<FeatureImportanceResponse>("/api/models/feature-importance", { signal }),
  segments: (algorithm: ClusterAlgorithm, signal?: AbortSignal) =>
    request<SegmentSummary>(`/api/segments?algorithm=${encodeURIComponent(algorithm)}`, { signal }),
  segmentCustomer: (id: string, signal?: AbortSignal) =>
    request<SegmentCustomer>(`/api/segments/customers/${encodeURIComponent(id)}`, { signal }),
  predictSegment: (payload: CustomerFeatures, algorithm: ClusterAlgorithm, signal?: AbortSignal) =>
    request<SegmentPrediction>(`/api/segments/predict?algorithm=${encodeURIComponent(algorithm)}`,
      { method: "POST", body: payload, signal }),
  monitoring: (signal?: AbortSignal) => request<MonitoringTelemetry>("/api/monitoring", { signal }),
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
