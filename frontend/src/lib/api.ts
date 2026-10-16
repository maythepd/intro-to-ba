/**
 * Typed client for the VioletBank churn model API.
 * Base URL comes from VITE_API_BASE_URL. When it is not set, every call throws
 * ApiNotConfiguredError — callers must never substitute fabricated results.
 */

export type Country = "France" | "Germany" | "Spain";
export type Gender = "Male" | "Female";
export type PreferredChannel = "Mobile App" | "Online Web" | "ATM" | "Branch";

/** Exact schema used by the model/xgboost branch. */
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
}

export interface HealthResponse {
  status?: string;
  [key: string]: unknown;
}

export type GenericRecord = Record<string, unknown>;

export class ApiNotConfiguredError extends Error {
  constructor() {
    super("API chưa được cấu hình (VITE_API_BASE_URL).");
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
    super("Yêu cầu quá thời gian chờ.");
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
  health: (signal?: AbortSignal) => request<HealthResponse>("/health", { signal, timeoutMs: 6000 }),
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
  segments: (signal?: AbortSignal) => request<GenericRecord | GenericRecord[]>("/api/segments", { signal }),
  monitoring: (signal?: AbortSignal) => request<GenericRecord>("/api/monitoring", { signal }),
};

export function normalizeCustomerList(data: CustomerFeatures[] | { items: CustomerFeatures[] }): CustomerFeatures[] {
  return Array.isArray(data) ? data : (data?.items ?? []);
}

export function describeError(err: unknown): string {
  if (err instanceof ApiNotConfiguredError) return err.message;
  if (err instanceof ApiTimeoutError) return "Máy chủ không phản hồi kịp thời. Vui lòng thử lại.";
  if (err instanceof ApiError) return `Lỗi máy chủ (${err.status}): ${err.message}`;
  if (err instanceof TypeError) return "Không thể kết nối tới API. Kiểm tra mạng hoặc cấu hình CORS.";
  return err instanceof Error ? err.message : "Lỗi không xác định.";
}