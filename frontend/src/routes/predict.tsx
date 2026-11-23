import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  Gauge,
  Layers,
  LineChart,
  Loader2,
  RotateCcw,
  Sparkles,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ErrorState, Notice, PageHeader, Panel, RiskBadge } from "@/components/common";
import { api, describeError, isApiConfigured, type CustomerFeatures } from "@/lib/api";
import { fmtPct } from "@/lib/format";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/predict")({
  validateSearch: (s: Record<string, unknown>): { customer_id?: string } => ({
    customer_id: typeof s.customer_id === "string" && s.customer_id ? s.customer_id : undefined,
  }),
  head: () => pageMeta("Churn Prediction", "Predict customer churn probability with selectable models and custom decision thresholds."),
  component: PredictPage,
});

const EMPTY: CustomerFeatures = {
  customer_id: "", credit_score: 650, country: "France", gender: "Female", age: 40, tenure: 5, balance: 0,
  products_number: 1, credit_card: 1, active_member: 1, estimated_salary: 0, recency_days: 1, monthly_transactions: 0,
  trans_amount_3m: 0, trans_trend_ratio: 1, app_logins_monthly: 0, customer_service_complaints: 0,
  has_unresolved_complaint: 0, preferred_channel: "Mobile App", primary_device: "iOS",
};

const DEVICES = ["iOS", "Android", "Desktop", "No_Device", "Web/Desktop"];

function F({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs font-medium">{label}</Label>
      {children}
      {hint && <p className="text-[11px] text-muted-foreground">{hint}</p>}
    </div>
  );
}

function PredictPage() {
  const { customer_id } = Route.useSearch();
  const [lookupId, setLookupId] = useState(customer_id ?? "");
  const [form, setForm] = useState<CustomerFeatures>(EMPTY);
  const [customDevice, setCustomDevice] = useState(false);
  const [prefillSource, setPrefillSource] = useState<"demo" | "api" | null>(null);

  // Model & Decision Threshold Controls
  const [selectedModel, setSelectedModel] = useState<"xgboost" | "random_forest" | "logistic_regression">("xgboost");
  const [threshold, setThreshold] = useState<number>(0.50);

  const set = <K extends keyof CustomerFeatures>(k: K, v: CustomerFeatures[K]) => setForm((f) => ({ ...f, [k]: v }));
  const num = (k: keyof CustomerFeatures) => (e: React.ChangeEvent<HTMLInputElement>) =>
    set(k, (e.target.value === "" ? 0 : Number(e.target.value)) as never);

  const lookup = useQuery({
    queryKey: ["customer", lookupId],
    queryFn: ({ signal }) => api.getCustomer(lookupId, signal),
    enabled: isApiConfigured && !!lookupId,
    retry: false,
  });

  useEffect(() => {
    setLookupId(customer_id ?? "");
  }, [customer_id]);

  useEffect(() => {
    const src = lookup.data;
    if (src) {
      setForm(src);
      setCustomDevice(!DEVICES.includes(src.primary_device));
      setPrefillSource("api");
    }
  }, [lookup.data]);

  const m = useMutation({
    mutationFn: (p: CustomerFeatures) =>
      api.predict(p, { model: selectedModel, threshold }),
  });

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!isApiConfigured) return;
    m.mutate({ ...form, customer_id: /^\d+$/.test(form.customer_id.trim()) ? form.customer_id.trim() : "" });
  };

  // Quick preset thresholds
  const applyPreset = (th: number) => {
    setThreshold(th);
    if (m.data) {
      m.mutate({ ...form, customer_id: /^\d+$/.test(form.customer_id.trim()) ? form.customer_id.trim() : "" });
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Predict"
        title="Customer Churn Prediction & Decision Engine"
        description="Score churn probability with full model flexibility (XGBoost, Random Forest, Logistic Regression) and adjustable decision thresholds."
      />

      <Notice title="Multi-Model Inference & Threshold Modulation">
        Hệ thống hỗ trợ lựa chọn giữa <strong>XGBoost Champion</strong>, <strong>Random Forest</strong>, và <strong>Logistic Regression SAGA</strong> kèm <strong>ngưỡng ra quyết định (Decision Threshold $\tau$)</strong> tùy biến từ 10% đến 90%. Hãy điều chỉnh $\tau$ để cân đối giữa độ nhạy phát hiện sớm (Recall) và độ chính xác ngân sách giữ chân (Precision).
      </Notice>
      {lookup.isError && (
        <div className="mt-3">
          <ErrorState message={describeError(lookup.error)} onRetry={() => lookup.refetch()} />
        </div>
      )}

      {prefillSource === "demo" && (
        <div className="mb-5">
          <Notice tone="warning" title={`Prefilled from demo record ${form.customer_id}`}>
            Synthetic preview data, not a real customer.
          </Notice>
        </div>
      )}

      {/* Model & Decision Threshold Configuration Panel */}
      <Panel
        className="mt-6 mb-6 border-primary/20 bg-lilac/5 shadow-sm"
        title="Decision Engine & Model Selection"
        subtitle="Chọn thuật toán mô hình dự đoán và điều chỉnh ngưỡng xác suất phân loại (Decision Threshold τ)"
      >
        <div className="grid gap-6 md:grid-cols-2">
          {/* Model Selector */}
          <div>
            <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Mô hình suy luận (Inference Model)
            </Label>
            <div className="mt-2 grid grid-cols-3 gap-2">
              <button
                type="button"
                onClick={() => setSelectedModel("xgboost")}
                className={`flex flex-col items-start rounded-xl border p-3 text-left transition-all ${
                  selectedModel === "xgboost"
                    ? "border-primary bg-primary/10 shadow-sm"
                    : "border-border/60 bg-background hover:bg-muted/50"
                }`}
              >
                <div className="flex items-center gap-1.5 text-xs font-bold text-foreground">
                  <Sparkles className="h-3.5 w-3.5 text-primary" />
                  XGBoost
                </div>
                <span className="mt-1 text-[11px] text-muted-foreground">Production Champion (ROC 0.8513)</span>
                <span className="mt-2 inline-flex rounded bg-primary/20 px-1.5 py-0.5 text-[9px] font-bold text-primary">
                  Champion
                </span>
              </button>

              <button
                type="button"
                onClick={() => setSelectedModel("random_forest")}
                className={`flex flex-col items-start rounded-xl border p-3 text-left transition-all ${
                  selectedModel === "random_forest"
                    ? "border-primary bg-primary/10 shadow-sm"
                    : "border-border/60 bg-background hover:bg-muted/50"
                }`}
              >
                <div className="flex items-center gap-1.5 text-xs font-bold text-foreground">
                  <Layers className="h-3.5 w-3.5 text-primary" />
                  Random Forest
                </div>
                <span className="mt-1 text-[11px] text-muted-foreground">Bagging Ensemble (500 Trees)</span>
                <span className="mt-2 inline-flex rounded bg-muted px-1.5 py-0.5 text-[9px] font-semibold text-muted-foreground">
                  Ensemble
                </span>
              </button>

              <button
                type="button"
                onClick={() => setSelectedModel("logistic_regression")}
                className={`flex flex-col items-start rounded-xl border p-3 text-left transition-all ${
                  selectedModel === "logistic_regression"
                    ? "border-primary bg-primary/10 shadow-sm"
                    : "border-border/60 bg-background hover:bg-muted/50"
                }`}
              >
                <div className="flex items-center gap-1.5 text-xs font-bold text-foreground">
                  <LineChart className="h-3.5 w-3.5 text-primary" />
                  Logistic Reg
                </div>
                <span className="mt-1 text-[11px] text-muted-foreground">Linear SAGA (Recall 83.3%)</span>
                <span className="mt-2 inline-flex rounded bg-muted px-1.5 py-0.5 text-[9px] font-semibold text-muted-foreground">
                  Linear Baseline
                </span>
              </button>
            </div>
          </div>

          {/* Decision Threshold Slider & Presets */}
          <div>
            <div className="flex items-center justify-between">
              <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Ngưỡng ra quyết định: <code className="text-xs font-bold text-primary">τ = {threshold.toFixed(2)} ({Math.round(threshold * 100)}%)</code>
              </Label>
              <span className="text-[11px] text-muted-foreground">
                {threshold <= 0.35
                  ? "⚡ Chế độ Cảnh báo sớm (High Recall)"
                  : threshold >= 0.65
                  ? "🎯 Chế độ Chọn lọc cao (High Precision)"
                  : "⚖️ Chế độ Tiêu chuẩn (Balanced)"}
              </span>
            </div>

            <div className="mt-3 flex items-center gap-3">
              <span className="text-xs font-mono text-muted-foreground">0.10</span>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-muted accent-primary"
              />
              <span className="text-xs font-mono text-muted-foreground">0.90</span>
            </div>

            {/* Threshold Preset Pills */}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-medium text-muted-foreground">Presets:</span>
              <button
                type="button"
                onClick={() => applyPreset(0.35)}
                className={`rounded-full px-2.5 py-1 text-xs font-semibold transition-all ${
                  threshold === 0.35
                    ? "bg-amber-500 text-white shadow-sm"
                    : "bg-muted text-muted-foreground hover:bg-muted/80"
                }`}
              >
                High Recall (τ = 0.35)
              </button>
              <button
                type="button"
                onClick={() => applyPreset(0.50)}
                className={`rounded-full px-2.5 py-1 text-xs font-semibold transition-all ${
                  threshold === 0.50
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "bg-muted text-muted-foreground hover:bg-muted/80"
                }`}
              >
                Balanced (τ = 0.50)
              </button>
              <button
                type="button"
                onClick={() => applyPreset(0.65)}
                className={`rounded-full px-2.5 py-1 text-xs font-semibold transition-all ${
                  threshold === 0.65
                    ? "bg-emerald-600 text-white shadow-sm"
                    : "bg-muted text-muted-foreground hover:bg-muted/80"
                }`}
              >
                High Precision (τ = 0.65)
              </button>
            </div>
          </div>
        </div>
      </Panel>

      <div className="grid gap-6 xl:grid-cols-[1fr_380px]">
        <form onSubmit={submit} className="space-y-5">
          <Panel title="1. Identity and Demographics" subtitle="Identity & demographics">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <F label="Customer ID (customer_id)">
                <div className="flex gap-2">
                  <Input
                    required
                    value={form.customer_id}
                    onChange={(e) => set("customer_id", e.target.value)}
                    placeholder="e.g. 15634602"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    disabled={!isApiConfigured || !/^\d+$/.test(form.customer_id.trim()) || lookup.isFetching}
                    onClick={() => {
                      const id = form.customer_id.trim();
                      if (id === lookupId) lookup.refetch();
                      else setLookupId(id);
                    }}
                  >
                    {lookup.isFetching ? <Loader2 className="animate-spin" /> : "Load"}
                  </Button>
                </div>
              </F>
              <F label="Country (country)">
                <Select value={form.country} onValueChange={(v) => set("country", v as CustomerFeatures["country"])}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["France", "Germany", "Spain"].map((v) => <SelectItem key={v} value={v}>{v}</SelectItem>)}
                  </SelectContent>
                </Select>
              </F>
              <F label="Gender (gender)">
                <Select value={form.gender} onValueChange={(v) => set("gender", v as CustomerFeatures["gender"])}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Female">Female</SelectItem>
                    <SelectItem value="Male">Male</SelectItem>
                  </SelectContent>
                </Select>
              </F>
              <F label="Age (age)"><Input type="number" required min={18} max={100} value={form.age} onChange={num("age")} /></F>
              <F label="Tenure (tenure)"><Input type="number" required min={0} max={50} value={form.tenure} onChange={num("tenure")} /></F>
              <F label="Credit Score (credit_score)"><Input type="number" required min={300} max={850} value={form.credit_score} onChange={num("credit_score")} /></F>
            </div>
          </Panel>

          <Panel title="2. Financial Profile" subtitle="Account and balance details">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <F label="Account Balance (balance)"><Input type="number" required min={0} step="0.01" value={form.balance} onChange={num("balance")} /></F>
              <F label="Estimated Salary (estimated_salary)"><Input type="number" required min={0} step="0.01" value={form.estimated_salary} onChange={num("estimated_salary")} /></F>
              <F label="Products (products_number)"><Input type="number" required min={1} max={4} value={form.products_number} onChange={num("products_number")} /></F>
              <label className="flex items-center justify-between self-end rounded-xl border bg-muted/50 px-4 py-2.5 text-sm">
                Has Credit Card (credit_card)
                <Switch checked={form.credit_card === 1} onCheckedChange={(c) => set("credit_card", c ? 1 : 0)} />
              </label>
            </div>
          </Panel>

          <Panel title="3. Behavioral Signals" subtitle="Engagement & interaction details">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <label className="flex items-center justify-between rounded-xl border bg-muted/50 px-4 py-2.5 text-sm">
                Active Member (active_member)
                <Switch checked={form.active_member === 1} onCheckedChange={(c) => set("active_member", c ? 1 : 0)} />
              </label>
              <F label="Recency Days (recency_days)"><Input type="number" required min={0} value={form.recency_days} onChange={num("recency_days")} /></F>
              <F label="Monthly Transactions (monthly_transactions)"><Input type="number" required min={0} value={form.monthly_transactions} onChange={num("monthly_transactions")} /></F>
              <F label="3-Month Transaction Volume (trans_amount_3m)"><Input type="number" required min={0} step="0.01" value={form.trans_amount_3m} onChange={num("trans_amount_3m")} /></F>
              <F label="Transaction Trend Ratio (trans_trend_ratio)"><Input type="number" required min={0} step="0.01" value={form.trans_trend_ratio} onChange={num("trans_trend_ratio")} /></F>
              <F label="App Logins Monthly (app_logins_monthly)"><Input type="number" required min={0} value={form.app_logins_monthly} onChange={num("app_logins_monthly")} /></F>
              <F label="Preferred Channel (preferred_channel)">
                <Select value={form.preferred_channel} onValueChange={(v) => set("preferred_channel", v as CustomerFeatures["preferred_channel"])}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["Mobile App", "Online Web", "ATM", "Branch", "No_Channel"].map((v) => <SelectItem key={v} value={v}>{v}</SelectItem>)}
                  </SelectContent>
                </Select>
              </F>
              <F label="Primary Device (primary_device)">
                {customDevice ? (
                  <Input required value={form.primary_device} onChange={(e) => set("primary_device", e.target.value)} placeholder="Enter a value" />
                ) : (
                  <Select
                    value={form.primary_device}
                    onValueChange={(v) => (v === "__other" ? (setCustomDevice(true), set("primary_device", "")) : set("primary_device", v))}
                  >
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {DEVICES.map((v) => <SelectItem key={v} value={v}>{v}</SelectItem>)}
                      <SelectItem value="__other">Other (enter manually)…</SelectItem>
                    </SelectContent>
                  </Select>
                )}
                {customDevice && (
                  <button type="button" className="text-[11px] font-medium text-primary" onClick={() => { setCustomDevice(false); set("primary_device", "iOS"); }}>Choose from list</button>
                )}
              </F>
            </div>
          </Panel>

          <Panel title="4. Customer Service" subtitle="Service & complaints">
            <div className="grid gap-4 sm:grid-cols-2">
              <F label="Complaint Count (customer_service_complaints)">
                <Input type="number" required min={0} value={form.customer_service_complaints} onChange={num("customer_service_complaints")} />
              </F>
              <label className="flex items-center justify-between self-end rounded-xl border bg-muted/50 px-4 py-2.5 text-sm">
                Unresolved Complaint (has_unresolved_complaint)
                <Switch checked={form.has_unresolved_complaint === 1} disabled />
              </label>
            </div>
          </Panel>

          <div className="flex flex-wrap items-center gap-3">
            <Button type="submit" size="lg" disabled={!isApiConfigured || m.isPending}>
              {m.isPending ? <Loader2 className="animate-spin" /> : <BrainCircuit />} Run Churn Inference
            </Button>
            <Button
              type="button"
              variant="outline"
              size="lg"
              onClick={() => {
                setForm(EMPTY);
                setLookupId("");
                setCustomDevice(false);
                setPrefillSource(null);
                m.reset();
              }}
            >
              <RotateCcw /> Reset
            </Button>
            {!isApiConfigured && <span className="text-xs text-muted-foreground">Connect the API to run predictions.</span>}
          </div>
        </form>

        <div className="xl:sticky xl:top-24 xl:self-start">
          <Panel title="Inference Decision" subtitle="Kết quả phân loại theo ngưỡng τ">
            {!isApiConfigured ? (
              <Notice tone="warning" title="API Not Connected">
                Unable to score customers. Connect the API to run predictions.
              </Notice>
            ) : m.isPending ? (
              <div className="flex flex-col items-center py-12 text-sm text-muted-foreground">
                <Loader2 className="mb-3 h-6 w-6 animate-spin text-primary" />
                Đang chạy mô hình {selectedModel} với ngưỡng τ = {threshold.toFixed(2)}…
              </div>
            ) : m.isError ? (
              <ErrorState
                message={describeError(m.error)}
                onRetry={() => m.mutate({ ...form, customer_id: /^\d+$/.test(form.customer_id.trim()) ? form.customer_id.trim() : "" })}
              />
            ) : m.data ? (
              <ResultView r={m.data} activeThreshold={threshold} />
            ) : (
              <p className="py-10 text-center text-sm text-muted-foreground">
                Nhập hồ sơ khách hàng, chọn mô hình & ngưỡng rồi bấm “Run Churn Inference”.
              </p>
            )}
          </Panel>
        </div>
      </div>
    </>
  );
}

function ResultView({
  r,
  activeThreshold,
}: {
  r: import("@/lib/api").PredictionResult;
  activeThreshold: number;
}) {
  const pct = Math.max(0, Math.min(1, r.churn_probability));
  const th = r.decision_threshold ?? activeThreshold;
  const thPct = Math.max(0, Math.min(1, th));
  const isLikelyToChurn = r.predicted_churn;

  return (
    <div>
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs text-muted-foreground">{r.customer_id || "Unsaved Profile"}</span>
        <RiskBadge level={r.risk_level} />
      </div>

      <p className="mt-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Xác suất Rời bỏ (Churn Probability)
      </p>
      <div className="flex items-baseline justify-between">
        <p className="font-display text-5xl font-bold tabular text-primary">{fmtPct(pct, 1)}</p>
        <span className="text-xs font-mono text-muted-foreground">τ = {th.toFixed(2)}</span>
      </div>

      {/* Visual Probability Bar with Threshold Marker */}
      <div className="relative mt-4 h-3 w-full rounded-full bg-lilac/40 overflow-hidden">
        {/* Fill bar */}
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            isLikelyToChurn ? "bg-danger" : "bg-primary"
          }`}
          style={{ width: `${pct * 100}%` }}
        />
      </div>

      {/* Threshold Pin Guide */}
      <div className="relative mt-1 text-[11px] font-mono text-muted-foreground flex justify-between">
        <span>0%</span>
        <span
          className="absolute -translate-x-1/2 font-semibold text-foreground flex flex-col items-center"
          style={{ left: `${thPct * 100}%` }}
        >
          <span>▲</span>
          <span className="text-[10px] bg-muted px-1 rounded">Cutoff {Math.round(th * 100)}%</span>
        </span>
        <span>100%</span>
      </div>

      {/* Classification Outcome Card */}
      <div
        className={`mt-6 rounded-xl border p-3.5 ${
          isLikelyToChurn
            ? "border-danger/30 bg-danger-soft/20 text-danger-foreground"
            : "border-success/30 bg-success-soft/20 text-success-foreground"
        }`}
      >
        <div className="flex items-center gap-2 font-bold text-sm">
          {isLikelyToChurn ? (
            <>
              <AlertTriangle className="h-4 w-4 text-danger" />
              <span>Likely to Churn (Có nguy cơ rời bỏ)</span>
            </>
          ) : (
            <>
              <CheckCircle2 className="h-4 w-4 text-success" />
              <span>Likely to Stay (Duy trì hoạt động)</span>
            </>
          )}
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          {isLikelyToChurn
            ? `Xác suất (${fmtPct(pct, 1)}) ≥ ngưỡng ra quyết định τ (${fmtPct(th, 0)}). Khuyến nghị triển khai chiến lược can thiệp giữ chân ngay.`
            : `Xác suất (${fmtPct(pct, 1)}) < ngưỡng ra quyết định τ (${fmtPct(th, 0)}). Khách hàng an toàn ở mức phân loại hiện tại.`}
        </p>
      </div>

      {/* Model Attribution Metadata */}
      <div className="mt-4 rounded-lg bg-muted/60 p-2.5 text-xs text-muted-foreground space-y-1">
        <div className="flex justify-between">
          <span>Mô hình sử dụng:</span>
          <span className="font-semibold text-foreground">{r.model_name}</span>
        </div>
        <div className="flex justify-between">
          <span>Quy tắc phân loại:</span>
          <span className="font-mono text-foreground">Churn = P(churn) ≥ {th.toFixed(2)}</span>
        </div>
      </div>

      {/* Behavioral Signals */}
      {r.signals?.length > 0 && (
        <div className="mt-5 space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Tín hiệu hành vi</p>
          {r.signals.map((s, i) => {
            const up = /up|increase|risk|positive/i.test(s.direction);
            return (
              <div key={i} className="flex gap-3 rounded-xl bg-muted p-3">
                {up ? (
                  <TrendingUp className="h-4 w-4 shrink-0 text-danger" />
                ) : (
                  <TrendingDown className="h-4 w-4 shrink-0 text-success" />
                )}
                <div>
                  <p className="text-sm font-medium">{s.title}</p>
                  <p className="text-xs text-muted-foreground">{s.description}</p>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <Button variant="outline" size="sm" className="mt-5 w-full" asChild>
        <Link to="/retention">Xem khuyến nghị giữ chân (Retention Actions)</Link>
      </Button>
    </div>
  );
}
