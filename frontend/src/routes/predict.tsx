import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { BrainCircuit, Loader2, RotateCcw, TrendingDown, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ErrorState, Notice, PageHeader, Panel, RiskBadge } from "@/components/common";
import { api, describeError, isApiConfigured, type CustomerFeatures } from "@/lib/api";
import { findDemoCustomer } from "@/lib/demo-data";
import { fmtPct } from "@/lib/format";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/predict")({
  validateSearch: (s: Record<string, unknown>): { customer_id?: string } => ({
    customer_id: typeof s.customer_id === "string" && s.customer_id ? s.customer_id : undefined,
  }),
  head: () => pageMeta("Dự đoán rời bỏ", "Chấm điểm xác suất rời bỏ của khách hàng bằng mô hình XGBoost."),
  component: PredictPage,
});

const EMPTY: CustomerFeatures = {
  customer_id: "", credit_score: 650, country: "France", gender: "Female", age: 40, tenure: 5, balance: 0,
  products_number: 1, credit_card: 1, active_member: 1, estimated_salary: 0, recency_days: 0, monthly_transactions: 0,
  trans_amount_3m: 0, trans_trend_ratio: 1, app_logins_monthly: 0, customer_service_complaints: 0,
  has_unresolved_complaint: 0, preferred_channel: "Mobile App", primary_device: "iOS",
};

const DEVICES = ["iOS", "Android", "Web/Desktop"];

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
  const [form, setForm] = useState<CustomerFeatures>(EMPTY);
  const [customDevice, setCustomDevice] = useState(false);
  const [prefillSource, setPrefillSource] = useState<"demo" | "api" | null>(null);
  const set = <K extends keyof CustomerFeatures>(k: K, v: CustomerFeatures[K]) => setForm((f) => ({ ...f, [k]: v }));
  const num = (k: keyof CustomerFeatures) => (e: React.ChangeEvent<HTMLInputElement>) =>
    set(k, (e.target.value === "" ? 0 : Number(e.target.value)) as never);

  const lookup = useQuery({
    queryKey: ["customer", customer_id],
    queryFn: ({ signal }) => api.getCustomer(customer_id!, signal),
    enabled: isApiConfigured && !!customer_id,
    retry: false,
  });

  useEffect(() => {
    if (!customer_id) return;
    const src = isApiConfigured ? lookup.data : findDemoCustomer(customer_id);
    if (src) {
      setForm(src);
      setCustomDevice(!DEVICES.includes(src.primary_device));
      setPrefillSource(isApiConfigured ? "api" : "demo");
    }
  }, [customer_id, lookup.data]);

  const m = useMutation({ mutationFn: (p: CustomerFeatures) => api.predict(p) });

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!isApiConfigured) return;
    m.mutate({ ...form, customer_id: form.customer_id.trim() });
  };

  return (
    <>
      <PageHeader eyebrow="Predict" title="Dự đoán khả năng rời bỏ" description="Nhập đầy đủ hồ sơ theo lược đồ của mô hình XGBoost. Kết quả chỉ đến từ API mô hình — không có ước tính cục bộ." />

      {prefillSource === "demo" && (
        <div className="mb-5"><Notice tone="warning" title={`Đã điền sẵn từ bản ghi Demo ${form.customer_id}`}>Dữ liệu mẫu tự tạo, không phải khách hàng thật.</Notice></div>
      )}

      <div className="grid gap-6 xl:grid-cols-[1fr_380px]">
        <form onSubmit={submit} className="space-y-5">
          <Panel title="1. Định danh & nhân khẩu học" subtitle="Identity & demographics">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <F label="Mã khách hàng (customer_id)"><Input required value={form.customer_id} onChange={(e) => set("customer_id", e.target.value)} placeholder="VD: 15634602" /></F>
              <F label="Quốc gia (country)">
                <Select value={form.country} onValueChange={(v) => set("country", v as CustomerFeatures["country"])}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>{["France", "Germany", "Spain"].map((v) => <SelectItem key={v} value={v}>{v}</SelectItem>)}</SelectContent>
                </Select>
              </F>
              <F label="Giới tính (gender)">
                <Select value={form.gender} onValueChange={(v) => set("gender", v as CustomerFeatures["gender"])}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="Female">Female</SelectItem><SelectItem value="Male">Male</SelectItem></SelectContent>
                </Select>
              </F>
              <F label="Tuổi (age)"><Input type="number" required min={18} max={100} value={form.age} onChange={num("age")} /></F>
              <F label="Điểm tín dụng (credit_score)"><Input type="number" required min={300} max={900} value={form.credit_score} onChange={num("credit_score")} /></F>
              <F label="Thu nhập ước tính (estimated_salary)"><Input type="number" required min={0} step="0.01" value={form.estimated_salary} onChange={num("estimated_salary")} /></F>
            </div>
          </Panel>

          <Panel title="2. Quan hệ sản phẩm" subtitle="Products & relationship">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <F label="Thâm niên năm (tenure)"><Input type="number" required min={0} max={50} value={form.tenure} onChange={num("tenure")} /></F>
              <F label="Số dư (balance)"><Input type="number" required min={0} step="0.01" value={form.balance} onChange={num("balance")} /></F>
              <F label="Số sản phẩm (products_number)"><Input type="number" required min={1} max={10} value={form.products_number} onChange={num("products_number")} /></F>
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {([["credit_card", "Có thẻ tín dụng (credit_card)"], ["active_member", "Thành viên tích cực (active_member)"]] as const).map(([k, l]) => (
                <label key={k} className="flex items-center justify-between rounded-xl border bg-muted/50 px-4 py-3 text-sm">
                  {l}<Switch checked={form[k] === 1} onCheckedChange={(v) => set(k, v ? 1 : 0)} />
                </label>
              ))}
            </div>
          </Panel>

          <Panel title="3. Hành vi giao dịch & số" subtitle="Transactions & digital engagement">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <F label="Ngày từ GD cuối (recency_days)"><Input type="number" required min={0} value={form.recency_days} onChange={num("recency_days")} /></F>
              <F label="Giao dịch/tháng (monthly_transactions)"><Input type="number" required min={0} value={form.monthly_transactions} onChange={num("monthly_transactions")} /></F>
              <F label="Giá trị GD 3 tháng (trans_amount_3m)"><Input type="number" required min={0} step="0.01" value={form.trans_amount_3m} onChange={num("trans_amount_3m")} /></F>
              <F label="Tỷ lệ xu hướng GD (trans_trend_ratio)" hint="< 1 nghĩa là giao dịch đang chậm lại"><Input type="number" required min={0} step="0.01" value={form.trans_trend_ratio} onChange={num("trans_trend_ratio")} /></F>
              <F label="Đăng nhập app/tháng (app_logins_monthly)"><Input type="number" required min={0} value={form.app_logins_monthly} onChange={num("app_logins_monthly")} /></F>
              <F label="Kênh ưa thích (preferred_channel)">
                <Select value={form.preferred_channel} onValueChange={(v) => set("preferred_channel", v as CustomerFeatures["preferred_channel"])}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>{["Mobile App", "Online Web", "ATM", "Branch"].map((v) => <SelectItem key={v} value={v}>{v}</SelectItem>)}</SelectContent>
                </Select>
              </F>
              <F label="Thiết bị chính (primary_device)">
                {customDevice ? (
                  <Input required value={form.primary_device} onChange={(e) => set("primary_device", e.target.value)} placeholder="Nhập giá trị" />
                ) : (
                  <Select value={form.primary_device} onValueChange={(v) => (v === "__other" ? (setCustomDevice(true), set("primary_device", "")) : set("primary_device", v))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {DEVICES.map((v) => <SelectItem key={v} value={v}>{v}</SelectItem>)}
                      <SelectItem value="__other">Khác (nhập tay)…</SelectItem>
                    </SelectContent>
                  </Select>
                )}
                {customDevice && (
                  <button type="button" className="text-[11px] font-medium text-primary" onClick={() => { setCustomDevice(false); set("primary_device", "iOS"); }}>Chọn từ danh sách</button>
                )}
              </F>
            </div>
          </Panel>

          <Panel title="4. Dịch vụ khách hàng" subtitle="Service & complaints">
            <div className="grid gap-4 sm:grid-cols-2">
              <F label="Số khiếu nại (customer_service_complaints)"><Input type="number" required min={0} value={form.customer_service_complaints} onChange={num("customer_service_complaints")} /></F>
              <label className="flex items-center justify-between self-end rounded-xl border bg-muted/50 px-4 py-2.5 text-sm">
                Khiếu nại chưa xử lý (has_unresolved_complaint)
                <Switch checked={form.has_unresolved_complaint === 1} onCheckedChange={(v) => set("has_unresolved_complaint", v ? 1 : 0)} />
              </label>
            </div>
          </Panel>

          <div className="flex flex-wrap items-center gap-3">
            <Button type="submit" size="lg" disabled={!isApiConfigured || m.isPending}>
              {m.isPending ? <Loader2 className="animate-spin" /> : <BrainCircuit />} Chấm điểm
            </Button>
            <Button type="button" variant="outline" size="lg" onClick={() => { setForm(EMPTY); setCustomDevice(false); setPrefillSource(null); m.reset(); }}>
              <RotateCcw /> Đặt lại
            </Button>
            {!isApiConfigured && <span className="text-xs text-muted-foreground">Cần kết nối API để chấm điểm.</span>}
          </div>
        </form>

        <div className="xl:sticky xl:top-24 xl:self-start">
          <Panel title="Kết quả mô hình" subtitle="Model output">
            {!isApiConfigured ? (
              <Notice tone="warning" title="API chưa kết nối">Không thể chấm điểm. Ứng dụng không tạo kết quả giả hay ước tính cục bộ.</Notice>
            ) : m.isPending ? (
              <div className="flex flex-col items-center py-12 text-sm text-muted-foreground"><Loader2 className="mb-3 h-6 w-6 animate-spin text-primary" />Đang gọi mô hình…</div>
            ) : m.isError ? (
              <ErrorState message={describeError(m.error)} onRetry={() => m.mutate(form)} />
            ) : m.data ? (
              <ResultView r={m.data} />
            ) : (
              <p className="py-10 text-center text-sm text-muted-foreground">Điền biểu mẫu và bấm “Chấm điểm”.</p>
            )}
          </Panel>
        </div>
      </div>
    </>
  );
}

function ResultView({ r }: { r: import("@/lib/api").PredictionResult }) {
  const pct = Math.max(0, Math.min(1, r.churn_probability));
  return (
    <div>
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs text-muted-foreground">{r.customer_id}</span>
        <RiskBadge level={r.risk_level} />
      </div>
      <p className="mt-4 text-xs text-muted-foreground">Xác suất rời bỏ</p>
      <p className="font-display text-5xl font-bold tabular text-primary">{fmtPct(pct, 1)}</p>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-lilac"><div className="h-full rounded-full bg-primary" style={{ width: `${pct * 100}%` }} /></div>
      <p className="mt-3 text-sm">Dự đoán: <strong>{r.predicted_churn ? "Có khả năng rời bỏ" : "Có khả năng ở lại"}</strong></p>
      <p className="text-[11px] text-muted-foreground">Mô hình: {r.model_name}</p>
      {r.signals?.length > 0 && (
        <div className="mt-5 space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Tín hiệu</p>
          {r.signals.map((s, i) => {
            const up = /up|increase|tăng|risk|positive/i.test(s.direction);
            return (
              <div key={i} className="flex gap-3 rounded-xl bg-muted p-3">
                {up ? <TrendingUp className="h-4 w-4 shrink-0 text-danger" /> : <TrendingDown className="h-4 w-4 shrink-0 text-success" />}
                <div><p className="text-sm font-medium">{s.title}</p><p className="text-xs text-muted-foreground">{s.description}</p></div>
              </div>
            );
          })}
        </div>
      )}
      <Button variant="outline" size="sm" className="mt-5 w-full" asChild><Link to="/retention">Xem gợi ý giữ chân</Link></Button>
    </div>
  );
}