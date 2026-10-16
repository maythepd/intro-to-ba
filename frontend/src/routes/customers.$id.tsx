import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, BrainCircuit } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DemoBadge, EmptyState, ErrorState, PageHeader, Panel } from "@/components/common";
import { api, ApiError, describeError, isApiConfigured, type CustomerFeatures } from "@/lib/api";
import { findDemoCustomer } from "@/lib/demo-data";
import { fmtInt, fmtMoney, yesNo } from "@/lib/format";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/customers/$id")({
  head: ({ params }) => pageMeta(`Khách hàng ${params.id}`, "Hồ sơ nhân khẩu học và hành vi giao dịch của khách hàng."),
  component: CustomerDetail,
});

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-dashed py-2.5 last:border-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm font-medium tabular">{value}</span>
    </div>
  );
}

function CustomerDetail() {
  const { id } = Route.useParams();
  const q = useQuery({
    queryKey: ["customer", id],
    queryFn: ({ signal }) => api.getCustomer(id, signal),
    enabled: isApiConfigured,
    retry: false,
  });
  const demo = !isApiConfigured ? findDemoCustomer(id) : undefined;
  const c: CustomerFeatures | undefined = isApiConfigured ? q.data : demo;

  const back = (
    <Button variant="ghost" size="sm" asChild className="mb-4 -ml-2"><Link to="/customers"><ArrowLeft /> Danh sách khách hàng</Link></Button>
  );

  if (isApiConfigured && q.isLoading) return <>{back}<div className="h-64 animate-pulse rounded-2xl bg-muted" /></>;
  if (isApiConfigured && q.isError) {
    if (q.error instanceof ApiError && q.error.status === 404)
      return <>{back}<EmptyState icon={<BrainCircuit />} title="Không tìm thấy khách hàng">Mã {id} không tồn tại trong hệ thống.</EmptyState></>;
    return <>{back}<ErrorState message={describeError(q.error)} onRetry={() => q.refetch()} retrying={q.isFetching} /></>;
  }
  if (!c) {
    return (
      <>{back}
        <EmptyState icon={<BrainCircuit />} title="Không có hồ sơ để hiển thị">
          API chưa kết nối và mã {id} không thuộc bản ghi Demo.
        </EmptyState>
      </>
    );
  }

  return (
    <>
      {back}
      <PageHeader
        eyebrow="Customer profile"
        title={c.customer_id}
        description={<span className="inline-flex items-center gap-2">{c.country} · {c.gender === "Female" ? "Nữ" : "Nam"} · {c.age} tuổi {demo && <DemoBadge />}</span>}
        actions={<Button asChild><Link to="/predict" search={{ customer_id: c.customer_id }}><BrainCircuit /> Chấm điểm rủi ro</Link></Button>}
      />
      {demo && <p className="-mt-4 mb-6 text-xs text-muted-foreground">Bản ghi Demo do nhóm phát triển tự tạo — không phải khách hàng thật.</p>}

      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["Số dư", fmtMoney(c.balance)],
          ["Giao dịch/tháng", fmtInt(c.monthly_transactions)],
          ["Giá trị GD 3 tháng", fmtMoney(c.trans_amount_3m)],
          ["Xu hướng giao dịch", c.trans_trend_ratio.toFixed(2)],
        ].map(([l, v]) => (
          <div key={l} className="rounded-2xl border bg-card p-5 shadow-soft">
            <p className="text-xs text-muted-foreground">{l}</p>
            <p className="mt-1 font-display text-2xl font-bold tabular">{v}</p>
          </div>
        ))}
      </div>

      <div className="mt-6 grid gap-5 lg:grid-cols-3">
        <Panel title="Nhân khẩu học" subtitle="Demographics">
          <Field label="Quốc gia" value={c.country} />
          <Field label="Giới tính" value={c.gender} />
          <Field label="Tuổi" value={c.age} />
          <Field label="Thu nhập ước tính" value={fmtMoney(c.estimated_salary)} />
          <Field label="Điểm tín dụng" value={c.credit_score} />
        </Panel>
        <Panel title="Quan hệ sản phẩm" subtitle="Products & tenure">
          <Field label="Thâm niên (năm)" value={c.tenure} />
          <Field label="Số sản phẩm" value={c.products_number} />
          <Field label="Thẻ tín dụng" value={yesNo(c.credit_card)} />
          <Field label="Thành viên tích cực" value={yesNo(c.active_member)} />
          <Field label="Số dư" value={fmtMoney(c.balance)} />
        </Panel>
        <Panel title="Hành vi & dịch vụ" subtitle="Behavior">
          <Field label="Ngày từ giao dịch cuối" value={c.recency_days} />
          <Field label="Đăng nhập app/tháng" value={c.app_logins_monthly} />
          <Field label="Khiếu nại" value={c.customer_service_complaints} />
          <Field label="Khiếu nại chưa xử lý" value={c.has_unresolved_complaint ? <span className="text-danger">Có</span> : "Không"} />
          <Field label="Kênh ưa thích" value={c.preferred_channel} />
          <Field label="Thiết bị chính" value={c.primary_device} />
        </Panel>
      </div>
    </>
  );
}