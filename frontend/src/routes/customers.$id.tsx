import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, BrainCircuit } from "lucide-react";
import { Button } from "@/components/ui/button";
import { EmptyState, ErrorState, PageHeader, Panel } from "@/components/common";
import { api, ApiError, describeError, isApiConfigured, type CustomerFeatures } from "@/lib/api";
import { fmtInt, fmtMoney, yesNo } from "@/lib/format";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/customers/$id")({
  head: ({ params }) => pageMeta(`Customers ${params.id}`, "Customer demographics and transaction behavior."),
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
  const c: CustomerFeatures | undefined = q.data;

  const back = (
    <Button variant="ghost" size="sm" asChild className="mb-4 -ml-2"><Link to="/customers"><ArrowLeft /> Customer Directory</Link></Button>
  );

  if (isApiConfigured && q.isLoading) return <>{back}<div className="h-64 animate-pulse rounded-2xl bg-muted" /></>;
  if (isApiConfigured && q.isError) {
    if (q.error instanceof ApiError && q.error.status === 404)
      return <>{back}<EmptyState icon={<BrainCircuit />} title="Customer Not Found">Customer ID {id} was not found in the system.</EmptyState></>;
    return <>{back}<ErrorState message={describeError(q.error)} onRetry={() => q.refetch()} retrying={q.isFetching} /></>;
  }
  if (!c) {
    return (
      <>{back}
        <EmptyState icon={<BrainCircuit />} title="No Customer Profile Available">
          The API is not connected and customer ID {id} is not available.
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
        description={<span className="inline-flex items-center gap-2">{c.country} · {c.gender === "Female" ? "Female" : "Male"} · {c.age} years old </span>}
        actions={<Button asChild><Link to="/predict" search={{ customer_id: c.customer_id }}><BrainCircuit /> Score Churn Risk</Link></Button>}
      />
      

      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["Balance", fmtMoney(c.balance)],
          ["Transactions per Month", fmtInt(c.monthly_transactions)],
          ["3-Month Transaction Amount", fmtMoney(c.trans_amount_3m)],
          ["Transaction Trend", c.trans_trend_ratio.toFixed(2)],
        ].map(([l, v]) => (
          <div key={l} className="rounded-2xl border bg-card p-5 shadow-soft">
            <p className="text-xs text-muted-foreground">{l}</p>
            <p className="mt-1 font-display text-2xl font-bold tabular">{v}</p>
          </div>
        ))}
      </div>

      <div className="mt-6 grid gap-5 lg:grid-cols-3">
        <Panel title="Demographics" subtitle="Demographics">
          <Field label="Country" value={c.country} />
          <Field label="Gender" value={c.gender} />
          <Field label="Age" value={c.age} />
          <Field label="Estimated Salary" value={fmtMoney(c.estimated_salary)} />
          <Field label="Credit Score" value={c.credit_score} />
        </Panel>
        <Panel title="Product Relationship" subtitle="Products & tenure">
          <Field label="Tenure (years)" value={c.tenure} />
          <Field label="Number of Products" value={c.products_number} />
          <Field label="Credit Card" value={yesNo(c.credit_card)} />
          <Field label="Active Member" value={yesNo(c.active_member)} />
          <Field label="Balance" value={fmtMoney(c.balance)} />
        </Panel>
        <Panel title="Behavior and Service" subtitle="Behavior">
          <Field label="Days Since Last Transaction" value={c.recency_days} />
          <Field label="App Logins per Month" value={c.app_logins_monthly} />
          <Field label="Complaints" value={c.customer_service_complaints} />
          <Field label="Unresolved Complaint" value={c.has_unresolved_complaint ? <span className="text-danger">Yes</span> : "No"} />
          <Field label="Preferred Channel" value={c.preferred_channel} />
          <Field label="Primary Device" value={c.primary_device} />
        </Panel>
      </div>
    </>
  );
}