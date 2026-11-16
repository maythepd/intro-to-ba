import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowRight, BrainCircuit, HeartHandshake, TrendingDown, UserMinus, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { IllustrativeStamp, PageHeader, Panel, SourceBadge, Notice } from "@/components/common";
import { api, ApiError, describeError, isApiConfigured } from "@/lib/api";
import { SOURCE_FACTS } from "@/lib/demo-data";
import { fmtInt, fmtPct } from "@/lib/format";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/")({
  head: () => pageMeta("Overview", "Customer retention overview covering portfolio size, historical churn rate, and risk signals."),
  component: Dashboard,
});

// Trend alone remains illustrative; risk buckets below require scored model output.
const TREND_ILLUS = ["M1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"].map((m, i) => ({ m, a: 40 + Math.round(10 * Math.sin(i / 1.5)) + i * 2 }));

function Kpi({ icon: Icon, label, value, note, badge, children }: { icon: typeof Users; label: string; value: string; note: string; badge?: React.ReactNode; children?: React.ReactNode }) {
  return (
    <div className="rounded-2xl border bg-card p-5 shadow-soft">
      <div className="flex items-center justify-between">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-lilac text-primary"><Icon className="h-[18px] w-[18px]" /></div>
        {badge}
      </div>
      <p className="mt-4 text-xs font-medium text-muted-foreground">{label}</p>
      <p className="mt-1 font-display text-3xl font-bold tabular">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{note}</p>
      {children}
    </div>
  );
}

function Dashboard() {
  const riskQuery = useQuery({
    queryKey: ["dashboard-risk-summary"],
    queryFn: ({ signal }) => api.riskSummary(signal),
    enabled: isApiConfigured,
    retry: false,
    staleTime: 60000,
  });
  const batch = useMutation({
    mutationFn: (force: boolean) => api.batchScore(force),
    onSuccess: () => { void riskQuery.refetch(); },
  });
  const risk = riskQuery.data;
  const riskBars = risk ? [
    { band: "Low", n: risk.counts.low },
    { band: "Medium", n: risk.counts.medium },
    { band: "High", n: risk.counts.high },
  ] : [];
  const needsScoring = isApiConfigured && riskQuery.error instanceof ApiError && riskQuery.error.status === 503;
  const runScoring = () => batch.mutate(Boolean(risk));
  const churned = Math.round(SOURCE_FACTS.totalCustomers * SOURCE_FACTS.baselineChurnRate);
  return (
    <>
      <PageHeader
        eyebrow="Dashboard"
        title="Customer Retention Overview"
        description="Portfolio insights and churn risk signals for customer retention teams. Baseline statistics are drawn from the EDA report."
        actions={
          <>
            <Button variant="outline" asChild><Link to="/customers">View Customers</Link></Button>
            <Button asChild><Link to="/predict"><BrainCircuit /> Score Churn Risk</Link></Button>
          </>
        }
      />

      <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi icon={Users} label="Total Customers" value={fmtInt(SOURCE_FACTS.totalCustomers)} note="Size of the study dataset" badge={<SourceBadge>EDA</SourceBadge>} />
        <Kpi icon={UserMinus} label="Historical Churn Rate" value={fmtPct(SOURCE_FACTS.baselineChurnRate)} note={`Approximately ${fmtInt(churned)} customers churned`} badge={<SourceBadge>EDA</SourceBadge>} />
        <Kpi icon={TrendingDown} label="Customers Requiring Attention"
          value={risk ? fmtInt(risk.customers_requiring_attention) : "—"}
          note={risk ? "High risk (churn probability ≥ 0.70)" : "Awaiting a completed XGBoost batch-scoring run"}
          badge={risk ? <SourceBadge>Scored data</SourceBadge> : undefined}>
          {risk && <p className="mt-2 text-xs text-muted-foreground">
            {fmtInt(risk.scored_customers)} profiles scored · {new Date(risk.scored_at).toLocaleString()}
          </p>}
        </Kpi>
        <Kpi icon={HeartHandshake} label="Historical Retention Rate" value={fmtPct(1 - SOURCE_FACTS.baselineChurnRate)} note="Derived from baseline churn" badge={<SourceBadge>EDA</SourceBadge>} />
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <Button type="button" variant="outline" onClick={runScoring}
          disabled={!isApiConfigured || batch.isPending || riskQuery.isLoading}>
          {batch.isPending ? "Scoring all customers…" : risk ? "Recompute risk scores" : "Run batch scoring"}
        </Button>
        {risk && <p className="text-xs text-muted-foreground">{risk.data_kind}</p>}
      </div>
      {!isApiConfigured && <div className="mt-4">
        <Notice tone="warning" title="API not configured">Set VITE_API_BASE_URL to see model-derived risk counts.</Notice>
      </div>}
      {needsScoring && !risk && <div className="mt-4">
        <Notice tone="info" title="Risk counts not available">
          Run batch scoring after the fitted XGBoost model and official feature table are loaded.
          No illustrative risk counts are substituted.
        </Notice>
      </div>}
      {riskQuery.isError && !needsScoring && <div className="mt-4">
        <Notice tone="warning" title="Risk summary unavailable">{describeError(riskQuery.error)}</Notice>
      </div>}
      {batch.isError && <div className="mt-4">
        <Notice tone="warning" title="Batch scoring failed">{describeError(batch.error)}</Notice>
      </div>}

      <div className="mt-6 grid gap-5 lg:grid-cols-5">
        <Panel className="lg:col-span-3" title="Churn Risk Trend" subtitle="Illustrative chart only; monitoring measurements will replace it when available." illustrative>
          <div className="stamp-illustrative h-64 rounded-xl">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={TREND_ILLUS} margin={{ left: -20, right: 8, top: 8 }}>
                <CartesianGrid stroke="var(--color-border)" vertical={false} />
                <XAxis dataKey="m" tickLine={false} axisLine={false} fontSize={12} stroke="var(--color-muted-foreground)" />
                <YAxis tickLine={false} axisLine={false} fontSize={12} stroke="var(--color-muted-foreground)" />
                <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid var(--color-border)" }} formatter={() => ["Illustrative", "Value"]} />
                <Area dataKey="a" type="monotone" stroke="var(--color-chart-1)" strokeWidth={2.5} fill="var(--color-lilac)" fillOpacity={0.7} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel className="lg:col-span-2" title="Risk Level Distribution"
          subtitle={risk ? "XGBoost predictions for the full scored customer snapshot" : "Waiting for batch-scored predictions"}>
          {risk ? (
            <div className="h-64 rounded-xl">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={riskBars} margin={{ left: -8, right: 8, top: 8, bottom: 8 }}>
                  <CartesianGrid stroke="var(--color-border)" vertical={false} />
                  <XAxis dataKey="band" tickLine={false} axisLine={false} fontSize={12} />
                  <YAxis allowDecimals={false} tickLine={false} axisLine={false} fontSize={12} />
                  <Tooltip cursor={{ fill: "var(--color-muted)" }} contentStyle={{ borderRadius: 12 }}
                    formatter={(value) => [Number(value).toLocaleString(), "Customers"]} />
                  <Bar dataKey="n" fill="var(--color-chart-1)" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : <p className="flex h-64 items-center justify-center text-center text-sm text-muted-foreground">
            No measured distribution yet. Run batch scoring to populate Low, Medium, and High risk counts.
          </p>}
        </Panel>
      </div>

      <div className="mt-6 grid gap-5 md:grid-cols-3">
        {[
          { to: "/predict" as const, title: "Score a Customer", desc: "Enter a profile using the XGBoost model schema.", icon: BrainCircuit },
          { to: "/retention" as const, title: "Retention Recommendations", desc: "Actions based on declining activity, complaints, and inactivity.", icon: HeartHandshake },
          { to: "/models" as const, title: "Model Metrics", desc: "Model evaluation metrics reported in the source study.", icon: TrendingDown },
        ].map((c) => (
          <Link key={c.to} to={c.to} className="group rounded-2xl border bg-card p-5 shadow-soft transition hover:border-lavender hover:shadow-card">
            <c.icon className="h-5 w-5 text-primary" />
            <p className="mt-3 font-semibold">{c.title}</p>
            <p className="mt-1 text-sm text-muted-foreground">{c.desc}</p>
            <span className="mt-3 inline-flex items-center gap-1 text-xs font-semibold text-primary">Open <ArrowRight className="h-3.5 w-3.5 transition group-hover:translate-x-0.5" /></span>
          </Link>
        ))}
      </div>
    </>
  );
}