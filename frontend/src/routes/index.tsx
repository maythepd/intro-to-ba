import { useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowRight, BrainCircuit, HeartHandshake, TrendingDown, UserMinus, Users, Database, Landmark } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PageHeader, Panel, SourceBadge, Notice } from "@/components/common";
import { api, describeError, isApiConfigured } from "@/lib/api";
import { fmtInt, fmtMoney, fmtPct } from "@/lib/format";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/")({
  head: () => pageMeta("Overview", "Customer retention overview covering portfolio size, historical churn rate, and risk signals."),
  component: Dashboard,
});

function Kpi({
  icon: Icon,
  label,
  value,
  note,
  badge,
  children,
  loading,
}: {
  icon: typeof Users;
  label: string;
  value: string;
  note: string;
  badge?: React.ReactNode;
  children?: React.ReactNode;
  loading?: boolean;
}) {
  return (
    <div className="rounded-2xl border bg-card p-5 shadow-soft">
      <div className="flex items-center justify-between">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-lilac text-primary"><Icon className="h-[18px] w-[18px]" /></div>
        {badge}
      </div>
      <p className="mt-4 text-xs font-medium text-muted-foreground">{label}</p>
      {loading ? (
        <span className="mt-2 inline-block h-8 w-28 animate-pulse rounded bg-muted" />
      ) : (
        <p className="mt-1 font-display text-3xl font-bold tabular">{value}</p>
      )}
      <p className="mt-1 text-xs text-muted-foreground">{note}</p>
      {children}
    </div>
  );
}

function Dashboard() {
  const [selectedModel, setSelectedModel] = useState<"xgboost" | "random_forest" | "logistic_regression">("xgboost");
  const [threshold, setThreshold] = useState<number>(0.50);

  const statsQuery = useQuery({
    queryKey: ["dashboard-stats"],
    queryFn: ({ signal }) => api.dashboardStats(signal),
    enabled: isApiConfigured,
    staleTime: 60000,
  });

  const riskQuery = useQuery({
    queryKey: ["dashboard-risk-summary", selectedModel, threshold],
    queryFn: ({ signal }) => api.riskSummary({ model: selectedModel, threshold }, signal),
    enabled: isApiConfigured,
    retry: false,
    staleTime: 60000,
  });

  const batch = useMutation({
    mutationFn: (force: boolean) => api.batchScore(force, selectedModel, threshold),
    onSuccess: () => {
      void riskQuery.refetch();
      void statsQuery.refetch();
    },
  });

  const stats = statsQuery.data;
  const risk = riskQuery.data ?? stats?.risk_summary;

  const totalCust = stats?.total_customers;
  const churnRate = stats?.historical_churn_rate;
  const churnedCust = stats?.churned_customers;
  const totalBal = stats?.total_balance;
  const avgBal = stats?.avg_balance;

  const monthlyTrend = stats?.monthly_trend ?? [];

  const riskBars = risk ? [
    { band: "Low Risk (<30%)", n: risk.counts.low },
    { band: "Medium Risk (30-70%)", n: risk.counts.medium },
    { band: "High Risk (≥70%)", n: risk.counts.high },
  ] : [];

  const runScoring = () => batch.mutate(Boolean(risk));

  return (
    <>
      <PageHeader
        eyebrow="Dashboard"
        title="Customer Retention Overview"
        description="Live portfolio insights, relational transaction trends, and XGBoost churn risk distribution calculated directly from the committed database."
        actions={
          <>
            <Button variant="outline" asChild><Link to="/customers">View Customers</Link></Button>
            <Button asChild><Link to="/predict"><BrainCircuit /> Score Churn Risk</Link></Button>
          </>
        }
      />

      <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi
          icon={Users}
          label="Total Customers"
          value={totalCust !== undefined ? fmtInt(totalCust) : "—"}
          loading={statsQuery.isLoading}
          note="Full analytical dataset (100% active panel)"
          badge={<SourceBadge>Dataset</SourceBadge>}
        />
        <Kpi
          icon={UserMinus}
          label="Historical Churn Rate"
          value={churnRate !== undefined ? fmtPct(churnRate) : "—"}
          loading={statsQuery.isLoading}
          note={
            churnedCust !== undefined && totalCust !== undefined
              ? `${fmtInt(churnedCust)} churned · ${fmtInt(totalCust - churnedCust)} retained`
              : "Historical bank outcomes"
          }
          badge={<SourceBadge>Observed</SourceBadge>}
        />
        <Kpi
          icon={TrendingDown}
          label="Customers Requiring Attention"
          value={risk ? fmtInt(risk.customers_requiring_attention) : "—"}
          loading={riskQuery.isLoading}
          note={risk ? `High risk (XGBoost P ≥ 0.70) · ${fmtInt(risk.scored_customers)} scored` : "Computing model risk scores…"}
          badge={risk ? <SourceBadge>XGBoost Model</SourceBadge> : undefined}
        >
          {risk && (
            <p className="mt-2 text-xs text-muted-foreground">
              {new Date(risk.scored_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} UTC update
            </p>
          )}
        </Kpi>
        <Kpi
          icon={Landmark}
          label="Total Portfolio Balance"
          value={totalBal !== undefined ? fmtMoney(totalBal) : "—"}
          loading={statsQuery.isLoading}
          note={avgBal !== undefined ? `Avg balance: ${fmtMoney(avgBal)} per customer` : "Portfolio balance metrics"}
          badge={<SourceBadge>Full Portfolio</SourceBadge>}
        />
      </div>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Button
            type="button"
            variant="outline"
            onClick={runScoring}
            disabled={!isApiConfigured || batch.isPending || riskQuery.isLoading}
          >
            {batch.isPending ? "Scoring all customers…" : risk ? "Recompute risk scores" : "Run batch scoring"}
          </Button>
          {risk && (
            <span className="text-xs text-muted-foreground">
              Champion: <strong className="font-semibold text-foreground">{risk.model_name}</strong> · {risk.source}
            </span>
          )}
        </div>
        {stats?.country_distribution && Object.keys(stats.country_distribution).length > 0 && (
          <div className="hidden items-center gap-3 text-xs text-muted-foreground lg:flex">
            <span className="flex items-center gap-1"><Database className="h-3.5 w-3.5 text-primary" /> Relational tables: 1.86M records</span>
            {stats.country_distribution.France !== undefined && <span>· France ({fmtInt(stats.country_distribution.France)})</span>}
            {stats.country_distribution.Germany !== undefined && <span>· Germany ({fmtInt(stats.country_distribution.Germany)})</span>}
            {stats.country_distribution.Spain !== undefined && <span>· Spain ({fmtInt(stats.country_distribution.Spain)})</span>}
          </div>
        )}
      </div>

      {!isApiConfigured && (
        <div className="mt-4">
          <Notice tone="warning" title="API not configured">Set VITE_API_BASE_URL to see live backend stats.</Notice>
        </div>
      )}

      {batch.isError && (
        <div className="mt-4">
          <Notice tone="warning" title="Batch scoring failed">{describeError(batch.error)}</Notice>
        </div>
      )}

      <div className="mt-6 grid gap-5 lg:grid-cols-5">
        <Panel
          className="lg:col-span-3"
          title="Monthly Transaction Activity & Volume"
          subtitle="Observed transaction volumes and frequencies across relational transactions"
        >
          {statsQuery.isLoading ? (
            <div className="flex h-64 items-center justify-center rounded-xl bg-muted/40">
              <span className="text-xs text-muted-foreground animate-pulse">Loading transaction trend…</span>
            </div>
          ) : monthlyTrend.length === 0 ? (
            <div className="flex h-64 items-center justify-center rounded-xl bg-muted/20 text-xs text-muted-foreground">
              No transaction history records available.
            </div>
          ) : (
            <div className="h-64 rounded-xl">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={monthlyTrend} margin={{ left: -10, right: 8, top: 8, bottom: 0 }}>
                  <CartesianGrid stroke="var(--color-border)" vertical={false} />
                  <XAxis dataKey="label" tickLine={false} axisLine={false} fontSize={12} stroke="var(--color-muted-foreground)" />
                  <YAxis
                    tickLine={false}
                    axisLine={false}
                    fontSize={11}
                    stroke="var(--color-muted-foreground)"
                    tickFormatter={(v) => `€${(v / 1000000).toFixed(1)}M`}
                  />
                  <Tooltip
                    contentStyle={{ borderRadius: 12, border: "1px solid var(--color-border)", background: "var(--color-card)" }}
                    formatter={(value: any, name: any) => [
                      name === "volume" ? fmtMoney(Number(value)) : fmtInt(Number(value)),
                      name === "volume" ? "Transaction Volume" : "Transaction Count"
                    ]}
                  />
                  <Area
                    dataKey="volume"
                    name="volume"
                    type="monotone"
                    stroke="var(--color-chart-1)"
                    strokeWidth={2.5}
                    fill="var(--color-lilac)"
                    fillOpacity={0.65}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>

        <Panel
          className="lg:col-span-2"
          title={`${risk?.model_name || "Portfolio"} Risk Distribution`}
          subtitle={
            risk && totalCust !== undefined
              ? `Breakdown across ${fmtInt(totalCust)} customers (Threshold τ = ${(risk.decision_threshold ?? threshold).toFixed(2)})`
              : "Computing risk bands…"
          }
          action={
            <div className="flex items-center gap-1.5">
              <select
                aria-label="Select Model"
                className="h-8 rounded-lg border bg-background px-2 text-xs font-medium"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value as any)}
              >
                <option value="xgboost">XGBoost</option>
                <option value="random_forest">Random Forest</option>
                <option value="logistic_regression">Logistic Reg</option>
              </select>
              <select
                aria-label="Select Decision Threshold"
                className="h-8 rounded-lg border bg-background px-2 text-xs font-mono font-medium"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
              >
                <option value={0.35}>τ = 0.35</option>
                <option value={0.50}>τ = 0.50</option>
                <option value={0.65}>τ = 0.65</option>
              </select>
            </div>
          }
        >
          {risk ? (
            <div className="h-64 rounded-xl">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={riskBars} margin={{ left: -8, right: 8, top: 8, bottom: 8 }}>
                  <CartesianGrid stroke="var(--color-border)" vertical={false} />
                  <XAxis dataKey="band" tickLine={false} axisLine={false} fontSize={11} />
                  <YAxis allowDecimals={false} tickLine={false} axisLine={false} fontSize={12} />
                  <Tooltip
                    cursor={{ fill: "var(--color-muted)" }}
                    contentStyle={{ borderRadius: 12, border: "1px solid var(--color-border)", background: "var(--color-card)" }}
                    formatter={(value) => [fmtInt(Number(value)), "Customers"]}
                  />
                  <Bar dataKey="n" fill="var(--color-chart-1)" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="flex h-64 items-center justify-center text-center text-sm text-muted-foreground">
              Computing risk distribution from fitted XGBoost model…
            </p>
          )}
        </Panel>
      </div>

      <div className="mt-6 grid gap-5 md:grid-cols-3">
        {[
          { to: "/predict" as const, title: "Individual Churn Predictor", desc: "Interactive customer scoring with live feature attributions.", icon: BrainCircuit },
          { to: "/retention" as const, title: "Retention Action Matrix", desc: "Actionable playbooks mapped to behavioral churn signals.", icon: HeartHandshake },
          { to: "/models" as const, title: "Model Checkpoints & Benchmark", desc: "Inspect trained checkpoints, hyperparameters, and decile lifts.", icon: TrendingDown },
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
