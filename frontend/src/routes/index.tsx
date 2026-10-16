import { createFileRoute, Link } from "@tanstack/react-router";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowRight, BrainCircuit, HeartHandshake, TrendingDown, UserMinus, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { IllustrativeStamp, PageHeader, Panel, SourceBadge } from "@/components/common";
import { SOURCE_FACTS } from "@/lib/demo-data";
import { fmtInt, fmtPct } from "@/lib/format";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/")({
  head: () => pageMeta("Overview", "Customer retention overview covering portfolio size, historical churn rate, and risk signals."),
  component: Dashboard,
});

// ILLUSTRATIVE shapes only — stamped "Illustrative" in the UI. Not measured data.
const SPARK = [12, 14, 13, 16, 15, 18, 17, 19, 21, 20, 22, 24].map((v, i) => ({ i, v }));
const RISK_ILLUS = [
  { band: "Low", n: 60 },
  { band: "Medium", n: 25 },
  { band: "High", n: 15 },
];
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
        <Kpi icon={TrendingDown} label="Customers Requiring Attention" value="—" note="Requires batch-scoring API" badge={<IllustrativeStamp />}>
          <div className="mt-3 h-10">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={SPARK}>
                <Area dataKey="v" type="monotone" stroke="var(--color-chart-1)" fill="var(--color-lilac)" strokeWidth={2} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Kpi>
        <Kpi icon={HeartHandshake} label="Historical Retention Rate" value={fmtPct(1 - SOURCE_FACTS.baselineChurnRate)} note="Derived from baseline churn" badge={<SourceBadge>EDA</SourceBadge>} />
      </div>

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
        <Panel className="lg:col-span-2" title="Risk Level Distribution" subtitle="Illustrative percentages, not model predictions." illustrative>
          <div className="stamp-illustrative h-64 rounded-xl">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={RISK_ILLUS} margin={{ left: -20, right: 8, top: 8 }}>
                <CartesianGrid stroke="var(--color-border)" vertical={false} />
                <XAxis dataKey="band" tickLine={false} axisLine={false} fontSize={12} stroke="var(--color-muted-foreground)" />
                <YAxis tickLine={false} axisLine={false} fontSize={12} stroke="var(--color-muted-foreground)" />
                <Tooltip cursor={{ fill: "var(--color-muted)" }} contentStyle={{ borderRadius: 12 }} formatter={() => ["Illustrative", "Share"]} />
                <Bar dataKey="n" fill="var(--color-chart-1)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
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