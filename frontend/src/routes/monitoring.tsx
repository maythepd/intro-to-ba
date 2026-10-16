import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis, CartesianGrid } from "recharts";
import { ErrorState, KeyValueGrid, Notice, PageHeader, Panel } from "@/components/common";
import { useHealth } from "@/components/layout/AppShell";
import { api, describeError, isApiConfigured } from "@/lib/api";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/monitoring")({
  head: () => pageMeta("Monitoring", "API health status and inference counts."),
  component: Monitoring,
});

const ILLUS = Array.from({ length: 12 }, (_, i) => ({ d: `N${i + 1}`, v: 50 + ((i * 7) % 11) }));

function Monitoring() {
  const health = useHealth();
  const mon = useQuery({ queryKey: ["monitoring"], queryFn: ({ signal }) => api.monitoring(signal), enabled: isApiConfigured, refetchInterval: 15000, retry: false });
  return (
    <>
      <PageHeader eyebrow="Monitoring" title="Model Monitoring" description="Live health and inference statistics from /health and /api/monitoring. Drift is not shown without measurements." />
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="API Readiness" subtitle="GET /health · refresh every 30 seconds">
          {!isApiConfigured ? <Notice tone="warning" title="API Not Connected" /> : health.isError ? <ErrorState message={describeError(health.error)} onRetry={() => health.refetch()} retrying={health.isFetching} /> : health.data ? <KeyValueGrid data={health.data} /> : <div className="h-20 animate-pulse rounded-xl bg-muted" />}
        </Panel>
        <Panel title="Inference Count" subtitle="GET /api/monitoring · refresh every 15 seconds">
          {!isApiConfigured ? <Notice tone="warning" title="API Not Connected" /> : mon.isError ? <ErrorState message={describeError(mon.error)} onRetry={() => mon.refetch()} retrying={mon.isFetching} /> : mon.data ? <KeyValueGrid data={mon.data} /> : <div className="h-20 animate-pulse rounded-xl bg-muted" />}
        </Panel>
      </div>
      <Panel className="mt-6" title="Latency and Traffic Chart" subtitle="Illustrative layout, not observed metrics. No drift measurements are available." illustrative>
        <div className="stamp-illustrative h-60 rounded-xl">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={ILLUS} margin={{ left: -20, right: 8, top: 8 }}>
              <CartesianGrid stroke="var(--color-border)" vertical={false} />
              <XAxis dataKey="d" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis fontSize={12} tickLine={false} axisLine={false} />
              <Line dataKey="v" stroke="var(--color-chart-1)" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Panel>
    </>
  );
}