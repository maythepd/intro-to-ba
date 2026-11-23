import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  Cpu,
  Database,
  HardDrive,
  Radio,
  Server,
  Zap,
} from "lucide-react";
import {
  ErrorState,
  KeyValueGrid,
  Notice,
  PageHeader,
  Panel,
  SourceBadge,
} from "@/components/common";
import { useHealth } from "@/components/layout/AppShell";
import { api, describeError, isApiConfigured } from "@/lib/api";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/monitoring")({
  head: () =>
    pageMeta(
      "Monitoring",
      "Live API health status, system memory/CPU telemetry, relational dataset scales, and observed request latency.",
    ),
  component: Monitoring,
});

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

export function Monitoring() {
  const health = useHealth();
  const mon = useQuery({
    queryKey: ["monitoring"],
    queryFn: ({ signal }) => api.monitoring(signal),
    enabled: isApiConfigured,
    refetchInterval: 10000,
    retry: false,
  });

  const trafficData = mon.data?.recent_traffic ?? [];
  const chartData = trafficData.map((t, idx) => ({
    name: t.time || `#${idx + 1}`,
    latency: t.latency_ms,
    path: t.path,
    status: t.status,
  }));

  const relTables = mon.data?.relational_tables;

  return (
    <>
      <PageHeader
        eyebrow="Monitoring"
        title="System Telemetry & Live Traffic"
        description="Real-time backend resource utilization, full relational database scale, and live HTTP request latency recorded by API telemetry middleware."
      />

      <Notice tone="info" title="100% Observed Metrics & Hardware Telemetry">
        All stats below are live measurements polled from the FastAPI server process via{" "}
        <code className="text-xs font-semibold">psutil</code>, the active SQLite/CSV storage engines, and the rolling request telemetry buffer.
      </Notice>

      {/* KPI Cards: Memory, CPU, Uptime, Avg Latency */}
      <div className="mt-6 grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-2xl border bg-card p-5 shadow-soft">
          <div className="flex items-center justify-between">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-lilac text-primary">
              <HardDrive className="h-[18px] w-[18px]" />
            </div>
            <SourceBadge>psutil</SourceBadge>
          </div>
          <p className="mt-4 text-xs font-medium text-muted-foreground">Process Memory RSS</p>
          <p className="mt-1 font-display text-3xl font-bold tabular">
            {mon.data?.process_memory_mb !== undefined ? `${mon.data.process_memory_mb} MB` : "—"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">Resident set size of backend process</p>
        </div>

        <div className="rounded-2xl border bg-card p-5 shadow-soft">
          <div className="flex items-center justify-between">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-lilac text-primary">
              <Cpu className="h-[18px] w-[18px]" />
            </div>
            <SourceBadge>Live OS</SourceBadge>
          </div>
          <p className="mt-4 text-xs font-medium text-muted-foreground">CPU Utilization</p>
          <p className="mt-1 font-display text-3xl font-bold tabular">
            {mon.data?.cpu_percent !== undefined ? `${mon.data.cpu_percent}%` : "—"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {mon.data?.process_cpu_percent !== undefined
              ? `Process: ${mon.data.process_cpu_percent}% · System: ${mon.data.system_cpu_percent ?? mon.data.cpu_percent}%`
              : "Active CPU core load"}
          </p>
        </div>

        <div className="rounded-2xl border bg-card p-5 shadow-soft">
          <div className="flex items-center justify-between">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-lilac text-primary">
              <Server className="h-[18px] w-[18px]" />
            </div>
            <SourceBadge>Uptime</SourceBadge>
          </div>
          <p className="mt-4 text-xs font-medium text-muted-foreground">Server Uptime</p>
          <p className="mt-1 font-display text-3xl font-bold tabular">
            {mon.data?.uptime_seconds !== undefined ? formatUptime(mon.data.uptime_seconds) : "—"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">Continuous running time</p>
        </div>

        <div className="rounded-2xl border bg-card p-5 shadow-soft">
          <div className="flex items-center justify-between">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-lilac text-primary">
              <Zap className="h-[18px] w-[18px]" />
            </div>
            <SourceBadge>Rolling Avg</SourceBadge>
          </div>
          <p className="mt-4 text-xs font-medium text-muted-foreground">Average Latency</p>
          <p className="mt-1 font-display text-3xl font-bold tabular">
            {mon.data?.avg_latency_ms !== undefined ? `${mon.data.avg_latency_ms} ms` : "—"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">Measured across recent requests</p>
        </div>
      </div>

      {/* Relational Dataset Breakdown */}
      <Panel
        className="mt-6"
        title="Full Relational Dataset Scale"
        subtitle="Dynamic row count in data/full_dataset/ — zero hardcoded counts"
      >
        <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4">
          <div className="rounded-xl border bg-muted/40 p-3.5">
            <div className="flex items-center justify-between">
              <p className="text-xs text-muted-foreground font-mono">transactions.csv</p>
              <Database className="h-3.5 w-3.5 text-primary" />
            </div>
            <p className="mt-2 text-xl font-bold tabular">{relTables?.transactions?.toLocaleString() ?? "—"}</p>
            <p className="text-[11px] text-muted-foreground">Financial ledger events</p>
          </div>
          <div className="rounded-xl border bg-muted/40 p-3.5">
            <div className="flex items-center justify-between">
              <p className="text-xs text-muted-foreground font-mono">login.csv</p>
              <Activity className="h-3.5 w-3.5 text-primary" />
            </div>
            <p className="mt-2 text-xl font-bold tabular">{relTables?.login?.toLocaleString() ?? relTables?.logins?.toLocaleString() ?? "—"}</p>
            <p className="text-[11px] text-muted-foreground">Digital banking sessions</p>
          </div>
          <div className="rounded-xl border bg-muted/40 p-3.5">
            <div className="flex items-center justify-between">
              <p className="text-xs text-muted-foreground font-mono">accounts.csv</p>
              <Server className="h-3.5 w-3.5 text-primary" />
            </div>
            <p className="mt-2 text-xl font-bold tabular">{relTables?.accounts?.toLocaleString() ?? "—"}</p>
            <p className="text-[11px] text-muted-foreground">Active checking & savings accounts</p>
          </div>
          <div className="rounded-xl border bg-muted/40 p-3.5">
            <div className="flex items-center justify-between">
              <p className="text-xs text-muted-foreground font-mono">complaints.csv</p>
              <Radio className="h-3.5 w-3.5 text-primary" />
            </div>
            <p className="mt-2 text-xl font-bold tabular">{relTables?.complaints?.toLocaleString() ?? "—"}</p>
            <p className="text-[11px] text-muted-foreground">Support & grievance tickets</p>
          </div>
        </div>
      </Panel>

      {/* Observed Live Request Latency Chart */}
      <Panel
        className="mt-6"
        title="Live Request Latency (Observed Telemetry)"
        subtitle="HTTP response duration recorded by FastAPI telemetry middleware in milliseconds"
      >
        {chartData.length === 0 ? (
          <div className="flex h-60 items-center justify-center rounded-xl bg-muted/30 text-sm text-muted-foreground">
            Awaiting incoming API requests to render live telemetry...
          </div>
        ) : (
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ left: -10, right: 16, top: 12, bottom: 0 }}>
                <CartesianGrid stroke="var(--color-border)" vertical={false} />
                <XAxis dataKey="name" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(val: number) => `${val}ms`}
                />
                <Tooltip
                  formatter={(val: number) => [`${val.toFixed(2)} ms`, "Latency"]}
                  labelFormatter={(_label, payload) => {
                    const item = payload?.[0]?.payload as { path?: string; status?: number; name?: string } | undefined;
                    return item ? `${item.path} (${item.status}) at ${item.name}` : "";
                  }}
                />
                <Line
                  dataKey="latency"
                  name="Latency (ms)"
                  stroke="var(--color-chart-1)"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: "var(--color-chart-1)" }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Live Requests Table */}
        {trafficData.length > 0 && (
          <div className="mt-5 overflow-x-auto">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Recent Request Trace Buffer (Last 20 Calls)
            </p>
            <table className="w-full min-w-[550px] text-left text-xs">
              <thead className="border-b text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">Time</th>
                  <th className="px-3 py-2 font-medium">Method</th>
                  <th className="px-3 py-2 font-medium">Endpoint</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y font-mono">
                {trafficData.slice().reverse().map((t, i) => (
                  <tr key={i} className="hover:bg-muted/40">
                    <td className="px-3 py-2 text-muted-foreground">{t.time}</td>
                    <td className="px-3 py-2">
                      <span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-bold ${
                        t.method === "POST" ? "bg-primary/10 text-primary" : "bg-muted text-foreground"
                      }`}>
                        {t.method}
                      </span>
                    </td>
                    <td className="px-3 py-2 truncate max-w-[260px]">{t.path}</td>
                    <td className="px-3 py-2">
                      <span className={`font-semibold ${t.status < 400 ? "text-success" : "text-danger"}`}>
                        {t.status}
                      </span>
                    </td>
                    <td className="px-3 py-2 tabular-nums font-semibold">{t.latency_ms.toFixed(2)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {/* API Health & Readiness Raw Details */}
      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        <Panel title="API Readiness" subtitle="GET /health · refreshed every 30 seconds">
          {!isApiConfigured ? (
            <Notice tone="warning" title="API Not Connected" />
          ) : health.isError ? (
            <ErrorState message={describeError(health.error)} onRetry={() => health.refetch()} />
          ) : health.data ? (
            <KeyValueGrid data={health.data} />
          ) : (
            <div className="h-20 animate-pulse rounded-xl bg-muted" />
          )}
        </Panel>

        <Panel title="Process & Serving Overview" subtitle="GET /api/monitoring · state machine verification">
          {!isApiConfigured ? (
            <Notice tone="warning" title="API Not Connected" />
          ) : mon.isError ? (
            <ErrorState message={describeError(mon.error)} onRetry={() => mon.refetch()} />
          ) : mon.data ? (
            <KeyValueGrid
              data={{
                status: mon.data.status,
                uptime_seconds: mon.data.uptime_seconds,
                process_memory_mb: mon.data.process_memory_mb,
                cpu_percent: mon.data.cpu_percent,
                total_inferences: mon.data.total_inferences,
                avg_latency_ms: mon.data.avg_latency_ms,
                total_relational_records: relTables?.total_records,
              }}
            />
          ) : (
            <div className="h-20 animate-pulse rounded-xl bg-muted" />
          )}
        </Panel>
      </div>
    </>
  );
}
