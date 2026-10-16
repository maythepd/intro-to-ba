import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis, CartesianGrid } from "recharts";
import { ErrorState, KeyValueGrid, Notice, PageHeader, Panel } from "@/components/common";
import { useHealth } from "@/components/layout/AppShell";
import { api, describeError, isApiConfigured } from "@/lib/api";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/monitoring")({
  head: () => pageMeta("Giám sát", "Trạng thái sẵn sàng của API và bộ đếm suy luận."),
  component: Monitoring,
});

const ILLUS = Array.from({ length: 12 }, (_, i) => ({ d: `N${i + 1}`, v: 50 + ((i * 7) % 11) }));

function Monitoring() {
  const health = useHealth();
  const mon = useQuery({ queryKey: ["monitoring"], queryFn: ({ signal }) => api.monitoring(signal), enabled: isApiConfigured, refetchInterval: 15000, retry: false });
  return (
    <>
      <PageHeader eyebrow="Monitoring" title="Giám sát mô hình" description="Trạng thái trực tiếp từ /health và /api/monitoring. Không hiển thị drift khi chưa có số đo thật." />
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Sẵn sàng API" subtitle="GET /health · tự làm mới 30 giây">
          {!isApiConfigured ? <Notice tone="warning" title="API chưa kết nối" /> : health.isError ? <ErrorState message={describeError(health.error)} onRetry={() => health.refetch()} retrying={health.isFetching} /> : health.data ? <KeyValueGrid data={health.data} /> : <div className="h-20 animate-pulse rounded-xl bg-muted" />}
        </Panel>
        <Panel title="Bộ đếm suy luận" subtitle="GET /api/monitoring · tự làm mới 15 giây">
          {!isApiConfigured ? <Notice tone="warning" title="API chưa kết nối" /> : mon.isError ? <ErrorState message={describeError(mon.error)} onRetry={() => mon.refetch()} retrying={mon.isFetching} /> : mon.data ? <KeyValueGrid data={mon.data} /> : <div className="h-20 animate-pulse rounded-xl bg-muted" />}
        </Panel>
      </div>
      <Panel className="mt-6" title="Biểu đồ độ trễ / lưu lượng" subtitle="Bố cục mẫu, không phải số đo thực tế. Không có dữ liệu drift được đo." illustrative>
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