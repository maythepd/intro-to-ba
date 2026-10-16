import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ErrorState, KeyValueGrid, Notice, PageHeader, Panel, SourceBadge } from "@/components/common";
import { api, describeError, isApiConfigured } from "@/lib/api";
import { SOURCE_FACTS } from "@/lib/demo-data";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/models")({
  head: () => pageMeta("Mô hình", "Chỉ số kiểm định XGBoost theo nghiên cứu nguồn."),
  component: Models,
});

function Models() {
  const x = SOURCE_FACTS.xgboost;
  const q = useQuery({ queryKey: ["metrics"], queryFn: ({ signal }) => api.metrics(signal), enabled: isApiConfigured, retry: false });
  return (
    <>
      <PageHeader eyebrow="Models" title="Mô hình XGBoost" description="Chỉ số kiểm định do nghiên cứu nguồn báo cáo." />
      <Notice tone="info" title="Đây là số liệu của nghiên cứu nguồn, không phải hiệu năng đo trong triển khai.">Chỉ số gần tuyệt đối có thể phản ánh rò rỉ dữ liệu hoặc tập kiểm định thuận lợi; cần đánh giá lại trên dữ liệu vận hành.</Notice>
      <div className="mt-6 grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        {[["ROC-AUC", x.rocAuc], ["PR-AUC", x.prAuc], ["F1", x.f1], ["Recall", x.recall]].map(([l, v]) => (
          <div key={l as string} className="rounded-2xl border bg-card p-5 shadow-soft">
            <div className="flex items-center justify-between"><p className="text-xs text-muted-foreground">{l}</p><SourceBadge /></div>
            <p className="mt-2 font-display text-3xl font-bold tabular">{(v as number).toFixed(4)}</p>
          </div>
        ))}
      </div>
      <Panel className="mt-6" title="Chỉ số từ API" subtitle="GET /api/metrics">
        {!isApiConfigured ? <p className="text-sm text-muted-foreground">API chưa kết nối.</p> : q.isError ? <ErrorState message={describeError(q.error)} onRetry={() => q.refetch()} retrying={q.isFetching} /> : q.data ? <KeyValueGrid data={q.data} /> : <div className="h-20 animate-pulse rounded-xl bg-muted" />}
      </Panel>
    </>
  );
}