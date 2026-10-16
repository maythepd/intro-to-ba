import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { PieChart } from "lucide-react";
import { EmptyState, ErrorState, KeyValueGrid, PageHeader, Panel } from "@/components/common";
import { api, ApiError, describeError, isApiConfigured } from "@/lib/api";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/segments")({
  head: () => pageMeta("Phân khúc", "Phân khúc khách hàng do mô hình phân cụm cung cấp."),
  component: Segments,
});

function Segments() {
  const q = useQuery({ queryKey: ["segments"], queryFn: ({ signal }) => api.segments(signal), enabled: isApiConfigured, retry: false });
  const notReady = !isApiConfigured || (q.error instanceof ApiError && q.error.status === 503);
  return (
    <>
      <PageHeader eyebrow="Segments" title="Phân khúc khách hàng" description="Chỉ hiển thị cụm do API cung cấp — không tự tạo điểm phân khúc." />
      {notReady ? (
        <EmptyState icon={<PieChart />} title="Chưa kết nối mô hình phân khúc">Mô hình phân cụm chưa sẵn sàng. Trang sẽ tự hiển thị khi API trả về dữ liệu.</EmptyState>
      ) : q.isError ? (
        <ErrorState message={describeError(q.error)} onRetry={() => q.refetch()} retrying={q.isFetching} />
      ) : q.isLoading ? (
        <div className="h-48 animate-pulse rounded-2xl bg-muted" />
      ) : (
        <div className="grid gap-5 md:grid-cols-2">
          {(Array.isArray(q.data) ? q.data : [q.data ?? {}]).map((s, i) => (
            <Panel key={i} title={String((s as Record<string, unknown>).name ?? `Cụm ${i + 1}`)}><KeyValueGrid data={s as Record<string, unknown>} /></Panel>
          ))}
        </div>
      )}
    </>
  );
}