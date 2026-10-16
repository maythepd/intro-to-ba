import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { ChevronRight, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ErrorState, Notice, PageHeader, Panel } from "@/components/common";
import { api, describeError, isApiConfigured, normalizeCustomerList, type CustomerFeatures } from "@/lib/api";
import { fmtMoney } from "@/lib/format";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/customers/")({
  validateSearch: (s: Record<string, unknown>): { search?: string } => ({
    search: typeof s.search === "string" && s.search ? s.search : undefined,
  }),
  head: () => pageMeta("Khách hàng", "Tra cứu hồ sơ khách hàng và hành vi giao dịch."),
  component: CustomersPage,
});

function CustomersPage() {
  const { search = "" } = Route.useSearch();
  const navigate = useNavigate({ from: "/customers/" });
  const [text, setText] = useState(search);
  useEffect(() => setText(search), [search]);
  useEffect(() => {
    const t = setTimeout(() => {
      if (text !== search) navigate({ search: { search: text.trim() || undefined }, replace: true });
    }, 300);
    return () => clearTimeout(t);
  }, [text, search, navigate]);

  const q = useQuery({
    queryKey: ["customers", search],
    queryFn: ({ signal }) => api.listCustomers({ search, limit: 50 }, signal).then(normalizeCustomerList),
    enabled: isApiConfigured,
    retry: false,
  });

  const rows: CustomerFeatures[] = q.data ?? [];

  return (
    <>
      <PageHeader eyebrow="Customers" title="Danh sách khách hàng" description="Tìm kiếm theo mã khách hàng, quốc gia, kênh hoặc thiết bị." />
      {!isApiConfigured && (
        <div className="mb-5">
          <Notice tone="warning" title="Chưa kết nối dữ liệu chính thức">
            Kết nối API và nạp dữ liệu Kaggle chính thức để hiển thị hồ sơ.
          </Notice>
        </div>
      )}
      <Panel>
        <div className="relative mb-5 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input value={text} onChange={(e) => setText(e.target.value)} placeholder="Tìm khách hàng…" className="h-10 rounded-xl pl-9" />
        </div>

        {q.isError ? (
          <ErrorState message={describeError(q.error)} onRetry={() => q.refetch()} retrying={q.isFetching} />
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Mã KH</TableHead>
                  <TableHead>Quốc gia</TableHead>
                  <TableHead className="text-right">Tuổi</TableHead>
                  <TableHead className="text-right">Số dư</TableHead>
                  <TableHead className="text-right">SP</TableHead>
                  <TableHead className="text-right">Giao dịch/tháng</TableHead>
                  <TableHead className="text-right">Ngày từ GD cuối</TableHead>
                  <TableHead>Khiếu nại</TableHead>
                  <TableHead>Kênh</TableHead>
                  <TableHead />
                </TableRow>
              </TableHeader>
              <TableBody>
                {q.isLoading &&
                  Array.from({ length: 5 }).map((_, i) => (
                    <TableRow key={i}><TableCell colSpan={10}><div className="h-5 animate-pulse rounded bg-muted" /></TableCell></TableRow>
                  ))}
                {!q.isLoading && rows.length === 0 && (
                  <TableRow><TableCell colSpan={10} className="py-10 text-center text-sm text-muted-foreground">Không có khách hàng phù hợp.</TableCell></TableRow>
                )}
                {rows.map((c) => (
                  <TableRow key={c.customer_id} className="cursor-pointer" onClick={() => navigate({ to: "/customers/$id", params: { id: c.customer_id } })}>
                    <TableCell className="font-mono text-xs font-semibold">
                      <span className="flex items-center gap-2">{c.customer_id}</span>
                    </TableCell>
                    <TableCell>{c.country}</TableCell>
                    <TableCell className="text-right tabular">{c.age}</TableCell>
                    <TableCell className="text-right tabular">{fmtMoney(c.balance)}</TableCell>
                    <TableCell className="text-right tabular">{c.products_number}</TableCell>
                    <TableCell className="text-right tabular">{c.monthly_transactions}</TableCell>
                    <TableCell className="text-right tabular">{c.recency_days}</TableCell>
                    <TableCell>
                      {c.has_unresolved_complaint ? <span className="text-xs font-semibold text-danger">Chưa xử lý</span> : <span className="text-xs text-muted-foreground">{c.customer_service_complaints}</span>}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">{c.preferred_channel}</TableCell>
                    <TableCell>
                      <Link to="/customers/$id" params={{ id: c.customer_id }} className="text-primary" aria-label={`Xem ${c.customer_id}`} onClick={(e) => e.stopPropagation()}>
                        <ChevronRight className="h-4 w-4" />
                      </Link>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </Panel>
    </>
  );
}