import { createFileRoute } from "@tanstack/react-router";
import { Layers, MessageSquareWarning, MoonStar, TrendingDown } from "lucide-react";
import { Notice, PageHeader } from "@/components/common";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/retention")({
  head: () => pageMeta("Giữ chân", "Gợi ý hành động giữ chân theo tín hiệu hành vi."),
  component: Retention,
});

const RECS = [
  { icon: TrendingDown, title: "Giao dịch chậm lại", signal: "trans_trend_ratio < 1, giao dịch/tháng giảm", actions: ["Liên hệ chủ động trong 7 ngày", "Ưu đãi phí giao dịch có thời hạn", "Kiểm tra nhu cầu thay đổi"] },
  { icon: MessageSquareWarning, title: "Khiếu nại", signal: "Có khiếu nại chưa xử lý hoặc nhiều khiếu nại", actions: ["Ưu tiên xử lý dứt điểm", "Gọi lại xác nhận hài lòng", "Chuyển chuyên viên quan hệ"] },
  { icon: MoonStar, title: "Ngủ đông", signal: "recency_days cao, ít đăng nhập app", actions: ["Nhắc nhở qua kênh ưa thích", "Hướng dẫn tính năng app", "Kích hoạt lại bằng ưu đãi nhỏ"] },
  { icon: Layers, title: "Gói sản phẩm", signal: "Chỉ 1 sản phẩm hoặc quá nhiều sản phẩm", actions: ["Tư vấn sản phẩm bổ trợ phù hợp", "Rà soát gói khi ≥3 sản phẩm", "Tránh bán chéo ép buộc"] },
];

function Retention() {
  return (
    <>
      <PageHeader eyebrow="Retention" title="Gợi ý giữ chân" description="Hành động gợi ý theo nhóm tín hiệu rủi ro." />
      <Notice tone="warning" title="Gợi ý mang tính tham khảo, không phải quan hệ nhân quả">Các tín hiệu có liên quan thống kê với rời bỏ nhưng chưa chứng minh nguyên nhân. Hãy kết hợp đánh giá của nhân viên.</Notice>
      <div className="mt-6 grid gap-5 md:grid-cols-2">
        {RECS.map((r) => (
          <div key={r.title} className="rounded-2xl border bg-card p-6 shadow-soft">
            <div className="flex items-center gap-3"><div className="flex h-10 w-10 items-center justify-center rounded-xl bg-lilac text-primary"><r.icon className="h-5 w-5" /></div><h2 className="font-semibold">{r.title}</h2></div>
            <p className="mt-3 text-xs text-muted-foreground">Tín hiệu: {r.signal}</p>
            <ul className="mt-3 space-y-2">{r.actions.map((a) => <li key={a} className="flex gap-2 text-sm"><span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />{a}</li>)}</ul>
          </div>
        ))}
      </div>
    </>
  );
}