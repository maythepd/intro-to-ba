import { createFileRoute, Link } from "@tanstack/react-router";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowRight, BrainCircuit, HeartHandshake, TrendingDown, UserMinus, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { IllustrativeStamp, PageHeader, Panel, SourceBadge } from "@/components/common";
import { SOURCE_FACTS } from "@/lib/demo-data";
import { fmtInt, fmtPct } from "@/lib/format";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/")({
  head: () => pageMeta("Tổng quan", "Tổng quan giữ chân khách hàng: quy mô danh mục, tỷ lệ rời bỏ lịch sử và tín hiệu rủi ro."),
  component: Dashboard,
});

// ILLUSTRATIVE shapes only — stamped "Minh họa" in the UI. Not measured data.
const SPARK = [12, 14, 13, 16, 15, 18, 17, 19, 21, 20, 22, 24].map((v, i) => ({ i, v }));
const RISK_ILLUS = [
  { band: "Thấp", n: 60 },
  { band: "Trung bình", n: 25 },
  { band: "Cao", n: 15 },
];
const TREND_ILLUS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"].map((m, i) => ({ m, a: 40 + Math.round(10 * Math.sin(i / 1.5)) + i * 2 }));

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
        title="Tổng quan giữ chân khách hàng"
        description="Bức tranh danh mục và tín hiệu rời bỏ dành cho đội ngũ chăm sóc khách hàng. Số liệu nền lấy từ báo cáo EDA hiện có."
        actions={
          <>
            <Button variant="outline" asChild><Link to="/customers">Xem khách hàng</Link></Button>
            <Button asChild><Link to="/predict"><BrainCircuit /> Chấm điểm rủi ro</Link></Button>
          </>
        }
      />

      <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi icon={Users} label="Tổng khách hàng" value={fmtInt(SOURCE_FACTS.totalCustomers)} note="Quy mô bộ dữ liệu phân tích" badge={<SourceBadge>EDA</SourceBadge>} />
        <Kpi icon={UserMinus} label="Tỷ lệ rời bỏ lịch sử (baseline)" value={fmtPct(SOURCE_FACTS.baselineChurnRate)} note={`≈ ${fmtInt(churned)} khách hàng đã rời bỏ`} badge={<SourceBadge>EDA</SourceBadge>} />
        <Kpi icon={TrendingDown} label="Khách hàng cần chú ý" value="—" note="Cần API chấm điểm hàng loạt" badge={<IllustrativeStamp />}>
          <div className="mt-3 h-10">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={SPARK}>
                <Area dataKey="v" type="monotone" stroke="var(--color-chart-1)" fill="var(--color-lilac)" strokeWidth={2} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Kpi>
        <Kpi icon={HeartHandshake} label="Tỷ lệ giữ chân lịch sử" value={fmtPct(1 - SOURCE_FACTS.baselineChurnRate)} note="Suy ra từ baseline rời bỏ" badge={<SourceBadge>EDA</SourceBadge>} />
      </div>

      <div className="mt-6 grid gap-5 lg:grid-cols-5">
        <Panel className="lg:col-span-3" title="Xu hướng rủi ro rời bỏ" subtitle="Hình dạng mẫu để xem bố cục — sẽ thay bằng dữ liệu khi API giám sát cung cấp." illustrative>
          <div className="stamp-illustrative h-64 rounded-xl">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={TREND_ILLUS} margin={{ left: -20, right: 8, top: 8 }}>
                <CartesianGrid stroke="var(--color-border)" vertical={false} />
                <XAxis dataKey="m" tickLine={false} axisLine={false} fontSize={12} stroke="var(--color-muted-foreground)" />
                <YAxis tickLine={false} axisLine={false} fontSize={12} stroke="var(--color-muted-foreground)" />
                <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid var(--color-border)" }} formatter={() => ["Minh họa", "Giá trị"]} />
                <Area dataKey="a" type="monotone" stroke="var(--color-chart-1)" strokeWidth={2.5} fill="var(--color-lilac)" fillOpacity={0.7} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel className="lg:col-span-2" title="Phân bổ mức rủi ro" subtitle="Tỷ trọng mẫu (%), không phải kết quả mô hình." illustrative>
          <div className="stamp-illustrative h-64 rounded-xl">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={RISK_ILLUS} margin={{ left: -20, right: 8, top: 8 }}>
                <CartesianGrid stroke="var(--color-border)" vertical={false} />
                <XAxis dataKey="band" tickLine={false} axisLine={false} fontSize={12} stroke="var(--color-muted-foreground)" />
                <YAxis tickLine={false} axisLine={false} fontSize={12} stroke="var(--color-muted-foreground)" />
                <Tooltip cursor={{ fill: "var(--color-muted)" }} contentStyle={{ borderRadius: 12 }} formatter={() => ["Minh họa", "Tỷ trọng"]} />
                <Bar dataKey="n" fill="var(--color-chart-1)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <div className="mt-6 grid gap-5 md:grid-cols-3">
        {[
          { to: "/predict" as const, title: "Chấm điểm một khách hàng", desc: "Nhập hồ sơ theo đúng lược đồ mô hình XGBoost.", icon: BrainCircuit },
          { to: "/retention" as const, title: "Gợi ý giữ chân", desc: "Hành động theo tín hiệu giao dịch, khiếu nại, ngủ đông.", icon: HeartHandshake },
          { to: "/models" as const, title: "Chỉ số mô hình", desc: "Kết quả kiểm định báo cáo trong nghiên cứu nguồn.", icon: TrendingDown },
        ].map((c) => (
          <Link key={c.to} to={c.to} className="group rounded-2xl border bg-card p-5 shadow-soft transition hover:border-lavender hover:shadow-card">
            <c.icon className="h-5 w-5 text-primary" />
            <p className="mt-3 font-semibold">{c.title}</p>
            <p className="mt-1 text-sm text-muted-foreground">{c.desc}</p>
            <span className="mt-3 inline-flex items-center gap-1 text-xs font-semibold text-primary">Mở <ArrowRight className="h-3.5 w-3.5 transition group-hover:translate-x-0.5" /></span>
          </Link>
        ))}
      </div>
    </>
  );
}