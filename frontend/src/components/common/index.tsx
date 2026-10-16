import type { ReactNode } from "react";
import { AlertTriangle, FlaskConical, Info, RefreshCw, WifiOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
      <div className="max-w-2xl">
        {eyebrow && <p className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-primary">{eyebrow}</p>}
        <h1 className="text-2xl font-bold text-foreground md:text-[28px]">{title}</h1>
        {description && <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

export function Panel({ title, subtitle, action, children, className, illustrative }: { title?: string; subtitle?: ReactNode; action?: ReactNode; children: ReactNode; className?: string; illustrative?: boolean }) {
  return (
    <section className={cn("relative rounded-2xl border bg-card p-6 shadow-soft", className)}>
      {(title || action) && (
        <div className="mb-5 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-base font-semibold">{title}</h2>}
            {subtitle && <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>}
          </div>
          <div className="flex items-center gap-2">
            {illustrative && <IllustrativeStamp />}
            {action}
          </div>
        </div>
      )}
      {children}
    </section>
  );
}

export function DemoBadge({ className }: { className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-md border border-warning/40 bg-warning-soft px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-warning", className)}>
      <FlaskConical className="h-3 w-3" /> Demo
    </span>
  );
}

export function IllustrativeStamp() {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-dashed border-lavender bg-lilac px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-secondary-foreground">
      Minh họa · không phải số liệu đo
    </span>
  );
}

export function SourceBadge({ children = "Số liệu nguồn" }: { children?: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md bg-lilac px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-secondary-foreground">
      <Info className="h-3 w-3" /> {children}
    </span>
  );
}

export function Notice({ tone = "info", title, children }: { tone?: "info" | "warning" | "danger"; title: string; children?: ReactNode }) {
  const tones = {
    info: "border-lavender/60 bg-lilac/60 text-secondary-foreground",
    warning: "border-warning/30 bg-warning-soft text-foreground",
    danger: "border-danger/30 bg-danger-soft text-foreground",
  };
  const Icon = tone === "info" ? Info : AlertTriangle;
  return (
    <div className={cn("flex gap-3 rounded-xl border px-4 py-3 text-sm", tones[tone])}>
      <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", tone === "danger" ? "text-danger" : tone === "warning" ? "text-warning" : "text-primary")} />
      <div>
        <p className="font-semibold">{title}</p>
        {children && <div className="mt-0.5 text-muted-foreground">{children}</div>}
      </div>
    </div>
  );
}

export function ErrorState({ message, onRetry, retrying }: { message: string; onRetry?: () => void; retrying?: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-danger/30 bg-danger-soft/50 px-6 py-10 text-center">
      <WifiOff className="h-6 w-6 text-danger" />
      <p className="mt-3 text-sm font-semibold">Không tải được dữ liệu</p>
      <p className="mt-1 max-w-md text-xs text-muted-foreground">{message}</p>
      {onRetry && (
        <Button variant="outline" size="sm" className="mt-4" onClick={onRetry} disabled={retrying}>
          <RefreshCw className={cn("h-3.5 w-3.5", retrying && "animate-spin")} /> Thử lại
        </Button>
      )}
    </div>
  );
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-lavender bg-card px-6 py-16 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-lilac text-primary">{icon}</div>
      <p className="mt-4 text-base font-semibold">{title}</p>
      {children && <div className="mt-2 max-w-md text-sm text-muted-foreground">{children}</div>}
    </div>
  );
}

export function RiskBadge({ level }: { level: "low" | "medium" | "high" }) {
  const map = {
    low: ["Thấp", "bg-success-soft text-success"],
    medium: ["Trung bình", "bg-warning-soft text-warning"],
    high: ["Cao", "bg-danger-soft text-danger"],
  } as const;
  const [label, cls] = map[level];
  return <span className={cn("inline-flex rounded-full px-2.5 py-0.5 text-xs font-semibold", cls)}>{label}</span>;
}

export function KeyValueGrid({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data);
  if (!entries.length) return <p className="text-sm text-muted-foreground">API trả về dữ liệu rỗng.</p>;
  return (
    <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {entries.map(([k, v]) => (
        <div key={k} className="rounded-xl bg-muted px-4 py-3">
          <dt className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">{k}</dt>
          <dd className="mt-1 break-words font-mono text-sm tabular">
            {typeof v === "object" && v !== null ? JSON.stringify(v) : String(v)}
          </dd>
        </div>
      ))}
    </dl>
  );
}