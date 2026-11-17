import { useState, type FormEvent, type ReactNode } from "react";
import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  Bell, BrainCircuit, ChevronsLeft, ChevronsRight, Gauge, HeartHandshake, LayoutDashboard,
  Menu, PieChart, Search, ShieldCheck, Users, WifiOff,
} from "lucide-react";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { DropdownMenu, DropdownMenuContent, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { api, isApiConfigured, API_BASE_URL } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Overview", sub: "Dashboard", icon: LayoutDashboard },
  { to: "/customers", label: "Customers", sub: "Customers", icon: Users },
  { to: "/predict", label: "Churn Prediction", sub: "Predict", icon: BrainCircuit },
  { to: "/segments", label: "Segmentation", sub: "Segments", icon: PieChart },
  { to: "/retention", label: "Retention", sub: "Retention", icon: HeartHandshake },
  { to: "/monitoring", label: "Monitoring", sub: "Monitoring", icon: Gauge },
  { to: "/models", label: "Models", sub: "Models", icon: ShieldCheck },
] as const;

function Logo({ compact }: { compact?: boolean }) {
  return (
    <Link to="/" className="flex items-center gap-2.5">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-soft">
        <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M5 5l7 14 7-14" /></svg>
      </div>
      {!compact && (
        <div className="leading-tight">
          <p className="font-display text-[15px] font-bold">BaBank</p>
          <p className="text-[11px] text-muted-foreground">Retention Analytics</p>
        </div>
      )}
    </Link>
  );
}

function NavList({ collapsed, onNavigate }: { collapsed?: boolean; onNavigate?: () => void }) {
  const path = useRouterState({ select: (s) => s.location.pathname });
  return (
    <nav className="flex flex-col gap-1">
      {NAV.map((item) => {
        const active = item.to === "/" ? path === "/" : path.startsWith(item.to);
        return (
          <Link
            key={item.to}
            to={item.to}
            onClick={onNavigate}
            title={collapsed ? item.label : undefined}
            className={cn(
              "group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors",
              active ? "bg-sidebar-accent font-semibold text-sidebar-accent-foreground" : "text-sidebar-foreground hover:bg-muted",
              collapsed && "justify-center px-0",
            )}
          >
            <item.icon className={cn("h-[18px] w-[18px] shrink-0", active ? "text-primary" : "text-muted-foreground group-hover:text-primary")} />
            {!collapsed && (
              <span className="flex flex-1 items-baseline justify-between gap-2">
                {item.label}
                <span className="text-[10px] font-normal uppercase tracking-wider text-muted-foreground/70">{item.sub}</span>
              </span>
            )}
          </Link>
        );
      })}
    </nav>
  );
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => api.health(signal),
    enabled: isApiConfigured,
    refetchInterval: 30000,
    retry: false,
  });
}

function TopBar({ onMenu }: { onMenu: () => void }) {
  const navigate = useNavigate();
  const [q, setQ] = useState("");
  const health = useHealth();
  const submit = (e: FormEvent) => {
    e.preventDefault();
    navigate({ to: "/customers", search: { search: q.trim() || undefined } });
  };
  const status = !isApiConfigured ? "off" : health.isSuccess && health.data?.model_loaded === true ? "ok" : health.isSuccess ? "err" : health.isError ? "err" : "wait";
  const notices = [
    status === "off" && "API is not configured. Customer records require a connection.",
    status === "err" && "Unable to reach the API health endpoint (/health).",
    status === "ok" && "The model API is available.",
    "Displayed model metrics come from the source study, not live deployment.",
  ].filter(Boolean) as string[];

  return (
    <header className="sticky top-0 z-20 flex h-16 items-center gap-3 border-b bg-background/85 px-4 backdrop-blur md:px-8">
      <button onClick={onMenu} className="rounded-lg p-2 text-muted-foreground hover:bg-muted lg:hidden" aria-label="Open menu">
        <Menu className="h-5 w-5" />
      </button>
      <form onSubmit={submit} className="relative max-w-md flex-1">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search customer ID or country…"
          aria-label="Search customers"
          className="h-10 w-full rounded-xl border bg-card pl-9 pr-3 text-sm outline-none transition focus:border-primary focus:ring-2 focus:ring-ring/20"
        />
      </form>
      <div className="ml-auto flex items-center gap-3">
        <span className={cn(
          "hidden items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium sm:inline-flex",
          status === "ok" ? "border-success/30 bg-success-soft text-success" : status === "err" ? "border-danger/30 bg-danger-soft text-danger" : "bg-card text-muted-foreground",
        )}>
          <span className={cn("h-1.5 w-1.5 rounded-full", status === "ok" ? "bg-success" : status === "err" ? "bg-danger" : "bg-muted-foreground")} />
          {status === "ok" ? "API online" : status === "err" ? "API error" : status === "wait" ? "Checking" : "API offline"}
        </span>
        <DropdownMenu>
          <DropdownMenuTrigger className="relative rounded-xl border bg-card p-2.5 text-muted-foreground hover:text-primary" aria-label="Notifications">
            <Bell className="h-4 w-4" />
            <span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-primary" />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-80">
            <DropdownMenuLabel>System Notifications</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {notices.map((n) => (
              <div key={n} className="px-2 py-2 text-xs leading-relaxed text-muted-foreground">{n}</div>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-lilac text-xs font-bold text-secondary-foreground">NV</div>
      </div>
    </header>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex min-h-screen w-full">
      <aside className={cn(
        "sticky top-0 hidden h-screen shrink-0 flex-col border-r bg-sidebar transition-[width] duration-200 lg:flex",
        collapsed ? "w-[76px] px-3" : "w-64 px-4",
      )}>
        <div className={cn("flex h-16 items-center", collapsed && "justify-center")}>
          <Logo compact={collapsed} />
        </div>
        <div className="mt-4 flex-1 overflow-y-auto">
          {!collapsed && <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Workspace</p>}
          <NavList collapsed={collapsed} />
        </div>
        <button
          onClick={() => setCollapsed((c) => !c)}
          className={cn("mb-4 flex items-center gap-2 rounded-xl px-3 py-2 text-xs text-muted-foreground hover:bg-muted", collapsed && "justify-center")}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronsRight className="h-4 w-4" /> : <><ChevronsLeft className="h-4 w-4" /> Collapse</>}
        </button>
      </aside>

      <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
        <SheetContent side="left" className="w-72 bg-sidebar p-4">
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          <div className="mb-6 flex h-10 items-center"><Logo /></div>
          <NavList onNavigate={() => setMobileOpen(false)} />
        </SheetContent>
      </Sheet>

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onMenu={() => setMobileOpen(true)} />
        {!isApiConfigured && (
          <div className="flex items-center gap-2 border-b border-warning/30 bg-warning-soft px-4 py-2 text-xs font-medium md:px-8">
            <WifiOff className="h-3.5 w-3.5 text-warning" />
            API not connected — preview mode only
            <span className="hidden text-muted-foreground md:inline">· Set VITE_API_BASE_URL to connect to the model.</span>
          </div>
        )}
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-8 md:px-8">{children}</main>
        <footer className="px-4 pb-6 text-[11px] text-muted-foreground md:px-8">
          BaBank Analytics · Internal staff analytics tool{isApiConfigured && <> · API: <span className="font-mono">{API_BASE_URL}</span></>}
        </footer>
      </div>
    </div>
  );
}