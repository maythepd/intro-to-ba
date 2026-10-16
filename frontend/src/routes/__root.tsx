import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Outlet, Link, createRootRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/layout/AppShell";
const queryClient = new QueryClient();
export const Route = createRootRoute({
 component: () => <QueryClientProvider client={queryClient}><AppShell><Outlet /></AppShell></QueryClientProvider>,
 notFoundComponent: () => <main className="p-8"><h1>404</h1><Link to="/">Về tổng quan</Link></main>,
});
