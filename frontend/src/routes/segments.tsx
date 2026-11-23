import { useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Layers3 } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Button } from "@/components/ui/button";
import { EmptyState, ErrorState, Notice, PageHeader, Panel } from "@/components/common";
import { api, ApiError, describeError, isApiConfigured, type ClusterAlgorithm } from "@/lib/api";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/segments")({
  head: () => pageMeta("Segmentation", "KMeans and Hierarchical customer segments from official data."),
  component: Segments,
});

function Segments() {
  const [algorithm, setAlgorithm] = useState<ClusterAlgorithm>("kmeans");
  const [customerId, setCustomerId] = useState("");
  const [searchedId, setSearchedId] = useState("");
  const q = useQuery({
    queryKey: ["segments", algorithm],
    queryFn: ({ signal }) => api.segments(algorithm, signal),
    enabled: isApiConfigured, retry: false,
  });
  const member = useQuery({
    queryKey: ["segment-customer", searchedId],
    queryFn: ({ signal }) => api.segmentCustomer(searchedId, signal),
    enabled: isApiConfigured && searchedId.length > 0, retry: false,
  });
  const unavailable = !isApiConfigured || (q.error instanceof ApiError && q.error.status === 503);
  const summary = q.data;
  return (
    <>
      <PageHeader eyebrow="Segments" title="Customer Segmentation"
        description="Compare two clustering algorithms fitted on the official customer-feature snapshot." />
      <div className="mt-4 flex flex-wrap gap-2">
        {(["kmeans", "hierarchical"] as const).map((name) => (
          <Button key={name} type="button" variant={algorithm === name ? "default" : "outline"}
            onClick={() => setAlgorithm(name)}>
            {name === "kmeans" ? "KMeans" : "Hierarchical (Ward)"}
          </Button>
        ))}
      </div>
      <div className="mt-5">
        <Notice tone="info" title="Descriptive model results, not live bank measurements">
          Churn is not used to fit either clustering model. Cluster IDs are categories, not risk rankings.
          Hierarchical is fitted with Ward linkage on a representative subset; assignments outside that subset
          are approximate nearest-centroid matches. The upstream research reports are kept separately.
        </Notice>
      </div>
      {unavailable ? (
        <div className="mt-6">
          <EmptyState icon={<Layers3 />} title="Clustering model unavailable">
            Run the official-data preparation and segmentation export, then start the API.
            No synthetic fallback or invented segment assignment is shown.
          </EmptyState>
        </div>
      ) : q.isError ? (
        <div className="mt-6"><ErrorState message={describeError(q.error)}
          onRetry={() => q.refetch()} retrying={q.isFetching} /></div>
      ) : q.isLoading || !summary ? (
        <div className="mt-6 h-48 animate-pulse rounded-2xl bg-muted" />
      ) : (
        <>
          <div className="mt-6 grid gap-4 sm:grid-cols-3">
            {[
              ["Customers", summary.customer_count.toLocaleString()],
              ["Number of Clusters", String(summary.n_clusters)],
              ["Sample Silhouette", summary.silhouette_sample == null ? "N/A" : summary.silhouette_sample.toFixed(4)],
            ].map(([label, value]) => (
              <div key={label} className="rounded-2xl border bg-card p-5 shadow-soft">
                <p className="text-xs text-muted-foreground">{label}</p>
                <p className="mt-2 font-display text-2xl font-bold tabular">{value}</p>
              </div>
            ))}
          </div>
          <Panel className="mt-6" title="Cluster size" subtitle={summary.assignment_method}>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={summary.clusters} margin={{ left: 0, right: 10, top: 8, bottom: 8 }}>
                  <CartesianGrid vertical={false} stroke="var(--color-border)" />
                  <XAxis dataKey="name" fontSize={12} tickLine={false} />
                  <YAxis fontSize={12} tickLine={false} />
                  <Tooltip formatter={(value) => Number(value).toLocaleString()} />
                  <Bar dataKey="customers" name="Customers" fill="var(--color-chart-1)" radius={[5, 5, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
          <div className="mt-6 grid gap-5 sm:grid-cols-2">
            {summary.clusters.map((cluster) => (
              <Panel key={cluster.id} title={cluster.name}
                subtitle={`${cluster.customers.toLocaleString()} customers · ${(cluster.share * 100).toFixed(1)}% of snapshot`}>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between"><span>Average balance</span><strong>{cluster.avg_balance.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong></div>
                  <div className="flex justify-between"><span>Transactions / month</span><strong>{cluster.avg_monthly_transactions.toFixed(1)}</strong></div>
                  <div className="flex justify-between"><span>App logins / month</span><strong>{cluster.avg_app_logins_monthly.toFixed(1)}</strong></div>
                  {cluster.observed_churn_rate != null && (
                    <div className="flex justify-between"><span>Observed churn (descriptive)</span><strong>{(cluster.observed_churn_rate * 100).toFixed(1)}%</strong></div>
                  )}
                </div>
              </Panel>
            ))}
          </div>
          <p className="mt-4 text-xs text-muted-foreground">{summary.study_note}</p>
        </>
      )}
      <Panel className="mt-6" title="Find a customer's clusters" subtitle="Look up both models from the fitted snapshot by customer ID">
        <form className="flex flex-wrap gap-2" onSubmit={(e) => { e.preventDefault(); setSearchedId(customerId.trim()); }}>
          <input aria-label="Customer ID" type="text" inputMode="numeric" value={customerId}
            onChange={(e) => setCustomerId(e.target.value)}
            placeholder="Customer ID" className="min-w-48 flex-1 rounded-xl border bg-background px-3 py-2 text-sm" />
          <Button type="submit" disabled={!isApiConfigured || customerId.trim().length === 0}>Look up</Button>
        </form>
        {searchedId && member.isLoading && <p className="mt-3 text-sm text-muted-foreground">Checking customer membership…</p>}
        {searchedId && member.isError && <div className="mt-3"><p className="text-sm text-muted-foreground">
          {member.error instanceof ApiError && member.error.status === 404 ? "Customer not in the trained snapshot." : describeError(member.error)}
        </p></div>}
        {member.data && (
          <div className="mt-4 space-y-2 text-sm">
            <p><strong>KMeans:</strong> Cluster {member.data.kmeans}</p>
            <p><strong>Hierarchical:</strong> Cluster {member.data.hierarchical} ({member.data.hierarchical_assignment_method.replaceAll("_", " ")})</p>
            <Link to="/customers/$id" params={{ id: member.data.customer_id }} className="text-primary underline">Open customer profile</Link>
          </div>
        )}
      </Panel>
    </>
  );
}
