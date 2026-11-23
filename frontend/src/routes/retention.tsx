import { createFileRoute } from "@tanstack/react-router";
import { Layers, MessageSquareWarning, MoonStar, TrendingDown } from "lucide-react";
import { Notice, PageHeader } from "@/components/common";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/retention")({
  head: () => pageMeta("Retention", "Recommended retention actions based on behavioral signals."),
  component: Retention,
});

const RECS = [
  { icon: TrendingDown, title: "Declining Transactions", signal: "trans_trend_ratio below 1 and fewer transactions per month", actions: ["Proactively contact the customer within 7 days", "Offer a limited-time transaction fee discount", "Review changes in customer needs"] },
  { icon: MessageSquareWarning, title: "Complaints", signal: "Unresolved or repeated complaints", actions: ["Prioritize full complaint resolution", "Follow up to confirm satisfaction", "Escalate to a relationship manager"] },
  { icon: MoonStar, title: "Dormancy", signal: "High recency_days and infrequent app logins", actions: ["Send a reminder through the preferred channel", "Help the customer use mobile app features", "Offer a small reactivation incentive"] },
  { icon: Layers, title: "Product Bundles", signal: "Only one product or an excessive number of products", actions: ["Recommend relevant complementary products", "Review product bundles when there are 3 or more products", "Avoid aggressive cross-selling"] },
];

function Retention() {
  return (
    <>
      <PageHeader eyebrow="Retention" title="Retention Recommendations" description="Suggested actions grouped by risk signals." />
      <Notice tone="warning" title="Guidance Only Not Proof of Causation">These signals correlate with churn but do not establish causality. Combine them with staff judgment.</Notice>
      <div className="mt-6 grid gap-5 md:grid-cols-2">
        {RECS.map((r) => (
          <div key={r.title} className="rounded-2xl border bg-card p-6 shadow-soft">
            <div className="flex items-center gap-3"><div className="flex h-10 w-10 items-center justify-center rounded-xl bg-lilac text-primary"><r.icon className="h-5 w-5" /></div><h2 className="font-semibold">{r.title}</h2></div>
            <p className="mt-3 text-xs text-muted-foreground">Signal: {r.signal}</p>
            <ul className="mt-3 space-y-2">{r.actions.map((a) => <li key={a} className="flex gap-2 text-sm"><span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />{a}</li>)}</ul>
          </div>
        ))}
      </div>
    </>
  );
}