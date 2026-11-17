import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ErrorState, KeyValueGrid, Notice, PageHeader, Panel, SourceBadge } from "@/components/common";
import { api, describeError, isApiConfigured } from "@/lib/api";
import { MODEL_BENCHMARKS } from "@/lib/demo-data";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/models")({
  head: () => pageMeta("Models", "Logistic Regression, Random Forest, and XGBoost evaluation results."),
  component: Models,
});

function Models() {
  const x = MODEL_BENCHMARKS.find((m) => m.model === "XGBoost")!;
  const q = useQuery({ queryKey: ["metrics"], queryFn: ({ signal }) => api.metrics(signal), enabled: isApiConfigured, retry: false });
  const metrics = [
    ["ROC-AUC", x.rocAuc],
    ["PR-AUC", x.prAuc],
    ["F1", x.f1],
    ["Recall", x.recall],
  ] as const;

  return (
    <>
      <PageHeader eyebrow="Models" title="Model Development Results" description="Comparison of Logistic Regression, Random Forest, and XGBoost for customer churn." />
      <Notice tone="warning" title="Module 5 checkpoint and study results">
        The API serves the saved Module 5 XGBoost checkpoint. Its preprocessing must reproduce the committed held-out study metrics before it is enabled. These figures describe the study test set; ongoing deployment performance requires new labels. At the reported threshold, XGBoost recalled 100% of churners with 20.4% precision and 20.4% accuracy.
      </Notice>
      <div className="mt-6 grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        {metrics.map(([label, value]) => (
          <div key={label} className="rounded-2xl border bg-card p-5 shadow-soft">
            <div className="flex items-center justify-between"><p className="text-xs text-muted-foreground">XGBoost {label}</p><SourceBadge /></div>
            <p className="mt-2 font-display text-3xl font-bold tabular">{value.toFixed(4)}</p>
          </div>
        ))}
      </div>
      <Panel className="mt-6" title="Three-model comparison" subtitle="Source: model_development/results/comparison/model_comparison_metrics.csv">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="border-b text-muted-foreground">
              <tr>{["Model", "ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Accuracy"].map((label) => <th key={label} className="px-3 py-3 font-medium">{label}</th>)}</tr>
            </thead>
            <tbody className="divide-y">
              {MODEL_BENCHMARKS.map((row) => (
                <tr key={row.model}>
                  <td className="px-3 py-3 font-medium">{row.model}</td>
                  {[row.rocAuc, row.prAuc, row.f1, row.precision, row.recall, row.accuracy].map((v, i) => <td key={i} className="px-3 py-3 tabular-nums">{v.toFixed(4)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          Source CSV is committed with the model-training pipeline and plots. These figures are not refreshed by demo inference.
        </p>
      </Panel>
      <Panel className="mt-6" title="Metrics from API" subtitle="GET /api/metrics · reads the same committed study CSV">
        {!isApiConfigured ? <p className="text-sm text-muted-foreground">API Not Connected. The source comparison above remains available.</p>
          : q.isError ? <ErrorState message={describeError(q.error)} onRetry={() => q.refetch()} retrying={q.isFetching} />
          : q.data ? <KeyValueGrid data={{ model: q.data.model, serving_source: q.data.serving_source, study_reproduced: q.data.study_reproduced, checkpoint_sha256: q.data.checkpoint_sha256, source: q.data.source, roc_auc: q.data.roc_auc, pr_auc: q.data.pr_auc, f1: q.data.f1, precision: q.data.precision, recall: q.data.recall, deployment_evaluated: q.data.deployment_evaluated }} />
          : <div className="h-20 animate-pulse rounded-xl bg-muted" />}
      </Panel>
    </>
  );
}
