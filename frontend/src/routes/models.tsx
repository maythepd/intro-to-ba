import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  AlertCircle,
  BrainCircuit,
  Check,
  CheckCircle2,
  Copy,
  Database,
  FileCode,
  HardDrive,
  Layers,
  Sparkles,
  Terminal,
} from "lucide-react";
import {
  ErrorState,
  KeyValueGrid,
  Notice,
  PageHeader,
  Panel,
  SourceBadge,
} from "@/components/common";
import { Button } from "@/components/ui/button";
import {
  api,
  describeError,
  isApiConfigured,
  type FeatureImportanceItem,
  type ModelComparisonRow,
} from "@/lib/api";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/models")({
  head: () =>
    pageMeta(
      "Models",
      "Repository model checkpoints, hyperparameters, comparison benchmarks, decile lift, and feature importance.",
    ),
  component: Models,
});

export function Models() {
  const [selectedFeatureModel, setSelectedFeatureModel] = useState<
    "xgboost" | "logistic_regression" | "random_forest" | "svm"
  >("xgboost");

  const [selectedLiftModel, setSelectedLiftModel] = useState<"xgboost" | "lr" | "rf">("xgboost");

  const [ckptFilter, setCkptFilter] = useState<"all" | "available" | "unavailable">("all");

  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(text);
    setTimeout(() => setCopiedCmd(null), 2500);
  };

  // 1. Model comparison metrics (prioritizing results/{model}/ over results/comparison/)
  const metricsQ = useQuery({
    queryKey: ["metrics"],
    queryFn: ({ signal }) => api.metrics(signal),
    enabled: isApiConfigured,
    retry: false,
  });

  // 2. Real saved model checkpoints in repo (models/saved/ and artifacts/)
  const ckptQ = useQuery({
    queryKey: ["model-checkpoints"],
    queryFn: ({ signal }) => api.modelCheckpoints(signal),
    enabled: isApiConfigured,
    retry: false,
  });

  // 3. Decile lift table (model-specific, prioritizing results/{model}/)
  const decileQ = useQuery({
    queryKey: ["model-decile-lift", selectedLiftModel],
    queryFn: ({ signal }) => api.decileLift(selectedLiftModel, signal),
    enabled: isApiConfigured,
    retry: false,
  });

  // 4. Feature importance from trained models (prioritizing results/{model}/)
  const featQ = useQuery({
    queryKey: ["model-feature-importance"],
    queryFn: ({ signal }) => api.featureImportance(signal),
    enabled: isApiConfigured,
    retry: false,
  });

  const x = metricsQ.data;

  const topKpis = [
    { label: "ROC-AUC", value: x?.roc_auc, note: "Champion test discrimination" },
    { label: "PR-AUC", value: x?.pr_auc, note: "Area under Precision-Recall curve" },
    { label: "F1 Score", value: x?.f1, note: "Balanced harmonic mean" },
    { label: "Recall", value: x?.recall, note: "True positive churn capture" },
  ];

  // Dynamic model comparison rows directly from prioritized results
  const comparisonRows = (metricsQ.data?.models as ModelComparisonRow[]) ?? [];

  // Feature importance data
  const currentFeatures: FeatureImportanceItem[] =
    (featQ.data?.features?.[selectedFeatureModel] as FeatureImportanceItem[]) ??
    (featQ.data?.[selectedFeatureModel] as FeatureImportanceItem[]) ??
    [];
  const currentFeatureSource = featQ.data?.sources?.[selectedFeatureModel] ?? "unavailable";
  const isFeatureAvailable = currentFeatures.length > 0 && currentFeatureSource !== "unavailable";

  // Checkpoints data
  const allCheckpoints = ckptQ.data ?? [];
  const totalCkpts = allCheckpoints.length;
  const availCkpts = allCheckpoints.filter((c) => c.available).length;
  const unavailCkpts = totalCkpts - availCkpts;

  const filteredCheckpoints = allCheckpoints.filter((ckpt) => {
    if (ckptFilter === "available") return ckpt.available;
    if (ckptFilter === "unavailable") return !ckpt.available;
    return true;
  });

  // Decile lift data
  const isDecileAvailable = decileQ.data?.available ?? false;
  const decileSource = decileQ.data?.source ?? "unavailable";
  const decilesList = decileQ.data?.deciles ?? [];

  return (
    <>
      <PageHeader
        eyebrow="Models"
        title="Model Checkpoints & Performance Benchmarks"
        description="Dynamic inspection of trained weights, hyperparameters, validation benchmarks, decile lift tables, and feature importances detected directly in repository directories."
      />

      <Notice tone="info" title="Priority Model Results Scanner">
        Hệ thống tự động ưu tiên quét trực tiếp thư mục kết quả riêng của từng mô hình trong{" "}
        <code className="text-xs font-semibold">results/&#123;model&#125;/</code> (như{" "}
        <code className="text-xs font-semibold">results/xgboost/</code>,{" "}
        <code className="text-xs font-semibold">results/lr/</code>) trước khi tìm trong{" "}
        <code className="text-xs font-semibold">results/comparison/</code>. Trạng thái{" "}
        <strong>Available</strong> và <strong>Unavailable</strong> được cập nhật tự động theo thực trạng file trên ổ đĩa.
      </Notice>

      {/* Top 4 Champion XGBoost KPI Cards */}
      <div className="mt-6 grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        {topKpis.map((kpi) => (
          <div key={kpi.label} className="rounded-2xl border bg-card p-5 shadow-soft">
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium text-muted-foreground">XGBoost {kpi.label}</p>
              <SourceBadge>Champion (results/xgboost/)</SourceBadge>
            </div>
            <p className="mt-2 font-display text-3xl font-bold tabular">
              {metricsQ.isLoading ? (
                <span className="inline-block h-8 w-24 animate-pulse rounded bg-muted" />
              ) : typeof kpi.value === "number" ? (
                kpi.value.toFixed(4)
              ) : (
                "—"
              )}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">{kpi.note}</p>
          </div>
        ))}
      </div>

      {/* Benchmark Comparison Table with Available / Unavailable Status */}
      <Panel
        className="mt-6"
        title="Model Benchmark Comparison & Availability"
        subtitle="Priority scan order: results/{model}/ -> results/comparison/ · Tracks performance across all model families"
      >
        {!isApiConfigured ? (
          <p className="text-sm text-muted-foreground">API not configured.</p>
        ) : metricsQ.isLoading ? (
          <div className="space-y-2 py-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-10 animate-pulse rounded-lg bg-muted" />
            ))}
          </div>
        ) : metricsQ.isError ? (
          <ErrorState message={describeError(metricsQ.error)} onRetry={() => metricsQ.refetch()} />
        ) : comparisonRows.length === 0 ? (
          <p className="text-sm text-muted-foreground">No comparison data returned by the server.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="border-b text-muted-foreground">
                <tr>
                  <th className="px-3 py-3 font-medium">Model</th>
                  <th className="px-3 py-3 font-medium">Status</th>
                  <th className="px-3 py-3 font-medium">Source Location</th>
                  <th className="px-3 py-3 font-medium">ROC-AUC</th>
                  <th className="px-3 py-3 font-medium">PR-AUC</th>
                  <th className="px-3 py-3 font-medium">F1-Score</th>
                  <th className="px-3 py-3 font-medium">Precision</th>
                  <th className="px-3 py-3 font-medium">Recall</th>
                  <th className="px-3 py-3 font-medium">Accuracy</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {comparisonRows.map((row) => {
                  const isChampion = row.model_id === "xgboost";
                  return (
                    <tr
                      key={row.model_id}
                      className={
                        isChampion
                          ? "bg-lilac/30 font-medium"
                          : !row.available
                          ? "opacity-60 bg-muted/20"
                          : ""
                      }
                    >
                      <td className="px-3 py-3">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold">{row.model}</span>
                          {isChampion && (
                            <span className="inline-flex items-center gap-1 rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-bold text-primary">
                              <Sparkles className="h-3 w-3" /> Champion
                            </span>
                          )}
                        </div>
                        <span className="text-[11px] text-muted-foreground">{row.family}</span>
                      </td>

                      <td className="px-3 py-3">
                        {row.available ? (
                          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] font-semibold text-emerald-600 dark:text-emerald-400">
                            <CheckCircle2 className="h-3 w-3" /> Available
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/10 px-2 py-0.5 text-[11px] font-semibold text-amber-600 dark:text-amber-400">
                            <AlertCircle className="h-3 w-3" /> Unavailable
                          </span>
                        )}
                      </td>

                      <td className="px-3 py-3 font-mono text-xs text-muted-foreground">
                        {row.available ? (
                          <span title={row.source_location}>{row.source_location}</span>
                        ) : (
                          <div className="flex items-center gap-1.5 text-amber-600/90 dark:text-amber-400/90">
                            <Terminal className="h-3 w-3" />
                            <code className="text-[11px] font-medium">{row.train_command}</code>
                          </div>
                        )}
                      </td>

                      <td className="px-3 py-3 tabular-nums font-semibold">
                        {row.roc_auc != null ? row.roc_auc.toFixed(4) : "—"}
                      </td>
                      <td className="px-3 py-3 tabular-nums font-semibold">
                        {row.pr_auc != null ? row.pr_auc.toFixed(4) : "—"}
                      </td>
                      <td className="px-3 py-3 tabular-nums">
                        {row.f1 != null ? row.f1.toFixed(4) : "—"}
                      </td>
                      <td className="px-3 py-3 tabular-nums">
                        {row.precision != null ? row.precision.toFixed(4) : "—"}
                      </td>
                      <td className="px-3 py-3 tabular-nums">
                        {row.recall != null ? row.recall.toFixed(4) : "—"}
                      </td>
                      <td className="px-3 py-3 tabular-nums">
                        {row.accuracy != null ? row.accuracy.toFixed(4) : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {/* Model Checkpoints & Parameter Inspector with Availability Filtering */}
      <Panel
        className="mt-6"
        title="Detected Repository Checkpoints & Parameter Inspector"
        subtitle="Automatic filesystem discovery of model weights, state dicts, and training configurations in models/saved/ & artifacts/"
        action={
          <div className="flex items-center gap-1.5">
            <Button
              size="sm"
              variant={ckptFilter === "all" ? "default" : "outline"}
              onClick={() => setCkptFilter("all")}
            >
              All ({totalCkpts})
            </Button>
            <Button
              size="sm"
              variant={ckptFilter === "available" ? "default" : "outline"}
              onClick={() => setCkptFilter("available")}
            >
              Available ({availCkpts})
            </Button>
            <Button
              size="sm"
              variant={ckptFilter === "unavailable" ? "default" : "outline"}
              onClick={() => setCkptFilter("unavailable")}
            >
              Unavailable ({unavailCkpts})
            </Button>
          </div>
        }
      >
        {!isApiConfigured ? (
          <p className="text-sm text-muted-foreground">API not configured.</p>
        ) : ckptQ.isLoading ? (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="h-48 animate-pulse rounded-xl bg-muted" />
            ))}
          </div>
        ) : ckptQ.isError ? (
          <ErrorState message={describeError(ckptQ.error)} onRetry={() => ckptQ.refetch()} />
        ) : filteredCheckpoints.length === 0 ? (
          <p className="text-sm text-muted-foreground">No checkpoints match the selected filter.</p>
        ) : (
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {filteredCheckpoints.map((ckpt) => {
              const isChampion = ckpt.status === "production_champion";
              return (
                <div
                  key={ckpt.name}
                  className={`flex flex-col justify-between rounded-xl border p-4 shadow-sm transition-all ${
                    isChampion
                      ? "border-primary/40 bg-lilac/10"
                      : !ckpt.available
                      ? "border-dashed border-border/80 bg-muted/20 opacity-80"
                      : "bg-background"
                  }`}
                >
                  <div>
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <div
                          className={`flex h-8 w-8 items-center justify-center rounded-lg ${
                            isChampion
                              ? "bg-primary text-primary-foreground"
                              : ckpt.available
                              ? "bg-lilac text-primary"
                              : "bg-muted text-muted-foreground"
                          }`}
                        >
                          {ckpt.type === "deep_learning_temporal" || ckpt.type === "deep_learning_gnn" ? (
                            <Layers className="h-4 w-4" />
                          ) : ckpt.type === "clustering" ? (
                            <Database className="h-4 w-4" />
                          ) : (
                            <BrainCircuit className="h-4 w-4" />
                          )}
                        </div>
                        <div>
                          <h4 className="text-sm font-semibold">{ckpt.name}</h4>
                          <span className="text-[11px] font-mono text-muted-foreground">{ckpt.path}</span>
                        </div>
                      </div>

                      {/* Status Badges */}
                      {isChampion ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-bold text-primary uppercase">
                          <Sparkles className="h-3 w-3" /> Champion
                        </span>
                      ) : ckpt.available ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-bold text-emerald-600 dark:text-emerald-400 uppercase">
                          <CheckCircle2 className="h-3 w-3" /> Available
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/10 px-2 py-0.5 text-[10px] font-bold text-amber-600 dark:text-amber-400 uppercase">
                          <AlertCircle className="h-3 w-3" /> Unavailable
                        </span>
                      )}
                    </div>

                    <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <HardDrive className="h-3.5 w-3.5" />{" "}
                        {ckpt.size_kb > 1024
                          ? `${(ckpt.size_kb / 1024).toFixed(1)} MB`
                          : ckpt.size_kb > 0
                          ? `${ckpt.size_kb} KB`
                          : "Not created"}
                      </span>
                      {ckpt.modified_at && (
                        <span className="flex items-center gap-1">
                          <FileCode className="h-3.5 w-3.5" /> {ckpt.modified_at}
                        </span>
                      )}
                    </div>

                    {/* Metrics if available */}
                    {ckpt.metrics && Object.keys(ckpt.metrics).length > 0 && (
                      <div className="mt-3 grid grid-cols-3 gap-1.5 rounded-lg bg-muted/60 p-2 text-center text-xs">
                        {ckpt.metrics.roc_auc !== undefined && (
                          <div>
                            <p className="text-[10px] text-muted-foreground">ROC-AUC</p>
                            <p className="font-bold tabular">{ckpt.metrics.roc_auc.toFixed(4)}</p>
                          </div>
                        )}
                        {ckpt.metrics.pr_auc !== undefined && (
                          <div>
                            <p className="text-[10px] text-muted-foreground">PR-AUC</p>
                            <p className="font-bold tabular">{ckpt.metrics.pr_auc.toFixed(4)}</p>
                          </div>
                        )}
                        {ckpt.metrics.f1 !== undefined && (
                          <div>
                            <p className="text-[10px] text-muted-foreground">F1 Score</p>
                            <p className="font-bold tabular">{ckpt.metrics.f1.toFixed(4)}</p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Source File Location */}
                    {ckpt.source_location && ckpt.source_location !== "unavailable" && (
                      <div className="mt-2 text-[11px] text-muted-foreground">
                        <span className="font-medium text-foreground">Scanned from:</span>{" "}
                        <code className="font-mono text-[10px] bg-muted/70 px-1 py-0.5 rounded">
                          {ckpt.source_location}
                        </code>
                      </div>
                    )}

                    {/* Hyperparameters / Architecture */}
                    {Object.keys(ckpt.parameters).length > 0 && (
                      <div className="mt-3">
                        <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                          Parameters & Architecture
                        </p>
                        <div className="mt-1.5 max-h-32 overflow-y-auto rounded-lg bg-muted/50 p-2 text-xs font-mono">
                          {Object.entries(ckpt.parameters).map(([k, v]) => (
                            <div key={k} className="flex justify-between py-0.5">
                              <span className="text-muted-foreground">{k}:</span>
                              <span className="font-medium text-foreground">
                                {Array.isArray(v) ? v.join(", ") : String(v)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* CLI Command Helper for Unavailable Models */}
                    {!ckpt.available && ckpt.train_command && (
                      <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/5 p-2.5 text-xs">
                        <p className="font-semibold text-amber-700 dark:text-amber-300">
                          To train this model and generate results:
                        </p>
                        <div className="mt-1.5 flex items-center justify-between rounded bg-background px-2 py-1 font-mono text-[11px]">
                          <span className="truncate">{ckpt.train_command}</span>
                          <button
                            type="button"
                            className="ml-2 text-muted-foreground hover:text-foreground"
                            onClick={() => copyToClipboard(ckpt.train_command!)}
                            title="Copy command"
                          >
                            {copiedCmd === ckpt.train_command ? (
                              <Check className="h-3.5 w-3.5 text-emerald-500" />
                            ) : (
                              <Copy className="h-3.5 w-3.5" />
                            )}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Panel>

      {/* Decile Lift & Cumulative Gains Table with Model Switcher */}
      <Panel
        className="mt-6"
        title="Decile Lift & Cumulative Churn Capture"
        subtitle={`Priority scan: results/${selectedLiftModel}/ -> results/comparison/ · Source: ${decileSource}`}
        action={
          <div className="flex gap-1.5">
            <Button
              size="sm"
              variant={selectedLiftModel === "xgboost" ? "default" : "outline"}
              onClick={() => setSelectedLiftModel("xgboost")}
            >
              XGBoost (results/xgboost/)
            </Button>
            <Button
              size="sm"
              variant={selectedLiftModel === "lr" ? "default" : "outline"}
              onClick={() => setSelectedLiftModel("lr")}
            >
              Logistic Regression (results/lr/)
            </Button>
            <Button
              size="sm"
              variant={selectedLiftModel === "rf" ? "default" : "outline"}
              onClick={() => setSelectedLiftModel("rf")}
            >
              Random Forest
            </Button>
          </div>
        }
      >
        {!isApiConfigured ? (
          <p className="text-sm text-muted-foreground">API not configured.</p>
        ) : decileQ.isLoading ? (
          <div className="h-48 animate-pulse rounded-xl bg-muted" />
        ) : decileQ.isError ? (
          <ErrorState message={describeError(decileQ.error)} onRetry={() => decileQ.refetch()} />
        ) : !isDecileAvailable || decilesList.length === 0 ? (
          <div className="rounded-xl border border-dashed p-6 text-center">
            <AlertCircle className="mx-auto h-8 w-8 text-amber-500" />
            <h4 className="mt-2 text-sm font-semibold">
              Decile lift table not available for {selectedLiftModel.toUpperCase()}
            </h4>
            <p className="mt-1 text-xs text-muted-foreground">
              Chưa có file{" "}
              <code className="font-mono">
                results/{selectedLiftModel}/{selectedLiftModel}_decile_lift_table.csv
              </code>
              . Chạy lệnh sau để huấn luyện và sinh bảng Decile Lift:
            </p>
            <div className="mx-auto mt-3 max-w-md flex items-center justify-between rounded-lg bg-muted px-3 py-1.5 font-mono text-xs">
              <code>python main.py --mode train --model {selectedLiftModel}</code>
              <button
                type="button"
                className="ml-2 text-muted-foreground hover:text-foreground"
                onClick={() =>
                  copyToClipboard(`python main.py --mode train --model ${selectedLiftModel}`)
                }
              >
                {copiedCmd === `python main.py --mode train --model ${selectedLiftModel}` ? (
                  <Check className="h-3.5 w-3.5 text-emerald-500" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
              </button>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[700px] text-left text-sm">
              <thead className="border-b text-muted-foreground">
                <tr>
                  <th className="px-3 py-3 font-medium">Decile</th>
                  <th className="px-3 py-3 font-medium">Customers</th>
                  <th className="px-3 py-3 font-medium">Churners</th>
                  <th className="px-3 py-3 font-medium">Decile Churn Rate</th>
                  <th className="px-3 py-3 font-medium">Decile Lift</th>
                  <th className="px-3 py-3 font-medium">Cum. Recall</th>
                  <th className="px-3 py-3 font-medium">Cum. Lift</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {decilesList.map((row, idx) => {
                  const isTopDecile = idx === 0;
                  const isHighPriority = idx <= 1;
                  return (
                    <tr
                      key={row.decile}
                      className={
                        isTopDecile
                          ? "bg-danger-soft/40 font-semibold"
                          : isHighPriority
                          ? "bg-warning-soft/30 font-medium"
                          : ""
                      }
                    >
                      <td className="px-3 py-2.5">
                        <span className="font-mono">{row.decile}</span>
                        {isTopDecile && (
                          <span className="ml-2 inline-flex rounded bg-danger/10 px-1.5 py-0.2 text-[10px] font-bold text-danger">
                            Top Target
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2.5 tabular-nums">{row.customers.toLocaleString()}</td>
                      <td className="px-3 py-2.5 tabular-nums font-semibold">{row.churners.toLocaleString()}</td>
                      <td className="px-3 py-2.5 tabular-nums">{row.churn_rate}</td>
                      <td className="px-3 py-2.5 tabular-nums font-semibold text-primary">{row.decile_lift}</td>
                      <td className="px-3 py-2.5 tabular-nums">{row.cum_recall}</td>
                      <td className="px-3 py-2.5 tabular-nums font-semibold">{row.cum_lift}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {/* Feature Importance Explorer with Model Switcher & Availability */}
      <Panel
        className="mt-6"
        title="Top Feature Importance Explorer"
        subtitle={`Priority scan: results/{model}/ -> results/comparison/ · Source: ${currentFeatureSource}`}
        action={
          <div className="flex gap-1.5">
            <Button
              size="sm"
              variant={selectedFeatureModel === "xgboost" ? "default" : "outline"}
              onClick={() => setSelectedFeatureModel("xgboost")}
            >
              XGBoost
            </Button>
            <Button
              size="sm"
              variant={selectedFeatureModel === "logistic_regression" ? "default" : "outline"}
              onClick={() => setSelectedFeatureModel("logistic_regression")}
            >
              Logistic Regression
            </Button>
            <Button
              size="sm"
              variant={selectedFeatureModel === "random_forest" ? "default" : "outline"}
              onClick={() => setSelectedFeatureModel("random_forest")}
            >
              Random Forest
            </Button>
            <Button
              size="sm"
              variant={selectedFeatureModel === "svm" ? "default" : "outline"}
              onClick={() => setSelectedFeatureModel("svm")}
            >
              Linear SVM
            </Button>
          </div>
        }
      >
        {!isApiConfigured ? (
          <p className="text-sm text-muted-foreground">API not configured.</p>
        ) : featQ.isLoading ? (
          <div className="h-48 animate-pulse rounded-xl bg-muted" />
        ) : featQ.isError ? (
          <ErrorState message={describeError(featQ.error)} onRetry={() => featQ.refetch()} />
        ) : !isFeatureAvailable ? (
          <div className="rounded-xl border border-dashed p-6 text-center">
            <AlertCircle className="mx-auto h-8 w-8 text-amber-500" />
            <h4 className="mt-2 text-sm font-semibold">
              Feature importance not available for {selectedFeatureModel.replace("_", " ").toUpperCase()}
            </h4>
            <p className="mt-1 text-xs text-muted-foreground">
              Mô hình này chưa có bảng feature importance xuất ra trong{" "}
              <code className="font-mono">results/</code>. Chạy lệnh sau để huấn luyện:
            </p>
            <div className="mx-auto mt-3 max-w-md flex items-center justify-between rounded-lg bg-muted px-3 py-1.5 font-mono text-xs">
              <code>python main.py --mode train --model {selectedFeatureModel === "logistic_regression" ? "lr" : selectedFeatureModel === "random_forest" ? "rf" : selectedFeatureModel}</code>
              <button
                type="button"
                className="ml-2 text-muted-foreground hover:text-foreground"
                onClick={() =>
                  copyToClipboard(
                    `python main.py --mode train --model ${
                      selectedFeatureModel === "logistic_regression"
                        ? "lr"
                        : selectedFeatureModel === "random_forest"
                        ? "rf"
                        : selectedFeatureModel
                    }`
                  )
                }
              >
                {copiedCmd?.includes(selectedFeatureModel) ? (
                  <Check className="h-3.5 w-3.5 text-emerald-500" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-2.5 pt-2">
            {currentFeatures.map((f) => (
              <div key={f.feature} className="grid grid-cols-12 items-center gap-3 text-xs">
                <span className="col-span-4 truncate font-mono text-foreground font-medium" title={f.feature}>
                  {f.feature}
                </span>
                <div className="col-span-6 flex items-center">
                  <div className="h-2.5 w-full rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-300"
                      style={{ width: `${Math.min(100, Math.max(2, f.relative_pct))}%` }}
                    />
                  </div>
                </div>
                <div className="col-span-2 text-right font-mono tabular-nums text-muted-foreground">
                  <span className="font-semibold text-foreground">{f.relative_pct.toFixed(1)}%</span>
                  <span className="ml-1 text-[10px]">({f.importance.toFixed(4)})</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </Panel>

      {/* Serving Metadata from API */}
      <Panel
        className="mt-6"
        title="Live Serving Metadata & Checkpoint Verification"
        subtitle="GET /api/metrics · verification of current inference engine"
      >
        {!isApiConfigured ? (
          <p className="text-sm text-muted-foreground">API Not Connected.</p>
        ) : metricsQ.isError ? (
          <ErrorState message={describeError(metricsQ.error)} onRetry={() => metricsQ.refetch()} />
        ) : metricsQ.data ? (
          <KeyValueGrid
            data={{
              model: metricsQ.data.model,
              serving_source: metricsQ.data.serving_source,
              study_reproduced: metricsQ.data.study_reproduced,
              checkpoint_sha256: metricsQ.data.checkpoint_sha256,
              report_path: metricsQ.data.report_path,
              roc_auc: metricsQ.data.roc_auc,
              pr_auc: metricsQ.data.pr_auc,
              f1: metricsQ.data.f1,
              precision: metricsQ.data.precision,
              recall: metricsQ.data.recall,
              deployment_evaluated: metricsQ.data.deployment_evaluated,
            }}
          />
        ) : (
          <div className="h-20 animate-pulse rounded-xl bg-muted" />
        )}
      </Panel>
    </>
  );
}
