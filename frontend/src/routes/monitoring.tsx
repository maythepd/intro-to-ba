import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Activity, Download, RefreshCw } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ErrorState, Notice, PageHeader, Panel } from "@/components/common";
import { Button } from "@/components/ui/button";
import { api, API_BASE_URL, describeError, isApiConfigured, type DistributionBin, type MonitoringSource } from "@/lib/api";
import { pageMeta } from "@/lib/seo";

export const Route = createFileRoute("/monitoring")({
  head: () => pageMeta("Monitoring", "Measured data drift, labeled performance and retraining alerts."), component: Monitoring,
});
const names: Record<string, string> = {
  stable: "Stable", warning: "Warning", critical: "Significant drift", ready: "Measured",
  insufficient_data: "Not enough samples", insufficient_labels: "Not enough labels", single_class: "Only one outcome class",
  awaiting_labels: "Waiting for labels", awaiting_outcome_horizon: "Waiting for outcome horizon",
  review_required: "Review required", retraining_required: "Retraining required", data_quality_alert: "Data quality alert",
  missing_data: "Missing values", unavailable: "Unavailable",
};
const pretty = (value: string) => names[value] ?? value.replaceAll("_", " ");
const number = (value: number | null | undefined, digits = 3) => value == null ? "N/A" : value.toFixed(digits);
const percent = (value: number | null | undefined) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const date = (value: string) => new Date(value).toLocaleString();
const control = "h-10 max-w-full rounded-xl border bg-background px-3 text-sm";

function State({ value }: { value: string }) {
  const tone = ["critical", "retraining_required", "data_quality_alert"].includes(value) ? "bg-danger-soft text-danger"
    : ["warning", "review_required"].includes(value) ? "bg-warning-soft text-warning" : "bg-muted text-muted-foreground";
  return <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ${tone}`}>{pretty(value)}</span>;
}
function Distribution({ data }: { data: DistributionBin[] }) {
  return <div className="h-64 min-w-0" role="img" aria-label="Reference and current population distributions">
    <ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 12, left: -14, right: 12, bottom: 24 }}>
      <CartesianGrid stroke="var(--color-border)" vertical={false} />
      <XAxis dataKey="bin" tick={{ fontSize: 10 }} interval={0} angle={-18} textAnchor="end" height={55} />
      <YAxis tickFormatter={(value: number) => `${Math.round(value * 100)}%`} tick={{ fontSize: 11 }} />
      <Tooltip formatter={(value: number) => percent(value)} /><Legend />
      <Bar dataKey="reference" name="Training reference" fill="var(--color-chart-2)" radius={[3, 3, 0, 0]} />
      <Bar dataKey="current" name="Selected window" fill="var(--color-chart-1)" radius={[3, 3, 0, 0]} />
    </BarChart></ResponsiveContainer>
  </div>;
}

function Monitoring() {
  const [source, setSource] = useState<MonitoringSource>("live");
  const [scenario, setScenario] = useState("stable");
  const [modelId, setModelId] = useState("");
  const [days, setDays] = useState(30);
  const [selectedId, setSelectedId] = useState("");
  const [selectedFeature, setSelectedFeature] = useState("");
  const query = useQuery({ queryKey: ["monitoring", source, scenario, modelId, days],
    queryFn: ({ signal }) => api.monitoring({ source, scenario: source === "study_replay" ? scenario : undefined, modelId: modelId || undefined, days }, signal),
    enabled: isApiConfigured, refetchInterval: 30000, retry: 1 });
  const data = query.data;
  const selected = useQuery({ queryKey: ["monitoring-run", selectedId],
    queryFn: ({ signal }) => api.monitoringRun(selectedId, signal), enabled: !!selectedId && isApiConfigured, retry: false });
  const report = selectedId ? selected.data : data?.latest;
  const features = report?.drift.features ?? [];
  const feature = features.find((item) => item.name === selectedFeature) ?? features[0];
  const history = data?.history ?? [];
  const synthetic = scenario.startsWith("synthetic_demo");
  const scenarios = Array.from(new Set(["stable", "behavior_drift", "performance_drop", ...(data?.scenarios ?? []),
    ...((data?.scenarios ?? []).some((name) => name.startsWith("synthetic_demo")) ? [] : ["synthetic_demo"])]));
  const scenarioName = (name: string) => {
    const info = data?.scenario_details?.find((item) => item.name === name);
    return name.startsWith("synthetic_demo") ? `Synthetic Demo · ${info?.outcome_kind === "simulated" ? `Simulated labels (seed ${info.label_seed})` : "Drift only"} · ${name.slice(15) || "No reports"}`
      : ({ stable: "Stable population", behavior_drift: "Behavioral drift", performance_drop: "Performance deterioration" }[name] ?? name);
  };
  const chart = history.map((run) => ({ window: run.synthetic_replay?.batch_id ?? new Date(run.window_end).toLocaleDateString(),
    roc_auc: run.performance.metrics?.roc_auc ?? null, average_precision: run.performance.metrics?.average_precision ?? null,
    recall: run.performance.metrics?.recall ?? null }));
  const resetSelection = () => { setSelectedId(""); setSelectedFeature(""); };
  const download = (format: "html" | "json") => report ? `${API_BASE_URL}/api/monitoring/runs/${encodeURIComponent(report.id)}/report?format=${format}` : "";

  return <>
    <PageHeader eyebrow="Monitoring" title="Model Monitoring"
      description="Compare populations with the frozen training reference, track labeled outcomes and inspect retraining signals."
      actions={<Button variant="outline" onClick={() => query.refetch()} disabled={!isApiConfigured || query.isFetching}><RefreshCw className={query.isFetching ? "animate-spin" : ""} /> Refresh</Button>} />
    <div className="mb-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <label className="grid gap-2 text-xs font-medium">Data source
        <select className={control} value={source} onChange={(e) => { setSource(e.target.value as MonitoringSource); resetSelection(); }}>
          <option value="live">Incoming API requests</option><option value="snapshot_batch">Study snapshot batch</option><option value="study_replay">Study replay / stress tests</option>
        </select></label>
      <label className="grid gap-2 text-xs font-medium">Scenario
        <select className={control} value={scenario} disabled={source !== "study_replay"} onChange={(e) => { setScenario(e.target.value); resetSelection(); }}>
          {scenarios.map((name) => <option key={name} value={name}>{scenarioName(name)}</option>)}
        </select></label>
      <label className="grid gap-2 text-xs font-medium">Model version
        <select className={control} value={modelId} onChange={(e) => { setModelId(e.target.value); resetSelection(); }}>
          <option value="">Current serving model</option>{data?.models.map((model) => <option key={model.id} value={model.id}>{model.name} · {model.id.slice(0, 12)}</option>)}
        </select></label>
      <label className="grid gap-2 text-xs font-medium">History range
        <select className={control} value={days} onChange={(e) => { setDays(Number(e.target.value)); resetSelection(); }}>
          {[7, 30, 90, 180].map((day) => <option key={day} value={day}>Last {day} days{source === "study_replay" ? " of replay" : ""}</option>)}
        </select></label>
    </div>
    {!isApiConfigured ? <Notice tone="warning" title="API not connected">Connect the model API to view monitoring history.</Notice>
      : query.isPending ? <Panel><div className="flex items-center gap-3 py-8 text-sm text-muted-foreground"><Activity className="animate-pulse" />Loading monitoring history. A sleeping server may take about a minute to start.</div></Panel>
      : query.isError ? <ErrorState message={describeError(query.error)} onRetry={() => query.refetch()} retrying={query.isFetching} />
      : data && <>
        {!data.storage_ready && <Notice tone="danger" title="Monitoring storage unavailable">History could not be collected or loaded. Contact the deployment administrator. Model inference may remain available.</Notice>}
        {data.storage_ready && !data.storage_persistent && <Notice tone="warning" title="Local monitoring history">This environment stores history locally. A cloud deployment needs persistent storage to retain it across restarts.</Notice>}
        {data.last_ingest_error && <div className="mt-3"><Notice tone="warning" title="Monitoring collection needs attention">Some telemetry could not be saved. {data.last_error_at && <>Last failure: {date(data.last_error_at)}.</>} The report retains its original measurement time.</Notice></div>}
        {source === "study_replay" && <div className="mt-3"><Notice tone="warning" title="Simulated study replay">
          {synthetic ? "Five newly synthesized customer batches share the 2026-10-01 feature snapshot. Window dates are simulated ingestion times. Optional outcomes use an independent illustrative policy and do not measure accuracy on real customers. Alerts are demonstrations." : "Timestamps are synthetic. Held-out customers are reused across scenarios and are distinct within each scenario's windows."}
          {scenario === "performance_drop" && " This stress test deliberately replaces study labels with adversarial synthetic outcomes."}
        </Notice></div>}
        {source === "snapshot_batch" && <div className="mt-3"><Notice title="Fixed study snapshot">Repeated scoring of the same snapshot is deduplicated. This is not a stream of newly observed customers.</Notice></div>}
        <div className="my-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {[
            ["Reference", data.reference_available ? "Prepared" : "Unavailable", `Model ${data.model_id?.slice(0, 12) ?? "—"}`],
            ["Selected-window samples", report?.sample_count.toLocaleString() ?? "—", `Minimum ${data.minimum_drift_samples} for drift`],
            ["Labeled coverage", percent(report?.performance.label_coverage), `${report?.performance.labeled_samples ?? 0} / ${report?.performance.mature_samples ?? 0} mature predictions`],
            ["Monitoring state", report ? pretty(report.status) : "No measurements", `${data.window_hours}-hour windows · ${(report?.policy ?? data.policy).consecutive_windows} consecutive windows for alerts`],
          ].map(([title, value, caption]) => <div className="rounded-2xl border bg-card p-5 shadow-soft" key={title}>
            <p className="text-xs text-muted-foreground">{title}</p><p className="mt-2 font-display text-xl font-bold">{value}</p><p className="mt-2 text-xs text-muted-foreground">{caption}</p>
          </div>)}
        </div>
        {!data.latest ? <Panel title="No completed monitoring window">
          <p className="text-sm text-muted-foreground">No report is available for this source and model. The scheduled monitoring job evaluates closed windows. A cohort below the minimum remains marked as insufficient data.</p>
          <p className="mt-3 text-sm text-muted-foreground">Live performance requires outcomes after the {data.outcome_horizon_days}-day horizon. Select Study replay to inspect demonstration scenarios after an operator has generated them.</p>
        </Panel> : <>
          <Panel title="Measurement window" subtitle="Window boundaries use UTC; displayed timestamps use your local timezone.">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <label className="grid min-w-0 max-w-full gap-2 text-xs font-medium">Select a completed window
                <select className={control} value={selectedId} onChange={(e) => { setSelectedId(e.target.value); setSelectedFeature(""); }}>
                  <option value="">Latest · {data.latest.synthetic_replay?.batch_id ?? date(data.latest.window_end)}</option>
                  {history.map((run) => <option key={run.id} value={run.id}>{run.synthetic_replay ? `${run.synthetic_replay.batch_index}/5 · ${run.synthetic_replay.batch_id} · ` : ""}{date(run.window_start)} → {date(run.window_end)} · {pretty(run.status)}</option>)}
                </select></label>
              {report && <div className="flex gap-2"><Button size="sm" variant="outline" asChild><a href={download("html")}><Download /> HTML report</a></Button>
                <Button size="sm" variant="outline" asChild><a href={download("json")}><Download /> JSON</a></Button></div>}
            </div>
            {selected.isError && <div className="mt-4"><ErrorState message={describeError(selected.error)} onRetry={() => selected.refetch()} /></div>}
            {report && <p className="mt-4 break-words text-xs text-muted-foreground">Measured {date(report.computed_at)} · Version {report.model_id.slice(0, 16)} · <State value={report.status} /></p>}
            {data.stale && <p className="mt-3 text-xs text-warning">The latest report is more than two monitoring periods old. Its original timestamp is retained.</p>}
          </Panel>
          {report && <>
            {report.synthetic_replay && <Panel className="mt-6" title={`Batch ${report.synthetic_replay.batch_index}/5 · ${report.synthetic_replay.batch_id}`} subtitle="Simulated time and outcomes; measured model probabilities and drift.">
              <p className="break-all text-xs text-muted-foreground">Dataset {report.synthetic_replay.dataset_sha256} · Generation seed {report.synthetic_replay.generation_seed} · Feature snapshot {report.synthetic_replay.feature_snapshot}.</p>
              <p className="mt-3 text-sm">{report.synthetic_replay.label_policy_version ? `Simulated outcomes: ${report.synthetic_replay.label_policy_version} · seed ${report.synthetic_replay.label_seed}.` : "N/A — no outcomes supplied. Only drift is measured."}</p>
              <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[820px] text-left text-xs"><thead className="border-b"><tr>{["Batch", "N", "Critical share", "Labels", "ROC-AUC", "AP", "Accuracy", "F1", "State"].map((label) => <th className="p-2" key={label}>{label}</th>)}</tr></thead>
                <tbody>{history.map((run) => <tr key={run.id} className="border-b"><td className="p-2"><button className="text-primary underline" onClick={() => setSelectedId(run.id)}>{run.synthetic_replay?.batch_id}</button></td><td className="p-2">{run.sample_count}</td><td className="p-2">{percent(run.drift.critical_share)}</td><td className="p-2">{percent(run.performance.label_coverage)}</td>{(["roc_auc", "average_precision", "accuracy", "f1"] as const).map((key) => <td className="p-2" key={key}>{number(run.performance.metrics?.[key])}</td>)}<td className="p-2"><State value={run.status} /></td></tr>)}</tbody>
              </table></div>
            </Panel>}
            {report.alerts.length > 0 && <Panel className="mt-6" title="Alerts and next actions" subtitle="Signals use this model, source and adjacent non-overlapping windows.">
              <div className="grid gap-3">{report.alerts.map((alert) => <Notice key={alert.id} tone={alert.status === "retraining_required" ? "danger" : "warning"} title={pretty(alert.status)}>
                <p>{alert.action}</p><p className="mt-2 text-xs">Evidence: {JSON.stringify(alert.evidence)}</p>
              </Notice>)}</div>
            </Panel>}
            <Panel className="mt-6" title="Feature drift" subtitle="PSI for numeric features; base-2 Jensen–Shannon distance for categories. Missingness is separate.">
              <State value={report.drift.status} />
              {report.drift.trigger_candidate && <p className="mt-3 text-xs text-warning">Qualifying drift windows: {report.consecutive_candidates?.drift ?? 1} / {(report.policy ?? data.policy).consecutive_windows} required for review.</p>}
              {!features.length ? <p className="mt-4 text-sm text-muted-foreground">Collected {report.sample_count} distinct subjects; at least {data.minimum_drift_samples} are needed.</p>
                : <><div className="mt-4 overflow-x-auto"><table className="w-full min-w-[720px] text-left text-sm">
                  <thead className="border-b text-xs text-muted-foreground"><tr>{["Feature", "Method", "Score", "Status", "Missing", "New categories"].map((label) => <th key={label} className="px-3 py-3 font-medium">{label}</th>)}</tr></thead>
                  <tbody className="divide-y">{features.map((item) => <tr key={item.name} className={feature?.name === item.name ? "bg-muted/60" : ""}>
                    <td className="px-3 py-3"><button onClick={() => setSelectedFeature(item.name)} className="font-medium text-primary underline decoration-dotted">{item.name}</button></td>
                    <td className="px-3 py-3 text-xs">{item.kind === "numeric" ? "PSI" : "JS distance"}</td><td className="px-3 py-3 tabular-nums">{number(item.score)}</td>
                    <td className="px-3 py-3"><State value={item.status} /></td><td className="px-3 py-3">{percent(item.missing_rate)}</td><td className="px-3 py-3">{percent(item.new_category_rate)}</td>
                  </tr>)}</tbody></table></div>
                  {feature && <div className="mt-6"><h3 className="mb-2 text-sm font-semibold">{feature.name}: reference vs selected window</h3><Distribution data={feature.distribution} /></div>}
                </>}
            </Panel>
            <Panel className="mt-6" title="Labeled performance" subtitle="Average Precision is the PR metric. Missing outcomes are never filled with historical customer labels.">
              <State value={report.performance.status} />
              {report.performance.trigger_candidate && <p className="mt-3 text-xs text-warning">Qualifying performance-drop windows: {report.consecutive_candidates?.performance ?? 1} / {(report.policy ?? data.policy).consecutive_windows} required for a retraining recommendation.</p>}
              <p className="mt-3 text-xs text-muted-foreground">{report.performance.labeled_samples} outcomes · {report.performance.positive_samples} positive / {report.performance.negative_samples} negative · Coverage {percent(report.performance.label_coverage)}.
                Alerts require {(report.policy ?? data.policy).min_labeled_samples} labels, {(report.policy ?? data.policy).min_class_samples} per class and {percent((report.policy ?? data.policy).min_label_coverage)} coverage.</p>
              <p className="mt-3 text-xs text-muted-foreground">{report.performance.baseline_note} · {report.performance.simulated ? "Simulated" : "Labeled"} churn rate {report.performance.churn_rate == null ? "N/A" : percent(report.performance.churn_rate)}.</p>
              <div className="mt-5 overflow-x-auto"><table className="w-full text-left text-sm"><thead className="border-b"><tr><th className="p-3">Metric</th><th className="p-3">Study baseline (different labels)</th>{synthetic && <th className="p-3">Normal synthetic baseline</th>}<th className="p-3">Window</th></tr></thead>
                <tbody className="divide-y">{([['ROC-AUC', 'ROC_AUC', 'roc_auc'], ['Average Precision', 'PR_AUC', 'average_precision'], ['Accuracy', 'Accuracy', 'accuracy'], ['Recall', 'Recall', 'recall'], ['Precision', 'Precision', 'precision'], ['F1', 'F1_Score', 'f1']] as const).map(([label, key, metric]) =>
                  <tr key={key}><td className="p-3">{label}</td><td className="p-3 tabular-nums">{number(report.performance.baseline[key])}</td>{synthetic && <td className="p-3 tabular-nums">{number(report.performance.synthetic_baseline?.metrics[metric])}</td>}<td className="p-3 tabular-nums">{number(report.performance.metrics?.[metric])}</td></tr>)}</tbody></table></div>
              {report.performance.metrics && <p className="mt-4 text-xs text-muted-foreground">Confusion matrix [[TN, FP], [FN, TP]]: {JSON.stringify(report.performance.metrics.confusion_matrix)} · Threshold {report.performance.threshold}.</p>}
              {chart.some((row) => row.roc_auc != null) && <div className="mt-6 h-64"><ResponsiveContainer width="100%" height="100%"><LineChart data={chart}>
                <CartesianGrid stroke="var(--color-border)" vertical={false} /><XAxis dataKey="window" tick={{ fontSize: 11 }} /><YAxis domain={[0, 1]} tick={{ fontSize: 11 }} /><Tooltip /><Legend />
                <Line dataKey="roc_auc" name="ROC-AUC" stroke="var(--color-chart-1)" connectNulls={false} /><Line dataKey="average_precision" name="Average Precision" stroke="var(--color-chart-2)" connectNulls={false} /><Line dataKey="recall" name="Recall" stroke="var(--color-chart-3)" connectNulls={false} />
              </LineChart></ResponsiveContainer></div>}
            </Panel>
            <div className="mt-6 grid gap-6 xl:grid-cols-2">
              <Panel title="Prediction distribution" subtitle="Prediction shift is a diagnostic signal; quality is measured separately with labels.">
                {report.prediction_drift ? <><State value={report.prediction_drift.status} /><Distribution data={report.prediction_drift.distribution} /></> : <p className="text-sm text-muted-foreground">Not enough scored samples.</p>}
              </Panel>
              <Panel title="Data quality and runtime" subtitle={report.quality.note ?? "Request errors include rejected inputs; model metrics deduplicate customer lookups."}>
                <dl className="grid grid-cols-2 gap-4 text-sm">{[
                  ["Requests", report.quality.request_count.toLocaleString()], ["Rejected / failed", report.quality.invalid_requests.toLocaleString()],
                  ["Error rate", percent(report.quality.error_rate)], ["Missingness above baseline", percent(report.quality.max_missing_excess)],
                  ["Inference p50", `${number(report.latency?.p50_ms, 1)} ms`], ["Inference p95", `${number(report.latency?.p95_ms, 1)} ms`],
                ].map(([label, value]) => <div key={label}><dt className="text-xs text-muted-foreground">{label}</dt><dd className="mt-1 font-semibold tabular-nums">{value}</dd></div>)}</dl>
              </Panel>
            </div>
            <Panel className="mt-6" title="Segment distribution" subtitle={report.segments?.note ?? "Available only with a matching frozen segmentation version."}>
              <State value={report.segments?.status ?? "unavailable"} /><p className="mt-3 text-xs text-muted-foreground">Coverage {percent(report.segments?.coverage)} · Segment version {report.segments?.model_id?.slice(0, 12) ?? "—"}.</p>
              {report.segments?.distribution && <Distribution data={report.segments.distribution} />}
            </Panel>
          </>}
        </>}
      </>}
  </>;
}
