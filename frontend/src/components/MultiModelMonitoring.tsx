import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ErrorState, Notice, PageHeader, Panel } from "@/components/common";
import { api, API_BASE_URL, describeError, isApiConfigured, type DriftFeature } from "@/lib/api";

const batches = ["normal_01", "normal_02", "normal_03", "engagement_shift", "friction_shift"];
export const monitoringModelNames = {xgboost: "XGBoost · champion", lr: "Logistic Regression · challenger", rf: "Random Forest · challenger",
  kmeans: "K-Means", hierarchical: "Hierarchical (Ward + nearest-centroid surrogate)"} as const;
export type MonitoringModel = "all" | keyof typeof monitoringModelNames;
const names: Record<string, string> = monitoringModelNames;
const colors = ["var(--color-chart-1)", "var(--color-chart-2)", "var(--color-chart-3)", "var(--color-chart-4)"];
const control = "h-10 w-full min-w-0 truncate rounded-xl border bg-background px-3 text-sm";
const num = (value: number | null | undefined) => value == null ? "N/A" : value.toFixed(3);
const pct = (value: number | null | undefined) => value == null ? "N/A" : `${(value * 100).toFixed(1)}%`;

function Histogram({feature}: {feature?: DriftFeature}) {
  if (!feature) return <p className="text-sm text-muted-foreground">N/A — no eligible distribution.</p>;
  return <div className="h-64 min-w-0" role="img" aria-label={`${feature.name} reference and current distribution`}><ResponsiveContainer width="100%" height="100%">
    <BarChart data={feature.distribution}><CartesianGrid vertical={false} /><XAxis dataKey="bin" tick={{fontSize: 9}} /><YAxis tickFormatter={(v) => pct(v)} /><Tooltip /><Legend />
      <Bar dataKey="reference" fill={colors[1]} /><Bar dataKey="current" fill={colors[0]} />
    </BarChart></ResponsiveContainer></div>;
}

export function MultiModelMonitoring({model = "all"}: {model?: MonitoringModel}) {
  const [campaignId, setCampaignId] = useState("");
  const [batch, setBatch] = useState("all");
  const [tab, setTab] = useState("classification");
  const [metric, setMetric] = useState("roc_auc");
  const [clusterMetric, setClusterMetric] = useState("silhouette");
  const [space, setSpace] = useState("classification");
  const [featureName, setFeatureName] = useState("");
  const [profileAlgorithm, setProfileAlgorithm] = useState("kmeans");
  const [profileCluster, setProfileCluster] = useState(0);
  const selectedTask = model === "all" ? null : ["xgboost", "lr", "rf"].includes(model) ? "classification" : "clustering";
  const activeTab = tab === "inputs" ? tab : selectedTask ?? tab;
  const activeSpace = selectedTask ?? space;
  const classifierAlgorithms = ["xgboost", "lr", "rf"].filter((name) => model === "all" || model === name);
  const clusterAlgorithms = ["kmeans", "hierarchical"].filter((name) => model === "all" || model === name);
  const activeProfileAlgorithm = selectedTask === "clustering" ? model : profileAlgorithm;
  const catalog = useQuery({queryKey: ["campaigns"], queryFn: ({signal}) => api.campaigns(signal), enabled: isApiConfigured, refetchInterval: 30000});
  const id = campaignId || catalog.data?.items.find((item) => item.status === "complete")?.id || catalog.data?.items[0]?.id || "";
  const detail = useQuery({queryKey: ["campaign", id], queryFn: ({signal}) => api.campaign(id, signal), enabled: !!id && isApiConfigured, refetchInterval: 30000});
  const inputs = useQuery({queryKey: ["campaign-inputs", id], queryFn: ({signal}) => api.campaignInputs(id, signal), enabled: !!id && isApiConfigured});
  const data = detail.data;
  const all = data?.reports ?? [];
  const visible = all.filter((item) => model === "all" || item.algorithm === model);
  const rows = visible.filter((item) => batch === "all" || item.batch_id === batch);
  const classifiers = rows.filter((item) => item.task === "classification");
  const clusterers = rows.filter((item) => item.task === "clustering");
  const selectedBatch = batch === "all" ? batches[0] : batch;
  const input = inputs.data?.items.find((item) => item.batch_id === selectedBatch && item.report.space === activeSpace);
  const features = input?.report.features ?? [];
  const feature = features.find((item) => item.name === featureName) ?? features[0];
  const lineData = (task: string, key: string) => batches.map((name) => ({batch: name, ...Object.fromEntries(visible.filter((item) => item.task === task && item.batch_id === name)
    .map((item) => [item.algorithm, task === "classification" ? (item.report.performance?.metrics as unknown as Record<string, number | null> | null)?.[key] ?? null
      : (item.report as unknown as Record<string, number | null>)[key] ?? null]))}));
  const exportUrl = (format: string, task = "classification") => `${API_BASE_URL}/api/monitoring/campaigns/${encodeURIComponent(id)}/export?format=${format}&task=${task}`;
  const profileReport = all.find((item) => item.algorithm === activeProfileAlgorithm && item.batch_id === selectedBatch)?.report;
  const activeProfileCluster = profileReport?.profiles?.some((item) => item.cluster_id === profileCluster)
    ? profileCluster : profileReport?.profiles?.[0]?.cluster_id ?? 0;
  const profile = profileReport?.profiles?.find((item) => item.cluster_id === activeProfileCluster);

  return <>
    <PageHeader eyebrow="Monitoring" title={model === "all" ? "Multi-model study replay" : `${names[model]} monitoring`} description="Measured input drift and model-specific performance on the same synthetic customer IDs. XGBoost remains the serving champion." />
    {!isApiConfigured ? <Notice title="API not connected">Configure the backend to read campaign reports.</Notice>
      : catalog.isError ? <ErrorState message={describeError(catalog.error)} onRetry={() => catalog.refetch()} />
      : catalog.isPending ? <Panel>Loading campaigns…</Panel>
      : !id ? <Panel title="No campaign available">An operator must prepare approved artifacts and run the authenticated multi-model replay. Metrics remain unavailable until measured.</Panel>
      : <>
        <div className="mb-6 grid min-w-0 grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <label className="grid min-w-0 gap-2 text-xs font-medium sm:col-span-2">Campaign<select className={control} value={id} onChange={(e) => {setCampaignId(e.target.value); setFeatureName("");}}>
            {catalog.data.items.map((item) => <option key={item.id} value={item.id}>{item.id.slice(0, 12)} · {item.mode === "simulated_labels" ? "Simulated labels" : "Drift only"} · {item.status}</option>)}
          </select></label>
          <label className="grid min-w-0 gap-2 text-xs font-medium">Batch<select className={control} value={batch} onChange={(e) => {setBatch(e.target.value); setFeatureName("");}}>
            <option value="all">All five batches</option>{batches.map((name, index) => <option key={name} value={name}>{index + 1}/5 · {name}</option>)}
          </select></label>
          <div className="flex flex-wrap items-end gap-3 text-xs" title="Downloads include the full campaign for the selected task."><a className="underline" href={exportUrl("json")}>Campaign JSON</a><a className="underline" href={exportUrl("html")}>Campaign HTML</a>
            <a className="underline" href={exportUrl("csv", activeTab === "clustering" || activeTab === "inputs" && activeSpace === "clustering" ? "clustering" : "classification")}>Campaign CSV</a></div>
        </div>
        {detail.isError ? <ErrorState message={describeError(detail.error)} onRetry={() => detail.refetch()} /> : !data ? <Panel>Loading campaign measurements…</Panel> : <>
          <Notice tone="warning" title={data.manifest.mode === "simulated_labels" ? "Independent simulated outcomes" : "Drift-only campaign"}>
            {data.manifest.mode === "simulated_labels" ? `${data.manifest.label_policy_version} · seed ${data.manifest.label_seed}. Classification metrics describe this illustrative policy, not real-customer accuracy.` : "No outcomes supplied. Classification performance is N/A; clustering quality can still be measured."}
            {" "}The three normal batches build each model's baseline. Stress reports use that fixed baseline; no model is automatically promoted or retrained.
          </Notice>
          {data.status !== "complete" && <div className="mt-4"><Notice tone="warning" title="Campaign incomplete">{data.report_count}/25 reports available. Missing units are pending, not zero. Comparison is not a complete ranking.</Notice></div>}
          <div className="my-6 grid min-w-0 grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {[["Selected model", model === "all" ? "All five models" : names[model]], ["Reports", `${model === "all" ? data.report_count : visible.length}/${model === "all" ? 25 : 5} · ${data.status}`], ["Customers", `${data.manifest.batches.reduce((n, b) => n + b.N, 0).toLocaleString()} unique IDs`],
              ["Feature snapshot / time", "2026-10-01 / simulated UTC"]].map(([label, value]) => <div key={label} className="min-w-0 rounded-2xl border bg-card p-5"><p className="text-xs text-muted-foreground">{label}</p><p className="mt-2 break-words text-sm font-semibold">{value}</p></div>)}
          </div>
          <p className="mb-5 break-all text-xs text-muted-foreground">Dataset SHA256 {data.manifest.dataset_sha256} · Anchor {data.manifest.anchor}.</p>
          {model !== "all" && <p className="mb-5 break-all text-xs text-muted-foreground">Model ID {data.manifest.models[model]?.model_id ?? "N/A"} · Input drift uses the shared {selectedTask} reference; output drift and performance belong to this model.</p>}
          <div className="mb-5 flex flex-wrap gap-2" role="tablist">{[["classification", "Classification monitoring"], ["clustering", "Clustering monitoring"], ["inputs", "Input data drift"]].map(([key, name]) =>
            (key === "inputs" || selectedTask === null || key === selectedTask) && <button key={key} role="tab" aria-selected={activeTab === key} className={`rounded-xl border px-4 py-2 text-sm ${activeTab === key ? "bg-primary text-primary-foreground" : "bg-card"}`} onClick={() => setTab(key)}>{name}</button>)}</div>
          {activeTab === "classification" && <><Panel title={model === "all" ? "Three classifiers, identical cohorts" : `${names[model]} performance`} subtitle="ROC-AUC, AP, F1 and recall: higher is better. Log loss/Brier: lower is better. AP must be read together with prevalence.">
            <label className="grid max-w-sm gap-2 text-xs">Chart metric<select className={control} value={metric} onChange={(e) => setMetric(e.target.value)}>{["roc_auc", "average_precision", "f1", "recall", "accuracy", "balanced_accuracy", "log_loss", "brier"].map((key) => <option key={key}>{key}</option>)}</select></label>
            <div className="mt-5 h-72 min-w-0"><ResponsiveContainer width="100%" height="100%"><LineChart data={lineData("classification", metric)}><CartesianGrid /><XAxis dataKey="batch" tick={{fontSize: 10}} /><YAxis /><Tooltip /><Legend />
              {classifierAlgorithms.map((name) => <Line key={name} dataKey={name} name={names[name]} stroke={colors[["xgboost", "lr", "rf"].indexOf(name)]} connectNulls={false} />)}</LineChart></ResponsiveContainer></div>
            <div className="mt-5 overflow-x-auto"><table className="w-full min-w-[1300px] text-left text-xs"><thead className="border-b"><tr>{["Algorithm / role", "Batch", "N", "AUC", "AP", "Accuracy", "Precision", "Recall", "F1", "Churn rate", "Predicted +", "Normal AP", "AP drop", "Confusion [[TN,FP],[FN,TP]]", "Labels"].map((key) => <th className="p-2" key={key}>{key}</th>)}</tr></thead>
              <tbody>{classifiers.map((item) => {const p = item.report.performance; return <tr className="border-b" key={item.algorithm + item.batch_id}><td className="p-2" title={item.model_id}>{names[item.algorithm]}</td><td className="p-2">{item.batch_id}</td><td className="p-2">{item.report.sample_count}</td>
                {(["roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"] as const).map((key) => <td className="p-2 tabular-nums" key={key}>{num(p?.metrics?.[key])}</td>)}
                <td className="p-2">{pct(p?.churn_rate)}</td><td className="p-2">{pct(p?.metrics?.predicted_positive_rate)}{p?.metrics?.predicted_positive_rate === 1 ? " · all positive" : ""}</td>
                <td className="p-2">{num(p?.synthetic_baseline?.metrics.average_precision)}</td><td className="p-2">{num(p?.drops.average_precision)}</td><td className="p-2">{p?.metrics ? JSON.stringify(p.metrics.confusion_matrix) : "N/A"}</td><td className="p-2">{p?.labeled_samples ?? 0} · {p?.status}</td></tr>;})}</tbody></table></div>
          </Panel><div className={`mt-6 grid min-w-0 gap-6 ${model === "all" ? "xl:grid-cols-3" : ""}`}>{classifierAlgorithms.map((algorithm) => {
            const item = all.find((row) => row.algorithm === algorithm && row.batch_id === selectedBatch);
            return <Panel key={algorithm} title={`${names[algorithm]} probabilities`} subtitle={`${selectedBatch} · original train reference · PSI ${num(item?.report.prediction_drift?.score)}`}>
              <Histogram feature={item?.report.prediction_drift ?? undefined} />
            </Panel>;
          })}</div></>}
          {activeTab === "clustering" && <>
            <Panel title="Fixed segmentation quality" subtitle={`Same encoded space and deterministic sample (up to ${data.manifest.config.silhouette_sample_size}) for intrinsic metrics. Distances use all customers. Cluster IDs are specific to each algorithm.`}>
              <label className="grid max-w-sm gap-2 text-xs">Chart metric<select className={control} value={clusterMetric} onChange={(e) => setClusterMetric(e.target.value)}>{["silhouette", "dbi", "chi", "p95_distance", "outlier_share"].map((key) => <option key={key}>{key}</option>)}</select></label>
              <div className="mt-5 h-72 min-w-0"><ResponsiveContainer width="100%" height="100%"><LineChart data={lineData("clustering", clusterMetric)}><CartesianGrid /><XAxis dataKey="batch" tick={{fontSize: 10}} /><YAxis /><Tooltip /><Legend />{clusterAlgorithms.map((name) => <Line key={name} dataKey={name} name={names[name]} stroke={colors[["kmeans", "hierarchical"].indexOf(name)]} connectNulls={false} />)}</LineChart></ResponsiveContainer></div>
              <p className="mt-3 text-xs text-muted-foreground">Silhouette/CHI ↑; DBI and fixed distance ↓. Normal baseline is the median of three equal-size normal cohorts. {data.manifest.config.threshold_note}</p>
              <div className="mt-5 overflow-x-auto"><table className="w-full min-w-[1100px] text-left text-xs"><thead className="border-b"><tr>{["Algorithm", "Batch", "N / eval N", "Silhouette", "DBI", "CHI", "p95 distance", "Outliers", "Share JS", "Silhouette drop", "Review"].map((key) => <th className="p-2" key={key}>{key}</th>)}</tr></thead><tbody>{clusterers.map((item) => <tr className="border-b" key={item.algorithm + item.batch_id}><td className="p-2" title={item.model_id}>{names[item.algorithm]}</td><td className="p-2">{item.batch_id}</td><td className="p-2">{item.report.sample_count} / {item.report.eval_N}</td>{(["silhouette", "dbi", "chi", "p95_distance"] as const).map((key) => <td className="p-2" key={key}>{num(item.report[key])}</td>)}<td className="p-2">{pct(item.report.outlier_share)}</td><td className="p-2">{num(item.report.assignment_drift?.score)}</td><td className="p-2">{num(item.report.drops?.silhouette)}</td><td className="p-2">{item.report.review_status ?? item.report.status}</td></tr>)}</tbody></table></div>
            </Panel>
            <div className={`mt-6 grid min-w-0 gap-6 ${model === "all" ? "xl:grid-cols-2" : ""}`}>{clusterAlgorithms.map((algorithm) => {
              const source = all.filter((item) => item.algorithm === algorithm);
              const bars = batches.map((name) => {const report = source.find((item) => item.batch_id === name)?.report; return {batch: name, ...Object.fromEntries((report?.cluster_shares ?? []).map((value, index) => ["cluster_" + index, value]))};});
              const count = source[0]?.report.k ?? 0;
              return <Panel key={algorithm} title={`${names[algorithm]} cluster shares`} subtitle="Population composition, not individual customer migration."><div className="h-72 min-w-0"><ResponsiveContainer width="100%" height="100%"><BarChart data={bars}><XAxis dataKey="batch" tick={{fontSize: 9}} /><YAxis tickFormatter={(value) => pct(value)} /><Tooltip /><Legend />{Array.from({length: count}, (_, index) => <Bar key={index} dataKey={`cluster_${index}`} stackId="shares" fill={colors[index % colors.length]} />)}</BarChart></ResponsiveContainer></div></Panel>;
            })}</div>
            <Panel className="mt-6" title={`Cluster profile drift · ${selectedBatch}`} subtitle="Each fixed cluster is compared with its own reference. Small groups stay insufficient; no churn accuracy is assigned to clusters.">
              <div className="grid gap-4 sm:grid-cols-2"><select className={control} aria-label="Profile algorithm" value={activeProfileAlgorithm} disabled={selectedTask === "clustering"} onChange={(e) => setProfileAlgorithm(e.target.value)}>{clusterAlgorithms.map((name) => <option key={name} value={name}>{names[name]}</option>)}</select>
                <select className={control} aria-label="Profile cluster" value={activeProfileCluster} onChange={(e) => setProfileCluster(Number(e.target.value))}>{(profileReport?.profiles ?? []).map((item) => <option key={item.cluster_id} value={item.cluster_id}>Cluster {item.cluster_id} · N={item.sample_count} · {item.status}</option>)}</select></div>
              <div className="mt-4 overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr><th className="p-2">Feature</th><th className="p-2">PSI / JS distance</th><th className="p-2">Missing</th><th className="p-2">State</th></tr></thead><tbody>{profile?.features.map((item) => <tr key={item.name}><td className="p-2">{item.name}</td><td className="p-2">{num(item.score)}</td><td className="p-2">{pct(item.missing_rate)}</td><td className="p-2">{item.status}</td></tr>)}</tbody></table>{!profile?.features.length && <p className="mt-3 text-sm">N/A — insufficient cluster profile samples.</p>}</div>
            </Panel>
          </>}
          {activeTab === "inputs" && <Panel title={`Shared input drift · ${selectedBatch}`} subtitle="One report per feature space/reference and batch. Classification uses the original train reference before SMOTE; segmentation uses its separate frozen snapshot.">
            {inputs.isError ? <ErrorState message={describeError(inputs.error)} onRetry={() => inputs.refetch()} /> : <><div className="grid gap-4 sm:grid-cols-2"><select className={control} aria-label="Feature space" value={activeSpace} disabled={selectedTask !== null} onChange={(e) => {setSpace(e.target.value); setFeatureName("");}}><option value="classification">Classifier domain profiles</option><option value="clustering">Segmentation inputs</option></select><select className={control} aria-label="Drift feature" value={feature?.name ?? ""} onChange={(e) => setFeatureName(e.target.value)}>{features.map((item) => <option key={item.name}>{item.name}</option>)}</select></div>
              <p className="mt-4 break-all text-xs text-muted-foreground">Reference {input?.reference_sha256 ?? "N/A"} · N={input?.report.sample_count ?? 0} · Critical share {pct(input?.report.critical_share)} · {input?.report.status}</p>
              <Histogram feature={feature} /><div className="mt-5 overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr>{["Feature", "Method", "Score", "Missing", "New categories", "State"].map((key) => <th className="p-2" key={key}>{key}</th>)}</tr></thead><tbody>{features.map((item) => <tr key={item.name}><td className="p-2"><button className="underline" onClick={() => setFeatureName(item.name)}>{item.name}</button></td><td className="p-2">{item.method}</td><td className="p-2">{num(item.score)}</td><td className="p-2">{pct(item.missing_rate)}</td><td className="p-2">{pct(item.new_category_rate)}</td><td className="p-2">{item.status}</td></tr>)}</tbody></table></div></>}
          </Panel>}
        </>}
      </>}
  </>;
}
