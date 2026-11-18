from html import escape
import json


def render_html(report: dict) -> str:
    def cell(value):
        return escape(str(value))
    rows = "".join("<tr>" + "".join(f"<td>{cell(value)}</td>" for value in
        (feature["name"], feature["method"], f"{feature['score']:.5f}" if feature["score"] is not None else "N/A",
         feature["status"], f"{feature['missing_rate']:.1%}")) + "</tr>" for feature in report["drift"]["features"])
    data = escape(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>BaBank monitoring report</title><style>body{{font:15px system-ui;margin:40px auto;max-width:1100px;padding:0 20px;color:#222}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:10px;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f2fa;padding:20px;border-radius:12px}}h1,h2{{color:#604395}}</style>
<h1>BaBank model monitoring</h1><p><strong>{cell(report['data_kind'])}</strong> · {cell(report['source'])} / {cell(report['scenario'] or 'default')}</p>
<p>Window: {cell(report['window_start'])} → {cell(report['window_end'])}<br>Computed: {cell(report['computed_at'])}<br>
Model version: {cell(report['model_id'])}<br>Status: <strong>{cell(report['status'])}</strong> · Samples: {report['sample_count']}</p>
<h2>Feature drift</h2><table><thead><tr><th>Feature</th><th>Method</th><th>Score</th><th>Status</th><th>Missing</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Full aggregate report and provenance</h2><pre>{data}</pre></html>'''
