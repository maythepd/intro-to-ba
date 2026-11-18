"""Inspect or deploy only the two existing BaBank services; never print credentials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from dotenv import dotenv_values
import yaml

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://babank.onrender.com"
FRONTEND_URL = "https://babank-dashboard.onrender.com"
EXPECTED_REPOSITORY = "https://github.com/nthne/Intro-to-Business-Analysis"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Update commands and required env variables only")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--commit", help="Exact already-pushed demo commit to deploy")
    parser.add_argument("--enable-demo", action="store_true")
    args = parser.parse_args()
    if args.deploy and (not args.apply or not args.commit):
        parser.error("Deployment requires --apply and an explicit --commit")
    values = dotenv_values(ROOT / ".env")
    token = values.get("RENDER_API_KEY")
    if not token:
        raise SystemExit("Add RENDER_API_KEY to local .env; its value must not be sent to chat")
    headers = {"Authorization":"Bearer " + token, "Accept":"application/json", "User-Agent":"BaBank-module7-setup"}

    def request(path, *, method="GET", payload=None, missing_ok=False):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request("https://api.render.com/v1" + path, data=data, headers=headers, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            if missing_ok and exc.code == 404:
                return None
            raise SystemExit(f"Render operation returned HTTP {exc.code}; credentials omitted") from None

    services = []
    cursor = None
    while True:
        query = urllib.parse.urlencode({"limit":100, **({"cursor":cursor} if cursor else {})})
        page = request("/services?" + query)
        services.extend(item["service"] for item in page)
        if len(page) < 100:
            break
        cursor = page[-1]["cursor"]
    targets = []
    for url in (API_URL, FRONTEND_URL):
        matches = [service for service in services if service.get("serviceDetails", {}).get("url", "").rstrip("/") == url]
        if len(matches) != 1:
            raise SystemExit("Could not uniquely identify the requested BaBank service: " + url)
        targets.append(matches[0])
    print(json.dumps([{"id":service["id"],"type":service["type"],"repo":service.get("repo"),
        "branch":service.get("branch"),"rootDir":service.get("rootDir"),"url":service["serviceDetails"]["url"],
        "command_layout":"envSpecificDetails" if "envSpecificDetails" in service["serviceDetails"] else "direct",
        "build_uses_module5":"api.prepare_module5" in service["serviceDetails"].get("envSpecificDetails",{}).get("buildCommand", ""),
        "start_runs_migrations":"monitoring.migrate" in service["serviceDetails"].get("envSpecificDetails",{}).get("startCommand", "")}
        for service in targets]))
    if not args.apply:
        return
    for service in targets:
        if (service.get("repo", "").removesuffix(".git").lower() != EXPECTED_REPOSITORY.lower()
                or service.get("branch") != "demo"):
            raise SystemExit("Review required: the service is not linked to the approved fork demo branch; no configuration changed")
    if not values.get("DATABASE_URL") or not values.get("MONITORING_ADMIN_TOKEN"):
        raise SystemExit("DATABASE_URL and MONITORING_ADMIN_TOKEN are required")
    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))["services"][0]
    backend, frontend = targets
    if backend.get("rootDir") not in (None, "", "./"):
        raise SystemExit("Review required: API rootDir is not repository root")
    if frontend.get("rootDir") not in (None, "", "./", "frontend", "frontend/"):
        raise SystemExit("Review required: unexpected frontend rootDir; no configuration changed")
    commands = {"buildCommand":blueprint["buildCommand"], "startCommand":blueprint["startCommand"]}
    request("/services/" + backend["id"], method="PATCH", payload={"serviceDetails":{"envSpecificDetails":commands}})
    def set_env(service, key, value):
        request(f"/services/{service['id']}/env-vars/{urllib.parse.quote(key, safe='')}", method="PUT", payload={"value":value})
    env = {"DATABASE_URL":values["DATABASE_URL"], "MONITORING_ADMIN_TOKEN":values["MONITORING_ADMIN_TOKEN"],
        "MONITORING_DATABASE_ROLE":"bank_monitoring_writer", "ENABLE_MONITORING_DEMO":"true" if args.enable_demo else "false",
        "SUPABASE_URL":values.get("SUPABASE_URL", ""), "CUSTOMERS_CSV_PATH":"data/official/module5_customer_features.csv",
        "SEGMENTATION_CUSTOMERS_CSV_PATH":"data/official/customer_features.csv"}
    for key, value in env.items():
        set_env(backend, key, value)
    cors = request(f"/services/{backend['id']}/env-vars/CORS_ORIGINS", missing_ok=True) or {}
    origins = [value.strip() for value in cors.get("envVar", cors).get("value", "").split(",") if value.strip()]
    if FRONTEND_URL not in origins:
        set_env(backend, "CORS_ORIGINS", ",".join(origins + [FRONTEND_URL]))
    if frontend.get("rootDir") in ("frontend", "frontend/"):
        build, publish = "npm install && npm run build", "dist"
    elif frontend.get("rootDir") in (None, "", "./"):
        build, publish = "cd frontend && npm install && npm run build", "frontend/dist"
    else:
        raise SystemExit("Review required: unexpected frontend rootDir")
    request("/services/" + frontend["id"], method="PATCH", payload={"serviceDetails":{"buildCommand":build, "publishPath":publish}})
    set_env(frontend, "VITE_API_BASE_URL", API_URL)
    output = {"updated_services":[service["id"] for service in targets], "secret_values":"not displayed"}
    if args.deploy:
        output["deploys"] = [{"service_id":service["id"], "deploy_id":request(f"/services/{service['id']}/deploys",
            method="POST", payload={"clearCache":"do_not_clear", "commitId":args.commit})["id"]} for service in targets]
    print(json.dumps(output))


if __name__ == "__main__":
    main()
