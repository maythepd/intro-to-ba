"""Install the API-only scheduler on the fork's main branch and encrypt its token."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request

from dotenv import dotenv_values
from nacl import encoding, public

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="nthne/Intro-to-Business-Analysis")
    parser.add_argument("--api-url", default="https://babank.onrender.com")
    parser.add_argument("--install-workflow-on-main", action="store_true")
    args = parser.parse_args()
    if not args.api_url.startswith("https://"):
        parser.error("Use the deployed HTTPS API URL")
    values = dotenv_values(ROOT / ".env")
    monitoring_token = values.get("MONITORING_ADMIN_TOKEN")
    if not monitoring_token:
        raise SystemExit("Configure MONITORING_ADMIN_TOKEN in .env first")
    credential = subprocess.run(["git", "credential", "fill"], cwd=ROOT,
        input=f"protocol=https\nhost=github.com\npath={args.repo}.git\n\n", text=True, capture_output=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT":"0", "GCM_INTERACTIVE":"Never"}, timeout=30)
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    secret = fields.get("password")
    if credential.returncode or not secret:
        raise SystemExit("Existing GitHub Git authentication is unavailable")
    headers = {"Authorization":"Bearer " + secret, "Accept":"application/vnd.github+json",
               "X-GitHub-Api-Version":"2022-11-28", "User-Agent":"BaBank-monitoring-setup"}

    def request(path, *, method="GET", payload=None, missing_ok=False):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request("https://api.github.com/repos/" + args.repo + path,
                                     data=data, headers=headers, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            if missing_ok and exc.code == 404:
                return None
            raise RuntimeError(f"GitHub operation failed: HTTP {exc.code}; credentials omitted") from None

    repository = request("")
    if args.install_workflow_on_main and repository["default_branch"] != "main":
        raise SystemExit("The requested main branch is not this repository's default branch")
    key = request("/actions/secrets/public-key")
    box = public.SealedBox(public.PublicKey(key["key"].encode(), encoding.Base64Encoder()))
    encrypted = base64.b64encode(box.encrypt(monitoring_token.encode())).decode()
    request("/actions/secrets/MONITORING_ADMIN_TOKEN", method="PUT", payload={"encrypted_value":encrypted, "key_id":key["key_id"]})
    existing = request("/actions/variables/MONITORING_API_URL", missing_ok=True)
    request("/actions/variables/MONITORING_API_URL" if existing else "/actions/variables",
            method="PATCH" if existing else "POST", payload={"name":"MONITORING_API_URL", "value":args.api_url.rstrip("/")})
    commit = None
    if args.install_workflow_on_main:
        path = "/contents/.github/workflows/monitoring-run.yml"
        current = request(path + "?ref=main", missing_ok=True)
        content = (ROOT / ".github/workflows/monitoring-run.yml").read_bytes()
        if current is None or base64.b64decode(current["content"]) != content:
            payload = {"message":"Schedule deployed model monitoring daily", "branch":"main", "content":base64.b64encode(content).decode()}
            if current:
                payload["sha"] = current["sha"]
            commit = request(path, method="PUT", payload=payload)["commit"]["sha"]
        workflow = request("/actions/workflows/monitoring-run.yml")
        if workflow["state"] != "active":
            request("/actions/workflows/monitoring-run.yml/enable", method="PUT")
    print(json.dumps({"repo":args.repo, "secret_configured":"MONITORING_ADMIN_TOKEN", "api_url":args.api_url,
                      "workflow_main_commit":commit, "workflow_installed":args.install_workflow_on_main}))


if __name__ == "__main__":
    main()
