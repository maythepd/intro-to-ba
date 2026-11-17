"""
Unified MLOps & API Serving Entrypoint for Digital Bank Analytics Platform.

Serves as the single runnable entrypoint for MLOps tasks:
  1. API Web Service: Run FastAPI serving for churn predictions & customer segmentation.
     Usage:
       python api.py                            # Starts both FastAPI (8000) & Frontend UI (5173)
       python api.py serve --no-frontend        # Starts API only
       python api.py serve --port 8000 --reload
       uvicorn api:app --reload

  2. Preprocessing & Artifact Preparation:
     Usage:
       python api.py prepare --download
       python api.py official-data --download
       python api.py segmentation --data data/official/customer_features.csv
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

# Expose FastAPI application instance for ASGI servers (uvicorn api:app)
# Dynamic import prevents IDE/Pylance name collision between api.py and api package
def _get_app():
    return importlib.import_module("api.main").app

try:
    app = _get_app()
except Exception:
    app = None

__all__ = ["app"]


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a process and all of its descendants cleanly on Windows and Unix."""
    if proc.poll() is not None:
        return
    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        pass


def _launch_frontend(port: int = 5173) -> subprocess.Popen | None:
    """Start Vite dev server concurrently in the background."""
    if not FRONTEND_DIR.is_dir():
        print("[WARN] frontend/ directory not found. Serving API only.")
        return None

    npm_bin = shutil.which("npm.cmd" if platform.system() == "Windows" else "npm")
    if not npm_bin:
        print("[WARN] npm executable not found in PATH. Skipping frontend startup.")
        return None

    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.is_dir():
        print("[INFO] Installing frontend dependencies (npm install)...")
        try:
            subprocess.run([npm_bin, "install"], cwd=str(FRONTEND_DIR), check=True)
        except Exception as exc:
            print(f"[WARN] Failed to install frontend dependencies: {exc}")
            return None

    print(f"[INFO] Launching Frontend (Vite) on http://localhost:{port}...")
    try:
        proc = subprocess.Popen(
            [npm_bin, "run", "dev", "--", "--port", str(port), "--host", "localhost"],
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        return proc
    except Exception as exc:
        print(f"[WARN] Failed to start frontend process: {exc}")
        return None


def _print_banner(api_host: str, api_port: int, frontend_port: int | None) -> None:
    display_host = "localhost" if api_host in ("0.0.0.0", "127.0.0.1") else api_host
    print("=" * 70)
    print("🚀  BaBank Customer Retention & Analytics Platform")
    print("=" * 70)
    if frontend_port:
        print(f"  💻 Frontend UI:    http://localhost:{frontend_port}")
    print(f"  ⚡ API Backend:    http://{display_host}:{api_port}")
    print(f"  📖 API Swagger:    http://{display_host}:{api_port}/docs")
    print(f"  🩺 Health Check:   http://{display_host}:{api_port}/health")
    print("-" * 70)
    print("  💡 Tip: On Windows browsers, navigate to http://localhost:8000")
    print("     (http://0.0.0.0:8000 is an internal bind address, not accessible).")
    print("  ⌨️  Press Ctrl+C to shut down both API and Frontend cleanly.")
    print("=" * 70 + "\n", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BaBank Analytics MLOps & API Serving CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="MLOps command to execute")

    # Command: serve (default)
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI web server & Frontend")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    serve_parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Backend port number")
    serve_parser.add_argument("--frontend-port", type=int, default=5173, help="Frontend UI port number")
    serve_parser.add_argument("--no-frontend", action="store_true", help="Do not start the frontend server")
    serve_parser.add_argument("--no-browser", action="store_true", help="Do not automatically pop up web browser")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # Command: prepare
    prep_parser = subparsers.add_parser("prepare", help="Reproduce Module 5 preprocessing & generate serving artifacts")
    prep_parser.add_argument("--download", action="store_true", help="Download official dataset from Kaggle")
    prep_parser.add_argument("--source", type=Path, default=None, help="Path to raw relational CSV tables directory")
    prep_parser.add_argument("--data", type=Path, default=None, help="Path to previously materialized customer table")

    # Command: official-data
    data_parser = subparsers.add_parser("official-data", help="Prepare model-ready customer features snapshot")
    data_parser.add_argument("--download", action="store_true", help="Download official Kaggle release")
    data_parser.add_argument("--source", type=Path, default=None, help="Path to raw relational tables")

    # Command: segmentation
    seg_parser = subparsers.add_parser("segmentation", help="Train and export segmentation serving models")
    seg_parser.add_argument("--data", type=Path, default=Path("data/official/customer_features.csv"), help="Features CSV path")
    seg_parser.add_argument("--output", type=Path, default=Path("artifacts/segmentation"), help="Export output directory")
    seg_parser.add_argument("--clusters", type=int, default=4, help="Number of clusters (k)")
    seg_parser.add_argument("--hierarchical-sample", type=int, default=3000, help="Ward subsample size")

    return parser


def _open_browser_when_ready(url: str, check_port: int, timeout: float = 12.0) -> None:
    """Poll for the port to become reachable, then automatically pop open the web browser."""
    import socket
    import threading
    import webbrowser

    def _worker():
        start_t = time.time()
        while time.time() - start_t < timeout:
            try:
                with socket.create_connection(("127.0.0.1", check_port), timeout=0.4):
                    time.sleep(0.3)
                    print(f"[INFO] 🌐 Opening browser UI at {url} ...\n", flush=True)
                    webbrowser.open(url)
                    return
            except (OSError, ConnectionRefusedError):
                time.sleep(0.3)
        try:
            print(f"[INFO] 🌐 Opening browser UI at {url} ...\n", flush=True)
            webbrowser.open(url)
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    frontend_port: int = 5173,
    start_frontend: bool = True,
    reload: bool = False,
    open_browser: bool = True,
) -> None:
    import uvicorn

    frontend_proc = None
    if start_frontend:
        frontend_proc = _launch_frontend(port=frontend_port)
        if frontend_proc:
            atexit.register(_terminate_process_tree, frontend_proc)

    def handle_signal(sig, frame):
        if frontend_proc:
            _terminate_process_tree(frontend_proc)
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except Exception:
        pass

    _print_banner(host, port, frontend_port if start_frontend else None)

    if open_browser:
        target_url = f"http://localhost:{frontend_port}" if start_frontend else f"http://localhost:{port}"
        check_port = frontend_port if start_frontend else port
        _open_browser_when_ready(target_url, check_port=check_port)

    try:
        uvicorn.run("api.main:app", host=host, port=port, reload=reload)
    finally:
        if frontend_proc:
            _terminate_process_tree(frontend_proc)


def main() -> None:
    # If invoked with no arguments, default to starting both API & Frontend
    if len(sys.argv) == 1:
        port = int(os.environ.get("PORT", 8000))
        run_server(host="0.0.0.0", port=port, frontend_port=5173, start_frontend=True, reload=False, open_browser=True)
        return

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command in (None, "serve"):
        host = getattr(args, "host", "0.0.0.0")
        port = getattr(args, "port", int(os.environ.get("PORT", 8000)))
        frontend_port = getattr(args, "frontend_port", 5173)
        start_frontend = not getattr(args, "no_frontend", False)
        reload = getattr(args, "reload", False)
        open_browser = not getattr(args, "no_browser", False)
        run_server(host=host, port=port, frontend_port=frontend_port, start_frontend=start_frontend, reload=reload, open_browser=open_browser)

    elif args.command == "prepare":
        from api.prepare_module5 import main as prep_main
        prep_main()

    elif args.command == "official-data":
        from api.official_data import main as data_main
        data_main()

    elif args.command == "segmentation":
        from api.segmentation_service import main as seg_main
        seg_main()


if __name__ == "__main__":
    main()
