# SPDX-License-Identifier: Apache-2.0
"""
Web UI routes for the OA2A proxy server dashboard.
"""

import json
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from local_openai2anthropic.config import Settings, get_config_file, load_config_from_file

web_router = APIRouter()

# --- In-memory request/error counters (per-day) ---
_stats_lock = Lock()
_stats = {
    "date": date.today().isoformat(),
    "total_requests": 0,
    "total_errors": 0,
}


def _reset_stats_if_new_day():
    """Reset counters if the day has changed."""
    today = date.today().isoformat()
    if _stats["date"] != today:
        _stats["date"] = today
        _stats["total_requests"] = 0
        _stats["total_errors"] = 0


def record_request(is_error: bool = False):
    """Record a request (and optionally an error)."""
    with _stats_lock:
        _reset_stats_if_new_day()
        _stats["total_requests"] += 1
        if is_error:
            _stats["total_errors"] += 1


def _get_current_stats() -> dict:
    """Get current stats."""
    with _stats_lock:
        _reset_stats_if_new_day()
        total = _stats["total_requests"]
        errors = _stats["total_errors"]
        error_rate = round(errors / total * 100, 1) if total > 0 else 0.0
        return {
            "today_requests": total,
            "request_change": 0,  # No historical data to compare
            "today_errors": errors,
            "error_rate": error_rate,
        }


# --- Server start time tracking ---
_server_started_at: float = time.time()


def get_server_started_at() -> float:
    """Get the server start timestamp.

    Tries daemon config first (for background mode), falls back to module-level.
    """
    try:
        from local_openai2anthropic.daemon import _load_daemon_config
        config = _load_daemon_config()
        if config and "started_at" in config:
            return config["started_at"]
    except Exception:
        pass
    return _server_started_at

# Get templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


def get_log_dir() -> Path:
    """Get the log directory path."""
    if sys.platform == 'win32':
        base_dir = os.environ.get('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
        return Path(base_dir) / 'local-openai2anthropic' / 'logs'
    else:
        return Path.home() / '.local' / 'share' / 'local-openai2anthropic' / 'logs'


def get_default_log_dir() -> str:
    """Get default log directory based on platform."""
    if sys.platform == 'win32':
        base_dir = os.environ.get('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
        return os.path.join(base_dir, 'local-openai2anthropic', 'logs')
    else:
        return os.path.expanduser("~/.local/share/local-openai2anthropic/logs")


# Page routes
@web_router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@web_router.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    """Configuration management page."""
    return templates.TemplateResponse("config.html", {"request": request})


@web_router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    """Log viewer page."""
    return templates.TemplateResponse("logs.html", {"request": request})


@web_router.get("/api-test", response_class=HTMLResponse)
async def api_test_page(request: Request):
    """API testing page."""
    return templates.TemplateResponse("api_test.html", {"request": request})


# API endpoints for the web UI
@web_router.get("/api/status")
async def get_server_status(request: Request):
    """Get server status information."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        from local_openai2anthropic.config import get_settings
        settings = get_settings()

    started_at = get_server_started_at()
    uptime = time.time() - started_at

    return {
        "running": True,
        "pid": os.getpid(),
        "uptime": uptime,
        "started_at": started_at,
        "host": settings.host,
        "port": settings.port,
    }


@web_router.get("/api/config")
async def get_config(request: Request):
    """Get current configuration."""
    config_file = get_config_file()
    config_data = load_config_from_file()

    # Mask API keys for security
    if config_data.get("openai_api_key"):
        config_data["openai_api_key"] = "***" + config_data["openai_api_key"][-4:] if len(config_data["openai_api_key"]) > 4 else "***"
    if config_data.get("api_key"):
        config_data["api_key"] = "***" + config_data["api_key"][-4:] if len(config_data["api_key"]) > 4 else "***"
    if config_data.get("tavily_api_key"):
        config_data["tavily_api_key"] = "***" + config_data["tavily_api_key"][-4:] if len(config_data["tavily_api_key"]) > 4 else "***"

    # Add config file info
    config_data["config_path"] = str(config_file)
    config_data["config_modified"] = config_file.stat().st_mtime if config_file.exists() else None

    return config_data


@web_router.put("/api/config")
async def update_config(request: Request):
    """Update configuration."""
    try:
        import tomli_w

        data = await request.json()
        config_file = get_config_file()
        config_dir = config_file.parent

        # Create directory if needed
        config_dir.mkdir(parents=True, exist_ok=True)

        # Build config dict
        toml_config = {
            "openai_api_key": data.get("openai_api_key", ""),
            "openai_base_url": data.get("openai_base_url", "https://api.openai.com/v1"),
            "openai_org_id": data.get("openai_org_id", ""),
            "openai_project_id": data.get("openai_project_id", ""),
            "host": data.get("host", "0.0.0.0"),
            "port": data.get("port", 8080),
            "request_timeout": data.get("request_timeout", 300.0),
            "api_key": data.get("api_key", ""),
            "cors_origins": data.get("cors_origins", ["*"]),
            "cors_credentials": data.get("cors_credentials", True),
            "cors_methods": data.get("cors_methods", ["*"]),
            "cors_headers": data.get("cors_headers", ["*"]),
            "log_level": data.get("log_level", "INFO"),
            "log_dir": data.get("log_dir", ""),
            "tavily_api_key": data.get("tavily_api_key", ""),
            "tavily_timeout": data.get("tavily_timeout", 30.0),
            "tavily_max_results": data.get("tavily_max_results", 5),
            "websearch_max_uses": data.get("websearch_max_uses", 5),
        }

        # Remove empty values
        toml_config = {k: v for k, v in toml_config.items() if v not in (None, "")}

        # Write config file
        with open(config_file, "wb") as f:
            tomli_w.dump(toml_config, f)

        # Set restrictive permissions on Unix
        if sys.platform != "win32":
            config_file.chmod(0o600)

        return {"success": True, "message": "配置已保存"}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


@web_router.get("/api/logs")
async def get_logs(request: Request, type: str = "server", lines: int = 100):
    """Get server logs."""
    log_dir = get_log_dir()
    log_file = log_dir / f"{type}.log"

    if not log_file.exists():
        return []

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        # Get last N lines
        all_lines = all_lines[-lines:]

        logs = []
        for i, line in enumerate(all_lines):
            line = line.strip()
            if not line:
                continue

            # Parse log line
            log_entry = {
                "id": i,
                "timestamp": "",
                "level": "INFO",
                "message": line
            }

            # Try to parse timestamp and level
            # Format: 2024-01-01 12:00:00,000 - name - LEVEL - message
            parts = line.split(" - ", 3)
            if len(parts) >= 4:
                log_entry["timestamp"] = parts[0]
                log_entry["level"] = parts[2].upper()
                log_entry["message"] = parts[3]
            elif len(parts) >= 3:
                log_entry["timestamp"] = parts[0]
                log_entry["message"] = parts[2]

            logs.append(log_entry)

        return logs

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@web_router.get("/api/stats")
async def get_stats(request: Request):
    """Get server statistics."""
    return _get_current_stats()


@web_router.post("/api/restart")
async def restart_server(request: Request):
    """Restart the server via daemon module."""
    try:
        from local_openai2anthropic.daemon import restart_daemon
        from local_openai2anthropic.config import get_settings

        settings = getattr(request.app.state, "settings", None)
        if settings is None:
            settings = get_settings()

        success = restart_daemon(
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
        )
        if success:
            return {"success": True, "message": "服务器正在重启"}
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "重启失败，请检查日志"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


@web_router.get("/api/config/file")
async def open_config_file(request: Request):
    """Get config file path."""
    config_file = get_config_file()
    return {"path": str(config_file), "exists": config_file.exists()}
