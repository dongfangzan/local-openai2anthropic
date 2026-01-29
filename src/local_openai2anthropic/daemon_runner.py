#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Daemon runner - this module is executed as a standalone script in the child process.
"""

import atexit
import os
import signal
import sys
from pathlib import Path
from datetime import datetime

PID_FILE = Path.home() / ".local" / "share" / "oa2a" / "oa2a.pid"


def log_message(msg: str) -> None:
    """Write message to both stdout and parent process communication"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, file=sys.stderr)
    sys.stderr.flush()


def _write_pid(pid: int) -> None:
    """Write PID to pidfile."""
    try:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(pid))
        log_message(f"PID written to {PID_FILE}: {pid}")
    except Exception as e:
        log_message(f"Failed to write PID file: {e}")


def _remove_pid() -> None:
    """Remove pidfile."""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
            log_message(f"PID file removed: {PID_FILE}")
    except OSError:
        pass


def _signal_handler(signum, frame):
    """Handle termination signals."""
    sig_name = signal.Signals(signum).name
    log_message(f"Received signal {sig_name}, shutting down...")
    _remove_pid()
    sys.exit(0)


def run_server():
    """Run the server in daemon mode."""
    try:
        # Write current PID to file (this is the correct PID for the daemon)
        current_pid = os.getpid()
        _write_pid(current_pid)

        # Register cleanup on exit
        atexit.register(_remove_pid)

        # Setup signal handlers
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        log_message(f"Starting daemon server (PID: {current_pid})...")

        # Add the src directory to path if needed
        current_file = Path(__file__).resolve()
        package_dir = current_file.parent

        if str(package_dir) not in sys.path:
            sys.path.insert(0, str(package_dir))

        # Import and run the main server
        from local_openai2anthropic.main import create_app
        from local_openai2anthropic.config import get_settings

        import uvicorn

        settings = get_settings()

        log_message(f"Configuration loaded:")
        log_message(f"  Host: {settings.host}")
        log_message(f"  Port: {settings.port}")
        log_message(f"  Log Level: {settings.log_level}")
        log_message(f"  OpenAI Base URL: {settings.openai_base_url}")

        # Validate required settings
        if not settings.openai_api_key:
            log_message("Error: OA2A_OPENAI_API_KEY is required but not set")
            sys.exit(1)

        app = create_app(settings)

        log_message(f"Starting uvicorn on {settings.host}:{settings.port}")

        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
            timeout_keep_alive=300,
        )

    except Exception as e:
        log_message(f"Fatal error in daemon: {e}")
        import traceback
        traceback.print_exc()
        _remove_pid()
        sys.exit(1)


if __name__ == "__main__":
    run_server()
