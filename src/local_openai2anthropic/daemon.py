# SPDX-License-Identifier: Apache-2.0
"""
Daemon process management for local-openai2anthropic server.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Constants
DATA_DIR = Path.home() / ".local" / "share" / "oa2a"
PID_FILE = DATA_DIR / "oa2a.pid"
CONFIG_FILE = DATA_DIR / "oa2a.json"
LOG_FILE = DATA_DIR / "oa2a.log"


def _ensure_dirs() -> None:
    """Ensure pid/log directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_pid() -> Optional[int]:
    """Read PID from pidfile."""
    try:
        if PID_FILE.exists():
            return int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        pass
    return None


def _remove_pid() -> None:
    """Remove pidfile."""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except OSError:
        pass


def _save_daemon_config(host: str, port: int) -> None:
    """Save daemon configuration to file."""
    _ensure_dirs()
    config = {
        "host": host,
        "port": port,
        "started_at": time.time(),
    }
    try:
        CONFIG_FILE.write_text(json.dumps(config))
    except OSError:
        pass


def _load_daemon_config() -> Optional[dict]:
    """Load daemon configuration from file."""
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _remove_daemon_config() -> None:
    """Remove daemon configuration file."""
    try:
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
    except OSError:
        pass


def _is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_port() -> Optional[int]:
    """Read port from daemon config file."""
    config = _load_daemon_config()
    if config and "port" in config:
        return config["port"]
    return None


def _find_pid_by_port(port: int) -> Optional[int]:
    """Find PID of process listening on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Return the first PID found
            return int(result.stdout.strip().split()[0])
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        pass
    return None


def _is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is already in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


def _cleanup_stale_pidfile() -> None:
    """Remove pidfile if the process is not running."""
    pid = _read_pid()
    if pid is not None and not _is_process_running(pid):
        _remove_pid()
        _remove_daemon_config()


def get_status() -> tuple[bool, Optional[int], Optional[dict]]:
    """
    Get daemon status.

    Returns:
        Tuple of (is_running, pid, config)
    """
    _cleanup_stale_pidfile()
    pid = _read_pid()
    config = _load_daemon_config()
    if pid is not None and _is_process_running(pid):
        return True, pid, config
    return False, None, None


def start_daemon(
    host: str = "0.0.0.0",
    port: int = 8080,
    log_level: str = "info",
) -> bool:
    """
    Start the server as a background daemon.

    Args:
        host: Server host
        port: Server port
        log_level: Logging level

    Returns:
        True if started successfully, False otherwise
    """
    _cleanup_stale_pidfile()

    pid = _read_pid()
    if pid is not None:
        config = _load_daemon_config()
        actual_port = config.get("port", port) if config else port
        print(f"Server is already running (PID: {pid}, port: {actual_port})", file=sys.stderr)
        print(f"Use 'oa2a logs' to view output", file=sys.stderr)
        return False

    # Check if port is already in use
    if _is_port_in_use(port):
        print(f"Error: Port {port} is already in use", file=sys.stderr)
        print(f"Another process may be listening on this port", file=sys.stderr)
        return False

    _ensure_dirs()

    # Prepare the command to run the daemon runner as a separate script
    daemon_runner_path = Path(__file__).parent / "daemon_runner.py"

    # Prepare environment - the daemon runner will use these env vars
    env = os.environ.copy()
    env["OA2A_HOST"] = host
    env["OA2A_PORT"] = str(port)
    env["OA2A_LOG_LEVEL"] = log_level.upper()

    cmd = [
        sys.executable,
        str(daemon_runner_path),
    ]

    try:
        # Open log file
        log_fd = open(LOG_FILE, "a")

        # Write a marker to log
        from datetime import datetime
        log_fd.write(f"\n\n[{datetime.now()}] Starting oa2a daemon...\n")
        log_fd.flush()

        # Start the process - use setsid on Unix to create new session
        kwargs = {
            "stdout": log_fd,
            "stderr": subprocess.STDOUT,
            "env": env,
        }

        if sys.platform != "win32":
            # On Unix, start in a new session so it survives parent exit
            kwargs["start_new_session"] = True

        process = subprocess.Popen(cmd, **kwargs)

        # Don't wait - close file descriptor in parent but child keeps it open
        log_fd.close()

        # Give the process a moment to fail (e.g., port in use)
        time.sleep(0.5)

        # Check if process is still running
        if process.poll() is not None:
            # Process exited immediately
            print("Failed to start server - check logs with 'oa2a logs'", file=sys.stderr)
            return False

        # Wait a bit more for the server to actually start
        time.sleep(0.5)

        # Check if port is now in use (server started successfully)
        for _ in range(10):
            if _is_port_in_use(port, "127.0.0.1"):
                break
            time.sleep(0.2)
        else:
            # Port never became active, check if process died
            if process.poll() is not None:
                print("Server process exited unexpectedly - check logs", file=sys.stderr)
                return False

        # Save the configuration
        _save_daemon_config(host, port)

        print(f"Server started (PID: {process.pid})")
        print(f"Listening on {host}:{port}")
        print(f"Logs: {LOG_FILE}")

        return True

    except Exception as e:
        print(f"Failed to start server: {e}", file=sys.stderr)
        return False


def stop_daemon(force: bool = False) -> bool:
    """
    Stop the background daemon.

    Args:
        force: If True, use SIGKILL instead of SIGTERM

    Returns:
        True if stopped successfully, False otherwise
    """
    _cleanup_stale_pidfile()

    pid = _read_pid()
    port = _read_port()

    # If pid file process not running but port still in use, find and kill the process
    if pid is not None and not _is_process_running(pid) and port is not None:
        if _is_port_in_use(port):
            print(f"Process with PID {pid} not found but port {port} is in use, searching...")
            found_pid = _find_pid_by_port(port)
            if found_pid is not None:
                pid = found_pid
                print(f"Found process on port {port}: PID {pid}")
            else:
                # Port is in use by another process, clear the pidfile
                _remove_pid()
                print(f"Port {port} is in use by another process, clearing pidfile")
                return True

    if pid is None:
        print("Server is not running")
        return True

    try:
        # Send signal
        signal_num = signal.SIGKILL if force else signal.SIGTERM
        os.kill(pid, signal_num)

        # Wait for process to terminate
        for _ in range(50):  # Wait up to 5 seconds
            if not _is_process_running(pid):
                break
            time.sleep(0.1)

        if _is_process_running(pid):
            if not force:
                print(f"Server did not stop gracefully, use -f to force kill", file=sys.stderr)
                return False
            # Force kill
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.2)

        _remove_pid()
        _remove_daemon_config()
        print(f"Server stopped (PID: {pid})")
        return True

    except (OSError, ProcessLookupError) as e:
        _remove_pid()
        _remove_daemon_config()
        print(f"Server stopped (PID: {pid})")
        return True
    except Exception as e:
        print(f"Failed to stop server: {e}", file=sys.stderr)
        return False


def restart_daemon(
    host: str = "0.0.0.0",
    port: int = 8080,
    log_level: str = "info",
) -> bool:
    """
    Restart the background daemon.

    Args:
        host: Server host
        port: Server port
        log_level: Logging level

    Returns:
        True if restarted successfully, False otherwise
    """
    print("Restarting server...")
    # Use force=True to ensure the old process is killed even if it's stuck
    stop_daemon(force=True)
    # Longer delay to ensure port is fully released
    time.sleep(1)
    return start_daemon(host, port, log_level)


def show_logs(follow: bool = False, lines: int = 50) -> bool:
    """
    Show server logs.

    Args:
        follow: If True, follow log output (like tail -f)
        lines: Number of lines to show from the end

    Returns:
        True if successful, False otherwise
    """
    if not LOG_FILE.exists():
        print("No log file found", file=sys.stderr)
        return False

    try:
        if follow:
            # Use subprocess to tail -f
            try:
                subprocess.run(
                    ["tail", "-f", "-n", str(lines), str(LOG_FILE)],
                    check=True,
                )
            except KeyboardInterrupt:
                pass
        else:
            # Read and print last N lines
            with open(LOG_FILE, "r") as f:
                content = f.readlines()
                # Print last N lines
                for line in content[-lines:]:
                    print(line, end="")

        return True

    except Exception as e:
        print(f"Failed to read logs: {e}", file=sys.stderr)
        return False


def run_foreground(
    host: str = "0.0.0.0",
    port: int = 8080,
    log_level: str = "info",
) -> None:
    """
    Run the server in foreground (blocking mode).

    This is the original behavior for compatibility.
    """
    # Import here to avoid circular imports
    from local_openai2anthropic.main import create_app
    from local_openai2anthropic.config import get_settings

    import uvicorn

    # Override settings with command line values
    os.environ["OA2A_HOST"] = host
    os.environ["OA2A_PORT"] = str(port)
    os.environ["OA2A_LOG_LEVEL"] = log_level.upper()

    settings = get_settings()

    app = create_app(settings)

    print(f"Starting server on {host}:{port}")
    print(f"Proxying to: {settings.openai_base_url}")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        timeout_keep_alive=300,
    )
