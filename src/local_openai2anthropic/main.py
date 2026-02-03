# SPDX-License-Identifier: Apache-2.0
"""
Main entry point for the local-openai2anthropic proxy server.
"""

import argparse
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from local_openai2anthropic.config import Settings, get_config_file, get_settings
from local_openai2anthropic.protocol import AnthropicError, AnthropicErrorResponse
from local_openai2anthropic.router import router


def get_default_log_dir() -> str:
    """Get default log directory based on platform.

    Returns:
        Path to log directory
    """
    if sys.platform == 'win32':
        # Windows: use %LOCALAPPDATA%\local-openai2anthropic\logs
        base_dir = os.environ.get('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
        return os.path.join(base_dir, 'local-openai2anthropic', 'logs')
    else:
        # macOS/Linux: use ~/.local/share/local-openai2anthropic/logs
        return os.path.expanduser("~/.local/share/local-openai2anthropic/logs")


def setup_logging(log_level: str, log_dir: str | None = None) -> None:
    """Setup logging with daily rotation, keeping only today's logs.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (platform-specific default)
    """
    # Default log directory based on platform
    if log_dir is None:
        log_dir = get_default_log_dir()

    # Expand user directory if specified
    log_dir = os.path.expanduser(log_dir)

    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_dir, "server.log")

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with daily rotation
    # backupCount=0 means no backup files are kept (only today's log)
    # when='midnight' rotates at midnight
    file_handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',
        interval=1,
        backupCount=0,  # Keep only today's log
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging configured. Log file: {log_file}")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    # Configure logging with daily rotation
    # Use platform-specific default if log_dir is not set
    log_dir = settings.log_dir if settings.log_dir else None
    setup_logging(settings.log_level, log_dir)

    # Create FastAPI app
    app = FastAPI(
        title="local-openai2anthropic",
        description="A proxy server that converts Anthropic Messages API to OpenAI API",
        version="0.3.7",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Store settings in app state
    app.state.settings = settings

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )

    # Add API key authentication middleware if configured
    if settings.api_key:

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            """Validate API key if configured."""
            # Skip auth for docs and health check
            if request.url.path in ["/docs", "/redoc", "/openapi.json", "/health"]:
                return await call_next(request)

            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                error_response = AnthropicErrorResponse(
                    error=AnthropicError(
                        type="authentication_error",
                        message="Missing or invalid Authorization header",
                    )
                )
                return JSONResponse(
                    status_code=401,
                    content=error_response.model_dump(),
                )

            token = auth_header[7:]  # Remove "Bearer " prefix
            if token != settings.api_key:
                error_response = AnthropicErrorResponse(
                    error=AnthropicError(
                        type="authentication_error",
                        message="Invalid API key",
                    )
                )
                return JSONResponse(
                    status_code=401,
                    content=error_response.model_dump(),
                )

            return await call_next(request)

    # Include routers
    app.include_router(router)

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Convert HTTP exceptions to Anthropic error format."""
        # Check if detail is already an AnthropicErrorResponse dict
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail,
            )

        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="api_error",
                message=str(exc.detail),
            )
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logging.exception("Unhandled exception")
        error_response = AnthropicErrorResponse(
            error=AnthropicError(
                type="internal_error",
                message=str(exc),
            )
        )
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(),
        )

    return app


def run_foreground(settings: Settings) -> None:
    """Run server in foreground mode (blocking)."""
    # Validate required settings
    if not settings.openai_api_key:
        config_file = get_config_file()
        print(
            f"Error: openai_api_key is required.\n"
            f"Please edit the configuration file:\n"
            f"  {config_file}\n"
            f"\nSet your OpenAI API key:\n"
            f'  openai_api_key = "your-api-key"',
            file=sys.stderr,
        )
        sys.exit(1)

    # Create app
    app = create_app(settings)

    # Run server
    print(f"Starting server on {settings.host}:{settings.port}")
    print(f"Proxying to: {settings.openai_base_url}")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        timeout_keep_alive=300,  # Keep connections alive for 5 minutes
    )


def main() -> None:
    """Main entry point with subcommand support."""
    # Create main parser
    parser = argparse.ArgumentParser(
        prog="oa2a",
        description="A proxy server that converts Anthropic Messages API to OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  oa2a start              # Start server in background
  oa2a stop               # Stop background server
  oa2a restart            # Restart background server
  oa2a status             # Check server status
  oa2a logs               # View server logs
  oa2a logs -f            # Follow server logs (tail -f)
  oa2a                    # Run server in foreground (default behavior)
        """.strip(),
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.3.7",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start command
    start_parser = subparsers.add_parser("start", help="Start server in background")
    start_parser.add_argument(
        "--host",
        default=None,
        help="Server host (default: 0.0.0.0)",
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 8080)",
    )
    start_parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    # stop command
    stop_parser = subparsers.add_parser("stop", help="Stop background server")
    stop_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force kill the server",
    )

    # restart command
    restart_parser = subparsers.add_parser("restart", help="Restart background server")
    restart_parser.add_argument(
        "--host",
        default=None,
        help="Server host (default: 0.0.0.0)",
    )
    restart_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 8080)",
    )
    restart_parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    # status command
    status_parser = subparsers.add_parser("status", help="Check server status")

    # logs command
    logs_parser = subparsers.add_parser("logs", help="View server logs")
    logs_parser.add_argument(
        "-f", "--follow",
        action="store_true",
        help="Follow log output (like tail -f)",
    )
    logs_parser.add_argument(
        "-n", "--lines",
        type=int,
        default=50,
        help="Number of lines to show (default: 50)",
    )

    args = parser.parse_args()

    # Import daemon module only when needed
    from local_openai2anthropic import daemon

    # Handle subcommands
    if args.command == "start":
        # Get settings for defaults
        settings = get_settings()
        host = args.host or settings.host
        port = args.port or settings.port

        success = daemon.start_daemon(
            host=host,
            port=port,
            log_level=args.log_level,
        )
        sys.exit(0 if success else 1)

    elif args.command == "stop":
        success = daemon.stop_daemon(force=args.force)
        sys.exit(0 if success else 1)

    elif args.command == "restart":
        # Get settings for defaults
        settings = get_settings()
        host = args.host or settings.host
        port = args.port or settings.port

        success = daemon.restart_daemon(
            host=host,
            port=port,
            log_level=args.log_level,
        )
        sys.exit(0 if success else 1)

    elif args.command == "status":
        running, pid, config = daemon.get_status()
        if running and config:
            host = config.get("host", "0.0.0.0")
            port = config.get("port", 8080)
            print(f"Server is running (PID: {pid})")
            print(f"Listening on: {host}:{port}")
        elif running:
            print(f"Server is running (PID: {pid})")
        else:
            print("Server is not running")
        sys.exit(0)

    elif args.command == "logs":
        success = daemon.show_logs(follow=args.follow, lines=args.lines)
        sys.exit(0 if success else 1)

    else:
        # No command - run in foreground (original behavior)
        settings = get_settings()
        run_foreground(settings)


if __name__ == "__main__":
    main()
