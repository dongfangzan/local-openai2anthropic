# SPDX-License-Identifier: Apache-2.0
"""
Main entry point for the local-openai2anthropic proxy server.
"""

import logging
import sys

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from local_openai2anthropic.config import Settings, get_settings
from local_openai2anthropic.protocol import AnthropicError, AnthropicErrorResponse
from local_openai2anthropic.router import router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="local-openai2anthropic",
        description="A proxy server that converts Anthropic Messages API to OpenAI API",
        version="0.1.0",
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


def main() -> None:
    """Main entry point."""
    # Load settings
    settings = get_settings()
    
    # Validate required settings
    if not settings.openai_api_key:
        print(
            "Error: OPENAI_API_KEY environment variable is required.\n"
            "Set it via:\n"
            "  - Environment variable: export OA2A_OPENAI_API_KEY='your-key'\n"
            "  - Or create a .env file with OPENAI_API_KEY=your-key",
            file=sys.stderr,
        )
        sys.exit(1)
    
    # Create app
    app = create_app(settings)
    
    # Run server
    print(f"Starting local-openai2anthropic server on {settings.host}:{settings.port}")
    print(f"Proxying to: {settings.openai_base_url}")
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
