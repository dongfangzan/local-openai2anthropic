# SPDX-License-Identifier: Apache-2.0
"""
Configuration settings for the proxy server.
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict


def get_config_dir() -> Path:
    """Get platform-specific config directory.

    Returns:
        Path to the config directory (~/.oa2a)
    """
    return Path.home() / ".oa2a"


def get_config_file() -> Path:
    """Get config file path.

    Returns:
        Path to the config file (~/.oa2a/config.toml)
    """
    return get_config_dir() / "config.toml"


def create_default_config() -> bool:
    """Create default config file if not exists.

    Returns:
        True if a new config file was created, False if it already exists
    """
    config_file = get_config_file()
    if config_file.exists():
        return False

    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    # Set restrictive permissions (0o600) for the config directory on Unix-like systems
    if sys.platform != "win32":
        config_dir.chmod(0o700)

    default_config = """# OA2A Configuration File
# Place this file at ~/.oa2a/config.toml

# OpenAI API Configuration
openai_api_key = ""
openai_base_url = "https://api.openai.com/v1"
openai_org_id = ""
openai_project_id = ""

# Server Configuration
host = "0.0.0.0"
port = 8080
request_timeout = 300.0

# API Key for authenticating requests to this server (optional)
api_key = ""

# CORS settings
cors_origins = ["*"]
cors_credentials = true
cors_methods = ["*"]
cors_headers = ["*"]

# Logging
log_level = "INFO"
log_dir = ""  # Empty uses platform-specific default

# Tavily Web Search Configuration
tavily_api_key = ""
tavily_timeout = 30.0
tavily_max_results = 5
websearch_max_uses = 5
"""
    config_file.write_text(default_config, encoding="utf-8")

    # Set restrictive permissions (0o600) for the config file on Unix-like systems
    if sys.platform != "win32":
        config_file.chmod(0o600)

    return True


def interactive_setup() -> dict:
    """Interactive configuration setup wizard.

    Guides user through setting up essential configuration values.

    Returns:
        Dictionary containing user-provided configuration
    """
    print("=" * 60)
    print("  Welcome to local-openai2anthropic Setup Wizard")
    print("=" * 60)
    print()
    print("This wizard will help you create the initial configuration.")
    print(f"Config file will be saved to: {get_config_file()}")
    print()

    config = {}

    # OpenAI API Key (required)
    print("[1/3] OpenAI API Configuration")
    print("-" * 40)
    while True:
        api_key = input("Enter your OpenAI API Key (required): ").strip()
        if api_key:
            config["openai_api_key"] = api_key
            break
        print("API Key is required. Please enter a valid key.")

    # Base URL (optional, with default)
    default_url = "https://api.openai.com/v1"
    base_url = input(f"Enter OpenAI Base URL [{default_url}]: ").strip()
    config["openai_base_url"] = base_url if base_url else default_url

    print()
    print("[2/3] Server Configuration")
    print("-" * 40)

    # Host (with default)
    default_host = "0.0.0.0"
    host = input(f"Enter server host [{default_host}]: ").strip()
    config["host"] = host if host else default_host

    # Port (with default)
    default_port = "8080"
    port_input = input(f"Enter server port [{default_port}]: ").strip()
    try:
        config["port"] = int(port_input) if port_input else int(default_port)
    except ValueError:
        print(f"Invalid port number, using default: {default_port}")
        config["port"] = int(default_port)

    # API Key for server authentication (optional)
    print()
    print("[3/3] Server API Authentication (Optional)")
    print("-" * 40)
    print("Set an API key to authenticate requests to this server.")
    print(
        "Leave empty to allow unauthenticated access (not recommended for production)."
    )
    server_api_key = input("Enter server API key (optional): ").strip()
    if server_api_key:
        config["api_key"] = server_api_key

    print()
    print("=" * 60)
    print("  Configuration Summary")
    print("=" * 60)
    print(f"OpenAI Base URL: {config.get('openai_base_url', default_url)}")
    print(
        f"Server: {config.get('host', default_host)}:{config.get('port', default_port)}"
    )
    print(f"OpenAI API Key: {config.get('openai_api_key', '')[:8]}... (configured)")
    if config.get("api_key"):
        print(f"Server Auth: {config['api_key'][:8]}... (configured)")
    print()

    return config


def create_config_from_dict(config: dict) -> None:
    """Create config file from dictionary.

    Args:
        config: Dictionary containing configuration values
    """
    import tomli_w

    config_file = get_config_file()
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    # Set restrictive permissions for the config directory on Unix-like systems
    if sys.platform != "win32":
        config_dir.chmod(0o700)

    # Build config dict with proper structure
    toml_config: dict = {
        "openai_api_key": config.get("openai_api_key", ""),
        "openai_base_url": config.get(
            "openai_base_url", "https://api.openai.com/v1"
        ),
        "host": config.get("host", "0.0.0.0"),
        "port": config.get("port", 8080),
        "request_timeout": config.get("request_timeout", 300.0),
        "cors_origins": ["*"],
        "cors_credentials": True,
        "cors_methods": ["*"],
        "cors_headers": ["*"],
        "log_level": "INFO",
        "log_dir": "",
        "tavily_timeout": 30.0,
        "tavily_max_results": 5,
        "websearch_max_uses": 5,
    }

    # Add optional values only if present
    if config.get("api_key"):
        toml_config["api_key"] = config["api_key"]

    if config.get("tavily_api_key"):
        toml_config["tavily_api_key"] = config["tavily_api_key"]

    # Write using proper TOML serialization (prevents injection attacks)
    with open(config_file, "wb") as f:
        tomli_w.dump(toml_config, f)

    # Set restrictive permissions for the config file on Unix-like systems
    if sys.platform != "win32":
        config_file.chmod(0o600)


def load_config_from_file() -> dict:
    """Load configuration from TOML file.

    Returns:
        Dictionary containing configuration values, empty dict if file doesn't exist
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    config_file = get_config_file()
    if not config_file.exists():
        return {}
    with open(config_file, "rb") as f:
        return tomllib.load(f)


class Settings(BaseModel):
    """Application settings loaded from config file."""

    model_config = ConfigDict(extra="ignore")

    # OpenAI API Configuration
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_org_id: Optional[str] = None
    openai_project_id: Optional[str] = None

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080
    request_timeout: float = 300.0  # 5 minutes

    # API Key for authenticating requests to this server (optional)
    api_key: Optional[str] = None

    # CORS settings
    cors_origins: list[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: list[str] = ["*"]
    cors_headers: list[str] = ["*"]

    # Logging
    log_level: str = "INFO"
    log_dir: str = ""  # Empty means use platform-specific default

    # Tavily Web Search Configuration
    tavily_api_key: Optional[str] = None
    tavily_timeout: float = 30.0
    tavily_max_results: int = 5
    websearch_max_uses: int = 5  # Default max_uses per request

    @property
    def openai_auth_headers(self) -> dict[str, str]:
        """Get OpenAI authentication headers."""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
        }
        if self.openai_org_id:
            headers["OpenAI-Organization"] = self.openai_org_id
        if self.openai_project_id:
            headers["OpenAI-Project"] = self.openai_project_id
        return headers

    @classmethod
    def from_toml(cls) -> "Settings":
        """Load settings from TOML config file.

        Returns:
            Settings instance populated from config file
        """
        config_data = load_config_from_file()
        return cls(**config_data)


def is_interactive() -> bool:
    """Check if running in an interactive terminal.

    Returns:
        True if stdin is a TTY (interactive), False otherwise
    """
    return sys.stdin.isatty()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Creates config file interactively if it doesn't exist and running in a TTY.
    Falls back to creating a default config file in non-interactive environments.

    Returns:
        Settings instance loaded from config file
    """
    config_file = get_config_file()
    if not config_file.exists():
        if is_interactive():
            # Interactive setup wizard
            config = interactive_setup()
            create_config_from_dict(config)
            print(f"\nConfiguration saved to: {config_file}")
            print("You can edit this file later to change settings.\n")
        else:
            # Non-interactive environment: create default config
            create_default_config()
            print(f"Created default config file: {config_file}")
            print("Please edit it to add your API keys and settings.")
    return Settings.from_toml()
