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


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Creates default config file if it doesn't exist and notifies the user.

    Returns:
        Settings instance loaded from config file
    """
    created = create_default_config()
    if created:
        config_file = get_config_file()
        print(f"Created default config file: {config_file}")
        print("Please edit it to add your API keys and settings.")
    return Settings.from_toml()
