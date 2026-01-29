# SPDX-License-Identifier: Apache-2.0
"""
Configuration settings for the proxy server.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_prefix="OA2A_",  # OpenAI-to-Anthropic prefix
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # OpenAI API Configuration
    openai_api_key: str
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


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
