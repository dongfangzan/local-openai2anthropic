"""
Tests for the config module.
"""

from pathlib import Path

import pytest

from local_openai2anthropic.config import (
    Settings,
    create_config_from_dict,
    create_default_config,
    get_config_dir,
    get_config_file,
    get_settings,
    interactive_setup,
    is_interactive,
    load_config_from_file,
)


class TestConfigFile:
    """Tests for config file management."""

    def test_get_config_dir(self):
        """Test getting config directory."""
        config_dir = get_config_dir()
        assert isinstance(config_dir, Path)
        assert config_dir.name == ".oa2a"

    def test_get_config_file(self):
        """Test getting config file path."""
        config_file = get_config_file()
        assert isinstance(config_file, Path)
        assert config_file.name == "config.toml"
        assert config_file.parent.name == ".oa2a"

    def test_create_default_config(self, tmp_path, monkeypatch):
        """Test creating default config file."""
        # Mock get_config_dir to use temp directory
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        # Should create new file
        created = create_default_config()
        assert created is True
        assert get_config_file().exists()

        # Verify content
        content = get_config_file().read_text()
        assert "OA2A Configuration File" in content
        assert 'openai_base_url = "https://api.openai.com/v1"' in content

        # Should not create if exists
        created = create_default_config()
        assert created is False

    def test_create_default_config_existing(self, tmp_path, monkeypatch):
        """Test that existing config is not overwritten."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        # Create config dir and file manually
        config_dir = tmp_path / ".oa2a"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text("custom_config = true")

        # Should not overwrite
        created = create_default_config()
        assert created is False
        assert config_file.read_text() == "custom_config = true"

    def test_load_config_from_file(self, tmp_path, monkeypatch):
        """Test loading config from TOML file."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        # Create config file
        create_default_config()

        # Load config
        config = load_config_from_file()
        assert "openai_base_url" in config
        assert config["openai_base_url"] == "https://api.openai.com/v1"
        assert config["port"] == 8080
        assert config["host"] == "0.0.0.0"

    def test_load_config_from_file_not_exists(self, tmp_path, monkeypatch):
        """Test loading config when file doesn't exist."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        # Should return empty dict
        config = load_config_from_file()
        assert config == {}

    def test_load_config_custom_values(self, tmp_path, monkeypatch):
        """Test loading config with custom values."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        # Create config dir and file with custom values
        config_dir = tmp_path / ".oa2a"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text("""
openai_api_key = "custom-key"
host = "127.0.0.1"
port = 9000
log_level = "INFO"
""")

        config = load_config_from_file()
        assert config["openai_api_key"] == "custom-key"
        assert config["host"] == "127.0.0.1"
        assert config["port"] == 9000
        assert config["log_level"] == "INFO"


class TestSettings:
    """Tests for Settings class."""

    def test_settings_initialization(self):
        """Test that Settings can be initialized with explicit values."""
        settings = Settings(
            openai_api_key="test-key",
            openai_base_url="https://custom.api.com",
            openai_org_id="org-123",
            openai_project_id="proj-456",
            host="127.0.0.1",
            port=9000,
            request_timeout=60.0,
            api_key="server-api-key",
            tavily_api_key="tavily-key",
            tavily_timeout=15.0,
            tavily_max_results=10,
            websearch_max_uses=3,
            log_level="INFO",
            cors_origins=["https://example.com"],
            cors_credentials=False,
            cors_methods=["GET", "POST"],
            cors_headers=["Authorization", "Content-Type"],
        )

        assert settings.openai_api_key == "test-key"
        assert settings.openai_base_url == "https://custom.api.com"
        assert settings.openai_org_id == "org-123"
        assert settings.openai_project_id == "proj-456"
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.request_timeout == 60.0
        assert settings.api_key == "server-api-key"
        assert settings.tavily_api_key == "tavily-key"
        assert settings.tavily_timeout == 15.0
        assert settings.tavily_max_results == 10
        assert settings.websearch_max_uses == 3
        assert settings.log_level == "INFO"
        assert settings.cors_origins == ["https://example.com"]
        assert settings.cors_credentials is False
        assert settings.cors_methods == ["GET", "POST"]
        assert settings.cors_headers == ["Authorization", "Content-Type"]

    def test_settings_from_toml(self, tmp_path, monkeypatch):
        """Test creating Settings from TOML file."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )
        monkeypatch.setattr(
            "local_openai2anthropic.config.load_config_from_file",
            lambda: {
                "openai_api_key": "test-key",
                "host": "127.0.0.1",
                "port": 9000,
            },
        )
        # Create config with custom values
        config_dir = tmp_path / ".oa2a"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text("""
openai_api_key = "test-key"
host = "127.0.0.1"
port = 9000
""")

        settings = Settings.from_toml()
        assert settings.openai_api_key == "test-key"
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        # Default values should still apply
        assert settings.openai_base_url == "https://api.openai.com/v1"

    def test_settings_from_toml_empty_file(self, tmp_path, monkeypatch):
        """Test creating Settings from empty TOML file."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )
        monkeypatch.setattr(
            "local_openai2anthropic.config.load_config_from_file",
            lambda: {},
        )
        # Create empty config file
        config_dir = tmp_path / ".oa2a"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text("")

        settings = Settings.from_toml()
        # Should use all default values
        assert settings.openai_base_url == "https://api.openai.com/v1"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8080

    def test_openai_auth_headers_basic(self):
        """Test OpenAI auth headers with just API key."""
        settings = Settings(
            openai_api_key="test-key",
        )

        headers = settings.openai_auth_headers

        assert headers == {"Authorization": "Bearer test-key"}

    def test_openai_auth_headers_with_org(self):
        """Test OpenAI auth headers with org ID."""
        settings = Settings(
            openai_api_key="test-key",
            openai_org_id="org-123",
        )

        headers = settings.openai_auth_headers

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["OpenAI-Organization"] == "org-123"

    def test_openai_auth_headers_with_project(self):
        """Test OpenAI auth headers with project ID."""
        settings = Settings(
            openai_api_key="test-key",
            openai_project_id="proj-456",
        )

        headers = settings.openai_auth_headers

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["OpenAI-Project"] == "proj-456"

    def test_openai_auth_headers_full(self):
        """Test OpenAI auth headers with all options."""
        settings = Settings(
            openai_api_key="test-key",
            openai_org_id="org-123",
            openai_project_id="proj-456",
        )

        headers = settings.openai_auth_headers

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["OpenAI-Organization"] == "org-123"
        assert headers["OpenAI-Project"] == "proj-456"

    def test_openai_auth_headers_no_api_key(self):
        """Test OpenAI auth headers without API key."""
        settings = Settings()

        headers = settings.openai_auth_headers

        # Should still work but with None in the f-string
        assert "Authorization" in headers
        assert "Bearer" in headers["Authorization"]

    def test_optional_api_keys(self):
        """Test that optional API keys can be set or None."""
        # With explicit values
        settings = Settings(
            openai_api_key="test-key",
            api_key="server-key",
            tavily_api_key="tavily-key",
        )

        assert settings.openai_api_key == "test-key"
        assert settings.api_key == "server-key"
        assert settings.tavily_api_key == "tavily-key"

    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored."""
        # This should not raise an error due to extra="ignore"
        settings = Settings(
            openai_api_key="test-key",
            custom_field="ignored",  # type: ignore[call-arg]
            another_extra=123,  # type: ignore[call-arg]
        )

        assert settings.openai_api_key == "test-key"

    def test_cors_settings_defaults(self):
        """Test default CORS settings."""
        settings = Settings()

        assert settings.cors_origins == ["*"]
        assert settings.cors_credentials is True
        assert settings.cors_methods == ["*"]
        assert settings.cors_headers == ["*"]

    def test_tavily_settings(self):
        """Test Tavily-specific settings."""
        settings = Settings(
            tavily_api_key="tvly-test",
            tavily_timeout=45.0,
            tavily_max_results=10,
            websearch_max_uses=8,
        )

        assert settings.tavily_api_key == "tvly-test"
        assert settings.tavily_timeout == 45.0
        assert settings.tavily_max_results == 10
        assert settings.websearch_max_uses == 8


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_instance(self, tmp_path, monkeypatch):
        """Test that get_settings returns a Settings instance."""
        # Clear the cache first
        get_settings.cache_clear()

        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_caching(self, tmp_path, monkeypatch):
        """Test that get_settings caches the result."""
        # Clear the cache first
        get_settings.cache_clear()

        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same object due to @lru_cache
        assert settings1 is settings2

    def test_get_settings_creates_default_config(self, tmp_path, monkeypatch, capsys):
        """Test that get_settings creates default config and notifies user."""
        # Clear the cache first
        get_settings.cache_clear()

        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        settings = get_settings()

        # Check that config file was created
        assert get_config_file().exists()

        # Check notification was printed
        captured = capsys.readouterr()
        assert "Created default config file" in captured.out
        assert "Please edit it to add your API keys and settings" in captured.out


class TestIsInteractive:
    """Tests for is_interactive function."""

    def test_is_interactive_returns_bool(self):
        """Test that is_interactive returns a boolean."""
        result = is_interactive()
        assert isinstance(result, bool)


class TestCreateConfigFromDict:
    """Tests for create_config_from_dict function."""

    def test_create_config_from_dict_basic(self, tmp_path, monkeypatch):
        """Test creating config file from dictionary with basic values."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        config = {
            "openai_api_key": "test-api-key",
            "openai_base_url": "https://api.openai.com/v1",
            "host": "0.0.0.0",
            "port": 8080,
        }

        create_config_from_dict(config)

        config_file = get_config_file()
        assert config_file.exists()

        content = config_file.read_text()
        assert 'openai_api_key = "test-api-key"' in content
        assert 'openai_base_url = "https://api.openai.com/v1"' in content
        assert 'host = "0.0.0.0"' in content
        assert "port = 8080" in content

    def test_create_config_from_dict_with_optional_values(self, tmp_path, monkeypatch):
        """Test creating config with optional values."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        config = {
            "openai_api_key": "test-api-key",
            "api_key": "server-api-key",
        }

        create_config_from_dict(config)

        config_file = get_config_file()
        content = config_file.read_text()

        assert 'api_key = "server-api-key"' in content

    def test_create_config_from_dict_without_optional_values(
        self, tmp_path, monkeypatch
    ):
        """Test creating config without optional values omits them."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        config = {
            "openai_api_key": "test-api-key",
        }

        create_config_from_dict(config)

        config_file = get_config_file()
        content = config_file.read_text()

        # Parse TOML to verify structure
        import tomllib

        with open(config_file, "rb") as f:
            parsed = tomllib.load(f)

        # Optional api_key should not be present when not provided
        assert "api_key" not in parsed
        # But openai_api_key should be present
        assert parsed["openai_api_key"] == "test-api-key"

    def test_create_config_from_dict_custom_host_port(self, tmp_path, monkeypatch):
        """Test creating config with custom host and port."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        config = {
            "openai_api_key": "test-key",
            "host": "127.0.0.1",
            "port": 9000,
        }

        create_config_from_dict(config)

        config_file = get_config_file()
        content = config_file.read_text()

        assert 'host = "127.0.0.1"' in content
        assert "port = 9000" in content

    def test_create_config_from_dict_creates_directory(self, tmp_path, monkeypatch):
        """Test that create_config_from_dict creates the config directory."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        config_dir = tmp_path / ".oa2a"
        assert not config_dir.exists()

        create_config_from_dict({"openai_api_key": "test"})

        assert config_dir.exists()
        assert config_dir.is_dir()

    def test_create_config_file_permissions(self, tmp_path, monkeypatch):
        """Test that created config file has correct permissions."""
        import sys

        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )

        create_config_from_dict({"openai_api_key": "test"})

        if sys.platform != "win32":
            import stat

            config_file = get_config_file()
            config_dir = get_config_dir()

            file_perms = stat.S_IMODE(config_file.stat().st_mode)
            dir_perms = stat.S_IMODE(config_dir.stat().st_mode)

            assert file_perms == 0o600
            assert dir_perms == 0o700


class TestInteractiveSetup:
    """Tests for interactive_setup function."""

    def test_interactive_setup_all_values(self, monkeypatch):
        """Test interactive setup with all values provided."""
        # Mock get_config_file to return a fixed path
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_file",
            lambda: Path("/tmp/.oa2a/config.toml"),
        )

        # Mock input for all inputs
        inputs = iter(
            [
                "openai-api-key",  # api_key (required)
                "https://custom.api.com",  # base_url
                "127.0.0.1",  # host
                "9000",  # port
                "server-api-key",  # server api_key
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        config = interactive_setup()

        assert config["openai_api_key"] == "openai-api-key"
        assert config["openai_base_url"] == "https://custom.api.com"
        assert config["host"] == "127.0.0.1"
        assert config["port"] == 9000
        assert config["api_key"] == "server-api-key"

    def test_interactive_setup_defaults(self, monkeypatch):
        """Test interactive setup with default values."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_file",
            lambda: Path("/tmp/.oa2a/config.toml"),
        )

        # Mock input - api key + all empty to accept defaults
        inputs = iter(["openai-api-key", "", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        config = interactive_setup()

        assert config["openai_api_key"] == "openai-api-key"
        assert config["openai_base_url"] == "https://api.openai.com/v1"
        assert config["host"] == "0.0.0.0"
        assert config["port"] == 8080
        assert "api_key" not in config

    def test_interactive_setup_invalid_port(self, monkeypatch):
        """Test interactive setup with invalid port falls back to default."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_file",
            lambda: Path("/tmp/.oa2a/config.toml"),
        )

        # Mock input with invalid port
        inputs = iter(["openai-api-key", "", "", "invalid", ""])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        config = interactive_setup()

        assert config["port"] == 8080  # Should fall back to default

    def test_interactive_setup_requires_api_key(self, monkeypatch):
        """Test that interactive setup requires API key."""
        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_file",
            lambda: Path("/tmp/.oa2a/config.toml"),
        )

        # Mock input - first empty (rejected), then valid
        inputs = iter(["", "openai-api-key", "", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        config = interactive_setup()

        assert config["openai_api_key"] == "openai-api-key"


class TestGetSettingsInteractive:
    """Tests for get_settings with interactive mode."""

    def test_get_settings_non_interactive_creates_default(
        self, tmp_path, monkeypatch, capsys
    ):
        """Test get_settings in non-interactive mode creates default config."""
        # Clear cache
        get_settings.cache_clear()

        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )
        monkeypatch.setattr(
            "local_openai2anthropic.config.is_interactive", lambda: False
        )

        settings = get_settings()

        # Should create default config
        assert get_config_file().exists()
        assert isinstance(settings, Settings)

        # Check notification
        captured = capsys.readouterr()
        assert "Created default config file" in captured.out

    def test_get_settings_interactive_mode(self, tmp_path, monkeypatch, capsys):
        """Test get_settings in interactive mode runs setup wizard."""
        # Clear cache
        get_settings.cache_clear()

        monkeypatch.setattr(
            "local_openai2anthropic.config.get_config_dir", lambda: tmp_path / ".oa2a"
        )
        monkeypatch.setattr(
            "local_openai2anthropic.config.is_interactive", lambda: True
        )

        # Mock interactive_setup
        mock_config = {
            "openai_api_key": "test-key",
            "host": "127.0.0.1",
            "port": 9000,
        }
        monkeypatch.setattr(
            "local_openai2anthropic.config.interactive_setup", lambda: mock_config
        )

        settings = get_settings()

        # Should create config with interactive values
        assert get_config_file().exists()
        assert settings.openai_api_key == "test-key"
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000

        # Check notification
        captured = capsys.readouterr()
        assert "Configuration saved to" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
