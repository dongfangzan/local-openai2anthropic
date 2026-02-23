# SPDX-License-Identifier: Apache-2.0
"""
Tests for logging functionality.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from local_openai2anthropic.main import get_default_log_dir, setup_logging


class TestGetDefaultLogDir:
    """Tests for get_default_log_dir function."""

    def test_windows_log_dir(self):
        """Test Windows log directory."""
        with patch.object(sys, "platform", "win32"):
            with patch.dict(
                os.environ, {"LOCALAPPDATA": "C:\\Users\\Test\\AppData\\Local"}
            ):
                log_dir = get_default_log_dir()
                # Check path components exist
                assert "C:\\Users\\Test\\AppData\\Local" in log_dir
                assert "local-openai2anthropic" in log_dir
                assert "logs" in log_dir

    def test_windows_fallback(self):
        """Test Windows fallback when LOCALAPPDATA not set."""
        with patch.object(sys, "platform", "win32"):
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(
                    os.path, "expanduser", return_value="C:\\Users\\Test"
                ):
                    log_dir = get_default_log_dir()
                    # Check path components exist
                    assert "local-openai2anthropic" in log_dir
                    assert "logs" in log_dir

    def test_macos_log_dir(self):
        """Test macOS log directory."""
        with patch.object(sys, "platform", "darwin"):
            log_dir = get_default_log_dir()
            assert ".local/share/local-openai2anthropic/logs" in log_dir

    def test_linux_log_dir(self):
        """Test Linux log directory."""
        with patch.object(sys, "platform", "linux"):
            log_dir = get_default_log_dir()
            assert ".local/share/local-openai2anthropic/logs" in log_dir


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_creates_directory(self):
        """Test that setup_logging creates log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs", "subdir")
            setup_logging("INFO", log_dir)

            assert os.path.exists(log_dir)
            assert os.path.isdir(log_dir)

    def test_setup_logging_creates_log_file(self):
        """Test that setup_logging creates log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging("INFO", tmpdir)

            log_file = os.path.join(tmpdir, "server.log")
            assert os.path.exists(log_file)

    def test_setup_logging_handlers(self):
        """Test that setup_logging configures handlers correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging("DEBUG", tmpdir)

            root_logger = logging.getLogger()

            # Should have 2 handlers: console and file
            assert len(root_logger.handlers) == 2

            # Check handler types
            handler_types = [type(h).__name__ for h in root_logger.handlers]
            assert "StreamHandler" in handler_types
            assert "TimedRotatingFileHandler" in handler_types

    def test_setup_logging_log_level(self):
        """Test that setup_logging sets correct log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging("WARNING", tmpdir)

            root_logger = logging.getLogger()
            assert root_logger.level == logging.WARNING

    def test_setup_logging_file_rotation_settings(self):
        """Test that file handler has correct rotation settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging("INFO", tmpdir)

            root_logger = logging.getLogger()
            file_handler = None
            for handler in root_logger.handlers:
                if isinstance(handler, logging.handlers.TimedRotatingFileHandler):
                    file_handler = handler
                    break

            assert file_handler is not None
            assert file_handler.when.upper() == "MIDNIGHT"
            assert file_handler.backupCount == 1  # Keep 2 days of logs

    def test_setup_logging_log_format(self):
        """Test that log format is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging("INFO", tmpdir)

            # Get file handler
            root_logger = logging.getLogger()
            file_handler = None
            for handler in root_logger.handlers:
                if isinstance(handler, logging.handlers.TimedRotatingFileHandler):
                    file_handler = handler
                    break

            assert file_handler is not None
            formatter = file_handler.formatter
            assert "%(asctime)s" in formatter._fmt
            assert "%(name)s" in formatter._fmt
            assert "%(levelname)s" in formatter._fmt
            assert "%(message)s" in formatter._fmt

    def test_setup_logging_clears_existing_handlers(self):
        """Test that setup_logging clears existing handlers."""
        # Add a dummy handler first
        root_logger = logging.getLogger()
        dummy_handler = logging.NullHandler()
        root_logger.addHandler(dummy_handler)
        initial_count = len(root_logger.handlers)

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging("INFO", tmpdir)

            # Should have exactly 2 handlers after setup
            assert len(root_logger.handlers) == 2

    def test_setup_logging_with_none_log_dir(self):
        """Test setup_logging with None log_dir uses default."""
        with patch("local_openai2anthropic.main.get_default_log_dir") as mock_get_dir:
            with tempfile.TemporaryDirectory() as tmpdir:
                mock_get_dir.return_value = tmpdir
                setup_logging("INFO", None)

                mock_get_dir.assert_called_once()
                assert os.path.exists(os.path.join(tmpdir, "server.log"))

    def test_setup_logging_expands_user_directory(self):
        """Test that ~ in log_dir is expanded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a path with ~
            home = os.path.expanduser("~")
            relative_path = tmpdir.replace(home, "~")

            setup_logging("INFO", relative_path)

            log_file = os.path.join(tmpdir, "server.log")
            assert os.path.exists(log_file)


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_log_messages_written_to_file(self):
        """Test that log messages are written to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging("INFO", tmpdir)

            # Get a logger and write messages
            logger = logging.getLogger("test_logger")
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")

            # Read log file
            log_file = os.path.join(tmpdir, "server.log")
            with open(log_file, "r") as f:
                content = f.read()

            assert "Test info message" in content
            assert "Test warning message" in content
            assert "Test error message" in content

    def test_log_level_filtering(self):
        """Test that log level filters messages correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging("WARNING", tmpdir)

            logger = logging.getLogger("test_filter")
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

            log_file = os.path.join(tmpdir, "server.log")
            with open(log_file, "r") as f:
                content = f.read()

            assert "Debug message" not in content
            assert "Info message" not in content
            assert "Warning message" in content
            assert "Error message" in content

    def test_log_message_format(self):
        """Test that log messages have correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging("INFO", tmpdir)

            logger = logging.getLogger("test_format")
            logger.info("Format test")

            log_file = os.path.join(tmpdir, "server.log")
            with open(log_file, "r") as f:
                lines = f.readlines()

            # Find the line with our message
            test_line = None
            for line in lines:
                if "Format test" in line:
                    test_line = line
                    break

            assert test_line is not None
            # Check format: timestamp - name - level - message
            parts = test_line.split(" - ")
            assert len(parts) >= 4
            assert "test_format" in parts[1]
            assert "INFO" in parts[2]
            assert "Format test" in parts[3]


class TestLoggingConfiguration:
    """Tests for logging configuration."""

    def test_config_default_log_level(self, monkeypatch):
        """Test default log level in config."""
        from local_openai2anthropic.config import Settings

        monkeypatch.setattr(
            "local_openai2anthropic.config.load_config_from_file",
            lambda: {},
        )
        monkeypatch.delenv("OA2A_LOG_LEVEL", raising=False)
        settings = Settings()
        assert settings.log_level.upper() == "INFO"

    def test_config_log_level_override(self):
        """Test log level can be overridden."""
        from local_openai2anthropic.config import Settings

        settings = Settings(log_level="WARNING")
        assert settings.log_level.upper() == "WARNING"

    def test_config_default_log_dir(self):
        """Test default log dir in config."""
        from local_openai2anthropic.config import Settings

        settings = Settings()
        assert settings.log_dir == ""  # Empty means use platform default

    def test_config_custom_log_dir(self):
        """Test custom log dir in config."""
        from local_openai2anthropic.config import Settings

        settings = Settings(log_dir="/custom/log/path")
        assert settings.log_dir == "/custom/log/path"
