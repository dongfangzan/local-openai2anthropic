"""
Tests for the daemon_runner module.
"""

import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_openai2anthropic import daemon_runner


class TestLogMessage:
    """Tests for log_message function."""

    def test_log_message(self):
        """Test logging a message."""
        with patch("builtins.print") as mock_print:
            with patch.object(sys.stderr, "flush"):
                daemon_runner.log_message("Test message")
                mock_print.assert_called_once()
                # Check that the message contains timestamp and text
                call_args = mock_print.call_args
                assert "Test message" in call_args[0][0]


class TestWritePid:
    """Tests for _write_pid function."""

    def test_write_pid_success(self):
        """Test successfully writing PID."""
        with patch.object(Path, "mkdir"):
            with patch.object(Path, "write_text") as mock_write:
                with patch.object(daemon_runner, "log_message"):
                    daemon_runner._write_pid(12345)
                    mock_write.assert_called_once_with("12345")

    def test_write_pid_failure(self):
        """Test handling write failure."""
        with patch.object(Path, "mkdir", side_effect=OSError("Cannot create")):
            with patch.object(daemon_runner, "log_message"):
                # Should not raise
                daemon_runner._write_pid(12345)


class TestRemovePid:
    """Tests for _remove_pid function."""

    def test_remove_pid_success(self):
        """Test successfully removing PID file."""
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "unlink") as mock_unlink:
                with patch.object(daemon_runner, "log_message"):
                    daemon_runner._remove_pid()
                    mock_unlink.assert_called_once()

    def test_remove_pid_not_exists(self):
        """Test removing when file doesn't exist."""
        with patch.object(Path, "exists", return_value=False):
            with patch.object(Path, "unlink") as mock_unlink:
                daemon_runner._remove_pid()
                mock_unlink.assert_not_called()

    def test_remove_pid_oserror(self):
        """Test handling OSError when removing."""
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "unlink", side_effect=OSError("Cannot remove")):
                # Should not raise
                daemon_runner._remove_pid()


class TestSignalHandler:
    """Tests for _signal_handler function."""

    def test_sigterm_handler(self):
        """Test SIGTERM signal handler."""
        with patch.object(daemon_runner, "_remove_pid") as mock_remove:
            with patch.object(daemon_runner, "log_message"):
                with pytest.raises(SystemExit) as exc_info:
                    daemon_runner._signal_handler(signal.SIGTERM, None)
                assert exc_info.value.code == 0
                mock_remove.assert_called_once()

    def test_sigint_handler(self):
        """Test SIGINT signal handler."""
        with patch.object(daemon_runner, "_remove_pid") as mock_remove:
            with patch.object(daemon_runner, "log_message"):
                with pytest.raises(SystemExit) as exc_info:
                    daemon_runner._signal_handler(signal.SIGINT, None)
                assert exc_info.value.code == 0
                mock_remove.assert_called_once()


class TestRunServer:
    """Tests for run_server function."""

    @patch("local_openai2anthropic.config.get_settings")
    def test_missing_api_key(self, mock_get_settings):
        """Test server exits when API key is missing."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = None
        mock_get_settings.return_value = mock_settings

        with patch("os.getpid", return_value=12345):
            with patch.object(daemon_runner, "_write_pid"):
                with patch("atexit.register"):
                    with patch("signal.signal"):
                        with patch.object(daemon_runner, "log_message"):
                            with pytest.raises(SystemExit) as exc_info:
                                daemon_runner.run_server()

                            assert exc_info.value.code == 1

    @patch("uvicorn.run")
    @patch("local_openai2anthropic.config.get_settings")
    @patch("local_openai2anthropic.main.create_app")
    def test_run_server_success(self, mock_create_app, mock_get_settings, mock_uvicorn_run):
        """Test successful server startup."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test-key"
        mock_settings.host = "127.0.0.1"
        mock_settings.port = 9000
        mock_settings.log_level = "INFO"
        mock_settings.openai_base_url = "https://api.openai.com"
        mock_get_settings.return_value = mock_settings

        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        with patch("os.getpid", return_value=12345):
            with patch.object(daemon_runner, "_write_pid"):
                with patch("atexit.register"):
                    with patch("signal.signal"):
                        with patch.object(daemon_runner, "log_message"):
                            daemon_runner.run_server()

        mock_create_app.assert_called_once_with(mock_settings)
        mock_uvicorn_run.assert_called_once()
        call_kwargs = mock_uvicorn_run.call_args.kwargs
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 9000
        assert call_kwargs["log_level"] == "info"

    @patch("local_openai2anthropic.config.get_settings")
    def test_run_server_exception(self, mock_get_settings):
        """Test handling exception during server startup."""
        mock_get_settings.side_effect = Exception("Config error")

        with patch("os.getpid", return_value=12345):
            with patch.object(daemon_runner, "_write_pid"):
                with patch("atexit.register"):
                    with patch("signal.signal"):
                        with patch.object(daemon_runner, "log_message"):
                            with pytest.raises(SystemExit) as exc_info:
                                daemon_runner.run_server()

                            assert exc_info.value.code == 1

    @patch("local_openai2anthropic.config.get_settings")
    @patch("local_openai2anthropic.main.create_app")
    @patch("uvicorn.run")
    def test_signal_handlers_registered(self, mock_uvicorn_run, mock_create_app, mock_get_settings):
        """Test that signal handlers are registered."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test-key"
        mock_settings.host = "127.0.0.1"
        mock_settings.port = 9000
        mock_settings.log_level = "INFO"
        mock_settings.openai_base_url = "https://api.openai.com"
        mock_get_settings.return_value = mock_settings

        signal_calls = []

        def mock_signal(signum, handler):
            signal_calls.append((signum, handler))

        with patch("os.getpid", return_value=12345):
            with patch.object(daemon_runner, "_write_pid"):
                with patch("atexit.register"):
                    with patch("signal.signal", side_effect=mock_signal):
                        with patch.object(daemon_runner, "log_message"):
                            daemon_runner.run_server()

        # Check that signal handlers were registered
        signums = [call[0] for call in signal_calls]
        assert signal.SIGTERM in signums
        assert signal.SIGINT in signums

    @patch("local_openai2anthropic.config.get_settings")
    @patch("local_openai2anthropic.main.create_app")
    @patch("uvicorn.run")
    def test_atexit_registered(self, mock_uvicorn_run, mock_create_app, mock_get_settings):
        """Test that atexit handler is registered."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "test-key"
        mock_settings.host = "127.0.0.1"
        mock_settings.port = 9000
        mock_settings.log_level = "INFO"
        mock_settings.openai_base_url = "https://api.openai.com"
        mock_get_settings.return_value = mock_settings

        registered_funcs = []

        def mock_atexit_register(func):
            registered_funcs.append(func)

        with patch("os.getpid", return_value=12345):
            with patch.object(daemon_runner, "_write_pid"):
                with patch("atexit.register", side_effect=mock_atexit_register):
                    with patch("signal.signal"):
                        with patch.object(daemon_runner, "log_message"):
                            daemon_runner.run_server()

        # Check that _remove_pid was registered
        assert daemon_runner._remove_pid in registered_funcs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
