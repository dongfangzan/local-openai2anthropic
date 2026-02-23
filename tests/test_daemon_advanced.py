"""
Advanced tests for daemon module covering subprocess and edge cases.
"""

import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from local_openai2anthropic import daemon


class TestStartDaemonAdvanced:
    """Advanced tests for start_daemon function."""

    def test_start_daemon_success(self):
        """Test successful daemon start."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll = MagicMock(return_value=None)

        with patch.object(daemon, "_cleanup_stale_pidfile"):
            with patch.object(daemon, "_read_pid", return_value=None):
                with patch.object(daemon, "_is_port_in_use", return_value=False):
                    with patch.object(daemon, "_ensure_dirs"):
                        with patch("subprocess.Popen", return_value=mock_process):
                            with patch("time.sleep"):
                                with patch.object(daemon, "_save_daemon_config") as mock_save:
                                    with patch.object(daemon, "_is_port_in_use", side_effect=[False, True]):
                                        result = daemon.start_daemon("127.0.0.1", 8080, "info")
                                        assert result is True
                                        mock_save.assert_called_once()

    def test_start_daemon_process_exits_immediately(self):
        """Test when daemon process exits immediately."""
        mock_process = MagicMock()
        mock_process.poll = MagicMock(return_value=1)  # Process exited

        with patch.object(daemon, "_cleanup_stale_pidfile"):
            with patch.object(daemon, "_read_pid", return_value=None):
                with patch.object(daemon, "_is_port_in_use", return_value=False):
                    with patch.object(daemon, "_ensure_dirs"):
                        with patch("subprocess.Popen", return_value=mock_process):
                            with patch("time.sleep"):
                                with patch("builtins.open", mock_open()):
                                    result = daemon.start_daemon()
                                    assert result is False

    def test_start_daemon_port_never_active(self):
        """Test when port never becomes active and process dies."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        # First poll returns None (running), second poll returns exit code (died)
        mock_process.poll = MagicMock(side_effect=[None, 1])

        with patch.object(daemon, "_cleanup_stale_pidfile"):
            with patch.object(daemon, "_read_pid", return_value=None):
                with patch.object(daemon, "_is_port_in_use", return_value=False):
                    with patch.object(daemon, "_ensure_dirs"):
                        with patch("subprocess.Popen", return_value=mock_process):
                            with patch("time.sleep"):
                                with patch("builtins.open", mock_open()):
                                    result = daemon.start_daemon()
                                    assert result is False

    def test_start_daemon_exception(self):
        """Test exception handling during start - exception in subprocess.Popen."""
        with patch.object(daemon, "_cleanup_stale_pidfile"):
            with patch.object(daemon, "_read_pid", return_value=None):
                with patch.object(daemon, "_is_port_in_use", return_value=False):
                    with patch.object(daemon, "_ensure_dirs"):
                        # Patch subprocess.Popen to raise exception (inside try block)
                        with patch("subprocess.Popen", side_effect=Exception("Failed to start")):
                            with patch("builtins.open", mock_open()):
                                result = daemon.start_daemon()
                                assert result is False

    def test_start_daemon_setsid_on_unix(self):
        """Test that setsid is used on Unix platforms."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll = MagicMock(return_value=None)

        with patch.object(daemon, "_cleanup_stale_pidfile"):
            with patch.object(daemon, "_read_pid", return_value=None):
                with patch.object(daemon, "_is_port_in_use", return_value=False):
                    with patch.object(daemon, "_ensure_dirs"):
                        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
                            with patch("time.sleep"):
                                with patch.object(daemon, "_save_daemon_config"):
                                    with patch.object(daemon, "_is_port_in_use", side_effect=[False, True]):
                                        with patch.object(sys, "platform", "linux"):
                                            daemon.start_daemon("127.0.0.1", 8080, "info")
                                            # Check that start_new_session was passed
                                            call_kwargs = mock_popen.call_args[1]
                                            assert call_kwargs.get("start_new_session") is True

    def test_start_daemon_no_setsid_on_windows(self):
        """Test that setsid is not used on Windows."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll = MagicMock(return_value=None)

        with patch.object(daemon, "_cleanup_stale_pidfile"):
            with patch.object(daemon, "_read_pid", return_value=None):
                with patch.object(daemon, "_is_port_in_use", return_value=False):
                    with patch.object(daemon, "_ensure_dirs"):
                        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
                            with patch("time.sleep"):
                                with patch.object(daemon, "_save_daemon_config"):
                                    with patch.object(daemon, "_is_port_in_use", side_effect=[False, True]):
                                        with patch.object(sys, "platform", "win32"):
                                            daemon.start_daemon("127.0.0.1", 8080, "info")
                                            # Check that start_new_session was not passed or is False
                                            call_kwargs = mock_popen.call_args[1]
                                            assert call_kwargs.get("start_new_session") is not True


class TestStopDaemonAdvanced:
    """Advanced tests for stop_daemon function."""

    def test_stop_daemon_force_kill_after_timeout(self):
        """Test force kill when graceful stop times out."""
        with patch.object(daemon, "_cleanup_stale_pidfile"):
            with patch.object(daemon, "_read_pid", return_value=12345):
                # Process keeps running
                with patch.object(daemon, "_is_process_running", return_value=True):
                    with patch("os.kill") as mock_kill:
                        with patch.object(daemon, "_remove_pid"):
                            with patch.object(daemon, "_remove_daemon_config"):
                                result = daemon.stop_daemon(force=True)
                                assert result is True
                                # Should be called with SIGKILL
                                mock_kill.assert_called_with(12345, signal.SIGKILL)

    def test_stop_daemon_force_flag_immediate(self):
        """Test immediate SIGKILL with force flag."""
        # Test that force=True uses SIGKILL - simplified test
        with patch.object(daemon, "_cleanup_stale_pidfile"):
            with patch.object(daemon, "_read_pid", return_value=12345):
                with patch.object(daemon, "_read_port", return_value=None):
                    # First call returns True (process running), second returns False (exits loop)
                    # Third call returns False (final check)
                    with patch.object(daemon, "_is_process_running", side_effect=[True, False, False]):
                        with patch("os.kill") as mock_kill:
                            with patch.object(daemon, "_remove_pid"):
                                with patch.object(daemon, "_remove_daemon_config"):
                                    result = daemon.stop_daemon(force=True)
                                    assert result is True
                                    # Verify SIGKILL was used (not SIGTERM)
                                    calls = mock_kill.call_args_list
                                    assert len(calls) >= 1
                                    assert calls[0][0][1] == signal.SIGKILL

    def test_stop_daemon_oserror(self):
        """Test stop_daemon with OSError."""
        with patch.object(daemon, "_cleanup_stale_pidfile"):
            with patch.object(daemon, "_read_pid", return_value=12345):
                with patch("os.kill", side_effect=OSError("Permission denied")):
                    with patch.object(daemon, "_remove_pid"):
                        with patch.object(daemon, "_remove_daemon_config"):
                            result = daemon.stop_daemon()
                            assert result is True

    def test_stop_daemon_generic_exception(self):
        """Test stop_daemon with generic exception."""
        with patch.object(daemon, "_cleanup_stale_pidfile"):
            with patch.object(daemon, "_read_pid", return_value=12345):
                with patch.object(daemon, "_read_port", return_value=None):
                    with patch.object(daemon, "_is_process_running", return_value=True):
                        with patch("os.kill", side_effect=Exception("Unexpected")):
                            result = daemon.stop_daemon()
                            assert result is False


class TestShowLogsAdvanced:
    """Advanced tests for show_logs function."""

    def test_show_logs_follow_mode(self):
        """Test show_logs with follow mode."""
        with patch.object(Path, "exists", return_value=True):
            with patch("subprocess.run") as mock_run:
                result = daemon.show_logs(follow=True, lines=50)
                assert result is True
                mock_run.assert_called_once()
                # Check that tail -f was called
                call_args = mock_run.call_args[0][0]
                assert "tail" in call_args
                assert "-f" in call_args

    def test_show_logs_follow_keyboard_interrupt(self):
        """Test show_logs follow mode with keyboard interrupt."""
        with patch.object(Path, "exists", return_value=True):
            with patch("subprocess.run", side_effect=KeyboardInterrupt):
                result = daemon.show_logs(follow=True)
                assert result is True  # Should return True after interrupt


class TestRunForeground:
    """Tests for run_foreground function."""

    def test_run_foreground(self):
        """Test running in foreground mode."""
        mock_settings = MagicMock()
        mock_settings.openai_base_url = "https://api.openai.com/v1"

        mock_app = MagicMock()

        with patch.dict("os.environ", {}, clear=True):
            with patch("local_openai2anthropic.main.create_app", return_value=mock_app):
                with patch("local_openai2anthropic.config.get_settings", return_value=mock_settings):
                    with patch("uvicorn.run") as mock_uvicorn:
                        daemon.run_foreground("127.0.0.1", 8080, "info")
                        mock_uvicorn.assert_called_once()
                        # Check environment variables were set
                        assert "127.0.0.1" == daemon.os.environ.get("OA2A_HOST")
                        assert "8080" == daemon.os.environ.get("OA2A_PORT")


class TestRemoveDaemonConfig:
    """Tests for _remove_daemon_config function."""

    def test_remove_existing_config(self):
        """Test removing existing config file."""
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "unlink") as mock_unlink:
                daemon._remove_daemon_config()
                mock_unlink.assert_called_once()

    def test_remove_missing_config(self):
        """Test removing non-existent config file."""
        with patch.object(Path, "exists", return_value=False):
            with patch.object(Path, "unlink") as mock_unlink:
                daemon._remove_daemon_config()
                mock_unlink.assert_not_called()

    def test_remove_config_oserror(self):
        """Test handling OSError when removing config."""
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "unlink", side_effect=OSError):
                # Should not raise
                daemon._remove_daemon_config()


class TestSaveDaemonConfigEdgeCases:
    """Edge case tests for _save_daemon_config."""

    def test_save_config_oserror(self):
        """Test handling OSError when saving config."""
        with patch.object(daemon, "_ensure_dirs"):
            with patch.object(Path, "write_text", side_effect=OSError):
                # Should not raise
                daemon._save_daemon_config("127.0.0.1", 8080)


class TestLoadDaemonConfigEdgeCases:
    """Edge case tests for _load_daemon_config."""

    def test_load_config_oserror(self):
        """Test handling OSError when loading config."""
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", side_effect=OSError):
                result = daemon._load_daemon_config()
                assert result is None


class TestRemovePidEdgeCases:
    """Edge case tests for _remove_pid."""

    def test_remove_pid_oserror(self):
        """Test handling OSError when removing PID."""
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "unlink", side_effect=OSError):
                # Should not raise
                daemon._remove_pid()


class TestEnsureDirsEdgeCases:
    """Edge case tests for _ensure_dirs."""

    def test_ensure_dirs_existing(self):
        """Test ensuring dirs when they already exist."""
        with patch.object(Path, "mkdir") as mock_mkdir:
            with patch.object(Path, "exists", return_value=True):
                daemon._ensure_dirs()
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
