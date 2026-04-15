"""
Tests for model name mapping functionality.
"""

import pytest

from local_openai2anthropic.config import ModelMappingRule, Settings


class TestModelMappingRule:
    """Tests for ModelMappingRule."""

    def test_mapping_rule_from_alias(self):
        """Test that 'from' key works via alias."""
        rule = ModelMappingRule(**{"from": "sonnet", "to": "kimi-k2.5"})
        assert rule.from_ == "sonnet"
        assert rule.to == "kimi-k2.5"

    def test_mapping_rule_direct(self):
        """Test that from_ works directly."""
        rule = ModelMappingRule(from_="opus", to="qwen-max")
        assert rule.from_ == "opus"
        assert rule.to == "qwen-max"


class TestResolveModel:
    """Tests for Settings.resolve_model."""

    def _settings(self, default_model="", model_mapping=None):
        rules = []
        if model_mapping:
            for from_pattern, to_model in model_mapping:
                rules.append(ModelMappingRule(**{"from": from_pattern, "to": to_model}))
        return Settings(default_model=default_model, model_mapping=rules)

    def test_exact_match(self):
        """Request model exactly matches a rule."""
        settings = self._settings(
            model_mapping=[("sonnet", "kimi-k2.5"), ("opus", "qwen-max")]
        )
        assert settings.resolve_model("sonnet") == "kimi-k2.5"
        assert settings.resolve_model("opus") == "qwen-max"

    def test_no_match_uses_default(self):
        """Request doesn't match any rule, falls back to default."""
        settings = self._settings(
            default_model="kimi-k2.5",
            model_mapping=[("sonnet", "model-a")],
        )
        assert settings.resolve_model("haiku") == "kimi-k2.5"

    def test_no_match_no_default_passthrough(self):
        """No rule matches and no default configured -> return original."""
        settings = self._settings(model_mapping=[("sonnet", "kimi-k2.5")])
        assert settings.resolve_model("haiku") == "haiku"

    def test_wildcard_match_prefix(self):
        """Request matches a wildcard rule with prefix."""
        settings = self._settings(
            model_mapping=[("*claude*", "kimi-k2.5")]
        )
        assert settings.resolve_model("claude-sonnet-4-20250514") == "kimi-k2.5"
        assert settings.resolve_model("claude-opus") == "kimi-k2.5"

    def test_wildcard_match_suffix(self):
        """Request matches a wildcard rule with suffix."""
        settings = self._settings(
            model_mapping=[("*opus*", "qwen-max")]
        )
        assert settings.resolve_model("claude-opus-4-20250514") == "qwen-max"

    def test_question_mark_wildcard(self):
        """Request matches single-char wildcard."""
        settings = self._settings(
            model_mapping=[("model-?", "kimi-k2.5")]
        )
        assert settings.resolve_model("model-a") == "kimi-k2.5"
        assert settings.resolve_model("model-ab") == "model-ab"  # doesn't match

    def test_exact_match_priority_over_wildcard(self):
        """Exact match should be checked before wildcard (exact is the first loop)."""
        settings = self._settings(
            default_model="fallback",
            model_mapping=[
                ("sonnet", "exact-hit"),
                ("*", "wildcard-hit"),
            ],
        )
        assert settings.resolve_model("sonnet") == "exact-hit"

    def test_first_wildcard_rule_wins(self):
        """First matching wildcard rule should win."""
        settings = self._settings(
            model_mapping=[
                ("*sonnet*", "sonnet-backend"),
                ("*claude*", "default-backend"),
            ],
        )
        assert settings.resolve_model("claude-sonnet-2025") == "sonnet-backend"

    def test_empty_mapping_returns_original(self):
        """No mapping configured at all -> return original."""
        settings = self._settings()
        assert settings.resolve_model("any-model") == "any-model"

    def test_default_model_only(self):
        """Only default_model configured, no rules."""
        settings = self._settings(default_model="my-default")
        assert settings.resolve_model("anything") == "my-default"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
