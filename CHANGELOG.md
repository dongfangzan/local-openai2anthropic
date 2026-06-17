# Changelog

## [0.6.7] - 2026-06-17

### Added

- **GLM (Zhipu/Z.ai) reasoning_effort semantic mapping.** SGLang-served GLM-4.5/4.6/4.7/5.x chat templates only wire two effective reasoning levels — `High` (dials reasoning down) and `Max` (highest, the default) — via a `Reasoning Effort: <level>` system line, and any value other than `"high"` falls through to `Max`. Forwarding Anthropic effort tiers verbatim therefore inverts the intent (e.g. Anthropic `low` → GLM `Max`). The converter now detects GLM models by name and maps:
  - Anthropic `low` / `medium` (want less reasoning) → GLM `"high"` (GLM's low band)
  - Anthropic `high` / `xhigh` / `max` / unset (want more reasoning) → unset → GLM `Max`

### Changed

- DeepSeek V4 keeps its existing verbatim effort forwarding (defaulting to `"high"`); only GLM takes the two-band remap.

### Fixed

- Corrected the `--version` flag string, which had fallen behind at `0.6.5`.

---

## [0.6.5] - 2026-05-06

### Added

- **OpenAI-native passthrough endpoint** (`POST /v1/chat/completions`): Proxy OpenAI-format requests directly to the upstream API without any conversion. This allows clients that already speak the OpenAI protocol to bypass the Anthropic to OpenAI conversion layer, reducing overhead and preserving all upstream-specific fields (e.g., `logprobs`, `token_ids`, `refusal`, `annotations`).
  - Non-streaming and streaming (SSE) modes both supported.
  - All request fields are forwarded as-is — no validation, no filtering, no model name mapping.
  - Error handling: timeout (504), connection error (502), invalid JSON (400).

### Changed

- None.

### Fixed

- None.

---

## [0.6.4] - 2026-04-21

### Added

- Anthropic `output_config.effort` to `reasoning_effort` mapping for DeepSeek V4.
- `preserve_thinking` in `chat_template_kwargs` when thinking mode is enabled.
- Model name mapping with wildcard support.

### Fixed

- Support both `reasoning` and `reasoning_content` in streaming responses.
