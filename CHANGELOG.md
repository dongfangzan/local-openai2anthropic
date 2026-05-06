# Changelog

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
