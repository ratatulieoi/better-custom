# better-custom

A better way to add custom providers for the Pi coding agent.

## Features

- Add, edit, or delete custom providers from an interactive wizard
- Supports:
  - OpenAI-compatible endpoints
  - Anthropic-compatible endpoints
  - Ollama-compatible endpoints
- API key modes:
  - API key (stored verbatim in `~/.pi/agent/models.json`)
  - none (writes a placeholder so the provider still loads)
  - existing `$ENV` and `!command` keys are still resolved when re-probing
- Auto-probe `/models` for OpenAI-compatible endpoints
- Multi-select model picker for probed models
- Unique provider names — the wizard refuses to overwrite an existing provider
- Image input enabled by default (`input: ["text", "image"]`) so vision-capable
  models receive images instead of having them silently dropped
- Reasoning enabled by default at the `xhigh` ceiling for newly added models
- Safe delete flow for whole providers or individual models

## Install

From npm:

```bash
pi install npm:better-custom
```

From GitHub:

```bash
pi install https://github.com/ratatulieoi/better-custom
```

## Usage

After installing, reload pi if needed, then run:

```text
/better-custom
```

The wizard can:

1. Add a provider
2. Edit a provider
3. Delete a provider

### Add a provider

Guides you through:

- provider style (OpenAI / Anthropic / Ollama)
- endpoint
- provider name (must be unique)
- API key method (API key or none)
- model discovery (auto-probe `/models`) or manual model entry

Newly added models default to `input: ["text", "image"]` and `reasoning: true`
at the `xhigh` ceiling. Tune any of this later via Edit provider.

### Edit a provider

Pick a provider, then choose:

- Re-probe for new models — query `/models` again and add ones not yet configured
- Set context window (all models) — apply one `contextWindow` to every model
- Edit per model — pick a model and edit a single field:
  - Reasoning ceiling (`off` → `xhigh`)
  - Vision (text+image vs text-only)
  - Context window
  - Max output tokens
  - Headers / endpoint override (per-model `baseUrl` and JSON `headers`)
  - Delete this model
- Add models manually

Per-model edits change one field in place, so untouched fields (cost, headers,
overrides) are preserved.

### Delete a provider

Lists configured providers and removes the selected one after confirmation.

## How reasoning maps to pi

pi exposes six thinking levels: `off, minimal, low, medium, high, xhigh`. When a
model has `reasoning: true`, pi treats `minimal` through `high` as available.
`xhigh` is opt-in and only unlocked when explicitly mapped, and any level set to
`null` is removed. The wizard writes a `thinkingLevelMap` to unlock `xhigh` or to
cap reasoning below `high`.

## Files

- `index.ts` — extension entry point
- `package.json` — pi package manifest

## License

MIT
