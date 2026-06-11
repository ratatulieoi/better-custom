# better-custom

A better way to add custom provider for Pi coding agnet

## Features

- Add or delete custom providers from an interactive wizard
- Supports:
  - Anthropic-compatible endpoints
  - OpenAI-compatible endpoints
  - Ollama-compatible endpoints
- API key modes:
  - literal API key
  - environment variable name (saved as `$NAME`, resolved at runtime)
  - shell command (saved as `!command`, stdout used as the key)
  - none (writes a placeholder so the provider still loads)
- Auto-probe `/models` for OpenAI-compatible endpoints
- Multi-select model picker for probed models
- Image input enabled by default (`input: ["text", "image"]`) so vision-capable
  models receive images instead of having them silently dropped
- Optional `reasoning: true` flag for all saved models
- Safe delete flow for existing providers

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
2. Delete a provider

When adding a provider, it will guide you through:

- provider style
- endpoint
- provider name 
- API key method (literal, env var, shell command, or none)
- model discovery or manual model entry
- reasoning flag

Every model saved by the wizard declares `input: ["text", "image"]`, so
vision-capable models receive image input. pi otherwise defaults custom models
to text-only and drops images before the request.

## Files

- `index.ts` — extension entry point
- `package.json` — pi package manifest

## License

MIT
