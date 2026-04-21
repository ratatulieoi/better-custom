# pi-custom-provider

A shareable [pi](https://github.com/badlogic/pi-mono) package that adds a `/better-custom` command for managing custom providers in `~/.pi/agent/models.json`.

## Features

- Add or delete custom providers from an interactive wizard
- Supports:
  - Anthropic-compatible endpoints
  - OpenAI-compatible endpoints
  - Ollama-compatible endpoints
- API key modes:
  - environment variable name
  - literal API key
  - shell command
  - none
- Auto-probe `/models` for OpenAI-compatible endpoints
- Multi-select model picker for probed models
- Optional `reasoning: true` flag for all saved models
- Safe delete flow for existing providers

## Install

Install directly from GitHub:

```bash
pi install git:github.com/ratatulieoi/pi-custom-provider
```

Or:

```bash
pi install https://github.com/ratatulieoi/pi-custom-provider
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
- provider id
- API key method
- model discovery or manual model entry
- reasoning flag

## Share with others

Anyone can install this package with:

```bash
pi install git:github.com/ratatulieoi/pi-custom-provider
```

## Files

- `index.ts` — extension entry point
- `package.json` — pi package manifest

## License

MIT
