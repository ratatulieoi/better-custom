import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Key, matchesKey, truncateToWidth } from "@mariozechner/pi-tui";
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname } from "node:path";

type ProviderApi = "openai-completions" | "anthropic-messages";
type ProviderStyle = "openai" | "anthropic" | "ollama";
type ApiKeyMode = "env" | "literal" | "shell" | "none";

type ModelsConfig = {
	providers?: Record<string, any>;
};

type ProbeItem = {
	value: string;
	label: string;
	description?: string;
};

type SelectItem = {
	value: string;
	label: string;
	suffix?: string;
	description?: string;
	searchText?: string;
};

type CommandContext = Parameters<Parameters<ExtensionAPI["registerCommand"]>[1]["handler"]>[1];

const MODELS_JSON_PATH = `${homedir()}/.pi/agent/models.json`;
const BUILTIN_PROVIDER_IDS = new Set([
	"anthropic",
	"openai",
	"azure-openai",
	"google",
	"vertex",
	"bedrock",
	"mistral",
	"groq",
	"cerebras",
	"xai",
	"openrouter",
	"vercel-ai-gateway",
	"zai",
	"huggingface",
	"kimi-for-coding",
	"minimax",
	"ollama",
]);

function ensureConfigDir() {
	mkdirSync(dirname(MODELS_JSON_PATH), { recursive: true });
}

function loadModelsConfig(): ModelsConfig {
	ensureConfigDir();
	if (!existsSync(MODELS_JSON_PATH)) {
		return { providers: {} };
	}

	const raw = readFileSync(MODELS_JSON_PATH, "utf8").trim();
	if (!raw) return { providers: {} };

	const parsed = JSON.parse(raw) as ModelsConfig;
	if (!parsed.providers || typeof parsed.providers !== "object") {
		parsed.providers = {};
	}
	return parsed;
}

function saveModelsConfig(config: ModelsConfig) {
	ensureConfigDir();
	writeFileSync(MODELS_JSON_PATH, `${JSON.stringify(config, null, 2)}\n`, "utf8");
}

function addDefaultScheme(input: string): string {
	if (/^[a-z]+:\/\//i.test(input)) return input;
	const lower = input.toLowerCase();
	const isLocal =
		lower.startsWith("localhost") ||
		lower.startsWith("127.") ||
		lower.startsWith("0.0.0.0") ||
		lower.startsWith("10.") ||
		lower.startsWith("192.168.") ||
		lower.startsWith("172.16.") ||
		lower.startsWith("172.17.") ||
		lower.startsWith("172.18.") ||
		lower.startsWith("172.19.") ||
		lower.startsWith("172.20.") ||
		lower.startsWith("172.21.") ||
		lower.startsWith("172.22.") ||
		lower.startsWith("172.23.") ||
		lower.startsWith("172.24.") ||
		lower.startsWith("172.25.") ||
		lower.startsWith("172.26.") ||
		lower.startsWith("172.27.") ||
		lower.startsWith("172.28.") ||
		lower.startsWith("172.29.") ||
		lower.startsWith("172.30.") ||
		lower.startsWith("172.31.") ||
		lower.startsWith("[");
	return `${isLocal ? "http" : "https"}://${input}`;
}

function stripSuffix(pathname: string, suffix: string): string {
	return pathname.endsWith(suffix) ? pathname.slice(0, -suffix.length) || "/" : pathname;
}

function normalizeEndpoint(input: string, api: ProviderApi): string {
	const url = new URL(addDefaultScheme(input.trim()));
	let pathname = url.pathname.replace(/\/+$/, "") || "/";

	if (api === "openai-completions") {
		pathname = stripSuffix(pathname, "/chat/completions");
		pathname = stripSuffix(pathname, "/responses");
		pathname = stripSuffix(pathname, "/completions");
		pathname = stripSuffix(pathname, "/models");
	} else {
		pathname = stripSuffix(pathname, "/messages");
	}

	pathname = pathname === "/" ? "" : pathname;
	const port = url.port ? `:${url.port}` : "";
	return `${url.protocol}//${url.hostname}${port}${pathname}`;
}

function slugify(value: string): string {
	return value
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.replace(/--+/g, "-");
}

function suggestProviderId(endpoint: string): string {
	const url = new URL(addDefaultScheme(endpoint));
	const host = url.hostname.replace(/^www\./, "").replace(/^api\./, "");
	const hostSlug = slugify(`${host}${url.port ? `-${url.port}` : ""}`) || "provider";
	return `custom-${hostSlug}`;
}

function dedupe(values: string[]): string[] {
	return Array.from(new Set(values));
}

function buildProbeUrl(baseUrl: string): string {
	const withSlash = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
	return new URL("models", withSlash).toString();
}

function resolveApiKeyForProbe(mode: ApiKeyMode, storedValue?: string): string | undefined {
	if (!storedValue || mode === "none") return undefined;
	if (mode === "literal") return storedValue;
	if (mode === "env") return process.env[storedValue]?.trim() || undefined;
	if (mode === "shell") {
		try {
			return execSync(storedValue, {
				encoding: "utf8",
				stdio: ["ignore", "pipe", "pipe"],
			}).trim();
		} catch {
			return undefined;
		}
	}
	return undefined;
}

function serializeApiKey(mode: ApiKeyMode, value?: string, style?: ProviderStyle): string | undefined {
	if (mode === "none") return style === "ollama" ? "ollama" : "dummy";
	if (!value) return undefined;
	if (mode === "shell") return value.startsWith("!") ? value : `!${value}`;
	return value;
}

async function probeOpenAIModels(baseUrl: string, apiKeyMode: ApiKeyMode, apiKeyValue?: string): Promise<ProbeItem[]> {
	const headers: Record<string, string> = {
		accept: "application/json",
	};
	const resolvedKey = resolveApiKeyForProbe(apiKeyMode, apiKeyValue);
	if (resolvedKey) {
		headers.authorization = `Bearer ${resolvedKey}`;
	}

	const response = await fetch(buildProbeUrl(baseUrl), { headers });
	if (!response.ok) {
		const body = await response.text().catch(() => "");
		throw new Error(`Probe failed (${response.status} ${response.statusText})${body ? `: ${body.slice(0, 200)}` : ""}`);
	}

	const json = (await response.json()) as any;
	const rawModels = Array.isArray(json) ? json : Array.isArray(json?.data) ? json.data : [];
	const ids = dedupe(
		rawModels
			.map((item: any) => (typeof item?.id === "string" ? item.id.trim() : ""))
			.filter(Boolean),
	).sort((a, b) => a.localeCompare(b));

	return ids.map((id) => ({ value: id, label: id }));
}

function normalizeSelectItems(items: Array<string | SelectItem>): SelectItem[] {
	return items.map((item) => (typeof item === "string" ? { value: item, label: item } : item));
}

async function selectOne(
	ctx: CommandContext,
	title: string,
	items: Array<string | SelectItem>,
	options?: { initialIndex?: number },
): Promise<string | null> {
	const normalizedItems = normalizeSelectItems(items);
	if (normalizedItems.length === 0) return null;

	return await ctx.ui.custom<string | null>((tui, theme, _kb, done) => {
		let cursor = Math.max(0, Math.min(options?.initialIndex ?? 0, normalizedItems.length - 1));
		let query = "";
		let cachedLines: string[] | undefined;
		const maxVisible = 12;

		function getVisibleItems() {
			const lowerQuery = query.trim().toLowerCase();
			if (!lowerQuery) return normalizedItems;
			return normalizedItems.filter((item) => {
				const haystack = `${item.label} ${item.suffix ?? ""} ${item.description ?? ""} ${item.searchText ?? ""}`.toLowerCase();
				return haystack.includes(lowerQuery);
			});
		}

		function refresh() {
			const visibleItems = getVisibleItems();
			if (visibleItems.length === 0) cursor = 0;
			else if (cursor >= visibleItems.length) cursor = visibleItems.length - 1;
			cachedLines = undefined;
			tui.requestRender();
		}

		return {
			render(width: number) {
				if (cachedLines) return cachedLines;

				const visibleItems = getVisibleItems();
				const safeWidth = Math.max(10, width);
				const lines: string[] = [];
				const add = (line = "") => lines.push(truncateToWidth(line, safeWidth));
				const border = theme.fg("accent", "─".repeat(safeWidth));

				add(border);
				add(` ${theme.fg("accent", theme.bold(title))}`);
				add(` ${theme.fg("text", `Search: ${query || "-"}`)}`);
				add();

				if (visibleItems.length === 0) {
					add(theme.fg("warning", " No matches."));
				} else {
					const start = Math.max(0, Math.min(cursor - Math.floor(maxVisible / 2), Math.max(0, visibleItems.length - maxVisible)));
					const end = Math.min(visibleItems.length, start + maxVisible);

					for (let i = start; i < end; i++) {
						const item = visibleItems[i];
						const active = i === cursor;
						const prefix = active ? theme.fg("accent", "> ") : "  ";
						const label = active ? theme.fg("accent", item.label) : theme.fg("text", item.label);
						const suffix = item.suffix ? theme.fg("dim", item.suffix) : "";
						add(`${prefix}${label}${suffix}`);
						if (item.description) {
							for (const line of item.description.split("\n")) {
								add(`   ${theme.fg("muted", line)}`);
							}
						}
					}

					if (visibleItems.length > maxVisible) {
						add();
						add(theme.fg("dim", ` ${start + 1}-${end} of ${visibleItems.length}`));
					}
				}

				add();
				add(theme.fg("dim", " Type to search • ↑↓ move (wraps) • enter confirm • backspace delete • esc cancel"));
				add(border);

				cachedLines = lines;
				return lines;
			},
			invalidate() {
				cachedLines = undefined;
			},
			handleInput(data: string) {
				const visibleItems = getVisibleItems();
				if (matchesKey(data, Key.up)) {
					if (visibleItems.length === 0) return;
					cursor = cursor === 0 ? visibleItems.length - 1 : cursor - 1;
					refresh();
					return;
				}
				if (matchesKey(data, Key.down)) {
					if (visibleItems.length === 0) return;
					cursor = cursor === visibleItems.length - 1 ? 0 : cursor + 1;
					refresh();
					return;
				}
				if (matchesKey(data, Key.enter)) {
					const item = visibleItems[cursor];
					done(item?.value ?? null);
					return;
				}
				if (matchesKey(data, Key.escape)) {
					done(null);
					return;
				}
				if (data === "\u007f" || data === "\b") {
					if (query.length > 0) {
						query = query.slice(0, -1);
						refresh();
					}
					return;
				}
				if (data >= " " && data !== "\u001b" && data !== "\r" && data !== "\n") {
					query += data;
					cursor = 0;
					refresh();
				}
			},
		};
	});
}

async function pickMany(
	ctx: CommandContext,
	title: string,
	items: ProbeItem[],
): Promise<string[] | null> {
	return await ctx.ui.custom<string[] | null>((tui, theme, _kb, done) => {
		let cursor = 0;
		const selected = new Set<string>();
		let cachedLines: string[] | undefined;
		const maxVisible = 12;

		function refresh() {
			cachedLines = undefined;
			tui.requestRender();
		}

		return {
			render(width: number) {
				if (cachedLines) return cachedLines;

				const safeWidth = Math.max(10, width);
				const lines: string[] = [];
				const add = (line = "") => lines.push(truncateToWidth(line, safeWidth));
				const border = theme.fg("accent", "─".repeat(safeWidth));

				add(border);
				add(` ${theme.fg("accent", theme.bold(title))}`);
				add(` ${theme.fg("muted", `${selected.size} selected`)}`);
				add();

				const start = Math.max(0, Math.min(cursor - Math.floor(maxVisible / 2), Math.max(0, items.length - maxVisible)));
				const end = Math.min(items.length, start + maxVisible);

				for (let i = start; i < end; i++) {
					const item = items[i];
					const active = i === cursor;
					const checked = selected.has(item.value);
					const prefix = active ? theme.fg("accent", "> ") : "  ";
					const box = checked ? theme.fg("success", "[x]") : theme.fg("muted", "[ ]");
					const label = active ? theme.fg("accent", item.label) : theme.fg("text", item.label);
					add(`${prefix}${box} ${label}`);
					if (item.description) {
						add(`     ${theme.fg("muted", item.description)}`);
					}
				}

				if (items.length > maxVisible) {
					add();
					add(theme.fg("dim", ` ${start + 1}-${end} of ${items.length}`));
				}

				add();
				add(theme.fg("dim", " ↑↓ move (wraps) • space toggle • enter confirm • esc cancel"));
				if (selected.size === 0) {
					add(theme.fg("warning", " Select at least one model before confirming."));
				}
				add(border);

				cachedLines = lines;
				return lines;
			},
			invalidate() {
				cachedLines = undefined;
			},
			handleInput(data: string) {
				if (matchesKey(data, Key.up)) {
					cursor = cursor === 0 ? items.length - 1 : cursor - 1;
					refresh();
					return;
				}
				if (matchesKey(data, Key.down)) {
					cursor = cursor === items.length - 1 ? 0 : cursor + 1;
					refresh();
					return;
				}
				if (data === " ") {
					const value = items[cursor]?.value;
					if (!value) return;
					if (selected.has(value)) selected.delete(value);
					else selected.add(value);
					refresh();
					return;
				}
				if (matchesKey(data, Key.enter)) {
					if (selected.size > 0) done(Array.from(selected));
					return;
				}
				if (matchesKey(data, Key.escape)) {
					done(null);
				}
			},
		};
	});
}

async function promptApiKey(
	ctx: CommandContext,
): Promise<{ mode: ApiKeyMode; value?: string } | null> {
	const choice = await selectOne(ctx, "API key", [
		"Environment variable name",
		"Literal API key",
		"Shell command",
		"None",
	]);
	if (!choice) return null;

	if (choice === "None") return { mode: "none" };

	if (choice === "Environment variable name") {
		const value = await ctx.ui.input("Environment variable name", "e.g. OPENAI_API_KEY");
		if (value === undefined) return null;
		const trimmed = value.trim();
		if (!trimmed) return { mode: "none" };
		return { mode: "env", value: trimmed };
	}

	if (choice === "Literal API key") {
		const value = await ctx.ui.input("Literal API key", "saved directly in ~/.pi/agent/models.json");
		if (value === undefined) return null;
		const trimmed = value.trim();
		if (!trimmed) return { mode: "none" };
		return { mode: "literal", value: trimmed };
	}

	const value = await ctx.ui.input("Shell command", "e.g. op read 'op://vault/item/credential'");
	if (value === undefined) return null;
	const trimmed = value.trim();
	if (!trimmed) return { mode: "none" };
	return { mode: "shell", value: trimmed };
}

async function promptModelIdsOneByOne(
	ctx: CommandContext,
	style: ProviderStyle,
	api: ProviderApi,
): Promise<string[] | null> {
	const modelIds: string[] = [];
	const firstPlaceholder =
		style === "anthropic"
			? "e.g. claude-sonnet-4-5 (blank to finish)"
			: style === "ollama"
				? "e.g. llama3.1:8b or qwen2.5-coder:7b (blank to finish)"
				: "e.g. gpt-4o-mini or qwen/qwen3-coder (blank to finish)";
	const nextPlaceholder =
		style === "anthropic"
			? "another Anthropic-style model id (blank to finish)"
			: style === "ollama"
				? "another Ollama model id (blank to finish)"
				: "another OpenAI-style model id (blank to finish)";

	while (true) {
		const value = await ctx.ui.input(modelIds.length === 0 ? "Model id" : "Add another model id", modelIds.length === 0 ? firstPlaceholder : nextPlaceholder);
		if (value === undefined) return null;
		const trimmed = value.trim();
		if (!trimmed) {
			if (modelIds.length === 0) {
				ctx.ui.notify("Add at least one model.", "warning");
				continue;
			}
			return modelIds;
		}
		if (modelIds.includes(trimmed)) {
			ctx.ui.notify(`Model already added: ${trimmed}`, "warning");
			continue;
		}
		modelIds.push(trimmed);
	}
}

function buildProviderConfig(
	style: ProviderStyle,
	api: ProviderApi,
	baseUrl: string,
	apiKey: { mode: ApiKeyMode; value?: string },
	modelIds: string[],
	reasoning: boolean,
) {
	const serializedApiKey = serializeApiKey(apiKey.mode, apiKey.value, style);
	const config: any = {
		baseUrl,
		api,
		...(serializedApiKey ? { apiKey: serializedApiKey } : {}),
		models: modelIds.map((id) => ({
			id,
			...(reasoning ? { reasoning: true } : {}),
		})),
	};

	if (style === "ollama") {
		if (!config.apiKey) config.apiKey = "ollama";
		config.compat = {
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
		};
	}

	return config;
}

function describeProvider(providerId: string, provider: any): string {
	const modelCount = Array.isArray(provider?.models) ? provider.models.length : 0;
	const endpoint = typeof provider?.baseUrl === "string" ? provider.baseUrl : "(no baseUrl)";
	const api = typeof provider?.api === "string" ? provider.api : "(no api)";
	return `${providerId}\n${api} • ${modelCount} model${modelCount === 1 ? "" : "s"}\n${endpoint}`;
}

function describeProviderInline(providerId: string, provider: any): { label: string; suffix: string; searchText: string } {
	const modelCount = Array.isArray(provider?.models) ? provider.models.length : 0;
	const endpoint = typeof provider?.baseUrl === "string" ? provider.baseUrl : "(no baseUrl)";
	const api = typeof provider?.api === "string" ? provider.api : "(no api)";
	const suffix = ` • ${api} • ${endpoint} • ${modelCount} model${modelCount === 1 ? "" : "s"}`;
	return {
		label: providerId,
		suffix,
		searchText: `${providerId} ${api} ${endpoint} ${modelCount}`,
	};
}

async function deleteProviderFlow(ctx: CommandContext) {
	let cursor = 0;
	let deletedAny = false;

	while (true) {
		let config: ModelsConfig;
		try {
			config = loadModelsConfig();
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}

		config.providers ||= {};
		const providerIds = Object.keys(config.providers).sort((a, b) => a.localeCompare(b));
		if (providerIds.length === 0) {
			ctx.ui.notify(
				deletedAny ? `No providers left in ${MODELS_JSON_PATH}` : `No providers found in ${MODELS_JSON_PATH}`,
				deletedAny ? "info" : "warning",
			);
			return;
		}

		const choice = await selectOne(
			ctx,
			"Delete provider",
			providerIds.map((providerId) => {
				const inline = describeProviderInline(providerId, config.providers?.[providerId]);
				return {
					value: providerId,
					label: inline.label,
					suffix: inline.suffix,
					searchText: inline.searchText,
				};
			}),
			{ initialIndex: Math.min(cursor, providerIds.length - 1) },
		);
		if (!choice) return;

		const provider = config.providers[choice];
		const confirmed = await ctx.ui.confirm("Delete provider?", describeProvider(choice, provider));
		const selectedIndex = providerIds.indexOf(choice);
		cursor = selectedIndex;
		if (!confirmed) continue;

		cursor = selectedIndex + 1;
		delete config.providers[choice];

		try {
			saveModelsConfig(config);
		} catch (error) {
			ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}

		deletedAny = true;
		ctx.ui.notify(`Deleted provider \"${choice}\" from ${MODELS_JSON_PATH}`, "info");
	}
}

export default function betterCustomWizard(pi: ExtensionAPI) {
	pi.registerCommand("better-custom", {
		description: "Wizard for adding or deleting custom providers in ~/.pi/agent/models.json",
		handler: async (_args, ctx) => {
			const action = await selectOne(ctx, "Better custom", ["Add provider", "Delete provider"]);
			if (!action) return;
			if (action === "Delete provider") {
				await deleteProviderFlow(ctx);
				return;
			}

			const providerStyleLabel = await selectOne(ctx, "Provider style", [
				"Anthropic-compatible",
				"OpenAI-compatible",
				"Ollama-compatible",
			]);
			if (!providerStyleLabel) return;

			const style: ProviderStyle =
				providerStyleLabel === "Anthropic-compatible"
					? "anthropic"
					: providerStyleLabel === "Ollama-compatible"
						? "ollama"
						: "openai";
			const api: ProviderApi = style === "anthropic" ? "anthropic-messages" : "openai-completions";

			const endpointInput = await ctx.ui.input(
				"Endpoint",
				style === "anthropic"
					? "e.g. https://api.anthropic-proxy.com/v1"
					: style === "ollama"
						? "e.g. http://localhost:11434/v1"
						: "e.g. https://api.example.com/v1 or http://localhost:11434/v1",
			);
			if (endpointInput === undefined) return;
			const trimmedEndpointInput = endpointInput.trim();
			if (!trimmedEndpointInput) {
				ctx.ui.notify("Endpoint is required.", "error");
				return;
			}

			let normalizedEndpoint: string;
			try {
				normalizedEndpoint = normalizeEndpoint(trimmedEndpointInput, api);
			} catch (error) {
				ctx.ui.notify(`Invalid endpoint: ${error instanceof Error ? error.message : String(error)}`, "error");
				return;
			}

			const providerIdSuggestion = suggestProviderId(normalizedEndpoint);
			const providerNameInput = await ctx.ui.input(
				`Provider name (blank = ${providerIdSuggestion})`,
				"e.g. custom-example-com",
			);
			if (providerNameInput === undefined) return;
			const providerId = slugify(providerNameInput.trim() || providerIdSuggestion);
			if (!providerId) {
				ctx.ui.notify("Provider name is required.", "error");
				return;
			}

			if (BUILTIN_PROVIDER_IDS.has(providerId)) {
				const ok = await ctx.ui.confirm(
					"Override built-in provider?",
					`"${providerId}" matches a built-in provider id. Saving this will override that provider in ~/.pi/agent/models.json. Continue?`,
				);
				if (!ok) return;
			}

			const apiKey = await promptApiKey(ctx);
			if (!apiKey) return;
			if (apiKey.mode === "none") {
				ctx.ui.notify(
					style === "ollama"
						? 'No API key selected. Using "ollama" automatically in models.json.'
						: 'No API key selected. Using "dummy" automatically in models.json.',
					"info",
				);
			}

			let modelIds: string[] | null = null;
			if (api === "openai-completions") {
				const modelMode = await selectOne(ctx, "Models", ["Auto probe from /models", "Add manually"]);
				if (!modelMode) return;

				if (modelMode === "Auto probe from /models") {
					try {
						ctx.ui.notify(`Probing ${buildProbeUrl(normalizedEndpoint)} ...`, "info");
						const probedModels = await probeOpenAIModels(normalizedEndpoint, apiKey.mode, apiKey.value);
						if (probedModels.length === 0) {
							ctx.ui.notify("Probe succeeded but returned no models. Switching to manual entry.", "warning");
							modelIds = await promptModelIdsOneByOne(ctx, style, api);
						} else {
							modelIds = await pickMany(ctx, "Select models", probedModels);
						}
					} catch (error) {
						ctx.ui.notify(
							`Auto probe failed: ${error instanceof Error ? error.message : String(error)}. Switching to manual entry.`,
							"warning",
						);
						modelIds = await promptModelIdsOneByOne(ctx, style, api);
					}
				} else {
					modelIds = await promptModelIdsOneByOne(ctx, style, api);
				}
			} else {
				modelIds = await promptModelIdsOneByOne(ctx, style, api);
			}

			if (!modelIds || modelIds.length === 0) return;

			const reasoningChoice = await selectOne(ctx, "Reasoning", [
				"Yes - set reasoning=true for all models",
				"No - leave reasoning unset",
			]);
			if (!reasoningChoice) return;
			const reasoning = reasoningChoice.startsWith("Yes");

			let config: ModelsConfig;
			try {
				config = loadModelsConfig();
			} catch (error) {
				ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
				return;
			}

			config.providers ||= {};
			if (config.providers[providerId]) {
				const replace = await ctx.ui.confirm(
					"Replace existing provider?",
					`Provider "${providerId}" already exists in ${MODELS_JSON_PATH}. Replace it?`,
				);
				if (!replace) return;
			}

			config.providers[providerId] = buildProviderConfig(style, api, normalizedEndpoint, apiKey, dedupe(modelIds), reasoning);

			try {
				saveModelsConfig(config);
			} catch (error) {
				ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
				return;
			}

			ctx.ui.notify(`Saved provider \"${providerId}\" to ${MODELS_JSON_PATH}`, "info");
			ctx.ui.notify("Open /model to use your new provider.", "info");
		},
	});
}
