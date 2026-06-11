import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Key, matchesKey, truncateToWidth } from "@mariozechner/pi-tui";
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname } from "node:path";

type ProviderApi = "openai-completions" | "anthropic-messages";
type ProviderStyle = "openai" | "anthropic" | "ollama";
type ApiKeyMode = "env" | "literal" | "shell" | "none";
// pi's reasoning ceilings. "off" means no reasoning; the rest are the levels a
// model is allowed to use. See pi-ai getSupportedThinkingLevels.
type ReasoningCeiling = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
const REASONING_LEVELS: ReasoningCeiling[] = ["off", "minimal", "low", "medium", "high", "xhigh"];

// Per-model knobs the wizard can write. apiKey lives at provider scope, not here.
type ModelOptions = {
	reasoning: ReasoningCeiling;
	vision: boolean;
};

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

function hasExplicitScheme(input: string): boolean {
	return /^[a-z]+:\/\//i.test(input.trim());
}

function addDefaultScheme(input: string): string {
	if (hasExplicitScheme(input)) return input;
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
	// pi resolves an apiKey by prefix: "!cmd" runs a shell command, "$VAR" reads an
	// env var, anything else is a literal. See pi-ai resolve-config-value.
	if (mode === "shell") return value.startsWith("!") ? value : `!${value}`;
	if (mode === "env") return value.startsWith("$") ? value : `$${value}`;
	return value;
}

async function probeOpenAIModels(baseUrl: string, apiKeyMode: ApiKeyMode, apiKeyValue?: string): Promise<ProbeItem[]> {
	const headers: Record<string, string> = {
		accept: "application/json",
		"accept-encoding": "identity",
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
		let query = "";
		const selected = new Set<string>();
		let cachedLines: string[] | undefined;
		const maxVisible = 12;

		function getVisibleItems() {
			const lowerQuery = query.trim().toLowerCase();
			if (!lowerQuery) return items;
			return items.filter((item) => {
				const haystack = `${item.label} ${item.value} ${item.description ?? ""}`.toLowerCase();
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
				add(` ${theme.fg("muted", `${selected.size} selected • ${visibleItems.length}/${items.length} shown`)}`);
				add();

				if (visibleItems.length === 0) {
					add(theme.fg("warning", " No matching models."));
				} else {
					const start = Math.max(0, Math.min(cursor - Math.floor(maxVisible / 2), Math.max(0, visibleItems.length - maxVisible)));
					const end = Math.min(visibleItems.length, start + maxVisible);

					for (let i = start; i < end; i++) {
						const item = visibleItems[i];
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

					if (visibleItems.length > maxVisible) {
						add();
						add(theme.fg("dim", ` ${start + 1}-${end} of ${visibleItems.length}`));
					}
				}

				add();
				add(theme.fg("dim", " Type to search • ↑↓ move (wraps) • space toggle • enter confirm • backspace delete • esc cancel"));
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
				if (data === " ") {
					const value = visibleItems[cursor]?.value;
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

async function promptApiKey(
	ctx: CommandContext,
): Promise<{ mode: ApiKeyMode; value?: string } | null> {
	const choice = await selectOne(ctx, "API key", [
		{ value: "literal", label: "Literal API key", description: "Stored verbatim in ~/.pi/agent/models.json" },
		{ value: "env", label: "Environment variable", description: "Stored as $NAME, read from the environment at runtime" },
		{ value: "shell", label: "Shell command", description: "Stored as !command, executed at runtime (stdout is the key)" },
		{ value: "none", label: "None", description: "No key; a placeholder is written so the provider still loads" },
	]);
	if (!choice) return null;
	if (choice === "none") return { mode: "none" };

	if (choice === "env") {
		const value = await ctx.ui.input("Environment variable name", "e.g. OPENAI_API_KEY");
		if (value === undefined) return null;
		const trimmed = value.trim().replace(/^\$/, "");
		if (!trimmed) return { mode: "none" };
		return { mode: "env", value: trimmed };
	}

	if (choice === "shell") {
		const value = await ctx.ui.input("Shell command", "e.g. cat ~/.secrets/openai or pass show openai");
		if (value === undefined) return null;
		const trimmed = value.trim().replace(/^!/, "");
		if (!trimmed) return { mode: "none" };
		return { mode: "shell", value: trimmed };
	}

	const value = await ctx.ui.input("API key", "saved directly in ~/.pi/agent/models.json");
	if (value === undefined) return null;
	const trimmed = value.trim();
	if (!trimmed) return { mode: "none" };
	return { mode: "literal", value: trimmed };
}

function reasoningLabel(level: ReasoningCeiling): string {
	if (level === "off") return "Off - no reasoning";
	if (level === "xhigh") return "xhigh - maximum (only if the model supports it)";
	return `${level} - cap reasoning at ${level}`;
}

// Prompts for a reasoning ceiling. Returns null if cancelled.
async function promptReasoning(ctx: CommandContext, current?: ReasoningCeiling): Promise<ReasoningCeiling | null> {
	const items: SelectItem[] = REASONING_LEVELS.map((level) => ({
		value: level,
		label: reasoningLabel(level),
	}));
	const initialIndex = current ? REASONING_LEVELS.indexOf(current) : 0;
	const choice = await selectOne(ctx, "Reasoning", items, { initialIndex: Math.max(0, initialIndex) });
	return (choice as ReasoningCeiling | null) ?? null;
}

// When a model is capped at xhigh, some providers name that level differently
// (e.g. "max"). Offer an optional override for the provider-facing string.
async function promptXhighProviderString(ctx: CommandContext, current?: string): Promise<string | undefined> {
	const value = await ctx.ui.input(
		"xhigh provider value (blank = xhigh)",
		current && current !== "xhigh" ? `current: ${current}` : 'e.g. max (leave blank to send "xhigh")',
	);
	if (value === undefined) return undefined;
	const trimmed = value.trim();
	return trimmed || undefined;
}

async function promptVision(ctx: CommandContext, current?: boolean): Promise<boolean | null> {
	const choice = await selectOne(ctx, "Image input (vision)", [
		{ value: "yes", label: "Yes - send text + images", description: "Sets input: [text, image]" },
		{ value: "no", label: "No - text only", description: "Sets input: [text]" },
	], { initialIndex: current === false ? 1 : 0 });
	if (!choice) return null;
	return choice === "yes";
}

// Read the reasoning ceiling + vision flags already stored on a model entry,
// mirroring pi's getSupportedThinkingLevels so edit defaults match reality.
function readModelOptions(model: any): ModelOptions {
	const vision = Array.isArray(model?.input) ? model.input.includes("image") : true;
	if (!model || model.reasoning !== true) return { reasoning: "off", vision };

	const map = model.thinkingLevelMap;
	let ceiling: ReasoningCeiling = "high";
	if (map && typeof map === "object") {
		if (map.xhigh !== undefined && map.xhigh !== null) {
			ceiling = "xhigh";
		} else {
			for (let i = REASONING_LEVELS.length - 1; i >= 1; i--) {
				const level = REASONING_LEVELS[i];
				if (level === "xhigh") continue;
				if (map[level] === null) continue;
				ceiling = level;
				break;
			}
		}
	}
	return { reasoning: ceiling, vision };
}

function readXhighProviderString(model: any): string | undefined {
	const v = model?.thinkingLevelMap?.xhigh;
	return typeof v === "string" ? v : undefined;
}

async function promptModelIdsOneByOne(
	ctx: CommandContext,
	style: ProviderStyle,
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

function buildModelEntry(id: string, opts: ModelOptions, providerStringOverride?: string): any {
	const entry: any = {
		id,
		// Default to text+image so pi forwards images upstream. Without this,
		// custom models default to text-only and images are silently dropped.
		input: opts.vision ? ["text", "image"] : ["text"],
	};

	if (opts.reasoning === "off") return entry;

	entry.reasoning = true;

	// pi's getSupportedThinkingLevels: off/minimal/low/medium/high are on by
	// default when reasoning is true; xhigh is available ONLY if explicitly
	// mapped; any level set to null is removed. So we only need a map to (a)
	// unlock xhigh, or (b) cap below high by nulling the higher levels.
	const ceilingIndex = REASONING_LEVELS.indexOf(opts.reasoning);
	const map: Record<string, string | null> = {};
	for (const level of REASONING_LEVELS) {
		if (level === "off") continue;
		const index = REASONING_LEVELS.indexOf(level);
		if (level === "xhigh") {
			// xhigh is off unless explicitly mapped. Only enable it at the xhigh
			// ceiling; below high it stays absent (no redundant null needed).
			if (ceilingIndex >= index) map.xhigh = providerStringOverride?.trim() || "xhigh";
		} else if (index > ceilingIndex) {
			map[level] = null;
		}
	}

	if (Object.keys(map).length > 0) entry.thinkingLevelMap = map;
	return entry;
}

function buildProviderConfig(
	style: ProviderStyle,
	api: ProviderApi,
	baseUrl: string,
	apiKey: { mode: ApiKeyMode; value?: string },
	modelIds: string[],
	opts: ModelOptions,
	providerStringOverride?: string,
) {
	const serializedApiKey = serializeApiKey(apiKey.mode, apiKey.value, style);
	const config: any = {
		baseUrl,
		api,
		...(serializedApiKey ? { apiKey: serializedApiKey } : {}),
		models: modelIds.map((id) => buildModelEntry(id, opts, providerStringOverride)),
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

function providerModelItems(provider: any): SelectItem[] {
	const models = Array.isArray(provider?.models) ? provider.models : [];
	return models
		.map((model: any) => {
			const id = typeof model === "string" ? model.trim() : typeof model?.id === "string" ? model.id.trim() : "";
			if (!id) return null;

			const details: string[] = [];
			if (model && typeof model === "object") {
				if (model.reasoning === true) {
					const opts = readModelOptions(model);
					details.push(`reasoning:${opts.reasoning}`);
				}
				if (Array.isArray(model.input) && model.input.includes("image")) details.push("vision");
				if (typeof model.contextWindow === "number") details.push(`context ${model.contextWindow}`);
				if (typeof model.maxTokens === "number") details.push(`max ${model.maxTokens}`);
			}

			return {
				value: id,
				label: id,
				suffix: details.length > 0 ? ` • ${details.join(" • ")}` : "",
				searchText: `${id} ${details.join(" ")}`,
			};
		})
		.filter((item): item is SelectItem => item !== null);
}

function normalizeStoredEndpoint(provider: any): string {
	const endpoint = typeof provider?.baseUrl === "string" ? provider.baseUrl.trim() : "";
	if (!endpoint) return "";
	const api: ProviderApi = provider?.api === "anthropic-messages" ? "anthropic-messages" : "openai-completions";
	try {
		return normalizeEndpoint(endpoint, api);
	} catch {
		return endpoint.replace(/\/+$/, "");
	}
}

function findProvidersByEndpoint(config: ModelsConfig, endpoint: string): string[] {
	return Object.entries(config.providers ?? {})
		.filter(([, provider]) => normalizeStoredEndpoint(provider) === endpoint)
		.map(([providerId]) => providerId)
		.sort((a, b) => a.localeCompare(b));
}

async function editProviderFlow(ctx: CommandContext) {
	let cursor = 0;

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
			ctx.ui.notify(`No providers found in ${MODELS_JSON_PATH}`, "warning");
			return;
		}

		const choice = await selectOne(
			ctx,
			"Edit provider",
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

		cursor = providerIds.indexOf(choice);
		await editSingleProvider(ctx, choice);
	}
}

// Per-provider action menu. Returns when the user backs out to the provider list.
async function editSingleProvider(ctx: CommandContext, providerId: string) {
	while (true) {
		let config: ModelsConfig;
		try {
			config = loadModelsConfig();
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}
		const provider = config.providers?.[providerId];
		if (!provider) {
			ctx.ui.notify(`Provider "${providerId}" no longer exists.`, "warning");
			return;
		}

		const modelCount = Array.isArray(provider.models) ? provider.models.length : 0;
		const action = await selectOne(ctx, `Edit ${providerId}`, [
			{ value: "models", label: "Edit a model", description: `${modelCount} model${modelCount === 1 ? "" : "s"} — set reasoning / vision` },
			{ value: "probe", label: "Re-probe for new models", description: "Query /models again and add ones not already configured" },
			{ value: "add", label: "Add models manually", description: "Type model ids to add" },
			{ value: "back", label: "Back", description: "Return to the provider list" },
		]);
		if (!action || action === "back") return;

		if (action === "models") {
			await editProviderModels(ctx, providerId);
		} else if (action === "probe") {
			await reprobeProvider(ctx, providerId);
		} else if (action === "add") {
			await addModelsToProvider(ctx, providerId);
		}
	}
}

// Load config, hand the provider to a mutator, and save if it returns true.
async function mutateProvider(
	ctx: CommandContext,
	providerId: string,
	mutate: (provider: any) => boolean | Promise<boolean>,
): Promise<boolean> {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}
	const provider = config.providers?.[providerId];
	if (!provider) {
		ctx.ui.notify(`Provider "${providerId}" no longer exists.`, "warning");
		return false;
	}

	const changed = await mutate(provider);
	if (!changed) return false;

	try {
		saveModelsConfig(config);
	} catch (error) {
		ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}
	return true;
}

// Pick a model, then set its reasoning ceiling and vision flag in place.
async function editProviderModels(ctx: CommandContext, providerId: string) {
	let cursor = 0;
	while (true) {
		let provider: any;
		try {
			provider = loadModelsConfig().providers?.[providerId];
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}
		const modelItems = providerModelItems(provider);
		if (modelItems.length === 0) {
			ctx.ui.notify(`Provider "${providerId}" has no models.`, "warning");
			return;
		}

		const choice = await selectOne(ctx, `Edit model in ${providerId}`, modelItems, {
			initialIndex: Math.min(cursor, modelItems.length - 1),
		});
		if (!choice) return;
		cursor = modelItems.findIndex((item) => item.value === choice);

		const current = readModelOptions(findModel(provider, choice));
		const reasoning = await promptReasoning(ctx, current.reasoning);
		if (reasoning === null) continue;

		let xhighProviderString: string | undefined;
		if (reasoning === "xhigh") {
			xhighProviderString = await promptXhighProviderString(ctx, readXhighProviderString(findModel(provider, choice)));
		}

		const vision = await promptVision(ctx, current.vision);
		if (vision === null) continue;

		const saved = await mutateProvider(ctx, providerId, (p) => {
			const models = Array.isArray(p.models) ? p.models : [];
			const index = models.findIndex((m: any) => modelIdOf(m) === choice);
			if (index === -1) return false;
			models[index] = buildModelEntry(choice, { reasoning, vision }, xhighProviderString);
			p.models = models;
			return true;
		});
		if (saved) ctx.ui.notify(`Updated "${choice}".`, "info");
	}
}

function modelIdOf(model: any): string {
	return typeof model === "string" ? model.trim() : typeof model?.id === "string" ? model.id.trim() : "";
}

function findModel(provider: any, id: string): any {
	const models = Array.isArray(provider?.models) ? provider.models : [];
	return models.find((m: any) => modelIdOf(m) === id);
}

// Resolve a stored provider's apiKey reference back into mode+value so we can
// reuse it for probing. Anything other than $VAR or !cmd is treated as literal.
function apiKeyFromProvider(provider: any): { mode: ApiKeyMode; value?: string } {
	const raw = typeof provider?.apiKey === "string" ? provider.apiKey : "";
	if (!raw || raw === "dummy" || raw === "ollama") return { mode: "none" };
	if (raw.startsWith("!")) return { mode: "shell", value: raw.slice(1) };
	if (raw.startsWith("$")) return { mode: "env", value: raw.slice(1) };
	return { mode: "literal", value: raw };
}

async function addModelEntriesToProvider(ctx: CommandContext, providerId: string, ids: string[]) {
	const existing = new Set<string>();
	try {
		const provider = loadModelsConfig().providers?.[providerId];
		for (const m of Array.isArray(provider?.models) ? provider.models : []) existing.add(modelIdOf(m));
	} catch {
		// fall through; mutateProvider re-reads and reports errors
	}
	const fresh = dedupe(ids).filter((id) => id && !existing.has(id));
	if (fresh.length === 0) {
		ctx.ui.notify("Nothing to add — all selected models already exist.", "info");
		return;
	}

	const reasoning = await promptReasoning(ctx);
	if (reasoning === null) return;
	let xhighProviderString: string | undefined;
	if (reasoning === "xhigh") xhighProviderString = await promptXhighProviderString(ctx);
	const vision = await promptVision(ctx);
	if (vision === null) return;

	const saved = await mutateProvider(ctx, providerId, (p) => {
		const models = Array.isArray(p.models) ? p.models : [];
		for (const id of fresh) models.push(buildModelEntry(id, { reasoning, vision }, xhighProviderString));
		p.models = models;
		return true;
	});
	if (saved) ctx.ui.notify(`Added ${fresh.length} model${fresh.length === 1 ? "" : "s"} to "${providerId}".`, "info");
}

async function reprobeProvider(ctx: CommandContext, providerId: string) {
	let provider: any;
	try {
		provider = loadModelsConfig().providers?.[providerId];
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}
	if (provider?.api === "anthropic-messages") {
		ctx.ui.notify("Anthropic-style endpoints don't expose /models. Use 'Add models manually'.", "warning");
		return;
	}
	const baseUrl = typeof provider?.baseUrl === "string" ? provider.baseUrl : "";
	if (!baseUrl) {
		ctx.ui.notify(`Provider "${providerId}" has no baseUrl to probe.`, "error");
		return;
	}

	const apiKey = apiKeyFromProvider(provider);
	let probed: ProbeItem[];
	try {
		ctx.ui.notify(`Probing ${buildProbeUrl(baseUrl)} ...`, "info");
		probed = await probeOpenAIModels(baseUrl, apiKey.mode, apiKey.value);
	} catch (error) {
		ctx.ui.notify(`Probe failed: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}

	const existing = new Set((Array.isArray(provider.models) ? provider.models : []).map(modelIdOf));
	const novel = probed.filter((item) => !existing.has(item.value));
	if (novel.length === 0) {
		ctx.ui.notify("No new models — everything the endpoint returned is already configured.", "info");
		return;
	}

	const picked = await pickMany(ctx, `New models for ${providerId}`, novel);
	if (!picked || picked.length === 0) return;
	await addModelEntriesToProvider(ctx, providerId, picked);
}

async function addModelsToProvider(ctx: CommandContext, providerId: string) {
	let provider: any;
	try {
		provider = loadModelsConfig().providers?.[providerId];
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}
	const style: ProviderStyle =
		provider?.api === "anthropic-messages" ? "anthropic" : provider?.compat ? "ollama" : "openai";
	const ids = await promptModelIdsOneByOne(ctx, style);
	if (!ids || ids.length === 0) return;
	await addModelEntriesToProvider(ctx, providerId, ids);
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

async function promptProviderStyle(
	ctx: CommandContext,
): Promise<{ style: ProviderStyle; api: ProviderApi } | null> {
	const providerStyleLabel = await selectOne(ctx, "Provider style", [
		"OpenAI-compatible",
		"Anthropic-compatible",
		"Ollama-compatible",
	]);
	if (!providerStyleLabel) return null;

	const style: ProviderStyle =
		providerStyleLabel === "Anthropic-compatible"
			? "anthropic"
			: providerStyleLabel === "Ollama-compatible"
				? "ollama"
				: "openai";
	const api: ProviderApi = style === "anthropic" ? "anthropic-messages" : "openai-completions";
	return { style, api };
}

async function promptEndpoint(
	ctx: CommandContext,
	style: ProviderStyle,
	api: ProviderApi,
): Promise<{ normalized: string; raw: string } | null> {
	const endpointInput = await ctx.ui.input(
		"Endpoint",
		style === "anthropic"
			? "e.g. https://api.anthropic-proxy.com/v1"
			: style === "ollama"
				? "e.g. http://localhost:11434/v1"
				: "e.g. https://api.example.com/v1 or http://localhost:11434/v1",
	);
	if (endpointInput === undefined) return null;
	const raw = endpointInput.trim();
	if (!raw) {
		ctx.ui.notify("Endpoint is required.", "error");
		return null;
	}

	try {
		return { normalized: normalizeEndpoint(raw, api), raw };
	} catch (error) {
		ctx.ui.notify(`Invalid endpoint: ${error instanceof Error ? error.message : String(error)}`, "error");
		return null;
	}
}

async function confirmEndpointReuse(ctx: CommandContext, normalizedEndpoint: string): Promise<boolean> {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}

	const providersWithSameEndpoint = findProvidersByEndpoint(config, normalizedEndpoint);
	if (providersWithSameEndpoint.length === 0) return true;

	return ctx.ui.confirm(
		"Endpoint already exists",
		`This endpoint is already used by:\n${providersWithSameEndpoint.map((id) => `- ${id}`).join("\n")}\n\nAdd another provider with the same endpoint?`,
	);
}

async function promptProviderId(ctx: CommandContext, normalizedEndpoint: string): Promise<string | null> {
	const providerIdSuggestion = suggestProviderId(normalizedEndpoint);
	const providerNameInput = await ctx.ui.input(
		`Provider name (blank = ${providerIdSuggestion})`,
		"e.g. custom-example-com",
	);
	if (providerNameInput === undefined) return null;
	const providerId = slugify(providerNameInput.trim() || providerIdSuggestion);
	if (!providerId) {
		ctx.ui.notify("Provider name is required.", "error");
		return null;
	}

	if (BUILTIN_PROVIDER_IDS.has(providerId)) {
		const ok = await ctx.ui.confirm(
			"Override built-in provider?",
			`"${providerId}" matches a built-in provider id. Saving this will override that provider in ~/.pi/agent/models.json. Continue?`,
		);
		if (!ok) return null;
	}
	return providerId;
}

async function persistProvider(ctx: CommandContext, providerId: string, providerConfig: any): Promise<boolean> {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}

	config.providers ||= {};
	if (config.providers[providerId]) {
		const replace = await ctx.ui.confirm(
			"Replace existing provider?",
			`Provider "${providerId}" already exists in ${MODELS_JSON_PATH}. Replace it?`,
		);
		if (!replace) return false;
	}

	config.providers[providerId] = providerConfig;
	try {
		saveModelsConfig(config);
	} catch (error) {
		ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}
	return true;
}

async function addProviderFlow(ctx: CommandContext) {
	const styleChoice = await promptProviderStyle(ctx);
	if (!styleChoice) return;
	const { style, api } = styleChoice;

	const endpoint = await promptEndpoint(ctx, style, api);
	if (!endpoint) return;
	if (!(await confirmEndpointReuse(ctx, endpoint.normalized))) return;

	const providerId = await promptProviderId(ctx, endpoint.normalized);
	if (!providerId) return;

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

	const modelIds = await collectModelIds(ctx, style, api, apiKey, endpoint.normalized, endpoint.raw);
	if (!modelIds || modelIds.length === 0) return;

	const reasoning = await promptReasoning(ctx);
	if (reasoning === null) return;

	let xhighProviderString: string | undefined;
	if (reasoning === "xhigh") {
		xhighProviderString = await promptXhighProviderString(ctx);
	}

	const vision = await promptVision(ctx);
	if (vision === null) return;

	const providerConfig = buildProviderConfig(
		style,
		api,
		endpoint.normalized,
		apiKey,
		dedupe(modelIds),
		{ reasoning, vision },
		xhighProviderString,
	);
	if (!(await persistProvider(ctx, providerId, providerConfig))) return;

	ctx.ui.notify(`Saved provider \"${providerId}\" to ${MODELS_JSON_PATH}`, "info");
	ctx.ui.notify("Open /model to use your new provider.", "info");
}

async function collectModelIds(
	ctx: CommandContext,
	style: ProviderStyle,
	api: ProviderApi,
	apiKey: { mode: ApiKeyMode; value?: string },
	normalizedEndpoint: string,
	trimmedEndpointInput: string,
): Promise<string[] | null> {
	if (api !== "openai-completions") {
		return promptModelIdsOneByOne(ctx, style);
	}

	const modelMode = await selectOne(ctx, "Models", ["Auto probe from /models", "Add manually"]);
	if (!modelMode) return null;
	if (modelMode !== "Auto probe from /models") {
		return promptModelIdsOneByOne(ctx, style);
	}

	try {
		ctx.ui.notify(`Probing ${buildProbeUrl(normalizedEndpoint)} ...`, "info");
		const probedModels = await probeOpenAIModels(normalizedEndpoint, apiKey.mode, apiKey.value);
		if (probedModels.length === 0) {
			ctx.ui.notify("Probe succeeded but returned no models. Switching to manual entry.", "warning");
			return promptModelIdsOneByOne(ctx, style);
		}
		return pickMany(ctx, "Select models", probedModels);
	} catch (error) {
		const schemeHint = hasExplicitScheme(trimmedEndpointInput) ? "" : "\n\nNo http:// or https:// was provided.";
		ctx.ui.notify(
			`Auto probe failed: ${error instanceof Error ? error.message : String(error)}.${schemeHint}\n\nSwitching to manual entry.`,
			"warning",
		);
		return promptModelIdsOneByOne(ctx, style);
	}
}

export default function betterCustomWizard(pi: ExtensionAPI) {
	pi.registerCommand("better-custom", {
		description: "Wizard for adding, editing, or deleting custom providers in ~/.pi/agent/models.json",
		handler: async (_args, ctx) => {
			const action = await selectOne(ctx, "Better custom", ["Add provider", "Edit provider", "Delete provider"]);
			if (!action) return;
			if (action === "Edit provider") {
				await editProviderFlow(ctx);
				return;
			}
			if (action === "Delete provider") {
				await deleteProviderFlow(ctx);
				return;
			}
			await addProviderFlow(ctx);
		},
	});
}
