from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from .entry_summary_provider import (
    ClaudeEntrySummaryProvider,
    EntrySummaryProvider,
    HybridEntrySummaryProvider,
    LocalEntrySummaryProvider,
    MockEntrySummaryProvider,
    OllamaEntrySummaryProvider,
)
from .state_label_provider import (
    ClaudeStateLabelProvider,
    LocalStateLabelProvider,
    MockStateLabelProvider,
    OllamaStateLabelProvider,
    StateLabelProvider,
)

logger = logging.getLogger("analysis-service.provider-registry")


@dataclass
class ProviderRegistry:
    summary_providers: dict[str, EntrySummaryProvider] = field(default_factory=dict)
    state_label_providers: dict[str, StateLabelProvider] = field(default_factory=dict)
    default_summary: str = "local"
    default_state_label: str = "local"
    fallback: str = "local"


_REGISTRY: ProviderRegistry | None = None


def get_provider_registry() -> ProviderRegistry:
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    summary_provider_name = os.environ.get("ENTRY_SUMMARY_PROVIDER", "auto").strip().lower()
    state_label_provider_name = os.environ.get("STATE_LABEL_PROVIDER", "auto").strip().lower()
    ollama_url = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip()

    local_summary = LocalEntrySummaryProvider()
    summary_providers: dict[str, EntrySummaryProvider] = {
        "mock": local_summary,  # backward compat: "mock" now uses local extraction
        "local": local_summary,
        "ollama": OllamaEntrySummaryProvider(ollama_url=ollama_url, model=ollama_model),
    }

    local_state = LocalStateLabelProvider()
    state_label_providers: dict[str, StateLabelProvider] = {
        "mock": local_state,  # backward compat: "mock" now uses same local engine
        "local": local_state,
        "ollama": OllamaStateLabelProvider(ollama_url=ollama_url, model=ollama_model),
    }

    if anthropic_api_key:
        summary_providers["anthropic"] = ClaudeEntrySummaryProvider(
            api_key=anthropic_api_key, model=anthropic_model,
        )
        summary_providers["hybrid"] = HybridEntrySummaryProvider(
            api_key=anthropic_api_key, model=anthropic_model,
        )
        state_label_providers["anthropic"] = ClaudeStateLabelProvider(
            api_key=anthropic_api_key, model=anthropic_model,
        )
        logger.info("anthropic_provider_registered", extra={"model": anthropic_model})
    else:
        logger.info("anthropic_provider_skipped: ANTHROPIC_API_KEY not set")

    default_summary = _resolve_default(summary_provider_name, summary_providers, "entry_summary")
    default_state_label = _resolve_default(state_label_provider_name, state_label_providers, "state_label")

    _REGISTRY = ProviderRegistry(
        summary_providers=summary_providers,
        state_label_providers=state_label_providers,
        default_summary=default_summary,
        default_state_label=default_state_label,
    )
    return _REGISTRY


def _resolve_default(
    provider_name: str,
    providers: dict,
    label: str,
) -> str:
    if provider_name == "auto":
        # Prefer hybrid (1 Claude call) over full anthropic (2 calls)
        # Fall back to local (0 calls) when no API key
        if "hybrid" in providers:
            return "hybrid"
        if "anthropic" in providers:
            return "anthropic"
        return "local" if "local" in providers else "ollama"
    if provider_name in providers:
        return provider_name
    logger.warning(
        f"invalid_{label}_provider_config",
        extra={
            "event": f"invalid_{label}_provider_config",
            "provider": provider_name,
            "fallback": "local",
        },
    )
    return "local"
