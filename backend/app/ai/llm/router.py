"""
Language-model router: provider fallback + validated structured output.

* ``complete`` sends a request through a ``FallbackChain`` of providers (primary first),
  each wrapped in its own circuit breaker and retry policy. Model calls have no external
  side effect, so they are declared idempotent: ambiguous timeouts may be retried and
  failed over.
* ``complete_structured`` additionally extracts one JSON object from the answer and
  validates it with a caller-supplied parser (e.g. ``parse_proposed_action``). Invalid or
  truncated output is regenerated with corrective feedback, a bounded number of times.
  Feedback contains only validation locations, never echoes of untrusted content.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from app.ai.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ModelRefusalError,
    Role,
    StopReason,
    StructuredOutputError,
)
from app.core.errors import DealPilotError
from app.core.resilience.circuit_breaker import CircuitBreaker
from app.core.resilience.fallback import FallbackCallback, FallbackChain, ProviderSlot
from app.core.resilience.retry import RetryPolicy
from app.domain.actions import InvalidActionError

UsageCallback = Callable[[LLMRequest, LLMResponse], Awaitable[None] | None]

_FEEDBACK_ECHO_LIMIT = 4_000


@dataclass(frozen=True, slots=True)
class StructuredResult[T]:
    value: T
    response: LLMResponse
    attempts: int


class LLMRouter:
    def __init__(
        self,
        chain: FallbackChain[LLMProvider],
        *,
        max_regenerations: int = 2,
        on_usage: UsageCallback | None = None,
    ) -> None:
        if max_regenerations < 0:
            raise ValueError("max_regenerations must be >= 0")
        self._chain = chain
        self._max_regenerations = max_regenerations
        self._on_usage = on_usage

    @classmethod
    def from_providers(
        cls,
        providers: Sequence[LLMProvider],
        *,
        retry_policy: RetryPolicy | None = None,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        on_fallback: FallbackCallback | None = None,
        on_usage: UsageCallback | None = None,
        max_regenerations: int = 2,
    ) -> LLMRouter:
        policy = retry_policy or RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=20.0)
        slots = [
            ProviderSlot(
                name=provider.name,
                client=provider,
                breaker=CircuitBreaker(
                    provider.name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                ),
                retry_policy=policy,
            )
            for provider in providers
        ]
        chain: FallbackChain[LLMProvider] = FallbackChain("llm", slots, on_fallback=on_fallback)
        return cls(chain, max_regenerations=max_regenerations, on_usage=on_usage)

    @property
    def provider_names(self) -> tuple[str, ...]:
        return self._chain.provider_names

    async def complete(self, request: LLMRequest) -> LLMResponse:
        result = await self._chain.execute(
            lambda provider: provider.complete(request),
            operation=f"llm.{request.operation}",
            idempotent=True,
        )
        response = result.value
        if self._on_usage is not None:
            outcome = self._on_usage(request, response)
            if inspect.isawaitable(outcome):
                await outcome
        if response.stop_reason is StopReason.REFUSED:
            raise ModelRefusalError(
                f"model declined the '{request.operation}' request", provider=response.provider
            )
        return response

    async def complete_structured[T](
        self, request: LLMRequest, parse: Callable[[str], T]
    ) -> StructuredResult[T]:
        conversation = list(request.messages)
        current = request.model_copy(update={"json_output": True})
        last_problem = "no attempt made"

        for attempt in range(1, self._max_regenerations + 2):
            response = await self.complete(current)
            try:
                if response.stop_reason is StopReason.MAX_TOKENS:
                    raise StructuredOutputError("output was cut off before it was complete")
                value = parse(extract_json_object(response.text))
            except (StructuredOutputError, ValueError, DealPilotError) as exc:
                if isinstance(exc, DealPilotError) and not _is_output_problem(exc):
                    raise
                last_problem = _describe(exc)
                conversation += [
                    ChatMessage(
                        role=Role.ASSISTANT,
                        content=response.text[:_FEEDBACK_ECHO_LIMIT] or "(empty)",
                    ),
                    ChatMessage(
                        role=Role.USER,
                        content=(
                            f"Your previous answer was invalid: {last_problem}. "
                            "Reply with exactly one corrected JSON object and nothing else."
                        ),
                    ),
                ]
                current = current.model_copy(update={"messages": tuple(conversation)})
                continue
            return StructuredResult(value=value, response=response, attempts=attempt)

        raise StructuredOutputError(
            f"no valid structured output for '{request.operation}' after "
            f"{self._max_regenerations + 1} attempts ({last_problem})"
        )


def _is_output_problem(exc: DealPilotError) -> bool:
    """Parser failures (bad model output) are regenerated; provider failures propagate."""
    return isinstance(exc, (StructuredOutputError, InvalidActionError))


def _describe(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        fields = sorted(
            {".".join(str(part) for part in err["loc"]) or "root" for err in exc.errors()}
        )
        return f"{exc.error_count()} validation error(s) in: {', '.join(fields)}"
    if isinstance(exc, DealPilotError):
        return exc.message
    return type(exc).__name__ if not str(exc) else str(exc)[:300]


def extract_json_object(text: str) -> str:
    """
    Return the first complete JSON object in ``text``. Tolerates surrounding prose and
    Markdown code fences, which models sometimes add despite instructions.
    """
    start = text.find("{")
    if start == -1:
        raise StructuredOutputError("answer contains no JSON object")
    try:
        _, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"answer is not valid JSON ({exc.msg})") from exc
    return text[start:end]


def pydantic_parser[T](model: type[T]) -> Callable[[str], T]:
    """Parser for ``complete_structured`` that validates JSON into ``model``."""
    adapter: TypeAdapter[T] = TypeAdapter(model)

    def parse(text: str) -> T:
        return adapter.validate_json(text)

    return parse
