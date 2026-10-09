"""HTTP client for llama.cpp OpenAI-compatible API."""

import json
import logging
import os
import time
from pathlib import Path

import asyncio

import httpx

# Structured file logger — zero console output
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "client.log"

_logger = logging.getLogger("llm_client")
_logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(_LOG_FILE, mode="a")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
_logger.addHandler(_fh)


class StallTimeout(RuntimeError):
    """Stream stalled: no chunks received for idle_timeout_s."""


class OutputOverflow(RuntimeError):
    """Response exceeded max_chars (runaway generation)."""


class TaskDeadlineExceeded(RuntimeError):
    """Task exceeded its adaptive per-task deadline."""


def _resolve_reasoning_effort(api_config: dict) -> str | None:
    """Resolve the reasoning_effort value to send for this run (pi-style).

    Mirrors pi's openai-completions buildParams(): gated on model.reasoning
    and compat.supportsReasoningEffort. For a non-"off" level: thinkingLevelMap[level],
    falling back to the raw level when the map has no entry. For off/unset:
    thinkingLevelMap["off"] only when it is a string (null means "don't send").
    """
    model = (api_config.get("models") or [{}])[0]
    if not model.get("reasoning", False):
        return None
    if not api_config.get("compat", {}).get("supportsReasoningEffort"):
        return None
    level = api_config.get("thinking_level")
    tmap = model.get("thinkingLevelMap") or {}
    if level and level != "off":
        value = tmap.get(level, level)
        return value if isinstance(value, str) else None
    off_value = tmap.get("off")
    return off_value if isinstance(off_value, str) else None


class LLMClient:
    """Async-compatible HTTP client for an OpenAI-compatible local server.

    Configured from configs/models.json (see that file for the schema):
    baseUrl, model id, capability metadata and the per-run choices
    (thinking_level). Sampling and reasoning-budget parameters are NOT sent here —
    the server is started manually by the user with its own flags.
    """

    def __init__(self, api_config: dict, **kwargs):
        """
        Args:
            api_config: Parsed contents of configs/models.json.
            timeout_config: dict from configs/timeout.json, passed via kwargs.
                          (idle_timeout_s, max_chars, warmup_tokens, deadline_factor,
                           min_deadline_s, fallback_deadline_s)

        Raises:
            ValueError: If api_config has no baseUrl or no models entry.
        """
        if not api_config.get("baseUrl"):
            raise ValueError("api_config requires a baseUrl")
        base = api_config["baseUrl"].rstrip("/")
        # Accept both 'http://host:port' and 'http://host:port/v1' forms
        self.base_url = base if base.endswith("/v1") else base + "/v1"
        self.api_config = api_config
        self.compat = api_config.get("compat", {}) or {}
        self.thinking_level = api_config.get("thinking_level")
        models = api_config.get("models") or []
        if not models:
            raise ValueError("api_config requires at least one model entry")
        model = models[0]
        self.model_id = model.get("id")
        self.model_reasoning = bool(model.get("reasoning", False))
        self.thinking_map = model.get("thinkingLevelMap", {}) or {}
        # Guardrails for the adaptive deadline (maxTokens = worst-case n_predict)
        self.max_tokens = model.get("maxTokens")
        # Timeout / guardrail params (separate file: configs/timeout.json)
        tc = kwargs.get("timeout_config", {}) or api_config.get("timeout_config", {})
        self.idle_timeout = float(tc.get("idle_timeout_s", 30))
        self.max_chars = int(tc.get("max_chars", 32000))
        self.warmup_tokens = int(tc.get("warmup_tokens", 32))
        self.deadline_factor = float(tc.get("deadline_factor", 2.5))
        self.min_deadline = float(tc.get("min_deadline_s", 60))
        self.fallback_deadline = float(tc.get("fallback_deadline_s", 300))
        self._client = httpx.AsyncClient(timeout=300.0)

    # ── public API ────────────────────────────────────────────────

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request and return the response text.

        Uses streaming mode to work around llama.cpp returning empty
        content in non-streaming responses. Completion is stream-driven
        ([DONE]); time gates are activity-based: idle watchdog (reset
        on every chunk), output cap, and an adaptive per-task deadline
        derived from observed token rate and n_predict.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.

        Returns:
            Response text from the model.

        Raises:
            RuntimeError: On invalid JSON response.
            httpx.HTTPError: On persistent network failure (after retry).
            StallTimeout: No stream chunks for idle_timeout_s.
            OutputOverflow: Response exceeded max_chars.
            TaskDeadlineExceeded: Task exceeded its adaptive deadline.
        """
        payload = self._build_payload(system_prompt, user_prompt)
        payload["stream"] = True  # llama.cpp needs streaming for content
        _logger.debug("Request payload: %s", json.dumps(payload, default=str))

        last_exc = None
        for attempt in range(2):  # 1 initial + 1 retry
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                ) as response:
                    # Handle 5xx / rate-limit with retry
                    if response.status_code in (429, 500, 502, 503, 504):
                        if attempt == 0:
                            wait = 2 ** attempt  # 2s backoff
                            _logger.warning(
                                "HTTP %s on attempt %d - retrying in %ds",
                                response.status_code, attempt + 1, wait,
                            )
                            await self._client.aclose()
                            await self._aclose_and_recreate()
                            await asyncio.sleep(min(wait, 1))
                            continue
                        else:
                            _logger.error(
                                "HTTP %s on retry - giving up", response.status_code,
                            )
                            raise httpx.HTTPError(
                                f"Server error {response.status_code} after retry"
                            )

                    response.raise_for_status()

                    # Consume SSE stream with activity-based gates
                    text = await self._stream_with_gates(response)
                    _logger.info("Response received (%d chars)", len(text))
                    return text

            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_exc = exc
                if attempt == 0:
                    _logger.warning(
                        "Connection failed on attempt %d - retrying", attempt + 1,
                    )
                    await self._client.aclose()
                    await self._aclose_and_recreate()
                    await asyncio.sleep(1)
                    continue
                _logger.error("Connection failed after retry: %s", exc)
                raise

        # Should not reach here, but just in case
        raise last_exc or httpx.HTTPError("Unknown failure")

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ── internals ─────────────────────────────────────────────────

    def _build_payload(self, system_prompt: str, user_prompt: str) -> dict:
        """Build the OpenAI-compatible chat completion payload."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {"messages": messages, "stream": False}

        payload["model"] = self.model_id

        # Reasoning params — mirrors pi's openai-completions buildParams().
        # The reasoning budget is NOT sent: it is set at server launch (recipe).
        effort = _resolve_reasoning_effort(self.api_config)
        if effort is not None:
            payload["reasoning_effort"] = effort

        return payload

    async def _stream_with_gates(self, response) -> str:
        """Consume the SSE stream with activity-based completion gates.

        Gates (all per-task, reset by activity, not by a static timer):
            - idle watchdog: no chunks for idle_timeout_s -> StallTimeout
            - output cap: more than max_chars -> OutputOverflow
            - adaptive deadline: after warmup_tokens, deadline is frozen as
              max(min_deadline_s, deadline_factor * n_predict / observed_rate);
              if elapsed exceeds it -> TaskDeadlineExceeded
            - [DONE] or stream end -> task complete (no timer involved)
        """
        loop = asyncio.get_running_loop()
        t_start = loop.time()
        t_first = None
        n_tokens = 0
        # maxTokens (declared in models.json) plays the role of worst-case n_predict
        n_predict = self.max_tokens
        deadline = self.fallback_deadline if n_predict is None else None
        full_text = ""

        # Idle watchdog: each wait for the next chunk is capped at idle_timeout
        aiter = response.aiter_lines()
        while True:
            try:
                line = await asyncio.wait_for(aiter.__anext__(), timeout=self.idle_timeout)
            except asyncio.TimeoutError:
                _logger.error("Idle: no chunks for %.0fs - aborting task", self.idle_timeout)
                raise StallTimeout(f"no chunks for {self.idle_timeout:.0f}s")
            except (StopIteration, StopAsyncIteration):
                break
            now = loop.time()

            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = data.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if not content:
                continue

            full_text += content
            n_tokens += 1
            if t_first is None:
                t_first = now

            # Adaptive deadline: freeze the estimate at warmup
            if deadline is None and n_tokens == self.warmup_tokens and t_first < now:
                rate = (n_tokens - 1) / (now - t_first)
                deadline = max(self.min_deadline, self.deadline_factor * float(n_predict) / rate)
                _logger.info(
                    "Adaptive deadline: %.0fs (rate=%.1f tok/s, n_predict=%s)",
                    deadline, rate, n_predict,
                )

            if deadline is not None and now - t_start > deadline:
                _logger.error("Task exceeded adaptive deadline %.0fs - aborting", deadline)
                raise TaskDeadlineExceeded(f"exceeded {deadline:.0f}s adaptive deadline")

            if len(full_text) > self.max_chars:
                _logger.error("Response exceeded %d chars - aborting", self.max_chars)
                raise OutputOverflow(f"response exceeded {self.max_chars} chars")
        return full_text

    def _extract_text(data: dict) -> str:
        """Extract response text from OpenAI-compatible JSON response."""
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices in response")
        message = choices[0].get("message", {})
        text = message.get("content", "")
        if text is None:
            text = ""
        return text

    async def _aclose_and_recreate(self):
        """Recreate the HTTP client for retry."""
        self._client = httpx.AsyncClient(timeout=300.0)
