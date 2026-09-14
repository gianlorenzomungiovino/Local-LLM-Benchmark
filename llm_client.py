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


class LLMClient:
    """Async-compatible HTTP client for llama.cpp /v1/chat/completions."""

    # llama.cpp OpenAI-compatible parameter mapping
    # llama.cpp OpenAI-compatible API parameter mapping
    # All sampling parameters accepted by the server
    _PARAM_MAP = {
        "temperature": "temperature",
        "top_k": "top_k",
        "top_p": "top_p",
        "min_p": "min_p",
        "repeat_penalty": "repeat_penalty",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "mirostat": "mirostat",
        "mirostat_tau": "mirostat_tau",
        "mirostat_eta": "mirostat_eta",
        "typical_p": "typical_p",
        "penalty_last_n": "penalty_last_n",
        "tfs_z": "tfs_z",
        "num_keep": "num_keep",
        "seed": "seed",
        "n_predict": "n_predict",
        "logit_bias": "logit_bias",
    }

    def __init__(self, base_url: str, **params):
        """
        Args:
            base_url: e.g. 'http://localhost:8080'
            **params: API parameters from configs/run.json
                      (temperature, top_k, top_p, min_p, repeat_penalty, presence_penalty)
            timeout_config: dict from configs/timeout.json (idle_timeout_s, max_chars, etc.)
        """
        self.base_url = base_url.rstrip("/")
        self.params = params
        # Timeout / guardrail params (separate file: configs/timeout.json)
        tc = params.get("timeout_config", {})
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
                    f"{self.base_url}/v1/chat/completions",
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

    async def fetch_model_info(self) -> dict | None:
        """Query the llama.cpp server's /v1/models endpoint and return full model metadata.

        Returns:
            Dict with model metadata (id, n_ctx, n_params, n_vocab, etc.),
            or None if no models found or on error.
        """
        try:
            response = await self._client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            if models:
                meta = models[0].get("meta", {})
                info = {
                    "id": models[0].get("id"),
                    "n_ctx": meta.get("n_ctx"),
                    "n_ctx_train": meta.get("n_ctx_train"),
                    "n_embd": meta.get("n_embd"),
                    "n_params": meta.get("n_params"),
                    "n_vocab": meta.get("n_vocab"),
                    "size": meta.get("size"),
                }
                _logger.info("Auto-detected model: %s (ctx=%s, params=%s)",
                           info["id"], info["n_ctx"], info["n_params"])
                return info
            _logger.debug("No models returned by /v1/models")
            return None
        except httpx.HTTPError as exc:
            _logger.warning("Failed to fetch model info: %s", exc)
            return None

    # ── internals ─────────────────────────────────────────────────

    def _build_payload(self, system_prompt: str, user_prompt: str) -> dict:
        """Build the OpenAI-compatible chat completion payload."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {"messages": messages, "stream": False}

        # Add model if configured
        model = self.params.get("model")
        if model:
            payload["model"] = model

        # Map config params to API parameters
        for config_key, api_key in self._PARAM_MAP.items():
            value = self.params.get(config_key)
            if value is not None:
                payload[api_key] = value

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
        n_predict = self.params.get("n_predict")
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
