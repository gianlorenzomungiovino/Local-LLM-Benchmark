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
        """
        self.base_url = base_url.rstrip("/")
        self.params = params
        self._client = httpx.AsyncClient(timeout=120.0)

    # ── public API ────────────────────────────────────────────────

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request and return the response text.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.

        Returns:
            Response text from the model.

        Raises:
            RuntimeError: On invalid JSON response.
            httpx.HTTPError: On persistent network failure (after retry).
        """
        payload = self._build_payload(system_prompt, user_prompt)
        _logger.debug("Request payload: %s", json.dumps(payload, default=str))

        last_exc = None
        for attempt in range(2):  # 1 initial + 1 retry
            try:
                response = await self._client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )

                # Handle 5xx / rate-limit with retry
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt == 0:
                        wait = 2 ** attempt  # 2s backoff
                        _logger.warning(
                            "HTTP %s on attempt %d — retrying in %ds",
                            response.status_code, attempt + 1, wait,
                        )
                        await self._client.aclose()
                        await self._aclose_and_recreate()
                        await asyncio.sleep(wait)
                        continue
                    else:
                        _logger.error(
                            "HTTP %s on retry — giving up", response.status_code,
                        )
                        raise httpx.HTTPError(
                            f"Server error {response.status_code} after retry"
                        )

                response.raise_for_status()

                # Parse JSON response
                try:
                    data = response.json()
                except json.JSONDecodeError as exc:
                    _logger.error("Invalid JSON response: %s", exc)
                    raise RuntimeError(f"Invalid JSON from server: {exc}") from exc

                # Extract text from OpenAI-compatible response shape
                text = self._extract_text(data)
                _logger.info("Response received (%d chars)", len(text))
                return text

            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_exc = exc
                if attempt == 0:
                    _logger.warning(
                        "Connection failed on attempt %d — retrying", attempt + 1,
                    )
                    await self._client.aclose()
                    await self._aclose_and_recreate()
                    await asyncio.sleep(2)
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

    @staticmethod
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
        self._client = httpx.AsyncClient(timeout=120.0)
