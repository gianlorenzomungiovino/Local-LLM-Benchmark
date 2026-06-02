"""HTTP client for OpenAI-compatible LLM servers."""

from __future__ import annotations

import httpx
from typing import Any


class LLMClient:
    """Client for sending chat completion requests to an OpenAI-compatible server."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _build_payload(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        """Build the chat completion payload."""
        return {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": params.get("temperature", 0.3),
            "top_p": params.get("top_p", 0.95),
            "top_k": params.get("top_k", 30),
            "min_p": params.get("min_p", 0.0),
            "repeat_penalty": params.get("repeat_penalty", 1.0),
            "presence_penalty": params.get("presence_penalty", 0.0),
            "max_tokens": 4096,
        }

    def send_request(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a chat completion request and return the response dict.

        Raises:
            httpx.ConnectError: If the server is unreachable.
            httpx.TimeoutException: If the request times out.
            httpx.HTTPStatusError: If the server returns an error status.
            RuntimeError: If the response is invalid.
        """
        payload = self._build_payload(prompt, params)
        url = f"{self.base_url}/v1/chat/completions"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()

        # Extract the assistant's message content
        choices = data.get("choices")
        if not choices:
            raise RuntimeError("Response has no choices")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if content is None:
            content = ""

        model = data.get("model", "unknown")

        return {
            "content": content,
            "model": model,
            "raw": data,
        }

    def send_with_retry(
        self,
        prompt: str,
        params: dict[str, Any],
        max_retries: int = 2,
        backoff: float = 2.0,
    ) -> dict[str, Any]:
        """Send a request with retry logic.

        Args:
            prompt: The prompt to send.
            params: Sampling parameters.
            max_retries: Maximum number of retry attempts.
            backoff: Base backoff multiplier for exponential retry.

        Returns:
            Response dict with content, model, and raw fields.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                return self.send_request(prompt, params)
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < max_retries:
                    import time
                    time.sleep(backoff * (2 ** attempt))
                continue

        raise RuntimeError(
            f"All {max_retries + 1} attempts failed. Last error: {last_error}"
        ) from last_error
