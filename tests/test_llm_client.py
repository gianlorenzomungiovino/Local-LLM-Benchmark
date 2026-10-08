"""Unit tests for llm_client.LLMClient — HTTP request construction, error handling, retry logic."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ── helpers ──────────────────────────────────────────────────────

def _make_response(status_code: int, json_data: dict) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def _make_ok_response(text: str = "Hello world") -> MagicMock:
    """Build a mock 200 response with a valid OpenAI-compatible body."""
    return _make_response(200, {
        "choices": [{"message": {"content": text}}],
    })


# ── fixtures ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _no_file_logger():
    """Prevent the real file logger from writing during tests."""
    with patch("llm_client._logger") as mock_logger:
        yield mock_logger


# ── tests: HTTP request construction ─────────────────────────────

class TestRequestConstruction:
    """Verify the payload sent to the server matches the OpenAI-compatible format."""

    @pytest.mark.asyncio
    async def test_basic_request_construction(self, _no_file_logger):
        """A basic call sends system + user messages with stream=False."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_make_ok_response("test"))
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080")
            result = await client.complete("sys", "user")

            assert result == "test"
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://localhost:8080/v1/chat/completions"
            payload = call_args[1]["json"]
            assert payload["messages"] == [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
            ]
            assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_config_params_mapped_to_payload(self, _no_file_logger):
        """Config params (temperature, top_k, etc.) appear in the payload."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_make_ok_response("ok"))
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080", temperature=0.5, top_k=20, top_p=0.9)
            await client.complete("s", "u")

            payload = mock_client.post.call_args[1]["json"]
            assert payload["temperature"] == 0.5
            assert payload["top_k"] == 20
            assert payload["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_model_param_included_when_set(self, _no_file_logger):
        """The 'model' field is added to the payload when configured."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_make_ok_response("ok"))
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080", model="llama-3")
            await client.complete("s", "u")

            payload = mock_client.post.call_args[1]["json"]
            assert payload["model"] == "llama-3"

    @pytest.mark.asyncio
    async def test_model_param_omitted_when_none(self, _no_file_logger):
        """The 'model' field is absent when not configured."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_make_ok_response("ok"))
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080")
            await client.complete("s", "u")

            payload = mock_client.post.call_args[1]["json"]
            assert "model" not in payload

    @pytest.mark.asyncio
    async def test_base_url_trailing_slash_stripped(self, _no_file_logger):
        """Trailing slash on base_url is removed before building the URL."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_make_ok_response("ok"))
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080/")
            await client.complete("s", "u")

            url = mock_client.post.call_args[0][0]
            assert url == "http://localhost:8080/v1/chat/completions"


# ── tests: error handling ────────────────────────────────────────

class TestErrorHandling:
    """Connection refused, HTTP 5xx, invalid JSON, empty choices."""

    @pytest.mark.asyncio
    async def test_connection_refused_raises(self, _no_file_logger):
        """ConnectError on first attempt triggers retry; second failure re-raises."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080")
            with pytest.raises(httpx.ConnectError):
                await client.complete("s", "u")

            # Should have been called twice (initial + retry)
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_http_500_triggers_retry_then_raises(self, _no_file_logger):
        """HTTP 500 on attempt 0 triggers retry; 500 on retry raises."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=[
                    _make_response(500, {}),
                    _make_response(500, {}),
                ]
            )
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080")
            with pytest.raises(Exception):
                await client.complete("s", "u")

            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_http_500_succeeds_on_retry(self, _no_file_logger):
        """HTTP 500 on attempt 0, then 200 on retry returns the text."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=[
                    _make_response(500, {}),
                    _make_ok_response("recovered"),
                ]
            )
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080")
            result = await client.complete("s", "u")

            assert result == "recovered"
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_json_raises_runtime_error(self, _no_file_logger):
        """Non-JSON response body raises RuntimeError."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            bad_resp = MagicMock()
            bad_resp.status_code = 200
            bad_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
            bad_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=bad_resp)
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080")
            with pytest.raises(RuntimeError, match="Invalid JSON"):
                await client.complete("s", "u")

    @pytest.mark.asyncio
    async def test_empty_choices_raises_runtime_error(self, _no_file_logger):
        """Response with no choices raises RuntimeError."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                return_value=_make_response(200, {"choices": []})
            )
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080")
            with pytest.raises(RuntimeError, match="No choices"):
                await client.complete("s", "u")

    @pytest.mark.asyncio
    async def test_http_429_triggers_retry(self, _no_file_logger):
        """HTTP 429 (rate limit) triggers retry like 5xx."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=[
                    _make_response(429, {}),
                    _make_ok_response("after-rate-limit"),
                ]
            )
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080")
            result = await client.complete("s", "u")

            assert result == "after-rate-limit"
            assert mock_client.post.call_count == 2


# ── tests: retry logic ───────────────────────────────────────────

class TestRetryLogic:
    """Verify retry behavior: attempt count, backoff, client recreation."""

    @pytest.mark.asyncio
    async def test_retry_count_is_two(self, _no_file_logger):
        """Maximum 2 attempts (initial + 1 retry) for connection errors."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("fail"))
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080")
            with pytest.raises(httpx.ConnectError):
                await client.complete("s", "u")

            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_client_recreated_on_retry(self, _no_file_logger):
        """On retry, the old client is closed and a new one is created."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client_1 = AsyncMock()
            mock_client_2 = AsyncMock()
            mock_client_1.post = AsyncMock(
                side_effect=[
                    _make_response(500, {}),
                ]
            )
            mock_client_2.post = AsyncMock(return_value=_make_ok_response("ok"))

            # First call returns client_1, second call (recreate) returns client_2
            MockClient.side_effect = [mock_client_1, mock_client_2]

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080")
            result = await client.complete("s", "u")

            assert result == "ok"
            # Two AsyncClient instances created (initial + recreate)
            assert MockClient.call_count == 2
            # Old client closed
            mock_client_1.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_retry_on_200(self, _no_file_logger):
        """Successful 200 response does not trigger retry."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_make_ok_response("ok"))
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient("http://localhost:8080")
            await client.complete("s", "u")

            assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_close_cleans_up_client(self, _no_file_logger):
        """close() calls aclose() on the underlying httpx client."""
        with patch("llm_client.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            from llm_client import LLMClient
            client = LLMClient({"baseUrl": "http://localhost:8080", "models": [{"id": "m"}]})
            await client.close()

            mock_client.aclose.assert_called_once()


