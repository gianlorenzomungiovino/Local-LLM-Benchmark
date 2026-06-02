"""Tests for llm_client.py."""

import json
import pytest
from unittest.mock import patch, MagicMock
from llm_client import LLMClient


class TestLLMClientInit:
    def test_init_default_timeout(self):
        client = LLMClient("http://127.0.0.1:8080")
        assert client.base_url == "http://127.0.0.1:8080"
        assert client.timeout == 120.0

    def test_init_custom_timeout(self):
        client = LLMClient("http://127.0.0.1:8080", timeout=30.0)
        assert client.timeout == 30.0

    def test_init_trailing_slash(self):
        client = LLMClient("http://127.0.0.1:8080/")
        assert client.base_url == "http://127.0.0.1:8080"


class TestBuildPayload:
    def test_build_payload_minimal(self):
        client = LLMClient("http://127.0.0.1:8080")
        payload = client._build_payload("test prompt", {})
        assert payload["messages"] == [{"role": "user", "content": "test prompt"}]
        assert payload["temperature"] == 0.3
        assert payload["top_p"] == 0.95
        assert payload["top_k"] == 30
        assert payload["max_tokens"] == 4096

    def test_build_payload_with_params(self):
        client = LLMClient("http://127.0.0.1:8080")
        params = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 50,
            "min_p": 0.1,
            "repeat_penalty": 1.2,
            "presence_penalty": 0.5,
        }
        payload = client._build_payload("test", params)
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.8
        assert payload["top_k"] == 50
        assert payload["min_p"] == 0.1
        assert payload["repeat_penalty"] == 1.2
        assert payload["presence_penalty"] == 0.5

    def test_build_payload_max_tokens_not_overridden(self):
        client = LLMClient("http://127.0.0.1:8080")
        payload = client._build_payload("test", {})
        assert payload["max_tokens"] == 4096


class TestSendRequest:
    def test_send_request_success(self):
        client = LLMClient("http://127.0.0.1:8080")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}],
            "model": "qwen3.6-35b",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("llm_client.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value = MagicMock(
                post=MagicMock(return_value=mock_response)
            )
            result = client.send_request("test prompt", {})

        assert result["content"] == "Hello world"
        assert result["model"] == "qwen3.6-35b"

    def test_send_request_empty_content(self):
        client = LLMClient("http://127.0.0.1:8080")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": None}}],
            "model": "test-model",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("llm_client.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value = MagicMock(
                post=MagicMock(return_value=mock_response)
            )
            result = client.send_request("test", {})

        assert result["content"] == ""

    def test_send_request_no_choices(self):
        client = LLMClient("http://127.0.0.1:8080")
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = MagicMock()

        with patch("llm_client.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value = MagicMock(
                post=MagicMock(return_value=mock_response)
            )
            with pytest.raises(RuntimeError, match="no choices"):
                client.send_request("test", {})

    def test_send_request_default_model(self):
        client = LLMClient("http://127.0.0.1:8080")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test"}}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("llm_client.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value = MagicMock(
                post=MagicMock(return_value=mock_response)
            )
            result = client.send_request("test", {})

        assert result["model"] == "unknown"


class TestSendWithRetry:
    def test_send_with_retry_success_on_first(self):
        client = LLMClient("http://127.0.0.1:8080")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "test",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("llm_client.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value = MagicMock(
                post=MagicMock(return_value=mock_response)
            )
            result = client.send_with_retry("test", {}, max_retries=2)

        assert result["content"] == "ok"

    def test_send_with_retry_success_after_fail(self):
        client = LLMClient("http://127.0.0.1:8080")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "test",
        }
        mock_response.raise_for_status = MagicMock()

        import httpx
        with patch("llm_client.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = MagicMock()
            mock_client_cls.return_value = mock_client

            # First call fails, second succeeds
            mock_client.__enter__.return_value.post.side_effect = [
                httpx.ConnectError("fail"),
                mock_response,
            ]
            result = client.send_with_retry("test", {}, max_retries=2)

        assert result["content"] == "ok"

    def test_send_with_retry_all_fail(self):
        client = LLMClient("http://127.0.0.1:8080")
        import httpx
        with patch("llm_client.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_client.__enter__.return_value.post.side_effect = httpx.ConnectError("fail")

            with pytest.raises(RuntimeError, match="All 3 attempts failed"):
                client.send_with_retry("test", {}, max_retries=2)
