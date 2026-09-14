"""Smoke test for activity-based completion gates in LLMClient._stream_with_gates."""
import asyncio
import json

from llm_client import (
    LLMClient,
    OutputOverflow,
    StallTimeout,
    TaskDeadlineExceeded,
)


class FakeStream:
    """Minimal duck-typed httpx stream (aiter_lines).

    stop_after: stop producing lines after N (simulates stream end).
    hang_after: after N lines, the stream stops producing but keeps
                waiting forever (simulates a stalled server).
    chunk_delay: seconds between chunks.
    """

    def __init__(self, lines, stop_after=None, hang_after=None, chunk_delay=0.0):
        self.lines = list(lines)
        self.stop_after = stop_after
        self.hang_after = hang_after
        self.chunk_delay = chunk_delay

    async def aiter_lines(self):
        for i, line in enumerate(self.lines):
            if self.stop_after is not None and i >= self.stop_after:
                return
            if self.chunk_delay:
                await asyncio.sleep(self.chunk_delay)
            yield line
            if self.hang_after is not None and i + 1 >= self.hang_after:
                # server stalls: no more chunks ever
                await asyncio.sleep(3600)


def chunk(text: str) -> str:
    return "data: " + json.dumps({"choices": [{"delta": {"content": text}}]})


def run(client, stream):
    try:
        return f"OK ({len(asyncio.run(client._stream_with_gates(stream)))} chars)"
    except (StallTimeout, OutputOverflow, TaskDeadlineExceeded) as exc:
        return f"{type(exc).__name__}: {exc}"


def main() -> None:
    cases = [
        ("1 normal completion (expect OK)",
         LLMClient("http://x"),
         FakeStream([chunk("hello "), chunk("world"), "data: [DONE]"])),
        ("2 idle watchdog (expect StallTimeout)",
         LLMClient("http://x", idle_timeout_s=0.5, n_predict=2048),
         FakeStream([chunk("a")], hang_after=1)),
        ("3 adaptive deadline (expect TaskDeadlineExceeded)",
         LLMClient("http://x", idle_timeout_s=30, warmup_tokens=5,
                  min_deadline_s=0.3, n_predict=10),
         FakeStream([chunk("a")] + [chunk("b")] * 200, chunk_delay=0.05)),
        ("4 output overflow (expect OutputOverflow)",
         LLMClient("http://x", max_chars=10, warmup_tokens=5, n_predict=10),
         FakeStream([chunk("x" * 20)])),
        ("5 fallback deadline, no n_predict (expect TaskDeadlineExceeded)",
         LLMClient("http://x", min_deadline_s=60, fallback_deadline_s=0.3),
         FakeStream([chunk("a")] * 50, chunk_delay=0.05)),
    ]

    for name, client, stream in cases:
        print(f"{name}: {run(client, stream)}")


if __name__ == "__main__":
    main()
