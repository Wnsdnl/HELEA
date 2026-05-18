"""
Async HTTP client for vLLM OpenAI-compatible endpoint.

Usage:
    client = AsyncVLLMClient(cfg)
    responses = client.run(
        prompts,
        system_prompt="...",
        temperature=0.0,
        max_tokens=256,
    )
"""

import asyncio
import json
import aiohttp
from tqdm.asyncio import tqdm as atqdm


DEFAULT_URL   = "http://localhost:18001"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_CONCURRENCY = 32


class AsyncVLLMClient:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.base_url    = getattr(cfg, "llm_url",     None) or DEFAULT_URL
        self.model       = getattr(cfg, "model",        None) or DEFAULT_MODEL
        self.concurrency = getattr(cfg, "concurrency",  None) or DEFAULT_CONCURRENCY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prompts: list[str],
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 256,
        disable_template: bool = True,   # kept for API compatibility
    ) -> list[str]:
        """Synchronous wrapper — call this from non-async code."""
        return asyncio.run(
            self._run_all(prompts, system_prompt, temperature, max_tokens)
        )

    # ------------------------------------------------------------------
    # Internal async helpers
    # ------------------------------------------------------------------

    async def _run_all(
        self,
        prompts: list[str],
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        sem = asyncio.Semaphore(self.concurrency)
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"

        connector = aiohttp.TCPConnector(limit=self.concurrency)
        timeout   = aiohttp.ClientTimeout(total=120)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                self._call_once(session, sem, url, prompt, system_prompt, temperature, max_tokens)
                for prompt in prompts
            ]
            results = await atqdm.gather(*tasks, desc="  LLM requests", unit="req")

        return list(results)

    async def _call_once(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        url: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }

        data = None   # ensure defined before try block for error reporting
        async with sem:
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return f"[Error {resp.status}]: {text[:200]}"
                    data = await resp.json()
                    choice = data["choices"][0]
                    if "message" in choice:          # OpenAI chat completions format
                        msg = choice["message"]
                        return msg.get("content") or msg.get("reasoning") or ""
                    elif "text" in choice:           # vLLM native / completions format
                        return choice["text"]
                    elif "delta" in choice:          # streaming format
                        return choice["delta"].get("content", "")
                    else:
                        raise KeyError(f"Unknown choice format: {list(choice.keys())}")
            except asyncio.TimeoutError:
                return "[Error: request timed out]"
            except aiohttp.ClientError as e:
                return f"[Error: {e}]"
            except (KeyError, json.JSONDecodeError) as e:
                import sys
                print(f"[LLM parse error] {e} | raw={str(data)[:300]}", file=sys.stderr)
                return f"[Error: unexpected response format — {e}]"
            except Exception as e:
                import sys
                print(f"[LLM unexpected error] {type(e).__name__}: {e}", file=sys.stderr)
                return f"[Error: {type(e).__name__}: {e}]"
