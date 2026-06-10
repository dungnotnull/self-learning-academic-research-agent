"""
academic-research-enhanced — UnifiedLLMClient
Claude / OpenAI / Ollama unified client with automatic fallback, retry, and cost tracking.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncGenerator, Optional

logger = logging.getLogger("academic_agent.llm_client")

# Cost per 1K tokens (USD)
COST_PER_1K: dict = {
    "claude-opus-4-8":   {"input": 0.015, "output": 0.075},
    "claude-opus-4-5":   {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-sonnet-3-5": {"input": 0.003, "output": 0.015},
    "gpt-4o":            {"input": 0.005, "output": 0.015},
    "gpt-4o-mini":       {"input": 0.00015, "output": 0.0006},
    "llama3":            {"input": 0.0,   "output": 0.0},
    "llama2":            {"input": 0.0,   "output": 0.0},
    "mistral":           {"input": 0.0,   "output": 0.0},
}

DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OLLAMA_MODEL = "llama3"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds


class UnifiedLLMClient:
    """Unified LLM client with provider fallback chain."""

    def __init__(self, memory_manager=None):
        self._memory_manager = memory_manager
        self._privacy_mode = os.environ.get("PRIVACY_MODE", "false").lower() == "true"
        self._last_provider = "unknown"
        self._last_model = "unknown"

        # Detect available providers
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self._ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._ollama_model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

        if self._privacy_mode:
            logger.info("PRIVACY_MODE enabled: using Ollama only")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        stream: bool = False,
        task: str = "general",
    ) -> str:
        """Complete a prompt using the best available provider."""
        if self._privacy_mode:
            return await self._complete_with_retry(
                self._call_ollama, prompt, system, max_tokens, temperature, task
            )

        # Try providers in order
        providers = []
        if self._anthropic_key:
            providers.append(("claude", DEFAULT_CLAUDE_MODEL))
        if self._openai_key:
            providers.append(("openai", DEFAULT_OPENAI_MODEL))
        providers.append(("ollama", self._ollama_model))

        last_exception = None
        for provider, model in providers:
            try:
                if provider == "claude":
                    result = await self._complete_with_retry(
                        self._call_claude, prompt, system, max_tokens, temperature, task, model
                    )
                elif provider == "openai":
                    result = await self._complete_with_retry(
                        self._call_openai, prompt, system, max_tokens, temperature, task, model
                    )
                else:
                    result = await self._complete_with_retry(
                        self._call_ollama, prompt, system, max_tokens, temperature, task
                    )
                self._last_provider = provider
                self._last_model = model
                return result
            except Exception as e:
                logger.warning("Provider %s failed: %s; trying next", provider, e)
                last_exception = e

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_exception}"
        )

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Claude (or fallback to full response)."""
        if not self._anthropic_key or self._privacy_mode:
            # Fallback: return full response as single chunk
            result = await self.complete(prompt, system, max_tokens)
            yield result
            return

        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self._anthropic_key)
            messages = [{"role": "user", "content": prompt}]
            kwargs = {
                "model": DEFAULT_CLAUDE_MODEL,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            async with client.messages.stream(**kwargs) as stream_obj:
                async for text in stream_obj.text_stream:
                    yield text
        except Exception as e:
            logger.warning("Streaming failed: %s; falling back to complete()", e)
            result = await self.complete(prompt, system, max_tokens)
            yield result

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    async def _complete_with_retry(self, fn, prompt, system, max_tokens, temperature, task, model=None):
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                if model:
                    return await fn(prompt, system, max_tokens, temperature, task, model)
                else:
                    return await fn(prompt, system, max_tokens, temperature, task)
            except Exception as e:
                last_exc = e
                error_str = str(e).lower()
                # Exponential backoff for rate limit errors
                if "rate" in error_str or "429" in error_str or "overload" in error_str:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.info("Rate limit hit; retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, MAX_RETRIES)
                    await asyncio.sleep(delay)
                elif attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY)
                else:
                    raise
        raise last_exc

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    async def _call_claude(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        task: str = "general",
        model: str = DEFAULT_CLAUDE_MODEL,
    ) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic SDK not installed: pip install anthropic")

        client = anthropic.AsyncAnthropic(api_key=self._anthropic_key)

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        t0 = time.time()
        response = await client.messages.create(**kwargs)
        latency = time.time() - t0

        output_text = response.content[0].text if response.content else ""
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        self._log_cost("claude", model, prompt_tokens, completion_tokens, cost, task)

        logger.debug(
            "Claude %s: %d+%d tokens, $%.4f, %.1fs",
            model, prompt_tokens, completion_tokens, cost, latency
        )
        return output_text

    async def _call_openai(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        task: str = "general",
        model: str = DEFAULT_OPENAI_MODEL,
    ) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai SDK not installed: pip install openai")

        client = AsyncOpenAI(api_key=self._openai_key)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        latency = time.time() - t0

        output_text = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        self._log_cost("openai", model, prompt_tokens, completion_tokens, cost, task)

        logger.debug(
            "OpenAI %s: %d+%d tokens, $%.4f, %.1fs",
            model, prompt_tokens, completion_tokens, cost, latency
        )
        return output_text

    async def _call_ollama(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        task: str = "general",
    ) -> str:
        import aiohttp

        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload = {
            "model": self._ollama_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        t0 = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._ollama_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        latency = time.time() - t0
        output_text = data.get("response", "")

        # Ollama has no token counting; estimate
        prompt_tokens = len(full_prompt.split()) * 4 // 3
        completion_tokens = len(output_text.split()) * 4 // 3

        self._log_cost("ollama", self._ollama_model, prompt_tokens, completion_tokens, 0.0, task)

        logger.debug("Ollama %s: ~%d tokens, $0.0000, %.1fs", self._ollama_model, completion_tokens, latency)
        return output_text

    # ------------------------------------------------------------------
    # Cost calculation
    # ------------------------------------------------------------------

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        rates = COST_PER_1K.get(model, {"input": 0.0, "output": 0.0})
        return (prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"]

    def _log_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        task: str,
    ):
        if self._memory_manager:
            try:
                self._memory_manager.log_llm_cost(
                    provider, model, prompt_tokens, completion_tokens, cost_usd, task
                )
            except Exception as e:
                logger.debug("Failed to log LLM cost: %s", e)

    # ------------------------------------------------------------------
    # Sync wrapper (for non-async callers)
    # ------------------------------------------------------------------

    def complete_sync(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper around complete()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.complete(prompt, **kwargs))
                    return future.result(timeout=120)
            else:
                return loop.run_until_complete(self.complete(prompt, **kwargs))
        except RuntimeError:
            return asyncio.run(self.complete(prompt, **kwargs))
