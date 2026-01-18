"""LLM client for LM Studio and Ollama integration."""

import logging
from typing import AsyncGenerator, Optional

import httpx
from openai import AsyncOpenAI

from .config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting LM Studio and Ollama."""

    def __init__(
        self,
        backend: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize LLM client.

        Args:
            backend: 'lmstudio' or 'ollama'
            base_url: API base URL
            model: Model name to use
        """
        self.backend = backend or settings.llm_backend

        if self.backend == "lmstudio":
            self.base_url = base_url or settings.lmstudio_base_url
            self.model = model or settings.lmstudio_model
            # LM Studio uses OpenAI-compatible API
            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key="not-needed",  # LM Studio doesn't require API key
            )
        else:  # ollama
            self.base_url = base_url or settings.ollama_base_url
            self.model = model or settings.ollama_model
            self.client = None  # Use httpx directly for Ollama

        self.timeout = 60.0

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        if self.backend == "lmstudio":
            return await self._generate_lmstudio(
                prompt, system_prompt, max_tokens, temperature
            )
        else:
            return await self._generate_ollama(
                prompt, system_prompt, max_tokens, temperature
            )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Generated text chunks
        """
        if self.backend == "lmstudio":
            async for chunk in self._stream_lmstudio(
                prompt, system_prompt, max_tokens, temperature
            ):
                yield chunk
        else:
            async for chunk in self._stream_ollama(
                prompt, system_prompt, max_tokens, temperature
            ):
                yield chunk

    async def _generate_lmstudio(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using LM Studio's OpenAI-compatible API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LM Studio generation error: {e}")
            raise

    async def _stream_lmstudio(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Stream using LM Studio's OpenAI-compatible API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LM Studio streaming error: {e}")
            raise

    async def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
            except Exception as e:
                logger.error(f"Ollama generation error: {e}")
                raise

    async def _stream_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        """Stream using Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            import json

                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
            except Exception as e:
                logger.error(f"Ollama streaming error: {e}")
                raise

    async def check_connection(self) -> bool:
        """Check if the LLM backend is available."""
        try:
            if self.backend == "lmstudio":
                # Try to list models
                await self.client.models.list()
                return True
            else:
                # Check Ollama health
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.base_url}/api/tags")
                    return response.status_code == 200
        except Exception as e:
            logger.warning(f"LLM connection check failed: {e}")
            return False
