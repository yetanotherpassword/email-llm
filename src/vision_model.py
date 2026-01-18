"""Vision model integration for deep image analysis using LLaVA via Ollama."""

import base64
import io
import logging
from typing import Optional

import httpx
from PIL import Image

from .config import settings

logger = logging.getLogger(__name__)


class VisionModel:
    """Vision model integration for image description and understanding."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize vision model client."""
        self.model = model or settings.ollama_vision_model
        self.base_url = base_url or settings.ollama_base_url
        self.timeout = 120.0  # Vision models can be slow

    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        # Resize if too large (vision models have limits)
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Encode to base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def describe_image(
        self,
        img: Image.Image,
        context: Optional[str] = None,
    ) -> str:
        """Generate a detailed description of an image.

        Args:
            img: PIL Image to describe
            context: Optional context about the image (e.g., email subject)

        Returns:
            Text description of the image
        """
        prompt = "Describe this image in detail. "
        if context:
            prompt += f"Context: This image is an email attachment. {context}. "
        prompt += (
            "Include: what the image shows, any visible text, "
            "document type if applicable, and key details that would help "
            "someone search for this image later."
        )

        return await self._query_vision(img, prompt)

    async def extract_text_from_image(self, img: Image.Image) -> str:
        """Extract and transcribe text from an image.

        Args:
            img: PIL Image containing text

        Returns:
            Extracted text content
        """
        prompt = (
            "Extract and transcribe ALL text visible in this image. "
            "Include headers, body text, labels, captions, and any other text. "
            "Preserve the structure and formatting as much as possible. "
            "If there's no text, respond with 'No text found.'"
        )

        return await self._query_vision(img, prompt)

    async def analyze_document_image(self, img: Image.Image) -> dict:
        """Analyze an image of a document.

        Args:
            img: PIL Image of a document

        Returns:
            Dict with document analysis including type, summary, and extracted text
        """
        prompt = (
            "Analyze this document image. Provide:\n"
            "1. Document type (invoice, receipt, letter, form, screenshot, etc.)\n"
            "2. Brief summary of content\n"
            "3. Key information (dates, amounts, names, etc.)\n"
            "4. Full text transcription\n"
            "Format your response with clear sections."
        )

        response = await self._query_vision(img, prompt)

        return {
            "analysis": response,
            "raw_response": response,
        }

    async def _query_vision(self, img: Image.Image, prompt: str) -> str:
        """Send a query to the vision model.

        Args:
            img: PIL Image to analyze
            prompt: Text prompt for the model

        Returns:
            Model response text
        """
        image_b64 = self._image_to_base64(img)

        # Ollama API format for vision models
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                return data.get("response", "")

            except httpx.TimeoutException:
                logger.error("Vision model request timed out")
                return "Error: Vision model request timed out"
            except httpx.HTTPStatusError as e:
                logger.error(f"Vision model HTTP error: {e}")
                return f"Error: {e}"
            except Exception as e:
                logger.error(f"Vision model error: {e}")
                return f"Error: {e}"

    def describe_image_sync(
        self,
        img: Image.Image,
        context: Optional[str] = None,
    ) -> str:
        """Synchronous version of describe_image."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.describe_image(img, context)
        )

    def extract_text_sync(self, img: Image.Image) -> str:
        """Synchronous version of extract_text_from_image."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self.extract_text_from_image(img)
        )


class LMStudioVisionModel:
    """Vision model via LM Studio (if it supports vision models)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize LM Studio vision client."""
        self.base_url = base_url or settings.lmstudio_base_url
        self.model = model or settings.lmstudio_model
        self.timeout = 120.0

    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 data URL."""
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        if img.mode != "RGB":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    async def describe_image(
        self,
        img: Image.Image,
        context: Optional[str] = None,
    ) -> str:
        """Generate a description using LM Studio's OpenAI-compatible API."""
        prompt = "Describe this image in detail for search indexing."
        if context:
            prompt += f" Context: {context}"

        image_url = self._image_to_base64(img)

        # OpenAI-compatible vision API format
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": 1000,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                return data["choices"][0]["message"]["content"]

            except Exception as e:
                logger.error(f"LM Studio vision error: {e}")
                return f"Error: {e}"
