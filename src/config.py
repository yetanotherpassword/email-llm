"""Configuration settings for email-llm."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="EMAIL_LLM_",
    )

    # Thunderbird profile path (Snap installation default)
    thunderbird_profile_path: Path = Field(
        default=Path.home() / "snap/thunderbird/common/.thunderbird",
        description="Path to Thunderbird profile directory",
    )

    # Data storage
    data_dir: Path = Field(
        default=Path(__file__).parent.parent / "data",
        description="Directory for storing embeddings and indexes",
    )

    # LLM settings
    llm_backend: Literal["lmstudio", "ollama"] = Field(
        default="lmstudio",
        description="LLM backend to use",
    )

    # LM Studio settings (OpenAI-compatible API)
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="LM Studio API base URL",
    )
    lmstudio_model: str = Field(
        default="local-model",
        description="Model name in LM Studio (usually 'local-model')",
    )

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    ollama_model: str = Field(
        default="mistral",
        description="Ollama model for text generation",
    )
    ollama_vision_model: str = Field(
        default="llava:13b",
        description="Ollama model for vision tasks",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model for embeddings",
    )

    # Embedding settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
    )

    # ChromaDB settings
    chroma_collection_name: str = Field(
        default="emails",
        description="ChromaDB collection name",
    )

    # YOLO settings
    yolo_model: str = Field(
        default="yolov8n.pt",
        description="YOLO model to use (n=nano, s=small, m=medium, l=large, x=extra-large)",
    )
    yolo_confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence for YOLO detections",
    )

    # Processing settings
    chunk_size: int = Field(
        default=512,
        description="Text chunk size for embeddings",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks",
    )
    max_attachment_size_mb: int = Field(
        default=50,
        description="Maximum attachment size to process in MB",
    )

    # Web UI settings
    web_host: str = Field(
        default="127.0.0.1",
        description="Web UI host",
    )
    web_port: int = Field(
        default=8000,
        description="Web UI port",
    )

    def get_chroma_path(self) -> Path:
        """Get path to ChromaDB storage."""
        return self.data_dir / "chroma"

    def get_attachment_cache_path(self) -> Path:
        """Get path to attachment cache."""
        return self.data_dir / "attachments"

    def get_yolo_cache_path(self) -> Path:
        """Get path to YOLO results cache."""
        return self.data_dir / "yolo_cache"

    def get_skip_log_path(self) -> Path:
        """Get path to skip/failure log file."""
        return self.data_dir / "skipped_files.json"

    def get_progress_log_path(self) -> Path:
        """Get path to progress checkpoint file (for crash recovery)."""
        return self.data_dir / "indexing_progress.json"


# Global settings instance
settings = Settings()
