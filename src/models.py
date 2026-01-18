"""Data models for email-llm."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class AttachmentType(str, Enum):
    """Types of attachments we can process."""

    IMAGE = "image"
    PDF = "pdf"
    WORD = "word"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    TEXT = "text"
    OTHER = "other"


class YoloDetection(BaseModel):
    """A single YOLO detection result."""

    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


class ImageAnalysis(BaseModel):
    """Analysis results for an image attachment."""

    # YOLO fast-filter results
    yolo_detections: list[YoloDetection] = Field(default_factory=list)
    has_faces: bool = False
    has_text: bool = False
    has_charts: bool = False
    detected_objects: list[str] = Field(default_factory=list)

    # Vision model deep analysis (populated on demand)
    vision_description: Optional[str] = None
    ocr_text: Optional[str] = None


class Attachment(BaseModel):
    """An email attachment."""

    filename: str
    content_type: str
    size_bytes: int
    attachment_type: AttachmentType
    extracted_text: Optional[str] = None
    image_analysis: Optional[ImageAnalysis] = None

    # Storage path for the extracted attachment
    cache_path: Optional[Path] = None


class EmailMessage(BaseModel):
    """A parsed email message."""

    # Unique identifier (message-id or generated)
    message_id: str

    # Basic headers
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: list[str] = Field(default_factory=list)
    cc_addresses: list[str] = Field(default_factory=list)
    date: Optional[datetime] = None

    # Content
    body_text: Optional[str] = None
    body_html: Optional[str] = None

    # Attachments
    attachments: list[Attachment] = Field(default_factory=list)

    # Source info
    mailbox_path: Optional[Path] = None
    folder_name: Optional[str] = None

    # Computed fields for search
    @property
    def full_text(self) -> str:
        """Get all searchable text from the email."""
        parts = []
        if self.subject:
            parts.append(f"Subject: {self.subject}")
        if self.from_address:
            parts.append(f"From: {self.from_address}")
        if self.to_addresses:
            parts.append(f"To: {', '.join(self.to_addresses)}")
        if self.date:
            parts.append(f"Date: {self.date.isoformat()}")
        if self.body_text:
            parts.append(self.body_text)

        # Include attachment text
        for att in self.attachments:
            if att.extracted_text:
                parts.append(f"[Attachment: {att.filename}]\n{att.extracted_text}")
            if att.image_analysis and att.image_analysis.vision_description:
                parts.append(
                    f"[Image: {att.filename}]\n{att.image_analysis.vision_description}"
                )
            if att.image_analysis and att.image_analysis.ocr_text:
                parts.append(f"[OCR from {att.filename}]\n{att.image_analysis.ocr_text}")

        return "\n\n".join(parts)


class SearchResult(BaseModel):
    """A search result with relevance info."""

    email: EmailMessage
    score: float
    matched_chunks: list[str] = Field(default_factory=list)
    highlight: Optional[str] = None


class SearchQuery(BaseModel):
    """A search query with filters."""

    query: str
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    from_address: Optional[str] = None
    folder: Optional[str] = None
    has_attachments: Optional[bool] = None
    attachment_type: Optional[AttachmentType] = None

    # YOLO filters
    has_faces: Optional[bool] = None
    has_charts: Optional[bool] = None
    contains_object: Optional[str] = None

    # Result settings
    limit: int = 20
