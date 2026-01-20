"""Email indexing pipeline - coordinates parsing, extraction, and embedding."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tqdm import tqdm

from .attachment_extractor import AttachmentExtractor
from .config import settings
from .email_parser import ThunderbirdParser
from .embeddings import VectorStore
from .models import AttachmentType, EmailMessage
from .vision_model import VisionModel
from .yolo_analyzer import YoloAnalyzer

logger = logging.getLogger(__name__)


class EmailIndexer:
    """Indexes emails from Thunderbird for search."""

    def __init__(
        self,
        profile_path: Optional[Path] = None,
        vector_store: Optional[VectorStore] = None,
        process_images: bool = True,
        use_vision_model: bool = False,
        skip_pdfs: bool = False,
    ):
        """Initialize the indexer.

        Args:
            profile_path: Path to Thunderbird profile
            vector_store: Vector store instance
            process_images: Whether to process images with YOLO
            use_vision_model: Whether to use vision model for deep image analysis
            skip_pdfs: Whether to skip PDF processing (to avoid crashes)
        """
        self.profile_path = profile_path or settings.thunderbird_profile_path
        self.vector_store = vector_store or VectorStore()
        self.process_images = process_images
        self.use_vision_model = use_vision_model
        self.skip_pdfs = skip_pdfs

        # Initialize components
        self.parser = ThunderbirdParser(self.profile_path)
        self.attachment_extractor = AttachmentExtractor()

        if process_images:
            self.yolo = YoloAnalyzer()
        else:
            self.yolo = None

        if use_vision_model:
            self.vision = VisionModel()
        else:
            self.vision = None

    def index_all(
        self,
        reindex: bool = False,
        max_emails: Optional[int] = None,
        incremental: bool = True,
    ) -> dict:
        """Index all emails from Thunderbird.

        Args:
            reindex: If True, clear existing index first
            max_emails: Maximum emails to process (for testing)
            incremental: If True (default), skip already indexed emails

        Returns:
            Dict with indexing statistics
        """
        if reindex:
            logger.info("Clearing existing index...")
            self.vector_store.clear()
            indexed_ids = set()
        elif incremental:
            logger.info("Loading list of already indexed emails...")
            indexed_ids = self.vector_store.get_indexed_email_ids()
            logger.info(f"Found {len(indexed_ids)} already indexed emails")
        else:
            indexed_ids = set()

        stats = {
            "emails_processed": 0,
            "emails_skipped": 0,
            "emails_failed": 0,
            "attachments_processed": 0,
            "images_analyzed": 0,
            "chunks_created": 0,
        }

        # Count emails first for progress bar
        logger.info("Scanning mailboxes...")
        emails = list(self.parser.parse_all())

        # Filter out already indexed emails if incremental
        if incremental and indexed_ids:
            new_emails = [e for e in emails if e.message_id not in indexed_ids]
            stats["emails_skipped"] = len(emails) - len(new_emails)
            logger.info(
                f"Found {len(emails)} total emails, "
                f"{stats['emails_skipped']} already indexed, "
                f"{len(new_emails)} new to process"
            )
            emails = new_emails

        total = len(emails) if max_emails is None else min(len(emails), max_emails)

        if total == 0:
            logger.info("No new emails to index.")
            return stats

        logger.info(f"Processing {total} emails...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("Indexing emails...", total=total)

            for i, email in enumerate(emails[:total]):
                try:
                    chunks = self._process_email(email)
                    stats["emails_processed"] += 1
                    stats["chunks_created"] += chunks
                    stats["attachments_processed"] += len(email.attachments)

                    for att in email.attachments:
                        if att.attachment_type == AttachmentType.IMAGE and att.image_analysis:
                            stats["images_analyzed"] += 1

                except Exception as e:
                    logger.warning(f"Failed to process email {email.message_id}: {e}")
                    stats["emails_failed"] += 1

                progress.update(task, advance=1)

        logger.info(
            f"Indexing complete: {stats['emails_processed']} new emails indexed, "
            f"{stats['emails_skipped']} skipped (already indexed), "
            f"{stats['chunks_created']} chunks, "
            f"{stats['attachments_processed']} attachments"
        )

        return stats

    def _process_email(self, email: EmailMessage) -> int:
        """Process a single email and add to vector store.

        Args:
            email: EmailMessage to process

        Returns:
            Number of chunks created
        """
        # Process attachments
        for attachment in email.attachments:
            self._process_attachment(email, attachment)

        # Add to vector store
        chunks = self.vector_store.add_email(email)
        return chunks

    def _process_attachment(self, email: EmailMessage, attachment) -> None:
        """Process an attachment and update its extracted content."""
        try:
            # Get raw attachment data
            data = self.parser.get_attachment_data(email, attachment)
            if not data:
                return

            # Extract text from documents
            if attachment.attachment_type == AttachmentType.PDF:
                # Skip PDFs if requested (they can cause crashes with corrupt files)
                if self.skip_pdfs:
                    logger.debug(f"Skipping PDF: {attachment.filename}")
                    return
                try:
                    text = self.attachment_extractor.extract_text(data, attachment)
                    if text:
                        attachment.extracted_text = text
                except Exception as e:
                    logger.warning(f"Failed to extract text from {attachment.filename}: {e}")
            elif attachment.attachment_type in (
                AttachmentType.WORD,
                AttachmentType.EXCEL,
                AttachmentType.POWERPOINT,
                AttachmentType.TEXT,
            ):
                try:
                    text = self.attachment_extractor.extract_text(data, attachment)
                    if text:
                        attachment.extracted_text = text
                except Exception as e:
                    logger.warning(f"Failed to extract text from {attachment.filename}: {e}")

            # Process images
            elif attachment.attachment_type == AttachmentType.IMAGE and self.process_images:
                try:
                    img = self.attachment_extractor.extract_image_for_analysis(data, attachment)
                    if img:
                        # Run YOLO analysis
                        if self.yolo:
                            try:
                                cache_key = f"{email.message_id}_{attachment.filename}"
                                analysis = self.yolo.analyze_image(img, cache_key=cache_key)

                                # Add YOLO detection summary as searchable text
                                yolo_summary = self.yolo.get_detection_summary(analysis)
                                attachment.image_analysis = analysis
                            except Exception as e:
                                logger.warning(f"YOLO analysis failed for {attachment.filename}: {e}")

                            # Run OCR (separate try block so it runs even if YOLO fails)
                            try:
                                ocr_text = self.attachment_extractor.ocr_image(img)
                                if ocr_text and attachment.image_analysis:
                                    attachment.image_analysis.ocr_text = ocr_text
                            except Exception as e:
                                logger.debug(f"OCR failed for {attachment.filename}: {e}")

                            # Deep vision analysis if enabled
                            if self.vision and self.use_vision_model and attachment.image_analysis:
                                try:
                                    description = asyncio.get_event_loop().run_until_complete(
                                        self.vision.describe_image(
                                            img, context=f"Email subject: {email.subject}"
                                        )
                                    )
                                    attachment.image_analysis.vision_description = description
                                except Exception as e:
                                    logger.warning(f"Vision analysis failed: {e}")

                        # Close image to free memory
                        img.close()
                except Exception as e:
                    logger.warning(f"Failed to process image {attachment.filename}: {e}")
        except Exception as e:
            logger.warning(f"Failed to process attachment {attachment.filename}: {e}")

    def index_single_mailbox(self, mailbox_path: Path) -> dict:
        """Index a single mailbox file.

        Args:
            mailbox_path: Path to mbox file

        Returns:
            Indexing statistics
        """
        stats = {
            "emails_processed": 0,
            "emails_failed": 0,
            "chunks_created": 0,
        }

        for email in self.parser.parse_mbox(mailbox_path):
            try:
                chunks = self._process_email(email)
                stats["emails_processed"] += 1
                stats["chunks_created"] += chunks
            except Exception as e:
                logger.warning(f"Failed to process email: {e}")
                stats["emails_failed"] += 1

        return stats

    def get_stats(self) -> dict:
        """Get current index statistics."""
        return {
            "total_chunks": self.vector_store.count(),
            "collection_name": self.vector_store.collection_name,
        }
