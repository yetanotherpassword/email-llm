"""Email indexing pipeline - coordinates parsing, extraction, and embedding."""

import asyncio
import json
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
from .skip_log import SkipLog
from .vision_model import VisionModel
from .yolo_analyzer import YoloAnalyzer

logger = logging.getLogger(__name__)


def save_progress(email: EmailMessage, index: int, total: int, attachment: Optional[str] = None):
    """Save current processing state for crash recovery."""
    progress_file = settings.get_progress_log_path()
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "index": index,
        "total": total,
        "email_id": email.message_id,
        "subject": email.subject or "(no subject)",
        "from_address": email.from_address or "(unknown)",
        "date": email.date.isoformat() if email.date else "(unknown)",
        "current_attachment": attachment,
    }

    with open(progress_file, "w") as f:
        json.dump(data, f, indent=2)


def get_last_crash_info() -> Optional[dict]:
    """Get info about the last email being processed when a crash occurred."""
    progress_file = settings.get_progress_log_path()
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def clear_progress():
    """Clear the progress file after successful completion."""
    progress_file = settings.get_progress_log_path()
    if progress_file.exists():
        progress_file.unlink()


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
        self.skip_log = SkipLog()

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

        # Check for previous crash
        crash_info = get_last_crash_info()
        if crash_info:
            logger.warning(
                f"Previous indexing crashed while processing email #{crash_info['index']+1}/{crash_info['total']}: "
                f"From: {crash_info['from_address']}, Subject: {crash_info['subject'][:50]}, "
                f"Attachment: {crash_info.get('current_attachment', 'N/A')}"
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("Indexing emails...", total=total)

            for i, email in enumerate(emails[:total]):
                # Save progress before processing (for crash recovery)
                save_progress(email, i, total)

                try:
                    chunks = self._process_email(email, i, total)
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

        # Clear progress file on successful completion
        clear_progress()

        logger.info(
            f"Indexing complete: {stats['emails_processed']} new emails indexed, "
            f"{stats['emails_skipped']} skipped (already indexed), "
            f"{stats['chunks_created']} chunks, "
            f"{stats['attachments_processed']} attachments"
        )

        return stats

    def _process_email(self, email: EmailMessage, index: int = 0, total: int = 0) -> int:
        """Process a single email and add to vector store.

        Args:
            email: EmailMessage to process
            index: Current email index (for progress tracking)
            total: Total emails to process (for progress tracking)

        Returns:
            Number of chunks created
        """
        # Process attachments
        for attachment in email.attachments:
            # Update progress with current attachment
            save_progress(email, index, total, attachment.filename)
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
                    self.skip_log.add(
                        email_id=email.message_id,
                        subject=email.subject,
                        from_address=email.from_address,
                        date=email.date,
                        attachment_filename=attachment.filename,
                        reason="PDF skipped (--skip-pdfs flag)",
                    )
                    return
                try:
                    text = self.attachment_extractor.extract_text(data, attachment)
                    if text:
                        attachment.extracted_text = text
                except Exception as e:
                    self.skip_log.add(
                        email_id=email.message_id,
                        subject=email.subject,
                        from_address=email.from_address,
                        date=email.date,
                        attachment_filename=attachment.filename,
                        reason="PDF extraction failed",
                        error_message=str(e),
                    )
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
                    self.skip_log.add(
                        email_id=email.message_id,
                        subject=email.subject,
                        from_address=email.from_address,
                        date=email.date,
                        attachment_filename=attachment.filename,
                        reason=f"{attachment.attachment_type.value} extraction failed",
                        error_message=str(e),
                    )

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
                    self.skip_log.add(
                        email_id=email.message_id,
                        subject=email.subject,
                        from_address=email.from_address,
                        date=email.date,
                        attachment_filename=attachment.filename,
                        reason="Image processing failed",
                        error_message=str(e),
                    )
        except Exception as e:
            self.skip_log.add(
                email_id=email.message_id,
                subject=email.subject,
                from_address=email.from_address,
                date=email.date,
                attachment_filename=attachment.filename,
                reason="Attachment processing failed",
                error_message=str(e),
            )

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
