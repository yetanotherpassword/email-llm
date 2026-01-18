"""Thunderbird email parser for mbox and maildir formats."""

import email
import hashlib
import logging
import mailbox
import re
from datetime import datetime
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Generator, Optional

from .models import Attachment, AttachmentType, EmailMessage

logger = logging.getLogger(__name__)


def decode_mime_header(header: Optional[str]) -> str:
    """Decode a MIME-encoded header value."""
    if not header:
        return ""

    decoded_parts = []
    for part, encoding in decode_header(header):
        if isinstance(part, bytes):
            try:
                decoded_parts.append(part.decode(encoding or "utf-8", errors="replace"))
            except (LookupError, UnicodeDecodeError):
                decoded_parts.append(part.decode("utf-8", errors="replace"))
        else:
            decoded_parts.append(part)

    return "".join(decoded_parts)


def parse_addresses(header: Optional[str]) -> list[str]:
    """Parse email addresses from a header."""
    if not header:
        return []

    # Handle multiple addresses separated by commas
    addresses = []
    for addr in header.split(","):
        _, email_addr = parseaddr(addr.strip())
        if email_addr:
            addresses.append(email_addr.lower())

    return addresses


def get_attachment_type(content_type: str, filename: str) -> AttachmentType:
    """Determine attachment type from content type and filename."""
    content_type = content_type.lower()
    filename = filename.lower()

    if content_type.startswith("image/"):
        return AttachmentType.IMAGE
    elif content_type == "application/pdf" or filename.endswith(".pdf"):
        return AttachmentType.PDF
    elif (
        "wordprocessingml" in content_type
        or "msword" in content_type
        or filename.endswith((".doc", ".docx"))
    ):
        return AttachmentType.WORD
    elif (
        "spreadsheetml" in content_type
        or "ms-excel" in content_type
        or filename.endswith((".xls", ".xlsx"))
    ):
        return AttachmentType.EXCEL
    elif (
        "presentationml" in content_type
        or "ms-powerpoint" in content_type
        or filename.endswith((".ppt", ".pptx"))
    ):
        return AttachmentType.POWERPOINT
    elif content_type.startswith("text/"):
        return AttachmentType.TEXT
    else:
        return AttachmentType.OTHER


def extract_text_from_part(part: email.message.Message) -> Optional[str]:
    """Extract text content from an email part."""
    content_type = part.get_content_type()

    if content_type == "text/plain":
        payload = part.get_payload(decode=True)
        if payload:
            charset = part.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset, errors="replace")
            except (LookupError, UnicodeDecodeError):
                return payload.decode("utf-8", errors="replace")

    elif content_type == "text/html":
        payload = part.get_payload(decode=True)
        if payload:
            charset = part.get_content_charset() or "utf-8"
            try:
                html = payload.decode(charset, errors="replace")
            except (LookupError, UnicodeDecodeError):
                html = payload.decode("utf-8", errors="replace")

            # Basic HTML to text conversion
            # Remove script and style elements
            html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.I)
            # Remove HTML tags
            text = re.sub(r"<[^>]+>", " ", html)
            # Decode common HTML entities
            text = text.replace("&nbsp;", " ")
            text = text.replace("&amp;", "&")
            text = text.replace("&lt;", "<")
            text = text.replace("&gt;", ">")
            text = text.replace("&quot;", '"')
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text

    return None


def parse_email_message(
    msg: email.message.Message,
    mailbox_path: Optional[Path] = None,
    folder_name: Optional[str] = None,
) -> EmailMessage:
    """Parse an email.message.Message into our EmailMessage model."""

    # Get or generate message ID
    message_id = msg.get("Message-ID", "")
    if not message_id:
        # Generate a hash-based ID
        content = str(msg).encode("utf-8", errors="replace")
        message_id = f"<generated-{hashlib.sha256(content).hexdigest()[:16]}>"

    # Parse date
    date = None
    date_str = msg.get("Date")
    if date_str:
        try:
            date = parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            pass

    # Parse body
    body_text = None
    body_html = None
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Check if this is an attachment
            if "attachment" in content_disposition or part.get_filename():
                filename = part.get_filename()
                if filename:
                    filename = decode_mime_header(filename)
                    payload = part.get_payload(decode=True)
                    if payload:
                        attachments.append(
                            Attachment(
                                filename=filename,
                                content_type=content_type,
                                size_bytes=len(payload),
                                attachment_type=get_attachment_type(content_type, filename),
                            )
                        )
            else:
                # Extract body text
                if content_type == "text/plain" and not body_text:
                    body_text = extract_text_from_part(part)
                elif content_type == "text/html" and not body_html:
                    body_html = extract_text_from_part(part)
    else:
        # Single part message
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            body_text = extract_text_from_part(msg)
        elif content_type == "text/html":
            body_html = extract_text_from_part(msg)

    # Use HTML-extracted text if no plain text body
    if not body_text and body_html:
        body_text = body_html
        body_html = None

    return EmailMessage(
        message_id=message_id,
        subject=decode_mime_header(msg.get("Subject")),
        from_address=parse_addresses(msg.get("From"))[0] if msg.get("From") else None,
        to_addresses=parse_addresses(msg.get("To")),
        cc_addresses=parse_addresses(msg.get("Cc")),
        date=date,
        body_text=body_text,
        body_html=body_html,
        attachments=attachments,
        mailbox_path=mailbox_path,
        folder_name=folder_name,
    )


class ThunderbirdParser:
    """Parser for Thunderbird email storage."""

    def __init__(self, profile_path: Path):
        """Initialize with Thunderbird profile path."""
        self.profile_path = profile_path
        self._find_profile_dir()

    def _find_profile_dir(self) -> None:
        """Find the actual profile directory within the Thunderbird path."""
        # Thunderbird stores profiles in subdirectories with random names
        # Look for profiles.ini or find directories with prefs.js
        if not self.profile_path.exists():
            raise FileNotFoundError(f"Thunderbird path not found: {self.profile_path}")

        # Check if this is already a profile directory
        if (self.profile_path / "prefs.js").exists():
            self.profile_dir = self.profile_path
            return

        # Look for profile directories
        for item in self.profile_path.iterdir():
            if item.is_dir() and (item / "prefs.js").exists():
                self.profile_dir = item
                logger.info(f"Found Thunderbird profile: {item}")
                return

        # Check one level deeper (for snap installation structure)
        for item in self.profile_path.iterdir():
            if item.is_dir():
                for subitem in item.iterdir():
                    if subitem.is_dir() and (subitem / "prefs.js").exists():
                        self.profile_dir = subitem
                        logger.info(f"Found Thunderbird profile: {subitem}")
                        return

        raise FileNotFoundError(
            f"No Thunderbird profile found in {self.profile_path}. "
            "Please check your profile path."
        )

    def find_mailboxes(self) -> list[Path]:
        """Find all mbox files in the profile."""
        mailboxes = []

        # Common locations for mail storage
        mail_dirs = [
            self.profile_dir / "Mail",
            self.profile_dir / "ImapMail",
        ]

        for mail_dir in mail_dirs:
            if mail_dir.exists():
                # Find all mbox files (files without extension that have a .msf counterpart)
                for msf_file in mail_dir.rglob("*.msf"):
                    mbox_file = msf_file.with_suffix("")
                    if mbox_file.exists() and mbox_file.is_file():
                        mailboxes.append(mbox_file)

                # Also look for standard folder names without .msf
                for name in ["Inbox", "Sent", "Drafts", "Trash", "Archives"]:
                    for mbox_file in mail_dir.rglob(name):
                        if mbox_file.is_file() and mbox_file not in mailboxes:
                            mailboxes.append(mbox_file)

        logger.info(f"Found {len(mailboxes)} mailboxes")
        return mailboxes

    def parse_mbox(self, mbox_path: Path) -> Generator[EmailMessage, None, None]:
        """Parse an mbox file and yield EmailMessage objects."""
        folder_name = mbox_path.stem

        try:
            mbox = mailbox.mbox(str(mbox_path))
            for msg in mbox:
                try:
                    yield parse_email_message(msg, mbox_path, folder_name)
                except Exception as e:
                    logger.warning(f"Failed to parse message in {mbox_path}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Failed to open mbox {mbox_path}: {e}")

    def parse_all(self) -> Generator[EmailMessage, None, None]:
        """Parse all mailboxes and yield EmailMessage objects."""
        mailboxes = self.find_mailboxes()

        for mbox_path in mailboxes:
            logger.info(f"Parsing {mbox_path}")
            yield from self.parse_mbox(mbox_path)

    def get_attachment_data(
        self, email_msg: EmailMessage, attachment: Attachment
    ) -> Optional[bytes]:
        """Retrieve the raw attachment data from the original email."""
        if not email_msg.mailbox_path:
            return None

        try:
            mbox = mailbox.mbox(str(email_msg.mailbox_path))
            for msg in mbox:
                if msg.get("Message-ID") == email_msg.message_id:
                    # Found the message, now find the attachment
                    for part in msg.walk():
                        filename = part.get_filename()
                        if filename:
                            filename = decode_mime_header(filename)
                            if filename == attachment.filename:
                                return part.get_payload(decode=True)
        except Exception as e:
            logger.error(f"Failed to retrieve attachment: {e}")

        return None
