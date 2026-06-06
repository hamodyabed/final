"""Send the narrative-coding Excel workbook via email (SMTP).

Configuration (env vars, typically loaded from ``.env``):

* ``SMTP_HOST``     — defaults to ``smtp.gmail.com``
* ``SMTP_PORT``     — defaults to ``465`` (SSL) ; use ``587`` for STARTTLS
* ``SMTP_USERNAME`` — full email address used to authenticate
* ``SMTP_PASSWORD`` — for Gmail this MUST be an *App Password* (not your
  account password). Create one at https://myaccount.google.com/apppasswords
* ``SMTP_FROM``     — optional, defaults to ``SMTP_USERNAME``

The function returns ``None`` on success and raises ``EmailerError`` otherwise.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Optional


class EmailerError(RuntimeError):
    """Raised when sending the workbook fails for any reason."""


def send_excel_email(
    *,
    to_address: str,
    subject: str,
    body: str,
    attachment_path: Path,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    from_address: Optional[str] = None,
) -> None:
    """Send ``attachment_path`` (an .xlsx) as an email attachment.

    Raises :class:`EmailerError` with a human-readable Arabic message on any
    configuration or transport failure.
    """
    attachment_path = Path(attachment_path)
    if not attachment_path.is_file():
        raise EmailerError(f"الملف غير موجود: {attachment_path}")

    smtp_host = smtp_host or os.environ.get("SMTP_HOST", "smtp.gmail.com")
    try:
        smtp_port = int(smtp_port or os.environ.get("SMTP_PORT", "465"))
    except ValueError as exc:
        raise EmailerError(f"SMTP_PORT غير صالح: {exc}") from exc
    username = username or os.environ.get("SMTP_USERNAME")
    password = password or os.environ.get("SMTP_PASSWORD")
    if not username or not password:
        raise EmailerError(
            "SMTP_USERNAME أو SMTP_PASSWORD غير معرّف. "
            "أضف بيانات Gmail App Password إلى .env قبل المحاولة."
        )
    from_address = from_address or os.environ.get("SMTP_FROM") or username

    # Build the message.
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_address
    msg["To"] = to_address
    msg.set_content(body)
    msg.add_attachment(
        attachment_path.read_bytes(),
        maintype="application",
        subtype=(
            "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if attachment_path.suffix.lower() == ".xlsx"
            else "octet-stream"
        ),
        filename=attachment_path.name,
    )

    # Send. Prefer SMTPS (port 465); fall back to STARTTLS (587) automatically
    # if the user configured that port instead.
    try:
        if smtp_port == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=ctx, timeout=30) as smtp:
                smtp.login(username, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
                smtp.starttls(context=ssl.create_default_context())
                smtp.login(username, password)
                smtp.send_message(msg)
    except smtplib.SMTPAuthenticationError as exc:
        raise EmailerError(
            "فشلت المصادقة مع خادم البريد. تأكد من استخدام Gmail App Password "
            "وليس كلمة المرور الاعتيادية."
        ) from exc
    except (smtplib.SMTPException, OSError) as exc:
        raise EmailerError(f"تعذّر إرسال البريد: {exc}") from exc
