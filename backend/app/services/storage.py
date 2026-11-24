"""Cloudflare R2 storage adapter with graceful local-development fallback."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import boto3
from botocore.config import Config

from app.core.config import settings

logger = logging.getLogger(__name__)


class R2Storage:
    def __init__(self) -> None:
        self._client: Any | None = None
        if not all(
            (
                settings.R2_ENDPOINT_URL,
                settings.R2_ACCESS_KEY_ID,
                settings.R2_SECRET_ACCESS_KEY,
                settings.R2_BUCKET_NAME,
            )
        ):
            logger.warning(
                "R2 environment variables are not fully configured; raw PDF persistence is disabled."
            )
            return
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            config=Config(s3={"addressing_style": "path"}),
        )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def upload_pdf(self, session_id: str, doc_id: str, file_bytes: bytes) -> bool:
        if not self._client:
            return False
        key = f"{session_id}/{doc_id}.pdf"
        try:
            await asyncio.to_thread(
                self._client.put_object,
                Bucket=settings.R2_BUCKET_NAME,
                Key=key,
                Body=file_bytes,
                ContentType="application/pdf",
            )
            return True
        except Exception as exc:
            logger.error("Failed to persist PDF in R2: %s", exc)
            return False

    async def delete_document(self, session_id: str, doc_id: str) -> bool:
        if not self._client:
            return False
        try:
            await asyncio.to_thread(
                self._client.delete_object,
                Bucket=settings.R2_BUCKET_NAME,
                Key=f"{session_id}/{doc_id}.pdf",
            )
            return True
        except Exception as exc:
            logger.error("Failed to delete PDF from R2: %s", exc)
            return False

    async def clear_session(self, session_id: str) -> bool:
        if not self._client:
            return False
        try:
            def delete_keys() -> None:
                paginator = self._client.get_paginator("list_objects_v2")
                for page in paginator.paginate(
                    Bucket=settings.R2_BUCKET_NAME,
                    Prefix=f"{session_id}/",
                ):
                    objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
                    if objects:
                        self._client.delete_objects(
                            Bucket=settings.R2_BUCKET_NAME,
                            Delete={"Objects": objects, "Quiet": True},
                        )

            await asyncio.to_thread(delete_keys)
            return True
        except Exception:
            logger.exception("Failed to clear R2 objects for session.")
            return False


r2_storage = R2Storage()
