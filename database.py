"""Async MongoDB access for saved documents."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

load_dotenv(Path(__file__).resolve().parent / ".env")

_client: AsyncIOMotorClient | None = None


def get_documents_collection() -> AsyncIOMotorCollection:
    """Return the shared documents collection, creating the client on first use."""

    global _client
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        raise MongoConfigurationError(
            "MONGODB_URI is not configured. Set it in the project .env file."
        )
    if _client is None:
        _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    return _client[_database_name(uri)]["documents"]


def close_mongo_client() -> None:
    """Close the shared client during application shutdown."""

    global _client
    if _client is not None:
        _client.close()
        _client = None


class MongoConfigurationError(Exception):
    """Raised when the application cannot select a MongoDB database."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _database_name(uri: str) -> str:
    configured = os.getenv("MONGODB_DB", "").strip()
    if configured:
        return configured
    client = _client
    if client is None:
        raise MongoConfigurationError(
            "MONGODB_DB is not configured, and the MongoDB client is unavailable."
        )
    try:
        return client.get_default_database().name
    except Exception as exc:
        raise MongoConfigurationError(
            "MONGODB_DB is not configured, and MONGODB_URI does not include a database name."
        ) from exc
