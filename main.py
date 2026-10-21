"""Headless FastAPI application for document classification and extraction.

Copy .env.example to .env, set GEMINI_API_KEY and MONGODB_URI, then run from this directory:

    uvicorn main:app --host 0.0.0.0 --port 8000

PDF, PNG, and JPEG uploads are sent to Gemini as the original file bytes.
Classification and field extraction happen in one Gemini request.
GEMINI_MODEL overrides the default gemini-2.5-flash model.
GEMINI_THINKING_LEVEL overrides the default low thinking level.
GEMINI_TIMEOUT_SECONDS overrides the default 60 second request timeout.
GOOGLE_API_KEY is also accepted by the Gemini client.
POST /documents stores a reviewed ProcessDocumentResponse in MongoDB.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pymongo.errors import ConnectionFailure, PyMongoError
from starlette.exceptions import HTTPException as StarletteHTTPException

from database import MongoConfigurationError, close_mongo_client, get_documents_collection
from schemas import (
    ErrorResponse,
    ProcessDocumentResponse,
    SaveDocumentRequest,
    SaveDocumentResponse,
)
from services import (
    MAX_UPLOAD_BYTES,
    DocumentProcessingError,
    DocumentService,
    calculate_features,
    prepare_document,
)

logger = logging.getLogger("document_analyzer")

_service: DocumentService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    close_mongo_client()


app = FastAPI(
    title="Document Analyzer API",
    version="1.0.0",
    summary="Headless document classification and extraction API.",
    lifespan=lifespan,
    # docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
)

_ERROR_RESPONSES: dict[int, dict[str, type[ErrorResponse]]] = {
    status_code: {"model": ErrorResponse}
    for status_code in (400, 413, 422, 429, 500, 502, 503, 504)
}


def get_document_service() -> DocumentService:
    """Return a shared Gemini-backed service, creating it on first use."""

    global _service
    if _service is None:
        _service = DocumentService()
    return _service


def problem_response(status_code: int, detail: str, request: Request | None = None) -> JSONResponse:
    """Return one error body that satisfies RFC 7807 and the flat error shape."""

    try:
        title = HTTPStatus(status_code).phrase
    except ValueError:
        title = "Error"
    payload = {
        "type": "about:blank",
        "title": title,
        "status": status_code,
        "detail": detail,
        "error": detail,
        "status_code": status_code,
    }
    if request is not None:
        payload["instance"] = request.url.path
    return JSONResponse(
        status_code=status_code,
        content=payload,
        media_type="application/problem+json",
    )


@app.exception_handler(DocumentProcessingError)
async def document_processing_error_handler(
    request: Request,
    exc: DocumentProcessingError,
) -> JSONResponse:
    return problem_response(exc.status_code, exc.message, request)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
    return problem_response(exc.status_code, detail, request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return problem_response(400, _validation_message(exc), request)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    return problem_response(
        500,
        "An unexpected error occurred while processing the document.",
        request,
    )


@app.post(
    "/api/v1/process-document",
    response_model=ProcessDocumentResponse,
    responses=_ERROR_RESPONSES,
    summary="Classify a document and extract structured fields",
)
async def process_document(file: UploadFile = File(...)) -> ProcessDocumentResponse:
    """Accept one PDF, PNG, or JPEG and return metadata, classification, fields, and features."""
    print(f"Processing document: {file.filename}")
    try:
        data = await _read_upload(file)
        prepared = prepare_document(file.filename, file.content_type, data)
        print(f"Sending {file.filename} as {prepared.input_mode}")
        service = get_document_service()
        started = time.perf_counter()
        analysis = await asyncio.to_thread(service.analyze, prepared)
        print(
            f"Processed {file.filename} as {prepared.input_mode} "
            f"in {time.perf_counter() - started:.2f}s"
        )
        features = calculate_features(
            analysis.classification.document_type,
            analysis.extracted_fields,
        )
        return ProcessDocumentResponse(
            metadata=analysis.metadata,
            classification=analysis.classification,
            extracted_fields=analysis.extracted_fields,
            features=features,
        )
    finally:
        await file.close()


async def _read_upload(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise DocumentProcessingError(413, "The file exceeds the 20 MB upload limit.")
        chunks.append(chunk)
    if total == 0:
        raise DocumentProcessingError(400, "The uploaded file is empty.")
    print(f"Read {total} bytes from {file.filename}")
    return b"".join(chunks)


@app.post(
    "/documents",
    response_model=SaveDocumentResponse,
    status_code=201,
    responses={
        status_code: {"model": ErrorResponse}
        for status_code in (400, 500, 502, 503)
    },
    summary="Save a reviewed document extraction",
)
async def save_document(body: SaveDocumentRequest) -> SaveDocumentResponse:
    """Store one user id and the reviewed process-document payload."""

    try:
        collection = get_documents_collection()
        result = await collection.insert_one(body.model_dump(mode="json"))
    except MongoConfigurationError as exc:
        raise DocumentProcessingError(503, exc.message) from exc
    except ConnectionFailure as exc:
        logger.warning("MongoDB is unavailable: %s", exc)
        raise DocumentProcessingError(503, "The database is unavailable. Try again shortly.") from exc
    except PyMongoError as exc:
        logger.warning("MongoDB insert failed: %s", exc)
        raise DocumentProcessingError(502, "The document could not be saved.") from exc

    return SaveDocumentResponse(id=str(result.inserted_id))


def _validation_message(exc: RequestValidationError) -> str:
    locations = [tuple(error.get("loc", ())) for error in exc.errors()]
    if any(location[-1:] == ("file",) for location in locations):
        return "A multipart form file field named 'file' is required."
    if any("user_id" in location or "document_data" in location for location in locations):
        return "The request body must include a user_id and a document_data object."
    return "The request is invalid. Send one multipart form file in the 'file' field."


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
