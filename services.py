"""Document preparation, Gemini extraction, and feature calculation."""

from __future__ import annotations

import io
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path, PurePath
from typing import Literal, TypeVar

import httpx
from dateutil import parser as date_parser
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from schemas import (
    ClassificationResponse,
    ClassificationResult,
    DocumentExtraction,
    DocumentMetadata,
    DocumentType,
    FlightTicketExtractor,
    StayExtractor,
    StayFeatures,
    TicketFeatures,
)

logger = logging.getLogger("document_analyzer")

load_dotenv(Path(__file__).resolve().parent / ".env")

Image.MAX_IMAGE_PIXELS = 36_000_000

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_TEXT_CHARS = 80
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 60.0

_MIME_TYPES = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpeg": "image/jpeg",
}
_PROVIDER_ALIASES = {
    "oyo": "OYO",
    "oyo rooms": "OYO",
    "oyorooms": "OYO",
    "indigo": "Indigo",
    "goindigo": "Indigo",
    "makemytrip": "MakeMyTrip",
    "make my trip": "MakeMyTrip",
    "mmt": "MakeMyTrip",
    "irctc": "IRCTC",
    "unknown": "UNKNOWN",
    "n/a": "UNKNOWN",
    "na": "UNKNOWN",
    "none": "UNKNOWN",
}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%b %d %Y",
    "%B %d %Y",
)
_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d %b %Y %I:%M %p",
    "%d %B %Y %I:%M %p",
    "%d %b %Y %H:%M",
    "%d %B %Y %H:%M",
    "%b %d %Y %I:%M %p",
    "%B %d %Y %I:%M %p",
    "%I:%M %p",
    "%H:%M:%S",
    "%H:%M",
)
_TIME_HINT = re.compile(r"\d{1,2}:\d{2}|\d{1,2}\s*[AP]M\b", re.IGNORECASE)
_MONTH_HINT = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_HINT = re.compile(r"\d{1,4}[./-]\d{1,2}(?:[./-]\d{1,4})?")

_ISO_8601_RULE = (
    "Standardize every extracted date and timestamp to pure ISO 8601 format (YYYY-MM-DD). "
    "When a clock time is printed, use YYYY-MM-DDTHH:MM:SS. "
    "Never return localized dates such as 12 Mar 2026, 12/03/2026, or 03-12-2026."
)

CLASSIFICATION_CONFIG = types.GenerateContentConfig(
    system_instruction=(
        "You route travel documents. Choose exactly one document_type. "
        "Use TICKET for flight, train, or bus tickets and boarding passes. "
        "Use STAY for hotels, hostels, homestays, and other lodging bookings. "
        "Use ACTIVITY for tours, attractions, experiences, and event bookings that are not transport or lodging. "
        "Use UNKNOWN when the document is unreadable or none of those categories. "
        "Set provider to the company that issued or sold the document, such as OYO, Indigo, or MakeMyTrip. "
        "Use UNKNOWN when no provider is visible. Do not guess a provider you cannot read. "
        f"{_ISO_8601_RULE}"
    ),
    temperature=0,
    response_mime_type="application/json",
    response_schema=ClassificationResponse,
)
STAY_CONFIG = types.GenerateContentConfig(
    system_instruction=(
        "Extract the lodging booking fields that are printed on the document. "
        "total_price is the numeric booking total, without a currency symbol. "
        "currency is the printed currency code or symbol, such as INR. "
        "capacity is the number of guests the booking covers. "
        "Use an empty string for missing text and 0 for missing numbers. "
        "Do not invent prices, dates, or guest counts. "
        "check_in and check_out must be pure ISO 8601 dates so stay duration can be calculated. "
        f"{_ISO_8601_RULE}"
    ),
    temperature=0,
    response_mime_type="application/json",
    response_schema=StayExtractor,
)
FLIGHT_CONFIG = types.GenerateContentConfig(
    system_instruction=(
        "Extract the transport ticket fields that are printed on the document. "
        "Use an empty string for text you cannot find, 0 for number_of_people when the traveler count is not shown, "
        "and an empty list for seat_details when no seats are listed. "
        "Do not invent passenger counts, cities, or seat numbers. "
        "departure_time must be pure ISO 8601 so schedule features can be calculated. "
        f"{_ISO_8601_RULE}"
    ),
    temperature=0,
    response_mime_type="application/json",
    response_schema=FlightTicketExtractor,
)
DOCUMENT_CONFIG = types.GenerateContentConfig(
    system_instruction=(
        "Classify this travel document and extract its printed fields in one pass. "
        "Choose exactly one document_type. "
        "Use TICKET for flight, train, or bus tickets and boarding passes. "
        "Use STAY for hotels, hostels, homestays, and other lodging bookings. "
        "Use ACTIVITY for tours, attractions, experiences, and event bookings that are not transport or lodging. "
        "Use UNKNOWN when the document is unreadable or none of those categories. "
        "Set provider to the company that issued or sold the document, such as OYO, Indigo, IRCTC, or MakeMyTrip. "
        "Use UNKNOWN when no provider is visible. Do not guess a provider you cannot read. "
        "When document_type is TICKET, fill the ticket fields and leave every stay field empty or zero. "
        "When document_type is STAY, fill the stay fields and leave every ticket field empty or zero. "
        "When document_type is ACTIVITY or UNKNOWN, leave every extracted field empty or zero. "
        "Use an empty string for missing text, 0 for missing numbers, and an empty list when no seats are listed. "
        "Do not invent passenger counts, prices, dates, cities, or seat numbers. "
        "total_price is the numeric booking total, without a currency symbol. "
        f"{_ISO_8601_RULE}"
    ),
    temperature=0,
    response_mime_type="application/json",
    response_schema=DocumentExtraction,
)

ParsedModel = TypeVar("ParsedModel", bound=BaseModel)
MediaKind = Literal["pdf", "png", "jpeg"]
InputMode = Literal["vision", "text"]


class DocumentProcessingError(Exception):
    """Expected processing failure with an HTTP status and a client-safe message."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(slots=True)
class PreparedDocument:
    filename: str
    content_type: str
    media_kind: MediaKind
    size_bytes: int
    page_count: int
    pages_processed: int
    input_mode: InputMode
    file_bytes: bytes
    warnings: list[str]
    text: str | None = None


@dataclass(slots=True)
class Analysis:
    metadata: DocumentMetadata
    classification: ClassificationResult
    extracted_fields: FlightTicketExtractor | StayExtractor | None


def _timeout_from_env() -> float:
    raw = os.getenv("GEMINI_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning("GEMINI_TIMEOUT_SECONDS is not a number. Using the default timeout.")
        return DEFAULT_TIMEOUT_SECONDS
    if value <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return value


def prepare_document(
    filename: str | None,
    content_type: str | None,
    data: bytes,
) -> PreparedDocument:
    """Validate an upload and keep the original bytes for Gemini."""

    if not data:
        raise DocumentProcessingError(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentProcessingError(400, "The file exceeds the 20 MB upload limit.")

    media_kind = _detect_media_kind(data)
    if media_kind == "pdf":
        prepared = _prepare_pdf(data)
    else:
        prepared = _prepare_image(data, media_kind)

    prepared.filename = _safe_filename(filename)
    prepared.content_type = _MIME_TYPES[media_kind]
    prepared.size_bytes = len(data)
    _warn_if_content_type_conflicts(prepared, content_type)
    return prepared


def calculate_features(
    document_type: DocumentType,
    extracted_fields: FlightTicketExtractor | StayExtractor | None,
) -> TicketFeatures | StayFeatures | None:
    """Build response features from a routing decision and its extractor payload."""

    if document_type is DocumentType.STAY:
        if not isinstance(extracted_fields, StayExtractor):
            raise DocumentProcessingError(422, "Stay document did not produce stay fields.")
        return _stay_features(extracted_fields)
    if document_type is DocumentType.TICKET:
        if not isinstance(extracted_fields, FlightTicketExtractor):
            raise DocumentProcessingError(422, "Ticket document did not produce ticket fields.")
        return _ticket_features(extracted_fields)
    return None


def classify_document(
    file_bytes: bytes,
    mime_type: str,
    *,
    client: genai.Client,
    model: str,
) -> ClassificationResponse:
    """Classify a document from its original file bytes."""

    return _generate_structured(
        client=client,
        model=model,
        file_bytes=file_bytes,
        mime_type=mime_type,
        instruction="Classify this document.",
        config=CLASSIFICATION_CONFIG,
        schema=ClassificationResponse,
    )


def extract_stay(
    file_bytes: bytes,
    mime_type: str,
    *,
    client: genai.Client,
    model: str,
) -> StayExtractor:
    """Extract hotel booking fields from the original file bytes."""

    return _generate_structured(
        client=client,
        model=model,
        file_bytes=file_bytes,
        mime_type=mime_type,
        instruction="Extract the hotel stay fields from this document.",
        config=STAY_CONFIG,
        schema=StayExtractor,
    )


def extract_flight(
    file_bytes: bytes,
    mime_type: str,
    *,
    client: genai.Client,
    model: str,
) -> FlightTicketExtractor:
    """Extract flight ticket fields from the original file bytes."""

    return _generate_structured(
        client=client,
        model=model,
        file_bytes=file_bytes,
        mime_type=mime_type,
        instruction="Extract the flight ticket fields from this document.",
        config=FLIGHT_CONFIG,
        schema=FlightTicketExtractor,
    )


def extract_document(
    file_bytes: bytes,
    mime_type: str,
    *,
    client: genai.Client,
    model: str,
    text: str | None = None,
) -> DocumentExtraction:
    """Classify a document and extract its fields in one Gemini request."""

    return _generate_structured(
        client=client,
        model=model,
        file_bytes=file_bytes,
        mime_type=mime_type,
        instruction="Classify this document and extract its fields.",
        config=DOCUMENT_CONFIG,
        schema=DocumentExtraction,
        text=text,
    )


class DocumentService:
    """Classify a prepared document with Gemini and extract the routed schema."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: genai.Client | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        resolved_model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
        self.model = resolved_model or DEFAULT_MODEL
        self._timeout = _timeout_from_env() if timeout_seconds is None else timeout_seconds
        if self._timeout <= 0:
            self._timeout = DEFAULT_TIMEOUT_SECONDS
        if client is not None:
            self.client = client
            return

        resolved_key = (api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")).strip()
        if not resolved_key:
            resolved_key = os.getenv("GOOGLE_API_KEY", "").strip()
        http_options = types.HttpOptions(timeout=int(self._timeout * 1000))
        try:
            if resolved_key:
                client = genai.Client(api_key=resolved_key, http_options=http_options)
            else:
                client = genai.Client(http_options=http_options)
        except ValueError as exc:
            raise DocumentProcessingError(
                503,
                "GEMINI_API_KEY is not configured. Set it in the project .env file.",
            ) from exc
        self.client = client

    def analyze(self, prepared: PreparedDocument) -> Analysis:
        started = time.perf_counter()
        decision = extract_document(
            prepared.file_bytes,
            prepared.content_type,
            client=self.client,
            model=self.model,
            text=prepared.text,
        )
        logger.info(
            "Gemini extraction finished in %.2fs using %s thinking",
            time.perf_counter() - started,
            _thinking_level_name(),
        )
        classification = ClassificationResult(
            document_type=decision.document_type,
            provider=_normalize_provider(decision.provider),
        )
        warnings = list(prepared.warnings)
        extracted: FlightTicketExtractor | StayExtractor | None = None

        if decision.document_type is DocumentType.TICKET:
            extracted = _clean_ticket(_ticket_from_extraction(decision))
            if _ticket_is_empty(extracted):
                raise DocumentProcessingError(
                    422,
                    "The document was classified as a ticket, but no ticket fields could be extracted.",
                )
        elif decision.document_type is DocumentType.STAY:
            extracted = _clean_stay(_stay_from_extraction(decision))
            if _stay_is_empty(extracted):
                raise DocumentProcessingError(
                    422,
                    "The document was classified as a stay, but no stay fields could be extracted.",
                )
        elif decision.document_type is DocumentType.ACTIVITY:
            warnings.append("No field extractor is configured for ACTIVITY documents.")
        else:
            warnings.append("The document type could not be determined, so no fields were extracted.")

        logger.info(
            "Processed %s document as %s (%s) using %s",
            prepared.media_kind,
            classification.document_type.value,
            classification.provider,
            prepared.input_mode,
        )
        return Analysis(
            metadata=DocumentMetadata(
                filename=prepared.filename,
                content_type=prepared.content_type,
                media_kind=prepared.media_kind,
                size_bytes=prepared.size_bytes,
                page_count=prepared.page_count,
                pages_processed=prepared.pages_processed,
                input_mode=prepared.input_mode,
                model=self.model,
                processed_at=datetime.now(timezone.utc).isoformat(),
                warnings=warnings,
            ),
            classification=classification,
            extracted_fields=extracted,
        )


def _generate_structured(
    *,
    client: genai.Client,
    model: str,
    file_bytes: bytes,
    mime_type: str,
    instruction: str,
    config: types.GenerateContentConfig,
    schema: type[ParsedModel],
    text: str | None = None,
) -> ParsedModel:
    contents = _model_contents(
        file_bytes=file_bytes,
        mime_type=mime_type,
        instruction=instruction,
        text=text,
    )
    request_config = config.model_copy(update={"thinking_config": _thinking_config(model)})
    started = time.perf_counter()
    for attempt in (1, 2):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=request_config,
            )
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise DocumentProcessingError(
                504,
                "The Gemini request timed out before the document could be processed.",
            ) from exc
        except errors.APIError as exc:
            logger.warning("Gemini status %s: %s", exc.code, exc.message)
            if attempt == 1 and exc.code in {429, 503}:
                logger.warning("Retrying Gemini once after status %s.", exc.code)
                time.sleep(1)
                continue
            if exc.code in {401, 403}:
                raise DocumentProcessingError(503, "The Gemini API key was rejected.") from exc
            if exc.code == 429:
                raise DocumentProcessingError(
                    429,
                    "The Gemini rate limit was reached. Try again shortly.",
                ) from exc
            if exc.code == 503:
                raise DocumentProcessingError(
                    503,
                    "Gemini is busy right now. Try again shortly.",
                ) from exc
            if exc.code in {408, 504}:
                raise DocumentProcessingError(
                    504,
                    "The Gemini request timed out before the document could be processed.",
                ) from exc
            raise DocumentProcessingError(
                502,
                "The Gemini API returned an error while processing the document.",
            ) from exc
        except ValidationError as exc:
            logger.info("Model output failed schema validation (%s errors).", exc.error_count())
            raise DocumentProcessingError(
                422,
                "The model response did not match the extraction schema.",
            ) from exc

        logger.info(
            "Gemini %s call finished in %.2fs",
            schema.__name__,
            time.perf_counter() - started,
        )
        return _coerce_schema(response, schema)
    raise DocumentProcessingError(502, "The Gemini API returned an error while processing the document.")


def _model_contents(
    *,
    file_bytes: bytes,
    mime_type: str,
    instruction: str,
    text: str | None,
) -> list[object]:
    """Send extracted text when the PDF has a text layer. Scans and images stay as files."""

    cleaned = (text or "").strip()
    if cleaned:
        return [f"{instruction}\n\nDocument text:\n{cleaned}"]
    return [
        types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
        instruction,
    ]


def _coerce_schema(response: object, schema: type[ParsedModel]) -> ParsedModel:
    feedback = getattr(response, "prompt_feedback", None)
    if getattr(feedback, "block_reason", None):
        raise DocumentProcessingError(422, "The model declined to process this document.")

    parsed = getattr(response, "parsed", None)
    try:
        if isinstance(parsed, schema):
            return parsed
        if parsed is not None:
            return schema.model_validate(parsed)
        text = getattr(response, "text", None) or ""
        if text.strip():
            return schema.model_validate_json(text)
    except ValidationError as exc:
        logger.info("Model output failed schema validation (%s errors).", exc.error_count())
        raise DocumentProcessingError(
            422,
            "The model response did not match the extraction schema.",
        ) from exc
    raise DocumentProcessingError(422, "The model returned an empty response.")


def _prepare_pdf(data: bytes) -> PreparedDocument:
    text, page_count = _read_pdf_text(data)
    warnings: list[str] = []
    if page_count == 0:
        raise DocumentProcessingError(400, "The PDF does not contain any pages.")
    if page_count is None:
        warnings.append(
            "The PDF page count could not be determined. The original file was sent to the model."
        )
        resolved_pages = 1
        input_mode: InputMode = "vision"
    else:
        resolved_pages = page_count
        input_mode = "text" if _usable_text(text) else "vision"

    document_text = text if input_mode == "text" else None
    pages_processed = min(resolved_pages, 8) if document_text else resolved_pages
    if document_text and resolved_pages > pages_processed:
        warnings.append(
            f"Only the first {pages_processed} of {resolved_pages} pages were sent to the model."
        )

    return PreparedDocument(
        filename="",
        content_type=_MIME_TYPES["pdf"],
        media_kind="pdf",
        size_bytes=len(data),
        page_count=resolved_pages,
        pages_processed=pages_processed,
        input_mode=input_mode,
        file_bytes=data,
        warnings=warnings,
        text=document_text,
    )


def _prepare_image(data: bytes, media_kind: MediaKind) -> PreparedDocument:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            allowed_formats = {"PNG"} if media_kind == "png" else {"JPEG", "MPO"}
            if image.format not in allowed_formats:
                raise DocumentProcessingError(400, "Unsupported image type. Upload a PNG or JPEG.")
    except DocumentProcessingError:
        raise
    except Image.DecompressionBombError as exc:
        raise DocumentProcessingError(400, "The image is too large to process.") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise DocumentProcessingError(400, "The image file is invalid or corrupted.") from exc

    return PreparedDocument(
        filename="",
        content_type=_MIME_TYPES[media_kind],
        media_kind=media_kind,
        size_bytes=len(data),
        page_count=1,
        pages_processed=1,
        input_mode="vision",
        file_bytes=data,
        warnings=[],
    )


def _read_pdf_text(data: bytes) -> tuple[str, int | None]:
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError:
        return "", None
    except Exception:
        logger.info("PDF text extraction could not open the file.", exc_info=True)
        return "", None

    if reader.is_encrypted and int(reader.decrypt("")) == 0:
        raise DocumentProcessingError(400, "Encrypted PDFs are not supported.")

    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise DocumentProcessingError(400, "The PDF could not be read.") from exc

    chunks: list[str] = []
    for page in reader.pages[:8]:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            logger.info("Skipping a PDF page whose text could not be extracted.", exc_info=True)
            chunks.append("")
    return "\n\n".join(chunks), page_count


def _warn_if_content_type_conflicts(prepared: PreparedDocument, content_type: str | None) -> None:
    declared = (content_type or "").split(";", 1)[0].strip().lower()
    compatible = {
        "pdf": {"application/pdf", "application/x-pdf"},
        "png": {"image/png"},
        "jpeg": {"image/jpeg", "image/jpg", "image/pjpeg"},
    }
    if not declared or declared == "application/octet-stream" or declared in compatible[prepared.media_kind]:
        return
    prepared.warnings.append(
        "The declared content type did not match the file contents. The detected file type was used."
    )


def _detect_media_kind(data: bytes) -> MediaKind:
    if data.startswith(b"%PDF"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    raise DocumentProcessingError(
        400,
        "Unsupported or unrecognized file. Upload a PDF, PNG, or JPEG.",
    )


def _safe_filename(filename: str | None) -> str:
    name = PurePath(filename or "").name.replace("\x00", "").strip()
    return name or "upload"


def _usable_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return len(compact) >= MIN_TEXT_CHARS


def _thinking_level_name() -> str:
    raw = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
    if raw in {"minimal", "low", "medium", "high"}:
        return raw
    return "low"


def _thinking_config(model: str) -> types.ThinkingConfig:
    """Use low thinking unless GEMINI_THINKING_LEVEL asks for more.

    gemini-3.8-flash defaults to medium thinking, which is slow for field extraction.
    Gemini 2.5 uses a token budget. Later Gemini models use a thinking level, and
    gemini-3.8-flash rejects the minimal level.
    """

    level = _thinking_level_name()
    name = model.lower()
    if "gemini-2.5" in name or "gemini-2-5" in name:
        budget = 0 if level in {"minimal", "low"} else -1
        return types.ThinkingConfig(thinking_budget=budget)
    if level == "minimal" and "3.8" in name:
        level = "low"
    levels = {
        "minimal": types.ThinkingLevel.MINIMAL,
        "low": types.ThinkingLevel.LOW,
        "medium": types.ThinkingLevel.MEDIUM,
        "high": types.ThinkingLevel.HIGH,
    }
    return types.ThinkingConfig(thinking_level=levels[level])


def _ticket_from_extraction(document: DocumentExtraction) -> FlightTicketExtractor:
    return FlightTicketExtractor(
        pnr=document.pnr,
        from_city=document.from_city,
        to_city=document.to_city,
        departure_time=document.departure_time,
        number_of_people=document.number_of_people,
        seat_details=document.seat_details,
    )


def _stay_from_extraction(document: DocumentExtraction) -> StayExtractor:
    return StayExtractor(
        check_in=document.check_in,
        check_out=document.check_out,
        hotel_name=document.hotel_name,
        location=document.location,
        total_price=document.total_price,
        currency=document.currency,
        capacity=document.capacity,
    )


def _normalize_provider(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return "UNKNOWN"
    alias = _PROVIDER_ALIASES.get(cleaned.casefold())
    if alias:
        return alias
    if len(cleaned) > 80:
        return cleaned[:80].rstrip()
    return cleaned


def _clean_ticket(ticket: FlightTicketExtractor) -> FlightTicketExtractor:
    return ticket.model_copy(
        update={
            "pnr": ticket.pnr.strip(),
            "from_city": " ".join(ticket.from_city.split()),
            "to_city": " ".join(ticket.to_city.split()),
            "departure_time": " ".join(ticket.departure_time.split()),
            "seat_details": [seat.strip() for seat in ticket.seat_details if seat and seat.strip()],
        }
    )


def _clean_stay(stay: StayExtractor) -> StayExtractor:
    currency = stay.currency.strip().upper()
    return stay.model_copy(
        update={
            "check_in": " ".join(stay.check_in.split()),
            "check_out": " ".join(stay.check_out.split()),
            "hotel_name": " ".join(stay.hotel_name.split()),
            "location": " ".join(stay.location.split()),
            "currency": currency,
        }
    )


def _ticket_is_empty(ticket: FlightTicketExtractor) -> bool:
    return not any(
        (
            ticket.pnr,
            ticket.from_city,
            ticket.to_city,
            ticket.departure_time,
            ticket.number_of_people,
            ticket.seat_details,
        )
    )


def _stay_is_empty(stay: StayExtractor) -> bool:
    return not any(
        (
            stay.check_in,
            stay.check_out,
            stay.hotel_name,
            stay.location,
            stay.total_price,
            stay.currency,
            stay.capacity,
        )
    )


def _stay_features(stay: StayExtractor) -> StayFeatures:
    try:
        check_in = _parse_calendar_date(stay.check_in)
        check_out = _parse_calendar_date(stay.check_out)
    except ValueError as exc:
        raise DocumentProcessingError(
            422,
            "Could not calculate stay_duration because check-in or check-out is not a recognizable date.",
        ) from exc

    stay_duration = (check_out - check_in).days
    if stay_duration < 0:
        raise DocumentProcessingError(
            422,
            "Could not calculate stay_duration because check-out is earlier than check-in.",
        )
    return StayFeatures(
        stay_duration=stay_duration,
        price_per_person=_price_per_person(stay.total_price, stay.capacity),
    )


def _ticket_features(ticket: FlightTicketExtractor) -> TicketFeatures:
    if ticket.number_of_people < 0:
        raise DocumentProcessingError(422, "Ticket extraction returned a negative number_of_people.")

    party_size = ticket.number_of_people
    seat_count = len(ticket.seat_details)
    seats_per_person = None
    if party_size > 0:
        seats_per_person = round(seat_count / party_size, 4)

    origin = ticket.from_city or "UNKNOWN"
    destination = ticket.to_city or "UNKNOWN"
    return TicketFeatures(
        route=f"{origin} -> {destination}",
        party_size=party_size,
        seat_count=seat_count,
        seats_per_person=seats_per_person,
        missing_seat_assignments=max(party_size - seat_count, 0),
        is_group_travel=party_size > 1,
        departure_hour=_parse_departure_hour(ticket.departure_time),
    )


def _price_per_person(total_price: float, capacity: int) -> float:
    if capacity <= 0:
        raise DocumentProcessingError(
            422,
            "Could not calculate price_per_person because capacity is not a positive number.",
        )
    if not math.isfinite(total_price) or total_price < 0:
        raise DocumentProcessingError(
            422,
            "Could not calculate price_per_person because total_price must be zero or greater.",
        )
    try:
        amount = (Decimal(str(total_price)) / Decimal(capacity)).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
    except (InvalidOperation, ZeroDivisionError) as exc:
        raise DocumentProcessingError(
            422,
            "Could not calculate price_per_person from total_price and capacity.",
        ) from exc
    return float(amount)


def _normalize_temporal_text(value: str) -> str:
    text = value.strip().replace(",", " ")
    text = re.sub(r"(\d)T(\d)", r"\1 \2", text)
    text = re.sub(r"\b(?:hrs|hours|ist|utc|gmt)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)(AM|PM)\b", r" \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(am|pm)\b", lambda match: match.group(1).upper(), text, flags=re.IGNORECASE)
    return " ".join(text.split())


def _parse_calendar_date(value: str) -> date:
    text = _normalize_temporal_text(value)
    if not text:
        raise ValueError("empty date")

    for fmt in (*_DATETIME_FORMATS, *_DATE_FORMATS):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    digits = sum(character.isdigit() for character in text)
    has_date_shape = bool(_MONTH_HINT.search(text) or _NUMERIC_DATE_HINT.search(text) or re.search(r"\d{4}", text))
    if digits < 4 or not has_date_shape:
        raise ValueError("unrecognized date")

    try:
        parsed = date_parser.parse(text, fuzzy=True, dayfirst=True)
    except (ValueError, OverflowError, TypeError) as exc:
        raise ValueError("unrecognized date") from exc
    return parsed.date()


def _parse_departure_hour(value: str) -> int | None:
    text = _normalize_temporal_text(value)
    if not text or not _TIME_HINT.search(text):
        return None

    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).hour
        except ValueError:
            continue

    try:
        return date_parser.parse(text, fuzzy=True, dayfirst=True).hour
    except (ValueError, OverflowError, TypeError):
        return None

