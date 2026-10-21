"""Pydantic models for routing, extraction, features, and API payloads."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentType(str, Enum):
    """Travel-document categories the router is allowed to return."""

    TICKET = "TICKET"
    STAY = "STAY"
    ACTIVITY = "ACTIVITY"
    UNKNOWN = "UNKNOWN"


class ClassificationResponse(BaseModel):
    """Strict classification schema sent to Gemini structured output.

    This model is sent as a Gemini response schema. extra='forbid' would emit
    additionalProperties, which the Gemini Developer API rejects.
    """

    document_type: DocumentType = Field(
        description=(
            "TICKET for flight, train, or bus tickets and boarding passes. "
            "STAY for hotels and other lodging. "
            "ACTIVITY for tours, attractions, and experiences. "
            "UNKNOWN when the document does not fit those categories or is unreadable."
        )
    )
    provider: str = Field(
        description=(
            "Company that issued or sold the document, for example OYO, Indigo, "
            "or MakeMyTrip. Use UNKNOWN when the provider is not visible."
        )
    )


RouterDecision = ClassificationResponse


class FlightTicketExtractor(BaseModel):
    """Fields extracted from a transport ticket."""

    pnr: str = Field(description="Booking reference or PNR. Empty string if it is not printed.")
    from_city: str = Field(description="Origin city or airport. Empty string if it is not printed.")
    to_city: str = Field(description="Destination city or airport. Empty string if it is not printed.")
    departure_time: str = Field(
        description=(
            "Departure in pure ISO 8601. Use YYYY-MM-DD when only a date is printed, "
            "or YYYY-MM-DDTHH:MM:SS when a time is printed. Empty string if it is not printed."
        )
    )
    number_of_people: int = Field(description="Traveler count. Use 0 when the count is not printed.")
    seat_details: list[str] = Field(
        description="Seat numbers printed on the ticket. Use an empty list when none are printed."
    )


class StayExtractor(BaseModel):
    """Fields extracted from a lodging booking."""

    check_in: str = Field(
        description="Check-in date in pure ISO 8601 format, YYYY-MM-DD. Empty string if it is not printed."
    )
    check_out: str = Field(
        description="Check-out date in pure ISO 8601 format, YYYY-MM-DD. Empty string if it is not printed."
    )
    hotel_name: str = Field(description="Property name. Empty string if it is not printed.")
    location: str = Field(description="City or address. Empty string if it is not printed.")
    total_price: float = Field(description="Numeric booking total without a currency symbol. Use 0 when missing.")
    currency: str = Field(description="Currency code or symbol, such as INR. Empty string if it is not printed.")
    capacity: int = Field(description="Number of guests the booking covers. Use 0 when the count is not printed.")


class DocumentExtraction(BaseModel):
    """One Gemini response that classifies the document and extracts its fields.

    Ticket fields stay empty unless document_type is TICKET.
    Stay fields stay empty unless document_type is STAY.
    """

    document_type: DocumentType = Field(
        description=(
            "TICKET for flight, train, or bus tickets and boarding passes. "
            "STAY for hotels and other lodging. "
            "ACTIVITY for tours, attractions, and experiences. "
            "UNKNOWN when the document does not fit those categories or is unreadable."
        )
    )
    provider: str = Field(
        description=(
            "Company that issued or sold the document, for example OYO, Indigo, IRCTC, "
            "or MakeMyTrip. Use UNKNOWN when the provider is not visible."
        )
    )
    pnr: str = Field(description="Ticket booking reference or PNR. Empty string unless document_type is TICKET.")
    from_city: str = Field(description="Ticket origin. Empty string unless document_type is TICKET.")
    to_city: str = Field(description="Ticket destination. Empty string unless document_type is TICKET.")
    departure_time: str = Field(
        description=(
            "Ticket departure in pure ISO 8601, YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS. "
            "Empty string unless document_type is TICKET."
        )
    )
    number_of_people: int = Field(description="Ticket traveler count. Use 0 unless document_type is TICKET.")
    seat_details: list[str] = Field(
        description="Ticket seat numbers. Use an empty list unless document_type is TICKET."
    )
    check_in: str = Field(
        description="Stay check-in date as YYYY-MM-DD. Empty string unless document_type is STAY."
    )
    check_out: str = Field(
        description="Stay check-out date as YYYY-MM-DD. Empty string unless document_type is STAY."
    )
    hotel_name: str = Field(description="Stay property name. Empty string unless document_type is STAY.")
    location: str = Field(description="Stay city or address. Empty string unless document_type is STAY.")
    total_price: float = Field(
        description="Stay booking total without a currency symbol. Use 0 unless document_type is STAY."
    )
    currency: str = Field(
        description="Stay currency code or symbol, such as INR. Empty string unless document_type is STAY."
    )
    capacity: int = Field(description="Stay guest count. Use 0 unless document_type is STAY.")


class ClassificationResult(BaseModel):
    """Router output returned to API clients."""

    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    provider: str


class DocumentMetadata(BaseModel):
    """File and processing context for one request."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    content_type: str
    media_kind: Literal["pdf", "png", "jpeg"]
    size_bytes: int
    page_count: int
    pages_processed: int
    input_mode: Literal["vision", "text"]
    model: str
    processed_at: str
    warnings: list[str] = Field(default_factory=list)


class StayFeatures(BaseModel):
    """Calculated stay features derived from extracted booking fields."""

    model_config = ConfigDict(extra="forbid")

    stay_duration: int = Field(
        ge=0,
        description="Calendar days between check-out and check-in.",
    )
    price_per_person: float = Field(
        ge=0,
        description="total_price divided by capacity, rounded to two decimal places.",
    )


class TicketFeatures(BaseModel):
    """Baseline ticket features derived from extracted ticket fields.

    seats_per_person is the per-traveler intensity feature, parallel to a stay's
    price_per_person. departure_hour is the schedule feature, parallel to stay_duration.
    """

    model_config = ConfigDict(extra="forbid")

    route: str = Field(description="Origin and destination joined as 'from -> to'.")
    party_size: int = Field(ge=0, description="Traveler count from the ticket.")
    seat_count: int = Field(ge=0, description="Number of seat assignments listed.")
    seats_per_person: float | None = Field(
        description="seat_count divided by party_size. Null when party_size is zero."
    )
    missing_seat_assignments: int = Field(
        ge=0,
        description="How many travelers have no listed seat. Zero when seats cover the party.",
    )
    is_group_travel: bool = Field(description="True when more than one person is traveling.")
    departure_hour: int | None = Field(
        ge=0,
        le=23,
        description="Departure hour from 0 to 23 when the departure value includes a clock time.",
    )


class ProcessDocumentResponse(BaseModel):
    """Successful processing payload."""

    model_config = ConfigDict(extra="forbid")

    metadata: DocumentMetadata
    classification: ClassificationResult
    extracted_fields: FlightTicketExtractor | StayExtractor | None = None
    features: TicketFeatures | StayFeatures | None = None


class SaveDocumentRequest(BaseModel):
    """Reviewed extraction payload stored for one user."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, description="Application user that owns the saved document.")
    document_data: ProcessDocumentResponse

    @field_validator("user_id")
    @classmethod
    def _user_id_required(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("user_id is required.")
        return cleaned


class SaveDocumentResponse(BaseModel):
    """Identifier of the MongoDB document created by POST /documents."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="String form of the inserted MongoDB document id.")


class ErrorResponse(BaseModel):
    """RFC 7807 problem details, plus the flat error fields used by API clients."""

    model_config = ConfigDict(extra="forbid")

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str | None = None
    error: str
    status_code: int
