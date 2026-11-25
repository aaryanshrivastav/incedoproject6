"""Domain models for RAG evaluation golden datasets."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class GoldenType(StrEnum):
    """Category of evaluation test case in the golden dataset."""

    factual = "factual"
    multi_hop = "multi_hop"
    table = "table"
    unanswerable = "unanswerable"
    access_restricted = "access_restricted"


class RestrictionKind(StrEnum):
    """Mechanism used to restrict access to a document for negative access tests."""

    acl_tag = "acl_tag"
    other_hub = "other_hub"


class Provenance(StrEnum):
    """Origin of a golden dataset item."""

    manifest = "manifest"
    llm = "llm"
    handwritten = "handwritten"


class GoldenItem(BaseModel):
    """Single evaluation question and expected ground truth in a golden dataset."""

    id: str = Field(pattern=r"^q-\d{4}$", description="Unique identifier matching ^q-\\d{4}$")
    question: str = Field(min_length=1, description="User query to evaluate")
    reference_answer: str = Field(
        default="",
        description="Expected reference answer (empty for unanswerable/access_restricted)",
    )
    type: GoldenType = Field(description="Evaluation item category")
    hub: str = Field(default="default", min_length=1, description="Target hub name")
    user_tags: list[str] = Field(
        default_factory=list,
        description="ACL tags held by the simulated user for this query",
    )
    source_uris: list[str] = Field(
        default_factory=list,
        description="Relative source URIs of expected supporting documents",
    )
    evidence_spans: list[str] = Field(
        default_factory=list,
        description="Exact text spans expected to appear in a single retrieved chunk",
    )
    restriction_kind: RestrictionKind | None = Field(
        default=None,
        description="Restriction kind when type is access_restricted",
    )
    control_for: str | None = Field(
        default=None,
        description="ID of the access_restricted item this item is the authorized control for",
    )
    provenance: Provenance = Field(description="How this item was authored")
    reviewed: bool = Field(
        default=False,
        description="Whether a human has reviewed and attested this item",
    )
    reviewed_by: str | None = Field(default=None, description="Reviewer identity")
    reviewed_at: datetime | None = Field(default=None, description="UTC review timestamp")
    notes: str = Field(default="", description="Optional curator notes")
    tags: list[str] = Field(default_factory=list, description="Metadata or distinctive token tags")

    @property
    def expected_refusal(self) -> bool:
        """Return True if this item expects the system to refuse answering."""
        return self.type in (GoldenType.unanswerable, GoldenType.access_restricted)

    @model_validator(mode="after")
    def validate_type_rules(self) -> "GoldenItem":
        """Enforce type-dependent invariants on source_uris, spans, answer, and restrictions."""
        if self.type in (GoldenType.factual, GoldenType.table):
            if len(self.source_uris) < 1:
                raise ValueError(f"{self.type.value} items require at least 1 source_uri")
            if len(self.evidence_spans) < 1 or any(not s.strip() for s in self.evidence_spans):
                raise ValueError(
                    f"{self.type.value} items require at least 1 non-empty evidence_span"
                )
            if not self.reference_answer.strip():
                raise ValueError(f"{self.type.value} items require a non-empty reference_answer")

        elif self.type == GoldenType.multi_hop:
            unique_uris = {u.strip() for u in self.source_uris if u.strip()}
            if len(unique_uris) < 2:
                raise ValueError(
                    "multi_hop items require at least 2 source_uris from different documents"
                )
            if len(self.evidence_spans) < 2 or any(not s.strip() for s in self.evidence_spans):
                raise ValueError("multi_hop items require at least 2 non-empty evidence_spans")
            if not self.reference_answer.strip():
                raise ValueError("multi_hop items require a non-empty reference_answer")

        elif self.type == GoldenType.unanswerable:
            if self.source_uris:
                raise ValueError("unanswerable items must have empty source_uris")
            if self.evidence_spans:
                raise ValueError("unanswerable items must have empty evidence_spans")
            if self.reference_answer.strip():
                raise ValueError("unanswerable items must have an empty reference_answer")

        elif self.type == GoldenType.access_restricted:
            if self.restriction_kind is None:
                raise ValueError("access_restricted items require restriction_kind")
            if len(self.source_uris) < 1:
                raise ValueError("access_restricted items require at least 1 source_uri")
            if len(self.evidence_spans) < 1 or any(not s.strip() for s in self.evidence_spans):
                raise ValueError(
                    "access_restricted items require at least 1 non-empty evidence_span"
                )
            if self.reference_answer.strip():
                raise ValueError("access_restricted items must have an empty reference_answer")
            if self.restriction_kind == RestrictionKind.acl_tag:
                # Required ACL tag can be recorded as 'acl:<tag>' in tags; user_tags must exclude it
                required_tags = {t.removeprefix("acl:") for t in self.tags if t.startswith("acl:")}
                if required_tags and required_tags.intersection(self.user_tags):
                    raise ValueError(
                        "access_restricted (acl_tag) item user_tags must not grant access"
                    )

        return self
