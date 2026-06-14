from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime


class EvidenceRef(BaseModel):
    """Reference to supporting evidence from historical policies."""

    document_id: str
    section_id: str
    excerpt: str
    relevance_score: float


class PolicyIssue(BaseModel):
    """An identified issue in the current policy."""

    section_title: str
    severity: Literal["low", "medium", "high"]
    issue_type: Literal[
        "missing_clause",
        "outdated_wording",
        "ambiguity",
        "redundancy",
        "weak_controls",
        "tone_problem",
        "grammar_problem",
        "structure_problem",
        "compliance_gap",
    ]
    description: str
    recommendation: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class PositioningRecommendation(BaseModel):
    """Strategic positioning recommendation for the policy."""

    summary: str
    recommended_positioning: Literal[
        "conservative", "modernized", "simplified",
        "stricter", "broader", "narrower", "role_specific"
    ]
    rationale: list[str]
    key_differences: list[str]
    retained_strengths: list[str]
    top_risks: list[str]


class RatingScorecard(BaseModel):
    """Multi-dimensional quality rating with weighted rubric."""

    overall_label: Literal["poor", "medium", "good", "excellent"]
    overall_score: float = Field(..., ge=0, le=100)

    # Individual dimension scores (0-100)
    structure_score: float = Field(..., ge=0, le=100)
    clarity_score: float = Field(..., ge=0, le=100)
    consistency_score: float = Field(..., ge=0, le=100)
    policy_alignment_score: float = Field(..., ge=0, le=100)
    language_quality_score: float = Field(..., ge=0, le=100)

    dimension_notes: dict[str, str] = Field(default_factory=dict)
    weaknesses_cited: list[str] = Field(default_factory=list)


class GrammarFix(BaseModel):
    """A specific grammar, spelling, or wording correction."""

    issue_type: Literal["grammar", "spelling", "punctuation", "wording", "style", "terminology"]
    original_text: str
    corrected_text: str
    explanation: str
    section_reference: str | None = None


class FinalPolicyPackage(BaseModel):
    """Complete output package from the policy review pipeline."""

    # Metadata
    workflow_id: str
    processed_at: datetime
    current_policy_filename: str
    historical_policies_used: list[str]

    # Analysis results
    similarity_report: list[dict]
    positioning: PositioningRecommendation
    issues: list[PolicyIssue]

    # Generated content
    improved_draft_markdown: str
    copyedited_draft_markdown: str

    # Quality assessment
    scorecard: RatingScorecard
    grammar_fixes: list[GrammarFix]

    # Audit trail
    retrieval_evidence: list[EvidenceRef]
    reviewer_notes: list[str] = Field(default_factory=list)
    approval_status: Literal["pending", "approved", "rejected", "revision_requested"] = "pending"
