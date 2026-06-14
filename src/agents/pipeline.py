import json
import logging
import uuid
from datetime import datetime, timezone

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..config.settings import settings
from ..retrieval.hybrid_retriever import HybridRetriever
from ..rating.rubric import RatingRubric
from ..schemas.documents import RetrievalResult
from ..schemas.outputs import (
    EvidenceRef,
    FinalPolicyPackage,
    GrammarFix,
    PolicyIssue,
    PositioningRecommendation,
    RatingScorecard,
)

logger = logging.getLogger(__name__)


def _parse_json_response(text: str, fallback):
    """
    Attempt to extract and parse the first JSON object or array from text.
    Returns fallback value on any parse error.
    """
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first JSON block inside the text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
    return fallback


class PolicyReviewPipeline:
    """
    Simple linear pipeline for Phase 1 MVP policy review.

    Stages (executed in order):
      1. retrieve   — hybrid BM25 + dense search for similar historical docs
      2. compare    — LLM comparison of current policy vs retrieved evidence
      3. position   — LLM strategic positioning recommendation
      4. identify   — LLM issue identification
      5. rewrite    — LLM policy rewrite addressing issues
      6. rate       — LLM quality scoring with weighted rubric
      7. review     — LLM grammar / language copyedit

    Returns a FinalPolicyPackage with all outputs bundled.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_model: str = settings.llm_model,
    ) -> None:
        self.retriever = retriever
        self.llm = ChatAnthropic(
            model=llm_model,
            api_key=settings.anthropic_api_key,
            max_tokens=8192,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _invoke(self, system: str, human: str) -> str:
        """Call the LLM with a system + human message pair."""
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=human),
        ]
        response = self.llm.invoke(messages)
        return response.content  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        policy_content: str,
        current_doc_id: str | None,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Stage 1: Hybrid retrieval of similar historical documents."""
        query = (
            "Find policy documents similar to:\n"
            + policy_content[:2000]
        )
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            exclude_document_id=current_doc_id,
        )

    def _compare(
        self,
        policy_content: str,
        retrieved: list[RetrievalResult],
    ) -> tuple[list[dict], list[dict]]:
        """Stage 2: Compare current policy against historical evidence."""
        evidence_text = "\n\n---\n\n".join(
            f"**{r.section_title}**\n{r.content}"
            for r in retrieved[:7]
        )

        system = (
            "You are a policy comparison expert. Compare the current policy "
            "against historical evidence and identify key similarities, differences, "
            "missing elements, outdated elements, and stronger/weaker controls. "
            "Return a JSON array of finding objects, each with keys: "
            "type, description, severity (low/medium/high)."
        )
        human = (
            "## Current Policy\n"
            "<policy_content>\n"
            f"{policy_content}\n"
            "</policy_content>\n\n"
            "## Historical Evidence\n"
            "<evidence>\n"
            f"{evidence_text}\n"
            "</evidence>\n\n"
            "Return JSON array of findings:"
        )

        raw = self._invoke(system, human)
        findings = _parse_json_response(raw, [])

        similarity_report = [
            {
                "document_id": r.document_id,
                "section_title": r.section_title,
                "rrf_score": r.rrf_score,
                "bm25_score": r.bm25_score,
                "dense_score": r.dense_score,
                "excerpt": r.content[:300],
            }
            for r in retrieved
        ]

        return findings, similarity_report

    def _position(
        self,
        policy_content: str,
        findings: list[dict],
    ) -> dict:
        """Stage 3: Generate strategic positioning recommendation."""
        system = (
            "You are a policy strategy advisor. Based on the comparison findings, "
            "recommend how the current policy should be positioned.\n\n"
            "Return JSON with keys:\n"
            '  "summary": str,\n'
            '  "recommended_positioning": one of '
            '["conservative","modernized","simplified","stricter","broader","narrower","role_specific"],\n'
            '  "rationale": [str],\n'
            '  "key_differences": [str],\n'
            '  "retained_strengths": [str],\n'
            '  "top_risks": [str]'
        )
        human = (
            "## Current Policy Summary\n"
            "<policy_content>\n"
            f"{policy_content[:3000]}\n"
            "</policy_content>\n\n"
            "## Comparison Findings\n"
            "<findings>\n"
            f"{json.dumps(findings, indent=2)}\n"
            "</findings>\n\n"
            "Recommend positioning as JSON:"
        )

        raw = self._invoke(system, human)
        positioning = _parse_json_response(
            raw,
            None,
        )
        if positioning is None:
            logger.warning(
                "LLM returned unparseable positioning response; using safe defaults. "
                "Raw response (first 200 chars): %.200s", raw
            )
            positioning = {
                "summary": "Analysis incomplete",
                "recommended_positioning": "modernized",
                "rationale": [],
                "key_differences": [],
                "retained_strengths": [],
                "top_risks": [],
            }
        return positioning

    def _identify_issues(
        self,
        policy_content: str,
        findings: list[dict],
        retrieved: list[RetrievalResult],
    ) -> list[dict]:
        """Stage 4: Identify specific issues in the current policy."""
        evidence_items = retrieved[:5]
        evidence_text = "\n".join(
            f"[{i}] {r.section_title} (doc={r.document_id}): {r.content[:200]}…"
            for i, r in enumerate(evidence_items)
        )

        system = (
            "You are a policy analyst. Identify specific issues in the policy.\n\n"
            "For each issue provide JSON with:\n"
            '  "section_title": str,\n'
            '  "severity": "low"|"medium"|"high",\n'
            '  "issue_type": one of '
            '["missing_clause","outdated_wording","ambiguity","redundancy",'
            '"weak_controls","tone_problem","grammar_problem","structure_problem","compliance_gap"],\n'
            '  "description": str,\n'
            '  "recommendation": str,\n'
            '  "evidence_indices": [int, ...]  '
            "// zero-based indices into the Historical Evidence list that support this issue\n\n"
            "Return a JSON array of issue objects."
        )
        human = (
            "## Current Policy\n"
            "<policy_content>\n"
            f"{policy_content}\n"
            "</policy_content>\n\n"
            "## Comparison Findings\n"
            "<findings>\n"
            f"{json.dumps(findings, indent=2)}\n"
            "</findings>\n\n"
            "## Historical Evidence (reference by index)\n"
            "<evidence>\n"
            f"{evidence_text}\n"
            "</evidence>\n\n"
            "Identify issues as JSON array:"
        )

        raw = self._invoke(system, human)
        return _parse_json_response(raw, [])

    def _rewrite(
        self,
        policy_content: str,
        positioning: dict,
        issues: list[dict],
        retrieved: list[RetrievalResult],
    ) -> str:
        """Stage 5: Rewrite the policy addressing identified issues."""
        evidence_text = "\n\n".join(
            f"### {r.section_title}\n{r.content}"
            for r in retrieved[:5]
        )

        system = (
            "You are an expert policy writer. Rewrite the policy to address the "
            "identified issues while following the positioning recommendation.\n\n"
            "Rules:\n"
            "- Preserve policy intent unless the issue explicitly requires a change\n"
            "- Improve structure, clarity, and professional tone\n"
            "- Reference strong language from historical evidence where appropriate\n"
            "- Mark any material change with [MATERIAL CHANGE] inline\n"
            "- Return clean Markdown only"
        )
        human = (
            "## Original Policy\n"
            "<policy_content>\n"
            f"{policy_content}\n"
            "</policy_content>\n\n"
            "## Positioning Recommendation\n"
            "<positioning>\n"
            f"{json.dumps(positioning, indent=2)}\n"
            "</positioning>\n\n"
            "## Issues to Address\n"
            "<issues>\n"
            f"{json.dumps(issues, indent=2)}\n"
            "</issues>\n\n"
            "## Historical Evidence\n"
            "<evidence>\n"
            f"{evidence_text}\n"
            "</evidence>\n\n"
            "Rewrite the improved policy in Markdown:"
        )

        return self._invoke(system, human)

    def _rate(self, policy_content: str) -> dict:
        """Stage 6: Score the draft with the weighted rubric."""
        system = (
            "You are a policy quality assessor. Score the policy on each dimension "
            "from 0-100 and return a JSON scorecard.\n\n"
            + RatingRubric.get_rating_prompt()
            + "\n\nReturn JSON with keys: "
            "overall_score, overall_label, structure_score, clarity_score, "
            "consistency_score, policy_alignment_score, language_quality_score, "
            "dimension_notes (dict), weaknesses_cited (list)."
        )
        human = (
            "## Policy to Rate\n"
            "<policy_content>\n"
            f"{policy_content}\n"
            "</policy_content>\n\n"
            "Return JSON scorecard:"
        )

        raw = self._invoke(system, human)
        scorecard = _parse_json_response(raw, None)

        if scorecard is None:
            logger.warning(
                "LLM returned unparseable rating response; scorecard will use defaults. "
                "Raw response (first 200 chars): %.200s", raw
            )
            scorecard = {}

        # Fill defaults for any missing keys
        defaults = {
            "overall_score": 60.0,
            "overall_label": "medium",
            "structure_score": 60.0,
            "clarity_score": 60.0,
            "consistency_score": 60.0,
            "policy_alignment_score": 60.0,
            "language_quality_score": 60.0,
            "dimension_notes": {},
            "weaknesses_cited": [],
        }
        missing_keys = [k for k in defaults if k not in scorecard]
        if missing_keys:
            logger.warning(
                "Rating scorecard missing keys %s; substituting defaults.", missing_keys
            )
        for k, v in defaults.items():
            scorecard.setdefault(k, v)

        # Recompute overall_score and label using the rubric for consistency
        computed_score = RatingRubric.compute_overall_score(
            structure=float(scorecard["structure_score"]),
            clarity=float(scorecard["clarity_score"]),
            consistency=float(scorecard["consistency_score"]),
            policy_alignment=float(scorecard["policy_alignment_score"]),
            language_quality=float(scorecard["language_quality_score"]),
        )
        scorecard["overall_score"] = round(computed_score, 2)
        scorecard["overall_label"] = RatingRubric.score_to_label(computed_score)

        return scorecard

    def _review(self, policy_content: str) -> tuple[list[dict], str]:
        """Stage 7: Grammar and language copyedit."""
        system = (
            "You are a professional editor. Review the document for grammar, "
            "spelling, punctuation, awkward wording, style inconsistencies, and "
            "terminology standardisation.\n\n"
            "Return JSON with:\n"
            '  "fixes": [{"issue_type": "grammar|spelling|punctuation|wording|style|terminology", '
            '"original_text": str, "corrected_text": str, "explanation": str, '
            '"section_reference": str|null}],\n'
            '  "copyedited_document": "full corrected document in Markdown"'
        )
        human = (
            "## Document to Review\n"
            "<document_content>\n"
            f"{policy_content}\n"
            "</document_content>\n\n"
            "Return JSON:"
        )

        raw = self._invoke(system, human)
        review = _parse_json_response(raw, None)
        if review is None:
            logger.warning(
                "LLM returned unparseable grammar-review response; "
                "copyedited document will fall back to improved draft. "
                "Raw response (first 200 chars): %.200s", raw
            )
            review = {}
        fixes = review.get("fixes", [])
        copyedited = review.get("copyedited_document", policy_content)
        if not review.get("copyedited_document"):
            logger.warning("Grammar review produced no copyedited_document; using input draft.")
        return fixes, copyedited

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        policy_content: str,
        policy_filename: str,
        policy_metadata: dict | None = None,
        historical_policy_filenames: list[str] | None = None,
        top_k: int = settings.retrieval_top_k,
    ) -> FinalPolicyPackage:
        """
        Execute the full linear policy review pipeline.

        Args:
            policy_content: Markdown text of the current policy.
            policy_filename: Original file name (for reporting).
            policy_metadata: Optional metadata dict (must contain document_id
                             if you want to exclude the current doc from retrieval).
            historical_policy_filenames: File names of indexed historical docs.
            top_k: Number of chunks to retrieve.

        Returns:
            FinalPolicyPackage with all analysis and generated content.
        """
        current_doc_id: str | None = (policy_metadata or {}).get("document_id")

        # Stage 1 – Retrieve
        retrieved = self._retrieve(policy_content, current_doc_id, top_k)

        # Stage 2 – Compare
        findings, similarity_report = self._compare(policy_content, retrieved)

        # Stage 3 – Position
        positioning_raw = self._position(policy_content, findings)

        # Stage 4 – Identify issues
        issues_raw = self._identify_issues(policy_content, findings, retrieved)

        # Stage 5 – Rewrite
        improved_draft = self._rewrite(
            policy_content, positioning_raw, issues_raw, retrieved
        )

        # Stage 6 – Rate
        scorecard_raw = self._rate(improved_draft)

        # Stage 7 – Grammar review
        grammar_fixes_raw, copyedited_draft = self._review(improved_draft)

        # ------------------------------------------------------------------
        # Build typed output objects
        # ------------------------------------------------------------------

        # Positioning
        try:
            positioning = PositioningRecommendation(**positioning_raw)
        except Exception as exc:
            logger.warning("Failed to build PositioningRecommendation from LLM output: %s", exc)
            positioning = PositioningRecommendation(
                summary=str(positioning_raw.get("summary", "N/A")),
                recommended_positioning="modernized",
                rationale=positioning_raw.get("rationale", []),
                key_differences=positioning_raw.get("key_differences", []),
                retained_strengths=positioning_raw.get("retained_strengths", []),
                top_risks=positioning_raw.get("top_risks", []),
            )

        # Build a lookup of evidence from retrieved results (up to 5, matching _identify_issues)
        evidence_items = retrieved[:5]
        evidence_ref_map: list[EvidenceRef] = [
            EvidenceRef(
                document_id=r.document_id,
                section_id=r.metadata.get("section_id", ""),
                excerpt=r.content[:300],
                relevance_score=r.rrf_score,
            )
            for r in evidence_items
        ]

        # Issues — resolve evidence_indices returned by the LLM into EvidenceRef objects
        issues: list[PolicyIssue] = []
        for raw_issue in (issues_raw if isinstance(issues_raw, list) else []):
            try:
                evidence_indices: list[int] = raw_issue.get("evidence_indices", []) or []
                issue_evidence = [
                    evidence_ref_map[i]
                    for i in evidence_indices
                    if isinstance(i, int) and 0 <= i < len(evidence_ref_map)
                ]
                issue_data = {k: v for k, v in raw_issue.items() if k != "evidence_indices"}
                issues.append(PolicyIssue(**issue_data, evidence=issue_evidence))
            except Exception as exc:
                logger.warning("Skipping malformed issue from LLM output: %s | issue: %s", exc, raw_issue)

        # Scorecard
        try:
            scorecard = RatingScorecard(**scorecard_raw)
        except Exception as exc:
            logger.warning("Failed to build RatingScorecard from LLM output: %s", exc)
            scorecard = RatingScorecard(
                overall_label="medium",
                overall_score=60.0,
                structure_score=60.0,
                clarity_score=60.0,
                consistency_score=60.0,
                policy_alignment_score=60.0,
                language_quality_score=60.0,
            )

        # Grammar fixes
        grammar_fixes: list[GrammarFix] = []
        for raw_fix in (grammar_fixes_raw if isinstance(grammar_fixes_raw, list) else []):
            try:
                grammar_fixes.append(GrammarFix(**raw_fix))
            except Exception as exc:
                logger.warning("Skipping malformed grammar fix from LLM output: %s | fix: %s", exc, raw_fix)

        # Retrieval evidence — cover all retrieved chunks, not just the first 5
        retrieval_evidence: list[EvidenceRef] = evidence_ref_map + [
            EvidenceRef(
                document_id=r.document_id,
                section_id=r.metadata.get("section_id", ""),
                excerpt=r.content[:300],
                relevance_score=r.rrf_score,
            )
            for r in retrieved[5:]
        ]

        return FinalPolicyPackage(
            workflow_id=str(uuid.uuid4()),
            processed_at=datetime.now(timezone.utc),
            current_policy_filename=policy_filename,
            historical_policies_used=historical_policy_filenames or [],
            similarity_report=similarity_report,
            positioning=positioning,
            issues=issues,
            improved_draft_markdown=improved_draft,
            copyedited_draft_markdown=copyedited_draft,
            scorecard=scorecard,
            grammar_fixes=grammar_fixes,
            retrieval_evidence=retrieval_evidence,
        )
