from typing import Literal


class RatingRubric:
    """
    Multi-dimensional weighted scoring rubric for policy documents.

    Weights:
      Structure       20%
      Clarity         25%
      Consistency     20%
      Policy Align    25%
      Language        10%

    Labels:
      poor      score < 50
      medium    50 ≤ score < 70
      good      70 ≤ score < 85
      excellent score ≥ 85
    """

    WEIGHTS: dict[str, float] = {
        "structure": 0.20,
        "clarity": 0.25,
        "consistency": 0.20,
        "policy_alignment": 0.25,
        "language_quality": 0.10,
    }

    @classmethod
    def compute_overall_score(
        cls,
        structure: float,
        clarity: float,
        consistency: float,
        policy_alignment: float,
        language_quality: float,
    ) -> float:
        """Return the weighted overall score (0–100)."""
        return (
            cls.WEIGHTS["structure"] * structure
            + cls.WEIGHTS["clarity"] * clarity
            + cls.WEIGHTS["consistency"] * consistency
            + cls.WEIGHTS["policy_alignment"] * policy_alignment
            + cls.WEIGHTS["language_quality"] * language_quality
        )

    @classmethod
    def score_to_label(
        cls, score: float
    ) -> Literal["poor", "medium", "good", "excellent"]:
        """Map a numeric score to a quality label."""
        if score < 50:
            return "poor"
        elif score < 70:
            return "medium"
        elif score < 85:
            return "good"
        else:
            return "excellent"

    @classmethod
    def get_rating_prompt(cls) -> str:
        """Return the rubric description for use in LLM prompts."""
        return """Score each dimension from 0-100:

**Structure (20% weight)**
- Clear section hierarchy and heading levels
- Logical flow between sections
- Consistent section naming

**Clarity (25% weight)**
- Unambiguous, precise language
- Readable sentence structure
- Clear scope definitions

**Consistency (20% weight)**
- Consistent terminology throughout
- No internal contradictions
- Proper cross-references

**Policy Alignment (25% weight)**
- Aligns with industry standards
- Consistent with historical intent
- Appropriate control strength and coverage

**Language Quality (10% weight)**
- Correct grammar and spelling
- Appropriate punctuation
- Professional tone

For any dimension scored below 70, cite specific evidence."""
