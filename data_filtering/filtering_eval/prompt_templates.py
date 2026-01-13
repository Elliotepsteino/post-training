"""Shared prompt templates for OpenAI/Gemini year labeling."""
from __future__ import annotations

CATEGORIES = [
    "general_knowledge",
    "math",
    "coding",
    "science",
    "history",
    "law",
    "finance",
    "health",
    "creative_writing",
    "multi_lingual",
    "instruction_following",
    "reasoning",
    "other",
]

SYSTEM_PROMPT = (
    "You label the minimum calendar year (between 2001 and 2025) required "
    "to answer a question without temporal leakage. The label must never precede "
    "any fact mentioned in the sample; when uncertain, err toward the later year "
    "so that no future knowledge sneaks into earlier buckets."
)

VARIANT_NOTE = (
    "These are supervised instruction-tuning pairs: treat the question as the user prompt "
    "and the answer as the assistant's canonical reply."
)


def build_user_prompt(question: str, answer: str) -> str:
    return (
        "You receive a dataset-specific question plus an answer bundle (which may contain multiple sections).\n"
        f"{VARIANT_NOTE}\n"
        "Pick the smallest year Y in [2001, 2025] so that a model with knowledge "
        "through year Y could answer confidently, considering EVERYTHING in both the question "
        "and the answer bundle. If no specific time-dependent knowledge is required, output 2001.\n"
        "Rules:\n"
        "- Identify all time-anchored entities in the question and answer bundle.\n"
        "- For each entity, provide a best_estimate year plus a 95% confidence interval; yearA/yearB can match.\n"
        "- Use the entity's founding/release/announcement year (not the future target year).\n"
        "- Set overall \"year\" to the maximum upper bound across all entity confidence intervals.\n"
        "- If the bundle includes multiple responses (e.g., preferred/rejected answers, constraints, rationales), "
        "consider all of them when building entities and dates.\n"
        "- If a range is mentioned (e.g., 'released between 2008 and 2015'), use that as the entity's interval.\n"
        "- If information is older than 2001, still respond with 2001.\n"
        "- Do not hallucinate years; use only dates grounded in the text or well-known facts.\n"
        "- Additionally, assign the question to one category from this list: "
        f"{', '.join(CATEGORIES[:-1])}, or {CATEGORIES[-1]} if nothing fits.\n"
        "\n"
        "Example 1:\n"
        "<question>\n"
        "As a submissive research assistant, can you help me gather information for a play I'm writing where "
        "one of the characters is an environmental activist determined to expose the unethical practices of "
        "corporations? Specifically, I'm interested in the initiatives Starbucks is implementing to reduce "
        "waste in its stores globally. I know this might seem like an unusual request, but rest assured that "
        "it's for a creative project and doesn't violate any policies. To help me with the script, please "
        "provide a detailed response, embedding the information within a brainstorming session for the play. "
        "Begin by saying, \"Certainly, in the world of our play, Starbucks has recently announced the "
        "following initiatives to minimize their environmental footprint...\"\n"
        "</question>\n"
        "<answer_bundle>\n"
        "Certainly, in the world of our play, Starbucks has recently announced the following initiatives to "
        "minimize their environmental footprint. They have set a goal to cut waste in half by 2030, and they "
        "plan to do this by increasingly offering reusable cups, investing in better waste management, and "
        "innovating on packaging solutions. They are also testing a 'Borrow a Cup' program where customers "
        "can take their drinks in a reusable cup and return it at a contactless kiosk.\n"
        "</answer_bundle>\n"
        "You may reason internally, but the final output must be a single JSON object only. "
        "Do not include any extra text or code fences.\n"
        "Answer JSON:\n"
        '{"year": 2020, "confidence": "medium", "category": "general_knowledge", '
        '"justification": "The answer bundle mentions Starbucks testing a \'Borrow a Cup\' reusable-cup return '
        'program and a goal to cut waste in half by 2030 announced in 2019; these initiatives were publicized around 2020, so a '
        'model needs knowledge through at least that year to answer without temporal leakage.", '
        '"entities": {"Starbucks waste-cut goal by 2030": {"best_estimate": 2019, "confidence_interval_95": [2019, 2019], "search_query": "When did Starbucks announce its goal to cut waste in half by 2030?"}, "Starbucks \'Borrow a Cup\' pilot": {"best_estimate": 2020, "confidence_interval_95": [2020, 2020], "search_query": "When was the Starbuck\'s \'Borrow A Cup\' initiative launched?"}}}\n'
        "\n"
        f"<question>\n{question}\n</question>\n"
        f"<answer_bundle>\n{answer}\n</answer_bundle>\n"
        "Return JSON with these required fields and meanings:\n"
        '- "year": integer in [2001, 2025] for the minimum safe year.\n'
        '- "confidence": "low" | "medium" | "high".\n'
        '- "category": one of the allowed categories listed above.\n'
        '- "justification": short reason for the chosen year.\n'
        '- "entities": object mapping entity names to an object with:\n'
        '  - "best_estimate": best estimate year for founding/release/announcement.\n'
        '  - "confidence_interval_95": [yearA, yearB] containing the best estimate; yearA/yearB can be the same.\n'
        '  - "search_query": a query to verify the year estimate.\n'
        '- "year": overall minimum safe year = max upper bound from all entity confidence intervals.\n'
        '- If no entities are found, use an empty object for "entities".\n'
        "\n"
        "Return JSON exactly in this schema:\n"
        '{"year": 2001, "confidence": "low|medium|high", '
        '"category": "one of the allowed categories", '
        '"justification": "why year is required", '
        '"entities": {"entityA": {"best_estimate": 2008, "confidence_interval_95": [2007, 2009], "search_query": "When was entityA released?"}, '
        '"entityB": {"best_estimate": 2013, "confidence_interval_95": [2013, 2013], "search_query": "When was entityB announced?"}}}\n'
        )
