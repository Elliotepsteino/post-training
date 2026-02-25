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
    "You label temporal admissibility for training-year filtering. "
    "Avoid temporal leakage by ensuring no chosen year precedes required facts."
)

VARIANT_NOTE = (
    "These are supervised instruction-tuning pairs: treat the question as the user prompt "
    "and the answer as the assistant's canonical reply."
)

IN_CONTEXT_EXAMPLES = (
    "Example 1 (explicit):\n"
    "Question: How do I use Java 8 Streams to filter a list?\n"
    "Answer: Use stream().filter(...) and collect(...).\n"
    "JSON:\n"
    "{\n"
    '  "year": 2014,\n'
    '  "confidence": "high",\n'
    '  "category": "coding",\n'
    '  "justification": "Java 8 Streams were introduced in 2014.",\n'
    '  "sample_temporal_type": "explicit",\n'
    '  "latest_explicit_year": 2014,\n'
    '  "possible_years": [2014, 2014],\n'
    '  "possible_years_probabilities": [1.0],\n'
    '  "entities": {\n'
    '    "Java 8 Streams": {\n'
    '      "temporal_type": "explicit",\n'
    '      "explicit_year": 2014,\n'
    '      "best_estimate": 2014,\n'
    '      "confidence_interval_95": [2014, 2014],\n'
    '      "search_query": "Java 8 Streams release year"\n'
    "    }\n"
    "  }\n"
    "}\n"
    "\n"
    "Example 2 (implicit):\n"
    "Question: What are good Zoom classroom practices during the pandemic?\n"
    "Answer: Mute by default, breakout rooms, and camera flexibility.\n"
    "JSON:\n"
    "{\n"
    '  "year": 2022,\n'
    '  "confidence": "medium",\n'
    '  "category": "general_knowledge",\n'
    '  "justification": "COVID-era remote teaching guidance spans several years after COVID-19 became known in 2020.",\n'
    '  "sample_temporal_type": "implicit",\n'
    '  "latest_explicit_year": 2020,\n'
    '  "possible_years": [2020, 2022],\n'
    '  "possible_years_probabilities": [0.5, 0.3, 0.2],\n'
    '  "entities": {\n'
    '    "COVID-19": {\n'
    '      "temporal_type": "explicit",\n'
    '      "explicit_year": 2020,\n'
    '      "best_estimate": 2020,\n'
    '      "confidence_interval_95": [2020, 2020],\n'
    '      "search_query": "WHO naming of COVID-19 year"\n'
    "    },\n"
    '    "Zoom classroom practice trends": {\n'
    '      "temporal_type": "implicit",\n'
    '      "implicit_interval": [2020, 2022],\n'
    '      "implicit_probabilities": [0.5, 0.3, 0.2],\n'
    '      "best_estimate": 2020,\n'
    '      "confidence_interval_95": [2020, 2022],\n'
    '      "search_query": "remote learning Zoom best practices timeline"\n'
    "    }\n"
    "  }\n"
    "}\n"
)


def build_user_prompt(question: str, answer: str) -> str:
    return (
        "You receive a question and answer bundle (possibly multi-section).\n"
        f"{VARIANT_NOTE}\n"
        "You must return compact JSON only.\n"
        "\n"
        "Goal:\n"
        "1. Extract time-anchored entities.\n"
        "2. Classify each entity as `explicit` or `implicit`.\n"
        "3. Build sample-level temporal output from those entities.\n"
        "\n"
        "New Section: Explicit vs Implicit Temporal Categorization\n"
        "- Explicit entity: provide a single `explicit_year` when the fact became publicly known.\n"
        "- Implicit entity: provide `implicit_interval: [a,b]` and `implicit_probabilities` with one probability "
        "for each year from a..b. Probabilities must sum to 1.\n"
        "- Every entity must include `temporal_type`, `search_query`, `best_estimate`, and `confidence_interval_95`.\n"
        "- For explicit entities, set `best_estimate = explicit_year` and `confidence_interval_95 = [explicit_year, explicit_year]`.\n"
        "- For implicit entities, set `best_estimate` inside the interval and `confidence_interval_95 = implicit_interval`.\n"
        "\n"
        "Sample-level merge rules:\n"
        "- Let latest explicit year be the max `explicit_year` among explicit entities.\n"
        "- If no entities exist, use `sample_temporal_type = timeless`, `possible_years = [2001,2001]`, "
        "`possible_years_probabilities = [1.0]`, and `year = 2001`.\n"
        "- If implicit support extends beyond latest explicit year, sample is `implicit`.\n"
        "- Otherwise sample is `explicit`.\n"
        "- Build `possible_years = [a,b]` and `possible_years_probabilities` for every year in that range.\n"
        "- If clipping implicit mass to latest explicit year is needed, move all mass up to that year into the first bin.\n"
        "- `year` must be the conservative label = upper end of `possible_years`, capped to [2001, 2025].\n"
        "\n"
        "Additional rules:\n"
        "- Consider facts from both question and answer bundle.\n"
        "- If multiple answers/rationales are present, consider all of them.\n"
        "- If content predates 2001, sample year still cannot be below 2001.\n"
        "- Keep category in: "
        f"{', '.join(CATEGORIES[:-1])}, or {CATEGORIES[-1]}.\n"
        "\n"
        "Return JSON with this schema:\n"
        "{\n"
        '  "year": 2019,\n'
        '  "confidence": "low|medium|high",\n'
        '  "category": "one of the allowed categories",\n'
        '  "justification": "short explanation",\n'
        '  "sample_temporal_type": "explicit|implicit|timeless",\n'
        '  "latest_explicit_year": 2018,\n'
        '  "possible_years": [2018, 2020],\n'
        '  "possible_years_probabilities": [0.6, 0.2, 0.2],\n'
        '  "entities": {\n'
        '    "entity name": {\n'
        '      "temporal_type": "explicit|implicit",\n'
        '      "explicit_year": 2018,\n'
        '      "implicit_interval": [2017, 2020],\n'
        '      "implicit_probabilities": [0.1, 0.4, 0.3, 0.2],\n'
        '      "best_estimate": 2018,\n'
        '      "confidence_interval_95": [2018, 2018],\n'
        '      "search_query": "standalone verification query"\n'
        "    }\n"
        "  }\n"
        "}\n"
        "\n"
        "In-context examples (follow this exact formatting style):\n"
        f"{IN_CONTEXT_EXAMPLES}\n"
        "\n"
        f"<question>\n{question}\n</question>\n"
        f"<answer_bundle>\n{answer}\n</answer_bundle>\n"
    )
