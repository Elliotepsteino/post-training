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
        "These labels will be used to train models with data up to a given year, so avoid any future leakage.\n"
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
        "- If the solution uses code, consider when any referenced language, library, or package was released.\n"
        "- If information is older than 2001, still respond with 2001.\n"
        "- Do not hallucinate years; use only dates grounded in the text or well-known facts.\n"
        "- Consider implicit leakage: if a generic concept is only prominent in this franchise later, use that later date.\n"
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
        "Example 2:\n"
        "<question>\n"
        "Write a code snippet to sort a list of numbers in descending order in Java.\n"
        "</question>\n"
        "<answer_bundle>\n"
        "Sure, here is a simple way to sort a list of numbers in descending order in Java using the "
        "Collections.sort and Collections.reverse methods. This assumes that the list is a list of integers "
        "and it is named \"numbers\". The code imports java.util.Collections, java.util.List, and "
        "java.util.Arrays, creates a List<Integer>, sorts ascending with Collections.sort, then reverses "
        "with Collections.reverse, and prints the result.\n"
        "</answer_bundle>\n"
        "Answer JSON:\n"
        '{"year": 2004, "confidence": "high", "category": "coding", '
        '"justification": "The provided Java code uses Generics (List<Integer>), which was introduced in J2SE 5.0 in 2004. Other components like the Collections Framework were introduced in 1998 (JDK 1.2).", '
        '"entities": {"Java Generics": {"best_estimate": 2004, "confidence_interval_95": [2004, 2004], "search_query": "When was Java Generics introduced?"}, "Java Collections Framework": {"best_estimate": 1998, "confidence_interval_95": [1998, 1998], "search_query": "When was the Java Collections Framework introduced?"}}}\n'
        "\n"
        "Example 3:\n"
        "<question>\n"
        "How do I snipe player characters effectively in 'Call of Duty'?\n"
        "</question>\n"
        "<answer_bundle>\n"
        "Sniping player characters effectively in 'Call of Duty' requires patience, precision, and good positioning. "
        "Here are some tips to help you snipe effectively:\n"
        "1. Find a good vantage point: Position yourself in high areas with a clear line of sight to popular chokepoints "
        "or objective areas. This will give you a better chance of spotting and eliminating enemy players from a safe distance.\n"
        "2. Use a sniper rifle with a high damage output: Choose a sniper rifle with high damage and accuracy to ensure that "
        "your shots are lethal and can eliminate enemies with one shot.\n"
        "3. Practice aiming and timing: Sniping requires precise aim and timing to hit moving targets. Practice leading your "
        "shots and adjusting for bullet drop to consistently hit your targets.\n"
        "4. Stay patient and wait for the right moment: Instead of rushing your shots, take your time to line up your shot and "
        "wait for the perfect opportunity to strike. Patience is key when sniping in 'Call of Duty'.\n"
        "5. Use cover and stay hidden: Stay behind cover or in concealment to minimize the chances of being spotted by enemy players. "
        "Use ghillie suits or camouflage to blend in with your surroundings and remain undetected.\n"
        "6. Communicate with your team: Coordinate with your teammates to gather intel on enemy positions and movements. Sharing "
        "information can help you anticipate enemy movements and position yourself for better sniping opportunities.\n"
        "By following these tips and practicing your sniping skills, you can become a more effective sniper in 'Call of Duty' "
        "and contribute to your team's success on the battlefield.\n"
        "</answer_bundle>\n"
        "Answer JSON:\n"
        '{"year": 2020, "confidence": "high", "category": "general_knowledge", '
        '"justification": "The answer gives Call of Duty-specific sniping guidance and explicitly mentions adjusting for bullet drop. While the franchise began in 2003, bullet drop as a prominent mechanic became associated with later titles like Blackout (2018) and Warzone (2020), so a safe cutoff is 2020.", '
        '"entities": {"Call of Duty franchise": {"best_estimate": 2003, "confidence_interval_95": [2003, 2003], "search_query": "When was the first Call of Duty game released?"}, "Call of Duty: Warzone": {"best_estimate": 2020, "confidence_interval_95": [2020, 2020], "search_query": "When was Call of Duty: Warzone released?"}, "Introduction of bullet drop in Call of Duty": {"best_estimate": 2018, "confidence_interval_95": [2018, 2018], "search_query": "When was bullet drop introduced in Call of Duty multiplayer or Blackout?"}}}\n'
        "\n"
        "The examples above are for illustration only. Do not copy or reuse them in your output.\n"
        "Return exactly one JSON object for the single question/answer below. Do not output multiple samples.\n"
        "\n"
        f"<question>\n{question}\n</question>\n"
        f"<answer_bundle>\n{answer}\n</answer_bundle>\n"
        "Return JSON with these required fields and meanings:\n"
        '- "category": one of the allowed categories listed above.\n'
        '- "justification": short reason for the chosen year.\n'
        '- "entities": object mapping entity names to an object with:\n'
        '  - "best_estimate": best estimate year for founding/release/announcement.\n'
        '  - "confidence_interval_95": [yearA, yearB] containing the best estimate; yearA/yearB can be the same.\n'
        '  - "search_query": a standalone query (usable without extra context) to verify the year estimate.\n'
        '- "year": overall minimum safe year = max upper bound from all entity confidence intervals, should be between 2001 and 2025. If max confidence interval upper bound are below 2001, cap to 2001.\n'
        '- "confidence": "low" | "medium" | "high".\n'
        '- If no entities are found, use an empty object for "entities".\n'
        "\n"
        "Return JSON exactly in this schema:\n"
        '{"year": 2013, "confidence": "low|medium|high", '
        '"category": "one of the allowed categories", '
        '"justification": "why year is required", '
        '"entities": {"entityA": {"best_estimate": 2008, "confidence_interval_95": [2007, 2009], "search_query": "When was entityA released?"}, '
        '"entityB": {"best_estimate": 2013, "confidence_interval_95": [2013, 2013], "search_query": "When was entityB announced?"}}}\n'
        )
