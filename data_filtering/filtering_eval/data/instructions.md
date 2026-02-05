# Annotation Instructions for gold_dataset_test.jsonl

## Purpose
Each JSONL record represents a prompt/response pair that should be annotated with the correct time-based metadata in the `entities` map and a single canonical year in `gold_year`. Your job is to replace those placeholders with the best-supported values or confirm that no time-based annotation is warranted. 

You should use the earliest year (gold_year) the prompt/response pair could likely have been written in.

## Fields to annotate
- `entities`: A JSON object that maps each time-relevant entity or event to a dictionary with:
  - `year`: A 4-digit year as a string ("YYYY").
  - `source`: A URL to a reputable source that supports the year.
  - `search_query`: The query string you used to find the source.
- `gold_year`: A single integer year for the most salient time reference in the answer. Use `2001` if no time-based entity/event is present or the answer is explicitly timeless.
- `model`: Put an identifier such as your name. 
- The goal is the have a gold dataset and compare it with LLM outputs, hence DONT use LLMs to fill in the fields.

## How to fill missing values
1. Read the `question` and `answer` and identify any real-world entities, events, products, releases, or historical facts that imply a year.
2. For each identified entity/event, add a key to `entities` using a concise name.
3. Find a reliable source for the year and record it in `source`.
4. Record the exact search query used in `search_query` (should be in English).
5. Set `gold_year` to the single most central year for the answer (typically the main event year or release year).
6. If **no** time-based entity/event is present, set `entities` to `2001`, and set `gold_year` to `2001`.


## Examples from this dataset

The first examples in gold_dataset_test.jsonl are already filled in, use them as examples.

## Deliverable

It suffices to fill out the id's up until 105 (i.e. the ids 106-141 can be skipped), and you dont need to adjust nr 1-5, as these are just examples for how to fill out the fields.

Send back the annotated dataset with the name gold_dataset_test_{your_name}.jsonl. Your name should be the same as the one you used in the `model` field.

Remember to change 'model' from 'human' to your name for all the examples (except the first 5) and remember to not use LLMs to fill in the fields.

If there are any questions, feel free to reach out to epsteine@stanford.edu for clarifications.



