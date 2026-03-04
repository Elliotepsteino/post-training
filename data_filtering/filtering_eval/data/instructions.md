# Annotation Instructions for gold_dataset_test.jsonl

## Purpose
Each JSONL record represents a prompt/response pair that should be annotated with the correct time-based metadata in the `entities` map and a single canonical year in `gold_year`. Your job is to replace those placeholders with the best-supported values or confirm that no time-based annotation is warranted. 

You should use the earliest year (gold_year) the prompt/response pair could likely have been written in.

## Fields to annotate
- `entities`: A JSON object that maps each time-relevant entity or event to a dictionary with:
  - `year`: A 4-digit year as a string ("YYYY").
  - `source`: A URL to a reputable source that supports the year.
  - `search_query`: The query string you used to find the source. (should be in English)
  - `entity_type`: Either `explicit` or `implicit`. An explicit entity is one where the knowledge related to it was released in a specific year. This could be for instance a product launch. An implicit entity is one where the knowledge related to it was not released in a specific year. This could be for instance a general statement such as "Global warming is a real problem", where the year this could have been said is multiple years.
  - `plausible_years`: A list of years that are plausible for the entity.
  - `plausible_years_probs`: A list of probabilities for the years in `plausible_years` (sum to 1).
- `gold_year`: A single integer year that is earliest year that the text could have been written. To determine this, take the max over the explicit years and if an event is implicit, use the min of the plausible years for that entity. Use `timeless` if the answer is explicitly timeless or if the answer is based on very old facts such as basic arithmetic.
- `model`: Put an identifier such as your name. 

## Additional details
1. Read the `question` and `answer` and identify any real-world entities, events, products, releases, or historical facts that imply a year.
2. For each identified entity/event, add a key to `entities` using a concise name.
3. For each entity, have a field `entity_type` that is either `explicit` or `implicit`. 
4. If `entity_type` is `explicit`, plausible_years should be a list with one single year equal to `year` that is the earliest year that the entity could have been created/released/discovered/etc, and plausible_years_probs should be a list with one single probability equal to 1.
5. If `entity_type` is `implicit`, plausible_years should be a list of years that are plausible for the entity, and plausible_years_probs should be a list of probabilities for the years in `plausible_years` (sum to 1).


## Examples from this dataset

The first examples in gold_dataset_test.jsonl are already filled in, use them as examples.

## Deliverable

It suffices to fill out the id's up until 105 (i.e. the ids 106-141 can be skipped), and you dont need to adjust nr 1-5, as these are just examples for how to fill out the fields.

Send back the annotated dataset with the name gold_dataset_test_{your_name}.jsonl. Your name should be the same as the one you used in the `model` field.

Remember to change 'model' from 'human' to your name for all the examples (except the first 5) and remember to not use LLMs to fill in the fields.

If there are any questions, feel free to reach out to epsteine@stanford.edu for clarifications.



