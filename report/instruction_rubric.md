# Instruction-following rubric (blind scoring)

Each generated response is scored 0, 1 or 2 on four criteria (total 0–8). The scorer works from
`results/metrics/rubric_sheet.csv`, which shuffles all models' outputs and hides which model wrote which
response. The key (`rubric_key.json`) is opened only after every row is scored.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| **Relevance** — is it about what was asked? | unrelated to the instruction | on topic but drifts or only partly addresses it | clearly addresses the instruction |
| **Format** — does it follow the requested form? | ignores the requested form (asked for a list/one word/summary, got something else) or is not a response at all | partly follows it | follows it (or no specific form was requested and the form is sensible) |
| **Answers, not echoes** | repeats or paraphrases the instruction/context without adding anything, or is empty | adds a little new content but mostly echoes | gives an actual answer / new content |
| **Coherence** | incoherent, broken text or visible loops (same phrase ≥ 3 times) | understandable but with repetition, contradictions or sudden topic jumps | fluent and internally consistent |

Factual correctness is **not** a separate criterion: a 4-layer model trained on children's stories cannot
be expected to know facts, and scoring it would mostly measure luck. Factual errors are noted in the
`notes` column and counted in the failure analysis instead.

## Scoring rules

1. Score one criterion at a time down the sheet; do not change earlier scores after seeing later items.
2. Judge only the visible response, not what the model "probably meant".
3. Empty response: 0 on all four criteria.
4. Use the `notes` column for failure tags: `echo`, `loop`, `off-topic`, `story-mode` (answers with a
   TinyStories-style story), `truncated`, `factual-error`, `format-violation`, `other`.

## Failure taxonomy (E4)

The tags above are counted per model and illustrated with 2–3 real examples each in `report/results.md`,
together with a plausible cause (e.g. `story-mode` → the pre-training corpus contains only stories;
`loop` → greedy decoding of a small model; `truncated` → 128-token limit or context window).
