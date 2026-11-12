# Final submission completion guide

This file separates work that can be completed in the repository from actions that require a real team member or independent rater.

## Current verified repository state

- The final report, evaluation protocol, results, and presentation/demo plan are present on `main`.
- The blind scoring sheet is `results/metrics/rubric_sheet.csv`.
- The scoring rubric is `report/instruction_rubric.md`.
- The scorer must not open `results/metrics/rubric_key.json` until every row has been scored.
- The scoring CLI is implemented in `src/evaluate.py` as the `rubric-score` subcommand.
- Do not replace blank scores with zero. Blank means not yet rated.

## A. Independent blind scoring — required external action

Assign one independent rater who did not generate the model outputs and who will not inspect the key before scoring.

1. Give the rater only:
   - `results/metrics/rubric_sheet.csv`
   - `report/instruction_rubric.md`
2. Do **not** share `results/metrics/rubric_key.json`, model labels, checkpoint names, or any model-to-item mapping.
3. Ask the rater to score every row from the visible response only:
   - `relevance`: 0–2
   - `format`: 0–2
   - `answers`: 0–2
   - `coherence`: 0–2
   - `notes`: optional failure tags from the rubric (comma-separated if multiple)
4. Require all four score cells to contain integers 0, 1, or 2. Keep the `item` and `prompt_id` columns unchanged.
5. The rater returns the completed CSV. Keep the original untouched as a backup.
6. Only after all 120 rows are scored, the project team may run the aggregation command below and open the key through the script.

### Validate and aggregate after the rater returns the sheet

Run from the repository root, after saving the completed sheet as `results/metrics/rubric_sheet_scored.csv`:

```bash
python src/evaluate.py rubric-score --sheet results/metrics/rubric_sheet_scored.csv --key results/metrics/rubric_key.json --out results/metrics/rubric_summary.json
```

Then inspect `results/metrics/rubric_summary.json` and update:
- `report/results.md`: per-model rubric total and four criterion means, plus failure-tag counts and examples.
- `report/final_report.md`: replace pending rubric cells and update the E2/E4 interpretation, without overstating a small one-rater sample.
- `README.md`: remove the blind-scoring-pending status only after the report and summary are committed.

Do not infer model identity from response style while scoring. The mapping is applied only during aggregation.

## B. Team review and presentation rehearsal — required team action

The team should complete and record the following before submission:

- [ ] All three members review the final report and verify their own implementation/results.
- [ ] Akash checks slides 6–8 and can explain the architecture, causal masks, linear-attention trade-offs, and pre-training results.
- [ ] Haneesh checks slides 4–5 and can explain the data split, tokenizer, SFT template, and response-only loss masking.
- [ ] Samvith checks slides 1–3 and 9–12, including the evaluation protocol, benchmark evidence, failure analysis, and limitations.
- [ ] Every member can explain the complete pipeline and answer questions outside their own slides.
- [ ] Run the live demo commands from `report/presentation_demo_plan.md` on a fresh Colab runtime with the final checkpoints.
- [ ] Verify every slide number against the committed metrics; keep the saved-output backup ready if Colab/Drive fails.
- [ ] Conduct one timed 12-minute presentation plus the planned demo and viva practice.

## C. Local repository sync and final verification

On the Windows machine, from the cloned repository directory:

```bash
git switch main
git pull origin main
python -m pip install -r requirements.txt
python -m pip check
python -m pytest tests -v
git status
```

Expected from the last recorded verification: 67 tests pass and `pip check` reports no broken requirements. Re-run these commands locally after pulling; do not assume an old local checkout includes the latest documentation commits.

## D. Presentation artifact

The slide deck was prepared as a separate PowerPoint artifact for the team. Add it to the repository only if the team wants the binary deck version-controlled; otherwise keep the deck alongside the submission materials. Confirm the final deck opens and that its figures and metrics match the report.

## Completion rule

The project may be marked fully submission-ready only when:
1. the independent rater has completed all 120 rows;
2. the aggregation JSON is generated and the report tables/failure analysis are updated;
3. the team review and timed rehearsal are complete;
4. local tests and dependency checks pass on the final pulled commit.

Until then, describe the implementation and experiments as complete, and the submission as pending the explicitly listed human/team actions.
