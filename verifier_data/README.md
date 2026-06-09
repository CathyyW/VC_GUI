# Critic-Augmented P3 Data

This folder builds pairwise verifier prompts that include the previous step's
critic feedback.

The final JSONL files contain prompt strings ready for V-Droid-style pairwise
preference tuning:

```json
{"chosen": "...prompt with helpful action...", "rejected": "...prompt with unhelpful action..."}
```

Debug JSONL files keep the structured fields and construction metadata for
manual inspection.

## Anti-Back Validation

Use `make_antiback_eval.py` to extract a focused diagnostic set for checking
whether the V-Droid verifier over-prefers `navigate_back`.

It writes two sets:

- `anti_back_reject`: `rejected.action_type == navigate_back`, while the previous
  critic recovery is not `navigate_back`/`close_dialog`.
- `valid_back_chosen`: `chosen.action_type == navigate_back`, while the previous
  critic recovery explicitly asks for `navigate_back`.

Run `evaluate_vdroid_pairs.py` on AutoDL inside the V-Droid environment to score
these pairs with the official verifier checkpoint.

## Experiment Report Materials

After running the full-test and navigate-back diagnostic evaluations, use
`summarize_v_experiment.py` to generate report- and PPT-ready materials:

```bash
python verifier_data/summarize_v_experiment.py \
  --results-root v_eval_results \
  --data-stats data/verifier_p3/androidworld_critic_augmented_p3_v2_20260607/stats.json \
  --model-dir ../V-Droid/saved/Llama-31-8B-vc_critic_aug_p3_v2_e1_20260607 \
  --output-dir v_eval_results/final_report
```

The output directory contains:

- `experiment_summary.md`: readable experiment summary and report-ready text.
- `ppt_outline.md`: slide-by-slide copy-ready content.
- `metrics_table.csv`: complete evaluation metrics.
- `chart_data.csv`: compact data for charts.
- `key_metrics.json`: structured headline results.
- `run_manifest.json`: source paths and training metadata.
