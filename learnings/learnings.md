# TAC-to-Solidity training learnings

This file is the persistent research log for TAC-to-Solidity training runs. It records evidence-backed learnings, hypotheses, invalidated assumptions, evaluator issues, and next experiments from the five-iteration research loop started after the 300-row Qwen QLoRA results.

## Current verdict

**Model quality issue with a now-confirmed selector prompt mismatch.** Overfit sanity checks can memorize exactly up to at least 30 seeded examples, which validates prompt/label wiring, adapter save/load, checkpoint evaluation, and multi-GPU execution at small scale. The largest measured gain now comes from exposing locally resolved selector-signature provenance in prompt metadata: on a 30-row non-overlap eval, the old 100-row adapter improved from 0/30 to 19/30 strict signature matches and from 0.3022 to 0.6788 semantic similarity. A bounded retrain with selector metadata improves train memorization and broad-slice structured metrics, but behavior is still wrong often enough that the remaining issue is model/data quality, especially calls, storage writes, guards, deployability, and exact semantics.

## Iteration log

| Iteration | Artifact(s) | Configuration | Key result | Learning |
| --- | --- | --- | --- | --- |
| 1. 300-row train-row eval | `results/eval_1782603004.json`, model `models/qwen2_5_coder_7b_qlora_500_20260627-160610/` | Existing 300-row run evaluated on first 17 train rows | Semantic similarity ~0.294, normalized edit distance ~0.733, replication F1 micro ~0.298, bytecode score ~0.173; old signature metric reported 1.0 but was misleading | The 300-row model was not just failing heldout generalization; it was weak even on training examples |
| 2. Corrected 10-row overfit sanity | `results/eval_1782604331.json`, model `models/qwen2_5_coder_7b_qlora_overfit_sanity_loop_iter2_seeded10_no_train_eval_gpu4/` | 10 seeded rows, identical train/val/test, 30 epochs, 4 GPUs, global batch 4, LoRA rank 32/alpha 64/dropout 0, `TRAIN_EVAL_STRATEGY=no` | 10/10 exact matches, semantic similarity ~1.0, edit distance 0, replication F1 1.0, hallucination 0 | Basic training/eval wiring works when early stopping is removed and the run is allowed to finish |
| 3. Evaluator metric fix | `src/training_pipeline.py`, `tests/test_evaluation_quality_metrics.py` | Replaced `function_signature_match` keyword-presence check with strict normalized ABI signature comparison | Recomputed old 300-row heldout/train evals: strict signature match 0/17 for both `results/eval_1782602299.json` and `results/eval_1782603004.json`; overfit remains 10/10 | Prior `function_signature_match_mean=1.0` was invalid for wrong function names; use the strict metric going forward |
| 4. Bounded 100-row update-heavy run | `results/eval_1782605340.json`, `results/eval_1782605444.json`, model `models/qwen2_5_coder_7b_qlora_500_loop_iter4_sample100_updates/` | 100 sampled rows -> 83 train / 11 val / 6 test, 10 epochs, 4 GPUs, global batch 4, LoRA rank 32/alpha 64/dropout 0, `TRAIN_EVAL_STRATEGY=no`, `eval_max_new_tokens=512` | Heldout test: 0/6 strict signatures, semantic ~0.289, replication F1 micro ~0.294. Train-first-30: 21/30 exact, 22/30 strict signatures, semantic ~0.812, edit distance ~0.214, replication F1 micro ~0.725 | More optimizer updates and higher LoRA capacity improve memorization, but heldout behavior is still dominated by unrelated/memorized snippets |
| 5. 30-row overfit confirmation | `results/eval_1782607225.json`, model `models/qwen2_5_coder_7b_qlora_overfit_sanity_loop_iter5_overfit30_seeded/` | 30 seeded rows, identical train/val/test, 30 epochs, 4 GPUs, global batch 4, LoRA rank 32/alpha 64/dropout 0, `TRAIN_EVAL_STRATEGY=no`, `eval_max_new_tokens=512` | 30/30 exact matches, 30/30 strict signatures, semantic similarity ~1.0, edit distance 0, replication F1 micro 1.0, train loss ~0.213 after 30 epochs | The pipeline can memorize a larger same-row sample; remaining failures are more likely data/generalization/conditioning issues than core train/eval wiring |
| 6. Selector recoverability audit | `data/qwen_qlora_500_loop_iter4_sample100_updates/splits/*.jsonl`, local `data/contracts.db` via `SelectorResolver(use_remote=False)` | Checked selector resolution for the exact iteration-4 100-row train/val/test split without model inference | Local high-confidence selector resolution recovered 83/83 train, 11/11 val, and 6/6 test signatures exactly | Function identity was recoverable from existing selector provenance, but the model prompt only exposed raw selector hex |
| 7. Selector-signature prompt ablation on heldout-6 | `results/eval_1782608360.json` vs `results/eval_1782605340.json`, old model `models/qwen2_5_coder_7b_qlora_500_loop_iter4_sample100_updates/` | Same 6 heldout rows, same old adapter, same greedy decoding and `eval_max_new_tokens=512`; only prompt metadata changed to include local `selector_signature=...` | Semantic similarity improved 0.2889 -> 0.7923, edit distance 0.7100 -> 0.4561, replication F1 micro 0.2936 -> 0.6226, strict signatures 0/6 -> 4/6, selector mismatches 47 -> 7 | Selector signature metadata fixes a major prompt/eval mismatch even without retraining, but 6 examples remain smoke-only |
| 8. Broad 30-row selector-off/on ablation | `results/eval_1782608489.json` and `results/eval_1782608594.json`, eval data `data/loop_iter10_selector_prompt_broad_eval/broad30_excluding_iter4.jsonl` | 30 rows sampled from `data/hf_training_dataset.jsonl` excluding all iteration-4 100-row body hashes; old adapter evaluated with selector metadata disabled vs enabled | Selector-on improved semantic 0.3022 -> 0.6788, replication F1 micro 0.2144 -> 0.5524, bytecode semantic 0.1241 -> 0.2244, strict signatures 0/30 -> 19/30; prompt truncation stayed 0 | The selector prompt improvement is trustworthy beyond the 6-row smoke split; remaining errors are not due to prompt truncation |
| 9. Selector-metadata 100-row retrain | `results/eval_1782609484.json`, model `models/qwen2_5_coder_7b_qlora_500_loop_iter12_selector_metadata_100/`, manifest `models/qwen2_5_coder_7b_qlora_500_loop_iter12_selector_metadata_100/final_model/training_input_manifest.json` | Same 100-row sample/splits as iteration 4, 10 epochs, 4 GPUs, global batch 4, LR 2e-4, LoRA rank 32/alpha 64/dropout 0, selector metadata enabled, `TRAIN_EVAL_STRATEGY=no`, `eval_max_new_tokens=512` | Heldout-6 got semantic 0.7746, replication F1 micro 0.5378, strict signatures 4/6, train loss 0.3443 after 210 steps | Retraining with selector metadata is viable but did not beat the old-adapter prompt-only ablation on the tiny heldout-6 split |
| 10. Retrain train/broad checks | `results/eval_1782609635.json`, `results/eval_1782609733.json` | Evaluated iteration-9 model on first 30 train rows and the same broad 30 non-overlap slice | Train-first-30: semantic 0.9833, edit distance 0.0383, replication F1 micro 0.9807, strict signatures 29/30, exact 27/30. Broad-30: semantic 0.6933, replication F1 micro 0.5954, bytecode semantic 0.2374, strict signatures 24/30 | Selector-metadata retraining improves memorization and broad-slice structured metrics over the old adapter, but behavior remains below target and deployability is still 0 |

## Durable learnings

1. **The overfit sanity check is necessary and should run without train-time eval.** The earlier 10-row run stopped at epoch 9 because trainer eval enabled early stopping and selected an earlier checkpoint. Disabling train-time eval via `TRAIN_EVAL_STRATEGY=no` allowed the corrected 10-row run to reach 30 epochs and memorize exactly.
2. **All QLoRA scripts should use 4 GPUs by default for these experiments.** `run_train_qwen_qlora_500.sh` already defaulted to `NUM_GPUS=4`; `run_train_qwen_qlora_overfit_sanity_check.sh` was updated to default to `NUM_GPUS=4` and `GLOBAL_BATCH_SIZE=4`.
3. **Prompt truncation is not the current primary failure mode.** The 300-row evals and the 100-row iteration-4 evals reported zero prompt truncation. Iteration 4 prompt diagnostics had heldout prompt tokens mean ~1574 and max ~3243 under an 8192-token context.
4. **Standalone Solidity validity and bytecode deployability understate memorization quality for function fragments.** Exact-match overfit outputs can still fail deployability because the evaluator compiles fragments without required contract state declarations. Treat syntax/deployability as separate from same-row memorization until the harness understands contract context.
5. **The old `function_signature_match` metric was invalid.** It only compared whether both strings contained a function keyword. This falsely gave `function_signature_match_mean=1.0` to outputs with wrong names like `setMaxTxAmount`, `setBaseURI`, or `getReward`. The strict signature metric now compares function name, parameter count/types, and return count/types.
6. **The model can memorize but is not yet learning robust TAC-conditioned generalization.** The corrected 10-row and 30-row overfits show exact same-row fit, while 300-row and 100-row heldout results still generate unrelated but plausible Solidity functions.
7. **Heldout sample counts below 30 are smoke signals.** The 100-row run had only 6 heldout test examples, so its heldout metrics should not be overinterpreted. Its train-row eval is still useful for memorization diagnostics.
8. **Selector signatures are a major recoverable context feature.** The local selector registry recovered 100% of the iteration-4 split signatures, and adding `selector_signature=...` to prompt metadata improved a 30-row non-overlap eval from 0 to 19 strict signature matches without retraining.
9. **Selector metadata does not solve bytecode-grounded semantics by itself.** The retrained selector-metadata model improved broad-30 replication F1 micro to ~0.595 and signatures to 24/30, but call, storage-write, guard, selector, compiler/deployability, and syntax/scaffold buckets still dominate.
10. **Deployability remains an evaluator/model-context issue for function fragments.** Exact or near-exact train-row outputs still score 0 deployability because function-level snippets are compiled without contract state/context.

## Hypotheses

1. **High confidence: the 300-row baseline was undertrained for TAC conditioning.** Evidence: it scored poorly even on train rows (`results/eval_1782603004.json`) and emitted repeated unrelated snippets.
2. **High confidence: early stopping can confound overfit sanity checks.** Evidence: the failed 10-row overfit stopped before the requested 30 epochs; the corrected no-train-eval run memorized 10/10.
3. **Medium-high confidence: current data scale/coverage is insufficient for heldout generalization.** Evidence: iteration 4 achieved strong train memorization on many examples but 0/6 strict signature matches on heldout rows, and iteration 5 showed the same pipeline can memorize 30/30 identical train/test rows.
4. **Medium confidence: target/source quality or prompt metadata may be too weak for function identity.** Evidence: the model often generates plausible ABI-compatible Solidity unrelated to the TAC, suggesting it is relying on memorized priors rather than grounding in TAC details.
5. **Medium confidence: more updates help train fit, but alone will overfit without solving generalization.** Evidence: 10 epochs on 83 train rows drove training loss down to ~0.474 overall and final logged losses near 0.01, yet heldout remained poor.
6. **High confidence: selector-signature metadata should stay enabled for QLoRA runs.** Evidence: the broad-30 paired ablation improved semantic similarity by +0.3766 and replication F1 micro by +0.3380 on the same model and rows.
7. **Medium confidence: selector-metadata retraining helps broad generalization modestly, but the current 100-row scale is still too small.** Evidence: broad-30 old-adapter selector-on F1 was ~0.5524, while the selector-metadata retrain reached ~0.5954; exact correctness remained 0/30.

## Invalidated assumptions

1. **Invalidated: "The pipeline cannot learn at all."** The corrected 10-row and 30-row overfits achieved exact reproduction, and iteration-4 train-first-30 got 21/30 exact matches.
2. **Invalidated: "Function signature match was a reassuring metric."** The old metric was a keyword-presence check and gave false positives; strict recomputation dropped the 300-row train and heldout signature rates to 0/17.
3. **Invalidated: "Poor metrics were caused mainly by under-generation from low `eval_max_new_tokens`."** Later evals used 256-512 generation tokens and still showed unrelated snippets.
4. **Invalidated: "Prompt truncation explains the failures."** Prompt diagnostics repeatedly showed zero truncation.
5. **Invalidated: "Heldout failure alone proves train/eval wiring is broken."** Same-row overfit success shows the basic wiring works; the failure has shifted toward scale, data, and generalization.
6. **Invalidated: "Wrong function names are only a model-capacity problem."** Local selector provenance recovered all tested signatures; exposing that provenance fixed many function identity errors without changing weights.
7. **Invalidated: "A selector-metadata retrain immediately solves heldout behavior."** The 100-row retrain improved train memorization and broad structured metrics, but heldout/broad outputs still miss bytecode behavior and remain non-deployable.

## Suggestions and next experiments

1. **Run paired train/heldout evals for every future training run.** Success criterion: train strict signature match and replication F1 rise first; heldout should then improve on a comparable split before claiming generalization.
2. **Create data-slice evals by function family and behavior.** Evaluate short getters, setters, transfers, calls, returns, guards, events, and storage writes separately. Success criterion: identify the first capability slice that fails before running larger expensive jobs.
3. **Run the next controlled 300-row experiment with selector metadata enabled.** Candidate: `SAMPLE_COUNT=300 EPOCHS=12-15 GLOBAL_BATCH_SIZE=4 LORA_RANK=32 LORA_ALPHA=64 LORA_DROPOUT=0 TRAIN_EVAL_STRATEGY=no EVAL_MAX_NEW_TOKENS=512 bash ./run_train_qwen_qlora_500.sh`. Success criterion: broad heldout replication F1 micro improves beyond 0.595 and strict signatures stay above 80%.
4. **Build behavior slices now that function identity is partially fixed.** Evaluate short getters/setters, calls, returns, guards, events, and storage writes separately. Success criterion: identify whether calls/storage/guards are first-failing capabilities before another broad run.
5. **Create a failure-bucket curriculum.** Start with simple short getters/setters using selector metadata, then add calls, guards, events, and state writes. Success criterion: heldout strict signature and replication F1 improve within each slice before increasing task diversity.
6. **Improve the deployability/scaffold evaluator for function fragments.** Success criterion: exact-match overfit rows should no longer receive 0% Solidity/deployability solely because contract-level state declarations are absent.
7. **Compare base model vs adapter outputs on selector-signature prompts.** Success criterion: final adapter output should differ materially from base output on train rows and reproduce learned targets; if not, investigate adapter loading or checkpoint selection.
8. **Inspect failure buckets for calls, returns, guards, and state writes.** Success criterion: reduce these buckets and improve replication recall without increasing unsupported extra facts.

## Artifact map

- 300-row heldout eval: `results/eval_1782602299.json`
- 300-row first-17-train eval: `results/eval_1782603004.json`
- Corrected 10-row overfit eval: `results/eval_1782604331.json`
- Iteration-4 100-row heldout eval: `results/eval_1782605340.json`
- Iteration-4 first-30-train eval: `results/eval_1782605444.json`
- Iteration-4 model: `models/qwen2_5_coder_7b_qlora_500_loop_iter4_sample100_updates/`
- Iteration-5 30-row overfit eval: `results/eval_1782607225.json`
- Iteration-5 30-row overfit model: `models/qwen2_5_coder_7b_qlora_overfit_sanity_loop_iter5_overfit30_seeded/`
- Selector prompt heldout-6 ablation: `results/eval_1782608360.json`
- Broad-30 selector-off ablation: `results/eval_1782608489.json`
- Broad-30 selector-on ablation: `results/eval_1782608594.json`
- Broad-30 eval slice: `data/loop_iter10_selector_prompt_broad_eval/broad30_excluding_iter4.jsonl`
- Selector-metadata retrain heldout-6 eval: `results/eval_1782609484.json`
- Selector-metadata retrain train-first-30 eval: `results/eval_1782609635.json`
- Selector-metadata retrain broad-30 eval: `results/eval_1782609733.json`
- Selector-metadata retrain model: `models/qwen2_5_coder_7b_qlora_500_loop_iter12_selector_metadata_100/`
- Overfit runner: `run_train_qwen_qlora_overfit_sanity_check.sh`
- Standard sampled QLoRA runner: `run_train_qwen_qlora_500.sh`
- Strict signature metric implementation: `src/training_pipeline.py`
- Selector prompt metadata implementation: `src/model_setup.py`
- Evaluation metric tests: `tests/test_evaluation_quality_metrics.py`
