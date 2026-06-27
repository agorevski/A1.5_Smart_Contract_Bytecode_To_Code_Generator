# Repository Review Findings

Verified on 2026-06-26 from the current checkout. Findings are prioritized by severity and include only actionable issues with supporting citations.

## High severity

### 1. Unauthenticated cross-origin decompile endpoint can trigger heavy inference
- **Severity:** High
- **Affected files/lines:** `web/app.py:102`, `web/app.py:336-365`, `web/app.py:530-684`, `web/app.py:903`
- **Evidence/impact:** API CORS allows any origin, the server binds to `0.0.0.0` by default, and `/api/decompile` can run per-function batched or single-function model inference after only size/shape checks. Any website or network peer that can reach the service can consume GPU/CPU resources.
- **Recommendation:** Restrict CORS origins, require authentication or a local-only default bind address, add rate limits, and queue/cap concurrent inference work.

### 2. Memory monitoring flag crashes training
- **Severity:** High
- **Affected files/lines:** `train.py:546-550`, `src/model_setup.py:245-252`, `src/model_setup.py:618-619`
- **Evidence/impact:** `--enable-memory-monitoring` enables `MemoryLoggingCallback(self.logger)`, but `SmartContractModelTrainer.__init__` initializes config/output/model fields and never defines `self.logger`. The flag therefore raises `AttributeError` before training can proceed.
- **Recommendation:** Initialize `self.logger = logger` in the trainer or pass the module logger directly.

### 3. CI lint job is currently red
- **Severity:** High
- **Affected files/lines:** `.github/workflows/ci.yml:17-24`, `src/model_setup.py:1121`
- **Evidence/impact:** CI runs `black --check`, `flake8`, and `mypy`. A local `flake8` check already reports `src/model_setup.py:1121:19 F821 undefined name 'BytecodeAnalyzer'` along with other violations, so the lint job cannot pass as configured.
- **Recommendation:** Fix the lint/type errors, or intentionally adjust CI rules to match the repository's accepted baseline before relying on CI status.

### 4. CI test dependency install omits a required import
- **Severity:** High
- **Affected files/lines:** `.github/workflows/ci.yml:33-39`, `requirements.txt:43-44`, `download_hf_contracts.py:43-44`, `tests/test_dataset_quality_issues.py:984-986`
- **Evidence/impact:** The CI test job installs a hand-written dependency subset that omits `huggingface_hub`, while tests import `download_hf_contracts.py`, which imports `HfApi` and `hf_hub_download`. Fresh CI test runs can fail before exercising the intended tests.
- **Recommendation:** Install `-r requirements.txt` in CI or maintain a tested CI constraints file that includes all imported test dependencies.

## Medium severity

### 5. Selector resolution leaks lookups to 4byte.directory and can stall requests
- **Severity:** Medium
- **Affected files/lines:** `web/app.py:404-410`, `src/selector_resolver.py:286-294`, `src/selector_resolver.py:469-476`
- **Evidence/impact:** Web decompilation calls `get_resolver(use_remote=True)`. Unknown selectors then trigger synchronous `requests.get()` calls to 4byte.directory with a 3-second timeout per selector, leaking selectors externally and delaying user-visible requests.
- **Recommendation:** Make remote lookup explicit/opt-in, run it in the background, cap remote lookups per request, and prefer local cache/DB results by default.

### 6. Security endpoints do not validate bytecode syntax
- **Severity:** Medium
- **Affected files/lines:** `web/app.py:367-379`, `web/app.py:781-790`, `web/app.py:823-832`, `web/app.py:856-865`
- **Evidence/impact:** `/api/decompile` normalizes and validates even-length hexadecimal bytecode, but vulnerability scan, classification, and audit report endpoints only check presence and maximum length before handing input to analyzers.
- **Recommendation:** Reuse a single bytecode normalization/validation helper for every POST endpoint that accepts bytecode.

### 7. Audit report decompilation calls a non-existent decompiler API
- **Severity:** Medium
- **Affected files/lines:** `web/app.py:870-878`, `src/audit_report.py:171-175`, `src/model_setup.py:1025-1030`
- **Evidence/impact:** `AuditReportGenerator` calls `self.decompiler.decompile(bytecode)`, but `SmartContractDecompiler` exposes `decompile_contract(...)` as its contract-level entry point. Audit reports with `include_decompilation=True` will fail or silently lose decompiled-source coverage depending on exception handling.
- **Recommendation:** Call `decompile_contract(...)` and store the returned Solidity field, or add a small adapter method named `decompile`.

### 8. Frontend permits stale and overlapping decompile requests
- **Severity:** Medium
- **Affected files/lines:** `web/static/app.js:125-135`, `web/static/app.js:806-823`, `web/static/app.js:944-947`, `web/static/app.js:963-990`
- **Evidence/impact:** Starting decompilation disables only the decompile button. Clear/input handlers and Ctrl+Enter remain active, and the fetch has no `AbortController` or request id. Older responses can update results after the input has changed or a newer request has started.
- **Recommendation:** Add an in-flight guard, abort stale fetches, disable or gate conflicting controls, and ignore responses whose request id no longer matches current UI state.

### 9. Gradient accumulation configuration is ignored
- **Severity:** Medium
- **Affected files/lines:** `src/training_pipeline.py:67-72`, `src/training_pipeline.py:667-673`, `src/model_setup.py:391-405`, `src/model_setup.py:601-608`
- **Evidence/impact:** The pipeline config exposes `gradient_accumulation_steps`, and `create_training_arguments` accepts it, but the pipeline train call and trainer call to `create_training_arguments` never pass the configured value. Training silently uses the trainer default of 4.
- **Recommendation:** Thread `gradient_accumulation_steps` through CLI/config, `SmartContractModelTrainer.train`, and `create_training_arguments`.

### 10. Training loss is computed over prompt/input tokens
- **Severity:** Medium
- **Affected files/lines:** `src/model_setup.py:162-169`, `src/model_setup.py:205-206`, `src/model_setup.py:588-591`
- **Evidence/impact:** The dataset formats instruction, TAC input, and response into one prompt, then copies all `input_ids` into `labels`. The collator masks only padding, so long TAC prompts dominate the causal-LM loss instead of training only on Solidity response tokens.
- **Recommendation:** Build labels with `-100` for instruction/input tokens and real token ids only for the response span.

### 11. Dataset split ratios are misleading
- **Severity:** Medium
- **Affected files/lines:** `train.py:179-183`, `train.py:210-223`
- **Evidence/impact:** Defaults appear to be 85% train and 10% validation, but `test_ratio = 1.0 - train_ratio` makes the test set 15%; validation is then 10/95 of the remaining data. The default effective split is about 76% train, 9% validation, 15% test.
- **Recommendation:** Compute `test_ratio = 1 - train_ratio - val_ratio`, or rename/re-document the parameters to describe the two-stage split.

### 12. Model augmentation tests can silently skip broken APIs
- **Severity:** Medium
- **Affected files/lines:** `tests/test_dataset_quality_issues.py:1095-1118`
- **Evidence/impact:** `_model_setup_supports_name_augmentation()` catches all exceptions and returns `False`, which triggers a class-level skip. Import errors or unexpected API breakage are converted into skipped coverage rather than failing tests.
- **Recommendation:** Catch only expected optional-dependency errors, and fail when repository-owned APIs are missing or raise unexpected exceptions.

### 13. Solidity OR pragmas are treated as AND constraints
- **Severity:** Medium
- **Affected files/lines:** `src/local_compiler.py:160-170`
- **Evidence/impact:** `_version_matches_pragma` extracts all version constraints and requires every constraint to match. For a pragma such as `^0.5.0 || ^0.6.0`, neither `0.5.x` nor `0.6.x` can satisfy both sides, so compatible compiler versions are rejected.
- **Recommendation:** Split pragma expressions on `||` and accept versions that satisfy any disjunct while preserving AND behavior within each disjunct.

### 14. Lookup DB reruns permanently skip failed or no-pair compile jobs
- **Severity:** Medium
- **Affected files/lines:** `scripts/build_lookup_db.py:205-219`, `scripts/build_lookup_db.py:456-464`, `scripts/build_lookup_db.py:514-523`, `src/tac_lookup.py:504-510`
- **Evidence/impact:** Worker setup/compile failures return `[]`, which is recorded as `no_pairs`; raised exceptions are recorded as `error`. Later reruns treat every row in `compiled_jobs` as completed regardless of status, so transient failures and empty results are never retried.
- **Recommendation:** Track retryable failures separately, skip only confirmed successful jobs, and add a `--force` or retry path for `error`/`no_pairs` statuses.

### 15. Missing Hugging Face addresses can collide and drop contracts
- **Severity:** Medium
- **Affected files/lines:** `download_hf_contracts.py:245-247`, `download_hf_contracts.py:476-484`, `download_hf_contracts.py:564-567`
- **Evidence/impact:** The `contracts` table primary key is `address`. When a row has no address, the fallback hashes only `src[:200]`; distinct contracts sharing a prefix collide, and `INSERT OR IGNORE` silently drops later rows.
- **Recommendation:** Use the full `source_hash` or a stable dataset row id for fallback addresses, and log actual inserted vs. ignored row counts.

### 16. ABI event enrichment is advertised but not implemented
- **Severity:** Medium
- **Affected files/lines:** `src/abi_enrichment.py:560-565`, `src/abi_enrichment.py:633-638`, `src/bytecode_analyzer.py:1208-1217`, `src/bytecode_analyzer.py:1694-1696`
- **Evidence/impact:** The enrichment docstring promises LOG/event annotations, and the analyzer stores LOG topics in metadata. The TAC formatter emits only `logN(memory[...])`, and the enrichment branch for log lines appends the original line and continues without resolving any event.
- **Recommendation:** Emit topic0 in TAC and resolve it via ABI events, or remove the advertised LOG enrichment behavior until it is implemented.

### 17. Full TAC output repeats every block under every function
- **Severity:** Medium
- **Affected files/lines:** `src/bytecode_analyzer.py:689-692`, `src/bytecode_analyzer.py:1589-1591`
- **Evidence/impact:** Dispatcher-discovered functions are created with `basic_blocks=[]`. `_format_function_tac` falls back to all basic blocks when a function has no assigned blocks, so full TAC can duplicate the whole contract body under each function.
- **Recommendation:** Populate reachable blocks for each function or reuse the per-function traversal used by `generate_function_tac()`.

### 18. Dataset export drops stored compiler and selector metadata
- **Severity:** Medium
- **Affected files/lines:** `src/dataset_pipeline.py:825-833`, `src/dataset_pipeline.py:1461-1479`, `src/dataset_pipeline.py:1583-1609`, `docs/data-format.md:11-20`
- **Evidence/impact:** Compiler/optimizer metadata is added to pairs and stored in the DB, and export selects the `metadata` column. The JSONL export then writes a fixed metadata object that omits stored compiler/version/selector fields expected by the documented format.
- **Recommendation:** Parse and merge stored metadata into exported records, with explicit fields winning only when intentionally overriding stored values.

## Low severity

### 19. Default web model path does not match the bundled model directory
- **Severity:** Low
- **Affected files/lines:** `web/app.py:85`, `web/app.py:152-166`, `models/final_model_378/model_config.json:1-18`
- **Evidence/impact:** The app hardcodes `models/final_model`, but the current checkout contains a configured model under `models/final_model_378`. A default non-mock startup will fail to load the bundled model unless users know to rename or reconfigure paths.
- **Recommendation:** Make the model path configurable via CLI/env and document or auto-discover available `final_model*` directories.

### 20. Quantization documentation contradicts code defaults
- **Severity:** Low
- **Affected files/lines:** `README.md:237-240`, `docs/model-details.md:28-30`, `src/model_setup.py:55-56`, `src/model_setup.py:311-316`
- **Evidence/impact:** Documentation says quantization is 8-bit via bitsandbytes, while `ModelConfig` defaults to quantization with `load_in_4bit=True` and configures 4-bit NF4 loading.
- **Recommendation:** Update docs to say 4-bit NF4, or add selectable 8-bit support and document when each mode is used.

### 21. Documentation links to a missing dataset-generation guide
- **Severity:** Low
- **Affected files/lines:** `docs/contributing.md:39-43`
- **Evidence/impact:** The contribution guide points readers to `docs/dataset-generation.md`, but that file is not tracked in the current checkout. New contributors following the link hit a dead end.
- **Recommendation:** Add the missing guide or update the link to the maintained data-generation docs/runbook section.

### 22. Tracked VS Code file conflicts with ignore policy
- **Severity:** Low
- **Affected files/lines:** `.gitignore:15-17`, `.vscode/launch.json:1-12`
- **Evidence/impact:** `.gitignore` ignores `.vscode/`, but `.vscode/launch.json` is tracked. Future IDE settings may be ignored locally yet still appear as repository-owned configuration, creating confusion.
- **Recommendation:** Either untrack `.vscode/launch.json` or explicitly allowlist it with a comment if it is intentional shared configuration.
