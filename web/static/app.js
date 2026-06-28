/* ===================================================================
   Smart Contract Bytecode Decompiler — Frontend Logic
   =================================================================== */

(function () {
  "use strict";

  // ---- DOM refs ----
  const inputEl = document.getElementById("bytecode-input");
  const charCountEl = document.getElementById("char-count");
  const btnDecompile = document.getElementById("btn-decompile");
  const btnSample = document.getElementById("btn-sample");
  const btnClear = document.getElementById("btn-clear");
  const apiKeyInput = document.getElementById("api-key-input");
  const bytecodeGuidance = document.getElementById("bytecode-guidance");
  const abiInput = document.getElementById("abi-input");
  const abiFileInput = document.getElementById("abi-file-input");
  const metadataInput = document.getElementById("metadata-input");
  const metadataFileInput = document.getElementById("metadata-file-input");
  const maxNewTokensInput = document.getElementById("max-new-tokens-input");
  const temperatureInput = document.getElementById("temperature-input");
  const doSampleInput = document.getElementById("do-sample-input");
  const repetitionPenaltyInput = document.getElementById("repetition-penalty-input");
  const btnResetGeneration = document.getElementById("btn-reset-generation");
  const loadingEl = document.getElementById("loading");
  const btnCancelDecompile = document.getElementById("btn-cancel-decompile");
  const errorBanner = document.getElementById("error-banner");
  const errorText = document.getElementById("error-text");
  const btnDismissError = document.getElementById("btn-dismiss-error");
  const resultsSection = document.getElementById("results-section");
  const solidityOutput = document.getElementById("solidity-output");
  const tacOutput = document.getElementById("tac-output");
  const analysisContent = document.getElementById("analysis-content");
  const modelWarning = document.getElementById("model-warning");
  const modelWarningText = document.getElementById("model-warning-text");
  const functionMappingContent = document.getElementById("function-mapping-content");
  const reconstructionContent = document.getElementById("reconstruction-content");
  const traceDiagnostics = document.getElementById("trace-diagnostics");
  const securityContent = document.getElementById("security-content");
  const btnVulnerabilityScan = document.getElementById("btn-vulnerability-scan");
  const btnClassify = document.getElementById("btn-classify");
  const btnAuditReport = document.getElementById("btn-audit-report");

  // Progress elements
  const progressMessage = document.getElementById("progress-message");
  const progressFill = document.getElementById("progress-fill");
  const progressPercent = document.getElementById("progress-percent");
  const progressStage = document.getElementById("progress-stage");
  const progressFunctions = document.getElementById("progress-functions");
  const progressFuncCount = document.getElementById("progress-func-count");
  const progressLog = document.getElementById("progress-log");
  const progressLogList = document.getElementById("progress-log-list");

  // GPU elements
  const gpuPanel = document.getElementById("gpu-panel");
  const gpuCards = document.getElementById("gpu-cards");
  const gpuStatusDot = document.getElementById("gpu-status-dot");
  const readinessContent = document.getElementById("readiness-content");
  const readinessStatusDot = document.getElementById("readiness-status-dot");

  // ---- Sample runtime bytecode for demo use ----
  const SAMPLE_BYTECODE =
    "0x608060405234801561000f575f80fd5b5060043610610034575f3560e01c8063893d20e814610038578063a6f9dae114610056575b5f80fd5b610040610072565b60405161004d919061014e565b60405180910390f35b610070600480360381019061006b9190610195565b610099565b005b5f805f9054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905090565b8073ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff160361010c57805f806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505b50565b5f73ffffffffffffffffffffffffffffffffffffffff82169050919050565b5f6101388261010f565b9050919050565b6101488161012e565b82525050565b5f6020820190506101615f83018461013f565b92915050565b5f80fd5b6101748161012e565b811461017e575f80fd5b50565b5f8135905061018f8161016b565b92915050565b5f602082840312156101aa576101a9610167565b5b5f6101b784828501610181565b9150509291505056fea264697066735822122020389b4014c2d8511dc28898fda8ea80a2215a6e55690e4770b83b23ce7e209364736f6c63430008180033";

  // ---- State tracking ----
  var functionStartTime = 0;
  var gpuPollTimer = null;
  var currentSelectorMap = null; // Stored from last decompilation
  var totalFunctionCount = 0;
  var completedFunctionCount = 0;
  var activeDecompileRequestId = 0;
  var activeDecompileController = null;
  var activeServerJobId = null;
  var resultRevealTimer = null;
  var healthInfo = null;
  var lastDecompileResult = null;
  var maxBytecodeHexLength = 200000;
  var generationDefaults = {
    max_new_tokens: 1024,
    temperature: 0.1,
    do_sample: false,
    repetition_penalty: 1.05
  };
  var generationControlsInitialized = false;

  // ---- Helpers ----

  function show(el) {
    el.classList.remove("hidden");
  }
  function hide(el) {
    el.classList.add("hidden");
  }

  function showError(msg) {
    errorText.textContent = msg;
    show(errorBanner);
  }
  function hideError() {
    hide(errorBanner);
  }

  function setDownloadButtonsEnabled(enabled) {
    document.querySelectorAll(".btn-download").forEach(function (btn) {
      btn.disabled = !enabled;
    });
  }

  function apiHeaders(baseHeaders) {
    var headers = Object.assign({}, baseHeaders || {});
    var apiKey = apiKeyInput ? apiKeyInput.value.trim() : "";
    if (apiKey) {
      headers.Authorization = "Bearer " + apiKey;
    }
    return headers;
  }

  function resetProgress() {
    progressFill.style.width = "0%";
    progressPercent.textContent = "0%";
    progressMessage.textContent = "Processing bytecode…";
    progressStage.textContent = "";
    hide(progressFunctions);
    progressFuncCount.textContent = "";
    hide(progressLog);
    progressLogList.innerHTML = "";
    functionStartTime = 0;
    totalFunctionCount = 0;
    completedFunctionCount = 0;
  }

  function setLoading(on) {
    if (on) {
      resetProgress();
      show(loadingEl);
      hide(resultsSection);
      hideError();
      setDownloadButtonsEnabled(false);
      btnDecompile.disabled = true;
      btnSample.disabled = true;
      inputEl.readOnly = true;
      if (abiInput) abiInput.readOnly = true;
      if (metadataInput) metadataInput.readOnly = true;
    } else {
      hide(loadingEl);
      btnDecompile.disabled = false;
      btnSample.disabled = false;
      inputEl.readOnly = false;
      if (abiInput) abiInput.readOnly = false;
      if (metadataInput) metadataInput.readOnly = false;
    }
  }

  function clearResultRevealTimer() {
    if (resultRevealTimer) {
      clearTimeout(resultRevealTimer);
      resultRevealTimer = null;
    }
  }

  function isCurrentDecompileRequest(requestId) {
    return requestId === activeDecompileRequestId;
  }

  function finishDecompileRequest(requestId) {
    if (!isCurrentDecompileRequest(requestId)) return;
    activeDecompileController = null;
    activeServerJobId = null;
    setLoading(false);
  }

  function abortActiveDecompile() {
    clearResultRevealTimer();
    var jobId = activeServerJobId;
    if (activeDecompileController) {
      activeDecompileController.abort();
      activeDecompileController = null;
      activeDecompileRequestId += 1;
    }
    if (jobId) {
      fetch("/api/decompile/" + encodeURIComponent(jobId) + "/cancel", {
        method: "POST",
        headers: apiHeaders()
      }).catch(function () {});
    }
    activeServerJobId = null;
    setLoading(false);
  }

  function newClientJobId() {
    if (window.crypto && window.crypto.randomUUID) {
      return window.crypto.randomUUID();
    }
    return "web-" + Date.now() + "-" + Math.random().toString(16).slice(2);
  }

  function updateProgress(pct, message) {
    progressFill.style.width = pct + "%";
    progressPercent.textContent = pct + "%";
    if (message) {
      progressMessage.textContent = message;
    }
  }

  function renderFunctionCount() {
    progressFuncCount.innerHTML =
      '<span class="func-count-highlight">' +
      completedFunctionCount +
      "</span> / " +
      totalFunctionCount +
      " decompiled";
  }

  function markFunctionProcessed() {
    completedFunctionCount += 1;
    if (totalFunctionCount > 0) {
      completedFunctionCount = Math.min(
        completedFunctionCount,
        totalFunctionCount
      );
    }
    renderFunctionCount();
  }

  function normalizeBytecodeInput(raw) {
    var text = (raw || "").trim();
    var body = text.toLowerCase().startsWith("0x") ? text.slice(2) : text;
    body = body.replace(/\s+/g, "");
    return {
      bytecode: "0x" + body,
      hexBody: body
    };
  }

  function validateBytecodeInput(raw) {
    var normalized = normalizeBytecodeInput(raw);
    var body = normalized.hexBody;
    if (!body) {
      return { valid: false, error: "Please enter EVM bytecode to decompile." };
    }
    if (body.length > maxBytecodeHexLength) {
      return {
        valid: false,
        error:
          "Bytecode is too large (" +
          body.length.toLocaleString() +
          " hex chars). Maximum is " +
          maxBytecodeHexLength.toLocaleString() +
          "."
      };
    }
    if (body.length % 2 !== 0) {
      return {
        valid: false,
        error: "Bytecode must contain an even number of hex characters."
      };
    }
    if (!/^[0-9a-fA-F]+$/.test(body)) {
      return {
        valid: false,
        error: "Bytecode contains non-hexadecimal characters."
      };
    }
    return {
      valid: true,
      bytecode: normalized.bytecode,
      hexLength: body.length,
      byteLength: body.length / 2
    };
  }

  function updateCharCount() {
    var normalized = normalizeBytecodeInput(inputEl.value);
    var hexLen = normalized.hexBody.length;
    var bytes = Math.floor(hexLen / 2);
    charCountEl.textContent =
      hexLen.toLocaleString() +
      " / " +
      maxBytecodeHexLength.toLocaleString() +
      " hex chars (" +
      bytes.toLocaleString() +
      " bytes)";
    if (bytecodeGuidance) {
      bytecodeGuidance.textContent =
        "Hex bytecode may include whitespace and an optional 0x prefix. Limit: " +
        maxBytecodeHexLength.toLocaleString() +
        " hex chars (" +
        Math.floor(maxBytecodeHexLength / 2).toLocaleString() +
        " bytes).";
    }
  }

  function applyGenerationDefaults(defaults) {
    generationDefaults = Object.assign(generationDefaults, defaults || {});
    if (maxNewTokensInput) maxNewTokensInput.value = generationDefaults.max_new_tokens;
    if (temperatureInput) temperatureInput.value = generationDefaults.temperature;
    if (doSampleInput) doSampleInput.value = generationDefaults.do_sample ? "true" : "false";
    if (repetitionPenaltyInput) {
      repetitionPenaltyInput.value = generationDefaults.repetition_penalty;
    }
  }

  function collectGenerationConfig() {
    var maxNew = parseInt(maxNewTokensInput ? maxNewTokensInput.value : generationDefaults.max_new_tokens, 10);
    var temp = parseFloat(temperatureInput ? temperatureInput.value : generationDefaults.temperature);
    var rep = parseFloat(repetitionPenaltyInput ? repetitionPenaltyInput.value : generationDefaults.repetition_penalty);
    if (!Number.isFinite(maxNew) || maxNew < 1) {
      throw new Error("Max new tokens must be at least 1.");
    }
    if (!Number.isFinite(temp) || temp < 0 || temp > 2) {
      throw new Error("Temperature must be between 0 and 2.");
    }
    if (!Number.isFinite(rep) || rep < 0.8 || rep > 2) {
      throw new Error("Repetition penalty must be between 0.8 and 2.");
    }
    return {
      max_new_tokens: maxNew,
      temperature: temp,
      do_sample: doSampleInput ? doSampleInput.value === "true" : false,
      repetition_penalty: rep
    };
  }

  function parseOptionalJsonInput(el, label) {
    if (!el) return null;
    var raw = el.value.trim();
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch (e) {
      throw new Error(label + " must be valid JSON: " + e.message);
    }
  }

  function collectUserMetadata() {
    var metadata = parseOptionalJsonInput(metadataInput, "Contract metadata");
    var abi = parseOptionalJsonInput(abiInput, "ABI");
    return { metadata: metadata, abi: abi };
  }

  function loadJsonFileIntoTextarea(fileInput, targetInput, label) {
    if (!fileInput || !targetInput || !fileInput.files || !fileInput.files[0]) return;
    var file = fileInput.files[0];
    var reader = new FileReader();
    reader.onload = function () {
      targetInput.value = String(reader.result || "");
      fileInput.value = "";
    };
    reader.onerror = function () {
      showError("Unable to read " + label + " file.");
      fileInput.value = "";
    };
    reader.readAsText(file);
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
  }

  function formatElapsed(ms) {
    if (ms < 1000) return ms + "ms";
    return (ms / 1000).toFixed(1) + "s";
  }

  // ---- Progress log helpers ----

  function addLogEntry(fname, status, elapsed, source, confidence) {
    show(progressLog);

    // Check if entry already exists (update it)
    var existing = document.getElementById("log-" + fname);
    if (existing) {
      existing.className = status;
      var icon = existing.querySelector(".log-icon");
      if (status === "in-progress") {
        icon.textContent = "⟳";
      } else if (status === "completed") {
        icon.textContent = "✓";
      } else if (status === "error") {
        icon.textContent = "✗";
      } else if (status === "exact_match") {
        icon.textContent = "⚡";
      } else if (status === "pending") {
        icon.textContent = "○";
      }
      if (elapsed !== undefined) {
        var timeEl = existing.querySelector(".log-time");
        if (timeEl) timeEl.textContent = formatElapsed(elapsed);
      }
      // Update source badge if provided
      if (source) {
        var badgeEl = existing.querySelector(".log-source");
        if (!badgeEl) {
          badgeEl = document.createElement("span");
          badgeEl.className = "log-source";
          existing.insertBefore(badgeEl, existing.querySelector(".log-time") || null);
        }
        badgeEl.className = "log-source " + getSourceClass(source);
        badgeEl.textContent = getSourceLabel(source);
        badgeEl.title = getSourceTooltip(source);
      }
      // Update confidence badge if provided
      if (confidence != null) {
        var confEl = existing.querySelector(".log-confidence");
        if (!confEl) {
          confEl = document.createElement("span");
          confEl.className = "log-confidence";
          var srcEl = existing.querySelector(".log-source");
          if (srcEl && srcEl.nextSibling) {
            existing.insertBefore(confEl, srcEl.nextSibling);
          } else {
            existing.insertBefore(confEl, existing.querySelector(".log-time") || null);
          }
        }
        confEl.className = "log-confidence " + getConfidenceClass(confidence);
        confEl.textContent = Math.round(confidence) + "%";
        confEl.title = "Confidence: " + confidence.toFixed(1) + "%";
      }
      // Auto-scroll
      progressLog.scrollTop = progressLog.scrollHeight;
      return;
    }

    var li = document.createElement("li");
    li.id = "log-" + fname;
    li.className = status;

    var iconSpan = document.createElement("span");
    iconSpan.className = "log-icon";
    if (status === "in-progress") {
      iconSpan.textContent = "⟳";
    } else if (status === "completed") {
      iconSpan.textContent = "✓";
    } else if (status === "error") {
      iconSpan.textContent = "✗";
    } else if (status === "exact_match") {
      iconSpan.textContent = "⚡";
    } else if (status === "pending") {
      iconSpan.textContent = "○";
    }

    var nameSpan = document.createElement("span");
    nameSpan.className = "log-name";
    nameSpan.textContent = fname;

    li.appendChild(iconSpan);
    li.appendChild(nameSpan);

    // Source badge
    if (source) {
      var sourceBadge = document.createElement("span");
      sourceBadge.className = "log-source " + getSourceClass(source);
      sourceBadge.textContent = getSourceLabel(source);
      sourceBadge.title = getSourceTooltip(source);
      li.appendChild(sourceBadge);
    }

    // Confidence badge
    if (confidence != null) {
      var confBadge = document.createElement("span");
      confBadge.className = "log-confidence " + getConfidenceClass(confidence);
      confBadge.textContent = Math.round(confidence) + "%";
      confBadge.title = "Confidence: " + confidence.toFixed(1) + "%";
      li.appendChild(confBadge);
    }

    if (elapsed !== undefined) {
      var timeSpan = document.createElement("span");
      timeSpan.className = "log-time";
      timeSpan.textContent = formatElapsed(elapsed);
      li.appendChild(timeSpan);
    }

    progressLogList.appendChild(li);

    // Auto-scroll to bottom
    progressLog.scrollTop = progressLog.scrollHeight;
  }

  function addBatchLogEntry(batchIndex, status, elapsedSeconds, size) {
    show(progressLog);
    var id = "batch-" + batchIndex;
    var existing = document.getElementById(id);
    var text = "Batch " + batchIndex + " (" + size + " function" +
      (size === 1 ? "" : "s") + ")";
    var elapsed = elapsedSeconds != null ? Number(elapsedSeconds) * 1000 : undefined;
    if (existing) {
      existing.className = status;
      var icon = existing.querySelector(".log-icon");
      if (icon) icon.textContent = status === "completed" ? "✓" : "⟳";
      var timeEl = existing.querySelector(".log-time");
      if (timeEl && elapsed !== undefined) timeEl.textContent = formatElapsed(Math.round(elapsed));
      return;
    }
    var li = document.createElement("li");
    li.id = id;
    li.className = status;
    var iconSpan = document.createElement("span");
    iconSpan.className = "log-icon";
    iconSpan.textContent = status === "completed" ? "✓" : "⟳";
    var nameSpan = document.createElement("span");
    nameSpan.className = "log-name";
    nameSpan.textContent = text;
    li.appendChild(iconSpan);
    li.appendChild(nameSpan);
    if (elapsed !== undefined) {
      var timeSpan = document.createElement("span");
      timeSpan.className = "log-time";
      timeSpan.textContent = formatElapsed(Math.round(elapsed));
      li.appendChild(timeSpan);
    }
    progressLogList.appendChild(li);
    progressLog.scrollTop = progressLog.scrollHeight;
  }

  // ---- Source/confidence formatting helpers ----

  function getSourceClass(source) {
    switch (source) {
      case "exact_match": return "source-lookup";
      case "model_inference": return "source-model";
      case "pending_inference": return "source-pending";
      case "error": return "source-error";
      default: return "source-unknown";
    }
  }

  function getSourceLabel(source) {
    switch (source) {
      case "exact_match": return "DB";
      case "model_inference": return "LLM";
      case "pending_inference": return "queued";
      case "error": return "err";
      default: return "?";
    }
  }

  function getSourceTooltip(source) {
    switch (source) {
      case "exact_match": return "Resolved via TAC hash lookup (exact match from database)";
      case "model_inference": return "Generated by Qwen2.5-Coder model inference";
      case "pending_inference": return "Awaiting model inference";
      case "error": return "Decompilation failed";
      default: return "";
    }
  }

  function getConfidenceClass(conf) {
    if (conf >= 80) return "conf-high";
    if (conf >= 50) return "conf-med";
    return "conf-low";
  }

  // ---- Tab switching ----

  function activateTab(targetId) {
    document.querySelectorAll(".tab").forEach(function (t) {
      t.classList.remove("active");
      if (t.getAttribute("data-tab") === targetId) {
        t.classList.add("active");
      }
    });
    document.querySelectorAll(".tab-panel").forEach(function (p) {
      p.classList.add("hidden");
      p.classList.remove("active");
    });
    const panel = document.getElementById(targetId);
    if (panel) {
      panel.classList.remove("hidden");
      panel.classList.add("active");
    }
  }

  document.querySelectorAll(".tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      activateTab(this.getAttribute("data-tab"));
    });
  });

  // ---- Copy buttons ----

  document.querySelectorAll(".btn-copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      const targetId = this.getAttribute("data-target");
      const el = document.getElementById(targetId);
      if (!el) return;
      navigator.clipboard.writeText(el.textContent).then(
        function () {
          const orig = btn.textContent;
          btn.textContent = "✓ Copied";
          setTimeout(function () {
            btn.textContent = orig;
          }, 1500);
        },
        function () {
          const range = document.createRange();
          range.selectNodeContents(el);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
        }
      );
    });
  });

  function safeFilenamePart(value) {
    return String(value || "decompile")
      .replace(/[^a-zA-Z0-9_.-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 80) || "decompile";
  }

  function downloadText(filename, mimeType, text) {
    var blob = new Blob([text], { type: mimeType });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function handleDownloadArtifact(kind) {
    if (!lastDecompileResult) return;
    var base = safeFilenamePart(lastDecompileResult.request_id || "decompile");
    if (kind === "solidity") {
      downloadText(base + ".sol", "text/plain;charset=utf-8",
        lastDecompileResult.solidity || "");
    } else if (kind === "tac") {
      downloadText(base + ".tac", "text/plain;charset=utf-8",
        lastDecompileResult.tac || "");
    } else if (kind === "json") {
      downloadText(base + ".json", "application/json;charset=utf-8",
        JSON.stringify(lastDecompileResult, null, 2));
    }
  }

  document.querySelectorAll(".btn-download").forEach(function (btn) {
    btn.addEventListener("click", function () {
      handleDownloadArtifact(this.getAttribute("data-artifact"));
    });
  });
  setDownloadButtonsEnabled(false);

  // ---- Build analysis cards ----

  function renderAnalysis(analysis) {
    analysisContent.innerHTML = "";

    function card(label, value, isSmall) {
      var cls = isSmall ? "value small" : "value";
      return (
        '<div class="analysis-card">' +
        '<div class="label">' +
        escapeHtml(label) +
        "</div>" +
        '<div class="' +
        cls +
        '">' +
        escapeHtml(String(value)) +
        "</div>" +
        "</div>"
      );
    }

    var html = "";
    html += card("Instructions", analysis.num_instructions || 0);
    html += card("Basic Blocks", analysis.num_basic_blocks || 0);
    html += card("Functions Identified", analysis.num_functions || 0);
    html += card(
      "TAC Generation Time",
      (analysis.tac_generation_time_s || 0) + "s"
    );
    html += card(
      "Solidity Generation Time",
      (analysis.solidity_generation_time_s || 0) + "s"
    );
    if (analysis.lookup_available !== undefined) {
      html += card(
        "TAC Lookup",
        analysis.lookup_available ? "Available" : "Unavailable"
      );
    }
    if (analysis.lookup_hits !== undefined) {
      html += card("Lookup Hits", analysis.lookup_hits);
    }
    if (analysis.source_summary) {
      html += card("DB Functions", analysis.source_summary.exact_match || 0);
      html += card("LLM Functions", analysis.source_summary.model_inference || 0);
      html += card("Failed Functions", analysis.source_summary.error || 0);
      html += card("ABI Functions", analysis.source_summary.abi_functions_used || 0);
      html += card("Validation Failed", analysis.source_summary.validation_failed || 0);
    }
    if (analysis.reconstruction_strategy) {
      html += card("Reconstruction", analysis.reconstruction_strategy, true);
    }
    if (analysis.semantic_chunk_count !== undefined) {
      html += card("Semantic Chunks", analysis.semantic_chunk_count);
    }
    if (analysis.detected_interfaces && analysis.detected_interfaces.length) {
      html += card("Interface Hints", analysis.detected_interfaces.join(", "), true);
    }
    if (analysis.proxy_like !== undefined) {
      html += card("Proxy-like", analysis.proxy_like ? "yes" : "no");
    }
    if (analysis.validation) {
      html += card(
        "Solidity Validation",
        (analysis.validation.valid ? "valid" : "invalid") +
          " (" + (analysis.validation.method || "unknown") + ")",
        true
      );
      if (analysis.validation.scaffold_errors && analysis.validation.scaffold_errors.length) {
        html += card("Validation Errors", analysis.validation.scaffold_errors.join(", "), true);
      }
    }
    if (analysis.quality) {
      var q = analysis.quality;
      html += card("Reconstruction Quality", qualitySummary(q), true);
      html += card("Compiler Checked", q.compiler_checked ? "yes" : "no");
      html += card("Deployable", q.deployable ? "yes" : "no");
      if (q.selector_coverage != null) {
        html += card("Selector Coverage", Math.round(Number(q.selector_coverage) * 100) + "%");
      }
      if (q.recommended_actions && q.recommended_actions.length) {
        html += card("Recommended Actions", q.recommended_actions.join(" "), true);
      }
    }
    if (analysis.effective_generation_config) {
      var gen = analysis.effective_generation_config;
      html += card(
        "Generation",
        "max_new_tokens=" +
          gen.max_new_tokens +
          ", temp=" +
          gen.temperature +
          ", sample=" +
          gen.do_sample +
          ", repeat=" +
          gen.repetition_penalty,
        true
      );
    }
    if (analysis.model_path) {
      html += card("Model Path", analysis.model_path, true);
    }

    if (analysis.model_config) {
      var mc = analysis.model_config;
      html += card("Base Model", mc.model_name || "—", true);
      html += card("LoRA Rank", mc.lora_rank || "—");
      html += card("LoRA Alpha", mc.lora_alpha || "—");
      html += card("Max Sequence Length", mc.max_sequence_length || "—");
      html += card(
        "Quantization",
        mc.use_quantization ? "4-bit (NF4)" : "None"
      );
      if (mc.target_modules) {
        html += card("Target Modules", mc.target_modules.join(", "), true);
      }
    }

    analysisContent.innerHTML = html;
  }

  // ================================================================== //
  //  Function Selector Mapping
  // ================================================================== //

  function diagnosticSummary(diag) {
    if (!diag) return "—";
    var parts = [];
    if (diag.tac_tokens_before != null) {
      parts.push("TAC " + diag.tac_tokens_before + "→" + diag.tac_tokens_after);
    }
    if (diag.prompt_tokens != null) parts.push("prompt " + diag.prompt_tokens);
    if (diag.generated_tokens != null) parts.push("gen " + diag.generated_tokens);
    if (diag.tac_truncated) parts.push("truncated");
    return parts.join(", ") || "—";
  }

  function qualitySummary(quality) {
    if (!quality) return "—";
    var parts = [];
    if (quality.severity) parts.push(quality.severity);
    if (quality.deployable === false) parts.push("non-deployable");
    else if (quality.deployable === true) parts.push("deployable");
    if (quality.compiler_checked === false) parts.push("unchecked");
    if (quality.scaffold_only) parts.push("scaffold-only");
    if (quality.score != null) parts.push("score " + Number(quality.score).toFixed(2));
    return parts.join(", ") || "—";
  }

  function provenanceSummary(provenance) {
    if (!provenance) return "—";
    var parts = [];
    if (provenance.build_id) parts.push("build " + provenance.build_id);
    if (provenance.tac_hash) parts.push("tac " + String(provenance.tac_hash).slice(0, 10));
    if (provenance.source_row_count != null) parts.push(provenance.source_row_count + " rows");
    if (provenance.compiler_version) parts.push("solc " + provenance.compiler_version);
    return parts.join(", ") || "lookup";
  }

  function renderTraceDiagnostics(data) {
    if (!traceDiagnostics) return;
    var functionResults = data.function_results || [];
    var truncated = 0;
    var generatedTokens = 0;
    for (var i = 0; i < functionResults.length; i++) {
      var diag = functionResults[i].diagnostics || {};
      if (diag.tac_truncated) truncated += 1;
      if (diag.generated_tokens != null) generatedTokens += Number(diag.generated_tokens) || 0;
    }
    var html = "";
    if (data.trace_path) {
      html += '<div class="diagnostic-card"><div class="label">Trace path</div><code>' +
        escapeHtml(data.trace_path) + "</code></div>";
    }
    html += '<div class="diagnostic-card"><div class="label">Token diagnostics</div>' +
      escapeHtml(functionResults.length + " function(s), " + truncated +
        " truncated, " + generatedTokens + " generated token(s)") + "</div>";
    if (data.contract_metadata && data.contract_metadata.abi &&
        data.contract_metadata.abi.provided) {
      var abi = data.contract_metadata.abi;
      html += '<div class="diagnostic-card"><div class="label">ABI metadata</div>' +
        escapeHtml((abi.function_count || 0) + " functions, " +
          (abi.event_count || 0) + " events, " + (abi.error_count || 0) +
          " errors from " + (abi.source || "request")) + "</div>";
    }
    if (data.validation) {
      html += '<div class="diagnostic-card"><div class="label">Validation</div>' +
        escapeHtml((data.validation.valid ? "valid" : "invalid") +
          " via " + (data.validation.method || "unknown")) + "</div>";
    }
    if (data.quality) {
      html += '<div class="diagnostic-card"><div class="label">Quality</div>' +
        escapeHtml(qualitySummary(data.quality)) + "</div>";
    }
    if (data.reconstruction) {
      html += '<div class="diagnostic-card"><div class="label">Reconstruction</div>' +
        escapeHtml((data.reconstruction.chunk_count || 0) + " semantic chunk(s), " +
          (data.reconstruction.assembly && data.reconstruction.assembly.mode ||
            "deterministic assembly")) + "</div>";
    }
    traceDiagnostics.innerHTML = html;
  }

  function renderReconstruction(reconstruction) {
    if (!reconstructionContent) return;
    if (!reconstruction) {
      reconstructionContent.innerHTML =
        '<p class="text-muted">No reconstruction plan returned.</p>';
      return;
    }

    var facts = reconstruction.contract_facts || {};
    var proxy = facts.proxy || {};
    var interfaces = facts.detected_interfaces || [];
    var chunks = reconstruction.semantic_chunks || [];
    var html = '<div class="reconstruction-summary">';
    html += '<div class="analysis-card"><div class="label">Strategy</div><div class="value small">' +
      escapeHtml(reconstruction.strategy || "semantic_function_chunks") + "</div></div>";
    html += '<div class="analysis-card"><div class="label">Assembly</div><div class="value small">' +
      escapeHtml((reconstruction.assembly && reconstruction.assembly.mode) ||
        "deterministic_reconciliation") + "</div></div>";
    html += '<div class="analysis-card"><div class="label">Chunks</div><div class="value">' +
      escapeHtml(String(reconstruction.chunk_count || chunks.length || 0)) + "</div></div>";
    html += '<div class="analysis-card"><div class="label">Selectors</div><div class="value small">' +
      escapeHtml((facts.selectors || []).join(", ") || "none") + "</div></div>";
    html += '<div class="analysis-card"><div class="label">Interface Hints</div><div class="value small">' +
      escapeHtml(interfaces.map(function (item) { return item.name; }).join(", ") || "none") +
      "</div></div>";
    html += '<div class="analysis-card"><div class="label">Proxy-like</div><div class="value small">' +
      escapeHtml(proxy.is_proxy_like ? "yes" : "no") + "</div></div>";
    html += "</div>";

    if (chunks.length) {
      html += '<table class="fn-map-table reconstruction-table"><thead><tr>' +
        "<th>#</th><th>Chunk</th><th>Kind</th><th>Selector</th>" +
        "<th>Blocks</th><th>Storage</th><th>Calls/Logs</th><th>Signature</th>" +
        "</tr></thead><tbody>";
      for (var i = 0; i < chunks.length; i++) {
        var chunk = chunks[i];
        var resolution = chunk.selector_resolution || {};
        var storage = "r" + ((chunk.storage_reads || []).length || 0) +
          " / w" + ((chunk.storage_writes || []).length || 0);
        var effects = "calls " + (chunk.external_calls || 0) +
          ", logs " + (chunk.logs || 0) +
          ", reverts " + (chunk.reverts || 0);
        html += "<tr>";
        html += "<td>" + escapeHtml(String(chunk.index || i + 1)) + "</td>";
        html += '<td class="fn-map-fname"><code>' + escapeHtml(chunk.name || "chunk") +
          "</code></td>";
        html += "<td>" + escapeHtml(chunk.kind || "semantic_region") + "</td>";
        html += "<td><code>" + escapeHtml(chunk.selector || "—") + "</code></td>";
        html += "<td>" + escapeHtml(String(chunk.basic_block_count || 0)) + "</td>";
        html += "<td>" + escapeHtml(storage) + "</td>";
        html += "<td>" + escapeHtml(effects) + "</td>";
        html += '<td class="fn-map-sig"><code>' +
          escapeHtml(resolution.signature || "—") + "</code></td>";
        html += "</tr>";
      }
      html += "</tbody></table>";
    }

    reconstructionContent.innerHTML = html;
  }

  function renderFunctionMapping(selectorMap, functionResults) {
    var hasSelectors = selectorMap && Object.keys(selectorMap).length > 0;
    var hasResults = functionResults && functionResults.length > 0;
    if (!hasSelectors && !hasResults) {
      functionMappingContent.innerHTML =
        '<p class="text-muted">No function data returned.</p>';
      return;
    }

    var html = '<table class="fn-map-table">';
    html +=
      "<thead><tr>" +
      "<th>Bytecode Function</th>" +
      "<th>Resolved Name</th>" +
      "<th>Confidence</th>" +
      "<th>Source</th>" +
      "<th>Lookup Provenance</th>" +
      "<th>Quality</th>" +
      "<th>Status</th>" +
      "<th>Elapsed</th>" +
      "<th>Diagnostics</th>" +
      "<th>Error</th>" +
      "</tr></thead><tbody>";

    var fnames = hasResults
      ? functionResults.map(function (r) { return r.name; })
      : Object.keys(selectorMap);
    for (var i = 0; i < fnames.length; i++) {
      var fname = fnames[i];
      var info = hasSelectors ? (selectorMap[fname] || {}) : {};
      var best = info.best_match || {};
      var result = hasResults ? functionResults[i] : {};

      var conf = result.confidence != null ? result.confidence : best.confidence;
      var confClass = "conf-high";
      if (conf < 50) confClass = "conf-low";
      else if (conf < 80) confClass = "conf-med";

      var selectorSource =
        best.source === "builtin" ? "Known Standard" :
        best.source === "4byte" ? "4byte.directory" :
        best.source === "abi" ? "User ABI" : (best.source || "Unknown");
      var provenance = result.source || "unknown";
      var status = result.status || "unknown";
      var error = result.error || "";
      var diag = result.diagnostics || {};
      var quality = result.quality || null;
      var provenanceInfo = result.lookup_provenance || null;
      var abi = result.abi || (info.abi || null);
      var validation = result.validation || null;

      html += "<tr>";
      html +=
        '<td class="fn-map-fname"><code>' +
        escapeHtml(fname) +
        "</code></td>";
      html +=
        '<td class="fn-map-sig"><code>' +
        escapeHtml(result.signature || best.signature || "—") +
        "</code></td>";
      if (conf != null) {
        html +=
          '<td><span class="conf-badge ' +
          confClass +
          '" title="' +
          Number(conf).toFixed(1) +
          '% confidence">' +
          Number(conf).toFixed(0) +
          "%</span></td>";
      } else {
        html += "<td>—</td>";
      }
      html += '<td class="fn-map-source">' +
        escapeHtml(getSourceLabel(provenance) + " / " + selectorSource) + "</td>";
      html += '<td class="fn-map-diagnostics">' +
        '<span title="' + escapeHtml(JSON.stringify(provenanceInfo || {}, null, 2)) + '">' +
        escapeHtml(provenanceSummary(provenanceInfo)) + "</span></td>";
      html += '<td class="fn-map-diagnostics">' +
        '<span title="' + escapeHtml(JSON.stringify(quality || {}, null, 2)) + '">' +
        escapeHtml(qualitySummary(quality)) + "</span></td>";
      html += "<td>" + escapeHtml(status) +
        (validation && !validation.valid ? ' <span class="diag-badge truncated">invalid</span>' : "") +
        (abi ? ' <span class="diag-badge abi">ABI</span>' : "") + "</td>";
      html += "<td>" + escapeHtml(result.elapsed_s != null ? result.elapsed_s + "s" : "—") + "</td>";
      html += '<td class="fn-map-diagnostics">' +
        '<span title="' + escapeHtml(JSON.stringify(diag || {}, null, 2)) + '">' +
        escapeHtml(diagnosticSummary(diag)) +
        "</span>" +
        (diag.tac_truncated ? ' <span class="diag-badge truncated">truncated</span>' : "") +
        "</td>";
      html += "<td>" + escapeHtml(error || "—") + "</td>";
      html += "</tr>";

      if (info.candidates && info.candidates.length > 1) {
        html +=
          '<tr class="fn-map-alt"><td></td><td colspan="9">' +
          escapeHtml((info.candidates.length - 1) + " alternate selector candidate(s) available") +
          "</td></tr>";
      }
    }

    html += "</tbody></table>";
    functionMappingContent.innerHTML = html;
  }

  // ================================================================== //
  //  GPU Stats
  // ================================================================== //

  function formatMB(mb) {
    if (mb >= 1024) return (mb / 1024).toFixed(1) + " GB";
    return mb.toFixed(0) + " MB";
  }

  function renderGpuStats(data) {
    if (!data) {
      gpuCards.innerHTML = '<div class="gpu-placeholder">Unable to fetch GPU stats.</div>';
      gpuStatusDot.className = "gpu-status-dot offline";
      gpuStatusDot.title = "Offline";
      return;
    }

    if (data.auth_required) {
      gpuCards.innerHTML =
        '<div class="gpu-placeholder">API key required to view GPU telemetry.</div>';
      gpuStatusDot.className = "gpu-status-dot offline";
      gpuStatusDot.title = "API key required";
      return;
    }

    if (!data.cuda_available || data.gpus.length === 0) {
      gpuCards.innerHTML =
        '<div class="gpu-placeholder">No CUDA GPU detected' +
        (data.error ? " — " + escapeHtml(data.error) : "") +
        "</div>";
      gpuStatusDot.className = "gpu-status-dot offline";
      gpuStatusDot.title = "No GPU";
      return;
    }

    gpuStatusDot.className = "gpu-status-dot online";
    gpuStatusDot.title = data.gpus.length + " GPU(s) available";

    var html = "";
    for (var i = 0; i < data.gpus.length; i++) {
      var g = data.gpus[i];
      html += '<div class="gpu-card">';

      // Title
      html +=
        '<div class="gpu-card-title">' +
        '<span class="gpu-card-name">' + escapeHtml(g.name) + "</span>" +
        '<span class="gpu-card-idx">GPU ' + g.index + "</span>" +
        "</div>";

      // Memory Bandwidth bar (primary metric for inference)
      var memCtrl = g.memory_controller_percent;
      var memCtrlStr = memCtrl != null ? memCtrl + "%" : "N/A";
      html +=
        '<div class="gpu-metric">' +
        '<div class="gpu-metric-label">Inference Load <span class="gpu-hint" title="Memory controller utilization — the true bottleneck for LLM inference (memory-bandwidth bound)">ⓘ</span></div>' +
        '<div class="gpu-bar-wrap">' +
        '<div class="gpu-bar">' +
        '<div class="gpu-bar-fill gpu-bar-inference" style="width:' +
        (memCtrl != null ? memCtrl : 0) +
        '%"></div>' +
        "</div>" +
        '<span class="gpu-bar-text">' +
        memCtrlStr +
        "</span>" +
        "</div></div>";

      // SM Compute utilization bar
      var gpuUtil = g.gpu_utilization_percent;
      var gpuUtilStr = gpuUtil != null ? gpuUtil + "%" : "N/A";
      html +=
        '<div class="gpu-metric">' +
        '<div class="gpu-metric-label">SM Compute <span class="gpu-hint" title="Streaming Multiprocessor utilization — % of time GPU cores were active">ⓘ</span></div>' +
        '<div class="gpu-bar-wrap">' +
        '<div class="gpu-bar">' +
        '<div class="gpu-bar-fill gpu-bar-util" style="width:' +
        (gpuUtil != null ? gpuUtil : 0) +
        '%"></div>' +
        "</div>" +
        '<span class="gpu-bar-text">' +
        gpuUtilStr +
        "</span>" +
        "</div></div>";

      // VRAM bar
      var memPct = g.memory_percent || 0;
      html +=
        '<div class="gpu-metric">' +
        '<div class="gpu-metric-label">VRAM</div>' +
        '<div class="gpu-bar-wrap">' +
        '<div class="gpu-bar">' +
        '<div class="gpu-bar-fill gpu-bar-mem" style="width:' +
        memPct +
        '%"></div>' +
        "</div>" +
        '<span class="gpu-bar-text">' +
        formatMB(g.memory_used_mb) +
        " / " +
        formatMB(g.memory_total_mb) +
        "</span>" +
        "</div></div>";

      // Extra stats row
      var extras = [];
      if (g.temperature_c != null) extras.push("🌡 " + g.temperature_c + "°C");
      if (g.power_w != null) {
        var pw = g.power_w + "W";
        if (g.power_limit_w != null) pw += " / " + g.power_limit_w + "W";
        extras.push("⚡ " + pw);
      }
      if (g.fan_speed_percent != null) extras.push("🌀 Fan " + g.fan_speed_percent + "%");
      if (g.clock_graphics_mhz != null) extras.push("Core " + g.clock_graphics_mhz + " MHz");
      if (g.clock_memory_mhz != null) extras.push("Mem " + g.clock_memory_mhz + " MHz");

      if (extras.length > 0) {
        html +=
          '<div class="gpu-extras">' + escapeHtml(extras.join("  ·  ")) + "</div>";
      }

      html += "</div>"; // .gpu-card
    }

    if (data.error) {
      html +=
        '<div class="gpu-warning">' + escapeHtml(data.error) + "</div>";
    }

    gpuCards.innerHTML = html;
  }

  function pollGpuStats() {
    fetch("/api/gpu-stats", { headers: apiHeaders() })
      .then(function (resp) {
        if (!resp.ok) {
          return resp.json().catch(function () {
            return {};
          }).then(function (data) {
            var message = data.error || "Unable to fetch GPU stats.";
            if (resp.status === 401 || resp.status === 403) {
              renderGpuStats({ auth_required: true });
              return null;
            }
            throw new Error(message);
          });
        }
        return resp.json();
      })
      .then(function (data) {
        if (data) renderGpuStats(data);
      })
      .catch(function () {
        renderGpuStats(null);
      });
  }

  function startGpuPolling() {
    pollGpuStats(); // initial
    gpuPollTimer = setInterval(pollGpuStats, 1000);
  }

  function stopGpuPolling() {
    if (gpuPollTimer) {
      clearInterval(gpuPollTimer);
      gpuPollTimer = null;
    }
  }

  function renderHealth(data) {
    healthInfo = data || null;
    if (!data) {
      readinessContent.innerHTML =
        '<div class="gpu-placeholder">Unable to fetch readiness status.</div>';
      readinessStatusDot.className = "gpu-status-dot offline";
      readinessStatusDot.title = "Unknown";
      return;
    }

    if (data.limits && data.limits.max_bytecode_hex_length) {
      maxBytecodeHexLength = data.limits.max_bytecode_hex_length;
      inputEl.maxLength = maxBytecodeHexLength + 2;
      updateCharCount();
    }
    if (data.generation_defaults && !generationControlsInitialized) {
      applyGenerationDefaults(data.generation_defaults);
      generationControlsInitialized = true;
    }

    var ready = !!data.inference_ready;
    readinessStatusDot.className = "gpu-status-dot " + (ready ? "online" : "offline");
    readinessStatusDot.title = ready ? "Inference model loaded" : "Model unavailable";

    var lookup = data.lookup || {};
    var limits = data.limits || {};
    var warmup = data.warmup || {};
    var html = '<div class="gpu-card">';
    html += '<div class="gpu-card-title"><span class="gpu-card-name">' +
      escapeHtml(ready ? "Model ready" : "TAC/lookup-only mode") + "</span></div>";
    html += "<div>Model: " + escapeHtml(
      data.model_path || (data.model_loaded ? "loaded" : "not loaded")
    ) + "</div>";
    if (data.model_error) {
      html += '<div class="gpu-warning">Load error: ' + escapeHtml(data.model_error) + "</div>";
    }
    html += "<div>Warmup: " +
      escapeHtml(warmup.status || (warmup.enabled ? "not started" : "disabled")) + "</div>";
    if (warmup.error) {
      html += '<div class="gpu-warning">Warmup error: ' + escapeHtml(warmup.error) + "</div>";
    }
    html += "<div>Lookup DB: " + escapeHtml(lookup.available ? "available" : "unavailable") + "</div>";
    html += "<div>Timeout: " + escapeHtml(String(limits.timeout_seconds || "disabled")) +
      "s (" + escapeHtml(limits.timeout_enforcement || "cooperative") + ")</div>";
    html += "<div>Bytecode limit: " +
      Number(limits.max_bytecode_hex_length || maxBytecodeHexLength).toLocaleString() +
      " hex chars</div>";
    html += "<div>Generation default: max_new_tokens=" +
      escapeHtml(String((data.generation_defaults || generationDefaults).max_new_tokens)) +
      "</div>";
    html += "</div>";
    readinessContent.innerHTML = html;

    if (ready) {
      btnDecompile.textContent = "⚡ Decompile";
      btnDecompile.title = "Generate Solidity with the loaded model.";
    } else {
      btnDecompile.textContent = "⚡ Analyze TAC / lookup";
      btnDecompile.title =
        "No model is loaded. The request can still analyze bytecode, emit TAC, and use DB lookup hits.";
    }
  }

  function pollHealth() {
    fetch("/api/health", { headers: apiHeaders() })
      .then(function (resp) { return resp.json(); })
      .then(renderHealth)
      .catch(function () { renderHealth(null); });
  }

  // ---- Handle SSE progress events ----

  function handleProgress(data) {
    var pct = data.percent || 0;
    var msg = data.message || "";
    if (data.request_id) activeServerJobId = data.request_id;

    updateProgress(pct, msg);

    // Store selector map when analysis is done
    if (data.selector_map) {
      currentSelectorMap = data.selector_map;
    }

    switch (data.stage) {
      case "readiness":
        progressStage.textContent = data.message || "Checking readiness…";
        if (data.health) renderHealth(data.health);
        break;

      case "analysis":
        progressStage.textContent = "Stage 1: Analyzing bytecode…";
        break;

      case "analysis_done":
        totalFunctionCount = data.num_functions || 0;
        completedFunctionCount = 0;
        progressStage.textContent =
          "Stage 1 complete — " +
          totalFunctionCount +
          " function(s) found";
        show(progressFunctions);
        renderFunctionCount();
        if (data.reconstruction) renderReconstruction(data.reconstruction);
        break;

      case "lookup":
        progressStage.textContent = "Checking TAC hash database…";
        break;

      case "chunking":
        progressStage.textContent = "Stage 1: Building semantic chunks…";
        if (data.reconstruction) renderReconstruction(data.reconstruction);
        break;

      case "function_resolved":
        // Per-function lookup result (exact_match or pending_inference)
        if (data.source === "exact_match") {
          addLogEntry(
            data.current_function, "exact_match", undefined,
            "exact_match", data.confidence
          );
          markFunctionProcessed();
        } else {
          addLogEntry(
            data.current_function, "pending", undefined,
            "pending_inference", null
          );
        }
        break;

      case "lookup_done":
        progressStage.textContent = msg;
        break;

      case "all_lookup":
        progressStage.textContent = "All functions resolved via database lookup!";
        completedFunctionCount = totalFunctionCount;
        renderFunctionCount();
        break;

      case "inference_start":
        progressStage.textContent = "Stage 2: Decompiling via LLM…";
        break;

      case "decompiling":
        progressStage.textContent = "Stage 2: Decompiling via LLM…";
        show(progressFunctions);
        renderFunctionCount();
        // Update existing pending entry to in-progress, or create new
        addLogEntry(data.current_function, "in-progress", undefined,
          "model_inference", null);
        functionStartTime = Date.now();
        break;

      case "batch_start":
        progressStage.textContent =
          "Stage 2: Batch " + (data.batch_index || "?") + "/" +
          (data.total_batches || "?") + " running (" +
          (data.batch_size || 0) + " function(s))";
        show(progressFunctions);
        addBatchLogEntry(data.batch_index || 0, "in-progress", null, data.batch_size || 0);
        break;

      case "batch_done":
        progressStage.textContent =
          "Stage 2: Batch " + (data.batch_index || "?") + "/" +
          (data.total_batches || "?") + " complete in " +
          (data.elapsed_s != null ? Number(data.elapsed_s).toFixed(3) + "s" : "server time");
        addBatchLogEntry(data.batch_index || 0, "completed", data.elapsed_s, data.batch_size || 0);
        break;

      case "function_done":
        var elapsed = data.elapsed_s != null
          ? Math.round(Number(data.elapsed_s) * 1000)
          : (functionStartTime ? Date.now() - functionStartTime : 0);
        var status = data.source === "error" ? "error" : "completed";
        addLogEntry(
          data.current_function, status, elapsed,
          data.source || "model_inference",
          data.confidence != null ? data.confidence : null
        );
        markFunctionProcessed();
        break;

      case "assembling":
        progressStage.textContent = "Stage 3: Reconstructing full contract…";
        break;
    }
  }

  // ---- Decompile action (SSE streaming) ----

  function decompile() {
    if (activeDecompileController) {
      return;
    }

    var validation = validateBytecodeInput(inputEl.value);
    if (!validation.valid) {
      showError(validation.error);
      return;
    }
    var generationConfig;
    var userMetadata;
    try {
      generationConfig = collectGenerationConfig();
      userMetadata = collectUserMetadata();
    } catch (e) {
      showError(e.message);
      return;
    }

    activeServerJobId = newClientJobId();
    var requestBody = {
      bytecode: validation.bytecode,
      generation: generationConfig,
      client_job_id: activeServerJobId
    };
    if (userMetadata.abi) requestBody.abi = userMetadata.abi;
    if (userMetadata.metadata) requestBody.metadata = userMetadata.metadata;

    clearResultRevealTimer();
    setLoading(true);
    currentSelectorMap = null;
    lastDecompileResult = null;
    activeDecompileRequestId += 1;
    var requestId = activeDecompileRequestId;
    activeDecompileController = new AbortController();

    // We use fetch with a ReadableStream to process SSE from a POST request
    // (EventSource only supports GET, so we parse the stream manually)
    fetch("/api/decompile", {
      method: "POST",
      headers: apiHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(requestBody),
      signal: activeDecompileController.signal,
    })
      .then(function (response) {
        if (!isCurrentDecompileRequest(requestId)) return;

        if (!response.ok) {
          return response.json().then(function (data) {
            throw new Error(data.error || "Server error (" + response.status + ")");
          });
        }

        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = "";

        function processChunk() {
          if (!isCurrentDecompileRequest(requestId)) return Promise.resolve();

          return reader.read().then(function (result) {
            if (!isCurrentDecompileRequest(requestId)) return;

            if (result.done) {
              // Process any remaining buffer
              if (buffer.trim()) {
                parseSSEBuffer(buffer, requestId);
              }
              return;
            }

            buffer += decoder.decode(result.value, { stream: true });

            // SSE messages are separated by double newlines
            var parts = buffer.split("\n\n");
            // Keep the last part as it may be incomplete
            buffer = parts.pop() || "";

            for (var i = 0; i < parts.length; i++) {
              parseSSEMessage(parts[i], requestId);
            }

            return processChunk();
          });
        }

        return processChunk();
      })
      .catch(function (err) {
        if (err.name === "AbortError" || !isCurrentDecompileRequest(requestId)) {
          return;
        }
        finishDecompileRequest(requestId);
        showError("Error: " + err.message);
      });
  }

  function parseSSEBuffer(buf, requestId) {
    var messages = buf.split("\n\n");
    for (var i = 0; i < messages.length; i++) {
      if (messages[i].trim()) {
        parseSSEMessage(messages[i], requestId);
      }
    }
  }

  function parseSSEMessage(raw, requestId) {
    if (!isCurrentDecompileRequest(requestId)) return;

    var eventType = "";
    var dataLines = [];

    var lines = raw.split("\n");
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (line.startsWith("event: ")) {
        eventType = line.substring(7).trim();
      } else if (line.startsWith("data: ")) {
        dataLines.push(line.substring(6));
      }
    }

    if (!eventType || dataLines.length === 0) return;

    var data;
    try {
      data = JSON.parse(dataLines.join("\n"));
    } catch (e) {
      console.error("Failed to parse SSE data:", e, dataLines);
      return;
    }

    switch (eventType) {
      case "progress":
        handleProgress(data);
        break;

      case "result":
        handleResult(data, requestId);
        break;

      case "error":
        finishDecompileRequest(requestId);
        showError(
          (data.error || "Unknown server error") +
            (data.trace_path ? " Trace: " + data.trace_path : "")
        );
        break;
    }
  }

  function handleResult(data, requestId) {
    if (!isCurrentDecompileRequest(requestId)) return;
    lastDecompileResult = data;
    setDownloadButtonsEnabled(true);

    // Populate results
    tacOutput.textContent = data.tac || "(no TAC output)";
    solidityOutput.textContent = data.solidity || "(no Solidity output)";

    // Model / partial-failure warning
    if (data.model_error || data.partial_success || data.success === false) {
      var warning = data.model_error || "";
      if (data.source_summary && data.source_summary.error) {
        warning +=
          (warning ? " " : "") +
          data.source_summary.error +
          " function(s) failed; see Function Mapping for details.";
      } else if (data.success === false) {
        warning += (warning ? " " : "") + "Decompilation did not fully succeed.";
      }
      if (data.validation && !data.validation.valid) {
        warning +=
          (warning ? " " : "") +
          "Generated Solidity failed validation (" +
          (data.validation.method || "unknown") +
          ").";
      }
      if (data.trace_path) {
        warning += (warning ? " " : "") + "Trace: " + data.trace_path;
      }
      modelWarningText.textContent = warning;
      show(modelWarning);
    } else {
      hide(modelWarning);
    }

    // Analysis
    if (data.analysis) {
      renderAnalysis(data.analysis);
    }
    renderTraceDiagnostics(data);
    renderReconstruction(data.reconstruction);

    // Function mapping
    var selMap = data.selector_map || currentSelectorMap;
    if (selMap || data.function_results) {
      renderFunctionMapping(selMap, data.function_results || []);
    }

    // Final progress update
    updateProgress(100, "Decompilation complete!");

    // Short delay so user sees 100% before switching to results
    clearResultRevealTimer();
    resultRevealTimer = setTimeout(function () {
      if (!isCurrentDecompileRequest(requestId)) return;
      resultRevealTimer = null;
      finishDecompileRequest(requestId);
      show(resultsSection);
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 600);
  }

  // ---- Security analysis actions ----

  function renderSecurityJson(title, data) {
    var html = '<div class="analysis-card"><div class="label">' +
      escapeHtml(title) + '</div><div class="value small">';
    html += "<pre>" + escapeHtml(JSON.stringify(data, null, 2)) + "</pre>";
    html += "</div></div>";
    securityContent.innerHTML = html;
  }

  function renderVulnerabilityResult(data) {
    var html = "";
    html += '<div class="analysis-card"><div class="label">Risk Score</div><div class="value">' +
      escapeHtml(String(data.risk_score != null ? data.risk_score : "—")) + "</div></div>";
    html += '<div class="analysis-card"><div class="label">Summary</div><div class="value small">' +
      escapeHtml(data.summary || "No vulnerabilities reported.") + "</div></div>";
    var vulns = data.vulnerabilities || [];
    if (vulns.length) {
      html += '<div class="analysis-card"><div class="label">Findings</div><div class="value small">';
      for (var i = 0; i < vulns.length; i++) {
        var v = vulns[i];
        html += "<p><strong>" + escapeHtml(v.severity || "severity") + "</strong> " +
          escapeHtml(v.type || "finding") + " (" +
          escapeHtml(String(v.confidence != null ? v.confidence : "n/a")) +
          " confidence)<br>" +
          escapeHtml(v.explanation || "") +
          "<br><em>" + escapeHtml(v.recommendation || "") + "</em></p>";
      }
      html += "</div></div>";
    }
    securityContent.innerHTML = html;
  }

  function renderClassificationResult(data) {
    var html = "";
    html += '<div class="analysis-card"><div class="label">Classification</div><div class="value">' +
      (data.is_malicious ? "Malicious" : "Legitimate / inconclusive") + "</div></div>";
    html += '<div class="analysis-card"><div class="label">Confidence</div><div class="value">' +
      escapeHtml(String(data.confidence != null ? data.confidence : "—")) + "</div></div>";
    html += '<div class="analysis-card"><div class="label">Explanation</div><div class="value small">' +
      escapeHtml(data.explanation || "") + "</div></div>";
    securityContent.innerHTML = html;
  }

  function runSecurityEndpoint(kind, endpoint) {
    var validation = validateBytecodeInput(inputEl.value);
    if (!validation.valid) {
      showError(validation.error);
      return;
    }
    hideError();
    show(resultsSection);
    activateTab("tab-security");
    securityContent.innerHTML =
      '<div class="gpu-placeholder">Running ' + escapeHtml(kind) + "…</div>";
    fetch(endpoint, {
      method: "POST",
      headers: apiHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ bytecode: validation.bytecode })
    })
      .then(function (resp) {
        return resp.json().then(function (data) {
          if (!resp.ok) throw new Error(data.error || "Server error (" + resp.status + ")");
          return data;
        });
      })
      .then(function (data) {
        if (kind === "vulnerability scan") renderVulnerabilityResult(data);
        else if (kind === "classification") renderClassificationResult(data);
        else renderSecurityJson("Audit Report", data.report || data);
      })
      .catch(function (err) {
        securityContent.innerHTML =
          '<div class="gpu-warning">' + escapeHtml(err.message) + "</div>";
      });
  }

  // ---- Event listeners ----

  btnDecompile.addEventListener("click", decompile);

  if (btnCancelDecompile) {
    btnCancelDecompile.addEventListener("click", function () {
      abortActiveDecompile();
      showError("Decompilation cancelled.");
    });
  }

  if (btnResetGeneration) {
    btnResetGeneration.addEventListener("click", function () {
      applyGenerationDefaults(healthInfo ? healthInfo.generation_defaults : generationDefaults);
    });
  }

  if (btnVulnerabilityScan) {
    btnVulnerabilityScan.addEventListener("click", function () {
      runSecurityEndpoint("vulnerability scan", "/api/vulnerability-scan");
    });
  }
  if (btnClassify) {
    btnClassify.addEventListener("click", function () {
      runSecurityEndpoint("classification", "/api/classify");
    });
  }
  if (btnAuditReport) {
    btnAuditReport.addEventListener("click", function () {
      runSecurityEndpoint("audit report", "/api/audit-report");
    });
  }

  btnSample.addEventListener("click", function () {
    abortActiveDecompile();
    inputEl.value = SAMPLE_BYTECODE;
    updateCharCount();
    inputEl.focus();
  });

  btnClear.addEventListener("click", function () {
    abortActiveDecompile();
    inputEl.value = "";
    if (abiInput) abiInput.value = "";
    if (metadataInput) metadataInput.value = "";
    updateCharCount();
    hide(resultsSection);
    hideError();
    lastDecompileResult = null;
    setDownloadButtonsEnabled(false);
    inputEl.focus();
  });

  // Cancel active decompilation if the bytecode changes.
  inputEl.addEventListener("input", function () {
    if (activeDecompileController) {
      abortActiveDecompile();
    }
  });

  [abiInput, metadataInput].forEach(function (el) {
    if (!el) return;
    el.addEventListener("input", function () {
      if (activeDecompileController) {
        abortActiveDecompile();
      }
    });
  });

  if (abiFileInput) {
    abiFileInput.addEventListener("change", function () {
      loadJsonFileIntoTextarea(abiFileInput, abiInput, "ABI");
    });
  }
  if (metadataFileInput) {
    metadataFileInput.addEventListener("change", function () {
      loadJsonFileIntoTextarea(metadataFileInput, metadataInput, "metadata");
    });
  }

  btnDismissError.addEventListener("click", hideError);

  if (apiKeyInput) {
    apiKeyInput.addEventListener("change", function () {
      pollGpuStats();
      pollHealth();
    });
  }

  inputEl.addEventListener("input", updateCharCount);

  // Allow Ctrl+Enter to decompile
  inputEl.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (activeDecompileController) return;
      decompile();
    }
  });

  // Initial char count
  updateCharCount();
  applyGenerationDefaults(generationDefaults);
  pollHealth();

  // Start GPU polling
  startGpuPolling();
})();