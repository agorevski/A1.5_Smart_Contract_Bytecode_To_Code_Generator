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
  const compilerVersionInput = document.getElementById("compiler-version-input");
  const optimizerEnabledInput = document.getElementById("optimizer-enabled-input");
  const optimizerRunsInput = document.getElementById("optimizer-runs-input");
  const loadingEl = document.getElementById("loading");
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

  // ---- Sample bytecode (Owner contract compiled with solc 0.8.24, runtime bytecode) ----
  const SAMPLE_BYTECODE =
    "0x608060405234801561000f575f80fd5b5060043610610034575f3560e01c8063893d20e814610038578063a6f9dae114610056575b5f80fd5b610040610072565b60405161004d919061014e565b60405180910390f35b610070600480360381019061006b9190610195565b610099565b005b5f805f9054906101000a900473ffffffffffffffffffffffffffffffffffffffff16905090565b8073ffffffffffffffffffffffffffffffffffffffff163373ffffffffffffffffffffffffffffffffffffffff160361010c57805f806101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055505b50565b5f73ffffffffffffffffffffffffffffffffffffffff82169050919050565b5f6101388261010f565b9050919050565b6101488161012e565b82525050565b5f6020820190506101615f83018461013f565b92915050565b5f80fd5b6101748161012e565b811461017e575f80fd5b50565b5f8135905061018f8161016b565b92915050565b5f602082840312156101aa576101a9610167565b5b5f6101b784828501610181565b9150509291505056fea264697066735822122020389b4014c2d8511dc28898fda8ea80a2215a6e55690e4770b83b23ce7e209364736f6c63430008180033";

  // Original Solidity source that was compiled into SAMPLE_BYTECODE
  const SAMPLE_ORIGINAL_SOLIDITY =
    "// SPDX-License-Identifier: GPL-3.0\n" +
    "pragma solidity >=0.7.0 <0.9.0;\n" +
    "\n" +
    "/**\n" +
    " * @title Owner\n" +
    " * @dev Set & change owner\n" +
    " */\n" +
    "contract Owner {\n" +
    "\n" +
    "    address private owner;\n" +
    "\n" +
    "    /**\n" +
    "     * @dev Set contract deployer as owner\n" +
    "     */\n" +
    "    constructor() {\n" +
    "        owner = msg.sender;\n" +
    "    }\n" +
    "\n" +
    "    /**\n" +
    "     * @dev Return owner address\n" +
    "     * @return address of owner\n" +
    "     */\n" +
    "    function getOwner() external view returns (address) {\n" +
    "        return owner;\n" +
    "    }\n" +
    "\n" +
    "    /**\n" +
    "     * @dev Change owner\n" +
    "     * @param newOwner address of new owner\n" +
    "     */\n" +
    "    function changeOwner(address newOwner) public {\n" +
    "        if (msg.sender == newOwner) {\n" +
    "            owner = newOwner;\n" +
    "        }\n" +
    "    }\n" +
    "}\n";

  // Track whether sample bytecode is loaded
  var isSampleLoaded = false;

  // ---- State tracking ----
  var functionStartTime = 0;
  var gpuPollTimer = null;
  var currentSelectorMap = null; // Stored from last decompilation
  var totalFunctionCount = 0;
  var completedFunctionCount = 0;
  var activeDecompileRequestId = 0;
  var activeDecompileController = null;
  var resultRevealTimer = null;

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
      btnDecompile.disabled = true;
      btnSample.disabled = true;
      inputEl.readOnly = true;
    } else {
      hide(loadingEl);
      btnDecompile.disabled = false;
      btnSample.disabled = false;
      inputEl.readOnly = false;
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
    setLoading(false);
  }

  function abortActiveDecompile() {
    clearResultRevealTimer();
    if (activeDecompileController) {
      activeDecompileController.abort();
      activeDecompileController = null;
      activeDecompileRequestId += 1;
    }
    setLoading(false);
  }

  function updateOriginalTab() {
    var tabBtn = document.getElementById("tab-btn-original");
    var tabPanel = document.getElementById("tab-original");
    var originalOutput = document.getElementById("original-output");

    if (isSampleLoaded) {
      show(tabBtn);
      originalOutput.textContent = SAMPLE_ORIGINAL_SOLIDITY;
    } else {
      hide(tabBtn);
      if (tabPanel) {
        tabPanel.classList.add("hidden");
        tabPanel.classList.remove("active");
      }
      // If original tab was active, switch to solidity tab
      if (tabBtn && tabBtn.classList.contains("active")) {
        tabBtn.classList.remove("active");
        var solTab = document.querySelector('[data-tab="tab-solidity"]');
        if (solTab) solTab.click();
      }
    }
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

  function updateCharCount() {
    const len = inputEl.value.trim().length;
    charCountEl.textContent = len.toLocaleString() + " characters";
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
      case "model_inference": return "Generated by Llama 3.2 3B model inference";
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

  document.querySelectorAll(".tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      const targetId = this.getAttribute("data-tab");

      document.querySelectorAll(".tab").forEach(function (t) {
        t.classList.remove("active");
      });
      document.querySelectorAll(".tab-panel").forEach(function (p) {
        p.classList.add("hidden");
        p.classList.remove("active");
      });

      this.classList.add("active");
      const panel = document.getElementById(targetId);
      if (panel) {
        panel.classList.remove("hidden");
        panel.classList.add("active");
      }
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

  function renderFunctionMapping(selectorMap) {
    if (!selectorMap || Object.keys(selectorMap).length === 0) {
      functionMappingContent.innerHTML =
        '<p class="text-muted">No function selectors detected.</p>';
      return;
    }

    var html = '<table class="fn-map-table">';
    html +=
      "<thead><tr>" +
      "<th>Bytecode Function</th>" +
      "<th>→</th>" +
      "<th>Resolved Name</th>" +
      "<th>Confidence</th>" +
      "<th>Source</th>" +
      "</tr></thead><tbody>";

    var fnames = Object.keys(selectorMap);
    for (var i = 0; i < fnames.length; i++) {
      var fname = fnames[i];
      var info = selectorMap[fname];
      var best = info.best_match;
      if (!best) continue;

      var conf = best.confidence; // 0-100
      var confClass = "conf-high";
      if (conf < 50) confClass = "conf-low";
      else if (conf < 80) confClass = "conf-med";

      var sourceLabel = best.source === "builtin" ? "Known Standard" :
                        best.source === "4byte" ? "4byte.directory" : "Unknown";

      html += "<tr>";
      html +=
        '<td class="fn-map-fname"><code>' +
        escapeHtml(fname) +
        "</code></td>";
      html += '<td class="fn-map-arrow">→</td>';
      html +=
        '<td class="fn-map-sig"><code>' +
        escapeHtml(best.signature) +
        "</code></td>";
      html +=
        '<td><span class="conf-badge ' +
        confClass +
        '" title="' +
        conf.toFixed(1) +
        '% confidence">' +
        conf.toFixed(0) +
        "%</span></td>";
      html +=
        '<td class="fn-map-source">' +
        escapeHtml(sourceLabel) +
        "</td>";
      html += "</tr>";

      // Additional candidates (if any beyond the best)
      if (info.candidates && info.candidates.length > 1) {
        for (var j = 0; j < info.candidates.length; j++) {
          var cand = info.candidates[j];
          if (cand.signature === best.signature) continue;
          var cConf = cand.confidence;
          var cClass = "conf-high";
          if (cConf < 50) cClass = "conf-low";
          else if (cConf < 80) cClass = "conf-med";

          var cSource = cand.source === "builtin" ? "Known Standard" :
                        cand.source === "4byte" ? "4byte.directory" : "Unknown";

          html += '<tr class="fn-map-alt">';
          html += "<td></td>";
          html += '<td class="fn-map-arrow">↳</td>';
          html +=
            '<td class="fn-map-sig"><code>' +
            escapeHtml(cand.signature) +
            "</code></td>";
          html +=
            '<td><span class="conf-badge ' +
            cClass +
            '">' +
            cConf.toFixed(0) +
            "%</span></td>";
          html +=
            '<td class="fn-map-source">' +
            escapeHtml(cSource) +
            "</td>";
          html += "</tr>";
        }
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

  // ---- Handle SSE progress events ----

  function handleProgress(data) {
    var pct = data.percent || 0;
    var msg = data.message || "";

    updateProgress(pct, msg);

    // Store selector map when analysis is done
    if (data.selector_map) {
      currentSelectorMap = data.selector_map;
    }

    switch (data.stage) {
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
        break;

      case "lookup":
        progressStage.textContent = "Checking TAC hash database…";
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

      case "function_done":
        var elapsed = functionStartTime ? Date.now() - functionStartTime : 0;
        var status = data.source === "error" ? "error" : "completed";
        addLogEntry(
          data.current_function, status, elapsed,
          data.source || "model_inference",
          data.confidence != null ? data.confidence : null
        );
        markFunctionProcessed();
        break;

      case "assembling":
        progressStage.textContent = "Assembling final contract…";
        break;
    }
  }

  // ---- Decompile action (SSE streaming) ----

  function decompile() {
    if (activeDecompileController) {
      return;
    }

    var bytecode = inputEl.value.trim();
    if (!bytecode) {
      showError("Please enter EVM bytecode to decompile.");
      return;
    }

    var requestBody = { bytecode: bytecode };
    var compilerVersion = compilerVersionInput
      ? compilerVersionInput.value.trim()
      : "";
    var optimizerEnabled = optimizerEnabledInput
      ? optimizerEnabledInput.value
      : "";
    var optimizerRuns = optimizerRunsInput
      ? optimizerRunsInput.value.trim()
      : "";
    if (compilerVersion) {
      requestBody.compiler_version = compilerVersion;
    }
    if (optimizerEnabled) {
      requestBody.optimizer_enabled = optimizerEnabled === "true";
    }
    if (optimizerRuns) {
      requestBody.optimizer_runs = optimizerRuns;
    }

    clearResultRevealTimer();
    setLoading(true);
    currentSelectorMap = null;
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
        showError(data.error || "Unknown server error");
        break;
    }
  }

  function handleResult(data, requestId) {
    if (!isCurrentDecompileRequest(requestId)) return;

    // Populate results
    tacOutput.textContent = data.tac || "(no TAC output)";
    solidityOutput.textContent = data.solidity || "(no Solidity output)";

    // Model warning
    if (data.model_error) {
      modelWarningText.textContent = data.model_error;
      show(modelWarning);
    } else {
      hide(modelWarning);
    }

    // Analysis
    if (data.analysis) {
      renderAnalysis(data.analysis);
    }

    // Function mapping
    var selMap = data.selector_map || currentSelectorMap;
    if (selMap) {
      renderFunctionMapping(selMap);
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

  // ---- Event listeners ----

  btnDecompile.addEventListener("click", decompile);

  btnSample.addEventListener("click", function () {
    abortActiveDecompile();
    inputEl.value = SAMPLE_BYTECODE;
    if (compilerVersionInput) compilerVersionInput.value = "0.8.24";
    if (optimizerEnabledInput) optimizerEnabledInput.value = "";
    if (optimizerRunsInput) optimizerRunsInput.value = "";
    isSampleLoaded = true;
    updateCharCount();
    updateOriginalTab();
    inputEl.focus();
  });

  btnClear.addEventListener("click", function () {
    abortActiveDecompile();
    inputEl.value = "";
    if (compilerVersionInput) compilerVersionInput.value = "";
    if (optimizerEnabledInput) optimizerEnabledInput.value = "";
    if (optimizerRunsInput) optimizerRunsInput.value = "";
    isSampleLoaded = false;
    updateCharCount();
    updateOriginalTab();
    hide(resultsSection);
    hideError();
    inputEl.focus();
  });

  // Detect if user modifies the input away from sample
  inputEl.addEventListener("input", function () {
    if (activeDecompileController) {
      abortActiveDecompile();
    }

    if (isSampleLoaded && inputEl.value.trim() !== SAMPLE_BYTECODE) {
      isSampleLoaded = false;
      updateOriginalTab();
    }
  });

  btnDismissError.addEventListener("click", hideError);

  if (apiKeyInput) {
    apiKeyInput.addEventListener("change", pollGpuStats);
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

  // Start GPU polling
  startGpuPolling();
})();