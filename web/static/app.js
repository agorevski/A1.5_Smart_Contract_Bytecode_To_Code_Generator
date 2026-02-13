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
  }

  function setLoading(on) {
    if (on) {
      resetProgress();
      show(loadingEl);
      hide(resultsSection);
      hideError();
      btnDecompile.disabled = true;
    } else {
      hide(loadingEl);
      btnDecompile.disabled = false;
    }
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

  function addLogEntry(fname, status, elapsed) {
    show(progressLog);

    // Check if entry already exists (update it)
    var existing = document.getElementById("log-" + fname);
    if (existing) {
      existing.className = status;
      var icon = existing.querySelector(".log-icon");
      if (status === "completed") {
        icon.textContent = "✓";
      } else if (status === "error") {
        icon.textContent = "✗";
      }
      if (elapsed !== undefined) {
        var timeEl = existing.querySelector(".log-time");
        if (timeEl) timeEl.textContent = formatElapsed(elapsed);
      }
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
    }

    var nameSpan = document.createElement("span");
    nameSpan.className = "log-name";
    nameSpan.textContent = fname;

    li.appendChild(iconSpan);
    li.appendChild(nameSpan);

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
    fetch("/api/gpu-stats")
      .then(function (resp) {
        return resp.json();
      })
      .then(function (data) {
        renderGpuStats(data);
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
        progressStage.textContent =
          "Stage 1 complete — " +
          (data.num_functions || 0) +
          " function(s) found";
        show(progressFunctions);
        progressFuncCount.innerHTML =
          '<span class="func-count-highlight">0</span> / ' +
          (data.num_functions || 0) +
          " decompiled";
        break;

      case "decompiling":
        progressStage.textContent = "Stage 2: Decompiling via LLM…";
        show(progressFunctions);
        progressFuncCount.innerHTML =
          '<span class="func-count-highlight">' +
          ((data.current_index || 1) - 1) +
          "</span> / " +
          (data.total_functions || 0) +
          " decompiled";
        addLogEntry(data.current_function, "in-progress");
        functionStartTime = Date.now();
        break;

      case "function_done":
        var elapsed = functionStartTime ? Date.now() - functionStartTime : 0;
        addLogEntry(data.current_function, "completed", elapsed);
        progressFuncCount.innerHTML =
          '<span class="func-count-highlight">' +
          (data.current_index || 0) +
          "</span> / " +
          (data.total_functions || 0) +
          " decompiled";
        break;

      case "assembling":
        progressStage.textContent = "Assembling final contract…";
        break;
    }
  }

  // ---- Decompile action (SSE streaming) ----

  function decompile() {
    var bytecode = inputEl.value.trim();
    if (!bytecode) {
      showError("Please enter EVM bytecode to decompile.");
      return;
    }

    setLoading(true);
    currentSelectorMap = null;

    // We use fetch with a ReadableStream to process SSE from a POST request
    // (EventSource only supports GET, so we parse the stream manually)
    fetch("/api/decompile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bytecode: bytecode }),
    })
      .then(function (response) {
        if (!response.ok) {
          return response.json().then(function (data) {
            throw new Error(data.error || "Server error (" + response.status + ")");
          });
        }

        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = "";

        function processChunk() {
          return reader.read().then(function (result) {
            if (result.done) {
              // Process any remaining buffer
              if (buffer.trim()) {
                parseSSEBuffer(buffer);
              }
              return;
            }

            buffer += decoder.decode(result.value, { stream: true });

            // SSE messages are separated by double newlines
            var parts = buffer.split("\n\n");
            // Keep the last part as it may be incomplete
            buffer = parts.pop() || "";

            for (var i = 0; i < parts.length; i++) {
              parseSSEMessage(parts[i]);
            }

            return processChunk();
          });
        }

        return processChunk();
      })
      .catch(function (err) {
        setLoading(false);
        showError("Error: " + err.message);
      });
  }

  function parseSSEBuffer(buf) {
    var messages = buf.split("\n\n");
    for (var i = 0; i < messages.length; i++) {
      if (messages[i].trim()) {
        parseSSEMessage(messages[i]);
      }
    }
  }

  function parseSSEMessage(raw) {
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
        handleResult(data);
        break;

      case "error":
        setLoading(false);
        showError(data.error || "Unknown server error");
        break;
    }
  }

  function handleResult(data) {
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
    setTimeout(function () {
      setLoading(false);
      show(resultsSection);
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 600);
  }

  // ---- Event listeners ----

  btnDecompile.addEventListener("click", decompile);

  btnSample.addEventListener("click", function () {
    inputEl.value = SAMPLE_BYTECODE;
    isSampleLoaded = true;
    updateCharCount();
    updateOriginalTab();
    inputEl.focus();
  });

  btnClear.addEventListener("click", function () {
    inputEl.value = "";
    isSampleLoaded = false;
    updateCharCount();
    updateOriginalTab();
    hide(resultsSection);
    hideError();
    inputEl.focus();
  });

  // Detect if user modifies the input away from sample
  inputEl.addEventListener("input", function () {
    if (isSampleLoaded && inputEl.value.trim() !== SAMPLE_BYTECODE) {
      isSampleLoaded = false;
      updateOriginalTab();
    }
  });

  btnDismissError.addEventListener("click", hideError);

  inputEl.addEventListener("input", updateCharCount);

  // Allow Ctrl+Enter to decompile
  inputEl.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      decompile();
    }
  });

  // Initial char count
  updateCharCount();

  // Start GPU polling
  startGpuPolling();
})();