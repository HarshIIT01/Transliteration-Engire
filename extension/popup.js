/**
 * Transliteration Engine — Chrome Extension
 * ==========================================
 * Handles:
 *   1. Server health check (green/red dot)
 *   2. Transliteration API calls
 *   3. Copy to clipboard
 *   4. Translation history (persisted in chrome.storage.local)
 *   5. Keyboard shortcuts (Ctrl+Enter)
 */

const API_BASE = "https://transliteration-engine.onrender.com";
const MAX_HISTORY = 10;

// ==========================================
// DOM ELEMENTS
// ==========================================
const inputText = document.getElementById("input-text");
const transliterateBtn = document.getElementById("transliterate-btn");
const btnText = transliterateBtn.querySelector(".btn-text");
const btnLoader = transliterateBtn.querySelector(".btn-loader");
const outputSection = document.getElementById("output-section");
const outputText = document.getElementById("output-text");
const latencyEl = document.getElementById("latency");
const copyBtn = document.getElementById("copy-btn");
const copyLabel = document.getElementById("copy-label");
const clearBtn = document.getElementById("clear-btn");
const charCount = document.getElementById("char-count");
const statusDot = document.getElementById("status-dot");
const historySection = document.getElementById("history-section");
const historyList = document.getElementById("history-list");
const clearHistoryBtn = document.getElementById("clear-history");

let serverOnline = false;

// ==========================================
// SERVER HEALTH CHECK
// ==========================================
async function checkServer() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    const data = await res.json();
    serverOnline = data.status === "ok";
  } catch {
    serverOnline = false;
  }

  statusDot.className = `status-dot ${serverOnline ? "online" : "offline"}`;
  statusDot.title = serverOnline ? "Server online" : "Server offline — start with: python -m uvicorn app.main:app";
  updateButtonState();
}

// ==========================================
// TRANSLITERATION
// ==========================================
async function transliterate() {
  const text = inputText.value.trim();
  if (!text || !serverOnline) return;

  // Show loading
  btnText.hidden = true;
  btnLoader.hidden = false;
  transliterateBtn.disabled = true;

  try {
    const startTime = performance.now();

    const res = await fetch(`${API_BASE}/transliterate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();
    const clientTime = (performance.now() - startTime).toFixed(1);

    // Show output
    outputSection.hidden = false;
    outputText.textContent = data.result;
    latencyEl.textContent = `Server: ${data.latency_ms}ms · Round-trip: ${clientTime}ms`;

    // Save to history
    addToHistory(text, data.result, data.latency_ms);

  } catch (err) {
    outputSection.hidden = false;
    outputText.textContent = "⚠ Error connecting to server";
    outputText.style.color = "#ef4444";
    latencyEl.textContent = "";
    console.error("Transliteration error:", err);
  } finally {
    btnText.hidden = false;
    btnLoader.hidden = true;
    updateButtonState();
  }
}

// ==========================================
// HISTORY
// ==========================================
function addToHistory(roman, devanagari, latency) {
  // Use chrome.storage if available, fallback to localStorage
  const storage = typeof chrome !== "undefined" && chrome.storage
    ? chrome.storage.local
    : null;

  const entry = { roman, devanagari, latency, time: Date.now() };

  if (storage) {
    storage.get("history", (result) => {
      const history = result.history || [];
      history.unshift(entry);
      if (history.length > MAX_HISTORY) history.pop();
      storage.set({ history }, renderHistory);
    });
  } else {
    // Fallback for testing outside Chrome
    const history = JSON.parse(localStorage.getItem("translit_history") || "[]");
    history.unshift(entry);
    if (history.length > MAX_HISTORY) history.pop();
    localStorage.setItem("translit_history", JSON.stringify(history));
    renderHistory();
  }
}

function loadHistory(callback) {
  const storage = typeof chrome !== "undefined" && chrome.storage
    ? chrome.storage.local
    : null;

  if (storage) {
    storage.get("history", (result) => callback(result.history || []));
  } else {
    callback(JSON.parse(localStorage.getItem("translit_history") || "[]"));
  }
}

function renderHistory() {
  loadHistory((history) => {
    if (history.length === 0) {
      historySection.hidden = true;
      return;
    }

    historySection.hidden = false;
    historyList.innerHTML = "";

    history.forEach((item) => {
      const div = document.createElement("div");
      div.className = "history-item";
      div.innerHTML = `
        <span class="roman">${escapeHtml(item.roman)}</span>
        <span class="devanagari">${escapeHtml(item.devanagari)}</span>
      `;
      div.addEventListener("click", () => {
        inputText.value = item.roman;
        outputSection.hidden = false;
        outputText.textContent = item.devanagari;
        outputText.style.color = "";
        latencyEl.textContent = `From history · Server: ${item.latency}ms`;
        updateCharCount();
      });
      historyList.appendChild(div);
    });
  });
}

function clearHistory() {
  const storage = typeof chrome !== "undefined" && chrome.storage
    ? chrome.storage.local
    : null;

  if (storage) {
    storage.set({ history: [] }, renderHistory);
  } else {
    localStorage.setItem("translit_history", "[]");
    renderHistory();
  }
}

// ==========================================
// CLIPBOARD
// ==========================================
async function copyToClipboard() {
  const text = outputText.textContent;
  if (!text) return;

  try {
    await navigator.clipboard.writeText(text);
    copyLabel.textContent = "✓ Copied!";
    copyBtn.classList.add("copied");
    setTimeout(() => {
      copyLabel.textContent = "📋 Copy";
      copyBtn.classList.remove("copied");
    }, 1500);
  } catch {
    // Fallback
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    copyLabel.textContent = "✓ Copied!";
    setTimeout(() => { copyLabel.textContent = "📋 Copy"; }, 1500);
  }
}

// ==========================================
// UI HELPERS
// ==========================================
function updateButtonState() {
  const hasText = inputText.value.trim().length > 0;
  transliterateBtn.disabled = !hasText || !serverOnline;
}

function updateCharCount() {
  const len = inputText.value.length;
  charCount.textContent = `${len} char${len !== 1 ? "s" : ""}`;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ==========================================
// EVENT LISTENERS
// ==========================================
inputText.addEventListener("input", () => {
  updateCharCount();
  updateButtonState();
  // Reset output color if it was error
  outputText.style.color = "";
});

inputText.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    transliterate();
  }
});

transliterateBtn.addEventListener("click", transliterate);
copyBtn.addEventListener("click", copyToClipboard);
clearBtn.addEventListener("click", () => {
  inputText.value = "";
  outputSection.hidden = true;
  updateCharCount();
  updateButtonState();
  inputText.focus();
});
clearHistoryBtn.addEventListener("click", clearHistory);

// ==========================================
// INITIALIZATION
// ==========================================
checkServer();
setInterval(checkServer, 10000);  // Re-check every 10s
renderHistory();
inputText.focus();
