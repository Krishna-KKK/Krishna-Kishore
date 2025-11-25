// =======================
//  app.js — Enhanced Chat + Auto Mode Renderer
// =======================
const API_BASE = "http://localhost:8500";

const chatWindow = document.getElementById("chat-window");
const sendBtn = document.getElementById("send-btn");
const sourceInput = document.getElementById("source-file");
const targetInput = document.getElementById("target-file");
const queryInput = document.getElementById("user-query");
const downloadExcelBtn = document.getElementById("downloadExcelBtn");
const downloadSqlBtn = document.getElementById("downloadSqlBtn");
const downloadStatus = document.getElementById("downloadStatus");
const statusEl = document.getElementById("status");
const modeToggleEls = document.getElementsByName("mode");

function getSelectedModeValue() {
  for (const r of modeToggleEls) if (r.checked) return r.value;
  return "chat";
}

function appendMessage(message, sender = "bot") {
  const wrapper = document.createElement("div");
  wrapper.className = "msg " + (sender === "user" ? "user" : "bot");
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (sender === "bot") {
    bubble.innerHTML = typeof message === "string" ? marked.parse(message) : marked.parse(JSON.stringify(message));
    bubble.querySelectorAll("pre code").forEach(block => hljs.highlightElement(block));
  } else {
    bubble.textContent = message;
  }

  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function setStatus(text) { statusEl.textContent = text || ""; }

// =======================
//  Helper Renderers
// =======================
function renderDatasetAnalysis(arr, title) {
  if (!Array.isArray(arr) || !arr.length) return "";
  let html = `<h6>${title}</h6><table class="table table-sm table-bordered"><thead><tr>
      <th>Column</th><th>Type</th><th>Nulls</th><th>% Null</th><th>Unique</th><th>Samples</th><th>Numeric Summary</th>
  </tr></thead><tbody>`;
  for (const c of arr) {
    const s = (c.Samples || []).slice(0, 5).join(", ");
    html += `<tr><td>${c.Column}</td><td>${c.Type}</td><td>${c.Nulls}</td><td>${c["% Null"]}</td><td>${c.Unique}</td><td>${s}</td><td>${c["Numeric Summary"]}</td></tr>`;
  }
  html += "</tbody></table>";
  return html;
}

function renderPreview(expected) {
  if (!expected || !expected.preview || !expected.preview.length) return "<p>No preview available.</p>";
  const rows = expected.preview;
  const headers = Object.keys(rows[0]);
  let html = "<table class='table table-sm table-bordered'><thead><tr>";
  headers.forEach(h => html += `<th>${h}</th>`);
  html += "</tr></thead><tbody>";
  rows.forEach(r => {
    html += "<tr>";
    headers.forEach(h => html += `<td>${r[h] ?? "-"}</td>`);
    html += "</tr>";
  });
  html += "</tbody></table>";
  return html;
}

// =======================
//  Chat Mode (Enhanced)
// =======================
async function restChatSend(text, sourceFile, targetFile) {
  try {
    setStatus("Processing...");
    const form = new FormData();
    form.append("message", text || "Generate transformation rules.");
    if (sourceFile) form.append("source_file", sourceFile);
    if (targetFile) form.append("target_file", targetFile);

    const res = await fetch(`${API_BASE}/chat`, { method: "POST", body: form });
    const data = await res.json();
    setStatus("");

    if (data.status === "error") {
      appendMessage("❌ " + data.message, "bot");
      return;
    }

    appendMessage("✅ Data Transformation Analysis Completed.", "bot");

    // 🧩 Full detailed sections — identical to Auto Mode
    appendMessage(renderDatasetAnalysis(data.source_analysis, "📊 Source Dataset Analysis"), "bot");
    appendMessage(renderDatasetAnalysis(data.target_analysis, "📊 Target Dataset Analysis"), "bot");

    appendMessage("📋 **Transformation Rules:**", "bot");
    appendMessage(data.mapping_rules || "No mapping rules found.", "bot");

    appendMessage("🧩 **SQL Transformation Query:**", "bot");
    appendMessage("```sql\n" + (data.sql_transformation_query || "-") + "\n```", "bot");

    appendMessage("🐍 **Python Example:**", "bot");
    appendMessage("```python\n" + (data.python_code_snippet || "-") + "\n```", "bot");

    appendMessage("✅ **Expected Result (Preview):**", "bot");
    appendMessage(renderPreview(data.expected_results), "bot");
  } catch (err) {
    appendMessage("❌ Error: " + err.message, "bot");
  }
}

// =======================
//  Auto Mode (ETL)
// =======================
async function analyzeSend(sourceFile, targetFile, query, etlMode) {
  setStatus("Processing...");
  const form = new FormData();
  form.append("source_file", sourceFile);
  if (targetFile) form.append("target_file", targetFile);
  form.append("mode", etlMode);
  if (query) form.append("query", query);

  try {
    const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });
    const data = await res.json();
    setStatus("");

    if (data.status === "error") {
      appendMessage("❌ " + data.message, "bot");
      return;
    }

    appendMessage("✅ Data Transformation Analysis Completed.", "bot");
    appendMessage(renderDatasetAnalysis(data.source_analysis, "📊 Source Dataset Analysis"), "bot");
    appendMessage(renderDatasetAnalysis(data.target_analysis, "📊 Target Dataset Analysis"), "bot");
    appendMessage("📋 **Transformation Rules:**", "bot");
    appendMessage(data.mapping_rules, "bot");
    appendMessage("🧩 **SQL Transformation Query:**", "bot");
    appendMessage("```sql\n" + data.sql_transformation_query + "\n```", "bot");
    appendMessage("🐍 **Python Example:**", "bot");
    appendMessage("```python\n" + data.python_code_snippet + "\n```", "bot");
    appendMessage("✅ **Expected Result (Preview):**", "bot");
    appendMessage(renderPreview(data.expected_results), "bot");
  } catch (err) {
    appendMessage("❌ Error: " + err.message, "bot");
  }
}

// =======================
//  Event Listener
// =======================
sendBtn.addEventListener("click", async () => {
  const mode = getSelectedModeValue();
  const text = queryInput.value.trim();
  const sourceFile = sourceInput.files[0];
  const targetFile = targetInput.files[0];

  if (mode === "chat") {
    appendMessage(text || "[Auto Message]", "user");
    await restChatSend(text, sourceFile, targetFile);
    queryInput.value = "";
  } else {
    if (!sourceFile) {
      appendMessage("⚠️ Please upload a source file.", "bot");
      return;
    }
    const etlMode = text ? "query" : "auto";
    appendMessage(text || "[Auto Mode Triggered – Analyzing files...]", "user");
    await analyzeSend(sourceFile, targetFile, text, etlMode);
    queryInput.value = "";
  }
});

// =======================
//  Download Buttons
// =======================
downloadExcelBtn.addEventListener("click", async () => {
  try {
    const res = await fetch(`${API_BASE}/download-excel`);
    if (!res.ok) throw new Error("No Excel file available.");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "Transformed_Output.xlsx";
    a.click();
    URL.revokeObjectURL(url);
    downloadStatus.textContent = "✅ Excel Downloaded";
  } catch (err) {
    appendMessage("❌ Excel download error: " + err.message, "bot");
  } finally {
    setTimeout(() => (downloadStatus.textContent = ""), 3000);
  }
});

downloadSqlBtn.addEventListener("click", async () => {
  try {
    const res = await fetch(`${API_BASE}/download-sql`);
    if (!res.ok) throw new Error("No SQL file available.");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "Transformation_Rules.sql";
    a.click();
    URL.revokeObjectURL(url);
    downloadStatus.textContent = "✅ SQL Downloaded";
  } catch (err) {
    appendMessage("❌ SQL download error: " + err.message, "bot");
  } finally {
    setTimeout(() => (downloadStatus.textContent = ""), 3000);
  }
});
