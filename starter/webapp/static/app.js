"use strict";

// ---- tiny DOM helper (textContent-based, so no HTML injection) -------------
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  (Array.isArray(children) ? children : [children]).forEach((c) => {
    if (c == null) return;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  });
  return node;
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    const old = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => (btn.textContent = old), 1200);
  } catch (e) {
    btn.textContent = "Copy failed";
  }
}

// ---- tab switching ----------------------------------------------------------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
    if (tab.dataset.tab === "history") loadHistory();
    if (tab.dataset.tab === "kb") loadKnowledgeBase();
  });
});

// ---- retrieval process rendering -------------------------------------------
function renderOutput(output) {
  if (output && typeof output === "object") {
    return el("pre", { class: "json", text: pretty(output) });
  }
  return el("div", { class: "text", text: String(output) });
}

function renderProcess(process) {
  const details = el("details", { class: "process" }, [
    el("summary", { text: "Retrieval process (" + process.length + " steps)" }),
  ]);
  process.forEach((step) => {
    const wrap = el("div", { class: "step " + step.type });
    if (step.type === "reasoning") {
      wrap.appendChild(el("div", { class: "step-head", text: "💭 Reasoning" }));
      wrap.appendChild(el("div", { class: "text", text: step.content }));
    } else if (step.type === "tool_call") {
      wrap.appendChild(el("div", { class: "step-head", text: "🔧 Call → " + step.tool }));
      wrap.appendChild(renderOutput(step.input));
    } else if (step.type === "tool_result") {
      wrap.appendChild(el("div", { class: "step-head", text: "📥 Result ← " + step.tool }));
      wrap.appendChild(renderOutput(step.output));
    } else if (step.type === "final_answer") {
      wrap.appendChild(el("div", { class: "step-head", text: "✅ Final answer" }));
      wrap.appendChild(el("div", { class: "text", text: step.content }));
    }
    details.appendChild(wrap);
  });
  return details;
}

// ---- answer card (shared by Ask + History) ---------------------------------
function renderAnswerCard(data) {
  const source = (data.answer_json && data.answer_json.source) || data.source || "unknown";
  const card = el("div", { class: "card" });

  card.appendChild(el("div", { class: "q" }, [
    (data.query || data.answer_json?.question || "") + "  ",
    el("span", { class: "badge " + source, text: source.replace("_", " ") }),
  ]));

  // content area toggled between NL and JSON
  const content = el("div");
  const nlNode = el("div", { class: "answer-nl", text: data.answer_nl || "" });
  const jsonNode = el("pre", { class: "json", text: pretty(data.answer_json || {}) });
  content.appendChild(nlNode);

  let mode = "nl";
  const nlBtn = el("button", { class: "active", text: "Natural language" });
  const jsonBtn = el("button", { text: "JSON" });
  const copyBtn = el("button", { class: "ghost copy-btn", text: "Copy" });

  function setMode(next) {
    mode = next;
    content.replaceChildren(mode === "nl" ? nlNode : jsonNode);
    nlBtn.classList.toggle("active", mode === "nl");
    jsonBtn.classList.toggle("active", mode === "json");
  }
  nlBtn.addEventListener("click", () => setMode("nl"));
  jsonBtn.addEventListener("click", () => setMode("json"));
  copyBtn.addEventListener("click", () =>
    copyText(mode === "nl" ? data.answer_nl || "" : pretty(data.answer_json || {}), copyBtn)
  );

  card.appendChild(el("div", { class: "toolbar" }, [
    el("span", { class: "toggle" }, [nlBtn, jsonBtn]),
    copyBtn,
  ]));
  card.appendChild(content);

  if (data.retrieval_process && data.retrieval_process.length) {
    card.appendChild(renderProcess(data.retrieval_process));
  }
  return card;
}

// ---- Ask --------------------------------------------------------------------
const askForm = document.getElementById("ask-form");
const askBtn = document.getElementById("ask-btn");
const askStatus = document.getElementById("ask-status");
const askResult = document.getElementById("ask-result");

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = document.getElementById("query").value.trim();
  if (!query) return;
  askBtn.disabled = true;
  askStatus.className = "status";
  askStatus.textContent = "Thinking… (retrieving, evaluating, maybe searching the web)";
  askResult.replaceChildren();
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    askStatus.textContent = "";
    askResult.appendChild(renderAnswerCard(data));
  } catch (err) {
    askStatus.className = "status error";
    askStatus.textContent = "Error: " + err.message;
  } finally {
    askBtn.disabled = false;
  }
});

// ---- History ----------------------------------------------------------------
const historyList = document.getElementById("history-list");
document.getElementById("refresh-history").addEventListener("click", loadHistory);

async function loadHistory() {
  historyList.replaceChildren(el("div", { class: "empty", text: "Loading…" }));
  try {
    const res = await fetch("/api/history");
    const items = await res.json();
    if (!items.length) {
      historyList.replaceChildren(el("div", { class: "empty", text: "No prompts yet. Ask something!" }));
      return;
    }
    historyList.replaceChildren();
    items.forEach((item) => {
      const source = item.source || "unknown";
      const row = el("details", { class: "row" }, [
        el("summary", {}, [
          el("span", { class: "title", text: item.query }),
          el("span", { class: "meta", text: item.timestamp }),
          el("span", { class: "badge " + source, text: source.replace("_", " ") }),
        ]),
      ]);
      const body = el("div", { class: "body" });
      // lazily render the card on first open
      row.addEventListener("toggle", () => {
        if (row.open && !body.dataset.rendered) {
          body.appendChild(renderAnswerCard(item));
          body.dataset.rendered = "1";
        }
      });
      row.appendChild(body);
      historyList.appendChild(row);
    });
  } catch (err) {
    historyList.replaceChildren(el("div", { class: "empty", text: "Error: " + err.message }));
  }
}

// ---- Knowledge Base ---------------------------------------------------------
const kbList = document.getElementById("kb-list");
document.getElementById("refresh-kb").addEventListener("click", loadKnowledgeBase);

const KB_FIELDS = ["Name", "Platform", "Genre", "Publisher", "YearOfRelease", "Description"];

async function loadKnowledgeBase() {
  kbList.replaceChildren(el("div", { class: "empty", text: "Loading…" }));
  try {
    const res = await fetch("/api/knowledge");
    const games = await res.json();
    if (!games.length) {
      kbList.replaceChildren(el("div", { class: "empty", text: "Knowledge base is empty." }));
      return;
    }
    kbList.replaceChildren();
    games.forEach((g) => {
      const row = el("details", { class: "row" }, [
        el("summary", {}, [
          el("span", { class: "title", text: g.Name || g.id }),
          el("span", { class: "meta", text: (g.Platform || "") + " · " + (g.YearOfRelease || "") }),
        ]),
      ]);
      const table = el("table", { class: "fields" });
      KB_FIELDS.forEach((f) => {
        table.appendChild(el("tr", {}, [
          el("td", { class: "k", text: f }),
          el("td", { text: g[f] == null ? "—" : String(g[f]) }),
        ]));
      });
      row.appendChild(el("div", { class: "body" }, [table]));
      kbList.appendChild(row);
    });
  } catch (err) {
    kbList.replaceChildren(el("div", { class: "empty", text: "Error: " + err.message }));
  }
}
