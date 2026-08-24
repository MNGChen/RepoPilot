const form = document.querySelector("#question-form");
const input = document.querySelector("#question");
const submit = document.querySelector("#submit");
const conversation = document.querySelector("#conversation");
const debugMode = document.querySelector("#debug-mode");
const SESSION_KEY = "devpilot-session-id";

function getSessionId() {
  let sessionId = sessionStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID ? crypto.randomUUID().replaceAll("-", "") : `${Date.now()}${Math.random().toString(36).slice(2)}`;
    sessionStorage.setItem(SESSION_KEY, sessionId);
  }
  return sessionId;
}

function addMessage(content, role, extraClass = "") {
  const item = document.createElement("article");
  item.className = `message ${role}-message ${extraClass}`;
  item.innerHTML = `<div class="avatar">${role === "user" ? "You" : "D"}</div><div class="bubble"><p></p></div>`;
  item.querySelector("p").textContent = content;
  conversation.append(item);
  item.scrollIntoView({ behavior: "smooth", block: "end" });
  return item;
}

function addBehaviorTree(message, tree) {
  const details = document.createElement("details");
  details.className = "behavior";
  details.open = true;
  const summary = document.createElement("summary");
  summary.textContent = "Agent behavior tree";
  details.append(summary, renderTree(tree));
  message.querySelector(".bubble").append(details);
}

function renderTree(node) {
  const list = document.createElement("ul");
  list.className = "tree";
  const item = document.createElement("li");
  const label = document.createElement("span");
  label.className = "tree-label";
  label.textContent = node.label;
  const status = document.createElement("i");
  status.className = `tree-status ${node.status || "success"}`;
  item.append(label, status);
  if (node.detail) { const detail = document.createElement("span"); detail.className = "tree-detail"; detail.textContent = node.detail; item.append(detail); }
  if (node.result) { const result = document.createElement("span"); result.className = "tree-detail"; result.textContent = node.result; item.append(result); }
  if (node.children?.length) { const children = document.createElement("ul"); node.children.forEach((child) => children.append(renderTree(child).firstChild)); item.append(children); }
  list.append(item);
  return list;
}

function resize() { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 180)}px`; }
input.addEventListener("input", resize);
input.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });
document.querySelectorAll("[data-question]").forEach((button) => button.addEventListener("click", () => { input.value = button.dataset.question; resize(); input.focus(); }));

async function showGitHubStatus() {
  const status = document.querySelector("#github-status");
  try {
    const response = await fetch("/api/capabilities");
    const { githubAnalysisAvailable } = await response.json();
    status.textContent = githubAnalysisAvailable
      ? "GitHub analysis is ready."
      : "GitHub analysis needs Docker Desktop and a GitHub token.";
  } catch (_) {
    status.textContent = "GitHub availability could not be checked.";
  }
}

showGitHubStatus();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || submit.disabled) return;
  addMessage(question, "user"); input.value = ""; resize(); submit.disabled = true;
  const pending = addMessage("Reviewing the repository and preparing an answer…", "assistant", "loading");
  try {
    const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, debug: debugMode.checked, sessionId: getSessionId() }) });
    const data = await response.json();
    if (data.sessionId) sessionStorage.setItem(SESSION_KEY, data.sessionId);
    pending.remove();
    const answer = addMessage(data.answer || data.error || "No valid response was received.", "assistant", data.error ? "error" : "");
    if (data.behaviorTree) addBehaviorTree(answer, data.behaviorTree);
  } catch (_) {
    pending.remove(); addMessage("Unable to reach the service. Make sure the web server is still running.", "assistant", "error");
  } finally { submit.disabled = false; input.focus(); }
});
