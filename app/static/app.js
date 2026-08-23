const form = document.querySelector("#question-form");
const input = document.querySelector("#question");
const submit = document.querySelector("#submit");
const conversation = document.querySelector("#conversation");

function addMessage(content, role, extraClass = "") {
  const item = document.createElement("article");
  item.className = `message ${role}-message ${extraClass}`;
  item.innerHTML = `<div class="avatar">${role === "user" ? "You" : "D"}</div><div class="bubble"><p></p></div>`;
  item.querySelector("p").textContent = content;
  conversation.append(item);
  item.scrollIntoView({ behavior: "smooth", block: "end" });
  return item;
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
    const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
    const data = await response.json();
    pending.remove();
    addMessage(data.answer || data.error || "No valid response was received.", "assistant", data.error ? "error" : "");
  } catch (_) {
    pending.remove(); addMessage("Unable to reach the service. Make sure the web server is still running.", "assistant", "error");
  } finally { submit.disabled = false; input.focus(); }
});
