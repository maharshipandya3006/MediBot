const form = document.querySelector("#chatForm");
const input = document.querySelector("#questionInput");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#sendButton");
const statusDot = document.querySelector("#statusDot");
const statusText = document.querySelector("#statusText");
const modelText = document.querySelector("#modelText");
const newChatButton = document.querySelector("#newChatButton");
const HISTORY_KEY = "medibot_session_history";
const MAX_HISTORY_MESSAGES = 12;
const TOPIC_KEYWORDS = [
  "abdomen",
  "acne",
  "allergy",
  "antibiotic",
  "anxiety",
  "asthma",
  "back pain",
  "bacteria",
  "bleeding",
  "blood",
  "blood pressure",
  "breathing",
  "cancer",
  "chest",
  "cold",
  "cough",
  "diabetes",
  "diarrhea",
  "dizziness",
  "ear",
  "fever",
  "flu",
  "headache",
  "heart",
  "hypertension",
  "infection",
  "injury",
  "kidney",
  "liver",
  "migraine",
  "nausea",
  "pain",
  "pneumonia",
  "rash",
  "sinus",
  "skin",
  "sore",
  "stomach",
  "stress",
  "stroke",
  "swelling",
  "symptom",
  "throat",
  "tonsil",
  "urine",
  "virus",
  "vomit",
  "wound",
  "bp",
];

const state = {
  busy: false,
  history: loadHistory(),
};

function loadHistory() {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed
      .filter((message) => {
        return (
          (message.role === "user" || message.role === "assistant") &&
          typeof message.content === "string" &&
          message.content.trim()
        );
      })
      .slice(-MAX_HISTORY_MESSAGES);
  } catch (error) {
    return [];
  }
}

function saveHistory() {
  sessionStorage.setItem(
    HISTORY_KEY,
    JSON.stringify(state.history.slice(-MAX_HISTORY_MESSAGES))
  );
}

function isLikelyTopicStart(content) {
  const normalized = content.toLowerCase().trim();

  if (normalized.length < 5) {
    return false;
  }

  return TOPIC_KEYWORDS.some((keyword) => normalized.includes(keyword));
}

function getContextHistoryForQuestion(question) {
  const currentQuestionStartsTopic = isLikelyTopicStart(question);
  const history = state.history;

  if (history.length === 0 || currentQuestionStartsTopic) {
    return [];
  }

  let startIndex = 0;

  for (let index = history.length - 1; index >= 0; index -= 1) {
    const message = history[index];

    if (message.role === "user" && isLikelyTopicStart(message.content)) {
      startIndex = index;
      break;
    }
  }

  return history.slice(startIndex);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

function setBusy(isBusy) {
  state.busy = isBusy;
  sendButton.disabled = isBusy;
  input.disabled = isBusy;
}

function addMessage(role, content, options = {}) {
  const shouldSave = options.save !== false;
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "MB";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const paragraph = document.createElement("p");
  paragraph.innerHTML = escapeHtml(content);
  bubble.appendChild(paragraph);

  article.append(avatar, bubble);
  messages.appendChild(article);

  if (shouldSave && (role === "user" || role === "assistant")) {
    state.history.push({ role, content });
    state.history = state.history.slice(-MAX_HISTORY_MESSAGES);
    saveHistory();
  }

  scrollToBottom();

  return article;
}

function hydrateSavedMessages() {
  if (state.history.length === 0) {
    return;
  }

  messages.innerHTML = "";

  state.history.forEach((message) => {
    addMessage(message.role, message.content, { save: false });
  });
}

function startNewChat() {
  state.history = [];
  saveHistory();
  messages.innerHTML = "";
  addMessage(
    "assistant",
    "Hello, I am Medibot! Feel free to ask me any medical related query :)",
    { save: false }
  );
  input.focus();
}

function addTypingMessage() {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = `
    <div class="avatar">MB</div>
    <div class="bubble">
      <div class="typing" aria-label="MediBot is thinking">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  messages.appendChild(article);
  scrollToBottom();
  return article;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");

    if (!response.ok) {
      throw new Error("Health check failed");
    }

    const data = await response.json();

    statusDot.classList.add("ready");
    statusText.textContent = data.vectorstore_ready
      ? "Service ready"
      : "Vector store missing";
    modelText.textContent = `Model: ${data.model}`;
  } catch (error) {
    statusDot.classList.add("error");
    statusText.textContent = "Service unavailable";
    modelText.textContent = "Check Render logs";
  }
}

async function askQuestion(question, history) {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question, history }),
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "MediBot could not answer right now.");
  }

  return data;
}

async function submitQuestion(question) {
  const cleanQuestion = question.trim();

  if (!cleanQuestion || state.busy) {
    return;
  }

  const contextHistory = getContextHistoryForQuestion(cleanQuestion);

  addMessage("user", cleanQuestion);
  input.value = "";
  resizeInput();
  setBusy(true);

  const typing = addTypingMessage();

  try {
    const data = await askQuestion(cleanQuestion, contextHistory);
    typing.remove();
    addMessage("assistant", data.answer);
  } catch (error) {
    typing.remove();
    addMessage(
      "assistant",
      `${error.message}\n\nPlease verify the backend environment variables and try again.`
    );
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(input.value);
});

input.addEventListener("input", resizeInput);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

newChatButton.addEventListener("click", startNewChat);

checkHealth();
hydrateSavedMessages();
resizeInput();
