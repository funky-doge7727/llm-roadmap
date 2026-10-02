import Prism from "prismjs";
import "prismjs/components/prism-clike.js";
import "prismjs/components/prism-javascript.js";
import "prismjs/components/prism-python.js";
import { STAGES } from "./stages.js";

const roadmapEl = document.getElementById("roadmap");
const tocEl = document.getElementById("tocList");
const STORAGE_KEY = "llm-roadmap-progress";

function loadProgress() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch {
    return {};
  }
}

function saveProgress(state) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function formatSource(fn) {
  const raw = fn.toString();
  const lines = raw.split("\n");
  if (lines.length <= 2) return raw.trim();
  const header = lines[0].trim();
  const footer = lines[lines.length - 1].trim();
  const body = lines.slice(1, -1);
  const indents = body.filter((l) => l.trim().length > 0).map((l) => l.match(/^\s*/)[0].length);
  const minIndent = indents.length ? Math.min(...indents) : 0;
  const dedented = body.map((l) => (l.length >= minIndent ? l.slice(minIndent) : l.trim()));
  return [header, ...dedented, footer].join("\n");
}

function buildCopyButton(getText) {
  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.type = "button";
  btn.textContent = "Copy";
  btn.addEventListener("click", () => {
    navigator.clipboard.writeText(getText()).then(() => {
      btn.textContent = "Copied";
      setTimeout(() => (btn.textContent = "Copy"), 1500);
    });
  });
  return btn;
}

const LANG_DISPLAY = { javascript: "JavaScript", python: "Python" };

// A small VS-Code-style editor surface: a line-number gutter kept in sync
// with a syntax-highlighted <pre><code>, plus a status bar underneath.
function buildEditor(text, lang) {
  const lines = text.split("\n");

  const body = document.createElement("div");
  body.className = "code-editor-body";

  const gutter = document.createElement("div");
  gutter.className = "line-numbers";
  gutter.setAttribute("aria-hidden", "true");
  lines.forEach((_, i) => {
    const n = document.createElement("span");
    n.textContent = String(i + 1);
    gutter.appendChild(n);
  });

  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.className = `language-${lang}`;
  const grammar = Prism.languages[lang];
  code.innerHTML = grammar ? Prism.highlight(text, grammar, lang) : escapeHtml(text);
  pre.appendChild(code);

  body.appendChild(gutter);
  body.appendChild(pre);

  const statusBar = document.createElement("div");
  statusBar.className = "code-status-bar";
  const langLabel = document.createElement("span");
  langLabel.className = "status-lang";
  langLabel.innerHTML = `<span class="status-dot"></span>${LANG_DISPLAY[lang] || lang}`;
  const lineCount = document.createElement("span");
  lineCount.className = "status-lines";
  lineCount.textContent = `${lines.length} lines`;
  const problems = document.createElement("span");
  problems.className = "status-problems ok";
  problems.innerHTML = "&#10003; No problems";

  statusBar.appendChild(langLabel);
  statusBar.appendChild(lineCount);
  statusBar.appendChild(problems);

  return { body, statusBar };
}

function buildStaticCodeBlock(label, text, lang) {
  const wrap = document.createElement("div");
  wrap.className = "code-block";

  const head = document.createElement("div");
  head.className = "code-head";
  head.innerHTML = `<span class="code-label">${escapeHtml(label)}</span>`;
  const btnGroup = document.createElement("div");
  btnGroup.className = "code-head-btns";
  btnGroup.appendChild(buildCopyButton(() => text));
  head.appendChild(btnGroup);

  const { body, statusBar } = buildEditor(text, lang);

  wrap.appendChild(head);
  wrap.appendChild(body);
  wrap.appendChild(statusBar);
  return wrap;
}

function buildRunnableBlock(stage) {
  const source = formatSource(stage.run);

  const wrap = document.createElement("div");
  wrap.className = "code-block runnable";

  const head = document.createElement("div");
  head.className = "code-head";
  head.innerHTML = `<span class="code-label">run-in-browser.js</span>`;

  const runBtn = document.createElement("button");
  runBtn.className = "run-btn";
  runBtn.type = "button";
  runBtn.textContent = "▶ Run";

  const btnGroup = document.createElement("div");
  btnGroup.className = "code-head-btns";
  btnGroup.appendChild(buildCopyButton(() => source));
  btnGroup.appendChild(runBtn);
  head.appendChild(btnGroup);

  const { body, statusBar } = buildEditor(source, "javascript");

  const output = document.createElement("div");
  output.className = "run-output";
  output.hidden = true;

  const statusLine = document.createElement("div");
  statusLine.className = "run-status";

  const logPre = document.createElement("pre");
  const logCode = document.createElement("code");
  logPre.appendChild(logCode);

  output.appendChild(statusLine);
  output.appendChild(logPre);

  wrap.appendChild(head);
  wrap.appendChild(body);
  wrap.appendChild(statusBar);
  if (stage.modelNote) {
    const note = document.createElement("p");
    note.className = "model-note";
    note.textContent = stage.modelNote;
    wrap.appendChild(note);
  }
  wrap.appendChild(output);

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    runBtn.textContent = "Running…";
    output.hidden = false;
    output.classList.remove("errored");
    statusLine.textContent = "";
    logCode.textContent = "";

    const ctx = {
      log: (msg) => {
        logCode.textContent += (logCode.textContent ? "\n" : "") + msg;
        logPre.scrollTop = logPre.scrollHeight;
      },
      status: (msg) => {
        statusLine.textContent = msg || "";
      }
    };

    const started = performance.now();
    try {
      await stage.run(ctx);
      statusLine.textContent = `Done in ${((performance.now() - started) / 1000).toFixed(1)}s.`;
    } catch (err) {
      output.classList.add("errored");
      statusLine.textContent = "Error — see below.";
      ctx.log(String(err && err.stack ? err.stack : err));
      console.error(err);
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = "▶ Run again";
    }
  });

  return wrap;
}

function buildLesson(stage) {
  const details = document.createElement("details");
  details.className = "lesson";

  const summary = document.createElement("summary");
  summary.innerHTML = `<span class="lesson-icon">&#9679;</span> Lesson, runnable code &amp; videos`;
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "lesson-body";

  if (stage.lesson) {
    const p = document.createElement("p");
    p.className = "lesson-text";
    p.textContent = stage.lesson;
    body.appendChild(p);
  }

  const tabs = document.createElement("div");
  tabs.className = "code-tabs";
  const tabBar = document.createElement("div");
  tabBar.className = "tab-bar";
  const runTabBtn = document.createElement("button");
  runTabBtn.type = "button";
  runTabBtn.className = "tab-btn active";
  runTabBtn.textContent = "Run in browser";
  const pyTabBtn = document.createElement("button");
  pyTabBtn.type = "button";
  pyTabBtn.className = "tab-btn";
  pyTabBtn.textContent = "Production reference (Python)";
  tabBar.appendChild(runTabBtn);
  tabBar.appendChild(pyTabBtn);

  const runPanel = buildRunnableBlock(stage);
  const pyPanel = buildStaticCodeBlock(stage.pythonCode.label, stage.pythonCode.text, "python");
  pyPanel.hidden = true;

  runTabBtn.addEventListener("click", () => {
    runTabBtn.classList.add("active");
    pyTabBtn.classList.remove("active");
    runPanel.hidden = false;
    pyPanel.hidden = true;
  });
  pyTabBtn.addEventListener("click", () => {
    pyTabBtn.classList.add("active");
    runTabBtn.classList.remove("active");
    pyPanel.hidden = false;
    runPanel.hidden = true;
  });

  tabs.appendChild(tabBar);
  tabs.appendChild(runPanel);
  tabs.appendChild(pyPanel);
  body.appendChild(tabs);

  if (stage.videos && stage.videos.length) {
    const videoHeading = document.createElement("p");
    videoHeading.className = "video-heading";
    videoHeading.textContent = "Watch";
    body.appendChild(videoHeading);

    const videoList = document.createElement("div");
    videoList.className = "video-list";
    stage.videos.forEach((v) => {
      const a = document.createElement("a");
      a.className = "video-link";
      a.href = v.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.innerHTML = `<span class="video-play">&#9654;</span><span class="video-info"><strong>${escapeHtml(v.title)}</strong><span>${escapeHtml(v.channel)}</span></span>`;
      videoList.appendChild(a);
    });
    body.appendChild(videoList);
  }

  details.appendChild(body);
  return details;
}

export function render() {
  const progress = loadProgress();
  let totalTopics = 0;

  STAGES.forEach((stage, sIdx) => {
    const stageId = `stage-${sIdx}`;

    const tocLi = document.createElement("li");
    const tocA = document.createElement("a");
    tocA.href = `#${stageId}`;
    tocA.innerHTML = `<span class="toc-num">${String(sIdx + 1).padStart(2, "0")}</span> ${stage.title}`;
    tocLi.appendChild(tocA);
    tocEl.appendChild(tocLi);

    const stageEl = document.createElement("section");
    stageEl.className = "stage";
    stageEl.id = stageId;

    const rail = document.createElement("div");
    rail.className = "stage-rail";
    rail.innerHTML = `<div class="stage-num">${sIdx + 1}</div>${sIdx < STAGES.length - 1 ? '<div class="stage-line"></div>' : ""}`;

    const card = document.createElement("div");
    card.className = "stage-card";

    const head = document.createElement("div");
    head.className = "stage-head";
    head.innerHTML = `<h2>${stage.title}</h2><span class="badge badge-${stage.level}">${stage.level}</span>`;

    const desc = document.createElement("p");
    desc.className = "stage-desc";
    desc.textContent = stage.desc;

    const list = document.createElement("ul");
    list.className = "topic-list";

    stage.topics.forEach(([name, blurb], tIdx) => {
      totalTopics++;
      const topicId = `${stageId}-t${tIdx}`;
      const li = document.createElement("li");
      li.className = "topic";
      if (progress[topicId]) li.classList.add("done");

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = !!progress[topicId];
      checkbox.addEventListener("change", () => {
        const state = loadProgress();
        state[topicId] = checkbox.checked;
        saveProgress(state);
        li.classList.toggle("done", checkbox.checked);
        updateProgressBar();
      });

      const textWrap = document.createElement("div");
      textWrap.className = "topic-text";
      textWrap.innerHTML = `<strong>${name}</strong><span>${blurb}</span>`;

      li.appendChild(checkbox);
      li.appendChild(textWrap);
      li.addEventListener("click", (e) => {
        if (e.target !== checkbox) checkbox.click();
      });

      list.appendChild(li);
    });

    card.appendChild(head);
    card.appendChild(desc);
    card.appendChild(list);
    card.appendChild(buildLesson(stage));

    stageEl.appendChild(rail);
    stageEl.appendChild(card);
    roadmapEl.appendChild(stageEl);
  });

  document.getElementById("progressTotal").textContent = totalTopics;
  updateProgressBar();
}

function updateProgressBar() {
  const progress = loadProgress();
  const done = Object.values(progress).filter(Boolean).length;
  const total = STAGES.reduce((sum, s) => sum + s.topics.length, 0);
  const pct = total ? Math.round((done / total) * 100) : 0;
  document.getElementById("progressFill").style.width = `${pct}%`;
  document.getElementById("progressCount").textContent = done;
}
