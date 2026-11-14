(function () {
  "use strict";

  var panel = document.getElementById("notes-panel");
  if (!panel) return;

  var input = document.getElementById("notes-input");
  var statusEl = document.getElementById("notes-status");
  var countEl = document.getElementById("notes-count");
  var feedbackEl = document.getElementById("notes-feedback");
  var linkEl = document.getElementById("notes-link");
  var syncBtn = document.getElementById("notes-sync");
  var downloadBtn = document.getElementById("notes-download");
  var clearBtn = document.getElementById("notes-clear");
  var settingsEl = document.getElementById("notes-settings");
  var repoInput = document.getElementById("notes-repo");
  var branchInput = document.getElementById("notes-branch");
  var tokenInput = document.getElementById("notes-token");
  var saveSettingsBtn = document.getElementById("notes-save-settings");
  var loadBtn = document.getElementById("notes-load");
  var forgetBtn = document.getElementById("notes-forget");
  var fab = document.getElementById("notes-fab");
  var closeBtn = document.getElementById("notes-close");
  var backdrop = document.getElementById("notes-backdrop");

  if (!input) return;

  function meta(name) {
    var el = document.querySelector('meta[name="' + name + '"]');
    return el ? (el.getAttribute("content") || "") : "";
  }

  var STORE = {
    note: "ttft-note:",
    repo: "ttft-notes-repo",
    branch: "ttft-notes-branch",
    token: "ttft-notes-token"
  };

  var pageKey = meta("notes-page") || location.pathname;
  var pageTitle = meta("notes-title") || document.title;
  var defaultRepo = (meta("notes-repo") || "").replace(/^https:\/\/github\.com\//, "").replace(/\.git$/, "");
  var defaultBranch = meta("notes-branch") || "main";
  var noteKey = STORE.note + (pageKey || "index");

  function get(key) {
    try { return localStorage.getItem(key) || ""; } catch (e) { return ""; }
  }
  function set(key, value) {
    try { localStorage.setItem(key, value); return true; } catch (e) { return false; }
  }
  function remove(key) {
    try { localStorage.removeItem(key); } catch (e) {}
  }

  function timeText() {
    try {
      return new Date().toLocaleTimeString("zh-CN", { hour12: false });
    } catch (e) {
      return new Date().toLocaleTimeString();
    }
  }

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
  }
  function setFeedback(text, isError) {
    if (!feedbackEl) return;
    feedbackEl.textContent = text || "";
    feedbackEl.classList.toggle("is-error", !!isError);
  }
  function updateCount() {
    if (!countEl) return;
    var n = Array.from(input.value).length;
    countEl.textContent = n + " 字";
  }
  function showLink(url) {
    if (!linkEl) return;
    if (url) {
      linkEl.href = url;
      linkEl.hidden = false;
    } else {
      linkEl.hidden = true;
      linkEl.removeAttribute("href");
    }
  }

  function readRepo() {
    var value = (repoInput ? repoInput.value : "") || defaultRepo || "";
    return value.trim()
      .replace(/^https:\/\/github\.com\//, "")
      .replace(/\.git$/, "")
      .replace(/\/+$/, "");
  }
  function readBranch() {
    var value = (branchInput ? branchInput.value : "") || defaultBranch || "main";
    return value.trim() || "main";
  }
  function readToken() {
    return ((tokenInput ? tokenInput.value : "") || "").trim();
  }

  function notePath() {
    var p = String(pageKey || "").replace(/^\.?\//, "").replace(/\/+$/, "");
    if (!p || p === "." || p === "./") p = "index";
    p = p.replace(/\.html$/, "");
    return "notes/" + p + ".md";
  }

  function encodedNotePath() {
    return notePath().split("/").map(encodeURIComponent).join("/");
  }

  function b64Encode(text) {
    var bytes = new TextEncoder().encode(text);
    var binary = "";
    for (var i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }
  function b64Decode(value) {
    var binary = atob(value);
    var bytes = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new TextDecoder().decode(bytes);
  }

  async function api(path, options) {
    options = options || {};
    var headers = {
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28"
    };
    var token = readToken();
    if (token) headers["Authorization"] = "Bearer " + token;
    if (options.body) headers["Content-Type"] = "application/json";

    var response = await fetch("https://api.github.com" + path, {
      method: options.method || "GET",
      headers: headers,
      body: options.body
    });
    var text = await response.text();
    var data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }
    if (!response.ok) {
      var err = new Error((data && data.message) || ("HTTP " + response.status));
      err.status = response.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  function errorText(err) {
    if (!err) return "未知错误。";
    if (err.status === 401) return "token 无效或已过期。";
    if (err.status === 403) return "token 没有权限，或触发了 GitHub API 限流。";
    if (err.status === 404) return "仓库或分支不存在，或 token 没有该仓库权限。";
    if (err.status === 409) return "文件冲突：先点“从仓库载入”，再同步。";
    return "同步失败：" + (err.message || String(err));
  }

  function openSettings() {
    if (settingsEl) settingsEl.open = true;
  }

  function saveSettings(quiet) {
    var repo = readRepo();
    if (!/^[^\/\s]+\/[^\/\s]+$/.test(repo)) {
      setFeedback("仓库格式应为 owner/repo。", true);
      openSettings();
      return false;
    }
    set(STORE.repo, repo);
    set(STORE.branch, readBranch());
    if (readToken()) set(STORE.token, readToken());
    else remove(STORE.token);
    if (!quiet) setFeedback("设置已保存到本浏览器。");
    return true;
  }

  function setBusy(busy, label) {
    if (!syncBtn) return;
    syncBtn.disabled = !!busy;
    syncBtn.textContent = busy ? (label || "处理中…") : "同步到仓库";
  }

  function saveLocalNow() {
    var text = input.value;
    if (!text) {
      remove(noteKey);
      setStatus("尚未输入");
      return;
    }
    if (set(noteKey, text)) {
      setStatus("已保存到浏览器 " + timeText());
    } else {
      setStatus("浏览器存储不可用");
    }
  }

  async function syncNote() {
    var text = input.value.trim();
    if (!text) {
      setFeedback("笔记是空的，先写点什么。", true);
      return;
    }
    if (!saveSettings(true)) return;

    var repo = readRepo();
    var branch = readBranch();
    var token = readToken();
    if (!token) {
      openSettings();
      setFeedback("同步需要 GitHub token：先填好并保存。", true);
      if (tokenInput) tokenInput.focus();
      return;
    }

    var path = notePath();
    setBusy(true, "同步中…");
    setFeedback("");
    try {
      var sha = null;
      var remoteText = null;
      try {
        var existing = await api("/repos/" + repo + "/contents/" + encodedNotePath() + "?ref=" + encodeURIComponent(branch));
        if (existing && existing.sha) sha = existing.sha;
        if (existing && existing.content) remoteText = b64Decode(String(existing.content).replace(/\n/g, ""));
      } catch (err) {
        if (err.status !== 404) throw err;
      }

      if (remoteText !== null && remoteText === text) {
        setFeedback("仓库里已经是最新版本，没有产生新提交。");
        setStatus("已是最新");
        return;
      }

      var payload = {
        message: "docs(notes): 更新「" + pageTitle + "」的笔记",
        content: b64Encode(text),
        branch: branch
      };
      if (sha) payload.sha = sha;

      await api("/repos/" + repo + "/contents/" + encodedNotePath(), {
        method: "PUT",
        body: JSON.stringify(payload)
      });

      var url = "https://github.com/" + repo + "/blob/" + encodeURIComponent(branch) + "/" +
        path.split("/").map(encodeURIComponent).join("/");
      showLink(url);
      setFeedback("已同步到 " + path + "。");
      setStatus("已同步 " + timeText());
    } catch (err) {
      setFeedback(errorText(err), true);
    } finally {
      setBusy(false);
    }
  }

  async function loadRemote(silent) {
    var repo = readRepo();
    var branch = readBranch();
    var token = readToken();
    if (!token || !repo) {
      if (!silent) {
        openSettings();
        setFeedback("先填好仓库与 token。", true);
      }
      return;
    }

    setBusy(true, "读取中…");
    try {
      var data = await api("/repos/" + repo + "/contents/" + encodedNotePath() + "?ref=" + encodeURIComponent(branch));
      var text = b64Decode(String(data.content || "").replace(/\n/g, ""));
      if (!text) {
        if (!silent) setFeedback("仓库里还没有这一页的笔记。");
        return;
      }
      if (silent && input.value.trim()) return;
      if (!silent && input.value.trim() && input.value.trim() !== text.trim()) {
        if (!confirm("本地笔记与仓库版本不同，用仓库版本覆盖本地吗？")) return;
      }
      input.value = text;
      saveLocalNow();
      updateCount();
      if (!silent) {
        setFeedback("已从仓库载入。");
        setStatus("已载入 " + timeText());
      }
    } catch (err) {
      if (err.status === 404) {
        if (!silent) setFeedback("仓库里还没有这一页的笔记。");
      } else if (!silent) {
        setFeedback(errorText(err), true);
      }
    } finally {
      setBusy(false);
    }
  }

  function downloadNote() {
    var text = input.value;
    if (!text) {
      setFeedback("笔记是空的。", true);
      return;
    }
    var blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = notePath().split("/").pop();
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function clearNote() {
    if (!confirm("清空这一页在浏览器里的笔记？仓库里的版本不会被删除。")) return;
    input.value = "";
    remove(noteKey);
    updateCount();
    setStatus("尚未输入");
    setFeedback("浏览器里的笔记已清空。");
    showLink("");
  }

  function setOpen(open) {
    document.documentElement.classList.toggle("notes-open", !!open);
    if (fab) fab.setAttribute("aria-expanded", open ? "true" : "false");
    if (open && input) {
      setTimeout(function () { input.focus(); }, 180);
    }
  }

  if (repoInput) repoInput.value = get(STORE.repo) || defaultRepo;
  if (branchInput) branchInput.value = get(STORE.branch) || defaultBranch;
  if (tokenInput) tokenInput.value = get(STORE.token) || "";

  var local = get(noteKey);
  if (local) {
    input.value = local;
    setStatus("已从本浏览器恢复");
  }
  updateCount();
  if (!local && readToken()) loadRemote(true);

  input.addEventListener("input", function () {
    updateCount();
    setStatus("正在保存…");
    setFeedback("");
    showLink("");
    clearTimeout(input._notesTimer);
    input._notesTimer = setTimeout(saveLocalNow, 300);
  });
  input.addEventListener("blur", saveLocalNow);

  if (syncBtn) syncBtn.addEventListener("click", syncNote);
  if (downloadBtn) downloadBtn.addEventListener("click", downloadNote);
  if (clearBtn) clearBtn.addEventListener("click", clearNote);
  if (saveSettingsBtn) saveSettingsBtn.addEventListener("click", function () {
    if (saveSettings(false)) setStatus("设置已保存");
  });
  if (loadBtn) loadBtn.addEventListener("click", function () {
    if (saveSettings(true)) loadRemote(false);
  });
  if (forgetBtn) forgetBtn.addEventListener("click", function () {
    remove(STORE.token);
    if (tokenInput) tokenInput.value = "";
    setFeedback("token 已从本浏览器清除。");
  });

  if (fab) fab.addEventListener("click", function () { setOpen(true); });
  if (closeBtn) closeBtn.addEventListener("click", function () { setOpen(false); });
  if (backdrop) backdrop.addEventListener("click", function () { setOpen(false); });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") setOpen(false);
  });
  window.addEventListener("pagehide", saveLocalNow);
})();
