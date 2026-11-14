(function () {
  "use strict";

  var root = document.querySelector(".book-progress");
  if (!root) return;

  var KEY = "ttft-book-progress:v2";
  var TOTAL = 31;
  var ALPHABET = "0123456789abcdefghijklmnopqrstuv";
  var boxes = Array.prototype.slice.call(root.querySelectorAll("input[data-chapter]"));
  var fill = document.getElementById("book-progress-fill");
  var text = document.getElementById("book-progress-text");
  var feedback = document.getElementById("book-progress-feedback");
  var exportBtn = document.getElementById("book-progress-export");
  var importInput = document.getElementById("book-progress-import");
  var linkBtn = document.getElementById("book-progress-link");
  var resetBtn = document.getElementById("book-progress-reset");

  function message(value, isError) {
    if (!feedback) return;
    feedback.textContent = value || "";
    feedback.classList.toggle("is-error", !!isError);
  }

  function read() {
    try {
      var raw = localStorage.getItem(KEY);
      var arr = JSON.parse(raw || "[]");
      if (!Array.isArray(arr)) return [];
      return arr.map(Number).filter(function (n) { return n >= 1 && n <= TOTAL; });
    } catch (e) {
      return [];
    }
  }

  function write(arr) {
    try {
      localStorage.setItem(KEY, JSON.stringify(arr));
      return true;
    } catch (e) {
      message("浏览器存储不可用，进度只在当前页面有效。", true);
      return false;
    }
  }

  function selected() {
    return boxes.filter(function (box) { return box.checked; }).map(function (box) {
      return Number(box.getAttribute("data-chapter"));
    });
  }

  function update() {
    var count = selected().length;
    if (fill) fill.style.width = Math.round(count / TOTAL * 100) + "%";
    if (text) text.textContent = count + " / " + TOTAL + " 章";
  }

  function save() {
    write(selected());
    update();
  }

  function setState(arr) {
    var set = {};
    arr.forEach(function (n) { set[n] = true; });
    boxes.forEach(function (box) {
      box.checked = !!set[Number(box.getAttribute("data-chapter"))];
    });
    update();
  }

  function encode(arr) {
    var bits = "";
    for (var i = 1; i <= TOTAL; i++) bits += arr.indexOf(i) >= 0 ? "1" : "0";
    var out = "";
    for (var j = 0; j < bits.length; j += 5) {
      out += ALPHABET[parseInt(bits.slice(j, j + 5).padEnd(5, "0"), 2)];
    }
    return out;
  }

  function decode(str) {
    var bits = "";
    for (var i = 0; i < str.length; i++) {
      var v = ALPHABET.indexOf(str.charAt(i).toLowerCase());
      if (v < 0) continue;
      bits += v.toString(2).padStart(5, "0");
    }
    var arr = [];
    for (var j = 0; j < TOTAL; j++) {
      if (bits.charAt(j) === "1") arr.push(j + 1);
    }
    return arr;
  }

  function fromHash() {
    var match = /(?:^#|&)p=([0-9a-v]+)/i.exec(location.hash || "");
    return match ? decode(match[1]) : null;
  }

  function download(arr) {
    var payload = { app: "ttft-book", version: 2, chapters: arr };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "reading-progress.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function importFile(file) {
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var data = JSON.parse(String(reader.result || "{}"));
        var arr = Array.isArray(data) ? data : data.chapters;
        if (!Array.isArray(arr)) throw new Error("缺少 chapters 字段");
        arr = arr.map(Number).filter(function (n) { return n >= 1 && n <= TOTAL; });
        setState(arr);
        write(arr);
        message("已导入 " + arr.length + " 章。");
      } catch (err) {
        message("导入失败：" + (err.message || String(err)), true);
      }
    };
    reader.onerror = function () { message("读取文件失败。", true); };
    reader.readAsText(file);
  }

  boxes.forEach(function (box) { box.addEventListener("change", save); });

  var initial = fromHash();
  if (initial) {
    setState(initial);
    write(initial);
    message("已从同步链接恢复进度。");
  } else {
    setState(read());
  }

  if (exportBtn) exportBtn.addEventListener("click", function () {
    download(selected());
    message("已导出 JSON。");
  });

  if (importInput) importInput.addEventListener("change", function () {
    if (importInput.files && importInput.files[0]) importFile(importInput.files[0]);
    importInput.value = "";
  });

  if (linkBtn) linkBtn.addEventListener("click", function () {
    var url = location.origin + location.pathname + "#p=" + encode(selected());
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).then(function () {
        message("同步链接已复制。");
      }, function () {
        window.prompt("复制这个链接：", url);
      });
    } else {
      window.prompt("复制这个链接：", url);
    }
  });

  if (resetBtn) resetBtn.addEventListener("click", function () {
    if (!window.confirm("重置本浏览器的阅读进度？已导出的 JSON 或同步链接不受影响。")) return;
    setState([]);
    write([]);
    message("进度已重置。");
  });

  window.addEventListener("pagehide", save);
})();
