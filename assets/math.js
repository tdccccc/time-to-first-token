(function () {
  "use strict";

  function render() {
    if (typeof window.renderMathInElement !== "function") return;
    var target = document.querySelector(".md-content__inner") || document.body;
    window.renderMathInElement(target, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false }
      ],
      throwOnError: false,
      strict: false
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();
