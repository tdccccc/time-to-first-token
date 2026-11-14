(function () {
  "use strict";

  function build() {
    var nav = document.getElementById("page-toc-nav");
    var panel = document.getElementById("page-toc");
    var article = document.querySelector(".md-content__inner");
    if (!nav || !panel || !article) return;

    var headings = Array.prototype.filter.call(
      article.querySelectorAll("h2, h3"),
      function (heading) {
        return heading.id && heading.textContent.trim();
      }
    );
    if (!headings.length) {
      panel.hidden = true;
      return;
    }

    var list = document.createElement("ol");
    list.className = "page-toc__list";

    var links = headings.map(function (heading) {
      var clone = heading.cloneNode(true);
      clone.querySelectorAll(".headerlink").forEach(function (link) {
        link.remove();
      });
      var text = clone.textContent.replace(/\s+/g, " ").trim();
      var item = document.createElement("li");
      var level = heading.tagName.toLowerCase() === "h3" ? "3" : "2";
      item.className = "page-toc__item page-toc__item--" + level;
      var link = document.createElement("a");
      link.href = "#" + heading.id;
      link.textContent = text;
      item.appendChild(link);
      list.appendChild(item);
      return link;
    });

    nav.appendChild(list);

    var ticking = false;
    function update() {
      ticking = false;
      var offset = window.scrollY + 120;
      var active = links[0];
      headings.forEach(function (heading, index) {
        if (heading.getBoundingClientRect().top + window.scrollY <= offset) {
          active = links[index];
        }
      });
      links.forEach(function (link) {
        link.classList.toggle("is-active", link === active);
      });
    }

    function requestUpdate() {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(update);
    }

    window.addEventListener("scroll", requestUpdate, { passive: true });
    window.addEventListener("resize", requestUpdate);
    panel.addEventListener("toggle", function () {
      if (panel.open) requestUpdate();
    });
    update();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
