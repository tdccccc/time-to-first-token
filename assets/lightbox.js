(function () {
  "use strict";

  var MIN_SCALE = 0.5;
  var MAX_SCALE = 3.5;
  var STEP_SCALE = 0.25;

  function createEl(tag, className, attrs) {
    var el = document.createElement(tag);
    if (className) el.className = className;
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === "text") {
          el.textContent = attrs[k];
        } else {
          el.setAttribute(k, attrs[k]);
        }
      });
    }
    return el;
  }

  function initLightbox() {
    var article = document.querySelector(".md-content__inner");
    if (!article) return;

    var images = Array.prototype.filter.call(
      article.querySelectorAll("img"),
      function (img) {
        if (img.classList.contains("twemoji") || img.classList.contains("emojione")) {
          return false;
        }
        if (img.closest("a")) return false;
        if (img.hasAttribute("data-no-lightbox")) return false;
        return true;
      }
    );

    if (!images.length) return;

    // Build modal once per document
    var overlay = document.getElementById("img-lightbox");
    if (!overlay) {
      overlay = createEl("div", "img-lightbox", {
        id: "img-lightbox",
        role: "dialog",
        "aria-modal": "true",
        "aria-label": "图片放大预览",
        hidden: ""
      });

      var toolbar = createEl("div", "img-lightbox__toolbar");
      var counter = createEl("span", "img-lightbox__counter");
      var controls = createEl("div", "img-lightbox__controls");

      var btnZoomOut = createEl("button", "img-lightbox__btn", {
        type: "button",
        "aria-label": "缩小",
        title: "缩小 (-)",
        text: "−"
      });
      var btnReset = createEl("button", "img-lightbox__btn img-lightbox__btn--pct", {
        type: "button",
        "aria-label": "重置缩放",
        title: "重置缩放 (0)",
        text: "100%"
      });
      var btnZoomIn = createEl("button", "img-lightbox__btn", {
        type: "button",
        "aria-label": "放大",
        title: "放大 (+)",
        text: "+"
      });
      var linkOpen = createEl("a", "img-lightbox__btn img-lightbox__btn--link", {
        target: "_blank",
        rel: "noopener noreferrer",
        title: "在新标签页打开原图",
        text: "原图 ↗"
      });
      var btnClose = createEl("button", "img-lightbox__btn img-lightbox__btn--close", {
        type: "button",
        "aria-label": "关闭预览",
        title: "关闭 (Esc)",
        text: "×"
      });

      controls.appendChild(btnZoomOut);
      controls.appendChild(btnReset);
      controls.appendChild(btnZoomIn);
      controls.appendChild(linkOpen);
      controls.appendChild(btnClose);
      toolbar.appendChild(counter);
      toolbar.appendChild(controls);

      var stage = createEl("div", "img-lightbox__stage");
      var btnPrev = createEl("button", "img-lightbox__nav img-lightbox__nav--prev", {
        type: "button",
        "aria-label": "上一张",
        title: "上一张 (←)",
        text: "‹"
      });
      var btnNext = createEl("button", "img-lightbox__nav img-lightbox__nav--next", {
        type: "button",
        "aria-label": "下一张",
        title: "下一张 (→)",
        text: "›"
      });

      var viewport = createEl("div", "img-lightbox__viewport");
      var modalImg = createEl("img", "img-lightbox__img", { alt: "", draggable: "false" });
      viewport.appendChild(modalImg);

      stage.appendChild(btnPrev);
      stage.appendChild(viewport);
      stage.appendChild(btnNext);

      var caption = createEl("div", "img-lightbox__caption");

      overlay.appendChild(toolbar);
      overlay.appendChild(stage);
      overlay.appendChild(caption);
      document.body.appendChild(overlay);
    }

    var counterEl = overlay.querySelector(".img-lightbox__counter");
    var btnZoomOutEl = overlay.querySelector('[aria-label="缩小"]');
    var btnResetEl = overlay.querySelector('[aria-label="重置缩放"]');
    var btnZoomInEl = overlay.querySelector('[aria-label="放大"]');
    var linkOpenEl = overlay.querySelector(".img-lightbox__btn--link");
    var btnCloseEl = overlay.querySelector('[aria-label="关闭预览"]');
    var btnPrevEl = overlay.querySelector(".img-lightbox__nav--prev");
    var btnNextEl = overlay.querySelector(".img-lightbox__nav--next");
    var stageEl = overlay.querySelector(".img-lightbox__stage");
    var viewportEl = overlay.querySelector(".img-lightbox__viewport");
    var modalImgEl = overlay.querySelector(".img-lightbox__img");
    var captionEl = overlay.querySelector(".img-lightbox__caption");

    var currentIndex = 0;
    var currentAspect = 16 / 9;
    var scale = 1;
    var panX = 0;
    var panY = 0;
    var dragging = false;
    var dragStartX = 0;
    var dragStartY = 0;
    var lastTrigger = null;

    function fitImageBox() {
      var ar = currentAspect > 0 ? currentAspect : 16 / 9;
      var availW = Math.min(window.innerWidth * 0.9, 1400);
      var availH = window.innerHeight * 0.72;
      var fitW = Math.min(availW, availH * ar);
      var fitH = fitW / ar;
      modalImgEl.style.width = Math.round(fitW) + "px";
      modalImgEl.style.height = Math.round(fitH) + "px";
    }

    function clampScale(s) {
      return Math.min(MAX_SCALE, Math.max(MIN_SCALE, Math.round(s * 100) / 100));
    }

    function applyTransform() {
      modalImgEl.style.transform =
        "translate3d(" + panX + "px, " + panY + "px, 0) scale(" + scale + ")";
      btnResetEl.textContent = Math.round(scale * 100) + "%";
      viewportEl.classList.toggle("is-zoomed", scale > 1.01);
    }

    function resetView() {
      scale = 1;
      panX = 0;
      panY = 0;
      applyTransform();
    }

    function zoomBy(delta) {
      var next = clampScale(scale + delta);
      if (next <= 1) {
        panX = 0;
        panY = 0;
      }
      scale = next;
      applyTransform();
    }

    function showImage(index) {
      currentIndex = (index + images.length) % images.length;
      var srcImg = images[currentIndex];
      var src = srcImg.currentSrc || srcImg.src;
      var rect = srcImg.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        currentAspect = rect.width / rect.height;
      } else if (srcImg.naturalWidth > 0 && srcImg.naturalHeight > 0) {
        currentAspect = srcImg.naturalWidth / srcImg.naturalHeight;
      } else {
        currentAspect = 16 / 9;
      }
      fitImageBox();
      modalImgEl.src = src;
      modalImgEl.alt = srcImg.alt || "";
      linkOpenEl.href = src;

      var figure = srcImg.closest("figure");
      var figcap = figure ? figure.querySelector("figcaption") : null;
      if (figcap) {
        captionEl.innerHTML = figcap.innerHTML;
        captionEl.hidden = false;
      } else if (srcImg.alt) {
        captionEl.textContent = srcImg.alt;
        captionEl.hidden = false;
      } else {
        captionEl.textContent = "";
        captionEl.hidden = true;
      }

      var multi = images.length > 1;
      btnPrevEl.hidden = !multi;
      btnNextEl.hidden = !multi;
      counterEl.textContent = multi
        ? (currentIndex + 1) + " / " + images.length + " · 滚轮缩放，拖拽平移"
        : "滚轮或按钮缩放 · 双击切换放大";

      resetView();
    }

    function openLightbox(index, triggerEl) {
      lastTrigger = triggerEl || null;
      showImage(index);
      overlay.hidden = false;
      document.documentElement.classList.add("img-lightbox-open");
      window.requestAnimationFrame(function () {
        overlay.classList.add("is-visible");
        btnCloseEl.focus();
      });
    }

    function closeLightbox() {
      if (overlay.hidden) return;
      overlay.classList.remove("is-visible");
      document.documentElement.classList.remove("img-lightbox-open");
      window.setTimeout(function () {
        overlay.hidden = true;
        modalImgEl.removeAttribute("src");
        if (lastTrigger && typeof lastTrigger.focus === "function") {
          lastTrigger.focus();
        }
      }, 150);
    }

    // Attach zoom triggers to all eligible images
    images.forEach(function (img, idx) {
      img.classList.add("zoomable-img");
      img.setAttribute("tabindex", "0");
      img.setAttribute("role", "button");
      img.setAttribute("aria-label", (img.alt ? img.alt + " — " : "") + "点击放大查看");

      var fig = img.closest(".book-figure");
      if (fig && !fig.querySelector(".book-figure__zoom-hint")) {
        var hint = createEl("span", "book-figure__zoom-hint", {
          "aria-hidden": "true",
          text: "点击放大"
        });
        fig.appendChild(hint);
      }

      img.addEventListener("click", function () {
        openLightbox(idx, img);
      });
      img.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openLightbox(idx, img);
        }
      });
    });

    // Bind overlay events once
    if (overlay.dataset.bound) return;
    overlay.dataset.bound = "1";

    btnCloseEl.addEventListener("click", closeLightbox);
    btnZoomInEl.addEventListener("click", function () { zoomBy(STEP_SCALE); });
    btnZoomOutEl.addEventListener("click", function () { zoomBy(-STEP_SCALE); });
    btnResetEl.addEventListener("click", resetView);
    btnPrevEl.addEventListener("click", function () { showImage(currentIndex - 1); });
    btnNextEl.addEventListener("click", function () { showImage(currentIndex + 1); });

    // Close when clicking empty backdrop around image
    stageEl.addEventListener("click", function (e) {
      if (e.target === stageEl || e.target === viewportEl) {
        closeLightbox();
      }
    });

    // Double-click image to toggle 1x <-> 1.8x
    modalImgEl.addEventListener("dblclick", function (e) {
      e.preventDefault();
      if (scale > 1.05) {
        resetView();
      } else {
        scale = 1.8;
        panX = 0;
        panY = 0;
        applyTransform();
      }
    });

    // Mouse wheel zoom
    stageEl.addEventListener(
      "wheel",
      function (e) {
        if (overlay.hidden) return;
        e.preventDefault();
        var delta = e.deltaY < 0 ? STEP_SCALE : -STEP_SCALE;
        zoomBy(delta);
      },
      { passive: false }
    );

    // Pointer drag to pan when zoomed
    modalImgEl.addEventListener("pointerdown", function (e) {
      if (scale <= 1.01) return;
      dragging = true;
      dragStartX = e.clientX - panX;
      dragStartY = e.clientY - panY;
      modalImgEl.classList.add("is-dragging");
      modalImgEl.setPointerCapture(e.pointerId);
      e.preventDefault();
    });

    modalImgEl.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      panX = e.clientX - dragStartX;
      panY = e.clientY - dragStartY;
      applyTransform();
    });

    function endDrag(e) {
      if (!dragging) return;
      dragging = false;
      modalImgEl.classList.remove("is-dragging");
      try {
        modalImgEl.releasePointerCapture(e.pointerId);
      } catch (_) {}
    }

    modalImgEl.addEventListener("pointerup", endDrag);
    modalImgEl.addEventListener("pointercancel", endDrag);

    window.addEventListener("resize", function () {
      if (!overlay.hidden) fitImageBox();
    });

    modalImgEl.addEventListener("load", function () {
      if (modalImgEl.naturalWidth > 0 && modalImgEl.naturalHeight > 0) {
        currentAspect = modalImgEl.naturalWidth / modalImgEl.naturalHeight;
        fitImageBox();
      }
    });

    // Keyboard shortcuts while open
    document.addEventListener("keydown", function (e) {
      if (overlay.hidden) return;
      if (e.key === "Escape") {
        e.preventDefault();
        closeLightbox();
      } else if (e.key === "ArrowLeft" && images.length > 1) {
        e.preventDefault();
        showImage(currentIndex - 1);
      } else if (e.key === "ArrowRight" && images.length > 1) {
        e.preventDefault();
        showImage(currentIndex + 1);
      } else if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        zoomBy(STEP_SCALE);
      } else if (e.key === "-" || e.key === "_") {
        e.preventDefault();
        zoomBy(-STEP_SCALE);
      } else if (e.key === "0") {
        e.preventDefault();
        resetView();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLightbox);
  } else {
    initLightbox();
  }
})();
