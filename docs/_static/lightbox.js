/* Click-to-zoom lightbox for gallery figures.
 *
 * Clicking a gallery image (``.sph-plot``) that is NOT already wrapped in a
 * link opens it full-size in an in-page overlay; clicking the backdrop or
 * pressing Escape closes it. Link-wrapped thumbnails (e.g. the plot-types
 * index, whose figures navigate to a detail page) are left untouched.
 *
 * Self-contained — no external lightbox dependency, matching the project's
 * other small _static scripts. Pairs with the ``.sph-lightbox`` rules in
 * custom.css.
 */
(function () {
  "use strict";

  function init() {
    var overlay = document.createElement("div");
    overlay.className = "sph-lightbox";
    overlay.setAttribute("aria-hidden", "true");
    var big = document.createElement("img");
    big.className = "sph-lightbox-img";
    overlay.appendChild(big);
    document.body.appendChild(overlay);

    function close() {
      overlay.classList.remove("is-open");
      overlay.setAttribute("aria-hidden", "true");
      big.removeAttribute("src");
    }
    function open(src, alt) {
      big.src = src;
      big.alt = alt || "";
      overlay.classList.add("is-open");
      overlay.setAttribute("aria-hidden", "false");
    }

    overlay.addEventListener("click", close);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") close();
    });

    var imgs = document.querySelectorAll("img.sph-plot");
    Array.prototype.forEach.call(imgs, function (el) {
      if (el.closest("a")) return; // keep link-wrapped thumbnails navigating
      el.classList.add("sph-zoomable");
      el.addEventListener("click", function () {
        // currentSrc respects the visible light/dark variant.
        open(el.currentSrc || el.src, el.alt);
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
