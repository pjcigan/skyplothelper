/* Priority+ navbar.
 *
 * pydata-sphinx-theme splits the header links into a "More" dropdown only once,
 * at build time (by a fixed count), and otherwise lets the links overflow until
 * the whole header collapses to the hamburger drawer at its breakpoint. This
 * adds the intermediate stages: as the window narrows, trailing section links
 * are moved into a "More" dropdown one at a time to fit the available width
 * (and restored when it widens). The search/tools group stays visible the
 * whole time. Below the theme's hamburger breakpoint the header nav is hidden
 * by the theme, so this script stands down and lets the drawer show everything.
 */
(function () {
  "use strict";

  function ready(fn) {
    if (document.readyState !== "loading") {
      fn();
    } else {
      document.addEventListener("DOMContentLoaded", fn);
    }
  }

  ready(function () {
    var nav = document.querySelector(".bd-header .bd-navbar-elements.navbar-nav");
    var wrap = document.querySelector(".bd-header .navbar-header-items");
    var center = document.querySelector(".bd-header .navbar-header-items__center");
    var end = document.querySelector(".bd-header .navbar-header-items__end");
    if (!nav || !wrap || !center || !end) {
      return;
    }

    // Collapse a link once the links group comes within this many pixels of the
    // tools group (search / theme / GitHub / PyPI). Higher = collapses sooner.
    var MIN_GAP = 24;

    // Build the (initially hidden) "More" dropdown at the end of the nav.
    var moreLi = document.createElement("li");
    moreLi.className = "nav-item dropdown pst-nav-more";
    moreLi.style.display = "none";
    moreLi.innerHTML =
      '<button class="nav-link dropdown-toggle" type="button" ' +
      'data-bs-toggle="dropdown" aria-expanded="false">More</button>' +
      '<ul class="dropdown-menu dropdown-menu-end"></ul>';
    nav.appendChild(moreLi);
    var moreMenu = moreLi.querySelector(".dropdown-menu");

    // Original top-level link items, in order (excludes the More dropdown).
    var items = Array.prototype.slice.call(nav.children).filter(function (li) {
      return li !== moreLi;
    });

    function resetAll() {
      items.forEach(function (li) {
        li.style.display = "";
      });
      moreMenu.innerHTML = "";
      moreLi.style.display = "none";
    }

    function overflowing() {
      // The header grows with its content (overflow spills to the page rather
      // than being clipped), so a scrollWidth check never fires. Instead, watch
      // the slack between the right edge of the links group and the left edge of
      // the tools group: when it shrinks below MIN_GAP we're about to collide.
      var gap = end.getBoundingClientRect().left - center.getBoundingClientRect().right;
      return gap < MIN_GAP;
    }

    function navIsHidden() {
      // The theme hides the whole header-items group at its hamburger
      // breakpoint; offsetParent is null when an ancestor is display:none.
      return wrap.offsetParent === null ||
        getComputedStyle(wrap).display === "none";
    }

    function reflow() {
      resetAll();
      if (navIsHidden() || !overflowing()) {
        return;
      }
      moreLi.style.display = "";
      // Move trailing links into "More" until the bar fits (keep >= 1 inline).
      for (var i = items.length - 1; i >= 1; i--) {
        if (!overflowing()) {
          break;
        }
        var li = items[i];
        if (li.style.display === "none") {
          continue;
        }
        var src = li.querySelector("a");
        if (!src) {
          continue;
        }
        li.style.display = "none";
        var entry = document.createElement("li");
        var link = document.createElement("a");
        link.className = "dropdown-item" +
          (src.classList.contains("active") ? " active" : "");
        link.href = src.getAttribute("href");
        link.textContent = src.textContent.trim();
        entry.appendChild(link);
        moreMenu.insertBefore(entry, moreMenu.firstChild);
      }
      if (!moreMenu.children.length) {
        moreLi.style.display = "none";
      }
    }

    var timer;
    function schedule() {
      window.clearTimeout(timer);
      timer = window.setTimeout(reflow, 100);
    }

    reflow();
    window.addEventListener("resize", schedule);
    // Re-measure once webfonts have loaded (label widths change).
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(reflow);
    }
  });
})();
