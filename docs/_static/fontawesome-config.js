/* Stop FontAwesome from re-scanning the document on every DOM mutation.
 *
 * pydata-sphinx-theme bundles FontAwesome in SVG-with-JS mode, which by default
 * installs a document-wide MutationObserver and re-runs querySelectorAll over
 * the whole document on every change, looking for <i class="fa-..."> to convert
 * to <svg>. Pages that build a large DOM *after* load go quadratic: the plotly
 * tutorial (~25 figures) spent 227 s, 79% of it inside querySelectorAll called
 * from FontAwesome's i2svg. With the observer off it loads in 4 s and renders
 * identically.
 *
 * Icons are still converted by FontAwesome's initial pass at load; we only stop
 * it watching for later mutations. That is safe here because every icon in this
 * site's chrome (search, navbar, theme toggle, breadcrumbs, "On this page")
 * exists in the initial HTML. The theme's only runtime-created icons are the
 * two banner close buttons (announcement + version-warning), which are
 * ReadTheDocs-only / unused here — see conf.py.
 *
 * Load order: this must run before fontawesome.js reads window.FontAwesomeConfig.
 * The theme emits fontawesome.js from a Jinja macro (head_js_preload() in
 * layout.html) that renders ahead of *all* Sphinx-managed scripts, so no
 * html_js_files priority can put this tag first — and it doesn't need to.
 * fontawesome.js is `defer`, so it executes only after the document is parsed,
 * whereas this file is a classic script that executes during parsing. Parsing
 * always wins, from anywhere in the page.
 */
window.FontAwesomeConfig = { observeMutations: false };
