// Flips every [data-hub-link] between "open" and "closed" by asking the hub
// whether it is running. Kept as its own file because index.html and lab.html
// are generated artifacts -- if either is regenerated, only the <script> tag
// needs re-adding, not this logic.
(function () {
  var HUB = "https://tobi.tail810f4b.ts.net";
  var POLL_MS = 30000;

  function apply(state) {
    var links = document.querySelectorAll("[data-hub-link]");
    for (var i = 0; i < links.length; i++) {
      var el = links[i];
      el.setAttribute("data-hub-state", state);
      if (state === "open") {
        el.setAttribute("href", HUB);
        el.style.pointerEvents = "";
        el.style.opacity = "";
      } else {
        el.removeAttribute("href");
        el.style.pointerEvents = "none";
        el.style.opacity = "0.45";
      }
    }
  }

  function check() {
    var done = false;
    var timer = setTimeout(function () {
      if (!done) { done = true; apply("closed"); }
    }, 3000);

    fetch(HUB + "/healthz", { cache: "no-store" })
      .then(function (r) {
        if (done) return;
        done = true; clearTimeout(timer);
        apply(r.ok ? "open" : "closed");
      })
      .catch(function () {
        if (done) return;
        done = true; clearTimeout(timer);
        apply("closed");
      });
  }

  apply("closed");  // fail safe: assume closed until proven open
  check();
  setInterval(check, POLL_MS);
})();
