(function () {
  var syncing = false;

  document.addEventListener(
    "scroll",
    function (e) {
      if (!e.target.classList || !e.target.classList.contains("voting-list"))
        return;
      if (syncing) return;
      syncing = true;
      var scrollTop = e.target.scrollTop;
      document.querySelectorAll(".voting-list").forEach(function (el) {
        if (el !== e.target) el.scrollTop = scrollTop;
      });
      syncing = false;
    },
    true
  );
})();
