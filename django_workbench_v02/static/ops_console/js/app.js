// PFD Toolkit · Ops Console — shared JS (UI-only)
// - inline expand rows (run review)
// - chip add/remove (search query / area / receiver tag editors)
// - mobile nav drawer

(function () {
  // ── expand rows (run review) ──
  document.querySelectorAll("[data-row-toggle]").forEach((row) => {
    row.addEventListener("click", (e) => {
      if (e.target.closest("a, button, input, select, textarea, .b")) return;
      const target = document.querySelector(row.dataset.rowToggle);
      if (!target) return;
      const open = row.classList.toggle("expanded");
      target.style.display = open ? "table-row" : "none";
    });
  });
  document.querySelectorAll("[data-row-open]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const row = btn.closest("[data-row-toggle]");
      if (!row) return;
      const target = document.querySelector(btn.dataset.rowOpen || row.dataset.rowToggle || "");
      if (!target) return;
      const open = row.classList.toggle("expanded");
      target.style.display = open ? "table-row" : "none";
    });
  });

  // ── chip add/remove ──
  document.querySelectorAll("[data-chips]").forEach((root) => {
    root.addEventListener("click", (e) => {
      const rm = e.target.closest("[data-chip-remove]");
      if (rm) { rm.closest(".chip").remove(); }
      const add = e.target.closest("[data-chip-add]");
      if (add) {
        const v = window.prompt("Add value");
        if (!v) return;
        const c = document.createElement("span");
        c.className = "chip";
        c.dataset.val = v;
        c.innerHTML = `${v} <button data-chip-remove aria-label="Remove">×</button>`;
        root.insertBefore(c, add);
      }
    });
  });

  // ── mobile nav drawer ──
  const menuBtn = document.querySelector("[data-menu-toggle]");
  const scrim = document.querySelector(".scrim");
  function closeNav() { document.body.classList.remove("nav-open"); }
  menuBtn && menuBtn.addEventListener("click", () => {
    document.body.classList.toggle("nav-open");
  });
  scrim && scrim.addEventListener("click", closeNav);
  document.querySelectorAll(".sidebar a, .mobile-tabbar a").forEach((a) => {
    a.addEventListener("click", closeNav);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeNav();
  });

  // ── method-gated config blocks (run review) ──
  document.querySelectorAll("[data-method]").forEach((cb) => {
    const sync = () => {
      const method = cb.dataset.method;
      const form = cb.closest("form") || document;
      const block = form.querySelector(`[data-method-block="${method}"]`);
      if (!block) return;
      block.hidden = !cb.checked;
      const card = cb.closest(".method-card");
      if (card) card.classList.toggle("is-active", cb.checked);
    };
    cb.addEventListener("change", sync);
    sync();
  });

  // ── schema-table: add/remove feature-field rows ──
  document.querySelectorAll(".schema-table").forEach((tbl) => {
    const addBtn = tbl.querySelector(".schema-add");
    addBtn && addBtn.addEventListener("click", () => {
      const row = document.createElement("div");
      row.className = "schema-row";
      row.setAttribute("role", "row");
      row.innerHTML = `
        <span role="cell" data-l="Field name"><input class="input mono" name="feature_field_name" placeholder="field_name" /></span>
        <span role="cell" data-l="Description"><input class="input" name="feature_field_description" placeholder="What this field captures" /></span>
        <span role="cell" data-l="Type"><select class="select" name="feature_field_type"><option value="decimal">number</option><option value="text">text</option><option value="boolean">True/False</option></select></span>
        <span role="cell" data-l="Required" class="schema-c-req"><label class="check tight"><input type="checkbox" /></label></span>
        <span role="cell" class="schema-c-x"><button class="iconbtn" type="button" aria-label="Remove">×</button></span>`;
      tbl.insertBefore(row, addBtn);
      row.querySelector("input").focus();
    });
    tbl.addEventListener("click", (e) => {
      const rm = e.target.closest(".iconbtn");
      if (!rm) return;
      const row = rm.closest(".schema-row");
      if (row) row.remove();
    });
  });

  // ── tabs (within a panel: data-tab-group / data-tab-target) ──
  document.querySelectorAll("[data-tab-group]").forEach((group) => {
    const buttons = group.querySelectorAll("[data-tab-target]");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const tgt = btn.dataset.tabTarget;
        buttons.forEach((b) => b.classList.toggle("is-active", b === btn));
        const scope = document.querySelector(group.dataset.tabGroup);
        if (!scope) return;
        scope.querySelectorAll("[data-tab-pane]").forEach((p) => {
          p.style.display = p.dataset.tabPane === tgt ? "" : "none";
        });
      });
    });
  });
})();
