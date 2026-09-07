/* ============================================================================
   Policy Badger - Raymond — shared shell script
   Icons, rail rendering, toasts, modals, small helpers.
   Loaded by every page. No dependencies.
   ========================================================================== */
(function (global) {
  "use strict";

  /* ── Icons ──────────────────────────────────────────────────────────────
     Every icon is a 24x24 stroke path rendered at the size the caller asks
     for, so weights stay consistent across the whole app.                   */
  function svg(paths, size, extra) {
    return (
      '<svg width="' + (size || 16) + '" height="' + (size || 16) + '" ' +
      'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" ' +
      'stroke-linecap="round" stroke-linejoin="round" ' + (extra || "") + ">" +
      paths + "</svg>"
    );
  }

  var Icons = {
    plus: function (s) { return svg('<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>', s); },
    search: function (s) { return svg('<circle cx="11" cy="11" r="7"/><line x1="20" y1="20" x2="16.7" y2="16.7"/>', s); },
    home: function (s) { return svg('<path d="M3 10.2 12 3l9 7.2V20a1.6 1.6 0 0 1-1.6 1.6H4.6A1.6 1.6 0 0 1 3 20z"/>', s); },
    chat: function (s) { return svg('<path d="M21 11.6a8 8 0 0 1-11.6 7.1L3 21l2.3-6.3A8 8 0 1 1 21 11.6z"/>', s); },
    folder: function (s) { return svg('<path d="M3 7.4a1.6 1.6 0 0 1 1.6-1.6h4l2 2.4h7.8A1.6 1.6 0 0 1 20 9.8v8.6a1.6 1.6 0 0 1-1.6 1.6H4.6A1.6 1.6 0 0 1 3 18.4z"/>', s); },
    settings: function (s) { return svg('<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.5 1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z"/>', s); },
    sparkle: function (s) { return svg('<path d="M12 3.2 13.7 8l4.8 1.7-4.8 1.7L12 16.2 10.3 11.4 5.5 9.7l4.8-1.7z"/><path d="M18.6 16.2l.7 1.9 1.9.7-1.9.7-.7 1.9-.7-1.9-1.9-.7 1.9-.7z"/>', s); },
    dots: function (s) { return svg('<circle cx="12" cy="5" r="1.4" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/><circle cx="12" cy="19" r="1.4" fill="currentColor" stroke="none"/>', s); },
    dotsH: function (s) { return svg('<circle cx="5" cy="12" r="1.5" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none"/><circle cx="19" cy="12" r="1.5" fill="currentColor" stroke="none"/>', s); },
    caret: function (s) { return svg('<polyline points="6 9 12 15 18 9"/>', s); },
    chevRight: function (s) { return svg('<polyline points="9 6 15 12 9 18"/>', s); },
    arrowRight: function (s) { return svg('<line x1="4" y1="12" x2="19" y2="12"/><polyline points="13 6 19 12 13 18"/>', s); },
    arrowUpRight: function (s) { return svg('<line x1="7" y1="17" x2="17" y2="7"/><polyline points="8 7 17 7 17 16"/>', s); },
    mic: function (s) { return svg('<rect x="9" y="2.6" width="6" height="11" rx="3"/><path d="M5.5 11.4a6.5 6.5 0 0 0 13 0"/><line x1="12" y1="18" x2="12" y2="21.4"/>', s); },
    paperclip: function (s) { return svg('<path d="M20.4 11.2 12.5 19a5 5 0 0 1-7.1-7.1l8-8a3.3 3.3 0 1 1 4.7 4.7l-8 8a1.7 1.7 0 0 1-2.4-2.4l7.4-7.3"/>', s); },
    copy: function (s) { return svg('<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1.6 1.6 0 0 1-1.6-1.6V4.6A1.6 1.6 0 0 1 4 3h8.8A1.6 1.6 0 0 1 14.4 4.6V5"/>', s); },
    refresh: function (s) { return svg('<path d="M20.5 12a8.5 8.5 0 1 1-2.5-6"/><polyline points="20.5 3.4 20.5 8.4 15.5 8.4"/>', s); },
    speaker: function (s) { return svg('<path d="M11 5 6.5 8.8H3.4v6.4h3.1L11 19z"/><path d="M15.6 9a4 4 0 0 1 0 6"/><path d="M18.2 6.4a7.6 7.6 0 0 1 0 11.2"/>', s); },
    thumbDown: function (s) { return svg('<path d="M10 15v4a2.5 2.5 0 0 0 2.5 2.5L15.5 15h4.1A1.9 1.9 0 0 0 21.5 13l-1-7A1.9 1.9 0 0 0 18.6 4.4H8.5a1.9 1.9 0 0 0-1.9 1.6"/><rect x="2.5" y="4.4" width="4.1" height="10.6" rx="1.3"/>', s); },
    check: function (s) { return svg('<polyline points="20 6.5 9.4 17.5 4 12.2"/>', s); },
    checkDouble: function (s) { return svg('<polyline points="1.5 12.5 6 17 14.5 8"/><polyline points="9 12.5 12 15.5 20.5 6.5"/>', s); },
    x: function (s) { return svg('<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>', s); },
    users: function (s) { return svg('<path d="M16.5 20.5v-1.8a3.6 3.6 0 0 0-3.6-3.6H6.1a3.6 3.6 0 0 0-3.6 3.6v1.8"/><circle cx="9.5" cy="7.8" r="3.6"/><path d="M21.5 20.5v-1.8a3.6 3.6 0 0 0-2.7-3.5"/><path d="M15.3 4.1a3.6 3.6 0 0 1 0 7"/>', s); },
    doc: function (s) { return svg('<path d="M14 2.6H6.6A1.7 1.7 0 0 0 5 4.3v15.4a1.7 1.7 0 0 0 1.6 1.7h10.8a1.7 1.7 0 0 0 1.6-1.7V7.4z"/><polyline points="14 2.6 14 7.6 19 7.6"/><line x1="8.6" y1="13" x2="15.4" y2="13"/><line x1="8.6" y1="16.6" x2="13" y2="16.6"/>', s); },
    layers: function (s) { return svg('<path d="m12 2.6 9.4 4.7L12 12 2.6 7.3z"/><polyline points="2.6 16.7 12 21.4 21.4 16.7"/><polyline points="2.6 12 12 16.7 21.4 12"/>', s); },
    shield: function (s) { return svg('<path d="M12 21.4s7.6-3.8 7.6-9.5V5.6L12 2.6 4.4 5.6v6.3C4.4 17.6 12 21.4 12 21.4z"/>', s); },
    logout: function (s) { return svg('<path d="M9.6 21.4H5.4a1.9 1.9 0 0 1-1.9-1.9V4.5a1.9 1.9 0 0 1 1.9-1.9h4.2"/><polyline points="16.2 16.8 21 12 16.2 7.2"/><line x1="21" y1="12" x2="9.6" y2="12"/>', s); },
    eye: function (s) { return svg('<path d="M1.6 12S5.8 4.4 12 4.4 22.4 12 22.4 12 18.2 19.6 12 19.6 1.6 12 1.6 12z"/><circle cx="12" cy="12" r="3.1"/>', s); },
    key: function (s) { return svg('<circle cx="7.6" cy="16.4" r="3.9"/><path d="m10.4 13.6 8.2-8.2"/><path d="m16.2 7.8 2.2 2.2"/><path d="m19 5 2.2 2.2"/>', s); },
    alert: function (s) { return svg('<circle cx="12" cy="12" r="9.4"/><line x1="12" y1="7.6" x2="12" y2="12.8"/><circle cx="12" cy="16.4" r=".9" fill="currentColor" stroke="none"/>', s); },
    menu: function (s) { return svg('<line x1="3.5" y1="7" x2="20.5" y2="7"/><line x1="3.5" y1="12" x2="20.5" y2="12"/><line x1="3.5" y1="17" x2="20.5" y2="17"/>', s); },
    trash: function (s) { return svg('<polyline points="3.5 6.4 20.5 6.4"/><path d="M18.4 6.4v13.2a1.9 1.9 0 0 1-1.9 1.9H7.5a1.9 1.9 0 0 1-1.9-1.9V6.4"/><path d="M8.6 6.4V4.5a1.9 1.9 0 0 1 1.9-1.9h3a1.9 1.9 0 0 1 1.9 1.9v1.9"/>', s); },
    edit: function (s) { return svg('<path d="M17 3.4a2.4 2.4 0 0 1 3.4 3.4L7.6 19.6l-4.5 1.1 1.1-4.5z"/>', s); },
    clock: function (s) { return svg('<circle cx="12" cy="12" r="9.4"/><polyline points="12 6.6 12 12 15.6 13.8"/>', s); }
  };

  /* The badger mark — a seven-dot cluster, echoing the reference design.
     Rendered in currentColor so it works on both the dark rail and light card. */
  function mark(size, color) {
    var s = size || 26;
    var c = color || "currentColor";
    var r = 2.55;
    var pts = [
      [12, 12], [12, 6.4], [16.85, 9.2], [16.85, 14.8],
      [12, 17.6], [7.15, 14.8], [7.15, 9.2]
    ];
    var circles = pts.map(function (p, i) {
      return '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="' + (i === 0 ? r * 0.86 : r) + '"/>';
    }).join("");
    return '<svg width="' + s + '" height="' + s + '" viewBox="0 0 24 24" fill="' + c + '">' + circles + "</svg>";
  }

  /* ── Helpers ────────────────────────────────────────────────────────────── */
  function escHtml(str) {
    return String(str == null ? "" : str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function el(id) { return document.getElementById(id); }

  function timeNow() {
    var d = new Date();
    return d.getHours().toString().padStart(2, "0") + ":" +
           d.getMinutes().toString().padStart(2, "0");
  }

  /* ── Toasts ─────────────────────────────────────────────────────────────── */
  function toast(message, type) {
    var host = el("toasts");
    if (!host) {
      host = document.createElement("div");
      host.id = "toasts";
      document.body.appendChild(host);
    }
    var node = document.createElement("div");
    node.className = "toast" + (type ? " t-" + type : "");
    node.textContent = message;
    host.appendChild(node);
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { node.classList.add("show"); });
    });
    setTimeout(function () {
      node.classList.remove("show");
      setTimeout(function () { node.remove(); }, 300);
    }, 3600);
  }

  /* ── Modal ──────────────────────────────────────────────────────────────── */
  function openModal(id) {
    var v = el(id);
    if (v) v.classList.add("show");
  }
  function closeModal(id) {
    var v = el(id);
    if (v) v.classList.remove("show");
  }
  // Clicking the veil (but not the modal itself) dismisses.
  document.addEventListener("click", function (e) {
    if (e.target.classList && e.target.classList.contains("modal-veil")) {
      e.target.classList.remove("show");
    }
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      document.querySelectorAll(".modal-veil.show").forEach(function (v) {
        v.classList.remove("show");
      });
    }
  });

  /* ── Rail ───────────────────────────────────────────────────────────────
     One definition of the sidebar, used by every page, so navigation and
     styling never drift between screens.

     opts.active   — "home" | "chats" | "studio" | "users" | "settings"
     opts.user     — { username, role } or null
     opts.folders  — [{ label, spine, href }]
     opts.threads  — [{ label, group, href, active }]
     opts.footer   — small print under the panel                            */
  function renderRail(opts) {
    opts = opts || {};
    var active = opts.active || "";
    var user = opts.user || null;
    var role = user && user.role ? user.role : null;
    var isAdmin = role === "super_admin" || role === "manager_admin";

    var nav = [
      { key: "home", label: "Home", icon: Icons.home, href: "/" },
      { key: "chats", label: "Chats", icon: Icons.chat, href: "/#chats" }
    ];
    if (isAdmin) {
      nav.push({ key: "studio", label: "Policy Studio", icon: Icons.layers, href: "/admin#studio" });
      nav.push({ key: "users", label: "Team Access", icon: Icons.users, href: "/admin#users" });
    }
    nav.push({ key: "settings", label: "Settings", icon: Icons.settings, href: "/settings" });

    var navHtml = nav.map(function (n) {
      return '<a class="rail-item' + (active === n.key ? " active" : "") + '" href="' + n.href + '">' +
             n.icon(17) + "<span>" + escHtml(n.label) + "</span></a>";
    }).join("");

    var folders = opts.folders || [];
    var foldersHtml = folders.map(function (f) {
      return '<a class="rail-folder" href="' + (f.href || "#") + '" style="--spine:' + (f.spine || "var(--mint)") + '">' +
             Icons.folder(15) +
             '<span class="row-label">' + escHtml(f.label) + "</span>" +
             '<span class="row-more" role="button" aria-label="More">' + Icons.dotsH(14) + "</span>" +
             "</a>";
    }).join("");

    var threads = opts.threads || [];
    var threadsHtml = "";
    var lastGroup = null;
    threads.forEach(function (t) {
      if (t.group && t.group !== lastGroup) {
        threadsHtml += '<div class="rail-date">' + escHtml(t.group) + "</div>";
        lastGroup = t.group;
      }
      threadsHtml +=
        '<button class="rail-thread' + (t.active ? " active" : "") + '" data-thread="' + escHtml(t.id || "") + '">' +
        Icons.chat(15) +
        '<span class="row-label">' + escHtml(t.label) + "</span>" +
        '<span class="row-more" role="button" aria-label="More">' + Icons.dotsH(14) + "</span>" +
        "</button>";
    });

    return '' +
      '<aside class="rail" id="rail">' +
        '<div class="rail-head">' +
          '<span class="rail-mark" style="color:var(--on-dark)">' + mark(27) + "</span>" +
          '<button class="icon-btn" id="rail-menu" aria-label="Rail menu">' + Icons.dotsH(17) + "</button>" +
        "</div>" +

        '<div class="rail-panel">' +
          '<button class="rail-item boxed" id="new-chat-btn">' + Icons.plus(17) + "<span>New Chat</span></button>" +
          '<button class="rail-item boxed" id="rail-search-btn">' + Icons.search(17) + "<span>Search</span></button>" +

          '<div class="rail-divider"></div>' +
          navHtml +

          (folders.length
            ? '<div class="rail-group" data-group="folders">' +
                '<span class="caret">' + Icons.caret(14) + "</span>" +
                '<span class="grp-label">Policy Areas</span>' +
                '<span class="grp-actions">' +
                  '<span class="grp-btn">' + Icons.plus(13) + "</span>" +
                  '<span class="grp-btn">' + Icons.dots(13) + "</span>" +
                "</span>" +
              "</div>" +
              '<div class="rail-collapse" data-for="folders">' + foldersHtml + "</div>"
            : "") +

          '<div class="rail-group" data-group="threads">' +
            '<span class="caret">' + Icons.caret(14) + "</span>" +
            '<span class="grp-label">Chats</span>' +
            '<span class="grp-actions">' +
              '<span class="grp-btn">' + Icons.plus(13) + "</span>" +
              '<span class="grp-btn">' + Icons.dots(13) + "</span>" +
            "</span>" +
          "</div>" +
          '<div class="rail-list scroll-y rail-collapse" data-for="threads" id="rail-threads">' +
            (threadsHtml || '<div class="rail-date">No conversations yet</div>') +
          "</div>" +

          '<div class="rail-foot">' + escHtml(opts.footer || "Answers come from your company's published policies. Confirm critical decisions with HR.") + "</div>" +
        "</div>" +
      "</aside>";
  }

  /* Wire the collapsible groups and the mobile rail toggle. Call once after
     the rail markup is in the DOM. */
  function bindRail() {
    document.querySelectorAll(".rail-group").forEach(function (g) {
      g.addEventListener("click", function (e) {
        if (e.target.closest(".grp-actions")) return;
        var key = g.getAttribute("data-group");
        var body = document.querySelector('.rail-collapse[data-for="' + key + '"]');
        g.classList.toggle("collapsed");
        if (body) body.classList.toggle("hidden");
      });
    });

    var toggle = el("rail-toggle");
    if (toggle) {
      toggle.addEventListener("click", function () {
        var r = el("rail");
        if (r) r.classList.toggle("open");
      });
    }
  }

  /* ── Session ────────────────────────────────────────────────────────────── */
  var _me = null;
  async function me() {
    if (_me) return _me;
    try {
      var res = await fetch("/api/auth/me");
      if (!res.ok) return null;
      _me = await res.json();
      return _me;
    } catch (e) {
      return null;
    }
  }

  function roleLabel(role) {
    return role === "super_admin" ? "Super Admin"
         : role === "manager_admin" ? "Manager Admin"
         : "Employee";
  }

  /* ── Auto-growing textarea ──────────────────────────────────────────────── */
  function autoGrow(textarea, max) {
    if (!textarea) return;
    var cap = max || 168;
    function fit() {
      textarea.style.height = "auto";
      textarea.style.height = Math.min(textarea.scrollHeight, cap) + "px";
    }
    textarea.addEventListener("input", fit);
    fit();
  }

  global.Badger = {
    Icons: Icons,
    mark: mark,
    svg: svg,
    escHtml: escHtml,
    el: el,
    timeNow: timeNow,
    toast: toast,
    openModal: openModal,
    closeModal: closeModal,
    renderRail: renderRail,
    bindRail: bindRail,
    me: me,
    roleLabel: roleLabel,
    autoGrow: autoGrow
  };
})(window);
