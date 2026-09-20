/* FastArt interactive focus, tooltips, and attention arcs */

(function () {
  let tip;

  function ensureTip() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "fa-tip";
      tip.setAttribute("role", "tooltip");
      document.body.appendChild(tip);
    }
    return tip;
  }

  function weightsMatrix() {
    const el = document.getElementById("attn-data");
    if (!el) return null;
    try {
      return JSON.parse(el.textContent);
    } catch {
      return null;
    }
  }

  function tokens() {
    return Array.from(document.querySelectorAll("#explainer .tok")).map((t) =>
      t.textContent.trim()
    );
  }

  function showTip(html, x, y) {
    const t = ensureTip();
    t.innerHTML = html;
    t.classList.add("show");
    const pad = 12;
    const tw = t.offsetWidth || 120;
    const th = t.offsetHeight || 28;
    let left = x + pad;
    let top = y - th - 8;
    if (left + tw > window.innerWidth - 8) left = x - tw - pad;
    if (top < 8) top = y + pad;
    t.style.left = left + "px";
    t.style.top = top + "px";
  }

  function hideTip() {
    if (tip) tip.classList.remove("show");
  }

  function updateArcs(focus) {
    const svg = document.getElementById("attn-arcs");
    const data = weightsMatrix();
    const toks = tokens();
    if (!svg || !data || !toks.length) return;

    const n = toks.length;
    const W = Math.max(220, Math.min(280, 40 + n * 18));
    const rowH = 22;
    const top = 18;
    const H = top + n * rowH + 12;
    const leftX = 8;
    const rightX = W - 8;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", String(W));
    svg.setAttribute("height", String(H));

    const row = data[focus] || [];
    let maxW = 0;
    for (let j = 0; j < n; j++) maxW = Math.max(maxW, row[j] || 0);
    if (maxW <= 0) maxW = 1;

    const parts = [];
    parts.push(
      `<text x="${W / 2}" y="12" text-anchor="middle" class="tok-node" style="font-size:10px;fill:#7a756c">attention links</text>`
    );

    for (let j = 0; j < n; j++) {
      const y0 = top + focus * rowH + rowH / 2;
      const y1 = top + j * rowH + rowH / 2;
      const w = row[j] || 0;
      const t = w / maxW;
      const stroke = 0.6 + t * 4.5;
      const opacity = 0.12 + t * 0.85;
      const bulge = 28 + Math.abs(j - focus) * 6;
      const midX = (leftX + rightX) / 2 + bulge;
      const d = `M ${leftX} ${y0} C ${midX} ${y0}, ${midX} ${y1}, ${rightX} ${y1}`;
      parts.push(
        `<path class="arc" d="${d}" stroke-width="${stroke.toFixed(2)}" opacity="${opacity.toFixed(3)}" />`
      );
    }

    for (let i = 0; i < n; i++) {
      const y = top + i * rowH + rowH / 2;
      const focusCls = i === focus ? " focus" : "";
      const dim = i === focus ? "" : " dim";
      parts.push(`<circle class="node-dot${dim}" cx="${leftX}" cy="${y}" r="${i === focus ? 4 : 2.5}" />`);
      parts.push(`<circle class="node-dot" cx="${rightX}" cy="${y}" r="2.5" opacity="0.55" />`);
      const label = (toks[i] || "").slice(0, 10);
      parts.push(
        `<text class="tok-node${focusCls}" x="${leftX + 8}" y="${y + 3.5}">${escapeXml(label)}</text>`
      );
    }

    svg.innerHTML = parts.join("");
  }

  function escapeXml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  window.selectToken = function selectToken(i) {
    const tokEls = document.querySelectorAll("#explainer .tok");
    tokEls.forEach((el, idx) => el.classList.toggle("active", idx === i));

    const svg = document.querySelector("#heatmap svg");
    if (svg) {
      const n = tokEls.length;
      svg.querySelectorAll("rect.cell").forEach((r) => {
        const q = Number(r.getAttribute("data-q"));
        const on = q === i;
        r.classList.toggle("hi", on);
        r.classList.toggle("dim", !on);
      });
      svg.querySelectorAll("text.wt").forEach((t) => {
        const q = Number(t.getAttribute("data-q"));
        t.classList.toggle("dim", q !== i);
      });
      svg.querySelectorAll("text.row-lbl").forEach((t) => {
        const q = Number(t.getAttribute("data-q"));
        t.classList.toggle("focus", q === i);
        t.classList.toggle("dim", q !== i);
      });
    }

    updateArcs(i);

    const hint = document.getElementById("focus-hint");
    const t = tokEls[i];
    if (hint && t) {
      hint.textContent =
        "Focusing query “" +
        t.textContent.trim() +
        "” — arc width ∝ attention; grid row stays vivid, others dim.";
    }
  };

  function bindHeatmapTips() {
    const wrap = document.getElementById("heatmap");
    if (!wrap || wrap.dataset.tipsBound) return;
    wrap.dataset.tipsBound = "1";
    wrap.addEventListener("pointermove", (e) => {
      const cell = e.target.closest && e.target.closest("rect.cell");
      if (!cell || !wrap.contains(cell)) {
        hideTip();
        return;
      }
      const q = Number(cell.getAttribute("data-q"));
      const k = Number(cell.getAttribute("data-k"));
      const w = Number(cell.getAttribute("data-w"));
      const toks = tokens();
      const qt = toks[q] || ("q" + q);
      const kt = toks[k] || ("k" + k);
      showTip(
        `<strong>${escapeXml(qt)}</strong> → <strong>${escapeXml(kt)}</strong> ` +
          `<span class="w">${w.toFixed(3)}</span>`,
        e.clientX,
        e.clientY
      );
    });
    wrap.addEventListener("pointerleave", hideTip);
  }

  function boot() {
    bindHeatmapTips();
    const active = document.querySelector("#explainer .tok.active");
    const toks = document.querySelectorAll("#explainer .tok");
    let idx = 0;
    if (active) {
      idx = Array.prototype.indexOf.call(toks, active);
      if (idx < 0) idx = 0;
    }
    if (toks.length) window.selectToken(idx);
  }

  document.addEventListener("DOMContentLoaded", boot);
  document.body.addEventListener("htmx:afterSwap", (e) => {
    if (e.target && (e.target.id === "explainer" || e.target.querySelector?.("#explainer"))) {
      bindHeatmapTips();
      boot();
    }
  });
})();
