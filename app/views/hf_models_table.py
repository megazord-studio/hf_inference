from __future__ import annotations
from typing import Optional, Dict, Any, List
import html
import json

from app.runners import RUNNERS

def render_models_table(task: Optional[str], rows: List[Dict[str, Any]]) -> str:
    # task selector (implemented tasks only)
    opts = []
    for t in sorted(RUNNERS.keys()):
        sel = " selected" if task == t else ""
        opts.append(f'<option value="{html.escape(t)}"{sel}>{html.escape(t)}</option>')

    # embed raw JSON (neutralize </script>)
    data_json = json.dumps(rows).replace("</", "<\\/")

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>mega model search</title>
  <link rel="stylesheet" href="https://unpkg.com/gridjs/dist/theme/mermaid.min.css" />
  <script src="https://unpkg.com/gridjs/dist/gridjs.umd.js"></script>
  <script src="https://unpkg.com/fuse.js@6.6.2"></script>
  <style>
    :root {{
      --bg:#0a0f14; --text:#e2ecff; --muted:#9aa8c0;
      --panel:#0e1522; --panel2:#0b1220;
      --line:rgba(148,163,184,.18);
      --neon:#00ffd0; --neon2:#8a2be2; --neon3:#ff5cf0;
      --ok:#16ffbd; --warn:#ffb703; --danger:#ff4d6d;
      --hover:rgba(0,255,208,.07);
      --zebra:rgba(138,43,226,.05);
    }}

    html,body {{
      margin:0; padding:0;
      background:
        radial-gradient(900px 480px at 12% -10%, rgba(138,43,226,.14), transparent 60%),
        radial-gradient(1100px 560px at 88% -6%, rgba(0,255,208,.10), transparent 60%),
        linear-gradient(180deg, rgba(255,92,240,.06), transparent 30%),
        var(--bg);
      color:var(--text);
      font:14px/1.45 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }}

    body::before {{
      content:""; position:fixed; inset:0; pointer-events:none;
      background-image:
        linear-gradient(rgba(0,255,200,.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,200,.06) 1px, transparent 1px);
      background-size:24px 24px;
    }}

    .container {{ max-width:1200px; margin:32px auto 56px; padding:0 20px; }}
    h1 {{ margin:0 0 18px; font-weight:800; text-shadow:0 0 16px rgba(0,255,208,.35), 0 0 24px rgba(255,92,240,.15); }}
    .muted {{ color:var(--muted); }}

    /* Inputs */
    .panel {{ display:grid; gap:10px; grid-template-columns: 1fr; }}
    .row   {{ display:grid; gap:12px; grid-template-columns: 1fr 220px 160px; align-items:center; }}
    select, input {{
      width:100%; background:var(--panel2); color:var(--text);
      border:1px solid var(--line); border-radius:12px; padding:10px 12px; outline:none;
      box-shadow: inset 0 0 0 1px rgba(0,255,208,.05);
    }}

    /* Neon Pulse button */
    .neon-btn {{
      appearance:none; border:0; cursor:pointer;
      padding:10px 14px; border-radius:12px; font-weight:800; letter-spacing:.2px;
      color:#0b0f14;
      background: linear-gradient(135deg, var(--neon), var(--neon3) 60%, var(--neon2));
      box-shadow: 0 0 16px rgba(0,255,208,.35), 0 0 22px rgba(255,92,240,.22);
      transition: transform .08s ease, filter .15s ease, box-shadow .15s ease;
    }}
    .neon-btn:hover {{ filter:brightness(1.05); transform: translateY(-1px); }}
    .neon-btn.is-active {{
      box-shadow: 0 0 28px rgba(0,255,208,.55), 0 0 36px rgba(255,92,240,.35);
      outline: 2px solid rgba(90,240,255,.4);
    }}

    /* Chips */
    .chip {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; border:1px solid var(--line); }}
    .chip-true   {{ background:rgba(255,77,109,.18);  border-color:rgba(255,77,109,.55);  color:#ff8aa3; }}
    .chip-manual {{ background:rgba(255,183,3,.16);   border-color:rgba(255,183,3,.55);   color:#ffd278; }}
    .chip-false  {{ background:rgba(22,255,189,.14);  border-color:rgba(22,255,189,.55);  color:var(--ok); }}

    /* Grid.js cyberpunk styling */
    .gridjs-container {{
      border:1px solid var(--line);
      border-radius:16px;
      background:
        linear-gradient(180deg, rgba(138,43,226,.07), rgba(0,0,0,.22)),
        radial-gradient(1000px 140px at 10% 0%, rgba(255,92,240,.08), transparent 70%);
      box-shadow:0 0 16px rgba(0,255,208,.16), inset 0 0 10px rgba(138,43,226,.08);
      overflow:hidden;
    }}
    .gridjs-table {{ width:100%; border-collapse:separate; border-spacing:0; }}

    /* Transparent body/headers; neon-ish header */
    .gridjs-tbody, td.gridjs-td, th.gridjs-th {{
      background:transparent !important;
      border-color:var(--line) !important;
    }}
    .gridjs-th {{
      color:var(--text) !important;
      font-weight:800;
      position:sticky; top:0; z-index:2;
      backdrop-filter:blur(3px) brightness(1.05);
      background: linear-gradient(180deg, rgba(0,255,208,.08), rgba(0,0,0,0)) !important;
    }}
    .gridjs-th:hover,
    .gridjs-th:focus,
    .gridjs-th:active {{
      background:linear-gradient(180deg, rgba(255,92,240,.12), rgba(0,0,0,0)) !important;
      color:var(--text) !important;
    }}

    /* numeric columns brighter + right aligned */
    td.gridjs-td:nth-child(3),
    td.gridjs-td:nth-child(4),
    td.gridjs-td:nth-child(5) {{
      text-align: right;
      color: var(--text) !important;
      font-weight: 800;
      letter-spacing: .15px;
      text-shadow: 0 0 6px rgba(0,255,208,.18);
    }}
    th.gridjs-th:nth-child(3),
    th.gridjs-th:nth-child(4),
    th.gridjs-th:nth-child(5) {{
      text-align: right;
      color: var(--text) !important;
    }}

    .gridjs-tr:nth-child(even) .gridjs-td {{ background:var(--zebra); }}
    .gridjs-tr:hover .gridjs-td {{ background:var(--hover); }}

    .link a {{ color:var(--neon); text-decoration:none; }}
    .link a:hover {{ color:#5af0ff; text-shadow:0 0 8px rgba(0,255,208,.35); }}

    .gridjs-footer {{ display:none; }}

    /* Count pill with glow */
    #count {{
      display:inline-block; padding:6px 10px; border-radius:10px;
      background: linear-gradient(90deg, rgba(0,255,208,.12), rgba(255,92,240,.12));
      border: 1px solid var(--line);
      box-shadow: 0 0 10px rgba(0,255,208,.15) inset;
    }}
  </style>
  <link rel="icon" type="image/x-icon" href="/static/favicon.ico" />
</head>
<body>
  <div class="container">
    <h1>Transformers HF Models</h1>

    <form method="get" class="panel">
      <div>
        <label for="task" class="muted">Task (implemented)</label>
        <select id="task" name="task" onchange="this.form.submit()">
          <option value="">(choose a task)</option>
          {''.join(opts)}
        </select>
      </div>

      {"" if not task else '''
      <div class="row">
        <input id="q" placeholder="Search model (fuzzy)" />
        <select id="gatedSel"></select>
        <button id="neonSort" type="button" class="neon-btn" title="Sort by combined popularity signal">⚡ Neon&nbsp;Pulse</button>
      </div>
      <div id="count" class="muted" style="margin-top:-4px;"></div>
      <div id="grid"></div>
      '''}
    </form>
  </div>

  <script id="__DATA__" type="application/json">{data_json}</script>
  <script>
  (function(){{
    const gridEl = document.getElementById('grid');
    if(!gridEl) return;

    let rows = JSON.parse(document.getElementById('__DATA__').textContent || '[]');
    rows = rows.map(r => {{
      const g = (r.gated ?? 'false');
      const gated = (typeof g === 'string' && g.trim()) ? g.trim() : (g ? 'true' : 'false');
      return {{
        id: r.id,
        downloads: Number(r.downloads||0),
        likes: Number(r.likes||0),
        trendingScore: Number(r.trendingScore||0),
        gated
      }};
    }});

    const gatedSel = document.getElementById('gatedSel');
    const gatedVals = Array.from(new Set(rows.map(r => r.gated))).sort();
    gatedSel.innerHTML = '<option value="">gated: any</option>' + gatedVals.map(v => `<option value="${{v}}">${{v}}</option>`).join('');

    const fuse = new Fuse(rows, {{ keys: ['id'], threshold: 0.33, ignoreLocation: true }});
    const q = document.getElementById('q');
    const countEl = document.getElementById('count');
    const neonBtn = document.getElementById('neonSort');

    const link = (id) => gridjs.html(`<span class="link"><a href="https://huggingface.co/${{id}}" target="_blank" title="${{id}}">${{id}}</a></span>`);
    const chip = (g) => {{
      if (g === 'manual') return gridjs.html('<span class="chip chip-manual">manual</span>');
      if (g === 'true')   return gridjs.html('<span class="chip chip-true">gated</span>');
      return gridjs.html('<span class="chip chip-false">open</span>');
    }};

    const mapToGrid = (arr) => arr.map(r => [
      link(r.id),
      chip(r.gated),
      r.downloads,
      r.likes,
      r.trendingScore
    ]);

    // Precompute normalization bounds for Neon Pulse score
    const maxLogD = Math.max(...rows.map(r => Math.log1p(r.downloads))) || 1;
    const maxLikes = Math.max(...rows.map(r => r.likes)) || 1;
    const maxTrend = Math.max(...rows.map(r => r.trendingScore)) || 1;
    function neonScore(r){{
      const nd = Math.log1p(r.downloads) / maxLogD;   // downloads (log)
      const nl = (r.likes||0) / maxLikes;            // likes
      const nt = (r.trendingScore||0) / maxTrend;    // trend
      return 0.5*nd + 0.3*nl + 0.2*nt;               // weighted blend
    }}

    let neonActive = false;

    let grid = new gridjs.Grid({{
      columns: [
        {{ name: "Model", width: "52%", sort: false }},
        {{ name: "Gated", width: "12%", sort: false }},
        {{ name: "Downloads", width: "12%", sort: {{ compare: (a,b) => a - b }}, formatter: v => gridjs.html(`<span>${{(v||0).toLocaleString()}}</span>`) }},
        {{ name: "Likes", width: "12%", sort: {{ compare: (a,b) => a - b }}, formatter: v => gridjs.html(`<span>${{(v||0).toLocaleString()}}</span>`) }},
        {{ name: "Trend", width: "12%", sort: {{ compare: (a,b) => a - b }} }},
      ],
      data: mapToGrid(rows),
      sort: true,
      search: false,
      pagination: false
    }}).render(gridEl);

    function filtered(){{
      const term = (q.value || '').trim();
      const gated = gatedSel.value;
      let base = rows;
      if (gated) base = base.filter(r => String(r.gated) === gated);
      if (term) {{
        const res = fuse.search(term);
        const keep = new Set(res.map(x => x.item.id));
        base = base.filter(r => keep.has(r.id));
      }}
      return base;
    }}

    function refresh(){{
      let data = filtered();
      if (neonActive) {{
        data = data.slice().sort((a,b) => neonScore(b) - neonScore(a));
      }}
      countEl.textContent = `${{data.length}} shown`;
      grid.updateConfig({{ data: mapToGrid(data) }}).forceRender();
    }}

    q.addEventListener('input', refresh);
    gatedSel.addEventListener('change', refresh);
    neonBtn.addEventListener('click', () => {{
      neonActive = !neonActive;
      neonBtn.classList.toggle('is-active', neonActive);
      refresh();
    }});

    countEl.textContent = `${{rows.length}} shown`;
  }})();
  </script>
</body>
</html>"""
