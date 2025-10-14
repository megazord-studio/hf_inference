# app/views/hf_models_table.py

from __future__ import annotations

import html
import json
from string import Template
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from app.runners import RUNNERS


def render_models_table(
    task: Optional[str], _rows_unused: List[Dict[str, Any]]
) -> str:
    # Build the task selector HTML
    opts = []
    for t in sorted(RUNNERS.keys()):
        sel = " selected" if task == t else ""
        opts.append(
            f'<option value="{html.escape(t)}"{sel}>{html.escape(t)}</option>'
        )
    opts_html = "".join(opts)

    # Safe JS literal for task (quoted string)
    task_js = json.dumps(task or "")

    template = Template(r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>mega model search</title>
  <link rel="stylesheet" href="https://unpkg.com/@shoelace-style/shoelace@2.15.1/cdn/themes/dark.css" />
  <script type="module" src="https://unpkg.com/@shoelace-style/shoelace@2.15.1/cdn/shoelace-autoloader.js"></script>
  <script src="https://unpkg.com/clusterize.js@0.18.1/clusterize.min.js"></script>
  <style>
    :root {
      --bg:#070b11; --text:#e3eeff; --muted:#9fb0cc;
      --panel:#0b1220; --panel2:#0b1220; --panel3:#0e1627;
      --line:rgba(148,163,184,.16);
      --neon:#00ffd0; --neon2:#8a2be2; --neon3:#ff5cf0;
      --ok:#16ffbd; --warn:#ffb703; --danger:#ff4d6d;
      --hover:rgba(0,255,208,.06);
      --zebra:rgba(138,43,226,.05);
      --shadow-neon:0 0 16px rgba(0,255,208,.28), 0 0 22px rgba(255,92,240,.18);
      --radius:16px;
    }

    html,body {
      margin:0; padding:0; height:100%;
      background:
        radial-gradient(1000px 520px at 12% -10%, rgba(138,43,226,.16), transparent 60%),
        radial-gradient(1200px 600px at 88% -6%, rgba(0,255,208,.12), transparent 60%),
        linear-gradient(180deg, rgba(255,92,240,.06), transparent 30%),
        var(--bg);
      color:var(--text);
      font:14px/1.55 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }

    body::before {
      content:""; position:fixed; inset:0; pointer-events:none;
      background-image:
        linear-gradient(rgba(0,255,200,.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,200,.06) 1px, transparent 1px);
      background-size:24px 24px;
    }

    .container { max-width:1200px; margin:28px auto 44px; padding:0 16px; }
    h1 { margin:0 0 14px; font-weight:900; letter-spacing:.2px;
         text-shadow:0 0 16px rgba(0,255,208,.35), 0 0 24px rgba(255,92,240,.2); }

    .panel { display:grid; gap:12px; }
    .row   { display:grid; gap:12px; grid-template-columns: 1fr; }

    .card {
      border:1px solid var(--line);
      border-radius:var(--radius);
      background:
        linear-gradient(180deg, rgba(138,43,226,.08), rgba(0,0,0,.22)),
        radial-gradient(1000px 140px at 10% 0%, rgba(255,92,240,.08), transparent 70%),
        var(--panel3);
      box-shadow:inset 0 0 10px rgba(138,43,226,.08), 0 0 28px rgba(0,255,208,.10);
      overflow:hidden;
    }

    .toolbar { display:flex; gap:12px; align-items:center; padding:16px; position:relative; }
    .toolbar::after {
      content:""; position:absolute; inset:0; pointer-events:none; border-radius:calc(var(--radius) - 2px);
      box-shadow:inset 0 0 0 1px rgba(0,255,208,.10), inset 0 0 18px rgba(138,43,226,.12);
    }
    .spacer { flex:1; }

    input, select, sl-button::part(base) {
      background:var(--panel2); color:var(--text);
      border:1px solid var(--line); border-radius:12px; padding:10px 12px; outline:none;
      box-shadow:inset 0 0 0 1px rgba(0,255,208,.05);
    }

    .neon-btn {
      appearance:none; border:0; cursor:pointer;
      padding:10px 14px; border-radius:12px; font-weight:800; letter-spacing:.2px;
      color:#0b0f14;
      background: linear-gradient(135deg, var(--neon), var(--neon3) 60%, var(--neon2));
      box-shadow: var(--shadow-neon);
      transition: transform .08s ease, filter .15s ease, box-shadow .15s ease;
    }
    .neon-btn:hover { filter:brightness(1.06); transform: translateY(-1px); }
    .neon-btn.is-active {
      box-shadow: 0 0 28px rgba(0,255,208,.55), 0 0 36px rgba(255,92,240,.35);
      outline: 2px solid rgba(90,240,255,.35);
    }

    #count {
      margin: 8px 16px 0;
      display:inline-block; padding:6px 10px; border-radius:10px;
      background: linear-gradient(90deg, rgba(0,255,208,.12), rgba(255,92,240,.12));
      border: 1px solid var(--line); box-shadow: 0 0 10px rgba(0,255,208,.15) inset;
    }
    #spinner { padding:12px 16px; color:var(--muted); }

    .table-wrap { height: 70vh; overflow: hidden; }
    table { width:100%; border-collapse:separate; border-spacing:0; }
    thead th {
      position:sticky; top:0; z-index:1;
      background:
        linear-gradient(180deg, rgba(0,255,208,.10), rgba(0,0,0,0)),
        var(--panel3);
      padding:12px 10px; text-align:left; font-weight:900; border-bottom:1px solid var(--line);
      letter-spacing:.2px;
    }
    tbody td { padding:10px; border-bottom:1px solid var(--line); }
    tbody tr:nth-child(even) td { background:var(--zebra); }
    tbody tr:hover td { background:var(--hover); }
    td.num { text-align:right; font-weight:800; letter-spacing:.15px; text-shadow:0 0 6px rgba(0,255,208,.18); }

    .link a { color:var(--neon); text-decoration:none; }
    .link a:hover { color:#5af0ff; text-shadow:0 0 8px rgba(0,255,208,.35); }

    .chip { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; border:1px solid var(--line); }
    .chip-true   { background:rgba(255,77,109,.18);  border-color:rgba(255,77,109,.55);  color:#ff8aa3; }
    .chip-manual { background:rgba(255,183,3,.16);   border-color:rgba(255,183,3,.55);   color:#ffd278; }
    .chip-false  { background:rgba(22,255,189,.14);  border-color:rgba(22,255,189,.55);  color:var(--ok); }

    .sortable { cursor:pointer; user-select:none; }
    .sortable .label { display:inline-flex; align-items:center; gap:8px; }
    .sort-icon { opacity:.35; transition: transform .12s ease, opacity .12s ease; font-size:10px; }
    .sortable:hover .sort-icon { opacity:.7; }
    .sort-asc .sort-icon { opacity:1; transform: rotate(180deg); text-shadow:0 0 8px rgba(0,255,208,.4); }
    .sort-desc .sort-icon { opacity:1; transform: rotate(0deg);   text-shadow:0 0 8px rgba(0,255,208,.4); }
  </style>
  <link rel="icon" type="image/x-icon" href="/static/favicon.ico" />
</head>
<body>
  <div class="container">
    <h1>Transformers HF Models</h1>

    <form id="panelForm" class="panel">
      <div>
        <label for="task" class="muted">Task (implemented)</label>
        <select id="task" name="task" onchange="location.href='/?task=' + encodeURIComponent(this.value)">
          <option value="">(choose a task)</option>
          $OPTS_HTML
        </select>
      </div>

      $MAIN_UI
    </form>
  </div>

  <script>
  (function(){
    const task = new URLSearchParams(location.search).get('task') || $TASK_JS;

    const form = document.getElementById('panelForm');
    const q = document.getElementById('q');
    const gatedSel = document.getElementById('gatedSel');
    const neonBtn = document.getElementById('neonSort');
    const countEl = document.getElementById('count');
    const spinner = document.getElementById('spinner');
    const contentArea = document.getElementById('contentArea');
    const scrollArea = document.getElementById('scrollArea');
    const dlCsvBtn = document.getElementById('downloadCsv');

    const thModel = document.querySelector('th[data-key="id"]');
    const thGated = document.querySelector('th[data-key="gated"]');
    const thDl    = document.querySelector('th[data-key="downloads"]');
    const thLikes = document.querySelector('th[data-key="likes"]');
    const thTrend = document.querySelector('th[data-key="trendingScore"]');
    const sortableHeaders = [thModel, thGated, thDl, thLikes, thTrend].filter(Boolean);

    if (form) form.addEventListener('submit', (e) => e.preventDefault());
    if (q) q.addEventListener('keydown', (e) => { if (e.key === 'Enter') e.preventDefault(); });

    if(!task || !contentArea) return;

    // --- Worker with SAFE maxima (no spread over huge arrays) ---
    const workerSrc = `
      let ALL = [];

      function safeMax(rows, getter, minVal=1){
        let max = minVal;
        for (let i = 0; i < rows.length; i++){
          const v = getter(rows[i]);
          if (v > max) max = v;
        }
        return max;
      }

      function neonScoreFactory(rows){
        const maxLogD = safeMax(rows, r => Math.log1p(r.downloads||0), 1);
        const maxLikes = safeMax(rows, r => (r.likes||0), 1);
        const maxTrend = safeMax(rows, r => (r.trendingScore||0), 1);
        return (r) => {
          const nd = Math.log1p(r.downloads||0)/maxLogD;
          const nl = (r.likes||0)/maxLikes;
          const nt = (r.trendingScore||0)/maxTrend;
          return 0.5*nd + 0.3*nl + 0.2*nt;
        };
      }

      function cmp(a,b){ return a<b ? -1 : a>b ? 1 : 0; }

      onmessage = (e) => {
        const {type, payload} = e.data || {};
        if(type === 'set'){
          ALL = payload || [];
          const gatedValues = Array.from(new Set(ALL.map(r => (r.gated||'false') + ''))).sort();
          postMessage({type:'ready', total: ALL.length, gatedValues});
          return;
        }
        if(type === 'query'){
          const { term, gated, neon, sortKey, sortDir } = payload;
          let base = ALL;

          if(gated){ base = base.filter(r => String(r.gated) === gated); }
          if(term){
            const t = term.toLowerCase();
            base = base.filter(r => (r.id||'').toLowerCase().includes(t));
          }

          if(neon){
            const score = neonScoreFactory(base);
            base = base.slice().sort((a,b) => score(b) - score(a));
          } else if (sortKey){
            const dir = (sortDir === 'asc') ? 1 : -1;
            base = base.slice().sort((a,b) => {
              const va = (sortKey==='id'||sortKey==='gated') ? (String(a[sortKey]||'')) : Number(a[sortKey]||0);
              const vb = (sortKey==='id'||sortKey==='gated') ? (String(b[sortKey]||'')) : Number(b[sortKey]||0);
              return dir * cmp(va,vb);
            });
          }

          postMessage({type:'result', rows: base});
        }

        if(type === 'csv'){
          const rows = payload || ALL;
          const head = ['id','gated','downloads','likes','trendingScore'];
          const esc = (s) => '"' + String(s).replaceAll('"','""') + '"';
          const lines = [head.join(',')].concat(rows.map(r => [r.id, r.gated, r.downloads||0, r.likes||0, r.trendingScore||0].map(esc).join(',')));
          postMessage({type:'csv', csv: lines.join('\\n')});
        }
      };
    `;
    const blob = new Blob([workerSrc], {type:'application/javascript'});
    const worker = new Worker(URL.createObjectURL(blob));

    const clusterize = new Clusterize({
      rows: [],
      scrollElem: scrollArea,
      contentElem: contentArea,
      rows_in_block: 60,
      blocks_in_cluster: 4,
      tag: 'tr',
      no_data_text: 'No data'
    });

    function rowHtml(r){
      const link = '<span class="link"><a href="https://huggingface.co/' + r.id + '" target="_blank" title="' + r.id + '">' + r.id + '</a></span>';
      let chip = '<span class="chip chip-false">open</span>';
      if(r.gated === 'true') chip = '<span class="chip chip-true">gated</span>';
      if(r.gated === 'manual') chip = '<span class="chip chip-manual">manual</span>';
      const fmt = (v) => (Number(v||0)).toLocaleString();
      return '<td style="width:52%">' + link + '</td>'
           + '<td style="width:12%">' + chip + '</td>'
           + '<td class="num" style="width:12%">' + fmt(r.downloads) + '</td>'
           + '<td class="num" style="width:12%">' + fmt(r.likes) + '</td>'
           + '<td class="num" style="width:12%">' + fmt(r.trendingScore) + '</td>';
    }

    let neonActive = false;
    let currentRows = [];
    let sortKey = null;
    let sortDir = 'desc';

    function applySortHeaderStyles(){
      const headers = [thModel, thGated, thDl, thLikes, thTrend].filter(Boolean);
      headers.forEach(th => th.classList.remove('sort-asc','sort-desc'));
      if (!sortKey) return;
      const th = document.querySelector('th[data-key="'+sortKey+'"]');
      if (th) th.classList.add(sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
    }

    function refresh(){
      worker.postMessage({
        type:'query',
        payload:{
          term: (q && q.value && q.value.trim()) || '',
          gated: (gatedSel && gatedSel.value) || '',
          neon: neonActive,
          sortKey,
          sortDir
        }
      });
      applySortHeaderStyles();
    }

    worker.onmessage = (e) => {
      const {type} = e.data || {};
      if(type === 'ready'){
        if (spinner) spinner.textContent = '';
        const { total, gatedValues } = e.data;
        if (countEl) countEl.textContent = total + ' total';
        const opts = ['<option value="">gated: any</option>'].concat((gatedValues||[]).map(v => '<option value="' + v + '">' + v + '</option>'));
        if(gatedSel) gatedSel.innerHTML = opts.join('');
        refresh(); return;
      }
      if(type === 'result'){
        const rows = e.data.rows || [];
        currentRows = rows;
        if (countEl) countEl.textContent = rows.length + ' shown';
        clusterize.update(rows.map(r => '<tr>' + rowHtml(r) + '</tr>'));
        return;
      }
      if(type === 'csv'){
        const url = URL.createObjectURL(new Blob([e.data.csv || ''], {type:'text/csv'}));
        const a = document.createElement('a'); a.href = url; a.download = (task || 'models') + '.csv';
        document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url), 2000);
      }
    };

    (async function loadAll(){
      try{
        const res = await fetch('/models?task=' + encodeURIComponent(task));
        const data = await res.json();
        if (!Array.isArray(data)) { if (spinner) spinner.textContent = 'No data for this task'; return; }
        if (spinner) spinner.textContent = 'Loaded ' + data.length + ' models';
        const rows = data.map(r => ({
          id: r.id,
          downloads: Number(r.downloads||0),
          likes: Number(r.likes||0),
          trendingScore: Number(r.trendingScore||0),
          gated: (function(g){ if (typeof g === 'string' && g.trim()) return g.trim(); return g ? 'true' : 'false'; })(r.gated)
        }));
        worker.postMessage({type:'set', payload: rows});
      } catch(err){ if (spinner) spinner.textContent = 'Failed to load models'; console.error(err); }
    })();

    let t = null;
    if(q){ q.addEventListener('input', () => { clearTimeout(t); t = setTimeout(refresh, 150); }); }
    if(gatedSel){ gatedSel.addEventListener('change', refresh); }
    if(neonBtn){
      neonBtn.addEventListener('click', () => {
        neonActive = !neonActive;
        if (neonActive) sortKey = null;
        neonBtn.classList.toggle('is-active', neonActive);
        refresh();
      });
    }
    if(dlCsvBtn){ dlCsvBtn.addEventListener('click', () => worker.postMessage({type:'csv', payload: currentRows})); }

    [thModel, thGated, thDl, thLikes, thTrend].filter(Boolean).forEach(th => {
      th.addEventListener('click', () => {
        const key = th.getAttribute('data-key'); if (!key) return;
        neonActive = false;
        if (sortKey === key) sortDir = (sortDir === 'asc') ? 'desc' : 'asc';
        else { sortKey = key; sortDir = (key === 'id' || key === 'gated') ? 'asc' : 'desc'; }
        neonBtn && neonBtn.classList.remove('is-active');
        refresh();
      });
    });
  })();
  </script>
</body>
</html>
""")

    main_ui = ""
    if task:
        main_ui = """
      <div class="card">
        <div class="toolbar">
          <input id="q" placeholder="Search model (substring)" style="flex:2" />
          <select id="gatedSel" style="flex:1"></select>
          <button id="neonSort" type="button" class="neon-btn" title="Sort by combined popularity signal">⚡ Neon&nbsp;Pulse</button>
          <div class="spacer"></div>
          <sl-button id="downloadCsv" size="small" outline>CSV</sl-button>
        </div>
        <div id="count" class="muted"></div>
        <div id="spinner" class="muted">Loading all models…</div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th class="sortable" data-key="id" style="width:52%">
                  <span class="label">Model <span class="sort-icon">▲</span></span>
                </th>
                <th class="sortable" data-key="gated" style="width:12%">
                  <span class="label">Gated <span class="sort-icon">▲</span></span>
                </th>
                <th class="sortable" data-key="downloads" style="width:12%; text-align:right">
                  <span class="label">Downloads <span class="sort-icon">▲</span></span>
                </th>
                <th class="sortable" data-key="likes" style="width:12%; text-align:right">
                  <span class="label">Likes <span class="sort-icon">▲</span></span>
                </th>
                <th class="sortable" data-key="trendingScore" style="width:12%; text-align:right">
                  <span class="label">Trend <span class="sort-icon">▲</span></span>
                </th>
              </tr>
            </thead>
          </table>
          <div id="scrollArea" style="height:calc(70vh - 44px); overflow:auto;">
            <table>
              <tbody id="contentArea"></tbody>
            </table>
          </div>
        </div>
      </div>
        """

    return template.substitute(
        TASK_JS=task_js, OPTS_HTML=opts_html, MAIN_UI=main_ui
    )
