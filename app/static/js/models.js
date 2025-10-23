(async function () {
  const formEl = document.getElementById('panelForm');
  const task = new URLSearchParams(location.search).get('task') || (formEl ? formEl.dataset.selectedTask : '') || '';

  const q = document.getElementById('q');
  const gatedSel = document.getElementById('gatedSel');
  const neonBtn = document.getElementById('neonSort');
  const countEl = document.getElementById('count');
  const spinner = document.getElementById('spinner');
  const contentArea = document.getElementById('contentArea');
  const scrollArea = document.getElementById('scrollArea');
  const dlCsvBtn = document.getElementById('downloadCsv');

  // Debug toggles via URL params
  const usp = new URLSearchParams(location.search);
  const NO_WORKER = usp.get('noworker') === '1';
  const NO_VIRT = usp.get('novirt') === '1';

  // Keep a local copy for fallback filtering/sorting
  let ALL_ROWS = [];

  const thead = scrollArea ? scrollArea.querySelector('thead') : null;

  if (formEl) formEl.addEventListener('submit', (e) => e.preventDefault());
  if (q) q.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') e.preventDefault();
  });
  if (!task || !contentArea) return;

  // Use external worker script to minimize inline code
  let workerOk = !NO_WORKER;
  const worker = workerOk ? new Worker('/static/js/models_worker.js') : null;
  if (!workerOk) {
    try {
      worker && worker.terminate();
    } catch (_) {
    }
  }
  if (worker) worker.onerror = (e) => {
    console.warn('Worker error', e);
    workerOk = false;
  };

  // Prefer Clusterize when available; otherwise, fallback to simple innerHTML rendering
  let clusterize = null;
  if (!NO_VIRT && window.Clusterize && scrollArea && contentArea) {
    try {
      clusterize = new Clusterize({
        rows: [],
        scrollElem: scrollArea,
        contentElem: contentArea,
        tag: 'tr'
      });
    } catch (e) {
      console.warn('Clusterize init failed, falling back to innerHTML.', e);
      clusterize = null;
    }
  } else if (NO_VIRT) {
    console.info('Virtualization disabled via ?novirt=1');
  }

  function renderRows(rows) {
    const htmlRows = rows.map(r => rowHtml(r));
    if (clusterize) {
      try {
        clusterize.update(htmlRows);
        return;
      } catch (e) {
        console.warn('Clusterize update failed; fallback rendering.', e);
        clusterize = null;
      }
    }
    contentArea.innerHTML = htmlRows.join('');
  }

  function badgeHtml(g) {
    if (g === 'true') return '<span class="badge badge-error badge-sm">gated</span>';
    if (g === 'manual') return '<span class="badge badge-warning badge-sm">manual</span>';
    return '<span class="badge badge-success badge-sm">open</span>';
  }

  function escapeAttr(s) {
    return String(s).replace(/"/g, '&quot;');
  }

  function rowHtml(r) {
    const link = '<a class="link" href="https://huggingface.co/' + r.id + '" target="_blank" title="' + r.id + '">' + r.id + '</a>';
    const fmt = (v) => (Number(v || 0)).toLocaleString('en-US');
    const runBtn = '<button class="btn btn-xs btn-accent" title="Run this model" data-run="' + escapeAttr(r.id) + '">Run</button>';
    return '<tr>'
      + '<td class="w-[46%]">' + link + '</td>'
      + '<td class="w-[12%]">' + badgeHtml(r.gated) + '</td>'
      + '<td class="text-right w-[12%]">' + fmt(r.downloads) + '</td>'
      + '<td class="text-right w-[12%]">' + fmt(r.likes) + '</td>'
      + '<td class="text-right w-[12%]">' + fmt(r.trendingScore) + '</td>'
      + '<td class="text-center w-[8%]">' + runBtn + '</td>'
      + '</tr>';
  }

  // Local fallback query
  function queryRowsLocal({term, gated, neon, sortKey, sortDir}) {
    let base = ALL_ROWS;
    if (gated) base = base.filter(r => String(r.gated) === gated);
    if (term) {
      const t = term.toLowerCase();
      base = base.filter(r => (r.id || '').toLowerCase().includes(t));
    }
    if (neon) {
      const maxLogD = base.reduce((m, r) => Math.max(m, Math.log1p(r.downloads || 0)), 1);
      const maxLikes = base.reduce((m, r) => Math.max(m, r.likes || 0), 1);
      const maxTrend = base.reduce((m, r) => Math.max(m, r.trendingScore || 0), 1);
      const score = (r) => 0.5 * (Math.log1p(r.downloads || 0) / maxLogD) + 0.3 * ((r.likes || 0) / maxLikes) + 0.2 * ((r.trendingScore || 0) / maxTrend);
      base = base.slice().sort((a, b) => score(b) - score(a));
    } else if (sortKey) {
      const dir = (sortDir === 'asc') ? 1 : -1;
      base = base.slice().sort((a, b) => {
        const va = (sortKey === 'id' || sortKey === 'gated') ? String(a[sortKey] || '') : Number(a[sortKey] || 0);
        const vb = (sortKey === 'id' || sortKey === 'gated') ? String(b[sortKey] || '') : Number(b[sortKey] || 0);
        return dir * (va < vb ? -1 : va > vb ? 1 : 0);
      });
    }
    return base;
  }

  let neonActive = false;
  let currentRows = [];
  let sortKey = null;
  let sortDir = 'desc';

  function applySortHeaderStyles() {
    if (!thead) return;
    thead.querySelectorAll('th[data-key]').forEach(h => h.classList.remove('underline', 'text-primary'));
    if (!sortKey) return;
    const th = thead.querySelector('th[data-key="' + sortKey + '"]');
    if (th) th.classList.add('underline', 'text-primary');
  }

  function refresh() {
    const payload = {
      term: (q && q.value && q.value.trim()) || '',
      gated: (gatedSel && gatedSel.value) || '',
      neon: neonActive,
      sortKey,
      sortDir
    };
    if (workerOk && worker) {
      try {
        worker.postMessage({type: 'query', payload});
      } catch (e) {
        workerOk = false;
      }
    }
    if (!workerOk) {
      const rows = queryRowsLocal(payload);
      currentRows = rows;
      if (countEl) countEl.textContent = rows.length.toLocaleString('en-US') + ' shown';
      renderRows(rows);
    }
    applySortHeaderStyles();
  }

  // CSV helpers (main-thread fallback)
  function buildCsvContent(rows) {
    const head = ['id', 'gated', 'downloads', 'likes', 'trendingScore'];
    const esc = (s) => '"' + String(s ?? '').replaceAll('"', '""') + '"';
    const lines = [head.join(',')].concat(
      (rows || []).map(r => [r.id, r.gated, r.downloads || 0, r.likes || 0, r.trendingScore || 0].map(esc).join(','))
    );
    return '\uFEFF' + lines.join('\r\n');
  }

  function downloadCsv(rows) {
    try {
      const csv = buildCsvContent(rows);
      const blob = new Blob([csv], {type: 'text/csv;charset=utf-8'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = (task || 'models') + '.csv';
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    } catch (e) {
      console.error('CSV export failed', e);
    }
  }

  if (worker) {
    worker.onmessage = (e) => {
      const {type} = e.data || {};
      if (type === 'ready') {
        if (spinner) spinner.textContent = '';
        const {total, gatedValues} = e.data;
        if (countEl) countEl.textContent = total.toLocaleString('en-US') + ' total';
        if (gatedSel) gatedSel.innerHTML = ['<option value="">gated: any</option>'].concat((gatedValues || []).map(v => '<option value="' + v + '">' + v + '</option>')).join('');
        refresh();
        return;
      }
      if (type === 'result') {
        const rows = e.data.rows || [];
        currentRows = rows;
        if (countEl) countEl.textContent = rows.length.toLocaleString('en-US') + ' shown';
        renderRows(rows);
        return;
      }
      if (type === 'csv') {
        const blob = new Blob([e.data.csv || ''], {type: 'text/csv;charset=utf-8'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = (task || 'models') + '.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 2000);
      }
    };
  }

  try {
    const res = await fetch('/models?task=' + encodeURIComponent(task));
    const data = await res.json();
    if (!Array.isArray(data)) {
      if (spinner) spinner.textContent = 'No data for this task';
      return;
    }
    if (spinner) spinner.textContent = 'Loaded ' + data.length.toLocaleString('en-US') + ' models';

    let filtered = data;
    if (task === 'image-to-image') {
      filtered = data.filter(r => (r && r.id) !== 'fal/AuraSR-v2');
    }

    const rows = filtered.map(r => ({
      id: r.id,
      downloads: Number(r.downloads || 0),
      likes: Number(r.likes || 0),
      trendingScore: Number(r.trendingScore || 0),
      gated: (function (g) {
        if (typeof g === 'string' && g.trim()) return g.trim();
        return g ? 'true' : 'false';
      })(r.gated)
    }));

    // Save locally for fallback
    ALL_ROWS = rows;

    // Populate gated filter options immediately
    if (gatedSel) {
      const gatedValues = Array.from(new Set(ALL_ROWS.map(r => String(r.gated)))).sort();
      gatedSel.innerHTML = ['<option value="">gated: any</option>'].concat(gatedValues.map(v => '<option value="' + v + '">' + v + '</option>')).join('');
    }

    // Render immediately (fallback) while worker warms up
    currentRows = rows;
    if (countEl) countEl.textContent = rows.length.toLocaleString('en-US') + ' shown';
    renderRows(rows);

    // Then hand off to worker for subsequent queries
    try {
      if (workerOk && worker) worker.postMessage({type: 'set', payload: rows});
    } catch (e) {
      workerOk = false;
    }
  } catch (err) {
    if (spinner) spinner.textContent = 'Failed to load models';
    console.error(err);
  }

  let t = null;
  if (q) {
    q.addEventListener('input', () => {
      clearTimeout(t);
      t = setTimeout(refresh, 150);
    });
  }
  if (gatedSel) {
    gatedSel.addEventListener('change', refresh);
  }
  if (neonBtn) {
    neonBtn.addEventListener('click', () => {
      neonActive = !neonActive;
      neonBtn.setAttribute('aria-pressed', neonActive ? 'true' : 'false');
      if (neonActive) sortKey = null;
      neonBtn.classList.toggle('btn-primary', neonActive);
      refresh();
    });
  }
  if (dlCsvBtn) {
    dlCsvBtn.addEventListener('click', () => {
      if (workerOk && worker) {
        try {
          worker.postMessage({type: 'csv', payload: currentRows});
          return;
        } catch (_) {
        }
      }
      downloadCsv(currentRows);
    });
  }

  // Single event listener: header sorting & run buttons
  if (thead) {
    thead.addEventListener('click', (ev) => {
      const th = ev.target.closest('th[data-key]');
      if (!th) return;
      const key = th.getAttribute('data-key');
      neonActive = false;
      if (sortKey === key) sortDir = (sortDir === 'asc') ? 'desc' : 'asc';
      else {
        sortKey = key;
        sortDir = (key === 'id' || key === 'gated') ? 'asc' : 'desc';
      }
      neonBtn && neonBtn.classList.remove('btn-primary');
      refresh();
    });
  }
  if (contentArea) {
    contentArea.addEventListener('click', (ev) => {
      const btn = ev.target.closest('[data-run]');
      if (!btn) return;
      const id = btn.getAttribute('data-run');
      if (id && typeof window.runModel === 'function') window.runModel(id);
    });
  }
})();
