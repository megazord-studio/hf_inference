let ALL = [];

function safeMax(rows, getter, minVal = 1) {
  let max = minVal;
  for (let i = 0; i < rows.length; i++) {
    const v = getter(rows[i]);
    if (v > max) max = v;
  }
  return max;
}

function neonScoreFactory(rows) {
  const maxLogD = safeMax(rows, r => Math.log1p(r.downloads || 0), 1);
  const maxLikes = safeMax(rows, r => (r.likes || 0), 1);
  const maxTrend = safeMax(rows, r => (r.trendingScore || 0), 1);
  return (r) => {
    const nd = Math.log1p(r.downloads || 0) / maxLogD;
    const nl = (r.likes || 0) / maxLikes;
    const nt = (r.trendingScore || 0) / maxTrend;
    return 0.5 * nd + 0.3 * nl + 0.2 * nt;
  };
}

function cmp(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

onmessage = (e) => {
  const {type, payload} = e.data || {};
  if (type === 'set') {
    ALL = payload || [];
    const gatedValues = Array.from(new Set(ALL.map(r => (r.gated || 'false') + ''))).sort();
    postMessage({type: 'ready', total: ALL.length, gatedValues});
    return;
  }
  if (type === 'query') {
    const {term, gated, neon, sortKey, sortDir} = payload;
    let base = ALL;
    if (gated) {
      base = base.filter(r => String(r.gated) === gated);
    }
    if (term) {
      const t = term.toLowerCase();
      base = base.filter(r => (r.id || '').toLowerCase().includes(t));
    }
    if (neon) {
      const score = neonScoreFactory(base);
      base = base.slice().sort((a, b) => score(b) - score(a));
    } else if (sortKey) {
      const dir = (sortDir === 'asc') ? 1 : -1;
      base = base.slice().sort((a, b) => {
        const va = (sortKey === 'id' || sortKey === 'gated') ? (String(a[sortKey] || '')) : Number(a[sortKey] || 0);
        const vb = (sortKey === 'id' || sortKey === 'gated') ? (String(b[sortKey] || '')) : Number(b[sortKey] || 0);
        return dir * cmp(va, vb);
      });
    }
    postMessage({type: 'result', rows: base});
  }
  if (type === 'csv') {
    const rows = payload || ALL;
    const head = ['id', 'gated', 'downloads', 'likes', 'trendingScore'];
    const esc = (s) => '"' + String(s).replaceAll('"', '""') + '"';
    const lines = [head.join(',')].concat(
      rows.map(r => [r.id, r.gated, r.downloads || 0, r.likes || 0, r.trendingScore || 0].map(esc).join(','))
    );
    postMessage({type: 'csv', csv: '\uFEFF' + lines.join('\r\n')});
  }
};
