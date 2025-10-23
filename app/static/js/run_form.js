(function () {
  function $(sel, ctx = document) { return ctx.querySelector(sel); }
  function escapeHtml(s) {
    return String(s)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function renderJSONPretty(wrap, text) {
    let formatted = text;
    try { formatted = JSON.stringify(JSON.parse(text), null, 2); } catch {}
    wrap.innerHTML = (
      '<div class="space-y-2">' +
        '<div class="text-sm opacity-70">Response (JSON)</div>' +
        '<pre class="p-3 bg-base-300/40 rounded-lg overflow-auto text-xs max-h-[70vh]"><code>' +
          escapeHtml(formatted) +
        '</code></pre>' +
      '</div>'
    );
  }

  function renderBlob(wrap, ct, blob) {
    const url = URL.createObjectURL(blob);
    let content;
    if (ct.startsWith('image/')) {
      content = '<img src="' + url + '" alt="Result image" class="max-w-full rounded-lg border border-base-300"/>';
    } else if (ct.startsWith('audio/')) {
      content = '<audio controls class="w-full"><source src="' + url + '" type="' + ct + '">Your browser does not support the audio element.</audio>';
    } else if (ct.startsWith('video/')) {
      content = '<video controls class="w-full max-h-[70vh] rounded-lg border border-base-300"><source src="' + url + '" type="' + ct + '">Your browser does not support the video tag.</video>';
    } else {
      content = '<a class="btn btn-sm btn-primary" href="' + url + '" download>Download file</a>';
    }
    wrap.innerHTML = '<div class="space-y-2">' +
      '<div class="text-sm opacity-70">Response (' + ct + ')</div>' +
      '<div class="p-2">' + content + '</div>' +
      '</div>';
    setTimeout(() => URL.revokeObjectURL(url), 120000);
  }

  async function sniffContentType(blob) {
    try {
      const buf = new Uint8Array(await blob.slice(0, 32).arrayBuffer());
      const b = (i) => buf[i];
      if (buf.length >= 8 && b(0) === 0x89 && b(1) === 0x50 && b(2) === 0x4E && b(3) === 0x47 && b(4) === 0x0D && b(5) === 0x0A && b(6) === 0x1A && b(7) === 0x0A) return 'image/png';
      if (buf.length >= 3 && b(0) === 0xFF && b(1) === 0xD8 && b(2) === 0xFF) return 'image/jpeg';
      if (buf.length >= 4 && b(0) === 0x47 && b(1) === 0x49 && b(2) === 0x46 && b(3) === 0x38) return 'image/gif';
      if (buf.length >= 12 && b(0) === 0x52 && b(1) === 0x49 && b(2) === 0x46 && b(3) === 0x46 && b(8) === 0x57 && b(9) === 0x45 && b(10) === 0x42 && b(11) === 0x50) return 'image/webp';
      if (buf.length >= 2 && b(0) === 0x42 && b(1) === 0x4D) return 'image/bmp';
      if (buf.length >= 12 && b(0) === 0x52 && b(1) === 0x49 && b(2) === 0x46 && b(3) === 0x46 && b(8) === 0x57 && b(9) === 0x41 && b(10) === 0x56 && b(11) === 0x45) return 'audio/wav';
      if ((buf.length >= 3 && b(0) === 0x49 && b(1) === 0x44 && b(2) === 0x33) || (buf.length >= 2 && b(0) === 0xFF && (b(1) & 0xE0) === 0xE0)) return 'audio/mpeg';
      if (buf.length >= 4 && b(0) === 0x4F && b(1) === 0x67 && b(2) === 0x67 && b(3) === 0x53) return 'audio/ogg';
      if (buf.length >= 4 && b(0) === 0x66 && b(1) === 0x4C && b(2) === 0x61 && b(3) === 0x43) return 'audio/flac';
      if (buf.length >= 12 && b(4) === 0x66 && b(5) === 0x74 && b(6) === 0x79 && b(7) === 0x70) return 'video/mp4';
      if (buf.length >= 4 && b(0) === 0x1A && b(1) === 0x45 && b(2) === 0xDF && b(3) === 0xA3) return 'video/webm';
    } catch {}
    return 'application/octet-stream';
  }

  async function handleResponse(wrap, res) {
    const ctHeader = (res.headers.get('content-type') || '').toLowerCase();
    if (ctHeader.includes('text/html')) {
      wrap.innerHTML = await res.text();
      return;
    }
    if (ctHeader.includes('application/json') || ctHeader.endsWith('+json')) {
      const text = await res.text();
      renderJSONPretty(wrap, text || '{}');
      return;
    }
    if (ctHeader.startsWith('text/plain')) {
      const text = await res.text();
      wrap.innerHTML = '<pre class="p-3 bg-base-300/40 rounded-lg overflow-auto text-sm">' + escapeHtml(text) + '</pre>';
      return;
    }
    const blob = await res.blob();
    let ct = ctHeader;
    if (!ct || ct === 'application/octet-stream') {
      ct = await sniffContentType(blob);
    }
    const mediaTypes = ['image/', 'audio/', 'video/'];
    if (mediaTypes.some(p => ct.startsWith(p)) || ct === 'application/octet-stream') {
      renderBlob(wrap, ct, blob);
      return;
    }
    try {
      const text = await blob.text();
      wrap.innerHTML = '<pre class="p-3 bg-base-300/40 rounded-lg overflow-auto text-sm">' + escapeHtml(text) + '</pre>';
    } catch {
      renderBlob(wrap, ct || 'binary', blob);
    }
  }

  function findWrap(start) {
    return start.closest('[data-run-wrap]') || $('#runWrap');
  }

  function findSpinner() {
    return document.querySelector('[data-run-spinner]');
  }

  // Event delegation for any dynamically inserted run form
  document.addEventListener('submit', async function (e) {
    const form = e.target;
    if (!(form instanceof HTMLFormElement)) return;
    if (!form.matches('[data-run-form]')) return;
    e.preventDefault();

    const wrap = findWrap(form);
    const spinner = findSpinner();
    if (wrap && spinner) wrap.innerHTML = spinner.innerHTML;

    try {
      const fd = new FormData(form);
      const res = await fetch('/run-form', { method: 'POST', body: fd });
      await handleResponse(wrap || document.body, res);
    } catch (err) {
      if (wrap) wrap.innerHTML = '<div class="alert alert-error"><span>Request failed.</span></div>';
      // eslint-disable-next-line no-console
      console.error('Run request error', err);
    }
  });
})();

