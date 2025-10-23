(function () {
  // --------- DOM helpers ---------
  const SELECTORS = {
    runForm: '[data-run-form]',
    runWrap: '[data-run-wrap]',
    spinner: '[data-run-spinner]'
  };
  function $(sel, ctx = document) { return ctx.querySelector(sel); }
  function $all(sel, ctx = document) { return Array.from(ctx.querySelectorAll(sel)); }

  // --------- Utilities ---------
  function escapeHtml(s) {
    return String(s)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function preBlock(text, extraClasses = 'text-sm') {
    return `\n<pre class="p-3 bg-base-300/40 rounded-lg overflow-auto ${extraClasses}"><code>${escapeHtml(text)}</code></pre>`;
  }

  function findWrap(start) {
    return start.closest(SELECTORS.runWrap) || $('#runWrap');
  }
  function findSpinner() {
    return document.querySelector(SELECTORS.spinner);
  }

  // --------- Response rendering ---------
  function renderJSONPretty(wrap, text) {
    let formatted = text;
    try { formatted = JSON.stringify(JSON.parse(text), null, 2); } catch {}
    wrap.innerHTML = (
      `<div class="space-y-2">
        <div class="text-sm opacity-70">Response (JSON)</div>
        ${preBlock(formatted, 'text-xs max-h-[70vh]')}
      </div>`
    );
  }

  function renderBlob(wrap, ct, blob) {
    const url = URL.createObjectURL(blob);
    let content;
    if (ct.startsWith('image/')) {
      content = `<img src="${url}" alt="Result image" class="max-w-full rounded-lg border border-base-300"/>`;
    } else if (ct.startsWith('audio/')) {
      content = `<audio controls class="w-full"><source src="${url}" type="${ct}">Your browser does not support the audio element.</audio>`;
    } else if (ct.startsWith('video/')) {
      content = `<video controls class="w-full max-h-[70vh] rounded-lg border border-base-300"><source src="${url}" type="${ct}">Your browser does not support the video tag.</video>`;
    } else {
      content = `<a class="btn btn-sm btn-primary" href="${url}" download>Download file</a>`;
    }
    wrap.innerHTML = (
      `<div class="space-y-2">
        <div class="text-sm opacity-70">Response (${escapeHtml(ct)})</div>
        <div class="p-2">${content}</div>
      </div>`
    );
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

  function isJSONType(ctHeader) {
    return ctHeader.includes('application/json') || ctHeader.endsWith('+json');
  }
  function isTextType(ctHeader) {
    return ctHeader.startsWith('text/plain');
  }
  function isMediaType(ct) {
    return ['image/', 'audio/', 'video/'].some((p) => ct.startsWith(p));
  }

  async function handleResponse(wrap, res) {
    const ctHeader = (res.headers.get('content-type') || '').toLowerCase();

    if (ctHeader.includes('text/html')) {
      wrap.innerHTML = await res.text();
      return;
    }

    if (isJSONType(ctHeader)) {
      const text = await res.text();
      renderJSONPretty(wrap, text || '{}');
      return;
    }

    if (isTextType(ctHeader)) {
      const text = await res.text();
      wrap.innerHTML = preBlock(text, 'text-sm');
      return;
    }

    const blob = await res.blob();
    let ct = ctHeader;
    if (!ct || ct === 'application/octet-stream') {
      ct = await sniffContentType(blob);
    }

    if (isMediaType(ct) || ct === 'application/octet-stream') {
      renderBlob(wrap, ct, blob);
      return;
    }

    try {
      const text = await blob.text();
      wrap.innerHTML = preBlock(text, 'text-sm');
    } catch {
      renderBlob(wrap, ct || 'binary', blob);
    }
  }

  // --------- Extra args JSON validation & formatting ---------
  function getExtraArgsElements(form) {
    const textarea = form.querySelector('[name="extra_args"]');
    const errorDiv = form.querySelector('#extraArgsError');
    return { textarea, errorDiv };
  }

  function validateAndFormatExtraArgs(textarea, errorDiv) {
    if (!textarea || !errorDiv) return true; // nothing to validate
    const text = textarea.value.trim();
    errorDiv.textContent = '';
    textarea.classList.remove('textarea-error', 'input-error');

    if (!text || text === '{}') {
      if (text === '') textarea.value = '{}';
      return true;
    }

    try {
      const parsed = JSON.parse(text);
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        errorDiv.textContent = 'Invalid JSON: Input must be a JSON object (e.g., {"key": "value"}).';
        textarea.classList.add('textarea-error', 'input-error');
        return false;
      }
      textarea.value = JSON.stringify(parsed, null, 2);
      return true;
    } catch (e) {
      errorDiv.textContent = 'Invalid JSON: ' + (e && e.message ? e.message : String(e));
      textarea.classList.add('textarea-error', 'input-error');
      return false;
    }
  }

  function attachExtraArgsValidation(form) {
    if (!(form instanceof HTMLFormElement)) return;
    const { textarea, errorDiv } = getExtraArgsElements(form);
    if (!textarea || !errorDiv) return;

    textarea.addEventListener('blur', function () {
      validateAndFormatExtraArgs(textarea, errorDiv);
    });

    if (!textarea.value.trim()) {
      textarea.value = '{}';
    }
  }

  function initExtraArgs() {
    $all(SELECTORS.runForm).forEach(attachExtraArgsValidation);
  }

  // Also validate on delegated blur for dynamically inserted forms
  document.addEventListener('focusout', function (e) {
    const el = e.target;
    if (!(el instanceof HTMLElement)) return;
    if (!el.matches('[name="extra_args"]')) return;
    const form = el.closest(SELECTORS.runForm);
    if (!form) return;
    const { textarea, errorDiv } = getExtraArgsElements(form);
    validateAndFormatExtraArgs(textarea, errorDiv);
  }, true);

  // --------- Submit handling ---------
  document.addEventListener('submit', async function (e) {
    const form = e.target;
    if (!(form instanceof HTMLFormElement)) return;
    if (!form.matches(SELECTORS.runForm)) return;
    e.preventDefault();

    const wrap = findWrap(form);

    // Pre-validate extra_args before sending
    const { textarea, errorDiv } = getExtraArgsElements(form);
    const ok = validateAndFormatExtraArgs(textarea, errorDiv);
    if (!ok) {
      // Do not submit if invalid JSON
      return;
    }

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

  // --------- Init ---------
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initExtraArgs);
  } else {
    initExtraArgs();
  }
})();
