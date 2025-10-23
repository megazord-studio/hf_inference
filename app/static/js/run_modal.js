(function () {
  const dlg = document.getElementById('runDialog');
  const runTmpl = document.getElementById('runModalTemplate');

  function currentTask() {
    const usp = new URLSearchParams(location.search);
    return usp.get('task') || (document.getElementById('panelForm')?.dataset.selectedTask || '');
  }

  function ensureDialog() {
    if (!dlg) return null;
    // Clear and create modal-box container
    dlg.innerHTML = '';
    const box = document.createElement('div');
    box.className = 'modal-box p-0 rounded-xl border border-base-300 bg-base-200/95 shadow-2xl w-[94vw] md:w-[80vw] lg:max-w-5xl';
    dlg.appendChild(box);

    // Close on outside click
    dlg.addEventListener('click', (e) => {
      if (e.target === dlg) dlg.close();
    }, {once: true});

    return box;
  }

  async function openDialog() {
    if (!dlg) return;
    if (typeof dlg.showModal === 'function') dlg.showModal();
    else dlg.setAttribute('open', '');
  }

  window.runModel = async function (id) {
    try {
      if (!dlg || !runTmpl) return;
      const box = ensureDialog();
      if (!box) return;

      // Close button
      const closeBtn = document.createElement('button');
      closeBtn.className = 'btn btn-md btn-circle btn-ghost absolute right-3 top-3 text-xl';
      closeBtn.textContent = '✕';
      closeBtn.addEventListener('click', () => dlg.close());
      box.appendChild(closeBtn);

      const node = runTmpl.content.cloneNode(true);
      const task = currentTask() || 'model';
      const hfUrl = 'https://huggingface.co/' + id.split('/').map(s => encodeURIComponent(s)).join('/');

      node.querySelectorAll('[data-model-title]').forEach(el => {
        el.textContent = id;
        el.setAttribute('title', id);
      });
      node.querySelectorAll('[data-task]').forEach(el => el.textContent = task);
      node.querySelectorAll('[data-href="hf"]').forEach(el => {
        el.href = hfUrl;
      });
      node.querySelectorAll('[data-href="start"]').forEach(el => {
        el.href = '/run?task=' + encodeURIComponent(task || '') + '&model=' + encodeURIComponent(id);
      });

      // Host for HTMX form
      const frag = document.createElement('div');
      frag.appendChild(node);
      box.appendChild(frag);

      const host = box.querySelector('[data-form-host]');
      if (host) {
        const url = '/run-form?task=' + encodeURIComponent(task || '') + '&model_id=' + encodeURIComponent(id);
        host.setAttribute('hx-get', url);
        host.setAttribute('hx-trigger', 'load');
        host.setAttribute('hx-target', 'this');
        host.setAttribute('hx-swap', 'innerHTML');
        if (window.htmx && typeof window.htmx.process === 'function') {
          window.htmx.process(host);
        }
      }

      await openDialog();
    } catch (e) {
      console.error('Run modal error', e);
    }
  };
})();
