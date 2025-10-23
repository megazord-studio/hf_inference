(function () {
  const EXAMPLE_TOKEN = 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx';

  // --------- Code generation functions ---------
  
  function generateCurl(task, modelId, payload, extraArgs, files) {
    const lines = ['curl -X POST http://localhost:8000/inference \\'];
    
    // Add Bearer token
    lines.push(`  -H "Authorization: Bearer ${EXAMPLE_TOKEN}" \\`);
    
    // Build spec object
    const spec = {
      model_id: modelId,
      task: task,
      payload: payload || {},
      extra_args: extraArgs || {}
    };
    
    lines.push(`  -F 'spec=${JSON.stringify(spec)}' \\`);
    
    // Add files
    if (files.image) {
      lines.push(`  -F 'image=@/path/to/image.jpg' \\`);
    }
    if (files.audio) {
      lines.push(`  -F 'audio=@/path/to/audio.wav' \\`);
    }
    if (files.video) {
      lines.push(`  -F 'video=@/path/to/video.mp4' \\`);
    }
    
    // Remove trailing backslash from last line
    const lastLine = lines[lines.length - 1];
    lines[lines.length - 1] = lastLine.replace(/ \\$/, '');
    
    return lines.join('\n');
  }
  
  function generateNodeFetch(task, modelId, payload, extraArgs, files) {
    const lines = [];
    lines.push('const FormData = require(\'form-data\');');
    lines.push('const fs = require(\'fs\');');
    lines.push('const fetch = require(\'node-fetch\');');
    lines.push('');
    lines.push('const formData = new FormData();');
    
    // Build spec
    const spec = {
      model_id: modelId,
      task: task,
      payload: payload || {},
      extra_args: extraArgs || {}
    };
    
    lines.push(`formData.append('spec', JSON.stringify(${JSON.stringify(spec)}));`);
    lines.push('');
    
    // Add files
    if (files.image) {
      lines.push('formData.append(\'image\', fs.createReadStream(\'/path/to/image.jpg\'));');
    }
    if (files.audio) {
      lines.push('formData.append(\'audio\', fs.createReadStream(\'/path/to/audio.wav\'));');
    }
    if (files.video) {
      lines.push('formData.append(\'video\', fs.createReadStream(\'/path/to/video.mp4\'));');
    }
    
    lines.push('');
    lines.push('const response = await fetch(\'http://localhost:8000/inference\', {');
    lines.push('  method: \'POST\',');
    lines.push('  headers: {');
    lines.push(`    'Authorization': 'Bearer ${EXAMPLE_TOKEN}'`);
    lines.push('  },');
    lines.push('  body: formData');
    lines.push('});');
    lines.push('');
    lines.push('const result = await response.json();');
    lines.push('console.log(result);');
    
    return lines.join('\n');
  }
  
  function generatePythonRequests(task, modelId, payload, extraArgs, files) {
    const lines = [];
    lines.push('import requests');
    lines.push('import json');
    lines.push('');
    
    // Build spec
    const spec = {
      model_id: modelId,
      task: task,
      payload: payload || {},
      extra_args: extraArgs || {}
    };
    
    lines.push('spec = json.dumps(' + JSON.stringify(spec) + ')');
    lines.push('');
    lines.push('data = {\'spec\': spec}');
    
    // Add files
    const fileLines = [];
    if (files.image) {
      fileLines.push('    \'image\': open(\'/path/to/image.jpg\', \'rb\')');
    }
    if (files.audio) {
      fileLines.push('    \'audio\': open(\'/path/to/audio.wav\', \'rb\')');
    }
    if (files.video) {
      fileLines.push('    \'video\': open(\'/path/to/video.mp4\', \'rb\')');
    }
    
    if (fileLines.length > 0) {
      lines.push('files = {');
      lines.push(fileLines.join(',\n'));
      lines.push('}');
    } else {
      lines.push('files = None');
    }
    
    lines.push('');
    lines.push('headers = {');
    lines.push(`    'Authorization': 'Bearer ${EXAMPLE_TOKEN}'`);
    lines.push('}');
    lines.push('');
    lines.push('response = requests.post(');
    lines.push('    \'http://localhost:8000/inference\',');
    lines.push('    data=data,');
    lines.push('    files=files,');
    lines.push('    headers=headers');
    lines.push(')');
    lines.push('');
    lines.push('print(response.json())');
    
    return lines.join('\n');
  }
  
  // --------- Form data extraction ---------
  
  function extractFormData(form) {
    if (!form) return null;
    
    const task = form.querySelector('[name="task"]')?.value || '';
    const modelId = form.querySelector('[name="model_id"]')?.value || '';
    
    // Extract payload fields (exclude task, model_id, extra_args, and file inputs)
    const payload = {};
    const formData = new FormData(form);
    
    for (const [key, value] of formData.entries()) {
      if (key === 'task' || key === 'model_id' || key === 'extra_args') continue;
      
      const input = form.querySelector(`[name="${key}"]`);
      if (!input) continue;
      
      // Skip file inputs
      if (input.type === 'file') continue;
      
      // Handle JSON textareas
      if (input.dataset.type === 'json' || input.classList.contains('font-mono')) {
        const text = (value || '').trim();
        if (text) {
          try {
            payload[key] = JSON.parse(text);
          } catch {
            payload[key] = text; // fallback to string if not valid JSON
          }
        }
      } else if (value) {
        payload[key] = value;
      }
    }
    
    // Extract extra_args
    let extraArgs = {};
    const extraArgsTextarea = form.querySelector('[name="extra_args"]');
    if (extraArgsTextarea) {
      const text = (extraArgsTextarea.value || '').trim();
      if (text && text !== '{}') {
        try {
          extraArgs = JSON.parse(text);
        } catch {
          // ignore invalid JSON
        }
      }
    }
    
    // Check which files are present
    const files = {
      image: !!form.querySelector('[name="image"]')?.files?.[0],
      audio: !!form.querySelector('[name="audio"]')?.files?.[0],
      video: !!form.querySelector('[name="video"]')?.files?.[0]
    };
    
    return { task, modelId, payload, extraArgs, files };
  }
  
  // --------- UI management ---------
  
  function createCurlHelperUI() {
    const container = document.createElement('div');
    container.className = 'collapse collapse-arrow bg-base-300/30 rounded-lg';
    container.id = 'curlHelperContainer';
    
    container.innerHTML = `
      <input type="checkbox" />
      <div class="collapse-title text-sm font-medium">
        📋 API Code Examples
      </div>
      <div class="collapse-content">
        <div class="space-y-3">
          <div class="text-xs opacity-70">
            Copy these examples to use this request in your code
          </div>
          
          <div role="tablist" class="tabs tabs-boxed">
            <a role="tab" class="tab tab-active" data-lang="curl">cURL</a>
            <a role="tab" class="tab" data-lang="node">Node.js</a>
            <a role="tab" class="tab" data-lang="python">Python</a>
          </div>
          
          <div class="relative">
            <pre class="p-3 bg-base-300/60 rounded-lg overflow-auto text-xs" style="max-height: 400px;"><code id="curlHelperCode"></code></pre>
            <button type="button" class="btn btn-xs btn-ghost absolute top-2 right-2" id="curlHelperCopy" title="Copy to clipboard">
              📋 Copy
            </button>
          </div>
        </div>
      </div>
    `;
    
    return container;
  }
  
  function updateCodeExample(container, lang, code) {
    const codeEl = container.querySelector('#curlHelperCode');
    if (codeEl) {
      codeEl.textContent = code;
    }
  }
  
  function setupTabNavigation(container, updateCallback) {
    const tabs = container.querySelectorAll('[role="tab"]');
    tabs.forEach(tab => {
      tab.addEventListener('click', (e) => {
        e.preventDefault();
        
        // Update active tab
        tabs.forEach(t => t.classList.remove('tab-active'));
        tab.classList.add('tab-active');
        
        // Update code
        const lang = tab.dataset.lang;
        updateCallback(lang);
      });
    });
  }
  
  function setupCopyButton(container, getCodeCallback) {
    const copyBtn = container.querySelector('#curlHelperCopy');
    if (!copyBtn) return;
    
    copyBtn.addEventListener('click', async () => {
      const code = getCodeCallback();
      try {
        await navigator.clipboard.writeText(code);
        const originalText = copyBtn.textContent;
        copyBtn.textContent = '✓ Copied!';
        copyBtn.classList.add('btn-success');
        setTimeout(() => {
          copyBtn.textContent = originalText;
          copyBtn.classList.remove('btn-success');
        }, 2000);
      } catch (err) {
        console.error('Failed to copy:', err);
        copyBtn.textContent = '✗ Failed';
        setTimeout(() => {
          copyBtn.textContent = '📋 Copy';
        }, 2000);
      }
    });
  }
  
  // --------- Main initialization ---------
  
  function initCurlHelper() {
    // Find the form container
    const runWrap = document.querySelector('[data-run-wrap]');
    if (!runWrap) return;
    
    // Check if already initialized
    if (document.getElementById('curlHelperContainer')) return;
    
    const form = runWrap.querySelector('[data-run-form]');
    if (!form) return;
    
    // Create and insert UI
    const helperUI = createCurlHelperUI();
    runWrap.appendChild(helperUI);
    
    let currentLang = 'curl';
    let cachedData = null;
    
    function updateCode() {
      cachedData = extractFormData(form);
      if (!cachedData) return;
      
      const { task, modelId, payload, extraArgs, files } = cachedData;
      
      let code = '';
      if (currentLang === 'curl') {
        code = generateCurl(task, modelId, payload, extraArgs, files);
      } else if (currentLang === 'node') {
        code = generateNodeFetch(task, modelId, payload, extraArgs, files);
      } else if (currentLang === 'python') {
        code = generatePythonRequests(task, modelId, payload, extraArgs, files);
      }
      
      updateCodeExample(helperUI, currentLang, code);
    }
    
    // Setup tab navigation
    setupTabNavigation(helperUI, (lang) => {
      currentLang = lang;
      updateCode();
    });
    
    // Setup copy button
    setupCopyButton(helperUI, () => {
      return helperUI.querySelector('#curlHelperCode')?.textContent || '';
    });
    
    // Initial update
    updateCode();
    
    // Listen for form changes
    form.addEventListener('input', updateCode);
    form.addEventListener('change', updateCode);
  }
  
  // --------- Bootstrap ---------
  
  // Watch for HTMX content swaps (when run form is loaded)
  document.body.addEventListener('htmx:afterSwap', (e) => {
    if (e.detail.target?.matches('[data-form-host]')) {
      // Give the browser a moment to render
      setTimeout(initCurlHelper, 100);
    }
  });
  
  // Also try to init on DOM ready (if form already exists)
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCurlHelper);
  } else {
    initCurlHelper();
  }
})();
