(function () {
  "use strict";

  const dialog = document.getElementById("runDialog");
  const template = document.getElementById("inferenceModalTemplate");
  const pageContext = window.__HF_MODELS_PAGE_CONTEXT || {};

  if (!dialog || !template) {
    window.runModel = function noop(modelId) { // graceful fallback if markup missing
      console.warn("Run modal not mounted; cannot open modal for", modelId);
    };
    return;
  }

  const schemaCache = new Map();
  const objectUrls = new Set();
  const PROGRESS_STEPS = [
    { title: "Prepare inputs", desc: "Validating form values." },
    { title: "Upload & queue", desc: "Sending payload to the runner." },
    { title: "Inference", desc: "Waiting on model output." },
    { title: "Complete", desc: "Rendering the latest result." }
  ];

  const state = {
    initialized: false,
    submitting: false,
    task: null,
    modelId: null,
    fieldDefs: [],
    fileDefs: [],
    fileInputs: new Map(),
    defaultRunText: "Run inference"
  };

  const refs = {
    root: null,
    form: null,
    inputMount: null,
    fileSection: null,
    fileGrid: null,
    advancedSection: null,
    advancedGrid: null,
    status: null,
    progress: null,
    progressBar: null,
    progressSteps: null,
    cancelButton: null,
    runButton: null,
    runLabel: null,
    output: null,
    outputMeta: null,
    outputBody: null,
    outputEmpty: null,
    outputText: null,
    outputJson: null,
    outputImage: null,
    outputAudio: null,
    outputVideo: null,
    outputFiles: null,
    chip: null,
    title: null,
    subtitle: null,
    meta: null
  };

  function init() {
    if (state.initialized) {
      return;
    }

    const fragment = template.content.cloneNode(true);
    const root = fragment.querySelector("[data-root]");
    if (!root) {
      return;
    }

    dialog.innerHTML = "";
    dialog.appendChild(root);

    refs.root = root;
    refs.form = root.querySelector("[data-run-form]");
    refs.inputMount = root.querySelector("[data-input-mount]");
    refs.fileSection = root.querySelector("[data-file-mount]");
    refs.fileGrid = root.querySelector("[data-file-grid]");
    refs.advancedSection = root.querySelector("[data-advanced]");
    refs.advancedGrid = root.querySelector("[data-advanced-grid]");
    refs.status = root.querySelector("[data-status]");
    refs.progress = root.querySelector("[data-progress]");
    refs.progressBar = root.querySelector("[data-progress-bar]");
    refs.progressSteps = root.querySelector("[data-progress-steps]");
    refs.cancelButton = root.querySelector("[data-cancel]");
    refs.runButton = root.querySelector("[data-run]");
    refs.runLabel = refs.runButton ? refs.runButton.querySelector(".neon-run__label") : null;
    refs.output = root.querySelector("[data-output]");
    refs.outputMeta = root.querySelector("[data-output-meta]");
    refs.outputBody = root.querySelector("[data-output-body]");
    refs.outputEmpty = root.querySelector("[data-output-empty]");
    refs.outputText = root.querySelector("[data-output-text]");
    refs.outputJson = root.querySelector("[data-output-json]");
    refs.outputImage = root.querySelector("[data-output-image]");
    refs.outputAudio = root.querySelector("[data-output-audio]");
    refs.outputVideo = root.querySelector("[data-output-video]");
    refs.outputFiles = root.querySelector("[data-output-files]");
    refs.chip = root.querySelector("[data-chip]");
    refs.title = root.querySelector("[data-model-title]");
    refs.subtitle = root.querySelector("[data-task-description]");
    refs.meta = root.querySelector("[data-model-meta]");

    if (refs.runLabel && refs.runLabel.textContent) {
      state.defaultRunText = refs.runLabel.textContent.trim() || state.defaultRunText;
    }

    attachListeners();
    state.initialized = true;
  }

  function attachListeners() {
    if (refs.form) {
      refs.form.addEventListener("submit", handleSubmit);
    }
    if (refs.cancelButton) {
      refs.cancelButton.addEventListener("click", function () {
        dialog.hide();
      });
    }
    dialog.addEventListener("sl-after-hide", resetForNextOpen);
    dialog.addEventListener("sl-request-close", function (event) {
      if (state.submitting) {
        event.preventDefault();
      }
    });
  }

  function getSelectedTask() {
    const selected = typeof pageContext.selectedTask === "string" ? pageContext.selectedTask.trim() : "";
    return selected || "";
  }

  function open(modelId) {
    init();
    state.modelId = modelId;
    state.task = getSelectedTask();

    resetStatus();
    resetProgress();
    resetOutput();
    resetFormView();

    dialog.label = "Run " + modelId;

    setFormInteractive(false);

    if (!state.task) {
      setStatus("Pick a task from the toolbar before running a model.", "error");
      dialog.show();
      return;
    }

    setStatus("Loading form…", "info");
    dialog.show();

    fetchSchema(state.task)
      .then(function (schema) {
        applySchema(modelId, schema);
        setStatus("", "info");
        setFormInteractive(true);
      })
      .catch(function (error) {
        setStatus(error.message || "Failed to load task form.", "error");
        setFormInteractive(false);
      });
  }

  function fetchSchema(task) {
    if (schemaCache.has(task)) {
      return Promise.resolve(schemaCache.get(task));
    }

    return fetch("/run?task=" + encodeURIComponent(task), { credentials: "same-origin" })
      .then(function (response) {
        if (!response.ok) {
          return response.json().catch(function () { return {}; }).then(function (body) {
            const detail = body && (body.error || body.reason || body.detail);
            throw new Error(detail ? String(detail) : "Unable to fetch schema.");
          });
        }
        return response.json();
      })
      .then(function (payload) {
        if (!payload || typeof payload !== "object" || !payload.schema) {
          throw new Error("Malformed schema response.");
        }
        schemaCache.set(task, payload.schema);
        return payload.schema;
      });
  }

  function applySchema(modelId, schema) {
    const safeSchema = schema || {};
    state.fieldDefs = Array.isArray(safeSchema.inputs) ? safeSchema.inputs.slice() : [];
    state.fileDefs = Array.isArray(safeSchema.files) ? safeSchema.files.slice() : [];

    renderHeader(modelId, safeSchema);
    renderInputs(state.fieldDefs);
    renderFiles(state.fileDefs);
    renderAdvanced(Array.isArray(safeSchema.advanced) ? safeSchema.advanced : []);

    if (refs.runLabel) {
      refs.runLabel.textContent = state.defaultRunText;
    }

    const firstInput = refs.form ? refs.form.querySelector("textarea, input, select") : null;
    if (firstInput && typeof firstInput.focus === "function") {
      firstInput.focus();
    }
  }

  function renderHeader(modelId, schema) {
    if (refs.title) {
      refs.title.textContent = modelId;
      refs.title.setAttribute("title", modelId);
    }
    if (refs.chip) {
      const label = schema.label || (schema.category ? schema.category : "Run");
      refs.chip.textContent = label;
    }
    if (refs.subtitle) {
      refs.subtitle.textContent = schema.description || "";
    }
    if (refs.meta) {
      var pieces = [];
      if (schema.category) {
        pieces.push("Category · " + schema.category);
      }
      if (state.task) {
        pieces.push("Task · " + state.task);
      }
      refs.meta.textContent = pieces.join("  •  ");
    }
  }

  function renderInputs(defs) {
    if (!refs.inputMount) {
      return;
    }
    refs.inputMount.innerHTML = "";
    if (!Array.isArray(defs) || defs.length === 0) {
      return;
    }

    defs.forEach(function (field) {
      const wrapper = document.createElement("div");
      wrapper.className = "inference-field";
      if (field && field.inline) {
        wrapper.classList.add("inference-field--inline");
      }

      const labelEl = document.createElement("label");
      labelEl.className = "inference-field__label";
      labelEl.textContent = field.label || field.name;
      labelEl.setAttribute("for", field.name);
      wrapper.appendChild(labelEl);

      const inputEl = createInputControl(field);
      wrapper.appendChild(inputEl);

      if (field.help) {
        const help = document.createElement("div");
        help.className = "inference-field__help muted";
        help.textContent = field.help;
        wrapper.appendChild(help);
      } else if (field.type === "chips") {
        const hint = document.createElement("div");
        hint.className = "inference-field__help muted";
        hint.textContent = "Comma separated list.";
        wrapper.appendChild(hint);
      }

      refs.inputMount.appendChild(wrapper);
    });
  }

  function createInputControl(field) {
    const type = field.type || "text";
    const required = field.required !== false;
    const placeholder = field.placeholder || "";
    const name = field.name;
    const rows = field.rows || 4;

    let control;
    if (type === "textarea" || type === "json") {
      control = document.createElement("textarea");
      control.rows = rows;
      if (type === "json") {
        control.setAttribute("spellcheck", "false");
      }
    } else {
      control = document.createElement("input");
      control.type = type === "number" ? "number" : "text";
    }

    control.name = name;
    if (placeholder) {
      control.placeholder = placeholder;
    }
    if (required) {
      control.required = true;
    }

    return control;
  }

  function renderFiles(defs) {
    if (!refs.fileGrid || !refs.fileSection) {
      return;
    }
    refs.fileGrid.innerHTML = "";
    state.fileInputs.forEach(function (entry) {
      if (entry && entry.url) {
        revokeObjectUrl(entry.url);
      }
    });
    state.fileInputs = new Map();

    if (!Array.isArray(defs) || defs.length === 0) {
      refs.fileSection.hidden = true;
      return;
    }

    refs.fileSection.hidden = false;

    defs.forEach(function (field) {
      const required = field.required !== false;
      const wrapper = document.createElement("div");
      wrapper.className = "inference-field inference-field--file";

      const labelEl = document.createElement("label");
      labelEl.className = "inference-field__label";
      labelEl.textContent = field.label || field.name;
      labelEl.setAttribute("for", field.name);
      wrapper.appendChild(labelEl);

      const input = document.createElement("input");
      input.type = "file";
      input.name = field.name;
      if (required) {
        input.required = true;
      }
      if (field.accept) {
        input.accept = field.accept;
      }
      wrapper.appendChild(input);

      const preview = document.createElement("div");
      preview.className = "inference-file-preview";
      const meta = document.createElement("div");
      meta.className = "inference-file-preview__meta muted";
      meta.textContent = "No file selected";
      preview.appendChild(meta);

      let viewer = null;
      if (field.preview === "image") {
        viewer = document.createElement("img");
        viewer.className = "inference-file-preview__image";
        viewer.hidden = true;
        preview.appendChild(viewer);
      } else if (field.preview === "audio") {
        viewer = document.createElement("audio");
        viewer.className = "inference-file-preview__audio";
        viewer.controls = true;
        viewer.hidden = true;
        preview.appendChild(viewer);
      } else if (field.preview === "video") {
        viewer = document.createElement("video");
        viewer.className = "inference-file-preview__video";
        viewer.controls = true;
        viewer.hidden = true;
        preview.appendChild(viewer);
      }

      wrapper.appendChild(preview);
      refs.fileGrid.appendChild(wrapper);

      state.fileInputs.set(field.name, {
        input: input,
        meta: meta,
        viewer: viewer,
        url: null
      });

      input.addEventListener("change", function () {
        updateFilePreview(field.name);
      });
    });
  }

  function renderAdvanced(defs) {
    if (!refs.advancedSection || !refs.advancedGrid) {
      return;
    }
    refs.advancedGrid.innerHTML = "";
    if (!Array.isArray(defs) || defs.length === 0) {
      refs.advancedSection.hidden = true;
      return;
    }
    refs.advancedSection.hidden = false;

    defs.forEach(function (field) {
      const wrapper = document.createElement("div");
      wrapper.className = "inference-field";

      const labelEl = document.createElement("label");
      labelEl.className = "inference-field__label";
      labelEl.textContent = field.label || field.name;
      labelEl.setAttribute("for", field.name);
      wrapper.appendChild(labelEl);

      const control = createInputControl(field);
      wrapper.appendChild(control);
      refs.advancedGrid.appendChild(wrapper);
    });
  }

  function updateFilePreview(name) {
    const entry = state.fileInputs.get(name);
    if (!entry) {
      return;
    }

    const files = entry.input.files;
    if (!files || files.length === 0) {
      entry.meta.textContent = "No file selected";
      clearViewer(entry);
      return;
    }

    const file = files[0];
    entry.meta.textContent = file.name + " · " + formatBytes(file.size);

    if (!entry.viewer) {
      return;
    }

    clearViewer(entry);
    const url = URL.createObjectURL(file);
    entry.viewer.src = url;
    entry.viewer.hidden = false;
    entry.url = url;
    objectUrls.add(url);
  }

  function clearViewer(entry) {
    if (!entry || !entry.viewer) {
      return;
    }
    if (entry.url) {
      revokeObjectUrl(entry.url);
      entry.url = null;
    }
    if (entry.viewer.tagName === "AUDIO" || entry.viewer.tagName === "VIDEO") {
      entry.viewer.pause();
      entry.viewer.removeAttribute("src");
      entry.viewer.load();
    } else {
      entry.viewer.removeAttribute("src");
    }
    entry.viewer.hidden = true;
  }

  function handleSubmit(event) {
    event.preventDefault();
    if (state.submitting) {
      return;
    }

    resetStatus();

    let payload;
    try {
      payload = collectPayload();
    } catch (error) {
      setStatus(error.message || "Form validation failed.", "error");
      if (error.field && refs.form && refs.form.elements[error.field]) {
        refs.form.elements[error.field].focus();
      }
      return;
    }

    const formData = new FormData();
    const spec = {
      model_id: state.modelId,
      task: state.task,
      payload: payload
    };

    formData.append("spec", JSON.stringify(spec));

    for (var i = 0; i < state.fileDefs.length; i += 1) {
      const field = state.fileDefs[i];
      const entry = state.fileInputs.get(field.name);
      if (!entry) {
        continue;
      }
      if (!entry.input.files || entry.input.files.length === 0) {
        if (field.required !== false) {
          entry.input.focus();
          setStatus("File required: " + (field.label || field.name), "error");
          return;
        }
        continue;
      }
      formData.append(field.name, entry.input.files[0]);
    }

    setSubmitting(true);
    prepareProgress();
    setProgressStep(1);
    setStatus("Uploading payload…", "info");

    fetch("/inference", {
      method: "POST",
      body: formData
    })
      .then(function (response) {
        setProgressStep(2);
        const contentType = response.headers.get("content-type") || "";
        if (!response.ok) {
          return response
            .json()
            .catch(function () { return {}; })
            .then(function (body) {
              const detail = body && (body.error || body.reason || body.detail);
              throw new Error(detail ? String(detail) : "Inference failed.");
            });
        }

        if (contentType.includes("application/json")) {
          return response.json().then(function (data) {
            return { kind: "json", data: data };
          });
        }

        const disposition = response.headers.get("content-disposition") || "";
        const filename = extractFilename(disposition) || "output.bin";
        return response.blob().then(function (blob) {
          const resolvedType = blob.type || contentType;
          return { kind: "blob", data: blob, filename: filename, contentType: resolvedType };
        });
      })
      .then(function (result) {
        if (!result || typeof result !== "object") {
          throw new Error("Empty inference response.");
        }
        setProgressStep(PROGRESS_STEPS.length - 1);
        setStatus("Inference complete", "success");
        if (result.kind === "json") {
          renderJsonOutput(result.data);
        } else if (result.kind === "blob") {
          renderBlobOutput(result.data, result.filename, result.contentType);
        }
      })
      .catch(function (error) {
        setStatus(error.message || "Inference failed.", "error");
      })
      .finally(function () {
        setSubmitting(false);
      });
  }

  function collectPayload() {
    const payload = {};
    if (!refs.form) {
      return payload;
    }

    for (var i = 0; i < state.fieldDefs.length; i += 1) {
      const field = state.fieldDefs[i];
      const element = refs.form.elements[field.name];
      if (!element) {
        continue;
      }
      const raw = typeof element.value === "string" ? element.value.trim() : "";
      const required = field.required !== false;

      if (!raw) {
        if (field.type === "chips") {
          if (required) {
            const err = new Error("Enter at least one label for " + (field.label || field.name) + ".");
            err.field = field.name;
            throw err;
          }
          payload[field.name] = [];
          continue;
        }
        if (field.type === "json") {
          if (required) {
            const err = new Error("JSON required for " + (field.label || field.name) + ".");
            err.field = field.name;
            throw err;
          }
          continue;
        }
        if (required) {
          const err = new Error((field.label || field.name) + " is required.");
          err.field = field.name;
          throw err;
        }
        continue;
      }

      if (field.type === "chips") {
        payload[field.name] = raw.split(",").map(function (part) {
          return part.trim();
        }).filter(Boolean);
        if (required && payload[field.name].length === 0) {
          const err = new Error("Enter at least one label for " + (field.label || field.name) + ".");
          err.field = field.name;
          throw err;
        }
        continue;
      }

      if (field.type === "json") {
        try {
          payload[field.name] = JSON.parse(raw);
        } catch (_) {
          const err = new Error("Invalid JSON in " + (field.label || field.name) + ".");
          err.field = field.name;
          throw err;
        }
        continue;
      }

      payload[field.name] = raw;
    }

    const advanced = refs.advancedGrid ? refs.advancedGrid.querySelectorAll("input, textarea, select") : [];
    if (advanced && advanced.length) {
      for (var j = 0; j < advanced.length; j += 1) {
        const el = advanced[j];
        if (!el.name) {
          continue;
        }
        const val = typeof el.value === "string" ? el.value.trim() : "";
        if (!val) {
          continue;
        }
        payload[el.name] = val;
      }
    }

    return payload;
  }

  function setSubmitting(active) {
    state.submitting = active;
    setFormInteractive(!active);
    if (refs.runLabel) {
      refs.runLabel.textContent = active ? "Running…" : state.defaultRunText;
    }
    if (refs.cancelButton) {
      refs.cancelButton.disabled = active;
    }
  }

  function setFormInteractive(interactive) {
    if (!refs.form) {
      return;
    }
    const fields = refs.form.querySelectorAll("input, textarea, select");
    fields.forEach(function (el) {
      if (el === refs.cancelButton) {
        return;
      }
      if (interactive) {
        el.removeAttribute("disabled");
      } else {
        el.setAttribute("disabled", "disabled");
      }
    });
    if (refs.runButton) {
      if (interactive && !state.submitting) {
        refs.runButton.removeAttribute("disabled");
      } else {
        refs.runButton.setAttribute("disabled", "disabled");
      }
    }
  }

  function prepareProgress() {
    if (!refs.progress || !refs.progressSteps || !refs.progressBar) {
      return;
    }
    refs.progress.hidden = false;
    refs.progressSteps.innerHTML = "";
    PROGRESS_STEPS.forEach(function (step, index) {
      const item = document.createElement("li");
      item.className = "inference-progress__step";
      item.dataset.index = String(index);

      const title = document.createElement("div");
      title.className = "inference-progress__step-title";
      title.textContent = step.title;
      item.appendChild(title);

      const desc = document.createElement("div");
      desc.className = "inference-progress__step-desc";
      desc.textContent = step.desc;
      item.appendChild(desc);

      refs.progressSteps.appendChild(item);
    });
    setProgressStep(0);
  }

  function setProgressStep(stepIndex) {
    if (!refs.progressSteps || !refs.progressBar) {
      return;
    }
    const steps = Array.prototype.slice.call(refs.progressSteps.children || []);
    const maxIndex = PROGRESS_STEPS.length - 1;
    const clamped = Math.max(0, Math.min(stepIndex, maxIndex));

    steps.forEach(function (stepEl, idx) {
      stepEl.classList.toggle("is-active", idx === clamped);
      stepEl.classList.toggle("is-complete", idx < clamped);
    });

    const percent = maxIndex === 0 ? 100 : Math.round((clamped / maxIndex) * 100);
    refs.progressBar.style.width = percent + "%";
  }

  function resetProgress() {
    if (refs.progress) {
      refs.progress.hidden = true;
    }
    if (refs.progressSteps) {
      refs.progressSteps.innerHTML = "";
    }
    if (refs.progressBar) {
      refs.progressBar.style.width = "0%";
    }
  }

  function renderJsonOutput(data) {
    if (!refs.output || !refs.outputBody) {
      return;
    }
    refs.output.hidden = false;
    refs.outputEmpty.hidden = true;

    const now = new Date();
    if (refs.outputMeta) {
      refs.outputMeta.textContent = "Updated " + now.toLocaleTimeString();
    }

    if (refs.outputText) {
      const primaryText = extractPrimaryText(data);
      if (primaryText) {
        refs.outputText.textContent = primaryText;
        refs.outputText.hidden = false;
      } else {
        refs.outputText.hidden = true;
      }
    }

    if (refs.outputJson) {
      refs.outputJson.textContent = JSON.stringify(data, null, 2);
      refs.outputJson.hidden = false;
    }

    if (refs.outputImage) {
      refs.outputImage.hidden = true;
      refs.outputImage.removeAttribute("src");
    }
    if (refs.outputAudio) {
      refs.outputAudio.hidden = true;
      refs.outputAudio.removeAttribute("src");
    }
    if (refs.outputVideo) {
      refs.outputVideo.hidden = true;
      refs.outputVideo.removeAttribute("src");
    }
    if (refs.outputFiles) {
      refs.outputFiles.hidden = true;
      refs.outputFiles.innerHTML = "";
    }
  }

  function renderBlobOutput(blob, filename, contentType) {
    if (!refs.output || !refs.outputFiles) {
      return;
    }

    refs.output.hidden = false;
    refs.outputEmpty.hidden = true;

    if (refs.outputMeta) {
      const now = new Date();
      refs.outputMeta.textContent = "Downloaded " + now.toLocaleTimeString();
    }

    if (refs.outputText) {
      refs.outputText.hidden = true;
    }
    if (refs.outputJson) {
      refs.outputJson.hidden = true;
      refs.outputJson.textContent = "";
    }

    const type = (contentType || blob.type || "").toLowerCase();
    const url = URL.createObjectURL(blob);
    objectUrls.add(url);

    let previewHandled = false;

    if (refs.outputImage) {
      if (type.startsWith("image/")) {
        refs.outputImage.src = url;
        refs.outputImage.hidden = false;
        previewHandled = true;
      } else {
        refs.outputImage.hidden = true;
        refs.outputImage.removeAttribute("src");
      }
    }

    if (refs.outputAudio) {
      if (type.startsWith("audio/")) {
        refs.outputAudio.src = url;
        refs.outputAudio.hidden = false;
        refs.outputAudio.load();
        previewHandled = true;
      } else {
        refs.outputAudio.pause();
        refs.outputAudio.hidden = true;
        refs.outputAudio.removeAttribute("src");
        refs.outputAudio.load();
      }
    }

    if (refs.outputVideo) {
      if (type.startsWith("video/")) {
        refs.outputVideo.src = url;
        refs.outputVideo.hidden = false;
        refs.outputVideo.load();
        previewHandled = true;
      } else {
        refs.outputVideo.pause();
        refs.outputVideo.hidden = true;
        refs.outputVideo.removeAttribute("src");
        refs.outputVideo.load();
      }
    }

    if (!previewHandled) {
      if (refs.outputImage) {
        refs.outputImage.hidden = true;
        refs.outputImage.removeAttribute("src");
      }
      if (refs.outputAudio) {
        refs.outputAudio.pause();
        refs.outputAudio.hidden = true;
        refs.outputAudio.removeAttribute("src");
        refs.outputAudio.load();
      }
      if (refs.outputVideo) {
        refs.outputVideo.pause();
        refs.outputVideo.hidden = true;
        refs.outputVideo.removeAttribute("src");
        refs.outputVideo.load();
      }
    }

    if (refs.outputFiles) {
      refs.outputFiles.hidden = false;
      refs.outputFiles.innerHTML = "";
      const link = document.createElement("a");
      link.href = url;
      link.download = filename || "output.bin";
      link.className = "btn-ghost";
      link.textContent = "Download " + (filename || "output");
      refs.outputFiles.appendChild(link);
    }
  }

  function extractPrimaryText(data) {
    if (!data || typeof data !== "object") {
      return "";
    }
    if (typeof data.result === "string") {
      return data.result;
    }
    if (Array.isArray(data) && data.length && typeof data[0] === "string") {
      return data.join("\n");
    }
    if (Array.isArray(data) && data.length && data[0] && typeof data[0].generated_text === "string") {
      return data.map(function (item) { return item.generated_text; }).join("\n\n");
    }
    if (typeof data.generated_text === "string") {
      return data.generated_text;
    }
    return "";
  }

  function renderBlobCleanup() {
    if (refs.outputFiles) {
      refs.outputFiles.innerHTML = "";
      refs.outputFiles.hidden = true;
    }
  }

  function resetOutput() {
    if (refs.output) {
      refs.output.hidden = true;
    }
    if (refs.outputMeta) {
      refs.outputMeta.textContent = "";
    }
    if (refs.outputEmpty) {
      refs.outputEmpty.hidden = false;
    }
    if (refs.outputText) {
      refs.outputText.hidden = true;
      refs.outputText.textContent = "";
    }
    if (refs.outputJson) {
      refs.outputJson.hidden = true;
      refs.outputJson.textContent = "";
    }
    if (refs.outputImage) {
      refs.outputImage.hidden = true;
      refs.outputImage.removeAttribute("src");
    }
    if (refs.outputAudio) {
      refs.outputAudio.hidden = true;
      refs.outputAudio.removeAttribute("src");
    }
    if (refs.outputVideo) {
      refs.outputVideo.hidden = true;
      refs.outputVideo.removeAttribute("src");
    }
    renderBlobCleanup();
  }

  function resetFormView() {
    if (refs.form) {
      refs.form.reset();
    }
    if (refs.inputMount) {
      refs.inputMount.innerHTML = "";
    }
    if (refs.advancedGrid) {
      refs.advancedGrid.innerHTML = "";
    }
    if (refs.advancedSection) {
      refs.advancedSection.hidden = true;
    }
    if (refs.fileGrid) {
      refs.fileGrid.innerHTML = "";
    }
    if (refs.fileSection) {
      refs.fileSection.hidden = true;
    }
    state.fieldDefs = [];
    state.fileDefs = [];
    state.fileInputs.forEach(function (entry) {
      clearViewer(entry);
    });
    state.fileInputs = new Map();
  }

  function resetStatus() {
    if (refs.status) {
      refs.status.hidden = true;
      refs.status.textContent = "";
      refs.status.classList.remove("inference-form__status--error");
    }
  }

  function setStatus(message, variant) {
    if (!refs.status) {
      return;
    }
    const text = (message || "").trim();
    if (!text) {
      refs.status.hidden = true;
      refs.status.textContent = "";
      refs.status.classList.remove("inference-form__status--error");
      return;
    }
    refs.status.hidden = false;
    refs.status.textContent = text;
    refs.status.classList.toggle("inference-form__status--error", variant === "error");
  }

  function resetForNextOpen() {
    setSubmitting(false);
    resetStatus();
    resetProgress();
    resetOutput();
    resetFormView();
    revokeAllObjectUrls();
  }

  function revokeAllObjectUrls() {
    objectUrls.forEach(revokeObjectUrl);
    objectUrls.clear();
  }

  function revokeObjectUrl(url) {
    try {
      URL.revokeObjectURL(url);
    } catch (_) {
      /* ignore */
    }
  }

  function extractFilename(disposition) {
    if (!disposition) {
      return "";
    }
    const match = disposition.match(/filename="?([^";]+)"?/i);
    return match ? match[1] : "";
  }

  function formatBytes(bytes) {
    if (!bytes) {
      return "0 B";
    }
    const units = ["B", "KB", "MB", "GB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    return value.toFixed(value >= 10 || unit === 0 ? 0 : 1) + " " + units[unit];
  }

  window.runModel = open;
})();
