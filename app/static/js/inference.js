/**
 * Inference Modal System
 * 
 * This file manages the interactive inference modal that appears when users click "Run" on a model.
 * 
 * Key Components:
 * - InferenceModal class: Main modal controller
 * - Schema loading: Fetches task-specific form fields from /run endpoint  
 * - Form rendering: Dynamically creates inputs, file uploads, and advanced options
 * - Progress tracking: Shows download/inference/completion status
 * - Result handling: Displays text, JSON, images, audio, video outputs
 * 
 * Flow:
 * 1. User clicks Run button → open(modelId, task) called
 * 2. Load schema from /run?task=<task>
 * 3. Render form fields based on schema
 * 4. User submits → POST /inference with FormData
 * 5. Display results in appropriate format
 */

const MODAL_ID = "runDialog";
const TEMPLATE_ID = "inferenceModalTemplate";
const SCHEMA_ENDPOINT = "/run";

const DEFAULT_DESCRIPTIONS = {
  text: "Send a prompt, receive text back. All responses are copied directly from the model.",
  vision: "Upload an image, run a vision model, and inspect the structured result.",
  audio: "Upload or generate audio and listen to the result in the browser.",
  multimodal: "Blend text with images or other media for multimodal reasoning.",
  video: "Upload a short video clip (mp4) to classify its content.",
};

const KNOWN_TEXT_KEYS = [
  "generated_text",
  "generated_texts",
  "summary_text",
  "translation_text",
  "text",
  "answer",
  "label",
  "reason",
  "message",
  "note",
  "detail",
];

function getPageContext() {
  const ctx = window.__HF_MODELS_PAGE_CONTEXT || {};
  return {
    selectedTask: ctx.selectedTask || "",
    tasks: ctx.tasks || [],
  };
}

async function fetchJson(url) {
  const res = await fetch(url, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      detail = data?.error || data?.reason || detail;
    } catch (err) {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json();
}

function parseChipsValue(raw) {
  if (!raw) return [];
  const parts = raw
    .split(/[\n,]/g)
    .map((p) => p.trim())
    .filter(Boolean);
  return Array.from(new Set(parts));
}

function tryParseJson(raw) {
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch (err) {
    throw new Error("Invalid JSON payload: " + (err?.message || err));
  }
}

function extractTextCandidate(payload) {
  if (payload == null) return null;
  if (typeof payload === "string") {
    return payload;
  }
  if (Array.isArray(payload)) {
    if (!payload.length) return null;
    const first = payload[0];
    if (typeof first === "string") return first;
    if (first && typeof first === "object") {
      for (const key of KNOWN_TEXT_KEYS) {
        if (typeof first[key] === "string" && first[key].trim()) {
          return first[key];
        }
      }
    }
  }
  if (typeof payload === "object") {
    for (const key of KNOWN_TEXT_KEYS) {
      if (typeof payload[key] === "string" && payload[key].trim()) {
        return payload[key];
      }
    }
  }
  return null;
}

function formatDuration(ms) {
  if (ms < 1000) {
    return `${ms.toFixed(0)} ms`;
  }
  const sec = ms / 1000;
  if (sec < 60) {
    return `${sec.toFixed(2)} s`;
  }
  const mins = sec / 60;
  return `${mins.toFixed(2)} min`;
}

class InferenceModal {
  constructor(dialog, template) {
    this.dialog = dialog;
    this.template = template;
    this.schemaCache = new Map();
    this.inputMap = new Map();
    this.fileMap = new Map();
    this.advancedMap = new Map();
    this.currentTask = "";
    this.currentModel = "";
    this.schema = null;
    this.progressDefinition = [];
    this.progressElements = [];
    this.progressIndexById = new Map();
    this.progressTimers = [];
    this.progressCurrentIndex = -1;
    this.root = null;
    this.panel = null;
    this.boundWindowResize = () => this.resizeToContent();
    this.handleAfterHide = () => this.onDialogHide();
    this.resizeObserver =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => this.resizeToContent())
        : null;

    if (this.dialog) {
      this.dialog.addEventListener("sl-after-hide", this.handleAfterHide);
    }
  }

  async open(modelId, task) {
    this.resetState();
    this.currentModel = modelId;
    this.currentTask = task;

    if (!task) {
      this.showMessage("Pick a task first", "Select a task in the dropdown before launching a run.");
      await this.dialog.show();
      return;
    }

    let schema;
    try {
      schema = await this.loadSchema(task);
    } catch (err) {
      this.showMessage("Schema unavailable", err?.message || String(err));
      await this.dialog.show();
      return;
    }
    this.schema = schema;

    const clone = this.template.content.cloneNode(true);
    this.root = clone.querySelector("[data-root]");
    this.form = clone.querySelector("[data-run-form]");
    this.statusEl = clone.querySelector("[data-status]");
    this.inputMount = clone.querySelector("[data-input-mount]");
    this.fileSection = clone.querySelector("[data-file-mount]");
    this.fileGrid = clone.querySelector("[data-file-grid]");
    this.advancedSection = clone.querySelector("[data-advanced]");
    this.advancedGrid = clone.querySelector("[data-advanced-grid]");
    this.cancelBtn = clone.querySelector("[data-cancel]");
    this.runBtn = clone.querySelector("[data-run]");
    this.progressSection = clone.querySelector("[data-progress]");
    this.progressBar = clone.querySelector("[data-progress-bar]");
    this.progressStepsList = clone.querySelector("[data-progress-steps]");
    this.outputSection = clone.querySelector("[data-output]");
    this.outputMeta = clone.querySelector("[data-output-meta]");
    this.outputBody = clone.querySelector("[data-output-body]");
    this.outputEmpty = clone.querySelector("[data-output-empty]");
    this.outputText = clone.querySelector("[data-output-text]");
    this.outputJson = clone.querySelector("[data-output-json]");
    this.outputImage = clone.querySelector("[data-output-image]");
    this.outputAudio = clone.querySelector("[data-output-audio]");
    this.outputVideo = clone.querySelector("[data-output-video]");
    this.outputFiles = clone.querySelector("[data-output-files]");

    const titleEl = clone.querySelector("[data-model-title]");
    titleEl.textContent = modelId;
    titleEl.title = modelId;

    const chip = clone.querySelector("[data-chip]");
    chip.textContent = schema.label || schema.category || "Run";

    const descEl = clone.querySelector("[data-task-description]");
    descEl.textContent = schema.description || DEFAULT_DESCRIPTIONS[schema.category] || "Live inference run.";

    const metaEl = clone.querySelector("[data-model-meta]");
    metaEl.innerHTML = `<span class="muted-badge">Task</span> <code>${task}</code>`;

    this.renderInputs(schema.inputs || []);
    this.renderAdvanced(schema.advanced || []);
    this.renderFiles(schema.files || []);
    this.setupProgress();

    this.form.addEventListener("submit", (ev) => this.handleSubmit(ev));
    this.cancelBtn.addEventListener("click", () => this.dialog.hide());

    this.dialog.innerHTML = "";
    this.dialog.appendChild(clone);
    this.dialog.label = `Run • ${modelId}`;

    if (this.resizeObserver && this.root) {
      this.resizeObserver.observe(this.root);
    }
    window.addEventListener("resize", this.boundWindowResize);

    await this.dialog.show();
    this.panel = this.dialog.shadowRoot?.querySelector('[part="panel"]') || null;
    requestAnimationFrame(() => this.resizeToContent());
  }

  resetState() {
    this.onDialogHide();
    this.inputMap.clear();
    this.fileMap.clear();
    this.advancedMap.clear();
    this.schema = null;
    this.clearProgressTimers();
    this.progressElements = [];
    this.progressDefinition = [];
    this.progressIndexById = new Map();
    this.progressCurrentIndex = -1;
    if (this.dialog) {
      this.dialog.innerHTML = "";
    }
    this.root = null;
  }

  onDialogHide() {
    if (this.resizeObserver && this.root) {
      this.resizeObserver.unobserve(this.root);
    }
    window.removeEventListener("resize", this.boundWindowResize);
    if (this.root) {
      this.root.classList.remove("inference-modal--scrollable");
    }
    if (this.dialog) {
      this.dialog.style.removeProperty("--height");
      this.dialog.style.removeProperty("--max-height");
    }
    if (this.root) {
      const outputBody = this.root.querySelector("[data-output-body]");
      if (outputBody) {
        outputBody.style.removeProperty("max-height");
        outputBody.style.removeProperty("overflow-y");
      }
    }
    if (this.panel) {
      this.panel.style.removeProperty("height");
      this.panel.style.removeProperty("max-height");
      this.panel.style.removeProperty("overflow-y");
      this.panel = null;
    }
  }

  resizeToContent() {
    if (!this.dialog || !this.dialog.open || !this.root) {
      return;
    }
    if (!this.panel) {
      this.panel = this.dialog.shadowRoot?.querySelector('[part="panel"]') || null;
    }
    const outputBody = this.root.querySelector("[data-output-body]");
    this.root.classList.add("inference-modal--scrollable");
    if (outputBody) {
      const panelRect = this.panel
        ? this.panel.getBoundingClientRect()
        : this.root.getBoundingClientRect();
      const outputRect = outputBody.getBoundingClientRect();
      const availableForOutput = Math.max(
        window.innerHeight - (outputRect.top - panelRect.top) - 96,
        200
      );
      outputBody.style.maxHeight = `${availableForOutput}px`;
      outputBody.style.overflowY = "auto";
    }
  }

  async loadSchema(task) {
    if (this.schemaCache.has(task)) {
      return this.schemaCache.get(task);
    }
    const data = await fetchJson(`${SCHEMA_ENDPOINT}?task=${encodeURIComponent(task)}`);
    if (!data || !data.schema) {
      throw new Error("Missing schema definition for task");
    }
    this.schemaCache.set(task, data.schema);
    return data.schema;
  }

  showMessage(title, message) {
    const container = document.createElement("div");
    container.className = "inference-modal__message";
    container.innerHTML = `
      <header class="inference-modal__header">
        <div class="inference-modal__chip">Heads-up</div>
        <h3 class="inference-modal__title">${title}</h3>
        <p class="muted">${message}</p>
      </header>
      <footer class="inference-form__actions">
        <button class="btn-ghost" type="button" data-dismiss>Close</button>
      </footer>
    `;
    this.dialog.innerHTML = "";
    this.dialog.appendChild(container);
    const dismiss = container.querySelector("[data-dismiss]");
    dismiss?.addEventListener("click", () => this.dialog.hide());
    this.dialog.label = title;
  }

  renderInputs(fields) {
    if (!fields.length) {
      this.inputMount.innerHTML = "<div class=\"muted\">This task has no textual inputs.</div>";
      return;
    }
    this.inputMount.innerHTML = "";
    fields.forEach((field) => {
      const wrapper = document.createElement("label");
      wrapper.className = "inference-field";
      wrapper.dataset.name = field.name;

      const label = document.createElement("span");
      label.className = "inference-field__label";
      label.textContent = field.required ? `${field.label} *` : field.label;
      wrapper.appendChild(label);

      let control;
      switch (field.type) {
        case "textarea": {
          control = document.createElement("textarea");
          control.rows = field.rows || 5;
          break;
        }
        case "json": {
          control = document.createElement("textarea");
          control.rows = field.rows || 6;
          control.dataset.role = "json";
          break;
        }
        case "chips": {
          control = document.createElement("textarea");
          control.rows = field.rows || 3;
          control.dataset.role = "chips";
          break;
        }
        case "select": {
          control = document.createElement("select");
          (field.options || []).forEach((opt) => {
            const option = document.createElement("option");
            option.value = opt.value;
            option.textContent = opt.label;
            if (opt.value === field.default) {
              option.selected = true;
            }
            control.appendChild(option);
          });
          break;
        }
        default: {
          control = document.createElement("input");
          control.type = field.type || "text";
          break;
        }
      }

      control.name = field.name;
      control.required = Boolean(field.required);
      control.placeholder = field.placeholder || "";
      control.autocomplete = "off";
      if (field.default != null) {
        control.value = field.default;
      }
      if (field.maxLength) {
        control.maxLength = field.maxLength;
      }

      wrapper.appendChild(control);

      if (field.help) {
        const help = document.createElement("div");
        help.className = "inference-field__help muted";
        help.textContent = field.help;
        wrapper.appendChild(help);
      }

      this.inputMount.appendChild(wrapper);
      this.inputMap.set(field.name, { field, element: control });
    });
  }

  renderAdvanced(fields) {
    if (!fields.length) {
      this.advancedSection.hidden = true;
      return;
    }
    this.advancedSection.hidden = false;
    this.advancedGrid.innerHTML = "";
    fields.forEach((field) => {
      const wrapper = document.createElement("label");
      wrapper.className = "inference-field inference-field--inline";

      const label = document.createElement("span");
      label.className = "inference-field__label";
      label.textContent = field.label;
      wrapper.appendChild(label);

      const control = document.createElement("input");
      control.type = field.type || "number";
      control.name = field.name;
      control.required = Boolean(field.required);
      if (field.min != null) control.min = field.min;
      if (field.max != null) control.max = field.max;
      if (field.step != null) control.step = field.step;
      if (field.default != null) control.value = field.default;
      wrapper.appendChild(control);

      if (field.help) {
        const help = document.createElement("div");
        help.className = "inference-field__help muted";
        help.textContent = field.help;
        wrapper.appendChild(help);
      }

      this.advancedGrid.appendChild(wrapper);
      this.advancedMap.set(field.name, { field, element: control });
    });
  }

  renderFiles(files) {
    if (!files.length) {
      this.fileSection.hidden = true;
      return;
    }
    this.fileSection.hidden = false;
    this.fileGrid.innerHTML = "";
    files.forEach((fileField) => {
      const wrapper = document.createElement("label");
      wrapper.className = "inference-field inference-field--file";

      const label = document.createElement("span");
      label.className = "inference-field__label";
      label.textContent = fileField.required ? `${fileField.label} *` : fileField.label;
      wrapper.appendChild(label);

      const input = document.createElement("input");
      input.type = "file";
      input.name = fileField.name;
      input.accept = fileField.accept || "";
      input.required = Boolean(fileField.required);
      wrapper.appendChild(input);

      const preview = document.createElement("div");
      preview.className = "inference-file-preview muted";
      preview.textContent = fileField.help || "";
      wrapper.appendChild(preview);

      input.addEventListener("change", () => {
        preview.innerHTML = "";
        const file = input.files?.[0];
        if (!file) {
          preview.textContent = fileField.help || "";
          return;
        }
        const info = document.createElement("div");
        info.className = "inference-file-preview__meta";
        info.textContent = `${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
        preview.appendChild(info);

        if (fileField.preview === "image") {
          const img = document.createElement("img");
          img.className = "inference-file-preview__image";
          img.alt = "Selected image";
          img.src = URL.createObjectURL(file);
          preview.appendChild(img);
        } else if (fileField.preview === "audio") {
          const audio = document.createElement("audio");
          audio.controls = true;
          audio.src = URL.createObjectURL(file);
          preview.appendChild(audio);
        } else if (fileField.preview === "video") {
          const video = document.createElement("video");
          video.controls = true;
          video.src = URL.createObjectURL(file);
          preview.appendChild(video);
        }
      });

      this.fileGrid.appendChild(wrapper);
      this.fileMap.set(fileField.name, { field: fileField, element: input });
    });
  }

  setupProgress() {
    if (!this.progressStepsList) return;
    this.clearProgressTimers();
    this.progressDefinition = [
      {
        id: "prepare",
        label: "Prepare request",
        description: "Validating inputs and packaging payload.",
      },
      {
        id: "download",
        label: "Load model",
        description: "Downloading weights or warming cache if needed.",
      },
      {
        id: "inference",
        label: "Run inference",
        description: "Executing the pipeline on the selected device.",
      },
      {
        id: "finalize",
        label: "Finalize",
        description: "Formatting response and preparing downloads.",
      },
    ];
    this.progressIndexById = new Map();
    this.progressElements = [];
    this.progressStepsList.innerHTML = "";
    this.progressDefinition.forEach((step, idx) => {
      const li = document.createElement("li");
      li.className = "inference-progress__step";

      const title = document.createElement("span");
      title.className = "inference-progress__step-title";
      title.textContent = step.label;
      li.appendChild(title);

      const desc = document.createElement("p");
      desc.className = "inference-progress__step-desc";
      desc.textContent = step.description;
      li.appendChild(desc);

      this.progressStepsList.appendChild(li);
      this.progressElements.push({ root: li, desc, def: step });
      this.progressIndexById.set(step.id, idx);
    });

    if (this.progressSection) {
      this.progressSection.hidden = true;
    }
    if (this.progressBar) {
      this.progressBar.style.width = "0%";
    }
    this.progressCurrentIndex = -1;
  }

  startProgress() {
    if (!this.progressElements.length) {
      this.setupProgress();
    }
    if (!this.progressElements.length) return;
    this.setProgressById("prepare", "Validating inputs…");
    this.progressTimers.push(
      window.setTimeout(() => {
        this.setProgressById(
          "download",
          "Downloading or loading model weights (only first run)."
        );
      }, 1200)
    );
  }

  setProgressById(id, message) {
    const idx = this.progressIndexById.get(id);
    if (idx == null) return;
    this.setProgress(idx, { message });
  }

  setProgress(index, { message, complete = false } = {}) {
    if (!this.progressElements.length) return;
    const total = this.progressElements.length;
    const clamped = Math.max(0, Math.min(index, total - 1));
    this.progressCurrentIndex = clamped;

    if (this.progressSection) {
      this.progressSection.hidden = false;
    }

    this.progressElements.forEach((entry, idx) => {
      const isActive = idx === clamped;
      const isComplete = idx < clamped || (complete && idx === clamped);
      if (!entry.root.classList.contains("is-error")) {
        entry.root.classList.toggle("is-complete", isComplete);
        entry.root.classList.toggle("is-active", isActive && !complete);
      }
      if (isActive && message) {
        entry.desc.textContent = message;
      } else if (!isActive && !entry.root.classList.contains("is-error")) {
        entry.desc.textContent = entry.def.description;
      }
    });

    if (this.progressBar) {
      let pct = total > 1 ? (clamped / (total - 1)) * 100 : 100;
      if (complete) pct = 100;
      this.progressBar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    }
  }

  completeProgress(durationLabel) {
    if (!this.progressElements.length) return;
    this.clearProgressTimers();
    const lastIndex = this.progressElements.length - 1;
    this.setProgress(lastIndex, {
      message: durationLabel || "Finished",
      complete: true,
    });
    this.progressElements.forEach((entry) => {
      entry.root.classList.remove("is-error");
      entry.root.classList.add("is-complete");
      entry.root.classList.remove("is-active");
    });
    if (this.progressBar) {
      this.progressBar.style.width = "100%";
    }
  }

  failProgress(message) {
    if (!this.progressElements.length) return;
    this.clearProgressTimers();
    const index = this.progressCurrentIndex >= 0 ? this.progressCurrentIndex : 0;
    this.setProgress(index, { message: message || "Run failed" });
    const entry = this.progressElements[index];
    if (entry) {
      entry.root.classList.add("is-error");
      entry.root.classList.remove("is-active");
    }
    if (this.progressBar) {
      const total = this.progressElements.length;
      const pct = total > 1 ? (index / (total - 1)) * 100 : 0;
      this.progressBar.style.width = `${pct}%`;
    }
  }

  clearProgressTimers() {
    this.progressTimers.forEach((t) => window.clearTimeout(t));
    this.progressTimers = [];
  }

  async handleSubmit(event) {
    event.preventDefault();
    if (!this.schema) return;

    const payload = {};
    try {
      for (const [name, entry] of this.inputMap.entries()) {
        const { field, element } = entry;
        const value = element.value.trim();
        if (!value && field.required) {
          throw new Error(`${field.label} is required.`);
        }
        if (!value) continue;
        if (element.dataset.role === "chips") {
          payload[name] = parseChipsValue(value);
        } else if (element.dataset.role === "json") {
          payload[name] = tryParseJson(value);
        } else {
          payload[name] = value;
        }
      }
      for (const [name, entry] of this.advancedMap.entries()) {
        const { element } = entry;
        if (!element.value) continue;
        const number = Number(element.value);
        if (!Number.isFinite(number)) {
          throw new Error(`Invalid numeric value for ${name}`);
        }
        payload[name] = number;
      }
    } catch (err) {
      this.showStatus(err?.message || String(err), true);
      return;
    }

    const formData = new FormData();
    formData.append(
      "spec",
      JSON.stringify({
        model_id: this.currentModel,
        task: this.currentTask,
        payload,
      })
    );

    // Check if remote inference is enabled
    const useRemoteCheckbox = this.form?.querySelector('[data-use-remote]');
    if (useRemoteCheckbox?.checked) {
      formData.append("use_remote", "true");
    }

    for (const [name, entry] of this.fileMap.entries()) {
      const file = entry.element.files?.[0];
      if (!file) {
        if (entry.field.required) {
          this.showStatus(`${entry.field.label} is required.`, true);
          return;
        }
        continue;
      }
      formData.append(name, file, file.name);
    }

    this.setRunning(true);
    const started = performance.now();
    let response;
    try {
      response = await fetch("/inference", {
        method: "POST",
        body: formData,
      });
    } catch (err) {
      this.setRunning(false);
      this.showStatus(err?.message || String(err), true);
      return;
    }

    const elapsed = performance.now() - started;
    await this.handleResponse(response, elapsed);
    this.setRunning(false);
  }

  async handleResponse(response, elapsedMs) {
    const duration = formatDuration(elapsedMs);
    const contentType = response.headers.get("content-type") || "";
    const disposition = response.headers.get("content-disposition") || "";

    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        if (contentType.includes("application/json")) {
          const data = await response.json();
          // Handle nested error objects
          if (data?.detail && typeof data.detail === 'object') {
            // If detail is an object, try to extract meaningful message
            detail = data.detail.reason || data.detail.error || data.detail.message || JSON.stringify(data.detail);
          } else {
            detail = data?.detail || data?.reason || data?.error || data?.message || detail;
          }
          // If it's still an object, convert to string
          if (typeof detail === 'object') {
            detail = JSON.stringify(detail, null, 2);
          }
        } else {
          detail = await response.text();
        }
      } catch (err) {
        // ignore
      }
      this.failProgress(String(detail));
      this.showStatus(String(detail), true);
      return;
    }

    this.setProgressById("inference", "Running inference on server…");
    this.outputSection.hidden = false;
    this.outputMeta.textContent = `Finished in ${duration}`;
    this.outputEmpty.hidden = true;
    this.resetOutputViews();

    if (contentType.startsWith("image/")) {
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      this.outputImage.src = url;
      this.outputImage.hidden = false;
      this.setProgressById("finalize", "Rendering generated image preview.");
      this.completeProgress(`Finished in ${duration}`);
      this.showStatus("Image generated", false);
      this.addDownloadLink(disposition, blob);
      this.resizeToContent();
      return;
    }

    if (contentType.startsWith("audio/")) {
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      this.outputAudio.src = url;
      this.outputAudio.hidden = false;
      this.setProgressById("finalize", "Audio ready for playback.");
      this.completeProgress(`Finished in ${duration}`);
      this.showStatus("Audio generated", false);
      this.addDownloadLink(disposition, blob);
      this.resizeToContent();
      return;
    }

    if (contentType.startsWith("video/")) {
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      this.outputVideo.src = url;
      this.outputVideo.hidden = false;
      this.setProgressById("finalize", "Video ready for playback.");
      this.completeProgress(`Finished in ${duration}`);
      this.showStatus("Video generated", false);
      this.addDownloadLink(disposition, blob);
      this.resizeToContent();
      return;
    }

    if (contentType.includes("application/json")) {
      const data = await response.json();
      const status = this.statusFromJson(data);
      this.renderJsonResult(data);
      if (status.isError) {
        this.failProgress(status.message);
      } else {
        this.setProgressById(
          "finalize",
          status.message || "Finalizing output"
        );
        this.completeProgress(`Finished in ${duration}`);
      }
      this.showStatus(status.message, status.isError);
      this.resizeToContent();
      return;
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    this.outputFiles.hidden = false;
    this.outputFiles.innerHTML = "";
    const link = document.createElement("a");
    link.href = url;
    link.textContent = disposition || "Download file";
    link.download = this.extractFilename(disposition) || `${this.currentTask}_output.bin`;
    link.className = "btn-ghost";
    this.outputFiles.appendChild(link);
    this.setProgressById("finalize", "File prepared for download.");
    this.completeProgress(`Finished in ${duration}`);
    this.showStatus("File ready", false);
    this.resizeToContent();
  }

  renderJsonResult(data) {
    const textCandidate = extractTextCandidate(data);

    if (textCandidate) {
      this.outputText.textContent = textCandidate;
      this.outputText.hidden = false;
    } else {
      this.outputText.hidden = true;
    }

    this.outputJson.textContent = JSON.stringify(data, null, 2);
    this.outputJson.hidden = false;
    this.resizeToContent();
  }

  statusFromJson(data) {
    if (data && typeof data === "object") {
      if (data.error) {
        const reason = typeof data.reason === "string" ? data.reason : data.error;
        return { message: `Error: ${reason}`, isError: true };
      }
      if (data.skipped) {
        const reason = typeof data.reason === "string" ? data.reason : "Model skipped";
        return { message: `Skipped: ${reason}`, isError: false };
      }
      if (typeof data.detail === "string") {
        return { message: data.detail, isError: false };
      }
      if (Array.isArray(data.detail) && data.detail.length) {
        const first = data.detail[0];
        if (typeof first === "string") {
          return { message: first, isError: false };
        }
        if (first && typeof first === "object") {
          const msg = first.msg || first.message || first.detail;
          if (msg) {
            return { message: String(msg), isError: true };
          }
        }
      }
    }
    return { message: "JSON result ready", isError: false };
  }

  resetOutputViews() {
    this.outputText.hidden = true;
    this.outputJson.hidden = true;
    this.outputImage.hidden = true;
    this.outputAudio.hidden = true;
    this.outputVideo.hidden = true;
    this.outputFiles.hidden = true;
    this.outputFiles.innerHTML = "";
  }

  addDownloadLink(disposition, blob) {
    const filename = this.extractFilename(disposition) || `${this.currentTask}_output`;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.textContent = `Download ${filename}`;
    link.className = "btn-ghost";
    this.outputFiles.hidden = false;
    this.outputFiles.innerHTML = "";
    this.outputFiles.appendChild(link);
    this.resizeToContent();
  }

  extractFilename(disposition) {
    if (!disposition) return "";
    const match = /filename\*?="?([^";]+)"?/i.exec(disposition);
    if (match && match[1]) {
      return decodeURIComponent(match[1]);
    }
    return "";
  }

  showStatus(message, isError) {
    if (!this.statusEl) return;
    this.statusEl.hidden = false;
    this.statusEl.textContent = message;
    this.statusEl.className = isError
      ? "inference-form__status inference-form__status--error"
      : "inference-form__status";
  }

  setRunning(running) {
    if (!this.runBtn) return;
    this.runBtn.disabled = running;
    this.cancelBtn.disabled = running;
    if (running) {
      this.runBtn.dataset.loading = "true";
      this.showStatus("Running inference…", false);
      this.startProgress();
    } else {
      delete this.runBtn.dataset.loading;
      this.clearProgressTimers();
    }
  }
}

function setupRunModal() {
  const dialog = document.getElementById(MODAL_ID);
  const template = document.getElementById(TEMPLATE_ID);
  if (!dialog || !template) return;

  if (dialog.parentElement !== document.body) {
    document.body.appendChild(dialog);
  }
  if (window.customElements && typeof customElements.whenDefined === "function") {
    customElements.whenDefined("sl-dialog").catch(() => {});
  }

  const modal = new InferenceModal(dialog, template);

  window.runModel = async (modelId) => {
    const ctx = getPageContext();
    const task = ctx.selectedTask;
    await modal.open(modelId, task);
  };

  window.__HF_UPDATE_SELECTED_TASK = (task) => {
    if (!window.__HF_MODELS_PAGE_CONTEXT) {
      window.__HF_MODELS_PAGE_CONTEXT = {};
    }
    window.__HF_MODELS_PAGE_CONTEXT.selectedTask = task;
  };

  dialog.addEventListener("sl-after-hide", () => {
    dialog.innerHTML = "";
  });

  window.__HF_PROGRESS_API = {
    advance: (id, message) => {
      modal.setProgressById(id, message);
    },
    error: (message) => {
      modal.failProgress(message);
    },
  };
}

document.addEventListener("DOMContentLoaded", setupRunModal);
