(() => {
  'use strict';
  const navToggle = document.querySelector('[data-nav-toggle]');
  const navigation = document.getElementById('main-navigation');
  if (navToggle && navigation) {
    const compact = window.matchMedia('(max-width:720px)');
    const setNavigation = open => {
      navigation.hidden = compact.matches && !open;
      navToggle.setAttribute('aria-expanded', String(open));
      navToggle.textContent = open ? 'Close menu' : 'Menu';
    };
    navToggle.hidden = false;
    setNavigation(false);
    navToggle.addEventListener('click', () => setNavigation(navToggle.getAttribute('aria-expanded') !== 'true'));
    compact.addEventListener('change', () => setNavigation(false));
    navigation.addEventListener('keydown', event => {
      if (event.key === 'Escape' && compact.matches) { setNavigation(false); navToggle.focus(); }
    });
  }
  const status = document.querySelector('[data-extraction-status]') || document.querySelector('[data-resume-status]');
  const editor = document.getElementById('extracted-text');
  let manuallyEdited = false;
  const confirmReplacement = () => new Promise(resolve => {
    const dialog = document.querySelector('[data-confirm-replace]');
    if (!dialog) { resolve(false); return; }
    dialog.returnValue = 'cancel';
    dialog.addEventListener('close', () => resolve(dialog.returnValue === 'replace'), {once: true});
    dialog.showModal();
  });
  const announce = message => { if (status) status.textContent = message; };
  const countWords = () => {
    const counter = document.querySelector('[data-word-count]');
    if (counter && editor) {
      const count = editor.value.trim() ? editor.value.trim().split(/\s+/u).length : 0;
      counter.textContent = `${count} ${count === 1 ? 'word' : 'words'}`;
    }
  };
  editor?.addEventListener('input', countWords);
  editor?.addEventListener('input', () => { manuallyEdited = true; });
  countWords();
  const showEngine = engine => {
    document.querySelectorAll('[data-engine-tab]').forEach(tab => {
      const selected = tab.dataset.engineTab === engine;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll('[data-engine-panel]').forEach(panel => {
      panel.hidden = panel.dataset.enginePanel !== engine;
    });
  };
  document.querySelectorAll('[data-engine-tab]').forEach(tab => {
    tab.addEventListener('click', () => showEngine(tab.dataset.engineTab));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const engine = event.key === 'Home' ? 'trocr' : event.key === 'End' ? 'tesseract' : tab.dataset.engineTab === 'trocr' ? 'tesseract' : 'trocr';
      showEngine(engine);
      document.querySelector(`[data-engine-tab="${engine}"]`).focus();
    });
  });
  const setEngineResults = results => {
    const container = document.querySelector('[data-recognition-results]');
    if (!container) return;
    container.hidden = !Object.keys(results).length;
    ['trocr', 'tesseract'].forEach(engine => {
      const result = results[engine] || {};
      document.querySelector(`[data-engine-output="${engine}"]`).value = result.text || '';
      document.querySelector(`[data-engine-error="${engine}"]`).textContent = result.error || (result.warnings || []).join(' ');
      document.querySelector(`[data-use-engine="${engine}"]`).disabled = !result.text;
    });
    showEngine(results.trocr?.text ? 'trocr' : 'tesseract');
  };
  document.querySelectorAll('[data-use-engine]').forEach(button => {
    button.disabled = !document.querySelector(`[data-engine-output="${button.dataset.useEngine}"]`).value;
    button.addEventListener('click', async () => {
      const text = document.querySelector(`[data-engine-output="${button.dataset.useEngine}"]`).value;
      if (manuallyEdited && editor.value !== text && !await confirmReplacement()) return;
      editor.value = text;
      manuallyEdited = false;
      countWords();
      announce(`Using ${button.dataset.useEngine === 'trocr' ? 'TrOCR' : 'Tesseract'} result. Review and edit the final text.`);
    });
  });
  showEngine('trocr');
  const engineSelect = document.querySelector('[data-ocr-engine]');
  const updateEngine = () => {
    if (!engineSelect) return;
    const cloud = engineSelect.value === 'gemini';
    const englishOnly = ['trocr', 'hybrid'].includes(engineSelect.value);
    const consent = document.querySelector('[data-cloud-consent]');
    if (consent) consent.hidden = !cloud;
    const checkbox = document.querySelector('[name=cloud_consent]');
    if (checkbox) { checkbox.required = cloud; checkbox.checked = false; }
    document.querySelectorAll('[name=languages]').forEach(input => {
      input.disabled = englishOnly && input.value !== 'eng';
      if (englishOnly) input.checked = input.value === 'eng';
    });
    const help = document.querySelector('[data-engine-help]');
    if (help) help.textContent = cloud ? 'Select one or more document languages. Gemini transcription may contain errors.' : englishOnly ? 'TrOCR supports English only. Processing stays on this computer.' : 'Select installed Tesseract languages. Printed-language support does not guarantee handwriting accuracy.';
  };
  engineSelect?.addEventListener('change', updateEngine);
  updateEngine();
  let previewUrl;
  document.querySelectorAll('[data-upload]').forEach(input => {
    const zone = input.closest('.upload-zone');
    const preview = document.querySelector('[data-preview]');
    const remove = zone.querySelector('[data-remove-file]');
    const update = () => {
      const file = input.files[0];
      zone.querySelector('[data-file-name]').textContent = file ? file.name : '';
      if (remove) remove.hidden = !file;
      if (!preview) return;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      preview.replaceChildren();
      if (!file) {
        const note = document.createElement('span');
        note.textContent = 'Your document preview will appear here';
        preview.append(note);
        return;
      }
      let element;
      const suffix = file.name.split('.').pop().toLowerCase();
      if (['jpg', 'jpeg', 'png'].includes(suffix)) {
        element = document.createElement('img');
        element.alt = 'Selected document preview';
      } else if (suffix === 'pdf') {
        element = document.createElement('iframe');
        element.title = 'Selected PDF preview';
      } else {
        element = document.createElement('span');
        element.className = 'preview-note';
        element.textContent = `${file.name} — DOCX preview is unavailable. Extract text to read its contents.`;
      }
      if (['jpg', 'jpeg', 'png', 'pdf'].includes(suffix)) {
        previewUrl = URL.createObjectURL(file);
        element.src = previewUrl;
      }
      preview.append(element);
      if (editor?.value) announce('New file selected. The editor still contains your previous text until extraction completes.');
    };
    input.addEventListener('change', update);
    remove?.addEventListener('click', () => { input.value = ''; update(); input.focus(); });
    ['dragenter', 'dragover'].forEach(name => zone.addEventListener(name, event => {
      event.preventDefault(); zone.classList.add('drag-over');
    }));
    ['dragleave', 'drop'].forEach(name => zone.addEventListener(name, event => {
      event.preventDefault(); zone.classList.remove('drag-over');
    }));
    zone.addEventListener('drop', event => {
      if (event.dataTransfer.files.length) {
        const selection = new DataTransfer();
        selection.items.add(event.dataTransfer.files[0]);
        input.files = selection.files;
        update();
      }
    });
  });
  window.addEventListener('pagehide', () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });

  let extractionPending = false;
  document.querySelector('[data-extract-form]')?.addEventListener('submit', async event => {
    event.preventDefault();
    const form = event.currentTarget;
    if (extractionPending) return;
    extractionPending = true;
    if (manuallyEdited && editor.value && !await confirmReplacement()) { extractionPending = false; return; }
    const button = form.querySelector('[type=submit]');
    const errors = document.querySelector('[data-extraction-errors]');
    const warnings = document.querySelector('[data-extraction-warnings]');
    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = 'Extracting…';
    errors.replaceChildren(); warnings.replaceChildren();
    announce(form.querySelector('[name=mode]')?.value === 'printed' ? 'Reading your document in the selected languages…' : engineSelect?.value === 'gemini' ? 'Transcribing with Gemini online. Please wait; do not resubmit.' : 'Processing your document. Handwriting recognition may take a few minutes on CPU.');
    status.classList.add('busy');
    editor.setAttribute('aria-busy', 'true');
    const wasReadOnly = editor.readOnly;
    editor.readOnly = true;
    try {
      const response = await fetch(form.action || location.href, {
        method: 'POST', body: new FormData(form), headers: {Accept: 'application/json'}, credentials: 'same-origin',
      });
      const data = await response.json();
      if (!response.ok) {
        const messages = Object.values(data.errors || {}).flat().map(item => item.message);
        throw new Error(messages.join(' ') || 'Extraction failed. Try a smaller or clearer document.');
      }
      editor.value = data.text;
      const metadata = document.querySelector('[data-result-metadata]');
      if (metadata) {
        metadata.hidden = !data.filename;
        metadata.textContent = [data.filename, data.engine_name || data.method, (data.language_names || []).join(' + ')].filter(Boolean).join(' · ');
      }
      manuallyEdited = false;
      setEngineResults(data.engine_results || {});
      (data.warnings || []).forEach(message => {
        const note = document.createElement('p'); note.className = 'notice'; note.textContent = message; warnings.append(note);
      });
      announce(data.engine_results && Object.keys(data.engine_results).length ? 'Recognition complete. Compare the outputs and choose a result for the final editor.' : data.text?.trim() ? `Extraction complete · ${data.engine_name || data.method}. Review the text before using it.` : 'No readable text was detected. Try a clearer document.');
      countWords();
    } catch (error) {
      const note = document.createElement('p'); note.className = 'notice error';
      note.textContent = error instanceof SyntaxError ? 'The server did not return a result. Check that it is running and retry.' : error.message;
      errors.append(note);
      announce('Extraction did not complete. Any previous text remains in the editor.');
    } finally {
      extractionPending = false;
      button.disabled = false; button.textContent = originalLabel;
      status.classList.remove('busy'); editor.removeAttribute('aria-busy'); editor.readOnly = wasReadOnly;
    }
  });
  document.querySelectorAll('[data-loading-form]').forEach(form => form.addEventListener('submit', event => {
    if (form.getAttribute('aria-busy') === 'true') { event.preventDefault(); return; }
    form.setAttribute('aria-busy', 'true');
    const matching = !!form.querySelector('[name=analysis_token]');
    const button = form.querySelector('[type=submit]');
    button.dataset.originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = matching ? 'Analyzing job coverage…' : 'Analyzing resume…';
    const feedback = form.querySelector('[data-resume-status]');
    if (feedback) feedback.textContent = matching ? 'Comparing the detected skills with your job description…' : 'Analyzing your resume. Image resumes may take longer…';
  }));
  window.addEventListener('pageshow', () => document.querySelectorAll('[data-loading-form] [type=submit]').forEach(button => {
    button.disabled = false; if (button.dataset.originalLabel) button.textContent = button.dataset.originalLabel;
    const form = button.closest('form');
    form.removeAttribute('aria-busy');
    const feedback = form.querySelector('[data-resume-status]');
    if (feedback) feedback.textContent = '';
  }));
  document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
    const source = document.getElementById(button.dataset.copy);
    if (!source.value.trim()) { announce('Extract or enter some text first.'); return; }
    try { await navigator.clipboard.writeText(source.value); announce('Text copied to clipboard.'); }
    catch { source.focus(); source.select(); announce('Press Ctrl+C or Command+C to copy the selected text.'); }
  }));
  document.querySelectorAll('[data-download]').forEach(button => button.addEventListener('click', () => {
    const source = document.getElementById(button.dataset.download);
    if (!source.value.trim()) { announce('Extract or enter some text first.'); return; }
    const url = URL.createObjectURL(new Blob([source.value], {type: 'text/plain;charset=utf-8'}));
    const link = document.createElement('a'); link.href = url; link.download = button.dataset.filename || 'smartdoc.txt';
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    announce('Your text download is ready.');
  }));
})();
