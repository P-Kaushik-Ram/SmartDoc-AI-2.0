// Run the production script against minimal DOM doubles; no browser automation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(dataset = {}) {
    this.dataset = dataset; this.value = ''; this.textContent = '';
    this.listeners = {}; this.attributes = {}; this.children = [];
    this.classList = {add() {}, remove() {}};
  }
  addEventListener(name, handler, options = {}) {
    (this.listeners[name] ||= []).push({handler, once:options.once});
  }
  async fire(name, extra = {}) {
    const handlers = [...(this.listeners[name] || [])];
    this.listeners[name] = handlers.filter(item => !item.once);
    await Promise.all(handlers.map(item => {
      const event = {currentTarget:this, preventDefault() {}, ...extra};
      const result = item.handler(event);
      event.currentTarget = null; // Native DOM clears this after dispatch, before awaited work resumes.
      return result;
    }));
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  append(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  focus() {} select() {} remove() {} click() { this.clicked = true; }
  showModal() { this.open = true; }
  close(value) { this.returnValue = value; this.open = false; return this.fire('close'); }
}

async function main() {
  const editor = new Element(), status = new Element(), counter = new Element();
  const dialog = new Element(), form = new Element(), submit = new Element();
  const errors = new Element(), warnings = new Element(), container = new Element();
  const engineSelect = new Element(), cloudConsent = new Element(), cloudNotice = new Element(), engineHelp = new Element();
  engineSelect.value = 'hybrid';
  const languages = ['eng','tam','hin','tel','mal'].map(code => { const input = new Element(); input.value = code; return input; });
  const copy = new Element({copy:'extracted-text'});
  const download = new Element({download:'extracted-text', filename:'smartdoc-handwriting.txt'});
  const tabs = {}, panels = {}, outputs = {}, uses = {}, engineErrors = {};
  const selectors = new Map([
    ['[data-extraction-status]', status], ['[data-word-count]', counter],
    ['[data-confirm-replace]', dialog], ['[data-extract-form]', form],
    ['[data-extraction-errors]', errors], ['[data-extraction-warnings]', warnings],
    ['[data-recognition-results]', container],
    ['[data-ocr-engine]', engineSelect], ['[data-cloud-consent]', cloudNotice],
    ['[name=cloud_consent]', cloudConsent], ['[data-engine-help]', engineHelp],
  ]);
  for (const engine of ['trocr', 'tesseract']) {
    tabs[engine] = new Element({engineTab:engine});
    panels[engine] = new Element({enginePanel:engine});
    outputs[engine] = new Element(); outputs[engine].value = `${engine} output`;
    uses[engine] = new Element({useEngine:engine}); engineErrors[engine] = new Element();
    for (const [key, element] of [['engine-tab', tabs[engine]], ['engine-output', outputs[engine]],
                                  ['use-engine', uses[engine]], ['engine-error', engineErrors[engine]]]) {
      selectors.set(`[data-${key}="${engine}"]`, element);
    }
  }
  form.querySelector = () => submit;
  const groups = {'[data-engine-tab]':Object.values(tabs), '[data-engine-panel]':Object.values(panels),
                  '[data-use-engine]':Object.values(uses), '[data-copy]':[copy], '[data-download]':[download], '[name=languages]':languages};
  let copied, blob, anchor, fetches = 0;
  const document = {
    querySelector: selector => selectors.get(selector),
    querySelectorAll: selector => groups[selector] || [],
    getElementById: id => id === 'extracted-text' ? editor : undefined,
    createElement: tag => { const element = new Element(); if (tag === 'a') anchor = element; return element; },
    body:new Element(),
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/core/js/studio.js'), 'utf8'), {
    document, window:{addEventListener() {}}, Blob,
    URL:{createObjectURL(value) {blob = value; return 'blob:test';}, revokeObjectURL() {}},
    navigator:{clipboard:{async writeText(text) {copied = text;}}},
    location:{href:'http://localhost/document-studio/'}, FormData:class {}, setTimeout() {},
    async fetch() {fetches++; return {ok:true, async json() {return {text:'', method:'handwriting_hybrid', warnings:[],
      engine_results:{trocr:{text:'New TrOCR'}, tesseract:{text:'', error:'Unavailable'}}};}};},
  });
  assert.equal(languages[1].disabled, true);
  engineSelect.value = 'gemini'; await engineSelect.fire('change');
  assert.equal(cloudNotice.hidden, false); assert.equal(cloudConsent.required, true);
  assert.equal(languages[1].disabled, false); assert.equal(fetches, 0);
  cloudConsent.checked = true;
  engineSelect.value = 'trocr'; await engineSelect.fire('change');
  assert.equal(cloudNotice.hidden, true); assert.equal(cloudConsent.checked, false);
  assert.equal(languages[1].disabled, true); assert.equal(languages[0].checked, true);
  await uses.trocr.fire('click'); assert.equal(editor.value, 'trocr output');
  editor.value = 'My manual edits'; await editor.fire('input');
  await tabs.tesseract.fire('click');
  assert.equal(editor.value, 'My manual edits'); assert.equal(panels.trocr.hidden, true);
  let action = uses.tesseract.fire('click'); assert.equal(dialog.open, true);
  await dialog.close('cancel'); await action; assert.equal(editor.value, 'My manual edits');
  action = uses.tesseract.fire('click'); await dialog.close('replace'); await action;
  assert.equal(editor.value, 'tesseract output');
  editor.value = 'Final edited text — தமிழ் हिन्दी తెలుగు മലയാളം'; await editor.fire('input');
  await copy.fire('click'); assert.equal(copied, 'Final edited text — தமிழ் हिन्दी తెలుగు മലയാളം');
  await download.fire('click'); assert.equal(await blob.text(), 'Final edited text — தமிழ் हिन्दी తెలుగు മലയാളം');
  assert.equal(anchor.download, 'smartdoc-handwriting.txt'); assert.equal(anchor.clicked, true);
  action = form.fire('submit'); await dialog.close('cancel'); await action;
  assert.equal(fetches, 0); assert.equal(editor.value, 'Final edited text — தமிழ் हिन्दी తెలుగు മലയാളം');
  action = form.fire('submit');
  await form.fire('submit'); // A pending confirmation must also reject duplicate submissions.
  await dialog.close('replace'); await action;
  assert.equal(fetches, 1); assert.equal(editor.value, '');
  assert.equal(outputs.trocr.value, 'New TrOCR'); assert.equal(uses.tesseract.disabled, true);
  assert.equal(engineErrors.tesseract.textContent, 'Unavailable');
  await uses.trocr.fire('click'); assert.equal(editor.value, 'New TrOCR');
  console.log('PASS: engine selection; tab/edit preservation; cancel/confirm replacement; edited clipboard and TXT contents; upload confirmation; engine failure display.');
}
main().catch(error => {console.error(error); process.exitCode = 1;});
