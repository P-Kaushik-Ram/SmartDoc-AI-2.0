const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/core/js/theme.js','utf8');
function boot(saved, blocked=false) {
  const events={},windowEvents={},attributes={},buttons=[{setAttribute:(key,value)=>attributes[key]=value,addEventListener:(key,handler)=>events['button:'+key]=handler}];
  let preference=saved, changes=0;
  const root={dataset:{}};
  const document={documentElement:root,querySelectorAll:()=>buttons,querySelector:()=>null,
    addEventListener:(key,handler)=>events[key]=handler,dispatchEvent:()=>changes++};
  vm.runInNewContext(source,{document,window:{addEventListener:(key,handler)=>windowEvents[key]=handler},
    localStorage:{getItem(){if(blocked)throw Error('blocked');return preference;},setItem(key,value){if(blocked)throw Error('blocked');preference=value;}},
    CustomEvent:class {constructor(type,options){this.type=type;this.detail=options.detail;}}});
  return {root,attributes,events,windowEvents,preference:()=>preference,changes:()=>changes};
}
for (const saved of [null,undefined,'invalid','DARK','<script>']) assert.equal(boot(saved).root.dataset.theme,'light');
for (const saved of ['light','dark']) assert.equal(boot(saved).root.dataset.theme,saved);
const normal=boot('dark');
assert.equal(normal.changes(),1); // Initialization occurs synchronously, before DOM ready.
normal.events.DOMContentLoaded();
assert.equal(normal.attributes['aria-label'],'Switch to light mode');
normal.events['button:click']();
assert.equal(normal.root.dataset.theme,'light');assert.equal(normal.preference(),'light');
assert.equal(normal.attributes['aria-label'],'Switch to dark mode');
normal.windowEvents.storage({key:'unrelated',newValue:'dark'});assert.equal(normal.root.dataset.theme,'light');
normal.windowEvents.storage({key:'smartdoc-theme',newValue:'dark'});assert.equal(normal.root.dataset.theme,'dark');
normal.windowEvents.storage({key:'smartdoc-theme',newValue:null});assert.equal(normal.root.dataset.theme,'light');
const blocked=boot('dark',true);assert.equal(blocked.root.dataset.theme,'light');blocked.events.DOMContentLoaded();
blocked.events['button:click']();assert.equal(blocked.root.dataset.theme,'dark');
console.log('PASS: synchronous initialization, light default, validated persistence, labels, toggling, storage synchronization, blocked storage.');
