/* Blocking local script: restore the theme before styles or content render. */
(() => {
 'use strict';
 const key = 'smartdoc-theme';
 const root = document.documentElement;
 const valid = value => value === 'dark' || value === 'light';
 const syncControls = () => {
   const dark = root.dataset.theme === 'dark';
   document.querySelectorAll('[data-theme-toggle]').forEach(button => {
     button.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
     button.setAttribute('title', dark ? 'Switch to light mode' : 'Switch to dark mode');
   });
   document.querySelector('meta[name="theme-color"]')?.setAttribute('content', dark ? '#080b1d' : '#f6f7ff');
 };
 const apply = (theme, persist = false) => {
   root.dataset.theme = valid(theme) ? theme : 'light';
   if (persist) { try { localStorage.setItem(key, root.dataset.theme); } catch { /* Current-page switching works with blocked storage. */ } }
   syncControls();
   document.dispatchEvent(new CustomEvent('smartdoc:themechange', {detail:{theme:root.dataset.theme}}));
 };
 let saved;
 try { saved = localStorage.getItem(key); } catch { /* First-visit light default. */ }
 apply(saved);
 document.addEventListener('DOMContentLoaded', () => {
   syncControls();
   document.querySelectorAll('[data-theme-toggle]').forEach(button => button.addEventListener('click', () => apply(root.dataset.theme === 'dark' ? 'light' : 'dark', true)));
 }, {once:true});
 window.addEventListener('storage', event => { if (event.key === key) apply(event.newValue); });
})();
