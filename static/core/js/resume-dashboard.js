(() => {
  'use strict';
  const data = document.getElementById('skill-chart-data');
  const svg = document.querySelector('[data-skill-doughnut]');
  if (data && svg && !svg.dataset.initialized) {
    svg.dataset.initialized = 'true';
    try {
      const categories = JSON.parse(data.textContent);
      const total = categories.reduce((sum, row) => sum + row.count, 0);
      const palette = () => {
        const style = getComputedStyle(document.documentElement);
        return Array.from({length:7}, (_, i) => style.getPropertyValue(`--chart-${i + 1}`).trim());
      };
      const colors = palette();
      const ns = 'http://www.w3.org/2000/svg';
      const circumference = 2 * Math.PI * 78;
      let offset = 0;
      const select = index => {
        document.querySelectorAll('[data-skill-category]').forEach(el => el.classList.toggle('highlighted', el.dataset.skillCategory === String(index)));
        document.querySelectorAll('.category-bar').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.category === String(index))));
        document.querySelector('[data-chart-caption]').textContent = `${categories[index].label}: ${categories[index].count} of ${total} detected skills.`;
      };
      if (total > 0) categories.forEach((row, i) => {
        const arc = document.createElementNS(ns, 'circle');
        for (const [key, value] of Object.entries({cx:110,cy:110,r:78,fill:'none',stroke:colors[i % colors.length],'stroke-width':29,'stroke-linecap':'butt','stroke-dasharray':`${row.count / total * circumference} ${circumference}`,'stroke-dashoffset':-offset,transform:'rotate(-90 110 110)',tabindex:0,role:'button','aria-label':`${row.label}: ${row.count} skills`,'data-category':i})) arc.setAttribute(key, value);
        offset += row.count / total * circumference;
        arc.addEventListener('click', () => select(i));
        arc.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(i); } });
        svg.append(arc);
      });
      const label = document.createElementNS(ns, 'text');
      label.setAttribute('x',110); label.setAttribute('y',120); label.setAttribute('text-anchor','middle'); label.setAttribute('fill','var(--ink)'); label.setAttribute('font-size',32); label.textContent = total; svg.append(label);
      document.addEventListener('smartdoc:themechange', () => {
        const next = palette();
        svg.querySelectorAll('circle[data-category]').forEach((arc, i) => arc.setAttribute('stroke', next[i % next.length]));
      });
      document.querySelectorAll('.category-bar').forEach(button => button.addEventListener('click', () => select(Number(button.dataset.category))));
    } catch { svg.hidden = true; }
  }
})();
