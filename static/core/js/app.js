(() => {
  const sidebar = document.getElementById('appSidebar');
  const scrim = document.querySelector('.sidebar-scrim');
  const toggle = document.querySelector('[data-sidebar-toggle]');
  const closeSidebar = () => {
    sidebar?.classList.remove('is-open');
    scrim?.classList.remove('is-visible');
    toggle?.setAttribute('aria-expanded', 'false');
  };
  toggle?.addEventListener('click', () => {
    const open = sidebar?.classList.toggle('is-open');
    scrim?.classList.toggle('is-visible', Boolean(open));
    toggle.setAttribute('aria-expanded', String(Boolean(open)));
  });
  scrim?.addEventListener('click', closeSidebar);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeSidebar(); });

  document.querySelectorAll('[data-toast]').forEach(toast => {
    toast.querySelector('.toast-dismiss')?.addEventListener('click', () => toast.remove());
    window.setTimeout(() => toast.remove(), 6500);
  });

  document.querySelectorAll('[data-toggle-target]').forEach(button => {
    button.addEventListener('click', () => {
      const panel = document.getElementById(button.dataset.toggleTarget);
      if (!panel) return;
      const expanded = button.getAttribute('aria-expanded') === 'true';
      button.setAttribute('aria-expanded', String(!expanded));
      panel.hidden = expanded;
    });
  });

  document.querySelectorAll('form[data-loading-form]').forEach(form => {
    form.addEventListener('submit', () => {
      const button = form.querySelector('[type="submit"]');
      if (!button || button.disabled) return;
      button.dataset.originalLabel = button.innerHTML;
      button.disabled = true;
      button.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin" aria-hidden="true"></i> Working…';
    });
  });

  const chartTheme = () => {
    const style = getComputedStyle(document.documentElement);
    return {
      palette:Array.from({length:7}, (_, i) => style.getPropertyValue(`--chart-${i + 1}`).trim()),
      text:style.getPropertyValue('--text-primary').trim(),
      muted:style.getPropertyValue('--text-secondary').trim(),
      border:style.getPropertyValue('--border').trim(),
      surface:style.getPropertyValue('--surface').trim(),
    };
  };
  document.querySelectorAll('[data-analytics-chart]').forEach(canvas => {
    const fallback = (message, labels = [], values = []) => {
      const panel = document.createElement('div');
      panel.className = 'chart-fallback';
      const note = document.createElement('p');
      note.textContent = message;
      panel.append(note);
      if (labels.length) {
        const list = document.createElement('ul');
        labels.forEach((label, index) => {
          const item = document.createElement('li');
          item.textContent = `${label}: ${values[index] ?? 0}`;
          list.append(item);
        });
        panel.append(list);
      }
      canvas.replaceWith(panel);
    };
    try {
      const source = document.getElementById(canvas.dataset.chartData);
      if (!source) throw new Error('Chart data is missing.');
      const data = JSON.parse(source.textContent);
      const labels = Array.isArray(data.labels) ? data.labels : [];
      const values = Array.isArray(data.values) ? data.values.map(Number) : [];
      if (!labels.length || !window.Chart) {
        fallback(window.Chart ? 'No chart data is available.' : 'Chart library unavailable; here are the underlying database counts.', labels, values);
        return;
      }
      const horizontal = canvas.dataset.analyticsChart === 'bar-horizontal';
      const type = canvas.dataset.analyticsChart === 'doughnut' ? 'doughnut' : 'bar';
      const theme = chartTheme();
      const chart = new Chart(canvas, {
        type,
        data: { labels, datasets: [{ data: values, backgroundColor: labels.map((_, index) => theme.palette[index % theme.palette.length]), borderColor: theme.border, borderWidth: 1, borderRadius: type === 'bar' ? 5 : 0, hoverOffset: type === 'doughnut' ? 4 : 0 }] },
        options: {
          indexAxis: horizontal ? 'y' : 'x',
          responsive: true,
          maintainAspectRatio: false,
          cutout: type === 'doughnut' ? '68%' : undefined,
          plugins: { legend: { display: false }, tooltip: { backgroundColor:theme.surface, titleColor:theme.text, bodyColor:theme.text, borderColor:theme.border, borderWidth:1, callbacks: { label: item => ` ${item.label}: ${item.raw}` } } },
          scales: type === 'bar' ? {
            x: { beginAtZero: true, ticks: { color: theme.muted, precision: 0 }, grid: { color: theme.border } },
            y: { beginAtZero: true, ticks: { color: theme.muted, precision: 0 }, grid: { display: false } }
          } : {}
        }
      });
      document.addEventListener('smartdoc:themechange', () => {
        const next = chartTheme();
        chart.data.datasets[0].backgroundColor = labels.map((_, i) => next.palette[i % next.palette.length]);
        chart.data.datasets[0].borderColor = next.border;
        Object.assign(chart.options.plugins.tooltip, {backgroundColor:next.surface, titleColor:next.text, bodyColor:next.text, borderColor:next.border});
        if (type === 'bar') {
          chart.options.scales.x.ticks.color = next.muted;
          chart.options.scales.y.ticks.color = next.muted;
          chart.options.scales.x.grid.color = next.border;
        }
        chart.update('none');
      });
    } catch (_) {
      fallback('Chart could not be rendered. Try refreshing the dashboard.');
    }
  });
})();
