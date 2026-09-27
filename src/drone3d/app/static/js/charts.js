// Canvas time-series and SVG gauges for the console's telemetry.

// Area/line chart over a fixed time window. series: [{ key, color, fill?, max?, label }]
export function areaChart(canvas, samples, series, { window = 600, max = null, now = Date.now() / 1000, grid = 4, unit = '' } = {}) {
  const dpr = devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const pad = { l: 34, r: 6, t: 6, b: 16 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const top = max ?? Math.max(1, ...samples.flatMap((s) => series.map((c) => s[c.key] ?? 0))) * 1.1;
  g.font = '10px "IBM Plex Mono", monospace';
  g.fillStyle = '#56677a'; g.strokeStyle = '#1b2633'; g.lineWidth = 1;
  for (let i = 0; i <= grid; i++) {
    const y = pad.t + (ih * i) / grid;
    g.beginPath(); g.moveTo(pad.l, y + 0.5); g.lineTo(w - pad.r, y + 0.5); g.stroke();
    const v = top * (1 - i / grid);
    g.fillText(`${v >= 100 ? Math.round(v) : v.toFixed(v >= 10 ? 0 : 1)}${unit}`, 2, y + 3);
  }
  for (const [i, lbl] of [[0, `-${Math.round(window / 60)} min`], [1, 'now']]) { g.fillText(lbl, i ? w - pad.r - 22 : pad.l, h - 3); }
  const x = (t) => pad.l + iw * (1 - (now - t) / window);
  const y = (v) => pad.t + ih * (1 - Math.min(1, (v ?? 0) / top));
  const pts = samples.filter((s) => now - s.t <= window);
  if (pts.length < 2) return;
  for (const c of series) {
    g.beginPath();
    pts.forEach((s, i) => (i ? g.lineTo(x(s.t), y(s[c.key])) : g.moveTo(x(s.t), y(s[c.key]))));
    if (c.fill) {
      const grad = g.createLinearGradient(0, pad.t, 0, pad.t + ih);
      grad.addColorStop(0, c.fill); grad.addColorStop(1, 'transparent');
      g.lineTo(x(pts[pts.length - 1].t), pad.t + ih); g.lineTo(x(pts[0].t), pad.t + ih); g.closePath();
      g.fillStyle = grad; g.fill();
      g.beginPath(); pts.forEach((s, i) => (i ? g.lineTo(x(s.t), y(s[c.key])) : g.moveTo(x(s.t), y(s[c.key]))));
    }
    g.strokeStyle = c.color; g.lineWidth = 1.6; g.stroke();
  }
}

// Circular gauge (0..1) as SVG markup.
export function ring(frac, { label = '', sub = '', color = '#4fd1e8', size = 120 } = {}) {
  const r = 48, c = 2 * Math.PI * r, f = Math.max(0, Math.min(1, frac || 0));
  return `<svg class="ring" viewBox="0 0 120 120" width="${size}" height="${size}">
    <circle cx="60" cy="60" r="${r}" fill="none" stroke="#1b2735" stroke-width="9"/>
    <circle cx="60" cy="60" r="${r}" fill="none" stroke="${color}" stroke-width="9" stroke-linecap="round"
      stroke-dasharray="${(c * f).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 60 60)" style="transition: stroke-dasharray .6s"/>
    <text x="60" y="62" text-anchor="middle" fill="#e7edf3" style="font: 800 30px 'Big Shoulders Display', sans-serif">${label}</text>
    <text x="60" y="82" text-anchor="middle" fill="#8b9bac" style="font: 500 10px 'IBM Plex Mono', monospace">${sub}</text></svg>`;
}

// Stacked horizontal bar: parts [{ value, color, label }] of total.
export function stack(parts, total) {
  return `<div class="bar stack" style="height:10px">${parts.map((p) => `<i title="${p.label}" style="width:${(100 * p.value) / Math.max(total, 1)}%;background:${p.color}"></i>`).join('')}</div>`;
}
