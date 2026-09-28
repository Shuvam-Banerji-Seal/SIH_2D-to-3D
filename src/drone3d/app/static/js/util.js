// Small helpers shared by every page.
export const $ = (sel, el = document) => el.querySelector(sel);
export const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
export const post = (path, body = {}) => api(path, { method: 'POST', body: JSON.stringify(body) });

export const fmtS = (s) => (s == null || Number.isNaN(s) ? '—' : s >= 5400 ? `${(s / 3600).toFixed(1)} h` : s >= 90 ? `${(s / 60).toFixed(1)} min` : `${s < 10 ? s.toFixed(1) : Math.round(s)} s`);
export const fmtN = (n) => (n == null ? '—' : n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M` : n >= 1e4 ? `${Math.round(n / 1e3)}k` : Math.round(n).toLocaleString());
export const pct = (x) => (x == null ? '—' : `${Math.round(100 * x)}%`);
export const gb = (mb) => (mb == null ? '—' : `${(mb / 1024).toFixed(mb >= 10240 ? 0 : 1)}`);
export const clip = (s, n) => (s && s.length > n ? `${s.slice(0, n - 1)}…` : s || '');
export const prettyVideo = (name) => (name || '').replace(/\s*\[[^\]]+\]\.\w+$/, '').replace(/\.\w+$/, '').replace(/[｜|].*$/, '').trim();
export const ago = (t) => { if (!t) return '—'; const s = Date.now() / 1000 - t; return s < 60 ? 'just now' : s < 3600 ? `${Math.round(s / 60)} min ago` : s < 86400 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`; };

export function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = kind; el.textContent = msg;
  $('#toast').appendChild(el);
  setTimeout(() => el.remove(), kind === 'bad' ? 9000 : 4500);
}

// Timers owned by the current page; the router clears them on navigation.
const timers = new Set();
export function every(ms, fn) { const id = setInterval(fn, ms); timers.add(id); return id; }
export function clearTimers() { timers.forEach(clearInterval); timers.clear(); }
const cleanups = new Set();
export function onLeave(fn) { cleanups.add(fn); }
export function leave() { cleanups.forEach((f) => { try { f(); } catch { /* page teardown */ } }); cleanups.clear(); clearTimers(); }

export const icon = {
  eye: '<svg viewBox="0 0 24 24"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 5v14l11-7z"/></svg>',
  stop: '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="1.5"/></svg>',
  power: '<svg viewBox="0 0 24 24"><path d="M12 3v8M6.3 6.3a8 8 0 1 0 11.4 0"/></svg>',
  bolt: '<svg viewBox="0 0 24 24"><path d="M13 3 5 14h6l-1 7 8-11h-6z"/></svg>',
  cam: '<svg viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg>',
  full: '<svg viewBox="0 0 24 24"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
  ruler: '<svg viewBox="0 0 24 24"><path d="m3 17 14-14 4 4L7 21zM7 13l2 2M10 10l2 2M13 7l2 2"/></svg>',
  target: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2"/></svg>',
  down: '<svg viewBox="0 0 24 24"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>',
  ext: '<svg viewBox="0 0 24 24"><path d="M14 4h6v6M20 4l-9 9M18 14v6H4V6h6"/></svg>',
};
