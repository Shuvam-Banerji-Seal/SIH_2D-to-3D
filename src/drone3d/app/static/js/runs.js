// Runs: the finished models first -- your builds and live sessions, the sample-video benchmark -- as a gallery;
// the development runs behind the paper's studies in a table of their own.
import { $, $$, api, ago, esc, every, fmtN, fmtS, icon, pct, prettyVideo } from './util.js';

const keep = { q: '', group: 'models', status: 'all' };
const SPLAT = ['splat', 'depth', 'mesh', 'render'];
const GROUPS = { models: ['builds', 'live', 'benchmark'], development: ['development'], all: ['builds', 'live', 'benchmark', 'development'] };

// the budget is for the mesh and point cloud; splat training comes after it
const deliverable = (p) => (p ? p.deliverable_seconds ?? Object.entries(p.per_stage_s || {}).filter(([st]) => !SPLAT.includes(st)).reduce((a, [, v]) => a + v, 0) : null);
const title = (r) => prettyVideo((r.summary.video || '').split('/').pop()) || r.name;

function card(r) {
  const s = r.summary, p = s.processing, used = deliverable(p), frac = p?.budget_seconds && used != null ? used / p.budget_seconds : null;
  const glb = (s.files || []).flatMap((m) => m.files).find((f) => f.endsWith('mesh_textured.glb')) || (s.files || []).flatMap((m) => m.files).find((f) => f.endsWith('.glb'));
  const enc = encodeURIComponent(r.name);
  return `<article class="rcard" data-n="${esc(r.name)}">
    <div class="rthumb" style="${r.thumb ? `background-image:url('${encodeURI(r.thumb)}')` : ''}">${r.thumb ? '' : '<span class="muted">no preview yet</span>'}
      <span class="st ${esc(r.status)}" style="position:absolute;left:10px;top:10px">${esc(r.status)}</span>${r.kind === 'live' ? '<span class="pill" style="position:absolute;right:10px;top:10px">live</span>' : ''}</div>
    <div style="padding:12px 14px">
      <div class="n" style="font-weight:600;line-height:1.3">${esc(title(r))}</div>
      <div class="note mono" style="margin-top:2px">${esc(r.name)} · ${ago(r.mtime)}</div>
      <div class="rstats">
        <div><span class="k">complete</span><b>${pct(s.completeness)}</b></div>
        <div><span class="k">mesh + cloud</span><b class="${frac != null && frac > 1 ? 'over' : ''}">${fmtS(used)}</b><span class="muted"> / ${fmtS(p?.budget_seconds)}${frac != null ? (frac <= 1 ? ' ✓' : ' ✗') : ''}</span></div>
        <div><span class="k">models</span><b>${s.models ?? '—'}</b></div>
        <div><span class="k">triangles</span><b>${fmtN(s.triangles)}</b></div>
      </div>
      <div class="row" style="gap:6px;margin-top:10px">
        <a class="btn tiny primary" href="#/run/${enc}">${icon.eye}Explore</a>
        ${glb ? `<a class="btn tiny" href="#/model/${enc}/${glb.split('/').map(encodeURIComponent).join('/')}" title="the textured GLB in the file viewer">GLB</a>` : ''}
        ${s.viewer ? `<a class="btn tiny" href="/runs/${enc}/export/index.html" target="_blank" title="the self-contained viewer shipped with the export">${icon.ext}viewer</a>` : ''}
      </div>
    </div></article>`;
}

export async function viewRuns(main) {
  main.innerHTML = `
    <div class="page-head rise"><h1>Runs<small>Every model built here, on the engine or from the command line — open one to explore it, or view a file on its own.</small></h1>
      <div class="row" style="margin-left:auto"><input type="text" id="q" placeholder="filter…" value="${esc(keep.q)}" style="width:220px">
        <div class="seg" id="grp">${Object.keys(GROUPS).map((g) => `<button data-g="${g}" class="${g === keep.group ? 'on' : ''}">${g}</button>`).join('')}</div>
        <div class="seg" id="st">${['all', 'running', 'done', 'failed'].map((s) => `<button data-s="${s}" class="${s === keep.status ? 'on' : ''}">${s}</button>`).join('')}</div></div></div>
    <div id="body"></div>`;
  let runs = [];
  const draw = () => {
    const q = keep.q.toLowerCase();
    const list = runs.filter((r) => GROUPS[keep.group].includes(r.group) && (!q || r.name.toLowerCase().includes(q) || title(r).toLowerCase().includes(q)) &&
      (keep.status === 'all' || (keep.status === 'running' ? ['running', 'queued', 'stopping'].includes(r.status) : r.status === keep.status)));
    if (keep.group === 'models') {
      const section = (g, head, sub) => { const rs = list.filter((r) => r.group === g); return rs.length ? `<section class="rise" style="margin-bottom:22px"><h2 class="sect">${head}<small>${sub}</small></h2><div class="rgrid">${rs.map(card).join('')}</div></section>` : ''; };
      $('#body').innerHTML = (section('builds', 'Your builds', 'videos you uploaded or chose in Build') + section('live', 'Live sessions', 'segment by segment')
        + section('benchmark', 'Sample-video benchmark', 'fifteen public drone videos, fast profile + Gaussian splats, one video at a time')) || '<div class="note">No runs match.</div>';
      $$('#body .rcard .rthumb').forEach((t) => t.addEventListener('click', () => { location.hash = `#/run/${encodeURIComponent(t.closest('.rcard').dataset.n)}`; }));
      return;
    }
    $('#body').innerHTML = `<section class="panel flush rise"><table class="t"><thead><tr><th>Run</th><th>Status</th><th>Video</th><th>Registered</th><th>Triangles</th><th>Completeness</th><th>Mesh + cloud vs budget</th><th>When</th></tr></thead><tbody>` +
      list.map((r) => {
        const s = r.summary, p = s.processing, used = deliverable(p), frac = p?.budget_seconds && used != null ? used / p.budget_seconds : null;
        return `<tr class="click" data-n="${esc(r.name)}"><td><b>${esc(r.name)}</b>${r.kind === 'live' ? ' <span class="pill">live</span>' : ''}</td>
          <td><span class="st ${esc(r.status)}">${esc(r.status)}</span></td><td class="muted">${esc(title(r).slice(0, 44))}</td>
          <td class="mono">${s.registered != null ? `${s.registered}/${s.keyframes}` : '—'}</td><td class="mono">${fmtN(s.triangles)}</td><td class="mono">${pct(s.completeness)}</td>
          <td>${p ? `<div class="row" style="flex-wrap:nowrap"><div class="bar" style="width:110px"><i class="${frac > 1 ? 'over' : ''}" style="width:${Math.min(100, 100 * frac)}%"></i></div><span class="mono muted" style="font-size:12px">${fmtS(used)} / ${fmtS(p.budget_seconds)}</span></div>` : '—'}</td>
          <td class="muted">${ago(r.mtime)}</td></tr>`;
      }).join('') + '</tbody></table></section>';
    $$('#body tr[data-n]').forEach((tr) => tr.addEventListener('click', () => { location.hash = `#/run/${encodeURIComponent(tr.dataset.n)}`; }));
  };
  const load = async () => { try { runs = await api('/api/runs'); draw(); } catch (e) { $('#body').innerHTML = `<div class="err">${esc(e.message)}</div>`; } };
  $('#q').addEventListener('input', (e) => { keep.q = e.target.value; draw(); });
  $$('#grp button').forEach((b) => b.addEventListener('click', () => { keep.group = b.dataset.g; $$('#grp button').forEach((c) => c.classList.toggle('on', c === b)); draw(); }));
  $$('#st button').forEach((b) => b.addEventListener('click', () => { keep.status = b.dataset.s; $$('#st button').forEach((c) => c.classList.toggle('on', c === b)); draw(); }));
  await load();
  every(8000, load);
}
