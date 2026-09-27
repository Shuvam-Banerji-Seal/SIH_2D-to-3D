// Runs: every run directory under outputs/, started here, on the engine, or from the command line.
import { $, $$, api, ago, esc, every, fmtN, fmtS, pct, prettyVideo } from './util.js';

const keep = { q: '', status: 'all' };

export async function viewRuns(main) {
  main.innerHTML = `
    <div class="page-head rise"><h1>Runs<small>Every reconstruction and live session under <span class="mono">outputs/</span>, with its time against the budget.</small></h1>
      <div class="row" style="margin-left:auto"><input type="text" id="q" placeholder="filter…" value="${esc(keep.q)}" style="width:220px">
        <div class="seg" id="st">${['all', 'running', 'done', 'failed', 'live'].map((s) => `<button data-s="${s}" class="${s === keep.status ? 'on' : ''}">${s}</button>`).join('')}</div></div></div>
    <section class="panel flush rise" style="--i:1"><table class="t" id="tbl"></table></section>`;
  let runs = [];
  const draw = () => {
    const q = keep.q.toLowerCase();
    const list = runs.filter((r) => (!q || r.name.toLowerCase().includes(q) || (r.summary.video || '').toLowerCase().includes(q)) &&
      (keep.status === 'all' || (keep.status === 'live' ? r.kind === 'live' : keep.status === 'running' ? ['running', 'queued', 'stopping'].includes(r.status) : r.status === keep.status)));
    $('#tbl').innerHTML = `<thead><tr><th>Run</th><th>Status</th><th>Video</th><th>Registered</th><th>Triangles</th><th>Completeness</th><th>Time vs budget</th><th>When</th></tr></thead><tbody>` +
      list.map((r) => {
        const s = r.summary, p = s.processing, frac = p?.budget_seconds ? p.seconds / p.budget_seconds : null;
        return `<tr class="click" data-n="${esc(r.name)}"><td><b>${esc(r.name)}</b>${r.kind === 'live' ? ' <span class="pill">live</span>' : ''}</td>
          <td><span class="st ${esc(r.status)}">${esc(r.status)}</span></td><td class="muted">${esc(prettyVideo(s.video).slice(0, 44))}</td>
          <td class="mono">${s.registered != null ? `${s.registered}/${s.keyframes}` : '—'}</td><td class="mono">${fmtN(s.triangles)}</td><td class="mono">${pct(s.completeness)}</td>
          <td>${p ? `<div class="row" style="flex-wrap:nowrap"><div class="bar" style="width:110px"><i class="${frac > 1 ? 'over' : ''}" style="width:${Math.min(100, 100 * frac)}%"></i></div><span class="mono muted" style="font-size:12px">${fmtS(p.seconds)} / ${fmtS(p.budget_seconds)}</span></div>` : '—'}</td>
          <td class="muted">${ago(r.mtime)}</td></tr>`;
      }).join('') + '</tbody>';
    $$('#tbl tr[data-n]').forEach((tr) => tr.addEventListener('click', () => { location.hash = `#/run/${encodeURIComponent(tr.dataset.n)}`; }));
  };
  const load = async () => { try { runs = await api('/api/runs'); draw(); } catch (e) { $('#tbl').innerHTML = `<tr><td class="err">${esc(e.message)}</td></tr>`; } };
  $('#q').addEventListener('input', (e) => { keep.q = e.target.value; draw(); });
  $$('#st button').forEach((b) => b.addEventListener('click', () => { keep.status = b.dataset.s; $$('#st button').forEach((c) => c.classList.toggle('on', c === b)); draw(); }));
  await load();
  every(5000, load);
}
