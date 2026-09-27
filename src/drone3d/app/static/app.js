// drone3d web app: configure a reconstruction, follow it, explore the result.
const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
};
const fmtS = (s) => (s == null ? '—' : s >= 90 ? `${(s / 60).toFixed(1)} min` : `${Math.round(s)} s`);
const fmtN = (n) => (n == null ? '—' : n >= 1e6 ? `${(n / 1e6).toFixed(1)} M` : n >= 1e4 ? `${Math.round(n / 1e3)} k` : n.toLocaleString());
const pct = (x) => (x == null ? '—' : `${Math.round(100 * x)} %`);
const human = (key) => key.replace(/_/g, ' ');

// ------------------------------------------------------------------ system bar
async function pollSystem() {
  try {
    const s = await api('/api/system');
    const g = s.gpu[0];
    $('#sys').innerHTML = g
      ? `${esc(g.name.replace('NVIDIA ', ''))} · <b>${Math.round(g.memory_used_mb / 1024)}/${Math.round(g.memory_total_mb / 1024)} GB</b> · <b>${g.util_pct}%</b>` +
        (s.queue.length ? ` · queue ${s.queue.length}` : '') + ` · disk ${s.disk_free_gb} GB free`
      : 'no GPU visible';
    $('#ver').textContent = `drone3d ${s.version}`;
  } catch { $('#sys').textContent = 'server unreachable'; }
}
setInterval(pollSystem, 4000); pollSystem();

// ------------------------------------------------------------------ router
let timer = null;
const routes = { new: viewNew, runs: viewRuns, run: viewRun, about: viewAbout };
function route() {
  clearInterval(timer);
  const [, page = 'new', arg] = location.hash.split('/');
  document.querySelectorAll('nav a').forEach((a) => a.classList.toggle('on', a.dataset.nav === (page === 'run' ? 'runs' : page)));
  (routes[page] || viewNew)(arg && decodeURIComponent(arg));
}
addEventListener('hashchange', route);

// ------------------------------------------------------------------ new reconstruction
const state = { schema: null, profiles: [], videos: [], logs: [], video: null, telemetry: '', profile: 'fast', overrides: {}, stages: null, filter: '', modifiedOnly: false };

function profileValue(section, key) {
  const p = state.profiles.find((x) => x.name === state.profile);
  const v = p && p.values;
  if (!v) return undefined;
  return section ? (v[section] || {})[key] : v[key];
}
function effective(section, f) {
  const dotted = section ? `${section}.${f.key}` : f.key;
  if (dotted in state.overrides) return state.overrides[dotted];
  const pv = profileValue(section, f.key);
  return pv !== undefined ? pv : f.default;
}

async function viewNew() {
  const main = $('#main');
  main.innerHTML = '<p class="muted">Loading…</p>';
  try {
    [state.schema, state.profiles, state.videos, state.logs] = await Promise.all([
      state.schema || api('/api/schema'), api('/api/profiles'), api('/api/videos'), api('/api/telemetry')]);
  } catch (e) { main.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  if (!state.profiles.some((p) => p.name === state.profile)) state.profile = state.profiles[0]?.name;
  main.innerHTML = `
    <h1>New reconstruction</h1>
    <p class="sub">One drone video in — a textured mesh, a dense point cloud and, with a GPS log, georeferenced outputs.</p>
    <div class="grid2">
      <div>
        <section class="card"><h2><span class="step">1</span>Video</h2>
          <div class="videos" id="videos"></div>
          <div class="drop" id="drop">Drop a video or flight log here, or <u>choose a file</u><input type="file" id="file" hidden></div>
          <div class="row" style="margin-top:12px">
            <label class="muted" for="telemetry">GPS / flight log</label>
            <select id="telemetry" style="max-width:360px"><option value="">none — model in SfM units, not georeferenced</option></select>
          </div>
          <div class="budget" id="budget" style="margin-top:10px"></div>
        </section>
        <section class="card"><h2><span class="step">2</span>Profile</h2><div class="profiles" id="profiles"></div></section>
        <section class="card"><h2><span class="step">3</span>Options</h2>
          <div class="row" style="margin-bottom:10px">
            <input type="text" class="search" id="search" placeholder="Search options…" value="${esc(state.filter)}">
            <label class="row muted"><span class="sw"><input type="checkbox" id="modonly" ${state.modifiedOnly ? 'checked' : ''}><span></span></span> changed only</label>
            <button class="link" id="resetall">reset all</button>
          </div>
          <div id="sections"></div>
        </section>
      </div>
      <div>
        <section class="card"><h2>Pipeline stages</h2><div class="chips" id="stages"></div>
          <p class="muted" style="font-size:12px;margin:10px 0 0">Stages exchange files only: any subset can be re-run on an existing run.</p></section>
        <section class="card"><h2><span class="step">4</span>Start</h2>
          <div class="row"><input type="text" id="runname" placeholder="run name (optional)" style="max-width:260px">
            <button class="btn primary" id="start" disabled>Reconstruct</button></div>
          <div id="starterr"></div>
          <details style="margin-top:12px"><summary class="muted" style="cursor:pointer">Changes against the profile</summary><pre class="yaml" id="yaml"></pre></details>
        </section>
      </div>
    </div>`;
  renderVideos(); renderLogs(); renderProfiles(); renderStages(); renderSections(); renderSummary();
  $('#search').addEventListener('input', (e) => { state.filter = e.target.value.toLowerCase(); renderSections(); });
  $('#modonly').addEventListener('change', (e) => { state.modifiedOnly = e.target.checked; renderSections(); });
  $('#resetall').addEventListener('click', () => { state.overrides = {}; renderSections(); renderSummary(); });
  $('#telemetry').addEventListener('change', (e) => { state.telemetry = e.target.value; renderSummary(); });
  $('#start').addEventListener('click', startRun);
  const drop = $('#drop'), file = $('#file');
  drop.addEventListener('click', () => file.click());
  file.addEventListener('change', () => file.files[0] && upload(file.files[0]));
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('hover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('hover'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('hover'); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
}

function renderVideos() {
  const el = $('#videos');
  if (!state.videos.length) { el.innerHTML = '<p class="muted">No videos in datasets/ or uploads/ yet.</p>'; return; }
  el.innerHTML = state.videos.map((v, i) => `
    <div class="vid ${state.video === v.path ? 'sel' : ''}" data-i="${i}" title="${esc(v.name)}">
      <div class="n">${esc(v.name.replace(/\s*\[[^\]]+\]\.\w+$/, '').replace(/\.\w+$/, ''))}</div>
      <div class="m">${v.width ? `${v.width}×${v.height} · ${v.fps} fps · ${fmtS(v.duration_s)}` : 'unreadable'} · ${v.size_mb} MB</div>
      <span class="pill ${v.origin === 'upload' ? 'up' : ''}">${v.origin}</span>
    </div>`).join('');
  el.querySelectorAll('.vid').forEach((d) => d.addEventListener('click', () => {
    state.video = state.videos[+d.dataset.i].path; renderVideos(); renderSummary();
  }));
}
function renderLogs() {
  const sel = $('#telemetry');
  sel.innerHTML = '<option value="">none — model in SfM units, not georeferenced</option>' +
    state.logs.map((l) => `<option value="${esc(l.path)}" ${state.telemetry === l.path ? 'selected' : ''}>${esc(l.name)} (${l.origin})</option>`).join('');
}
async function upload(f) {
  const drop = $('#drop');
  drop.textContent = `Uploading ${f.name}…`;
  const fd = new FormData(); fd.append('file', f);
  try {
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || r.statusText);
    [state.videos, state.logs] = await Promise.all([api('/api/videos'), api('/api/telemetry')]);
    if (state.videos.some((v) => v.path === body.path)) state.video = body.path; else state.telemetry = body.path;
    renderVideos(); renderLogs(); renderSummary();
    drop.innerHTML = `Uploaded <b>${esc(f.name)}</b>. Drop another file, or <u>choose one</u><input type="file" id="file" hidden>`;
  } catch (e) { drop.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  const file = $('#file'); if (file) file.addEventListener('change', () => file.files[0] && upload(file.files[0]));
}
function renderProfiles() {
  $('#profiles').innerHTML = state.profiles.map((p) => `
    <div class="prof ${p.name === state.profile ? 'sel' : ''}" data-n="${esc(p.name)}">
      <div class="n">${esc(p.name)}</div><div class="d">${esc((p.description || '').slice(0, 170))}${(p.description || '').length > 170 ? '…' : ''}</div></div>`).join('');
  document.querySelectorAll('.prof').forEach((d) => d.addEventListener('click', () => {
    state.profile = d.dataset.n; state.stages = null; renderProfiles(); renderStages(); renderSections(); renderSummary();
  }));
}
function currentStages() {
  if (state.stages) return state.stages;
  const pv = profileValue(null, 'stages');
  return pv || state.schema.stages;
}
function renderStages() {
  const on = new Set(currentStages());
  $('#stages').innerHTML = state.schema.stages.map((s) => `<span class="chip ${on.has(s) ? 'on' : ''}" data-s="${s}">${s}</span>`).join('');
  document.querySelectorAll('#stages .chip').forEach((c) => c.addEventListener('click', () => {
    const set = new Set(currentStages());
    set.has(c.dataset.s) ? set.delete(c.dataset.s) : set.add(c.dataset.s);
    state.stages = state.schema.stages.filter((s) => set.has(s)); renderStages(); renderSummary();
  }));
}
function control(section, f) {
  const dotted = section ? `${section}.${f.key}` : f.key;
  const v = effective(section, f);
  const id = `f_${dotted.replace(/\./g, '__')}`;
  if (f.type === 'bool') return `<label class="sw"><input type="checkbox" id="${id}" ${v ? 'checked' : ''}><span></span></label>`;
  if (f.choices) return `<select id="${id}">${(f.optional ? ['(none)'] : []).concat(f.choices).map((c) => `<option ${c === v || (c === '(none)' && v == null) ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select>`;
  if (f.type === 'int' || f.type === 'float') return `<input type="number" id="${id}" step="${f.type === 'int' ? 1 : 'any'}" value="${v ?? ''}" placeholder="${f.optional ? 'none' : ''}">`;
  if (f.type.startsWith('list')) return `<input type="text" id="${id}" value="${esc((v || []).join(', '))}" placeholder="comma-separated">`;
  return `<input type="text" id="${id}" value="${esc(v ?? '')}" placeholder="${f.optional ? 'none' : ''}">`;
}
function parseValue(f, el) {
  if (f.type === 'bool') return el.checked;
  const raw = el.value.trim();
  if (f.choices) return raw === '(none)' ? null : raw;
  if (raw === '' && f.optional) return null;
  if (f.type === 'int') return raw === '' ? f.default : parseInt(raw, 10);
  if (f.type === 'float') return raw === '' ? f.default : parseFloat(raw);
  if (f.type.startsWith('list')) {
    const items = raw ? raw.split(',').map((x) => x.trim()).filter(Boolean) : [];
    return f.type === 'list[int]' ? items.map((x) => parseInt(x, 10)) : f.type === 'list[float]' ? items.map(parseFloat) : items;
  }
  return raw;
}
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
function renderSections() {
  const skip = new Set(['run_name', 'stages']);
  const groups = [{ key: null, title: 'Run', fields: state.schema.top.filter((f) => !skip.has(f.key)) }, ...state.schema.sections];
  const q = state.filter;
  $('#sections').innerHTML = groups.map((g) => {
    const fields = g.fields.filter((f) => {
      const dotted = g.key ? `${g.key}.${f.key}` : f.key;
      if (g.key === 'ingest' && (f.key === 'video' || f.key === 'telemetry')) return false;
      if (state.modifiedOnly && !(dotted in state.overrides)) return false;
      return !q || dotted.toLowerCase().includes(q) || (f.help || '').toLowerCase().includes(q);
    });
    if (!fields.length) return '';
    const mods = fields.filter((f) => (g.key ? `${g.key}.${f.key}` : f.key) in state.overrides).length;
    const open = q || state.modifiedOnly || ['keyframes', 'sfm', 'dense', 'export'].includes(g.key);
    return `<details class="sec" ${open ? 'open' : ''}><summary><span class="t">${esc(g.key || 'run')}</span><span class="dsc">${esc(g.title)}</span>${mods ? `<span class="cnt">${mods} changed</span>` : ''}</summary><div class="body">` +
      fields.map((f) => {
        const dotted = g.key ? `${g.key}.${f.key}` : f.key;
        const mod = dotted in state.overrides;
        const pv = profileValue(g.key, f.key);
        const base = pv !== undefined ? `${state.profile}: ${JSON.stringify(pv)}` : `default: ${JSON.stringify(f.default)}`;
        return `<div class="field ${mod ? 'mod' : ''}" data-d="${esc(dotted)}"><label for="f_${dotted.replace(/\./g, '__')}">${esc(f.key)}</label>
          <div class="ctl">${control(g.key, f)}${mod ? `<button class="link" data-reset="${esc(dotted)}">reset</button>` : ''}</div>
          <div class="help">${esc(f.help)}${f.help ? ' · ' : ''}<span class="muted">${esc(base)}</span></div></div>`;
      }).join('') + '</div></details>';
  }).join('') || '<p class="muted">No option matches.</p>';
  for (const g of groups) for (const f of g.fields) {
    const dotted = g.key ? `${g.key}.${f.key}` : f.key;
    const el = document.getElementById(`f_${dotted.replace(/\./g, '__')}`);
    if (!el) continue;
    el.addEventListener('change', () => {
      const val = parseValue(f, el);
      const base = profileValue(g.key, f.key) !== undefined ? profileValue(g.key, f.key) : f.default;
      if (same(val, base)) delete state.overrides[dotted]; else state.overrides[dotted] = val;
      renderSections(); renderSummary();
    });
  }
  document.querySelectorAll('[data-reset]').forEach((b) => b.addEventListener('click', () => {
    delete state.overrides[b.dataset.reset]; renderSections(); renderSummary();
  }));
}
function renderSummary() {
  const v = state.videos.find((x) => x.path === state.video);
  $('#budget').innerHTML = v && v.duration_s
    ? `Time budget (problem statement: 15 min per 10-min video): <b>${fmtS(1.5 * v.duration_s)}</b> for this ${fmtS(v.duration_s)} video`
    : '<span class="muted">Choose a video.</span>';
  $('#start').disabled = !state.video;
  const lines = [`profile: ${state.profile}`, `video: ${state.video || '—'}`];
  if (state.telemetry) lines.push(`telemetry: ${state.telemetry}`);
  if (state.stages) lines.push(`stages: [${state.stages.join(', ')}]`);
  for (const [k, val] of Object.entries(state.overrides)) lines.push(`${k}: ${JSON.stringify(val)}`);
  $('#yaml').textContent = lines.join('\n');
}
async function startRun() {
  $('#starterr').innerHTML = '';
  $('#start').disabled = true;
  try {
    const r = await api('/api/runs', { method: 'POST', body: JSON.stringify({
      name: $('#runname').value.trim(), profile: state.profile, video: state.video, telemetry: state.telemetry || null,
      stages: state.stages, overrides: state.overrides }) });
    location.hash = `#/run/${encodeURIComponent(r.name)}`;
  } catch (e) { $('#starterr').innerHTML = `<div class="err">${esc(e.message)}</div>`; $('#start').disabled = false; }
}

// ------------------------------------------------------------------ runs
async function viewRuns() {
  const main = $('#main');
  const draw = async () => {
    let runs;
    try { runs = await api('/api/runs'); } catch (e) { main.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
    main.innerHTML = `<h1>Runs</h1><p class="sub">Every run directory under outputs/, whether started here or from the command line.</p>
      <section class="card"><table class="runs"><thead><tr><th>Run</th><th>Status</th><th>Video</th><th>Keyframes</th><th>Completeness</th><th>Time vs budget</th></tr></thead><tbody>` +
      runs.map((r) => {
        const s = r.summary, p = s.processing;
        const frac = p && p.budget_seconds ? p.seconds / p.budget_seconds : null;
        return `<tr data-n="${esc(r.name)}"><td><b>${esc(r.name)}</b></td><td><span class="st ${esc(r.status)}">${esc(r.status)}</span></td>
          <td class="muted">${esc((s.video || '').replace(/\s*\[[^\]]+\]\.\w+$/, '').slice(0, 42))}</td>
          <td>${s.registered != null ? `${s.registered}/${s.keyframes}` : '—'}</td><td>${pct(s.completeness)}</td>
          <td>${p ? `<div class="row"><div class="bar" style="width:110px"><i class="${frac > 1 ? 'over' : ''}" style="width:${Math.min(100, 100 * frac)}%"></i></div><span class="muted">${fmtS(p.seconds)} / ${fmtS(p.budget_seconds)}</span></div>` : '—'}</td></tr>`;
      }).join('') + '</tbody></table></section>';
    main.querySelectorAll('tr[data-n]').forEach((tr) => tr.addEventListener('click', () => { location.hash = `#/run/${encodeURIComponent(tr.dataset.n)}`; }));
  };
  await draw();
  timer = setInterval(draw, 5000);
}

// ------------------------------------------------------------------ one run
async function viewRun(name) {
  const main = $('#main');
  let viewerShown = false;
  const draw = async () => {
    let r;
    try { r = await api(`/api/runs/${encodeURIComponent(name)}`); } catch (e) { main.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
    const s = r.summary, p = s.processing;
    const planned = (r.config.stages && r.config.stages.length ? r.config.stages : Object.keys(r.stages));
    const now = Date.now() / 1000;
    const elapsed = r.job.started ? ((r.job.finished || now) - r.job.started) : null;
    const budget = p?.budget_seconds ?? (s.video_seconds ? 1.5 * s.video_seconds : null);
    const used = p?.seconds ?? elapsed;
    const frac = budget && used != null ? used / budget : null;
    if (!$('#runhead')) {
      main.innerHTML = `<div id="runhead"></div><section class="card"><h2>Pipeline</h2><div class="stages" id="stg"></div>
        <div class="row" style="margin-top:12px"><span class="muted">Time vs budget</span><div class="bar" style="flex:1;max-width:420px"><i id="bb"></i></div><span class="budget" id="bt"></span></div></section>
        <div id="results"></div>
        <section class="card"><h2>Log</h2><div class="log" id="log"></div></section>`;
    }
    $('#runhead').innerHTML = `<div class="row"><h1 style="margin-right:8px">${esc(r.name)}</h1><span class="st ${esc(r.status)}">${esc(r.status)}</span>
      ${['running', 'queued'].includes(r.status) ? '<button class="btn danger small" id="stop">Stop</button>' : ''}
      <span style="margin-left:auto" class="row">${s.report ? `<a class="btn small" href="/runs/${encodeURIComponent(r.name)}/report.html" target="_blank">HTML report</a>` : ''}
      ${s.viewer ? `<a class="btn small" href="/runs/${encodeURIComponent(r.name)}/export/index.html" target="_blank">Open viewer</a>` : ''}</span></div>
      <p class="sub">${esc(s.video || '')}${s.resolution && s.resolution[0] ? ` · ${s.resolution[0]}×${s.resolution[1]}` : ''}${s.video_seconds ? ` · ${fmtS(s.video_seconds)}` : ''} · profile config: <span class="muted">outputs/${esc(r.name)}/ui_config.yaml</span></p>`;
    if ($('#stop')) $('#stop').onclick = async () => { await api(`/api/runs/${encodeURIComponent(r.name)}/stop`, { method: 'POST' }); draw(); };
    $('#stg').innerHTML = planned.map((st) => {
      const x = r.stages[st] || { status: 'pending' };
      return `<div class="stg ${esc(x.status)}"><div class="n">${esc(st)}</div><div class="s"><span class="st ${esc(x.status)}">${esc(x.status)}</span> ${x.seconds != null ? fmtS(x.seconds) : ''}</div></div>`;
    }).join('');
    $('#bb').style.width = frac != null ? `${Math.min(100, 100 * frac)}%` : '0';
    $('#bb').className = frac > 1 ? 'over' : '';
    $('#bt').innerHTML = used != null ? `<b>${fmtS(used)}</b> of ${fmtS(budget)}${frac != null ? ` (${Math.round(100 * frac)} %)` : ''}` : '—';
    const log = $('#log'); const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
    log.textContent = (r.log_tail || []).join('\n'); if (atEnd) log.scrollTop = log.scrollHeight;
    if (['done', 'failed', 'stopped', 'unknown'].includes(r.status) && !viewerShown) {
      viewerShown = true;
      const geo = s.georef.length ? s.georef.map((g) => `${g.model}: ${g.mode}, held-out ${g.loo_rmse_horizontal_m?.toFixed(2)} m`).join('; ') : 'no GPS log: SfM units, levelled';
      $('#results').innerHTML = `<section class="card"><h2>Result</h2><div class="tiles">
          <div class="tile"><div class="k">Keyframes registered</div><div class="v">${s.registered ?? '—'}<small> / ${s.keyframes ?? '—'}</small></div></div>
          <div class="tile"><div class="k">Models (passes)</div><div class="v">${s.models ?? '—'}</div></div>
          <div class="tile"><div class="k">Mesh triangles</div><div class="v">${fmtN(s.triangles)}</div></div>
          <div class="tile"><div class="k">Completeness</div><div class="v">${pct(s.completeness)}<small> of what the camera saw</small></div></div>
          <div class="tile"><div class="k">Processing</div><div class="v">${fmtS(p?.seconds)}<small> / ${fmtS(p?.budget_seconds)}</small></div></div>
          <div class="tile"><div class="k">Georeferencing</div><div class="v" style="font-size:13px;font-weight:500">${esc(geo)}</div></div></div></section>
        ${s.viewer ? `<section class="card"><h2>3D model</h2><iframe class="viewer" src="/runs/${encodeURIComponent(r.name)}/export/index.html" title="3D viewer"></iframe></section>` : ''}
        ${s.files.length ? `<section class="card"><h2>Downloads</h2>${s.files.map((m) => `<div style="margin-bottom:8px"><b>model ${esc(m.model)}</b> <span class="muted">${fmtN(m.triangles)} triangles · ${fmtN(m.points)} points${m.epsg ? ` · EPSG:${m.epsg}` : ''}</span><div class="files">${m.files.map((f) => `<a href="/runs/${encodeURIComponent(r.name)}/export/${esc(f)}" download>${esc(f.split('/').pop())}</a>`).join('')}</div></div>`).join('')}</section>` : ''}`;
    }
    if (!['running', 'queued', 'stopping'].includes(r.status)) clearInterval(timer);
  };
  main.innerHTML = '';
  await draw();
  timer = setInterval(draw, 2000);
}

// ------------------------------------------------------------------ about
function viewAbout() {
  $('#main').innerHTML = `<div class="about"><h1>About</h1>
    <section class="card"><h2>What it does</h2><p>drone3d turns a single-pass drone video into a textured 3D mesh and a dense point cloud of the visible scene, georeferenced in UTM when a GPS log (DJI SRT, CSV, GPX or JSON) is supplied. Outputs: OBJ, PLY, LAS, GeoTIFF (surface model and orthophoto), glTF/GLB and FBX, with a web viewer that measures distances and heights.</p></section>
    <section class="card"><h2>Profiles</h2><p><b>fast</b> — the problem statement's budget (15 minutes per 10-minute video): one GPU decode, motion-adaptive analysis, camera poses from optical-flow tracks, depth from flow triangulation completed by Depth Anything V2, GPU TSDF fusion and GPU texturing. <b>quality</b> — the same method with denser keyframes and full-resolution depth. <b>accurate / default</b> — SIFT structure from motion, Marigold v2 depth and 3D Gaussian Splatting; hours per clip.</p></section>
    <section class="card"><h2>How to read a run</h2><p>Each stage writes its own result files under <code>outputs/&lt;run&gt;/</code>, so a stage can be re-run alone. <i>Completeness</i> is the share of each registered keyframe's non-sky pixels whose ray hits the reconstructed mesh; <i>time vs budget</i> compares processing time with 1.5× the video length.</p></section></div>`;
}

route();
