// Build: one drone video in, a complete 3D model out. Source, profile, resolution, modules, every option.
import { $, $$, api, esc, fmtS, post, prettyVideo, toast } from './util.js';
import { MODULES, Options, renderModules, renderOptions, renderProfiles, renderResolution, renderStages } from './options.js';

const keep = { video: null, telemetry: '', opts: null, next: true };

export async function viewNew(main) {
  main.innerHTML = '<div class="note">loading…</div>';
  let schema, profiles, videos, logs, engine;
  try {
    [schema, profiles, videos, logs, engine] = await Promise.all([api('/api/schema'), api('/api/profiles'), api('/api/videos'), api('/api/telemetry'), api('/api/engine')]);
  } catch (e) { main.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  let o = keep.opts && keep.opts.profiles.length === profiles.length ? keep.opts : null;
  const defaults = (opts) => { // a new build makes everything: mesh, cloud, depth, texture, Gaussian splats, STL and Blender files
    if (!opts.stages().includes('splat')) MODULES.find((m) => m.key === 'splat').set(opts, true);
    if (opts.profile === 'fast') opts.override('splat.models', 'all');
  };
  if (!o) {
    o = new Options(schema, profiles, profiles.some((p) => p.name === 'fast') ? 'fast' : profiles[0]?.name);
    defaults(o);
  }
  o.onProfile = () => defaults(o);
  o.schema = schema; o.profiles = profiles; keep.opts = o;
  const st = { videos, logs, engine };

  main.innerHTML = `
  <div class="page-head rise"><h1>Build a 3D model<small>A single-pass drone video in; a textured mesh, a dense point cloud and — with a flight log — georeferenced deliverables out.</small></h1></div>
  <div class="grid g-12">
    <div class="s8 stack">
      <section class="panel rise" style="--i:1"><h2>1 · Source video <span class="tag">datasets/ and uploads/</span></h2>
        <input type="text" id="vq" placeholder="filter videos…" style="max-width:280px;margin-bottom:10px">
        <div class="choices scroll" id="videos"></div>
        <div class="drop" id="drop">Drop a video or flight log here, or <u>choose a file</u><input type="file" id="file" hidden></div>
        <div class="row" style="margin-top:12px"><span class="muted" style="min-width:130px">GPS / flight log</span>
          <select id="telemetry" style="max-width:420px"></select></div>
      </section>
      <section class="panel rise" style="--i:2"><h2>2 · Profile</h2><div class="choices" id="profiles"></div></section>
      <section class="panel rise" style="--i:3"><h2>3 · Resolution</h2><div id="res"></div></section>
      <section class="panel rise" style="--i:4"><h2>4 · Modules</h2><div id="mods"></div></section>
      <section class="panel rise" style="--i:5"><h2>5 · Every option <span class="tag">from the config dataclasses — nothing hidden</span><span class="grow"></span>
          <input type="text" id="oq" placeholder="search options…" style="width:200px">
          <label class="row muted" style="gap:6px;font-size:12.5px"><span class="sw"><input type="checkbox" id="modonly"><span></span></span>changed</label>
          <button class="linkbtn" id="resetall">reset all</button></h2>
        <div id="sections"></div></section>
    </div>
    <div class="s4">
      <section class="panel rise" style="--i:2;position:sticky;top:calc(var(--top) + 18px)" id="launch"></section>
    </div>
  </div>`;

  const redraw = () => { renderProfiles($('#profiles'), o, redraw); renderResolution($('#res'), o, redraw); renderModules($('#mods'), o, redraw); renderOptions($('#sections'), o); launch(); };
  o.onChange = () => { renderResolution($('#res'), o, redraw); renderModules($('#mods'), o, redraw); launch(); };

  function drawVideos() {
    const q = ($('#vq').value || '').toLowerCase();
    const list = st.videos.filter((v) => !q || v.name.toLowerCase().includes(q));
    $('#videos').innerHTML = list.map((v) => `<div class="choice ${keep.video === v.path ? 'sel' : ''}" data-p="${esc(v.path)}" title="${esc(v.name)}">
      <div class="n">${esc(prettyVideo(v.name))}</div>
      <div class="m">${v.width ? `${v.width}×${v.height} · ${v.fps} fps · ${fmtS(v.duration_s)}` : 'unreadable'} · ${v.size_mb} MB</div>
      <div class="row" style="margin-top:6px"><span class="pill">${esc(v.codec || '?')}</span>${v.origin === 'upload' ? '<span class="pill" style="color:var(--cyan)">upload</span>' : ''}</div></div>`).join('') || '<span class="note">No videos match.</span>';
    $$('#videos .choice').forEach((d) => d.addEventListener('click', () => { keep.video = d.dataset.p; drawVideos(); launch(); }));
  }
  function drawLogs() {
    $('#telemetry').innerHTML = '<option value="">none — levelled model in SfM units</option>' + st.logs.map((l) => `<option value="${esc(l.path)}" ${keep.telemetry === l.path ? 'selected' : ''}>${esc(l.name)} (${l.origin})</option>`).join('');
  }
  // XHR rather than fetch: a 10-minute 4K flight is gigabytes, and only XHR reports upload progress.
  const send = (f, onProgress) => new Promise((resolve, reject) => {
    const x = new XMLHttpRequest(), fd = new FormData(); fd.append('file', f);
    x.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    x.onload = () => { let body = {}; try { body = JSON.parse(x.responseText); } catch { /* not JSON */ }
      x.status < 400 ? resolve(body) : reject(new Error(body.detail || `${x.status} ${x.statusText}`)); };
    x.onerror = () => reject(new Error('the upload was interrupted'));
    x.open('POST', '/api/upload'); x.send(fd);
  });
  let uploading = false;
  async function upload(f) {
    if (uploading) { toast('An upload is still running', 'bad'); return; }
    uploading = true;
    const drop = $('#drop'), t0 = performance.now();
    drop.innerHTML = `<div>Uploading <b>${esc(f.name)}</b> (${(f.size / 1e6).toFixed(0)} MB) <span class="mono" id="upPct">0 %</span></div><div class="bar" style="margin-top:8px"><i id="upBar" style="width:0"></i></div>`;
    try {
      const body = await send(f, (q) => { const s = (performance.now() - t0) / 1000;
        $('#upPct').textContent = `${(100 * q).toFixed(0)} %${q > 0.02 && q < 1 ? ` · ${fmtS(s * (1 - q) / q)} left` : ''}`; $('#upBar').style.width = `${100 * q}%`; });
      [st.videos, st.logs] = await Promise.all([api('/api/videos'), api('/api/telemetry')]);
      if (st.videos.some((v) => v.path === body.path)) keep.video = body.path; else keep.telemetry = body.path;
      drawVideos(); drawLogs(); launch();
      const what = body.reused ? `<b>${esc(body.name)}</b> was already uploaded — using it` : `Uploaded <b>${esc(body.name)}</b>`;
      drop.innerHTML = `${what} — drop another, or <u>choose a file</u><input type="file" id="file" hidden>`;
      toast(body.reused ? `${body.name}: already uploaded` : `Uploaded ${body.name}`, 'ok');
    } catch (e) { drop.innerHTML = `<span class="err">${esc(e.message)}</span> — drop a file to try again, or <u>choose one</u><input type="file" id="file" hidden>`; }
    uploading = false; bindFile();
  }
  function bindFile() { const file = $('#file'); if (file) file.onchange = () => file.files[0] && upload(file.files[0]); }

  function launch() {
    const v = st.videos.find((x) => x.path === keep.video);
    const warm = (st.engine.models || []).filter((m) => m.status === 'loaded').map((m) => m.title);
    const budget = v?.duration_s ? 1.5 * v.duration_s : null;
    $('#launch').innerHTML = `
      <h2>Launch</h2>
      ${v ? `<div class="n" style="font-weight:600">${esc(prettyVideo(v.name))}</div>${v.width ? `<div class="note mono">${v.width}×${v.height} · ${v.fps} fps · ${esc(v.codec)} · ${fmtS(v.duration_s)}</div>` : '<div class="err" style="margin-top:6px">This file cannot be read as a video.</div>'}` : '<div class="note">Choose a video.</div>'}
      <div class="hr"></div>
      <div class="between"><span class="muted">Time budget</span><span class="fig sm">${budget ? fmtS(budget) : '—'}</span></div>
      <div class="note">problem statement: under 15 minutes of processing per 10 minutes of video</div>
      <div class="hr"></div>
      <div class="between"><span class="muted">Profile</span><b class="mono">${esc(o.profile)} · ${esc(o.resolution())}</b></div>
      <div style="margin-top:6px"><div class="between"><span class="muted">Runs on</span><span class="row" style="gap:6px;flex-wrap:nowrap"><span class="led ${st.engine.online ? 'ok' : ''}"></span><b class="mono" style="font-size:12px">${st.engine.online ? 'warm engine' : 'cold subprocess'}</b></span></div>
        ${warm.length ? `<div class="note mono" style="text-align:right;margin-top:2px">${esc(warm.join(' · '))}</div>` : ''}</div>
      <div style="margin-top:12px"><div class="muted" style="margin-bottom:6px">Stages</div><div class="chips" id="stages"></div></div>
      <div class="hr"></div>
      <label class="row" style="gap:8px;margin-bottom:8px;font-size:13px"><span class="sw"><input type="checkbox" id="runnext" ${keep.next ? 'checked' : ''}><span></span></span>Run next — ahead of anything queued</label>
      <input type="text" id="runname" placeholder="run name (optional)" value="${esc($('#runname')?.value || '')}">
      <button class="btn primary" id="start" style="width:100%;justify-content:center;margin-top:10px;padding:12px" ${v?.width ? '' : 'disabled'}>Build the 3D model</button>
      <div id="starterr"></div>
      <details class="fold" style="margin-top:12px"><summary class="muted">Changes against the profile</summary><pre class="yaml">${esc(o.summary())}</pre></details>`;
    renderStages($('#stages'), o, redraw);
    $('#start').onclick = start;
    $('#runnext').onchange = (e) => { keep.next = e.target.checked; };
  }

  async function start() {
    $('#start').disabled = true; $('#starterr').innerHTML = '';
    try {
      const r = await post('/api/runs', { name: $('#runname').value.trim(), video: keep.video, telemetry: keep.telemetry || null, next: !!keep.next, ...o.payload() });
      toast(`${r.name}: ${r.via === 'engine' ? 'started on the warm engine' : 'queued as a subprocess'}`, 'ok');
      location.hash = `#/run/${encodeURIComponent(r.name)}`;
    } catch (e) { $('#starterr').innerHTML = `<div class="err">${esc(e.message)}</div>`; $('#start').disabled = false; }
  }

  drawVideos(); drawLogs(); redraw(); bindFile();
  $('#vq').addEventListener('input', drawVideos);
  $('#telemetry').addEventListener('change', (e) => { keep.telemetry = e.target.value; launch(); });
  $('#oq').addEventListener('input', (e) => { o.filter = e.target.value.toLowerCase(); renderOptions($('#sections'), o); });
  $('#modonly').addEventListener('change', (e) => { o.modifiedOnly = e.target.checked; renderOptions($('#sections'), o); });
  $('#resetall').addEventListener('click', () => { o.overrides = {}; o.stageList = null; redraw(); });
  const drop = $('#drop');
  drop.addEventListener('click', (e) => { const f = $('#file'); if (f && e.target.id !== 'file') f.click(); }); // no input while uploading
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('hover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('hover'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('hover'); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });
}
