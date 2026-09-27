// Console: is everything alive and warm? The engine, its models, the GPU, the host, the work in flight.
import { $, api, ago, esc, every, fmtN, fmtS, gb, icon, post, prettyVideo, toast } from './util.js';
import { areaChart, ring, stack } from './charts.js';

const state = { series: [], since: 0, profiles: [], warmProfile: 'fast', caps: null, capsPid: null };

export async function viewConsole(main) {
  main.innerHTML = `
  <div class="page-head rise">
    <h1>Mission console<small>The warm GPU engine, its models and the card they share — everything a flight needs before it lands.</small></h1>
    <div class="row" style="margin-left:auto">
      <span class="pill">≤ 15 min per 10-min video</span><span class="pill">≤ 1 m spatial accuracy</span><span class="pill">OBJ · PLY · LAS · GeoTIFF · GLB · FBX</span>
    </div>
  </div>
  <div class="grid g-12">
    <section class="panel s4 rise" style="--i:1" id="enginePanel"></section>
    <section class="panel s5 rise" style="--i:2" id="gpuPanel"></section>
    <section class="panel s3 rise" style="--i:3" id="hostPanel"></section>
    <section class="panel s12 rise" style="--i:4">
      <h2>Models <span class="tag">kept on the GPU between runs — a video starts without loading anything</span><span class="grow"></span>
        <select id="warmProfile" style="width:auto"></select>
        <button class="btn small" id="warmBtn">${icon.bolt}Warm for profile</button>
        <button class="btn small ghost" id="unloadAll">Unload all</button></h2>
      <div class="models" id="models"></div>
    </section>
    <section class="panel s8 rise" style="--i:5">
      <h2>GPU, last 10 minutes <span class="grow"></span><span class="legend"><span><i style="background:#4fd1e8"></i>SM utilisation</span><span><i style="background:#ff9f1c"></i>power, % of limit</span><span><i style="background:#a78bfa"></i>NVDEC</span></span></h2>
      <canvas class="chart tall" id="chUtil"></canvas>
      <div class="hr"></div>
      <h2>Memory <span class="grow"></span><span class="legend"><span><i style="background:#8b9bac"></i>all processes</span><span><i style="background:#ff9f1c"></i>drone3d</span></span></h2>
      <canvas class="chart" id="chMem"></canvas>
    </section>
    <section class="panel s4 rise" style="--i:6">
      <h2>On the GPU <span class="tag">ours in saffron — other users' work is never touched</span></h2>
      <div id="procs"></div>
    </section>
    <section class="panel s12 rise" style="--i:7">
      <h2>What this machine can do <span class="tag">checked by the engine, not assumed</span><span class="grow"></span><button class="btn tiny ghost" id="recheck">check</button></h2>
      <div class="caps" id="caps"><span class="note">Start the engine to run the checks.</span></div>
    </section>
    <section class="panel s6 rise" style="--i:8"><h2>In flight</h2><div id="work"></div></section>
    <section class="panel s6 rise" style="--i:9"><h2>Recent runs <span class="grow"></span><a class="btn tiny ghost" href="#/runs">all runs</a></h2><div id="recent"></div></section>
  </div>`;

  state.profiles = await api('/api/profiles').catch(() => []);
  $('#warmProfile').innerHTML = state.profiles.map((p) => `<option ${p.name === state.warmProfile ? 'selected' : ''}>${esc(p.name)}</option>`).join('');
  $('#warmProfile').onchange = (e) => { state.warmProfile = e.target.value; };
  $('#warmBtn').onclick = () => act(() => post('/api/engine/warm', { profile: state.warmProfile }), 'Warming models…', (r) => `Warm: ${Object.entries(r.result || {}).map(([k, v]) => `${k} ${v}`).join(', ') || 'nothing to load'}`);
  $('#unloadAll').onclick = () => act(() => post('/api/engine/unload'), 'Unloading…', () => 'Models unloaded; GPU memory returned');
  $('#recheck').onclick = () => { state.caps = null; loadCaps(); };

  await Promise.all([gpuTick(), engineTick(), activityTick()]);
  every(1000, gpuTick);
  every(2000, engineTick);
  every(4000, activityTick);
}

async function act(fn, pending, done) {
  toast(pending);
  try { const r = await fn(); toast(done(r), 'ok'); } catch (e) { toast(e.message, 'bad'); }
  engineTick();
}

// ---------------------------------------------------------------- engine
async function engineTick() {
  let e;
  try { e = await api('/api/engine'); } catch { return; }
  const panel = $('#enginePanel'); if (!panel) return;
  const busy = e.online && e.current;
  const led = !e.online ? 'bad' : e.poisoned ? 'warn' : busy ? 'busy' : 'ok';
  const loaded = (e.models || []).filter((m) => m.status === 'loaded');
  panel.innerHTML = `
    <h2>Engine <span class="grow"></span><span class="st ${e.online ? (busy ? 'running' : 'done') : 'failed'}">${e.online ? (busy ? 'busy' : 'online') : 'offline'}</span></h2>
    <div class="row" style="gap:16px;align-items:flex-end">
      <span class="led ${led}" style="width:14px;height:14px"></span>
      <div class="fig md">${e.online ? `${loaded.length}<small>warm model${loaded.length === 1 ? '' : 's'}</small>` : 'cold'}</div>
    </div>
    <div class="hr"></div>
    <dl class="kv">
      ${e.online ? `<dt>process</dt><dd>pid ${e.pid} · up ${fmtS(e.uptime_s)}</dd><dt>runtime</dt><dd>torch ${esc(e.torch)} · ${esc(e.device)}</dd>
      <dt>held by engine</dt><dd>${gb(e.memory?.allocated_mb)} GB used / ${gb(e.memory?.reserved_mb)} GB reserved</dd>
      <dt>work</dt><dd>${busy ? esc((e.running || [e.current]).map((j) => j.name).join(', ')) : 'idle'}${e.queue?.length ? ` · ${e.queue.length} queued` : ''}</dd>
      <dt>slots</dt><dd>${e.slots ?? 1} run${(e.slots ?? 1) > 1 ? 's' : ''} at once</dd>` : `<dt>address</dt><dd>${esc(e.url)}</dd><dt>state</dt><dd>${esc(e.error ? 'not running' : '')}</dd>`}
      <dt>restarts</dt><dd>${e.restarts}${e.last_exit != null ? ` · last exit ${e.last_exit}` : ''}</dd>
    </dl>
    ${e.poisoned ? `<div class="err">${esc(e.poisoned)}</div>` : ''}
    <div class="row" style="margin-top:14px">
      ${e.online ? `<button class="btn small danger" id="engStop">${icon.power}Stop engine</button>` : `<button class="btn primary" id="engStart">${icon.power}Bring engine online</button>`}
    </div>`;
  const start = $('#engStart'), stop = $('#engStop');
  if (start) start.onclick = () => act(() => post('/api/engine/start', { warm: null }), 'Starting the engine (torch + CUDA)…', () => 'Engine online');
  if (stop) stop.onclick = () => act(() => post('/api/engine/stop', { force: false }), 'Stopping the engine…', () => 'Engine stopped; its GPU memory is free');
  renderModels(e);
  if (e.online && !state.caps && state.capsPid !== e.pid) { state.capsPid = e.pid; loadCaps(); }
}

function renderModels(e) {
  const el = $('#models'); if (!el) return;
  const models = e.models || [];
  if (!e.online) { el.innerHTML = '<span class="note">The engine is offline: runs start cold in a subprocess (models loaded per run). Bring it online to keep them warm.</span>'; return; }
  el.innerHTML = models.map((m) => `
    <div class="model ${m.status}">
      <div class="between"><div class="name">${esc(m.title)}</div><span class="st ${m.status}">${m.unload_pending ? 'unloading after run' : m.status === 'loaded' ? 'warm' : m.status}</span></div>
      <div class="role">${esc(m.role)}</div>
      <div class="meta">${m.status === 'loaded' ? `${fmtN(m.vram_mb)} MB · loaded in ${fmtS(m.load_seconds)} · used ${m.uses}×` : `needs ~${m.need_gb} GB free`} · ${esc(m.license)}</div>
      ${m.error ? `<div class="err" style="margin:0">${esc(m.error)}</div>` : ''}
      <div class="row">${m.status === 'loaded' ? `<button class="btn tiny" data-unload="${m.key}">Unload</button>` : `<button class="btn tiny" data-load="${m.key}">${icon.bolt}Load</button>`}
        <span class="faint mono" style="font-size:11px">${esc(m.stages.join(' · '))}</span></div>
    </div>`).join('');
  el.querySelectorAll('[data-load]').forEach((b) => (b.onclick = () => { b.disabled = true; b.textContent = 'loading…'; act(() => post(`/api/engine/models/${b.dataset.load}/load`), `Loading ${b.dataset.load}…`, (r) => `${b.dataset.load}: ${Object.values(r)[0]}`); }));
  el.querySelectorAll('[data-unload]').forEach((b) => (b.onclick = () => act(() => post(`/api/engine/models/${b.dataset.unload}/unload`), `Unloading ${b.dataset.unload}…`, (r) => `${b.dataset.unload}: ${Object.values(r)[0]}`)));
}

async function loadCaps() {
  const el = $('#caps'); if (!el) return;
  el.innerHTML = '<span class="note">checking codecs, libraries and weights…</span>';
  try {
    state.caps = await api('/api/engine/capabilities');
    const label = { cuda: 'CUDA', nvdec: 'NVDEC decode', nvenc: 'NVENC encode', nvjpeg: 'nvJPEG', open3d: 'Open3D GPU', pycolmap: 'COLMAP / GLOMAP', spirula: 'Gaussian splat trainer', meshconv: 'FBX writer', depth_anything: 'Depth Anything V2', marigold: 'Marigold v2' };
    el.innerHTML = Object.entries(state.caps).map(([k, c]) => `<div class="cap"><span class="led ${c.ok ? 'ok' : 'bad'}"></span><div><b>${esc(label[k] || k)}</b><span>${esc(c.detail)}</span></div></div>`).join('');
  } catch (e) { el.innerHTML = `<span class="note">${esc(e.message)}</span>`; }
}

// ------------------------------------------------------------------- gpu
async function gpuTick() {
  let r;
  try { r = await api(`/api/gpu?since=${state.since}`); } catch { return; }
  state.series.push(...r.series);
  if (r.series.length) state.since = r.series[r.series.length - 1].t;
  state.series = state.series.filter((s) => r.now - s.t < 600);
  const g = r.latest?.gpus?.[0];
  const panel = $('#gpuPanel'); if (!panel) return;
  if (!g) { panel.innerHTML = '<h2>GPU</h2><span class="note">NVML unavailable</span>'; return; }
  const own = g.processes.filter((p) => p.own).reduce((a, p) => a + p.used_mb, 0);
  const others = Math.max(0, g.mem_used_mb - own);
  panel.innerHTML = `
    <h2>${esc(g.name)} <span class="tag">driver ${esc(r.latest.driver)} · CUDA ${esc(r.latest.cuda)}</span></h2>
    <div class="row" style="gap:18px;flex-wrap:nowrap">
      ${ring(g.util / 100, { label: `${g.util}%`, sub: 'SM busy', color: '#4fd1e8' })}
      ${ring(g.mem_used_mb / g.mem_total_mb, { label: gb(g.mem_used_mb), sub: `of ${gb(g.mem_total_mb)} GB`, color: '#ff9f1c' })}
      <dl class="kv" style="flex:1">
        <dt>power</dt><dd>${g.power_w} / ${g.power_limit_w} W</dd>
        <dt>temperature</dt><dd>${g.temp_c} °C</dd>
        <dt>SM clock</dt><dd>${g.sm_clock_mhz} MHz</dd>
        <dt>NVDEC · NVENC</dt><dd>${g.decoder_util ?? '—'}% · ${g.encoder_util ?? '—'}%</dd>
      </dl>
    </div>
    <div style="margin-top:12px">${stack([{ value: own, color: '#ff9f1c', label: 'drone3d' }, { value: others, color: '#5b6b7c', label: 'other users' }], g.mem_total_mb)}</div>
    <div class="legend" style="margin-top:6px"><span><i style="background:#ff9f1c"></i>drone3d ${gb(own)} GB</span><span><i style="background:#5b6b7c"></i>others ${gb(others)} GB</span><span><i style="background:#1b2735"></i>free ${gb(g.mem_free_mb)} GB</span></div>`;
  const now = r.now;
  const lim = g.power_limit_w || 300;
  const utilSeries = state.series.map((s) => ({ ...s, power_pct: (100 * (s.power_w || 0)) / lim }));
  areaChart($('#chUtil'), utilSeries, [{ key: 'util', color: '#4fd1e8', fill: '#4fd1e833' }, { key: 'power_pct', color: '#ff9f1c' }, { key: 'dec', color: '#a78bfa' }], { max: 100, now, unit: '%' });
  const memSeries = state.series.map((s) => ({ ...s, used_gb: s.mem_used_mb / 1024, own_gb: s.own_mb / 1024 }));
  areaChart($('#chMem'), memSeries, [{ key: 'used_gb', color: '#8b9bac', fill: '#8b9bac22' }, { key: 'own_gb', color: '#ff9f1c', fill: '#ff9f1c33' }], { max: g.mem_total_mb / 1024, now, unit: 'G' });
  $('#procs').innerHTML = g.processes.length ? `<table class="t"><thead><tr><th>pid</th><th>process</th><th style="text-align:right">GPU memory</th></tr></thead><tbody>${g.processes.slice(0, 9).map((p) => `
    <tr><td class="mono ${p.own ? 'own' : 'faint'}">${p.pid}</td><td class="${p.own ? 'own' : ''}">${esc(p.name || '?')}${p.own ? ' <span class="pill" style="border-color:#ff9f1c55;color:#ff9f1c">ours</span>' : ''}</td><td class="mono" style="text-align:right">${fmtN(p.used_mb)} MB</td></tr>`).join('')}</tbody></table>` : '<span class="note">no compute processes</span>';
}

// --------------------------------------------------------------- activity
async function activityTick() {
  let sys, runs;
  try { [sys, runs] = await Promise.all([api('/api/system'), api('/api/runs')]); } catch { return; }
  const h = sys.host;
  const hp = $('#hostPanel'); if (!hp) return;
  hp.innerHTML = `<h2>Host</h2>
    <div class="fig md">${h.load1}<small>load · ${h.cpus} cores</small></div>
    <div class="hr"></div>
    <dl class="kv"><dt>memory free</dt><dd>${h.mem_available_gb} / ${h.mem_total_gb} GB</dd><dt>disk free</dt><dd>${sys.disk_free_gb} GB</dd>
    <dt>cold queue</dt><dd>${sys.queue.length ? esc(sys.queue.join(', ')) : 'empty'}</dd></dl>
    <div style="margin-top:12px">${stack([{ value: h.mem_total_gb - h.mem_available_gb, color: '#4fd1e8', label: 'in use' }], h.mem_total_gb)}</div>`;
  const active = runs.filter((r) => ['running', 'queued', 'stopping'].includes(r.status));
  $('#work').innerHTML = active.length ? active.map((r) => runRow(r, true)).join('') : '<span class="note">Nothing running. <a href="#/new">Build a model</a> or <a href="#/live">go live</a>.</span>';
  $('#recent').innerHTML = runs.filter((r) => !active.includes(r)).slice(0, 6).map((r) => runRow(r)).join('') || '<span class="note">no runs yet</span>';
}

function runRow(r, live = false) {
  const p = r.summary.processing;
  const frac = p?.budget_seconds ? p.seconds / p.budget_seconds : null;
  const stages = Object.entries(r.stages || {});
  const done = stages.filter(([, s]) => ['ok', 'skipped', 'failed'].includes(s.status)).length;
  return `<a class="between" href="#/run/${encodeURIComponent(r.name)}" style="padding:9px 0;border-top:1px solid #18222e;color:inherit;text-decoration:none">
    <div style="min-width:0"><b>${esc(r.name)}</b> <span class="st ${esc(r.status)}">${esc(r.status)}</span>${r.kind === 'live' ? ' <span class="pill">live</span>' : ''}
      <div class="note">${esc(prettyVideo(r.summary.video))}${live && r.current ? ` · stage <span class="mono">${esc(r.current)}</span> (${done}/${stages.length})` : ` · ${ago(r.mtime)}`}</div></div>
    ${frac != null ? `<div style="text-align:right"><div class="mono" style="font-size:12px">${fmtS(p.seconds)} / ${fmtS(p.budget_seconds)}</div><div class="bar" style="width:120px"><i class="${frac > 1 ? 'over' : ''}" style="width:${Math.min(100, 100 * frac)}%"></i></div></div>` : ''}
  </a>`;
}
