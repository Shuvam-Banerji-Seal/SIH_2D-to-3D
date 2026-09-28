// One run (or live session): the pipeline as it happens, then the model -- mesh, points, splats, depth.
import { VIEWABLE } from './model.js';
import { $, $$, api, esc, every, fmtN, fmtS, icon, onLeave, pct, post, prettyVideo, toast } from './util.js';
import { mountExplorer } from './explore.js';

const EXTRA = ['splat', 'depth', 'mesh', 'render']; // the 3DGS stages: after the deliverable, outside its budget
const stageSum = (r, pick) => Object.entries(r.stages || {}).filter(([st]) => pick(st)).reduce((a, [, x]) => a + (x.seconds || 0), 0);
const deliverableS = (r) => stageSum(r, (st) => !EXTRA.includes(st));
const extraS = (r) => stageSum(r, (st) => EXTRA.includes(st));

const ACTIVE = ['running', 'queued', 'stopping'];

export async function viewRun(main, name) {
  const enc = encodeURIComponent(name);
  main.innerHTML = `
    <div class="page-head rise" id="head"></div>
    <div class="grid g-12">
      <section class="panel s12 rise" style="--i:1" id="pipe"></section>
      <section class="panel s12 rise" style="--i:2" id="results" hidden></section>
      <section class="panel s12 rise" style="--i:3" id="catalog" hidden></section>
      <section class="s12 rise" style="--i:3"><div id="xp"></div></section>
      <section class="panel s12 rise" style="--i:4" id="filmPanel" hidden>
        <h2>Keyframes <span class="tag" id="filmTag"></span><span class="grow"></span>
          <div class="seg"><button data-fm="photo" class="on">photo</button><button data-fm="depth">fused depth</button><button data-fm="split">split</button></div></h2>
        <div class="film" id="film"></div>
        <div class="note">Click a keyframe to look through it: the camera moves to where the drone was, with its field of view, and the photo lies over the model.</div>
      </section>
      <section class="panel s12 rise" style="--i:5" id="depthPanel" hidden>
        <h2>Depth <span class="tag">fused per keyframe: flow triangulation, Depth Anything fill, sky left empty — drag across to compare</span><span class="grow"></span>
          <button class="btn tiny" id="dPrev">‹</button><span class="mono" id="dIdx" style="font-size:12px"></span><button class="btn tiny" id="dNext">›</button></h2>
        <div class="compare" id="compare"><img id="cPhoto" alt="keyframe"><div class="clip" id="cClip"><img id="cDepth" alt="depth"></div><div class="handle" id="cHandle"></div></div>
      </section>
      <section class="panel s12 rise" style="--i:5" id="downloads" hidden></section>
      <section class="panel s12 rise" style="--i:6"><details class="fold" id="logFold"><summary><h2 style="margin:0">Log</h2></summary><div class="log" id="log" style="margin-top:12px"></div></details></section>
    </div>`;
  const xp = mountExplorer($('#xp'), { imageBase: `/runs/${enc}` });
  onLeave(() => xp.dispose());
  const seen = { scene: false, liveModels: new Set(), frames: 0, films: null, mode: 'photo' };

  $$('[data-fm]').forEach((b) => b.addEventListener('click', () => { seen.mode = b.dataset.fm; $$('[data-fm]').forEach((c) => c.classList.toggle('on', c === b)); drawFilm(); }));

  async function tick() {
    let r;
    try { r = await api(`/api/runs/${enc}`); } catch (e) { $('#head').innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
    const live = r.job?.kind === 'live' || !!r.live;
    head(r, live);
    if (live) await liveTick(r); else pipeline(r);
    const log = $('#log'); const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
    log.textContent = (r.log_tail || []).join('\n'); if (atEnd) log.scrollTop = log.scrollHeight;
    if (!live && r.summary.viewer && !seen.scene) { seen.scene = true; await loadScene(); results(r); downloads(r); catalog(); }
    if (!live && !ACTIVE.includes(r.status) && seen.scene) { results(r); }
    if (seen.frames === 0 || live || ACTIVE.includes(r.status) || !seen.depthDone) await frames(); // keyframes, then depth, appear as stages finish
    if (!ACTIVE.includes(r.status)) seen.depthDone = true;
  }

  function head(r, live) {
    const s = r.summary;
    const via = r.job?.engine ? 'warm engine' : r.job?.pid ? 'subprocess' : 'command line';
    $('#head').innerHTML = `<h1>${esc(r.name)}<small>${esc(prettyVideo((s.video || '').split('/').pop()) || '')}${s.resolution?.[0] ? ` · ${s.resolution[0]}×${s.resolution[1]}` : ''}${s.video_seconds ? ` · ${fmtS(s.video_seconds)}` : ''} · ${via}</small></h1>
      <div class="row" style="margin-left:auto">
        <span class="st ${esc(r.status)}">${esc(r.status)}</span>${live ? '<span class="pill" style="color:var(--red);border-color:#6b2a2a">live</span>' : ''}
        ${ACTIVE.includes(r.status) ? `<button class="btn small danger" id="stop">${icon.stop}${live ? 'Stop recording' : 'Stop'}</button>` : ''}
        ${s.report ? `<a class="btn small" href="/runs/${enc}/report.html" target="_blank">${icon.ext}Report</a>` : ''}
        ${s.viewer ? `<a class="btn small" href="/runs/${enc}/export/index.html" target="_blank">${icon.ext}Standalone viewer</a>` : ''}
      </div>`;
    const stop = $('#stop');
    if (stop) stop.onclick = async () => { try { await post(live ? `/api/live/${enc}/stop` : `/api/runs/${enc}/stop`); toast(live ? 'Recording stops; the open segment is still processed' : 'Stopping after the current stage'); } catch (e) { toast(e.message, 'bad'); } };
  }

  function pipeline(r) {
    const s = r.summary, p = s.processing;
    const planned = r.config?.stages?.length ? r.config.stages : Object.keys(r.stages);
    const now = Date.now() / 1000;
    const elapsed = r.job?.started ? (r.job.finished || now) - r.job.started : null;
    const budget = p?.budget_seconds ?? (s.video_seconds ? 1.5 * s.video_seconds : null);
    // the budget is for the deliverable (mesh + cloud); splat training comes after it and is shown on its own
    const deliverable = planned.filter((st) => !EXTRA.includes(st)).reduce((a, st) => a + (r.stages[st]?.seconds || 0), 0);
    const extra = planned.filter((st) => EXTRA.includes(st)).reduce((a, st) => a + (r.stages[st]?.seconds || 0), 0);
    const used = ACTIVE.includes(r.status) && !deliverable ? elapsed : deliverable || (p?.seconds ?? elapsed);
    const frac = budget && used != null ? used / budget : null;
    const total = Math.max(1, planned.reduce((a, st) => a + (r.stages[st]?.seconds || 0), 0));
    $('#pipe').innerHTML = `<h2>Pipeline <span class="tag">every stage writes its own files; any subset can be re-run</span></h2>
      <div class="timeline">${planned.map((st) => {
        const x = r.stages[st] || { status: 'pending' };
        return `<div class="stage ${esc(x.status)}" title="${esc(x.message || '')}"><div class="n">${esc(st)}</div><div class="s">${x.status === 'running' ? 'running…' : x.seconds != null ? fmtS(x.seconds) : esc(x.status)}</div>
          ${x.seconds ? `<div class="fill" style="width:${Math.min(100, (100 * x.seconds) / total)}%"></div>` : ''}</div>`;
      }).join('')}</div>
      <div class="budget-meter" style="margin-top:14px"><div class="used ${frac > 1 ? 'over' : ''}" style="width:${frac != null ? Math.min(100, 100 * frac) : 0}%"></div>
        <div class="lbl"><span>mesh + cloud vs budget · 15 min per 10-min video</span><span><b>${fmtS(used)}</b> of ${fmtS(budget)}${frac != null ? ` · ${Math.round(100 * frac)}%` : ''}${extra ? ` <span class="muted">· + ${fmtS(extra)} Gaussian splats</span>` : ''}</span></div></div>
      ${planned.filter((st) => r.stages[st]?.status === 'failed').map((st) => `<div class="err"><b>${esc(st)} failed:</b> ${esc(r.stages[st].message || 'see the log below')}${hint(r.stages[st].message) ? `\n${esc(hint(r.stages[st].message))}` : ''}</div>`).join('')}`;
    if (!ACTIVE.includes(r.status) && !s.viewer) { // finished without a model: say so, not "appears when export finishes"
      const failed = planned.find((st) => r.stages[st]?.status === 'failed');
      const empty = $('#xpEmpty'); if (empty) empty.textContent = failed ? `This run made no 3D model: the ${failed} stage failed (see above).` : `This run made no 3D model (${r.status}).`;
    }
  }

  // What a user can do about the failures a single pass can meet.
  function hint(msg = '') {
    if (/recoverable 3D structure|no pass/i.test(msg)) return 'The camera did not move enough relative to the scene — a pan or hover on the spot, a very distant or flat scene, or a clip too short. A single pass needs parallax: fly past or around the subject.';
    if (/no SfM model|registered/i.test(msg)) return 'Too few keyframes could be placed together; footage with more overlap between views (slower flight, less motion blur) helps.';
    if (/out of memory|CUDA/i.test(msg)) return 'The GPU ran out of memory or faulted; the engine restarts itself — run it again, or choose the draft resolution.';
    if (/ffprobe|decode|stream/i.test(msg)) return 'The video could not be decoded; re-encode it to H.264 or HEVC.';
    return '';
  }

  function results(r) {
    const s = r.summary, p = s.processing;
    const geo = s.georef?.length ? s.georef.map((g) => `${esc(g.model)} ${esc(g.mode)}${g.loo_rmse_horizontal_m != null ? `, ${g.loo_rmse_horizontal_m.toFixed(2)} m held-out` : ''}`).join('; ') : 'no flight log — levelled, SfM units';
    const el = $('#results'); el.hidden = false;
    el.innerHTML = `<h2>Result</h2><div class="tiles">
      <div class="tile"><div class="k">Keyframes registered</div><div class="v">${s.registered ?? '—'}<small> / ${s.keyframes ?? '—'}</small></div></div>
      <div class="tile"><div class="k">Models</div><div class="v">${s.models ?? '—'}<small> passes</small></div></div>
      <div class="tile"><div class="k">Mesh</div><div class="v">${fmtN(s.triangles)}<small> triangles</small></div></div>
      <div class="tile accent"><div class="k">Completeness</div><div class="v">${pct(s.completeness)}<small> of what the camera saw</small></div></div>
      <div class="tile ${deliverableS(r) <= (p?.budget_seconds ?? Infinity) ? 'accent' : ''}"><div class="k">Mesh + cloud time</div><div class="v">${fmtS(deliverableS(r))}<small> / ${fmtS(p?.budget_seconds)} budget</small></div>${extraS(r) ? `<div class="note">+ ${fmtS(extraS(r))} Gaussian splats</div>` : ''}</div>
      <div class="tile"><div class="k">Georeferencing</div><div class="v" style="font:500 13px var(--sans);margin-top:10px">${geo}</div></div></div>`;
  }

  // The run's models, main pieces first: a model is the part of the scene one set of camera passes saw.
  let showFragments = false, cat = null;
  async function catalog() {
    try { cat = await api(`/api/runs/${enc}/models`); } catch { return; }
    if (!cat.models.length) return;
    const el = $('#catalog'); el.hidden = false;
    const main = cat.models.filter((m) => m.role === 'main'), frag = cat.models.filter((m) => m.role !== 'main');
    const merged = cat.merged ? `${cat.models_before_merge} camera passes merged into ${cat.models_after_merge} models where they see the same scene` : 'each camera pass is its own model';
    const cardOf = (m) => `<div class="mcard ${m.role}" data-model="${esc(m.model)}">
        <div class="mthumb" style="${m.thumb ? `background-image:url('${encodeURI(m.thumb)}')` : ''}"><span class="pill">${m.role === 'main' ? 'main' : 'fragment'}</span></div>
        <div style="padding:9px 11px"><b>Model ${esc(m.model)}</b> <span class="muted" style="font-size:12px">${m.shots.length > 1 ? `${m.shots.length} shots merged` : '1 shot'} · ${Math.round(100 * m.share)} % of keyframes</span>
          <div class="rstats" style="margin-top:6px">
            <div><span class="k">keyframes</span><b>${m.images ?? '—'}</b></div><div><span class="k">complete</span><b>${pct(m.completeness)}</b></div>
            <div><span class="k">triangles</span><b>${fmtN(m.triangles)}</b></div><div><span class="k">splat PSNR</span><b>${m.splat_psnr != null ? `${m.splat_psnr.toFixed(1)} dB` : '—'}</b></div></div>
          <div class="row" style="gap:6px;margin-top:8px"><button class="btn tiny" data-solo="${esc(m.model)}">${icon.eye}show alone</button>
            ${m.glb ? `<a class="btn tiny" href="#/model/${enc}/${m.glb.split('/').map(encodeURIComponent).join('/')}">GLB</a>` : ''}</div></div></div>`;
    el.innerHTML = `<h2>Models <span class="tag">${esc(merged)}</span><span class="grow"></span>
        ${frag.length ? `<button class="linkbtn" id="fragBtn">${showFragments ? 'hide' : 'show'} ${frag.length} fragment${frag.length > 1 ? 's' : ''}</button>` : ''}</h2>
      <div class="note" style="margin:-4px 0 10px">A model is the part of the scene one set of camera passes saw. Main models hold the bulk of the video; fragments are short shots nothing else overlaps.</div>
      <div class="mgrid">${main.map(cardOf).join('')}${showFragments ? frag.map(cardOf).join('') : ''}</div>`;
    const fb = $('#fragBtn'); if (fb) fb.onclick = () => { showFragments = !showFragments; catalog(); };
    $$('#catalog [data-solo]').forEach((b) => b.addEventListener('click', () => {
      const i = xp.x.models.findIndex((m) => m.spec.dir === `model_${b.dataset.solo}` || m.spec.name === `model ${b.dataset.solo}`);
      if (i >= 0) { xp.x.solo(i); document.querySelector('#xp').scrollIntoView({ behavior: 'smooth' }); }
    }));
  }

  function downloads(r) {
    const s = r.summary;
    if (!s.files?.length) return;
    const el = $('#downloads'); el.hidden = false;
    el.innerHTML = `<h2>Deliverables <span class="tag">OBJ · PLY · LAS · GeoTIFF · GLB · FBX · STL · Blender · web splats</span></h2>` + s.files.map((m) => `
      <div style="padding:10px 0;border-top:1px solid var(--line)"><div class="between"><b>model ${esc(m.model)}</b><span class="note mono">${fmtN(m.triangles)} triangles · ${fmtN(m.points)} points${m.epsg ? ` · EPSG:${m.epsg}` : ''}</span></div>
        <div class="row" style="margin-top:8px;gap:6px">${m.files.map((f) => `<span class="row" style="gap:0">${VIEWABLE.includes(f.split('.').pop().toLowerCase()) ? `<a class="btn tiny" href="#/model/${enc}/${f.split('/').map(encodeURIComponent).join('/')}" title="open this file in the 3D viewer" style="border-top-right-radius:0;border-bottom-right-radius:0">${icon.eye}view</a>` : ''}<a class="btn tiny" href="/runs/${enc}/export/${esc(f)}" download ${VIEWABLE.includes(f.split('.').pop().toLowerCase()) ? 'style="border-top-left-radius:0;border-bottom-left-radius:0;margin-left:-1px"' : ''}>${icon.down}${esc(f.split('/').pop())}</a></span>`).join('')}</div></div>`).join('');
  }

  async function loadScene() {
    try { const scn = await api(`/api/runs/${enc}/scene`); await xp.load(scn.models); } catch (e) { toast(`3D scene: ${e.message}`, 'bad'); }
  }

  async function frames() {
    try {
      const f = await api(`/api/runs/${enc}/frames?limit=600`);
      const depthN = f.frames.filter((x) => x.depth).length;
      if (f.frames.length === seen.frames && depthN === seen.depthN && seen.films) return;
      seen.depthN = depthN;
      seen.films = f.frames; seen.frames = f.frames.length;
      drawFilm();
    } catch { /* no frames yet */ }
  }

  function drawCompare() {
    const withDepth = (seen.films || []).filter((f) => f.depth);
    if (!withDepth.length) return;
    $('#depthPanel').hidden = false;
    seen.ci = Math.min(seen.ci ?? Math.floor(withDepth.length / 2), withDepth.length - 1);
    const f = withDepth[seen.ci];
    $('#cPhoto').src = f.full; $('#cDepth').src = f.depth;
    $('#dIdx').textContent = `${seen.ci + 1} / ${withDepth.length} · ${f.pass}`;
  }
  function bindCompare() {
    const box = $('#compare'), set = (x) => { const r = box.getBoundingClientRect(); const k = Math.max(0, Math.min(1, (x - r.left) / r.width));
      $('#cClip').style.clipPath = `inset(0 ${100 - 100 * k}% 0 0)`; $('#cHandle').style.left = `${100 * k}%`; };
    let drag = false;
    box.addEventListener('pointerdown', (e) => { drag = true; box.setPointerCapture(e.pointerId); set(e.clientX); });
    box.addEventListener('pointermove', (e) => { if (drag) set(e.clientX); });
    box.addEventListener('pointerup', () => { drag = false; });
    $('#dPrev').onclick = () => { seen.ci = Math.max(0, (seen.ci || 0) - 1); drawCompare(); };
    $('#dNext').onclick = () => { seen.ci = (seen.ci || 0) + 1; drawCompare(); };
  }
  bindCompare();

  function drawFilm() {
    const list = seen.films || [];
    if (!list.length) return;
    drawCompare();
    $('#filmPanel').hidden = false;
    const withDepth = list.filter((f) => f.depth).length;
    $('#filmTag').textContent = `${list.length} keyframes · ${withDepth} with fused depth`;
    const shown = seen.mode === 'photo' ? list : list.filter((f) => f.depth);
    $('#film').innerHTML = shown.map((f, i) => {
      const name = f.full.split('/dataset/images/')[1] || '';
      const img = seen.mode === 'depth' ? f.depth : f.image;
      return `<div class="frame" data-n="${esc(name)}" data-full="${esc(f.full)}" title="${esc(name)}"><img loading="lazy" src="${esc(img)}" alt="">
        ${seen.mode === 'split' ? `<img class="d" loading="lazy" src="${esc(f.depth)}" style="clip-path:inset(0 0 0 50%)" alt="">` : ''}
        <span class="lbl">${esc(f.pass)} · ${i + 1}</span></div>`;
    }).join('');
    $$('#film .frame').forEach((d) => d.addEventListener('click', () => {
      if (!xp.lookThrough(d.dataset.n, d.dataset.full)) toast('This keyframe is not in a reconstructed model', 'bad');
      else $('#xp').scrollIntoView({ behavior: 'smooth', block: 'center' });
    }));
  }

  // ------------------------------------------------------------- live
  async function liveTick() {
    let L;
    try { L = await api(`/api/live/${enc}`); } catch { return; }
    const done = L.segments.filter((s) => s.status === 'done');
    const lat = done.map((s) => s.latency_s).filter((x) => x != null);
    $('#pipe').innerHTML = `<h2>Live session <span class="tag">${esc(L.source.split('/').pop())}${L.simulate ? ' · replayed at its frame rate' : ''} · ${L.segment_s} s segments</span><span class="grow"></span>
        <span class="st ${L.recording ? 'recording' : 'done'}">${L.recording ? 'recording' : 'recorder stopped'}</span></h2>
      <div class="tiles" style="margin-bottom:14px">
        <div class="tile"><div class="k">Recorded</div><div class="v">${fmtS(L.recorded_s)}</div></div>
        <div class="tile"><div class="k">Segments</div><div class="v">${done.length}<small> / ${L.segments.length} modelled</small></div></div>
        <div class="tile accent"><div class="k">Real-time factor</div><div class="v">${L.mean_realtime_factor != null ? `${L.mean_realtime_factor.toFixed(2)}×` : '—'}<small> processing ÷ footage</small></div></div>
        <div class="tile"><div class="k">Model latency</div><div class="v">${lat.length ? fmtS(lat.reduce((a, b) => a + b, 0) / lat.length) : '—'}<small> after a segment closes</small></div></div>
      </div>
      <div class="segs">${L.segments.map((s) => `<div class="segc ${esc(s.status)}"><div class="n">seg ${s.index}</div><div class="m">${fmtS(s.duration_s)} · <span class="st ${esc(s.status)}">${esc(s.status)}</span></div>
        <div class="m">${s.processing_s != null ? `${fmtS(s.processing_s)} · ${s.realtime_factor?.toFixed(2)}×` : 'waiting'}</div></div>`).join('')}
        ${L.recording ? '<div class="segc recording"><div class="n">recording</div><div class="m">next segment</div></div>' : ''}</div>
      ${L.error ? `<div class="err">${esc(L.error)}</div>` : ''}`;
    const fresh = done.filter((s) => !seen.liveModels.has(s.index));
    if (fresh.length) {
      const scn = await api(`/api/runs/${enc}/scene`).catch(() => ({ models: [] }));
      for (const s of fresh) {
        seen.liveModels.add(s.index);
        for (const m of scn.models.filter((x) => x.name.startsWith(`segment ${s.index} ·`))) await xp.add(m);
        toast(`Segment ${s.index} modelled ${fmtS(s.latency_s)} after it closed`, 'ok');
      }
    }
  }

  await tick();
  every(2000, tick);
}
