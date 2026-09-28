// The explorer panel: layers, navigation, render resolution, models, measuring, keyframe overlay.
import { Explorer } from '/viewer/explorer.js';
import { $, $$, esc, fmtN, icon, toast } from './util.js';

const LAYERS = [['mesh', 'Mesh'], ['texture', 'Texture'], ['shaded', 'Shaded'], ['wireframe', 'Wireframe'], ['points', 'Points'], ['splats', 'Gaussian splats'],
  ['cameras', 'Flight path'], ['photos', 'Keyframe photos'], ['depth', 'Depth maps'], ['video', 'Source video'], ['grid', 'Grid'], ['focus', 'Focus subject']];
const BACKGROUNDS = { dark: 0x0b1118, grey: 0x2a3038, sky: 0x9fb8cf };

export function mountExplorer(el, { imageBase = null, height = null } = {}) {
  el.classList.add('xp');
  if (height) el.style.height = height;
  el.innerHTML = `<div class="stagebox"></div><div class="overlay" hidden></div>
    <div class="empty" id="xpEmpty">No 3D model yet — it appears here when the export stage finishes.</div>
    <div class="status"></div>
    <div class="tools">
      <div class="box"><h3>Layers</h3><div class="layers">${LAYERS.map(([k, t]) => `<div class="lay" data-l="${k}"><i></i>${t}</div>`).join('')}</div></div>
      <div class="box"><h3>Navigate</h3>
        <div class="seg" style="width:100%"><button data-nav="orbit" class="on" style="flex:1">Orbit</button><button data-nav="fly" style="flex:1">Fly</button><button data-nav="follow" style="flex:1">Follow flight</button></div>
        <div class="note" style="margin-top:7px;font-size:11.5px" id="xpKeys">drag orbit · right-drag pan · wheel zoom · double-click focus</div></div>
      <div class="box"><h3>Render</h3>
        <div class="seg" style="width:100%">${[0.5, 1, 1.5, 2].map((s) => `<button data-rs="${s}" class="${s === 1 ? 'on' : ''}" style="flex:1">${s}×</button>`).join('')}</div>
        <div class="seg" style="width:100%;margin-top:6px">${Object.keys(BACKGROUNDS).map((b) => `<button data-bg="${b}" class="${b === 'dark' ? 'on' : ''}" style="flex:1">${b}</button>`).join('')}</div>
        <div class="row" style="margin-top:8px;font-size:12px"><span class="muted">point size</span><input type="range" min="0.5" max="6" step="0.5" value="1.5" id="xpPs" style="flex:1"></div>
        <div class="row" style="margin-top:4px;font-size:12px"><span class="muted">keyframes</span><input type="range" min="0.5" max="8" step="0.25" value="2.5" id="xpIs" style="flex:1"></div></div>
      <div class="box"><h3>Models <span class="faint mono" id="xpCount"></span></h3><div class="modellist" id="xpModels"></div></div>
      <div class="box"><h3>Measure <button class="btn tiny ghost" id="xpClear">clear</button></h3>
        <button class="btn tiny" id="xpMeasure" style="width:100%;justify-content:center">${icon.ruler}Measure distance / height</button>
        <div class="measures" id="xpMeasures" style="margin-top:6px"></div></div>
      <div class="box" id="xpPhoto" hidden><h3>Keyframe overlay <button class="btn tiny ghost" id="xpPhotoOff">close</button></h3>
        <div class="row" style="font-size:12px"><span class="muted">photo</span><input type="range" min="0" max="1" step="0.05" value="0.5" id="xpOpacity" style="flex:1"><span class="muted">model</span></div></div>
    </div>
    <div class="pip" id="xpPip" hidden><video id="xpVideo" muted playsinline preload="metadata"></video><div class="pipbar"><span id="xpVt">—</span><button class="btn tiny ghost" id="xpVplay">play</button></div></div>
    <div class="zoom"><button class="btn small" id="xpIn" title="zoom in">+</button><button class="btn small" id="xpOut" title="zoom out">−</button>
      <button class="btn small" id="xpFit" title="fit the model">⤢</button><button class="btn small" id="xpAll" title="every model">◎</button></div>
    <div class="topright"><button class="btn small" id="xpShot">${icon.cam}Screenshot</button><button class="btn small" id="xpFull">${icon.full}Fullscreen</button></div>
    <div class="hud" id="xpHud"></div>`;
  const x = new Explorer($('.stagebox', el), { background: BACKGROUNDS.dark });
  const ui = { x, el, specs: [] };

  x.addEventListener('status', (e) => { $('.status', el).textContent = e.detail; });
  x.addEventListener('tick', () => {
    const s = x.stats();
    $('#xpHud', el).textContent = `${x.idle ? 'idle' : `${s.fps} fps`} · ${fmtN(s.triangles)} tris · ${fmtN(s.points)} pts · ${fmtN(s.splats)} splats · ${s.pixelRatio.toFixed(2)} px`;
  });
  x.addEventListener('models', () => drawModels());
  x.addEventListener('measure', () => {
    $('#xpMeasures', el).innerHTML = x.measures.map((m, i) => `<div>#${i + 1} ${m.d.toFixed(2)} ${esc(m.units)} · Δh ${m.dz >= 0 ? '+' : ''}${m.dz.toFixed(2)}</div>`).join('');
  });
  x.addEventListener('followend', () => setNavButtons('fly'));
  x.addEventListener('nav', () => setNavButtons(x.follow ? 'follow' : x.nav));

  const video = { on: false, spec: null };
  const drawLayers = () => {
    $$('.lay', el).forEach((d) => {
      const k = d.dataset.l;
      const avail = k === 'video' ? ui.specs.some((m) => m.video) : !ui.specs.length || x.layerAvailable(k);
      d.classList.toggle('on', k === 'video' ? video.on : !!x.layers[k]);
      d.classList.toggle('off-avail', !avail);
    });
  };
  $$('.lay', el).forEach((d) => d.addEventListener('click', async () => {
    if (d.dataset.l === 'video') { setVideo(!video.on); drawLayers(); return; }
    await x.setLayer(d.dataset.l, !x.layers[d.dataset.l]); drawLayers();
  }));
  // Source video, picture in picture: follows the camera during "follow flight" and seeks to a clicked keyframe.
  const vid = $('#xpVideo', el);
  const frameTime = (spec, name) => { const k = /(\d+)(?!.*\d)/.exec(name || ''); return k && spec?.fps ? +k[1] / spec.fps : null; };
  function setVideo(on, spec = null) {
    video.on = on;
    $('#xpPip', el).hidden = !on;
    if (!on) { vid.pause(); return; }
    const s = spec || x.models.find((m) => m.visible)?.spec || ui.specs.find((m) => m.video);
    if (s?.video && video.spec !== s) { video.spec = s; vid.src = s.video; }
  }
  vid.addEventListener('timeupdate', () => { $('#xpVt', el).textContent = `${vid.currentTime.toFixed(1)} s / ${(vid.duration || 0).toFixed(0)} s`; });
  $('#xpVplay', el).addEventListener('click', () => { vid.paused ? vid.play() : vid.pause(); $('#xpVplay', el).textContent = vid.paused ? 'play' : 'pause'; });
  let lastSeek = 0;
  x.addEventListener('follow', (e) => {
    if (!video.on) return;
    const { model, index, images } = e.detail;
    if (model?.spec && video.spec !== model.spec) setVideo(true, model.spec);
    const i = Math.floor(index), a = frameTime(model?.spec, images[i]), b = frameTime(model?.spec, images[Math.min(images.length - 1, i + 1)]);
    const now = performance.now();
    if (a != null && now - lastSeek > 120) { vid.currentTime = a + (b != null ? (b - a) * (index - i) : 0); lastSeek = now; }
  });
  ui.seekVideo = (name) => { if (!video.on) return; const s = video.spec; const t = frameTime(s, name); if (t != null) vid.currentTime = t; };

  function setNavButtons(mode) {
    $$('[data-nav]', el).forEach((b) => b.classList.toggle('on', b.dataset.nav === mode));
    $('#xpKeys', el).innerHTML = mode === 'fly' ? '<kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move · <kbd>Q</kbd><kbd>E</kbd> down/up · drag look · wheel speed · <kbd>Shift</kbd> ×4'
      : mode === 'follow' ? 'replaying the drone’s own path and view' : 'drag orbit · right-drag pan · wheel zoom · double-click focus';
  }
  $$('[data-nav]', el).forEach((b) => b.addEventListener('click', () => {
    if (b.dataset.nav === 'follow') { x.followFlight(true); x.renderer.domElement.focus(); }
    else { x.setNavigation(b.dataset.nav); x.renderer.domElement.focus(); }
    setNavButtons(b.dataset.nav);
  }));
  $$('[data-rs]', el).forEach((b) => b.addEventListener('click', () => { x.setRenderScale(+b.dataset.rs); $$('[data-rs]', el).forEach((c) => c.classList.toggle('on', c === b)); }));
  $$('[data-bg]', el).forEach((b) => b.addEventListener('click', () => { x.setBackground(BACKGROUNDS[b.dataset.bg]); $$('[data-bg]', el).forEach((c) => c.classList.toggle('on', c === b)); }));
  $('#xpPs', el).addEventListener('input', (e) => x.setPointSize(+e.target.value));
  $('#xpIs', el).addEventListener('input', (e) => x.setImageScale(+e.target.value));
  $('#xpMeasure', el).addEventListener('click', (e) => { x.setMeasuring(!x.measuring); e.currentTarget.classList.toggle('primary', x.measuring); });
  $('#xpClear', el).addEventListener('click', () => x.clearMeasures());
  $('#xpIn', el).addEventListener('click', () => x.zoom(0.7));
  $('#xpOut', el).addEventListener('click', () => x.zoom(1.45));
  $('#xpFit', el).addEventListener('click', () => { x.follow = null; x.setNavigation('orbit'); setNavButtons('orbit'); x.frame(x.models.findIndex((m) => m.visible)); });
  $('#xpAll', el).addEventListener('click', () => { x.showAll(); setNavButtons('orbit'); });
  $('#xpShot', el).addEventListener('click', async () => {
    const blob = await x.screenshot();
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `drone3d_${Date.now()}.png`; a.click();
    toast('Screenshot saved', 'ok');
  });
  $('#xpFull', el).addEventListener('click', () => { el.classList.toggle('full'); x.resize(); });
  addEventListener('keydown', (e) => { if (e.key === 'Escape' && el.classList.contains('full')) { el.classList.remove('full'); x.resize(); } });
  const overlay = $('.overlay', el);
  $('#xpOpacity', el).addEventListener('input', (e) => { overlay.style.opacity = 1 - +e.target.value; });
  $('#xpPhotoOff', el).addEventListener('click', () => { overlay.hidden = true; $('#xpPhoto', el).hidden = true; x.setNavigation('orbit'); setNavButtons('orbit'); });

  function drawModels() {
    $('#xpCount', el).textContent = x.models.length ? `${x.models.filter((m) => m.visible).length}/${x.models.length}` : '';
    $('#xpModels', el).innerHTML = x.models.map((m, i) => `<label><input type="checkbox" data-mi="${i}" ${m.visible ? 'checked' : ''}>${esc(m.spec.name)}<span class="go" data-solo="${i}">solo</span></label>`).join('') || '<span class="note">—</span>';
    $$('[data-mi]', el).forEach((c) => c.addEventListener('change', () => x.showModel(+c.dataset.mi, c.checked)));
    $$('[data-solo]', el).forEach((s) => s.addEventListener('click', (e) => { e.preventDefault(); x.solo(+s.dataset.solo); }));
  }

  ui.load = async (models) => {
    ui.specs = models;
    $('#xpEmpty', el).hidden = models.length > 0;
    drawLayers();
    if (models.length) await x.setScene(models);
    drawLayers(); drawModels();
  };
  ui.add = async (model) => {
    ui.specs.push(model);
    $('#xpEmpty', el).hidden = true;
    await x.addModel(model, { visible: true, frame: x.models.length === 0 });
    if (x.models.length === 1) x.frame(0);
    drawLayers(); drawModels();
  };
  // Put the camera where the drone was for this keyframe and lay its photo over the render.
  ui.lookThrough = (imageName, fullUrl) => {
    for (const [mi, m] of x.models.entries()) {
      const k = (m.spec.images || []).indexOf(imageName);
      if (k < 0) continue;
      if (!m.visible) x.solo(mi);
      const r = x.lookThrough(mi, k);
      if (!r) return false;
      setNavButtons('fly');
      overlay.style.backgroundImage = `url("${fullUrl}")`; overlay.style.opacity = 0.5; overlay.hidden = false;
      if (video.on) { if (video.spec !== m.spec) setVideo(true, m.spec); ui.seekVideo(imageName); }
      $('#xpOpacity', el).value = 0.5; $('#xpPhoto', el).hidden = false;
      return true;
    }
    return false;
  };
  ui.dispose = () => { x.dispose(); if (window.__explorer === ui) delete window.__explorer; };
  window.__explorer = ui; // for scripted demos and tests
  ui.imageBase = imageBase;
  setNavButtons('orbit'); drawLayers();
  return ui;
}
