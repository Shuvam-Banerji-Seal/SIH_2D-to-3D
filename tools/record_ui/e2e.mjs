// End-to-end check of the console in a GPU-accelerated headless Chrome: every page, every explorer layer and control.
//
//   bun tools/record_ui/e2e.mjs --url http://127.0.0.1:8080 --run RUN_NAME [--out .playwright-mcp/e2e.json]
//
// Fails (exit 1) on console errors, page exceptions, failed requests (>= 400), a layer that draws nothing, or a
// control that does not do what it says. A layer "draws" when the canvas changes by more than 0.5 % of its pixels.
import { chromium } from 'playwright-core';
import { writeFileSync } from 'node:fs';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const URL_ = arg('url', 'http://127.0.0.1:8080');
const RUN = arg('run');
const report = { pages: {}, layers: {}, controls: {}, errors: [], failures: [] };
const fail = (what) => { report.failures.push(what); console.log('FAIL', what); };

const browser = await chromium.launch({ executablePath: arg('chrome', '/usr/bin/google-chrome'), headless: true,
  args: ['--use-gl=angle', '--use-angle=gl-egl', '--ignore-gpu-blocklist', '--enable-gpu-rasterization'] });
const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
const page = await context.newPage();
page.on('console', (m) => { if (m.type() === 'error') report.errors.push(`console: ${m.text().slice(0, 200)}`); });
page.on('pageerror', (e) => report.errors.push(`exception: ${e.message.slice(0, 200)}`));
page.on('response', (r) => { if (r.status() >= 400 && !r.url().includes('favicon')) report.errors.push(`http ${r.status()}: ${r.url().slice(0, 160)}`); });
const sleep = (ms) => page.waitForTimeout(ms);

// ------------------------------------------------------------------ pages
for (const [hash, expect] of [['#/console', 'Mission console'], ['#/new', 'Build a 3D model'], ['#/live', 'Live reconstruction'], ['#/runs', 'Runs'], ['#/about', 'Method']]) {
  const t0 = Date.now();
  await page.goto(`${URL_}/${hash}`);
  try { await page.getByText(expect, { exact: false }).first().waitFor({ timeout: 15000 }); report.pages[hash] = { ok: true, ms: Date.now() - t0 }; }
  catch { report.pages[hash] = { ok: false }; fail(`page ${hash} did not show "${expect}"`); }
  await sleep(1500);
}
// the options form: every section opens and has fields with help
await page.goto(`${URL_}/#/new`); await page.getByText('Every option').first().waitFor();
const sections = await page.evaluate(() => { document.querySelectorAll('.opt-sec').forEach((d) => { d.open = true; });
  return [...document.querySelectorAll('.opt-sec')].map((d) => ({ k: d.dataset.k, fields: d.querySelectorAll('.field').length, help: [...d.querySelectorAll('.field .help')].every((h) => h.textContent.trim().length > 0) })); });
report.controls.options = sections;
if (!sections.length || sections.some((s) => !s.fields || !s.help)) fail('an options section is empty or a field has no help');

// ------------------------------------------------------------------ one run: explorer, layers, controls
if (RUN) {
  await page.goto(`${URL_}/#/run/${encodeURIComponent(RUN)}`);
  await page.waitForFunction(() => window.__explorer?.x.models.length && window.__explorer.x.models[0].meshObj, null, { timeout: 90000 });
  await page.evaluate(() => document.querySelector('#xp').scrollIntoView());
  await sleep(2500);
  // Pixels of the explorer's canvas right after a render (the drawing buffer is not preserved).
  const grab = () => page.evaluate(() => { const x = window.__explorer.x; x.wake(); x.renderer.render(x.scene, x.camera);
    const gl = x.renderer.getContext(), w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const step = 16, s = []; for (let i = 0; i < px.length; i += 4 * step) s.push(px[i] + px[i + 1] * 256 + px[i + 2] * 65536); return s; });
  const changed = (a, b) => a.filter((v, i) => Math.abs(v - b[i]) > 0).length / a.length;
  const avail = await page.evaluate(() => Object.fromEntries(['mesh', 'texture', 'shaded', 'wireframe', 'points', 'splats', 'cameras', 'photos', 'depth', 'grid'].map((k) => [k, window.__explorer.x.layerAvailable(k)])));
  const base = await grab();
  for (const layer of ['texture', 'shaded', 'wireframe', 'points', 'splats', 'cameras', 'photos', 'depth', 'grid']) {
    if (!avail[layer]) { report.layers[layer] = 'not in this run'; continue; }
    const before = await grab();
    await page.locator(`.lay[data-l="${layer}"]`).click();
    await sleep(layer === 'splats' ? 6000 : layer === 'points' ? 3000 : 1200);
    const after = await grab();
    const d = changed(before, after);
    report.layers[layer] = +d.toFixed(4);
    const least = ['cameras', 'photos', 'depth', 'grid'].includes(layer) ? 0.001 : 0.005; // thin overlays cover little of the frame
    if (d < least) fail(`layer ${layer}: the canvas did not change (${(100 * d).toFixed(2)} %)`);
    await page.locator(`.lay[data-l="${layer}"]`).click(); await sleep(600); // back to the default
  }
  const meshOff = await (async () => { await page.locator('.lay[data-l="mesh"]').click(); await sleep(800); const g = await grab(); await page.locator('.lay[data-l="mesh"]').click(); await sleep(800); return changed(base, g); })();
  report.layers.mesh = +meshOff.toFixed(4);
  if (meshOff < 0.01) fail('turning the mesh off changed nothing');
  // navigation and view controls
  const cam = () => page.evaluate(() => window.__explorer.x.camera.position.toArray().map((v) => +v.toFixed(3)));
  const c0 = await cam();
  await page.locator('#xpIn').click(); await sleep(400);
  const c1 = await cam(); report.controls.zoom_in = c1; if (JSON.stringify(c0) === JSON.stringify(c1)) fail('zoom in did not move the camera');
  await page.locator('#xpOut').click(); await sleep(300); await page.locator('#xpFit').click(); await sleep(400);
  const box = await page.locator('#xp').boundingBox();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.5); await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.55, { steps: 20 }); await page.mouse.up(); await sleep(500);
  const c2 = await cam(); if (JSON.stringify(c2) === JSON.stringify(c1)) fail('orbit drag did not move the camera');
  await page.locator('[data-nav="fly"]').click(); await page.locator('#xp canvas').focus();
  await page.keyboard.down('KeyW'); await sleep(700); await page.keyboard.up('KeyW');
  const c3 = await cam(); report.controls.fly = c3; if (JSON.stringify(c3) === JSON.stringify(c2)) fail('fly (W) did not move the camera');
  await page.locator('[data-nav="follow"]').click(); await sleep(2500);
  const c4 = await cam(); report.controls.follow = c4; if (JSON.stringify(c4) === JSON.stringify(c3)) fail('follow flight did not move the camera');
  await page.locator('[data-nav="orbit"]').click(); await page.locator('#xpAll').click(); await sleep(1500);
  report.controls.all_models = await page.evaluate(() => window.__explorer.x.models.filter((m) => m.visible).length);
  await page.locator('#xpFit').click();
  for (const rs of ['0.5', '2', '1']) { await page.locator(`[data-rs="${rs}"]`).click(); await sleep(300); }
  report.controls.pixel_ratio = await page.evaluate(() => window.__explorer.x.renderer.getPixelRatio());
  // measure: two clicks on the model
  await page.locator('#xpMeasure').click();
  for (const [fx, fy] of [[0.55, 0.55], [0.7, 0.6]]) { await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy); await sleep(300); }
  report.controls.measures = await page.evaluate(() => window.__explorer.x.measures.length);
  await page.locator('#xpMeasure').click();
  // screenshot button -> a PNG download
  const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }).catch(() => null), page.locator('#xpShot').click()]);
  report.controls.screenshot = dl ? dl.suggestedFilename() : null; if (!dl) fail('the screenshot button gave no download');
  // video picture-in-picture
  if (await page.locator('.lay[data-l="video"]:not(.off-avail)').count()) {
    await page.locator('.lay[data-l="video"]').click(); await sleep(2000);
    report.controls.video = await page.evaluate(() => { const v = document.querySelector('#xpVideo'); return { src: !!v.src, ready: v.readyState, duration: v.duration }; });
    if (!report.controls.video.duration) fail('the source video did not load');
    await page.locator('.lay[data-l="video"]').click();
  }
  // filmstrip and depth comparison
  const film = await page.locator('#film .frame').count(); report.controls.filmstrip = film; if (!film) fail('no keyframes in the filmstrip');
  if (film) {
    await page.locator('#film .frame').nth(Math.min(3, film - 1)).click(); await sleep(1500);
    report.controls.look_through = await page.evaluate(() => !document.querySelector('.overlay').hidden);
    if (!report.controls.look_through) fail('look-through-keyframe showed no photo');
    await page.locator('#xpPhotoOff').click();
  }
  if (await page.locator('#depthPanel:not([hidden])').count()) {
    await page.locator('#compare').scrollIntoViewIfNeeded(); await sleep(300);
    const cb = await page.locator('#compare').boundingBox();
    await page.mouse.move(cb.x + cb.width * 0.5, cb.y + cb.height / 2); await page.mouse.down(); await page.mouse.move(cb.x + cb.width * 0.8, cb.y + cb.height / 2, { steps: 8 }); await page.mouse.up();
    report.controls.depth_compare = await page.evaluate(() => document.querySelector('#cClip').style.clipPath);
    if (!report.controls.depth_compare) fail('dragging the depth comparison did nothing');
  } else fail('no depth comparison panel');
  report.controls.downloads = await page.locator('#downloads a[download]').count();
  report.controls.download_types = await page.evaluate(() => [...new Set([...document.querySelectorAll('#downloads a[download]')].map((a) => a.textContent.split('.').pop()))]);
  // every download link answers
  const links = await page.evaluate(() => [...document.querySelectorAll('#downloads a[download]')].map((a) => a.href));
  for (const href of links) { const r = await page.request.head(href).catch(() => null); if (!r || r.status() >= 400) fail(`download ${href.split('/').slice(-2).join('/')} -> ${r ? r.status() : 'error'}`); }
}

report.errors = [...new Set(report.errors)];
for (const e of report.errors) fail(e);
writeFileSync(arg('out', '.playwright-mcp/e2e.json'), JSON.stringify(report, null, 1));
console.log(JSON.stringify({ pages: report.pages, layers: report.layers, failures: report.failures.length }, null, 1));
await browser.close();
process.exit(report.failures.length ? 1 : 0);
