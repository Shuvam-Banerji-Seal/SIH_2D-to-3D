// Record a guided tour of the drone3d console to an MP4 (for the promo film and the paper's supplement).
//
//   bun tools/record_ui/tour.mjs --url http://127.0.0.1:8080 --run map_jal_mahal_jaipur_india_drone_cinematic --out promo/build/ui_tour.mp4
//
// Headless Chrome renders WebGL on the GPU (ANGLE on EGL; ANGLE on Vulkan drew 4K textures black); frames come from the DevTools screencast
// and are piped to ffmpeg, which re-times them to a constant 30 fps and encodes H.264 (NVENC if present).
import { chromium } from 'playwright-core';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const URL_ = arg('url', 'http://127.0.0.1:8080');
const RUN = arg('run', null);
const OUT = resolve(arg('out', 'ui_tour.mp4'));
const W = +arg('width', 1920), H = +arg('height', 1080), FPS = 30;
const FFMPEG = arg('ffmpeg', existsSync(resolve('.tools/ffmpeg/bin/ffmpeg')) ? resolve('.tools/ffmpeg/bin/ffmpeg') : 'ffmpeg');
mkdirSync(dirname(OUT), { recursive: true });

const browser = await chromium.launch({ executablePath: arg('chrome', '/usr/bin/google-chrome'), headless: true,
  args: ['--use-gl=angle', '--use-angle=gl-egl', '--ignore-gpu-blocklist', '--enable-gpu-rasterization', `--window-size=${W},${H}`] });
const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
const page = await context.newPage();
const cdp = await context.newCDPSession(page);

// ffmpeg reads JPEG frames with their wall-clock timestamps and emits constant-rate video.
const enc = spawn(FFMPEG, ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-c:v', 'mjpeg', '-use_wallclock_as_timestamps', '1', '-i', '-',
  '-vf', `fps=${FPS},scale=${W}:${H}:flags=lanczos,format=yuv420p`, '-c:v', arg('codec', 'h264_nvenc'), '-preset', 'p5', '-b:v', '16M', '-movflags', '+faststart', OUT],
{ stdio: ['pipe', 'inherit', 'inherit'] });
let frames = 0;
cdp.on('Page.screencastFrame', async ({ data, sessionId }) => {
  enc.stdin.write(Buffer.from(data, 'base64')); frames++;
  try { await cdp.send('Page.screencastFrameAck', { sessionId }); } catch { /* closing */ }
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function hold(ms) { const t = Date.now(); while (Date.now() - t < ms) await sleep(50); }
async function smoothScroll(y, ms = 1400) { await page.evaluate(async ({ y, ms }) => { const y0 = scrollY, t0 = performance.now();
  await new Promise((done) => { const step = (t) => { const k = Math.min(1, (t - t0) / ms), e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
    scrollTo(0, y0 + (y - y0) * e); k < 1 ? requestAnimationFrame(step) : done(); }; requestAnimationFrame(step); }); }, { y, ms }); }
async function to(sel, ms = 1200) { const y = await page.evaluate((s) => { const e = document.querySelector(s); return e ? e.getBoundingClientRect().top + scrollY - 70 : 0; }, sel); await smoothScroll(y, ms); }
async function drag(x0, y0, dx, dy, ms = 1800, button = 'left') { await page.mouse.move(x0, y0); await page.mouse.down({ button });
  const n = Math.round(ms / 16); for (let i = 1; i <= n; i++) { const k = i / n, e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2; await page.mouse.move(x0 + dx * e, y0 + dy * e); await sleep(16); }
  await page.mouse.up({ button }); }
async function click(sel) { const b = await page.locator(sel).first().boundingBox(); if (!b) return; await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 18 }); await sleep(150); await page.locator(sel).first().click(); }
async function go(hash, settle = 1800) { await page.evaluate((h) => { location.hash = h; }, hash); await hold(settle); }

await page.goto(`${URL_}/#/console`);
await page.waitForLoadState('networkidle').catch(() => {});
await hold(1200);
await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: W, maxHeight: H, everyNthFrame: 1 });

// 1. Console: engine, models, GPU
await hold(4500);
await to('#models', 1600); await hold(2500);
await to('#chUtil', 1400); await hold(3500);
await to('#caps', 1200); await hold(3000);
await smoothScroll(0, 1400); await hold(800);

// 2. Build: pick a video, resolution, modules, the options
await go('#/new', 2200);
await click('#videos .choice:nth-child(10)'); await hold(900);
await to('#res', 1200); await click('[data-r="ultra"]'); await hold(1400); await click('[data-r="high"]'); await hold(700);
await to('#mods', 1100); await click('[data-m="splat"]'); await hold(1600);
await to('#sections', 1300);
await page.locator('#oq').click(); await page.keyboard.type('trunc', { delay: 110 }); await hold(2600);
await page.locator('#oq').fill(''); await page.keyboard.press('Backspace'); await hold(500);
await smoothScroll(0, 1400); await hold(1200);

// 3. Live
await go('#/live', 2200);
await click('#mode [data-m="url"]'); await hold(1600); await click('#mode [data-m="replay"]'); await hold(1400);
await click('#src .choice:nth-child(10)'); await hold(1800);

// 4. A finished run: pipeline, results, the explorer
if (RUN) {
  await go(`#/run/${encodeURIComponent(RUN)}`, 3000);
  await page.waitForFunction(() => window.__explorer && window.__explorer.x.models.length && window.__explorer.x.models[0].meshObj, null, { timeout: 60000 }).catch(() => {});
  await hold(1500);
  await to('#xp', 1400); await hold(1200);
  const box = await page.locator('#xp').boundingBox();
  const cx = box.x + box.width * 0.62, cy = box.y + box.height * 0.5;
  await drag(cx, cy, -260, 40, 2600); await hold(600);
  await drag(cx, cy, 200, -60, 2200); await hold(600);
  await page.mouse.move(cx, cy); for (let i = 0; i < 10; i++) { await page.mouse.wheel(0, -120); await sleep(90); } await hold(900);
  await click('.lay[data-l="texture"]'); await hold(1600); await click('.lay[data-l="texture"]'); await hold(700);
  await click('.lay[data-l="shaded"]'); await hold(1700); await click('.lay[data-l="shaded"]'); await hold(500);
  await click('.lay[data-l="wireframe"]'); await hold(1600); await click('.lay[data-l="wireframe"]'); await hold(500);
  await click('.lay[data-l="points"]'); await click('.lay[data-l="mesh"]'); await hold(2400);
  await click('.lay[data-l="splats"]'); await click('.lay[data-l="points"]'); await hold(3500);
  await click('.lay[data-l="mesh"]'); await click('.lay[data-l="splats"]'); await hold(800);
  await click('[data-nav="follow"]'); await hold(9000);
  await click('[data-nav="orbit"]'); await hold(800);
  await to('#filmPanel', 1400); await hold(900);
  await click('[data-fm="depth"]'); await hold(2200); await click('[data-fm="split"]'); await hold(2200); await click('[data-fm="photo"]'); await hold(600);
  await click('#film .frame:nth-child(6)'); await hold(3500);
  await page.locator('#xpOpacity').fill('0.85'); await page.locator('#xpOpacity').dispatchEvent('input'); await hold(1500);
  await page.locator('#xpOpacity').fill('0.15'); await page.locator('#xpOpacity').dispatchEvent('input'); await hold(1500);
  await click('#xpPhotoOff'); await hold(900);
  await click('#xpFull'); await hold(1200);
  const fb = { x: W * 0.55, y: H * 0.5 };
  await drag(fb.x, fb.y, -300, 30, 2800); await hold(900);
  await page.keyboard.press('Escape'); await hold(900);
  await to('#downloads', 1400); await hold(2500);
}

// 5. Runs and method
await go('#/runs', 2600);
await go('#/about', 3200);
await to('.pipe', 800); await hold(2500);
await go('#/console', 3500);

await cdp.send('Page.stopScreencast');
enc.stdin.end();
await new Promise((r) => enc.on('close', r));
await browser.close();
console.log(`${OUT}: ${frames} frames captured`);
