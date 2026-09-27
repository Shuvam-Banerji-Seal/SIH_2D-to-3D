// Screenshot a console page with GPU WebGL, optionally after running actions in the page.
//
//   bun tools/record_ui/shot.mjs --url 'http://127.0.0.1:8080/#/run/NAME' --out paper/figures/ui_run.png \
//       --wait 'window.__explorer?.x.models[0]?.meshObj' --js 'await window.__explorer.x.setLayer("splats", true)' --settle 4000
import { chromium } from 'playwright-core';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const out = resolve(arg('out', 'shot.png'));
const W = +arg('width', 1920), H = +arg('height', 1080);
mkdirSync(dirname(out), { recursive: true });
const browser = await chromium.launch({ executablePath: arg('chrome', '/usr/bin/google-chrome'), headless: true,
  args: (arg('gl', 'egl') === 'egl' ? ['--use-gl=angle', '--use-angle=gl-egl'] : ['--use-angle=vulkan', '--enable-features=Vulkan']).concat(['--ignore-gpu-blocklist', '--enable-gpu-rasterization']) });
const page = await (await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: +arg('scale', 1) })).newPage();
page.on('pageerror', (e) => console.error('page error:', e.message));
page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) console.error('console', m.type(), m.text().slice(0, 300)); });
await page.goto(arg('url', 'http://127.0.0.1:8080/'));
if (arg('wait')) await page.waitForFunction(arg('wait'), null, { timeout: +arg('timeout', 90000) });
if (arg('js')) await page.evaluate(`(async () => { ${arg('js')} })()`);
await page.waitForTimeout(+arg('settle', 2500));
if (arg('select')) await page.locator(arg('select')).first().screenshot({ path: out });
else await page.screenshot({ path: out, fullPage: arg('full') === '1' });
const info = await page.evaluate(() => window.__explorer ? window.__explorer.x.stats() : null);
if (info) console.log(JSON.stringify(info));
await browser.close();
console.log(out);
