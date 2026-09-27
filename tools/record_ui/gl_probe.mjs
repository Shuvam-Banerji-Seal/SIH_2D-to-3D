import { chromium } from 'playwright-core';
const flags = process.argv.slice(2);
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: flags });
const page = await browser.newPage();
await page.goto('about:blank');
const info = await page.evaluate(() => { const c = document.createElement('canvas'); const gl = c.getContext('webgl2'); if (!gl) return 'no webgl2';
  const e = gl.getExtension('WEBGL_debug_renderer_info'); return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); });
console.log(flags.join(' ') || '(no flags)', '->', info);
await browser.close();
