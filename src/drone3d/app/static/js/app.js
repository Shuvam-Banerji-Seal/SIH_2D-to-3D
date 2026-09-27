// drone3d console: router, status bar and shared state.
import { $, $$, api, gb, leave } from './util.js';
import { viewConsole } from './console.js';
import { viewNew } from './build.js';
import { viewLive } from './live.js';
import { viewRuns } from './runs.js';
import { viewRun } from './run.js';
import { viewAbout } from './about.js';

const routes = { console: viewConsole, new: viewNew, live: viewLive, runs: viewRuns, run: viewRun, about: viewAbout };

function route() {
  leave();
  const [, page = 'console', ...rest] = location.hash.split('/');
  const arg = rest.length ? decodeURIComponent(rest.join('/')) : undefined;
  const nav = page === 'run' ? 'runs' : page;
  $$('.rail a.nav').forEach((a) => a.classList.toggle('on', a.dataset.nav === nav));
  const main = $('#main');
  main.innerHTML = '';
  window.scrollTo(0, 0);
  (routes[page] || viewConsole)(main, arg);
}
addEventListener('hashchange', route);

// ------------------------------------------------------------- status bar
async function status() {
  try {
    const [sys, g] = await Promise.all([api('/api/system'), api(`/api/gpu?since=${Date.now() / 1000}`)]);
    $('#ledApi').className = 'led ok';
    $('#ver').textContent = `v${sys.version}`;
    const e = $('#ledEngine');
    e.className = `led ${sys.engine_online ? 'ok' : ''}`;
    $('#engineTxt').textContent = sys.engine_online ? 'engine online' : 'engine offline';
    const gpu = g.latest?.gpus?.[0];
    if (gpu) {
      $('#ledGpu').className = `led ${gpu.util > 5 ? 'busy' : 'ok'}`;
      $('#gpuTxt').textContent = `${(gpu.name || 'GPU').replace('NVIDIA ', '').replace(' PCIe', '')} · ${gpu.util}% · ${gb(gpu.mem_used_mb)}/${gb(gpu.mem_total_mb)} GB`;
    } else { $('#ledGpu').className = 'led bad'; $('#gpuTxt').textContent = 'no GPU'; }
  } catch {
    $('#ledApi').className = 'led bad';
    $('#engineTxt').textContent = 'console unreachable';
  }
}
setInterval(status, 3000);
status();
route();
