// Live: reconstruct while the drone flies. A stream URL, a camera, or a recorded flight replayed in real time.
import { $, $$, api, esc, fmtS, post, prettyVideo, toast } from './util.js';
import { Options, renderModules, renderProfiles, renderResolution } from './options.js';

const keep = { mode: 'replay', video: null, source: 'rtsp://192.168.1.1:554/live', segment: 30, telemetry: '' };

export async function viewLive(main) {
  let schema, profiles, videos, logs, engine, runs;
  try {
    [schema, profiles, videos, logs, engine, runs] = await Promise.all([api('/api/schema'), api('/api/profiles'), api('/api/videos'), api('/api/telemetry'), api('/api/engine'), api('/api/runs')]);
  } catch (e) { main.innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  const o = new Options(schema, profiles, profiles.some((p) => p.name === 'fast') ? 'fast' : profiles[0]?.name);
  main.innerHTML = `
  <div class="page-head rise"><h1>Live reconstruction<small>The stream is cut into segments at keyframes as it arrives; each closed segment is modelled on the warm engine while the next one records.</small></h1></div>
  <div class="grid g-12">
    <div class="s8 stack">
      <section class="panel rise" style="--i:1"><h2>Source</h2>
        <div class="seg" id="mode"><button data-m="replay">Replay a recorded flight</button><button data-m="url">Stream URL</button><button data-m="camera">Camera</button></div>
        <div id="src" style="margin-top:14px"></div>
        <div class="row" style="margin-top:14px"><span class="muted" style="min-width:130px">Segment length</span>
          <input type="range" min="10" max="120" step="5" value="${keep.segment}" id="seglen" style="flex:1;max-width:320px"><b class="mono" id="segval">${keep.segment} s</b></div>
        <div class="note" style="margin-top:6px">Shorter segments give the first model sooner; longer ones keep more of a pass together (a segment boundary splits a camera move).</div>
        <div class="row" style="margin-top:12px"><span class="muted" style="min-width:130px">GPS / flight log</span><select id="telemetry" style="max-width:420px"></select></div>
        <div class="note">With a log every segment is georeferenced into one ENU frame (origin at its first fix), so segments line up in one scene.</div>
      </section>
      <section class="panel rise" style="--i:2"><h2>Profile</h2><div class="choices" id="profiles"></div></section>
      <section class="panel rise" style="--i:3"><h2>Resolution</h2><div id="res"></div></section>
      <section class="panel rise" style="--i:4"><h2>Modules</h2><div id="mods"></div></section>
    </div>
    <div class="s4 stack">
      <section class="panel rise" style="--i:2" id="go"></section>
      <section class="panel rise" style="--i:3"><h2>Sessions</h2><div id="sessions"></div></section>
    </div>
  </div>`;

  const redraw = () => { renderProfiles($('#profiles'), o, redraw); renderResolution($('#res'), o, redraw); renderModules($('#mods'), o, redraw); go(); };
  function src() {
    $$('#mode button').forEach((b) => b.classList.toggle('on', b.dataset.m === keep.mode));
    if (keep.mode === 'replay') {
      $('#src').innerHTML = `<div class="note" style="margin-bottom:8px">Rehearse a flight: the file is fed to the recorder at its own frame rate, exactly as a drone link would deliver it.</div>
        <div class="choices scroll">${videos.map((v) => `<div class="choice ${keep.video === v.path ? 'sel' : ''}" data-p="${esc(v.path)}"><div class="n">${esc(prettyVideo(v.name))}</div><div class="m">${v.width}×${v.height} · ${fmtS(v.duration_s)}</div></div>`).join('')}</div>`;
      $$('#src .choice').forEach((d) => d.addEventListener('click', () => { keep.video = d.dataset.p; src(); go(); }));
    } else if (keep.mode === 'url') {
      $('#src').innerHTML = `<input type="text" id="url" value="${esc(keep.source.startsWith('/dev/') ? 'rtsp://192.168.1.1:554/live' : keep.source)}" placeholder="rtsp:// rtmp:// srt:// udp:// http(s)://">
        <div class="note" style="margin-top:6px">RTSP (DJI, most IP links), RTMP, SRT, UDP/MPEG-TS, HLS or HTTP. Recorded without re-encoding; a dead link is dropped after 15 s.</div>`;
      $('#url').addEventListener('input', (e) => { keep.source = e.target.value.trim(); go(); });
      keep.source = $('#url').value;
    } else {
      $('#src').innerHTML = `<input type="text" id="cam" value="${esc(keep.source.startsWith('/dev/') ? keep.source : '/dev/video0')}">
        <div class="note" style="margin-top:6px">A V4L2 capture device (HDMI grabber on the ground station); frames are encoded on NVENC with a keyframe every second.</div>`;
      $('#cam').addEventListener('input', (e) => { keep.source = e.target.value.trim(); go(); });
      keep.source = $('#cam').value;
    }
  }
  function go() {
    const ready = keep.mode === 'replay' ? !!keep.video : /^(rtsp|rtmp|srt|udp|http|https|tcp):\/\/\S+$|^\/dev\/video\d+$/.test(keep.source);
    $('#go').innerHTML = `<h2>Go live</h2>
      <div class="between"><span class="muted">Engine</span><span class="row" style="gap:6px"><span class="led ${engine.online ? 'ok' : ''}"></span><span class="mono" style="font-size:12px">${engine.online ? 'online' : 'starts on demand'}</span></span></div>
      <div class="between" style="margin-top:6px"><span class="muted">Profile</span><b class="mono">${esc(o.profile)} · ${esc(o.resolution())}</b></div>
      <div class="between" style="margin-top:6px"><span class="muted">Segments</span><b class="mono">${keep.segment} s</b></div>
      <div class="hr"></div>
      <input type="text" id="name" placeholder="session name (optional)">
      <button class="btn primary" id="start" style="width:100%;justify-content:center;margin-top:10px;padding:12px" ${ready ? '' : 'disabled'}>Start live reconstruction</button>
      <div id="err"></div>`;
    $('#start').onclick = async () => {
      $('#start').disabled = true; $('#err').innerHTML = '';
      const name = $('#name').value.trim() || `live_${new Date().toISOString().slice(5, 19).replace(/[-:T]/g, '')}`;
      try {
        await post('/api/live', { name, ...o.payload(), segment_s: keep.segment, telemetry: keep.telemetry || null,
          ...(keep.mode === 'replay' ? { video: keep.video } : { source: keep.source }) });
        toast(`${name}: recording`, 'ok');
        location.hash = `#/run/${encodeURIComponent(name)}`;
      } catch (e) { $('#err').innerHTML = `<div class="err">${esc(e.message)}</div>`; $('#start').disabled = false; }
    };
  }
  $$('#mode button').forEach((b) => b.addEventListener('click', () => { keep.mode = b.dataset.m; src(); go(); }));
  $('#seglen').addEventListener('input', (e) => { keep.segment = +e.target.value; $('#segval').textContent = `${keep.segment} s`; go(); });
  $('#telemetry').innerHTML = '<option value="">none — each segment in its own SfM frame</option>' + logs.map((l) => `<option value="${esc(l.path)}">${esc(l.name)}</option>`).join('');
  $('#telemetry').addEventListener('change', (e) => { keep.telemetry = e.target.value; });
  const sessions = runs.filter((r) => r.kind === 'live');
  $('#sessions').innerHTML = sessions.length ? sessions.map((r) => `<a class="between" href="#/run/${encodeURIComponent(r.name)}" style="padding:8px 0;border-top:1px solid var(--line);color:inherit">
      <b>${esc(r.name)}</b><span class="st ${esc(r.status)}">${esc(r.status)}</span></a>`).join('') : '<span class="note">none yet</span>';
  src(); redraw();
  if (!engine.online) toast('The engine starts when the session does (about 5 s)');
}
