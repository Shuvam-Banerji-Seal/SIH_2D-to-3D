// A viewer for one exported file: GLB, OBJ (+MTL), PLY (mesh or points), STL, FBX -- the file itself, as a user
// would open it in a 3D tool, next to the layered explorer of the run page.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { $, $$, api, esc, fmtN, icon, onLeave, toast } from './util.js';

export const VIEWABLE = ['glb', 'gltf', 'obj', 'ply', 'stl', 'fbx'];
const Z_UP = new Set(['obj', 'ply', 'stl']); // the exporter writes these in the survey frame (z up); glTF is y-up
const BG = { dark: 0x0b1118, grey: 0x3a4250, light: 0xe9edf2 };

export async function viewModel(main, arg = '') {
  const [run, ...rest] = arg.split('/');
  const file = rest.join('/');
  const ext = file.split('.').pop().toLowerCase();
  const url = `/runs/${encodeURIComponent(run)}/export/${file.split('/').map(encodeURIComponent).join('/')}`;
  main.innerHTML = `
  <div class="page-head rise"><h1>${esc(file.split('/').pop())}<small><a href="#/run/${encodeURIComponent(run)}">${esc(run)}</a> · ${esc(file)}</small></h1>
    <div class="row" style="margin-left:auto"><a class="btn small" href="${url}" download>${icon.down}Download</a></div></div>
  <div class="grid g-12">
    <section class="panel s9 rise" style="--i:1;padding:0;overflow:hidden;position:relative">
      <div id="mv" style="height:74vh;min-height:420px;position:relative"></div>
      <div id="mvStatus" class="note mono" style="position:absolute;left:14px;bottom:10px">loading…</div>
      <div class="zoom" style="position:absolute;right:12px;top:12px;display:flex;flex-direction:column;gap:6px">
        <button class="btn small" id="mvIn" title="zoom in">+</button><button class="btn small" id="mvOut" title="zoom out">−</button>
        <button class="btn small" id="mvFit" title="fit">⤢</button><button class="btn small" id="mvShot" title="screenshot">${icon.cam}</button></div>
    </section>
    <aside class="s3 stack">
      <section class="panel rise" style="--i:2"><h2>Display</h2>
        <div class="muted" style="font-size:12px;margin-bottom:4px">shading</div>
        <div class="seg" style="width:100%" id="mvShade">${['authored', 'clay', 'normals'].map((s) => `<button data-s="${s}" style="flex:1" class="${s === 'authored' ? 'on' : ''}">${s}</button>`).join('')}</div>
        <label class="row" style="gap:8px;margin-top:10px;font-size:13px"><span class="sw"><input type="checkbox" id="mvWire"><span></span></span>wireframe</label>
        <label class="row" style="gap:8px;margin-top:6px;font-size:13px"><span class="sw"><input type="checkbox" id="mvGrid" checked><span></span></span>ground grid</label>
        <div class="muted" style="font-size:12px;margin:10px 0 4px">up axis</div>
        <div class="seg" style="width:100%" id="mvUp">${['z', 'y'].map((u) => `<button data-u="${u}" style="flex:1" class="${(Z_UP.has(ext) ? 'z' : 'y') === u ? 'on' : ''}">${u} up</button>`).join('')}</div>
        <div class="muted" style="font-size:12px;margin:10px 0 4px">background</div>
        <div class="seg" style="width:100%" id="mvBg">${Object.keys(BG).map((b) => `<button data-b="${b}" style="flex:1" class="${b === 'dark' ? 'on' : ''}">${b}</button>`).join('')}</div>
        <div class="muted" style="font-size:12px;margin:10px 0 4px">render scale</div>
        <div class="seg" style="width:100%" id="mvRs">${[0.5, 1, 2].map((r) => `<button data-r="${r}" style="flex:1" class="${r === 1 ? 'on' : ''}">${r}×</button>`).join('')}</div>
        <div class="row" style="margin-top:10px;font-size:12px" id="mvPsRow" hidden><span class="muted">point size</span><input type="range" min="0.5" max="8" step="0.5" value="2" id="mvPs" style="flex:1"></div>
        <div class="note" style="margin-top:10px">drag to orbit · right-drag to pan · wheel to zoom · double-click to focus</div>
      </section>
      <section class="panel rise" style="--i:3"><h2>File</h2><div id="mvInfo" class="note mono">—</div></section>
      <section class="panel rise" style="--i:4"><h2>Other files of this model</h2><div id="mvFiles" class="stack" style="gap:6px"></div></section>
    </aside>
  </div>`;

  // ---------------------------------------------------------------- scene
  const host = $('#mv');
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG.dark);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1e6);
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true; orbit.dampingFactor = 0.12;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404858, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(1, 1.6, 1.2); scene.add(sun);
  const root = new THREE.Group(); scene.add(root);
  let grid = null, object = null, frames = 0, running = true;
  const originals = new Map();
  const wake = () => { frames = 90; };
  const resize = () => { const w = host.clientWidth, h = host.clientHeight; renderer.setSize(w, h, false); renderer.domElement.style.width = '100%'; renderer.domElement.style.height = '100%'; camera.aspect = w / h; camera.updateProjectionMatrix(); wake(); };
  const ro = new ResizeObserver(resize); ro.observe(host); resize();
  orbit.addEventListener('change', wake);
  (function loop() { if (!running) return; requestAnimationFrame(loop); if (frames > 0) { frames--; orbit.update(); renderer.render(scene, camera); } })();
  onLeave(() => { running = false; ro.disconnect(); orbit.dispose(); renderer.dispose(); scene.traverse((o) => { o.geometry?.dispose?.(); [].concat(o.material || []).forEach((m) => { m.map?.dispose?.(); m.dispose?.(); }); }); });

  const setUp = (u) => { root.rotation.set(u === 'z' ? -Math.PI / 2 : 0, 0, 0); root.updateMatrixWorld(true); fit(); };
  function fit() {
    if (!object) return;
    const box = new THREE.Box3().setFromObject(root);
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3()), r = size.length() / 2 || 1;
    orbit.target.copy(c);
    camera.position.copy(c).add(new THREE.Vector3(0.6, 0.55, 0.9).normalize().multiplyScalar(r * 2.0));
    camera.near = r / 2000; camera.far = r * 200; camera.updateProjectionMatrix();
    if (grid) scene.remove(grid);
    const span = Math.max(size.x, size.z) * 1.3, step = 10 ** Math.floor(Math.log10(span / 10));
    grid = new THREE.GridHelper(Math.ceil(span / step) * step, Math.ceil(span / step), 0x3b4b5e, 0x1f2a36);
    grid.position.set(c.x, box.min.y, c.z); grid.visible = $('#mvGrid').checked; scene.add(grid);
    wake();
  }
  function shade(mode) {
    object?.traverse((o) => {
      if (!o.isMesh) return;
      if (!originals.has(o)) originals.set(o, o.material);
      if (mode === 'authored') o.material = originals.get(o);
      else if (mode === 'clay') o.material = new THREE.MeshStandardMaterial({ color: 0xbfc6cf, roughness: 0.85, metalness: 0, flatShading: false, side: THREE.DoubleSide });
      else o.material = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
      [].concat(o.material).forEach((m) => { m.wireframe = $('#mvWire').checked; });
    });
    wake();
  }

  // ---------------------------------------------------------------- load
  const t0 = performance.now();
  const status = (s) => { $('#mvStatus').textContent = s; };
  const progress = (e) => { if (e.lengthComputable) status(`loading… ${Math.round((100 * e.loaded) / e.total)} % of ${(e.total / 1e6).toFixed(0)} MB`); };
  const colorMaterial = (geom) => new THREE.MeshStandardMaterial({ vertexColors: !!geom.attributes.color, color: geom.attributes.color ? 0xffffff : 0xbfc6cf, roughness: 0.9, side: THREE.DoubleSide });
  try {
    if (ext === 'glb' || ext === 'gltf') object = (await new GLTFLoader().loadAsync(url, progress)).scene;
    else if (ext === 'obj') {
      const obj = new OBJLoader();
      try { // the textured OBJ names its material library; plain ones have none
        const mtlName = file.replace(/\.obj$/i, '.mtl'), base = url.slice(0, url.lastIndexOf('/') + 1);
        const mtl = await new MTLLoader().setResourcePath(base).loadAsync(`/runs/${encodeURIComponent(run)}/export/${mtlName.split('/').map(encodeURIComponent).join('/')}`);
        mtl.preload(); obj.setMaterials(mtl);
      } catch { /* no material library */ }
      object = await obj.loadAsync(url, progress);
    } else if (ext === 'ply') {
      const geom = await new PLYLoader().loadAsync(url, progress);
      if (geom.index) { geom.computeVertexNormals(); object = new THREE.Mesh(geom, colorMaterial(geom)); } // faces: a mesh
      else { // no faces: the dense point cloud
        object = new THREE.Points(geom, new THREE.PointsMaterial({ size: 2, sizeAttenuation: false, vertexColors: !!geom.attributes.color, color: geom.attributes.color ? 0xffffff : 0xbfc6cf }));
        $('#mvPsRow').hidden = false;
      }
    } else if (ext === 'stl') {
      const geom = await new STLLoader().loadAsync(url, progress); geom.computeVertexNormals();
      object = new THREE.Mesh(geom, colorMaterial(geom));
    } else if (ext === 'fbx') object = await new FBXLoader().loadAsync(url, progress);
    else throw new Error(`.${ext} cannot be shown in the browser — download it instead`);
  } catch (e) { status(''); $('#mv').innerHTML = `<div class="err" style="margin:24px">${esc(e.message || e)}</div>`; return; }
  // A photo texture already holds the scene's light: show it unlit, as it was captured; vertex colours stay lit but
  // matte (glTF exporters default to a metallic PBR material, which renders photogrammetry near black).
  object.traverse((o) => {
    if (!o.isMesh) return;
    o.material = [].concat(o.material).map((m) => (m.map ? new THREE.MeshBasicMaterial({ map: m.map, side: THREE.DoubleSide })
      : Object.assign(m, { metalness: 0, roughness: 1, side: THREE.DoubleSide })));
    if (o.material.length === 1) o.material = o.material[0];
  });
  root.add(object);
  let verts = 0, tris = 0, points = 0, textures = 0;
  object.traverse((o) => {
    const g = o.geometry; if (!g) return;
    const n = g.attributes.position?.count || 0;
    if (o.isPoints) points += n; else { verts += n; tris += g.index ? g.index.count / 3 : n / 3; }
    [].concat(o.material || []).forEach((m) => { if (m.map) textures++; });
  });
  setUp(Z_UP.has(ext) ? 'z' : 'y');
  const size = new THREE.Box3().setFromObject(root).getSize(new THREE.Vector3());
  $('#mvInfo').innerHTML = [['format', ext.toUpperCase()], ['vertices', fmtN(verts)], ['triangles', fmtN(tris)], ['points', fmtN(points)], ['textures', String(textures)],
    ['size', `${size.x.toPrecision(3)} × ${size.z.toPrecision(3)} × ${size.y.toPrecision(3)}`], ['loaded in', `${((performance.now() - t0) / 1000).toFixed(1)} s`]]
    .filter(([, v]) => v !== '—' && v !== '0').map(([k, v]) => `<div class="between"><span class="muted">${k}</span><span>${esc(v)}</span></div>`).join('');
  status('');
  toast(`${file.split('/').pop()}: ${fmtN(tris || points)} ${tris ? 'triangles' : 'points'}`, 'ok');

  // ---------------------------------------------------------------- controls
  $$('#mvShade button').forEach((b) => b.addEventListener('click', () => { $$('#mvShade button').forEach((x) => x.classList.toggle('on', x === b)); shade(b.dataset.s); }));
  $('#mvWire').addEventListener('change', () => shade($('#mvShade button.on').dataset.s));
  $('#mvGrid').addEventListener('change', (e) => { if (grid) grid.visible = e.target.checked; wake(); });
  $$('#mvUp button').forEach((b) => b.addEventListener('click', () => { $$('#mvUp button').forEach((x) => x.classList.toggle('on', x === b)); setUp(b.dataset.u); }));
  $$('#mvBg button').forEach((b) => b.addEventListener('click', () => { $$('#mvBg button').forEach((x) => x.classList.toggle('on', x === b)); scene.background = new THREE.Color(BG[b.dataset.b]); wake(); }));
  $$('#mvRs button').forEach((b) => b.addEventListener('click', () => { $$('#mvRs button').forEach((x) => x.classList.toggle('on', x === b)); renderer.setPixelRatio(Number(b.dataset.r) * Math.min(devicePixelRatio, 2)); resize(); }));
  $('#mvPs').addEventListener('input', (e) => { object.traverse((o) => { if (o.isPoints) o.material.size = Number(e.target.value); }); wake(); });
  const zoom = (k) => { camera.position.sub(orbit.target).multiplyScalar(k).add(orbit.target); wake(); };
  $('#mvIn').onclick = () => zoom(0.8); $('#mvOut').onclick = () => zoom(1.25); $('#mvFit').onclick = fit;
  $('#mvShot').onclick = () => { renderer.render(scene, camera); const a = document.createElement('a'); a.href = renderer.domElement.toDataURL('image/png'); a.download = `${file.split('/').pop()}.png`; a.click(); };
  renderer.domElement.addEventListener('dblclick', (e) => { // focus on the point under the cursor
    const r = renderer.domElement.getBoundingClientRect(), ray = new THREE.Raycaster();
    ray.params.Points.threshold = size.length() / 500;
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
    const hit = ray.intersectObject(root, true)[0];
    if (hit) { orbit.target.copy(hit.point); wake(); }
  });

  // ---------------------------------------------------------------- the model's other files
  try {
    const r = await api(`/api/runs/${encodeURIComponent(run)}`);
    const dir = file.includes('/') ? file.slice(0, file.lastIndexOf('/') + 1) : '';
    const mine = (r.summary.files || []).flatMap((m) => m.files).filter((f) => f.startsWith(dir) && VIEWABLE.includes(f.split('.').pop().toLowerCase()));
    $('#mvFiles').innerHTML = mine.map((f) => `<a class="btn tiny ${f === file ? 'primary' : ''}" href="#/model/${encodeURIComponent(run)}/${f.split('/').map(encodeURIComponent).join('/')}">${esc(f.split('/').pop())}</a>`).join('') || '<span class="note">—</span>';
  } catch { $('#mvFiles').innerHTML = '<span class="note">—</span>'; }
}
