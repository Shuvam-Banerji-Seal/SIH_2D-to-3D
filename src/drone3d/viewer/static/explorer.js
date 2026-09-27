// drone3d explorer: the textured mesh, dense cloud and Gaussian splats of a reconstruction, in one scene.
//
// Used by the standalone viewer that ships with every export (index.html next to it) and by the web
// console. Models arrive as scene.json entries: { name, mesh (GLB), points (PLY), splat (.splat),
// cameras [[x,y,z]], camera_rotations [[9 floats, world-from-camera]], intrinsics {f, width, height},
// images [names], view {eye, target}, bounds, units, georeferenced, up }.
//
// Navigation: orbit (drag, right-drag, wheel) or fly (WASD + QE, drag to look, shift x4, wheel = speed);
// "follow flight" replays the drone's own path and view; "look through keyframe" puts the camera where
// the drone was, with its field of view, so the model can be compared with the photo it came from.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const CLAY = 0xb9c2cc;

export class Explorer extends EventTarget {
  constructor(container, { background = 0x0b1118, sparkUrl = null } = {}) {
    super();
    this.container = container;
    this.sparkUrl = sparkUrl;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: false, powerPreference: 'high-performance' });
    this.renderScale = 1;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.classList.add('x3d-canvas');
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(background);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1e7);
    this.camera.up.set(0, 0, 1);
    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.12;
    this.hemi = new THREE.HemisphereLight(0xeef3ff, 0x3a3226, 1.6);
    this.sun = new THREE.DirectionalLight(0xffffff, 2.2);
    this.sun.position.set(0.4, -0.6, 1);
    this.scene.add(this.hemi, this.sun);
    this.root = new THREE.Group();
    this.marks = new THREE.Group();
    this.scene.add(this.root, this.marks);
    this.models = [];
    this.layers = { mesh: true, texture: true, wireframe: false, shaded: false, points: false, splats: false, cameras: true, grid: false };
    this.pointSize = 1.5;
    this.nav = 'orbit';
    this.keys = new Set();
    this.flySpeed = 1;
    this.follow = null;
    this.measuring = false;
    this.measures = [];
    this.pending = null;
    this.clock = new THREE.Clock();
    this.fps = 0;
    this._frames = 0;
    this._fpsT = performance.now();
    this._resize = () => this.resize();
    new ResizeObserver(this._resize).observe(container);
    this._bindInput();
    this.resize();
    this.renderer.setAnimationLoop(() => this._tick());
  }

  // ------------------------------------------------------------------ scene
  status(text) { this.dispatchEvent(new CustomEvent('status', { detail: text })); }

  clear() {
    for (const m of this.models) { this.root.remove(m.group); m.group.traverse((o) => { o.geometry?.dispose?.(); }); m.splatObj?.dispose?.(); }
    this.models = [];
    this.marks.clear();
    this.measures = [];
    this.pending = null;
  }

  async setScene(models, { frame = true, showAll = null } = {}) {
    this.clear();
    const all = showAll ?? (models.length > 1 && models.every((m) => m.georeferenced));
    for (const [i, m] of models.entries()) await this.addModel(m, { visible: all || i === 0, frame: false });
    this.current = 0;
    if (frame) this.frame();
    this.dispatchEvent(new CustomEvent('models'));
  }

  async addModel(m, { visible = true, frame = false } = {}) {
    const entry = { spec: m, group: new THREE.Group(), meshObj: null, pointsObj: null, splatObj: null, camObj: null, box: new THREE.Box3(),
      visible, triangles: 0, points: 0, splats: 0 };
    entry.group.visible = visible;
    this.root.add(entry.group);
    this.models.push(entry);
    this._cameraTrack(entry);
    if (m.mesh) await this._loadMesh(entry).catch((e) => this.status(`mesh failed: ${e.message || e}`));
    else if (m.points) { this.layers.points = true; await this._loadPoints(entry); }
    if (this.layers.points && m.points && !entry.pointsObj) await this._loadPoints(entry);
    if (this.layers.splats && m.splat && !entry.splatObj) await this._loadSplat(entry);
    this._apply(entry);
    if (frame) this.frame(this.models.length - 1);
    this.dispatchEvent(new CustomEvent('models'));
    return entry;
  }

  async _loadMesh(entry) {
    this.status(`loading ${entry.spec.name} mesh…`);
    const gltf = await new GLTFLoader().loadAsync(entry.spec.mesh);
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      const map = o.material?.map || null;
      if (map) { // triangle-soup atlas: mip levels blend neighbouring cells into a visible grid
        map.generateMipmaps = false; map.minFilter = THREE.LinearFilter; map.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
        map.colorSpace = THREE.SRGBColorSpace; map.needsUpdate = true;
      }
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
      o.userData.map = map;
      o.userData.vc = !!o.geometry.attributes.color;
      entry.triangles += (o.geometry.index ? o.geometry.index.count : o.geometry.attributes.position.count) / 3;
    });
    entry.meshObj = gltf.scene;
    entry.group.add(gltf.scene);
    entry.box.expandByObject(gltf.scene);
    this.status('');
  }

  async _loadPoints(entry) {
    if (!entry.spec.points || entry.pointsObj) return;
    this.status(`loading ${entry.spec.name} point cloud…`);
    const geo = await new PLYLoader().loadAsync(entry.spec.points);
    const mat = new THREE.PointsMaterial({ size: this.pointSize, sizeAttenuation: false, vertexColors: !!geo.attributes.color });
    entry.pointsObj = new THREE.Points(geo, mat);
    entry.points = geo.attributes.position.count;
    entry.group.add(entry.pointsObj);
    geo.computeBoundingBox();
    if (!entry.meshObj) entry.box.union(geo.boundingBox);
    this.status('');
  }

  async _loadSplat(entry) {
    if (!entry.spec.splat || entry.splatObj) return;
    this.status(`loading ${entry.spec.name} Gaussian splats…`);
    if (!this.spark) {
      const mod = await import(this.sparkUrl || new URL('./vendor/spark/spark.module.min.js', import.meta.url).href);
      this.sparkMod = mod;
      this.spark = new mod.SparkRenderer({ renderer: this.renderer });
      this.scene.add(this.spark);
    }
    const splat = new this.sparkMod.SplatMesh({ url: entry.spec.splat, fileType: 'splat' });
    await splat.initialized;
    entry.splatObj = splat;
    entry.splats = splat.packedSplats?.numSplats ?? 0;
    entry.group.add(splat);
    this.status('');
  }

  _cameraTrack(entry) {
    const c = entry.spec.cameras || [];
    if (c.length < 2) return;
    const g = new THREE.Group();
    const pts = c.map((p) => new THREE.Vector3(...p));
    g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xff9f1c, transparent: true, opacity: 0.9 })));
    const rots = entry.spec.camera_rotations;
    const box = new THREE.Box3().setFromPoints(pts);
    const size = Math.max(box.getSize(new THREE.Vector3()).length() / 60, 1e-3);
    if (rots && rots.length === c.length && entry.spec.intrinsics) { // small frusta: where the drone looked
      const { f, width, height } = entry.spec.intrinsics;
      const hw = (width / 2 / f) * size, hh = (height / 2 / f) * size;
      const corners = [[-hw, -hh, size], [hw, -hh, size], [hw, hh, size], [-hw, hh, size]];
      const seg = [];
      const step = Math.max(1, Math.round(c.length / 60));
      for (let i = 0; i < c.length; i += step) {
        const R = rots[i], o = pts[i];
        const w = corners.map(([x, y, z]) => new THREE.Vector3(R[0] * x + R[1] * y + R[2] * z, R[3] * x + R[4] * y + R[5] * z, R[6] * x + R[7] * y + R[8] * z).add(o));
        for (let k = 0; k < 4; k++) seg.push(o, w[k], w[k], w[(k + 1) % 4]);
      }
      g.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg), new THREE.LineBasicMaterial({ color: 0xffc46b, transparent: true, opacity: 0.55 })));
    }
    g.add(new THREE.Points(new THREE.BufferGeometry().setFromPoints(pts), new THREE.PointsMaterial({ color: 0xff9f1c, size: 4, sizeAttenuation: false })));
    entry.camObj = g;
    entry.group.add(g);
    pts.forEach((p) => entry.box.expandByPoint(p));
  }

  // ----------------------------------------------------------------- layers
  async setLayer(name, on) {
    this.layers[name] = on;
    if (name === 'points' && on) for (const m of this.models) if (m.visible) await this._loadPoints(m).catch((e) => this.status(`points failed: ${e}`));
    if (name === 'splats' && on) for (const m of this.models) if (m.visible) await this._loadSplat(m).catch((e) => this.status(`splats failed: ${e.message || e}`));
    this.models.forEach((m) => this._apply(m));
    if (name === 'grid') this._grid(on);
    this.dispatchEvent(new CustomEvent('layers'));
  }

  _apply(entry) {
    const L = this.layers;
    if (entry.meshObj) {
      entry.meshObj.visible = L.mesh;
      entry.meshObj.traverse((o) => {
        if (!o.isMesh) return;
        const map = L.texture ? o.userData.map : null;
        const vc = !map && L.texture !== false && o.userData.vc;
        const params = { map, vertexColors: !map && o.userData.vc && L.texture, color: map || vc ? 0xffffff : CLAY, side: THREE.DoubleSide, wireframe: L.wireframe };
        const want = L.shaded ? 'std' : 'basic';
        if (o.userData.kind !== want) { o.material?.dispose?.(); o.material = want === 'std' ? new THREE.MeshStandardMaterial({ ...params, roughness: 0.9, metalness: 0 }) : new THREE.MeshBasicMaterial(params); o.userData.kind = want; }
        else { Object.assign(o.material, params); o.material.needsUpdate = true; }
      });
    }
    if (entry.pointsObj) { entry.pointsObj.visible = L.points; entry.pointsObj.material.size = this.pointSize; }
    if (entry.splatObj) entry.splatObj.visible = L.splats;
    if (entry.camObj) entry.camObj.visible = L.cameras;
  }

  setPointSize(s) { this.pointSize = s; this.models.forEach((m) => this._apply(m)); }

  setRenderScale(s) { this.renderScale = s; this.resize(); }

  setBackground(hex) { this.scene.background = new THREE.Color(hex); }

  showModel(i, on) {
    const m = this.models[i]; if (!m) return;
    m.visible = m.group.visible = on;
    if (on) { if (this.layers.points) this._loadPoints(m).then(() => this._apply(m)); if (this.layers.splats) this._loadSplat(m).then(() => this._apply(m)); }
    this.dispatchEvent(new CustomEvent('models'));
  }

  solo(i) { this.models.forEach((m, k) => this.showModel(k, k === i)); this.current = i; this.frame(i); }

  _grid(on) {
    if (this.grid) { this.scene.remove(this.grid); this.grid = null; }
    if (!on) return;
    const box = this._visibleBox();
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const span = Math.max(size.x, size.y) * 1.4;
    const step = 10 ** Math.floor(Math.log10(span / 10));
    const n = Math.ceil(span / step);
    this.grid = new THREE.GridHelper(n * step, n, 0x3b4b5e, 0x1f2a36);
    this.grid.rotation.x = Math.PI / 2;
    const c = box.getCenter(new THREE.Vector3());
    this.grid.position.set(c.x, c.y, box.min.z);
    this.scene.add(this.grid);
  }

  // ------------------------------------------------------------- navigation
  _visibleBox() {
    const box = new THREE.Box3();
    this.models.filter((m) => m.visible).forEach((m) => {
      if (m.spec.bounds?.length) m.spec.bounds.forEach((q) => box.expandByPoint(new THREE.Vector3(...q)));
      else box.union(m.box);
    });
    return box;
  }

  frame(i = null) {
    const m = i != null ? this.models[i] : this.models.find((x) => x.visible);
    const box = i != null && m ? new THREE.Box3().setFromPoints((m.spec.bounds || []).map((q) => new THREE.Vector3(...q))) : this._visibleBox();
    if (box.isEmpty() && m) box.copy(m.box);
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3()), r = box.getSize(new THREE.Vector3()).length() / 2;
    this.camera.near = Math.max(r / 5000, 1e-4); this.camera.far = r * 200; this.camera.fov = 50;
    const view = (i != null || this.models.filter((x) => x.visible).length === 1) && m?.spec.view;
    if (view) { // where the drone was, looking where it looked, pulled back a little
      const eye = new THREE.Vector3(...view.eye), tgt = new THREE.Vector3(...view.target), back = eye.clone().sub(tgt);
      this.orbit.target.copy(tgt);
      this.camera.position.copy(eye).addScaledVector(back, 0.25).addScaledVector(this.camera.up, 0.15 * back.length());
    } else {
      this.orbit.target.copy(c);
      this.camera.position.copy(c).addScaledVector(new THREE.Vector3(0.55, -0.8, 0.65).normalize(), r * 2.1);
    }
    this.camera.updateProjectionMatrix();
    this.flySpeed = r / 8;
    if (this.layers.grid) this._grid(true);
  }

  setNavigation(mode) {
    this.nav = mode;
    this.orbit.enabled = mode === 'orbit';
    if (mode === 'orbit') { // orbit about what is in front of the camera
      const d = this.camera.position.distanceTo(this.orbit.target) || this.flySpeed * 4;
      this.orbit.target.copy(this.camera.position).addScaledVector(this.camera.getWorldDirection(new THREE.Vector3()), d);
    }
    this.follow = null;
    this.dispatchEvent(new CustomEvent('nav'));
  }

  lookThrough(modelIndex, k) {
    const m = this.models[modelIndex]; if (!m) return null;
    const c = m.spec.cameras?.[k], R = m.spec.camera_rotations?.[k], K = m.spec.intrinsics;
    if (!c || !R) return null;
    this.follow = null;
    this._pose(new THREE.Vector3(...c), R);
    if (K) this._matchPhoto(K);
    this.setNavigation('fly');
    return { image: m.spec.images?.[k], aspect: K ? K.width / K.height : null };
  }

  // Field of view of the keyframe as the photo is shown letterboxed ("contain") in this canvas.
  _matchPhoto(K) {
    const imgAspect = K.width / K.height, tanV = K.height / 2 / K.f;
    const t = this.camera.aspect >= imgAspect ? tanV : (tanV * imgAspect) / this.camera.aspect;
    this.camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(t));
    this.camera.updateProjectionMatrix();
  }

  _pose(pos, R) { // world-from-camera rotation (OpenCV: x right, y down, z forward) -> three.js camera
    const m = new THREE.Matrix4().set(R[0], -R[1], -R[2], 0, R[3], -R[4], -R[5], 0, R[6], -R[7], -R[8], 0, 0, 0, 0, 1);
    this.camera.position.copy(pos);
    this.camera.quaternion.setFromRotationMatrix(m);
  }

  followFlight(on, { modelIndex = null, speed = 1 } = {}) {
    if (!on) { this.follow = null; this.dispatchEvent(new CustomEvent('nav')); return; }
    const m = this.models[modelIndex ?? this.models.findIndex((x) => x.visible)];
    if (!m || !m.spec.cameras || m.spec.cameras.length < 2) return;
    const pts = m.spec.cameras.map((p) => new THREE.Vector3(...p));
    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
    const rots = m.spec.camera_rotations;
    const quats = rots ? rots.map((R) => { const q = new THREE.Quaternion(); q.setFromRotationMatrix(new THREE.Matrix4().set(R[0], -R[1], -R[2], 0, R[3], -R[4], -R[5], 0, R[6], -R[7], -R[8], 0, 0, 0, 0, 1)); return q; }) : null;
    if (m.spec.intrinsics) this._matchPhoto(m.spec.intrinsics);
    this.orbit.enabled = false;
    this.follow = { curve, quats, t: 0, n: pts.length, speed, seconds: Math.max(8, pts.length * 0.35) };
    this.dispatchEvent(new CustomEvent('nav'));
  }

  // ---------------------------------------------------------------- input
  _bindInput() {
    const el = this.renderer.domElement;
    el.tabIndex = 0;
    let drag = null, down = null;
    el.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; if (this.nav === 'fly') { drag = [e.clientX, e.clientY]; el.setPointerCapture(e.pointerId); } el.focus(); });
    el.addEventListener('pointermove', (e) => {
      if (this.nav !== 'fly' || !drag) return;
      const dx = e.clientX - drag[0], dy = e.clientY - drag[1]; drag = [e.clientX, e.clientY];
      const yaw = new THREE.Quaternion().setFromAxisAngle(this.camera.up, -dx * 0.0035);
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
      const pitch = new THREE.Quaternion().setFromAxisAngle(right, -dy * 0.0035);
      this.camera.quaternion.premultiply(yaw).premultiply(pitch);
    });
    el.addEventListener('pointerup', (e) => {
      drag = null;
      if (this.measuring && down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4) this._measureClick(e);
    });
    el.addEventListener('wheel', (e) => { if (this.nav === 'fly') { e.preventDefault(); this.flySpeed *= e.deltaY > 0 ? 0.85 : 1.18; } }, { passive: false });
    el.addEventListener('dblclick', (e) => { const p = this._pick(e); if (p) { this.orbit.target.copy(p); if (this.nav !== 'orbit') this.setNavigation('orbit'); } });
    el.addEventListener('keydown', (e) => { this.keys.add(e.code); if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'Space'].includes(e.code)) e.preventDefault(); });
    el.addEventListener('keyup', (e) => this.keys.delete(e.code));
    el.addEventListener('blur', () => this.keys.clear());
  }

  _pick(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.camera);
    ray.params.Points.threshold = this._visibleBox().getSize(new THREE.Vector3()).length() / 2000;
    const targets = this.models.filter((m) => m.visible).flatMap((m) => [this.layers.mesh && m.meshObj, this.layers.points && m.pointsObj].filter(Boolean));
    const hit = ray.intersectObjects(targets, true)[0];
    return hit ? hit.point.clone() : null;
  }

  setMeasuring(on) { this.measuring = on; this.pending = null; }

  clearMeasures() { this.marks.clear(); this.measures = []; this.pending = null; this.dispatchEvent(new CustomEvent('measure')); }

  _measureClick(e) {
    const p = this._pick(e); if (!p) return;
    const s = this._visibleBox().getSize(new THREE.Vector3()).length() / 350;
    const dot = (q, color) => { const o = new THREE.Mesh(new THREE.SphereGeometry(s, 12, 8), new THREE.MeshBasicMaterial({ color, depthTest: false })); o.position.copy(q); o.renderOrder = 10; this.marks.add(o); };
    if (!this.pending) { this.pending = p; dot(p, 0xff9f1c); return; }
    dot(p, 0x39d98a);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([this.pending, p]), new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }));
    line.renderOrder = 9; this.marks.add(line);
    const d = this.pending.distanceTo(p), dz = p.z - this.pending.z;
    this.measures.push({ d, dz, units: this.models.find((m) => m.visible)?.spec.units || 'units' });
    this.pending = null;
    this.dispatchEvent(new CustomEvent('measure'));
  }

  // ---------------------------------------------------------------- output
  async screenshot() {
    this.renderer.render(this.scene, this.camera);
    return new Promise((ok) => this.renderer.domElement.toBlob(ok, 'image/png'));
  }

  stats() {
    const vis = this.models.filter((m) => m.visible);
    return { fps: this.fps, models: this.models.length, visible: vis.length,
      triangles: vis.reduce((a, m) => a + (this.layers.mesh ? m.triangles : 0), 0),
      points: vis.reduce((a, m) => a + (this.layers.points ? m.points : 0), 0),
      splats: vis.reduce((a, m) => a + (this.layers.splats ? m.splats : 0), 0),
      calls: this.renderer.info.render.calls, pixelRatio: this.renderer.getPixelRatio() };
  }

  resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2) * this.renderScale);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%'; this.renderer.domElement.style.height = '100%';
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  _tick() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    if (this.follow) {
      const f = this.follow;
      f.t = Math.min(1, f.t + (dt * f.speed) / f.seconds);
      this.camera.position.copy(f.curve.getPointAt(f.t));
      if (f.quats) { const x = f.t * (f.n - 1), i = Math.min(f.n - 2, Math.floor(x)); this.camera.quaternion.slerpQuaternions(f.quats[i], f.quats[i + 1], x - i); }
      else this.camera.lookAt(f.curve.getPointAt(Math.min(1, f.t + 0.02)));
      if (f.t >= 1) { this.follow = null; this.setNavigation('fly'); this.dispatchEvent(new CustomEvent('followend')); }
    } else if (this.nav === 'fly') {
      const k = this.keys, v = new THREE.Vector3();
      if (k.has('KeyW')) v.z -= 1; if (k.has('KeyS')) v.z += 1; if (k.has('KeyA')) v.x -= 1; if (k.has('KeyD')) v.x += 1;
      if (v.lengthSq()) this.camera.position.add(v.normalize().applyQuaternion(this.camera.quaternion).multiplyScalar(this.flySpeed * dt * (k.has('ShiftLeft') || k.has('ShiftRight') ? 4 : 1)));
      if (k.has('KeyE') || k.has('Space')) this.camera.position.addScaledVector(this.camera.up, this.flySpeed * dt);
      if (k.has('KeyQ')) this.camera.position.addScaledVector(this.camera.up, -this.flySpeed * dt);
    } else this.orbit.update();
    this.renderer.render(this.scene, this.camera);
    this._frames++;
    const now = performance.now();
    if (now - this._fpsT > 500) { this.fps = Math.round((this._frames * 1000) / (now - this._fpsT)); this._frames = 0; this._fpsT = now; this.dispatchEvent(new CustomEvent('tick')); }
  }

  dispose() { this.renderer.setAnimationLoop(null); this.clear(); this.renderer.dispose(); this.renderer.domElement.remove(); }
}
