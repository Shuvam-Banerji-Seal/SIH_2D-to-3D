// Every tunable option, from the config schema: search, changed-only, reset, and sections a profile
// does not run dimmed. Also the presets and module switches that write into the same overrides.
import { $, $$, esc } from './util.js';

export const SECTION_STAGE = { ingest: 'ingest', keyframes: 'keyframes', sfm: 'sfm', dense: 'dense', depth: 'depth', splat: 'splat',
  mesh: 'mesh', geo: 'georef', export: 'export', render: 'render', metrics: 'metrics' };

export const RESOLUTIONS = {
  draft: { label: 'Draft', note: '1280 px keyframes · 384 px depth · 2K texture · 300k-triangle viewer mesh — fastest', o: {
    'keyframes.output_long_side': 1280, 'dense.long_side': 384, 'export.texture_size': 2048, 'export.max_triangles': 300000 } },
  high: { label: 'High', note: 'the profile as tuned: 1920 px keyframes · 480 px depth · 4K texture', o: {} },
  ultra: { label: 'Ultra', note: '2048 px keyframes · 960 px depth on every keyframe · 8K texture · 1.5M-triangle viewer mesh — slower', o: {
    'keyframes.output_long_side': 2048, 'dense.long_side': 960, 'dense.keyframe_stride': 1, 'export.texture_size': 8192, 'export.max_triangles': 1500000 } },
};
const RES_KEYS = new Set(Object.values(RESOLUTIONS).flatMap((r) => Object.keys(r.o)));
const DA = 'depth-anything/Depth-Anything-V2-Large-hf';

// Module switches: each reads its state from the effective config and writes overrides (and stages).
export const MODULES = [
  { key: 'depth', title: 'Monocular depth fill', desc: 'Depth Anything V2 Large completes what flow triangulation cannot see (untextured roofs, water, far ground)',
    get: (s) => !!s.eff('dense', 'mono_model'), set: (s, on) => s.override('dense.mono_model', on ? DA : null) },
  { key: 'texture', title: 'Photo texture', desc: 'bake the keyframes onto the mesh (GPU, best three views per triangle)',
    get: (s) => s.eff('export', 'texture') !== false, set: (s, on) => s.override('export.texture', on) },
  { key: 'splat', title: 'Gaussian splats', desc: '3D Gaussian Splatting from the dense cloud, for photoreal fly-throughs (~35 s per model at the fast settings)',
    get: (s) => s.stages().includes('splat'), set: (s, on) => {
      s.toggleStage('splat', on);
      if (on && s.profile === 'fast') { s.override('splat.models', 'largest'); s.override('splat.quality', 'low'); s.override('splat.iterations', 7000); s.override('splat.depth_weight', 0); }
      if (!on) ['splat.models', 'splat.quality', 'splat.iterations', 'splat.depth_weight'].forEach((k) => s.override(k, undefined));
    } },
  { key: 'geo', title: 'Georeference', desc: 'with a flight log: ENU metres, UTM with an EPSG code in LAS and GeoTIFF',
    get: (s) => s.eff('geo', 'enabled') !== false && s.stages().includes('georef'), set: (s, on) => { s.override('geo.enabled', on); s.toggleStage('georef', on); } },
  { key: 'las', title: 'LAS point cloud', desc: 'LAS 1.4, point format 7 (RGB)', get: (s) => s.eff('export', 'las') !== false, set: (s, on) => s.override('export.las', on) },
  { key: 'geotiff', title: 'GeoTIFF DSM + orthophoto', desc: 'top-down surface model and colour raster', get: (s) => s.eff('export', 'geotiff') !== false, set: (s, on) => s.override('export.geotiff', on) },
  { key: 'fbx', title: 'FBX', desc: 'for DCC tools (assimp)', get: (s) => (s.eff('export', 'mesh_formats') || []).includes('fbx'),
    set: (s, on) => { const f = new Set(s.eff('export', 'mesh_formats') || ['ply', 'obj', 'glb']); on ? f.add('fbx') : f.delete('fbx'); s.override('export.mesh_formats', ['ply', 'obj', 'glb', 'fbx'].filter((x) => f.has(x))); } },
];

// Form state shared by the Build and Live pages.
export class Options {
  constructor(schema, profiles, profile) {
    this.schema = schema; this.profiles = profiles; this.profile = profile; this.overrides = {}; this.stageList = null;
    this.filter = ''; this.modifiedOnly = false; this.onChange = () => {};
  }
  values() { return this.profiles.find((p) => p.name === this.profile)?.values || {}; }
  profileValue(section, key) { const v = this.values(); return section ? (v[section] || {})[key] : v[key]; }
  field(section, key) { return (section ? this.schema.sections.find((s) => s.key === section)?.fields : this.schema.top)?.find((f) => f.key === key); }
  base(section, key) { const pv = this.profileValue(section, key); return pv !== undefined ? pv : this.field(section, key)?.default; }
  eff(section, key) { const d = section ? `${section}.${key}` : key; return d in this.overrides ? this.overrides[d] : this.base(section, key); }
  override(dotted, value) {
    const [section, key] = dotted.includes('.') ? dotted.split('.') : [null, dotted];
    if (value === undefined || JSON.stringify(value) === JSON.stringify(this.base(section, key))) delete this.overrides[dotted];
    else this.overrides[dotted] = value;
  }
  stages() { return this.stageList || this.profileValue(null, 'stages') || this.schema.stages; }
  toggleStage(stage, on) {
    const set = new Set(this.stages());
    on ? set.add(stage) : set.delete(stage);
    this.stageList = this.schema.stages.filter((s) => set.has(s));
    if (JSON.stringify(this.stageList) === JSON.stringify(this.profileValue(null, 'stages'))) this.stageList = null;
  }
  setProfile(name) { this.profile = name; this.overrides = {}; this.stageList = null; }
  resolution() {
    for (const [k, r] of Object.entries(RESOLUTIONS)) if (Object.entries(r.o).every(([d, v]) => JSON.stringify(this.overrides[d]) === JSON.stringify(v)) &&
      [...RES_KEYS].filter((d) => !(d in r.o)).every((d) => !(d in this.overrides))) return k;
    return 'custom';
  }
  setResolution(key) { RES_KEYS.forEach((d) => delete this.overrides[d]); Object.entries(RESOLUTIONS[key].o).forEach(([d, v]) => this.override(d, v)); }
  payload() { return { profile: this.profile, overrides: { ...this.overrides }, stages: this.stageList }; }
  summary() {
    const lines = [`profile: ${this.profile}`];
    if (this.stageList) lines.push(`stages: [${this.stageList.join(', ')}]`);
    for (const [k, v] of Object.entries(this.overrides)) lines.push(`${k}: ${JSON.stringify(v)}`);
    return lines.join('\n');
  }
}

function control(o, section, f) {
  const dotted = section ? `${section}.${f.key}` : f.key;
  const v = o.eff(section, f.key);
  const id = `f_${dotted.replace(/\./g, '__')}`;
  if (f.type === 'bool') return `<label class="sw"><input type="checkbox" id="${id}" ${v ? 'checked' : ''}><span></span></label>`;
  if (f.choices) return `<select id="${id}">${(f.optional ? ['(none)'] : []).concat(f.choices).map((c) => `<option ${c === v || (c === '(none)' && v == null) ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select>`;
  if (f.type === 'int' || f.type === 'float') return `<input type="number" id="${id}" step="${f.type === 'int' ? 1 : 'any'}" value="${v ?? ''}" placeholder="${f.optional ? 'none' : ''}">`;
  if (f.type.startsWith('list')) return `<input type="text" id="${id}" value="${esc((v || []).join(', '))}" placeholder="comma-separated">`;
  if (f.type === 'dict') return `<input type="text" id="${id}" value="${esc(JSON.stringify(v || {}))}" placeholder='{"key": value}'>`;
  return `<input type="text" id="${id}" value="${esc(v ?? '')}" placeholder="${f.optional ? 'none' : ''}">`;
}

function parse(f, el) {
  if (f.type === 'bool') return el.checked;
  const raw = el.value.trim();
  if (f.choices) return raw === '(none)' ? null : raw;
  if (raw === '' && f.optional) return null;
  if (f.type === 'int') return raw === '' ? f.default : parseInt(raw, 10);
  if (f.type === 'float') return raw === '' ? f.default : parseFloat(raw);
  if (f.type === 'dict') { try { return JSON.parse(raw || '{}'); } catch { return f.default; } }
  if (f.type.startsWith('list')) {
    const items = raw ? raw.split(',').map((x) => x.trim()).filter(Boolean) : [];
    return f.type === 'list[int]' ? items.map((x) => parseInt(x, 10)) : f.type === 'list[float]' ? items.map(parseFloat) : items;
  }
  return raw;
}

export function renderOptions(el, o) {
  const skip = new Set(['run_name', 'stages']);
  const groups = [{ key: null, title: 'Run', fields: o.schema.top.filter((f) => !skip.has(f.key)) }, ...o.schema.sections];
  const q = o.filter, stages = new Set(o.stages());
  const open = new Set($$('.opt-sec[open]', el).map((d) => d.dataset.k));
  el.innerHTML = groups.map((g) => {
    const fields = g.fields.filter((f) => {
      const dotted = g.key ? `${g.key}.${f.key}` : f.key;
      if (g.key === 'ingest' && ['video', 'telemetry', 'telemetry_offset_s'].includes(f.key)) return false;
      if (o.modifiedOnly && !(dotted in o.overrides)) return false;
      return !q || dotted.toLowerCase().includes(q) || (f.help || '').toLowerCase().includes(q);
    });
    if (!fields.length) return '';
    const mods = fields.filter((f) => (g.key ? `${g.key}.${f.key}` : f.key) in o.overrides).length;
    const stage = SECTION_STAGE[g.key];
    const unused = stage && !stages.has(stage) && !(g.key === 'metrics' && stages.has('report'));
    const isOpen = q || o.modifiedOnly || open.has(g.key || 'run');
    return `<details class="opt-sec fold ${unused ? 'unused' : ''}" data-k="${esc(g.key || 'run')}" ${isOpen ? 'open' : ''}>
      <summary><span class="t">${esc(g.key || 'run')}</span><span class="dsc">${esc(g.title)}${unused ? ' · not run by this profile' : ''}</span>${mods ? `<span class="cnt">${mods} changed</span>` : ''}</summary><div class="body">` +
      fields.map((f) => {
        const dotted = g.key ? `${g.key}.${f.key}` : f.key;
        const mod = dotted in o.overrides;
        const pv = o.profileValue(g.key, f.key);
        const base = pv !== undefined ? `${o.profile}: ${JSON.stringify(pv)}` : `default: ${JSON.stringify(f.default)}`;
        return `<div class="field ${mod ? 'mod' : ''}"><label for="f_${dotted.replace(/\./g, '__')}">${esc(f.key)}</label>
          <div class="ctl">${control(o, g.key, f)}${mod ? `<button class="linkbtn" data-reset="${esc(dotted)}">reset</button>` : ''}</div>
          <div class="help">${esc(f.help)}${f.help ? ' · ' : ''}<span class="faint">${esc(base)}</span></div></div>`;
      }).join('') + '</div></details>';
  }).join('') || '<span class="note">No option matches.</span>';
  for (const g of groups) for (const f of g.fields) {
    const dotted = g.key ? `${g.key}.${f.key}` : f.key;
    const input = document.getElementById(`f_${dotted.replace(/\./g, '__')}`);
    if (input) input.addEventListener('change', () => { o.override(dotted, parse(f, input)); renderOptions(el, o); o.onChange(); });
  }
  $$('[data-reset]', el).forEach((b) => b.addEventListener('click', () => { delete o.overrides[b.dataset.reset]; renderOptions(el, o); o.onChange(); }));
}

// Profile cards, resolution presets, module switches and stage chips.
export function renderProfiles(el, o, onPick) {
  el.innerHTML = o.profiles.map((p) => `<div class="choice ${p.name === o.profile ? 'sel' : ''}" data-n="${esc(p.name)}">
    <div class="prof-name">${esc(p.name)}</div><div class="d">${esc((p.description || '').slice(0, 190))}${(p.description || '').length > 190 ? '…' : ''}</div></div>`).join('');
  $$('.choice', el).forEach((d) => d.addEventListener('click', () => { o.setProfile(d.dataset.n); onPick(); }));
}

export function renderResolution(el, o, onPick) {
  const cur = o.resolution();
  el.innerHTML = `<div class="seg">${Object.entries(RESOLUTIONS).map(([k, r]) => `<button data-r="${k}" class="${k === cur ? 'on' : ''}">${r.label}</button>`).join('')}${cur === 'custom' ? '<button class="on" disabled>Custom</button>' : ''}</div>
    <div class="note" style="margin-top:8px">${esc(RESOLUTIONS[cur]?.note || 'set through the options below')}</div>`;
  $$('[data-r]', el).forEach((b) => b.addEventListener('click', () => { o.setResolution(b.dataset.r); onPick(); }));
}

export function renderModules(el, o, onPick) {
  el.innerHTML = MODULES.map((m) => `<div class="toggle-row"><div class="grow"><div class="t">${esc(m.title)}</div><div class="d">${esc(m.desc)}</div></div>
    <label class="sw"><input type="checkbox" data-m="${m.key}" ${m.get(o) ? 'checked' : ''}><span></span></label></div>`).join('');
  $$('[data-m]', el).forEach((c) => c.addEventListener('change', () => { MODULES.find((m) => m.key === c.dataset.m).set(o, c.checked); onPick(); }));
}

export function renderStages(el, o, onPick) {
  const on = new Set(o.stages());
  el.innerHTML = o.schema.stages.map((s) => `<span class="chip ${on.has(s) ? 'on' : ''}" data-s="${s}">${s}</span>`).join('');
  $$('.chip', el).forEach((c) => c.addEventListener('click', () => { o.toggleStage(c.dataset.s, !on.has(c.dataset.s)); onPick(); }));
}
