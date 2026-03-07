import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from 'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js';
import { buildPuzzleMjcf, generatePuzzleSpec } from './puzzle-generator.js';
import { clamp, polylineProgress } from './math.js';

const UI = {
  canvasWrap: document.getElementById('canvasWrap'),
  statusBadge: document.getElementById('statusBadge'),
  complexity: document.getElementById('complexity'),
  complexityValue: document.getElementById('complexityValue'),
  seedInput: document.getElementById('seedInput'),
  randomizeSeed: document.getElementById('randomizeSeed'),
  generateBtn: document.getElementById('generateBtn'),
  resetBtn: document.getElementById('resetBtn'),
  pauseBtn: document.getElementById('pauseBtn'),
  shareBtn: document.getElementById('shareBtn'),
  selfTestBtn: document.getElementById('selfTestBtn'),
  downloadMjcfBtn: document.getElementById('downloadMjcfBtn'),
  guideToggle: document.getElementById('guideToggle'),
  followToggle: document.getElementById('followToggle'),
  turboHintToggle: document.getElementById('turboHintToggle'),
  turnCount: document.getElementById('turnCount'),
  gateCount: document.getElementById('gateCount'),
  wireLength: document.getElementById('wireLength'),
  progressValue: document.getElementById('progressValue'),
  timerValue: document.getElementById('timerValue'),
  seedEcho: document.getElementById('seedEcho'),
  modeValue: document.getElementById('modeValue'),
  diagnosticSummary: document.getElementById('diagnosticSummary'),
  diagnosticLog: document.getElementById('diagnosticLog'),
  overlayMessage: document.getElementById('overlayMessage'),
};

class CapsuleGeometry extends THREE.BufferGeometry {
  constructor(radius = 1, length = 1, capSegments = 8, radialSegments = 16) {
    const path = new THREE.Path();
    path.absarc(0, -length / 2, radius, Math.PI * 1.5, 0, false);
    path.absarc(0, length / 2, radius, 0, Math.PI * 0.5, false);
    const latheGeometry = new THREE.LatheGeometry(path.getPoints(capSegments), radialSegments);
    super();
    this.setIndex(latheGeometry.getIndex());
    this.setAttribute('position', latheGeometry.getAttribute('position'));
    this.setAttribute('normal', latheGeometry.getAttribute('normal'));
    this.setAttribute('uv', latheGeometry.getAttribute('uv'));
    this.type = 'CapsuleGeometry';
    this.parameters = { radius, length, capSegments, radialSegments };
  }
}

class MujocoThreeBridge {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x05070e);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-1.8, -3.3, 1.7);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);

    this.simGroup = new THREE.Group();
    this.guideGroup = new THREE.Group();
    this.scene.add(this.simGroup);
    this.scene.add(this.guideGroup);

    this.meshes = [];
    this.geometryCache = new Map();

    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    this.scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(2.2, -2.8, 4.8);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    this.scene.add(keyLight);

    const rimLight = new THREE.PointLight(0x6fd4ff, 1.0, 14, 2);
    rimLight.position.set(-2.5, 2.0, 2.5);
    this.scene.add(rimLight);

    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  clearGuide() {
    while (this.guideGroup.children.length > 0) {
      const child = this.guideGroup.children[0];
      this.guideGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }
  }

  setGuide(points) {
    this.clearGuide();
    const vectors = points.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    const geometry = new THREE.BufferGeometry().setFromPoints(vectors);
    const material = new THREE.LineDashedMaterial({ color: 0x63d7ff, dashSize: 0.08, gapSize: 0.05, transparent: true, opacity: 0.65 });
    const line = new THREE.Line(geometry, material);
    line.computeLineDistances();
    this.guideGroup.add(line);
  }

  setGuideVisible(visible) {
    this.guideGroup.visible = visible;
  }

  clearSimulationMeshes() {
    this.meshes.forEach((mesh) => {
      if (mesh.parent) mesh.parent.remove(mesh);
      if (mesh.material) {
        if (Array.isArray(mesh.material)) {
          mesh.material.forEach((material) => material.dispose());
        } else {
          mesh.material.dispose();
        }
      }
    });
    this.meshes.length = 0;
    this.geometryCache.forEach((geometry) => geometry.dispose());
    this.geometryCache.clear();
  }

  getGeometry(mujoco, mjvGeom) {
    const key = JSON.stringify([
      mjvGeom.type,
      Array.from(mjvGeom.size || []),
      mjvGeom.dataid,
    ]);

    if (this.geometryCache.has(key)) {
      return { key, geometry: this.geometryCache.get(key) };
    }

    let geometry;
    if (mjvGeom.type === mujoco.mjtGeom.mjGEOM_PLANE.value) {
      geometry = new THREE.PlaneGeometry(
        2 * (mjvGeom.size[0] || 1000),
        2 * (mjvGeom.size[1] || 1000),
      );
      const uv = geometry.getAttribute('uv');
      for (let i = 0; i < uv.count; i += 1) {
        uv.setY(i, 1 - uv.getY(i));
      }
    } else if (mjvGeom.type === mujoco.mjtGeom.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(mjvGeom.size[0], 24, 16);
    } else if (mjvGeom.type === mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
      geometry = new CapsuleGeometry(mjvGeom.size[0], 2 * mjvGeom.size[2], 12, 16);
      geometry.rotateX(0.5 * Math.PI);
    } else if (mjvGeom.type === mujoco.mjtGeom.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(2 * mjvGeom.size[0], 2 * mjvGeom.size[1], 2 * mjvGeom.size[2]);
    } else if (mjvGeom.type === mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(mjvGeom.size[0], mjvGeom.size[0], 2 * mjvGeom.size[2], 24);
      geometry.rotateX(0.5 * Math.PI);
    } else {
      geometry = new THREE.SphereGeometry(0.01, 8, 8);
    }

    this.geometryCache.set(key, geometry);
    return { key, geometry };
  }

  sync(mujoco, model, data, mjvScene, mjvOption, mjvPerturb, mjvCamera) {
    mujoco.mjv_updateScene(
      model,
      data,
      mjvOption,
      mjvPerturb,
      mjvCamera,
      mujoco.mjtCatBit.mjCAT_ALL.value,
      mjvScene,
    );

    const geoms = mjvScene.geoms;
    const count = geoms.size();

    for (let i = 0; i < count; i += 1) {
      const mjvGeom = geoms.get(i);
      if (!mjvGeom) continue;

      const { key, geometry } = this.getGeometry(mujoco, mjvGeom);
      let mesh = this.meshes[i];
      if (!mesh || mesh.userData.key !== key) {
        if (mesh) {
          this.simGroup.remove(mesh);
          if (mesh.material) mesh.material.dispose();
        }
        const material = new THREE.MeshPhongMaterial({
          color: new THREE.Color(mjvGeom.rgba[0], mjvGeom.rgba[1], mjvGeom.rgba[2]),
          transparent: mjvGeom.rgba[3] < 0.999,
          opacity: mjvGeom.rgba[3],
          side: mjvGeom.type === mujoco.mjtGeom.mjGEOM_PLANE.value ? THREE.DoubleSide : THREE.FrontSide,
        });
        mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData.key = key;
        this.meshes[i] = mesh;
        this.simGroup.add(mesh);
      }

      mesh.material.color.setRGB(mjvGeom.rgba[0], mjvGeom.rgba[1], mjvGeom.rgba[2]);
      mesh.material.opacity = mjvGeom.rgba[3];
      mesh.material.transparent = mjvGeom.rgba[3] < 0.999;
      mesh.matrixAutoUpdate = false;
      mesh.matrix.set(
        mjvGeom.mat[0], mjvGeom.mat[1], mjvGeom.mat[2], mjvGeom.pos[0],
        mjvGeom.mat[3], mjvGeom.mat[4], mjvGeom.mat[5], mjvGeom.pos[1],
        mjvGeom.mat[6], mjvGeom.mat[7], mjvGeom.mat[8], mjvGeom.pos[2],
        0, 0, 0, 1,
      );
      mesh.matrixWorldNeedsUpdate = true;
      mjvGeom.delete();
    }

    for (let i = count; i < this.meshes.length; i += 1) {
      const mesh = this.meshes[i];
      if (!mesh) continue;
      if (mesh.parent) mesh.parent.remove(mesh);
      if (mesh.material) mesh.material.dispose();
    }
    this.meshes.length = count;
    geoms.delete();
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.clearGuide();
    this.clearSimulationMeshes();
    this.controls.dispose();
    this.renderer.dispose();
  }
}

function formatTime(seconds) {
  const whole = Math.floor(seconds);
  const minutes = Math.floor(whole / 60);
  const secs = whole % 60;
  const dec = Math.floor((seconds - whole) * 10);
  return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${dec}`;
}

function bodyQuaternionFromQpos(qpos) {
  return new THREE.Quaternion(qpos[4], qpos[5], qpos[6], qpos[3]);
}

function safeErrorString(error) {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }
  return String(error);
}

function buildMinimalDiagnosticMjcf() {
  return `
<mujoco model="diagnostic_minimal">
  <option gravity="0 0 0" timestep="0.002"/>
  <worldbody>
    <body name="probe" pos="0 0 0">
      <joint name="probe_free" type="free"/>
      <geom name="probe_geom" type="sphere" size="0.08" rgba="0.2 0.8 1 1"/>
    </body>
  </worldbody>
</mujoco>
`.trim();
}

function buildCompatibilityMjcf(xml) {
  return xml
    .replace(/<compiler[^>]*autolimits="true"[^>]*\/>/, '<compiler angle="radian" inertiafromgeom="true"/>')
    .replace(/<option[^>]*integrator="implicitfast"[^>]*\/>/, '<option timestep="0.0025" gravity="0 0 0" iterations="70"/>')
    .replace(/\s*<visual>[\s\S]*?<\/visual>/, '');
}

function browserXmlError(xml) {
  const doc = new DOMParser().parseFromString(xml, 'application/xml');
  const parseError = doc.querySelector('parsererror');
  return parseError ? parseError.textContent?.trim() || 'Unknown XML parser error.' : null;
}

function triggerTextDownload(filename, text) {
  const blob = new Blob([text], { type: 'application/xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export class PuzzleApp {
  constructor() {
    this.viewer = new MujocoThreeBridge(UI.canvasWrap);
    this.mujoco = null;
    this.model = null;
    this.data = null;
    this.mjvScene = null;
    this.mjvOption = null;
    this.mjvPerturb = null;
    this.mjvCamera = null;

    this.currentSpec = null;
    this.initialQpos = null;
    this.initialQvel = null;
    this.playerBodyId = 1;

    this.keyState = Object.create(null);
    this.lastFrameTime = performance.now();
    this.accumulator = 0;
    this.physicsStep = 0.0025;
    this.elapsed = 0;
    this.paused = true;
    this.solved = false;
    this.ready = false;
    this.currentProgress = 0;
    this.lastXml = '';
    this.lastCompatibilityXml = '';
    this.diagnosticReport = null;

    this.handleKeyDown = this.handleKeyDown.bind(this);
    this.handleKeyUp = this.handleKeyUp.bind(this);
    this.animate = this.animate.bind(this);
  }

  async init() {
    this.bindUi();
    this.bindKeyboard();
    this.readUrlState();
    this.setMode('初期化中');
    try {
      this.mujoco = await loadMujoco();
      this.mujoco.FS.mkdir('/working');
      this.mujoco.FS.mount(this.mujoco.MEMFS, { root: '.' }, '/working');
      this.ready = true;
      this.setStatus('MuJoCo Ready', 'ready');
      this.setDiagnosticSummary('待機中');
      this.setDiagnosticLog(`User-Agent: ${navigator.userAgent}
MjModel.loadFromXML: ${typeof this.mujoco.MjModel?.loadFromXML}
MjModel.mj_loadXML: ${typeof this.mujoco.MjModel?.mj_loadXML}`);
      await this.generateFromUi();
      requestAnimationFrame(this.animate);
    } catch (error) {
      console.error(error);
      this.setStatus('Load Error', 'error');
      this.showOverlay(`MuJoCo の初期化に失敗しました。\n\n${error instanceof Error ? error.message : String(error)}`, 'error', false);
      this.setMode('エラー');
    }
  }

  bindUi() {
    UI.complexity.addEventListener('input', () => {
      UI.complexityValue.textContent = UI.complexity.value;
    });

    UI.randomizeSeed.addEventListener('click', () => {
      UI.seedInput.value = String(Math.floor(Math.random() * 2_147_483_647));
    });

    UI.generateBtn.addEventListener('click', async () => {
      await this.generateFromUi();
    });

    UI.resetBtn.addEventListener('click', () => this.resetPuzzle());
    UI.pauseBtn.addEventListener('click', () => this.togglePause());

    UI.guideToggle.addEventListener('change', () => {
      this.viewer.setGuideVisible(UI.guideToggle.checked);
    });

    UI.shareBtn.addEventListener('click', async () => {
      const url = this.buildShareUrl();
      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(url);
          this.showOverlay('現在のパズル URL をクリップボードにコピーしました。', 'info', true);
        } else {
          window.prompt('URL をコピーしてください', url);
        }
      } catch (error) {
        console.warn(error);
        window.prompt('URL をコピーしてください', url);
      }
    });

    UI.selfTestBtn?.addEventListener('click', async () => {
      await this.runSelfTest({ includePuzzle: true, includeCompatibility: true, reason: 'manual' });
    });

    UI.downloadMjcfBtn?.addEventListener('click', () => {
      this.downloadCurrentMjcf();
    });
  }

  bindKeyboard() {
    window.addEventListener('keydown', this.handleKeyDown);
    window.addEventListener('keyup', this.handleKeyUp);
    window.addEventListener('blur', () => {
      this.keyState = Object.create(null);
    });
    window.addEventListener('beforeunload', () => this.dispose());
  }

  handleKeyDown(event) {
    const tag = document.activeElement?.tagName?.toLowerCase();
    const typing = tag === 'input' || tag === 'textarea';

    if (!typing) {
      this.keyState[event.code] = true;
    }

    if (event.code === 'Space' && !typing) {
      event.preventDefault();
      this.togglePause();
    } else if (event.code === 'Backspace' && !typing) {
      event.preventDefault();
      this.resetPuzzle();
    } else if (event.code === 'KeyG' && !typing) {
      UI.guideToggle.checked = !UI.guideToggle.checked;
      this.viewer.setGuideVisible(UI.guideToggle.checked);
    } else if (event.code === 'KeyC' && !typing) {
      this.recenterCamera();
    }
  }

  handleKeyUp(event) {
    this.keyState[event.code] = false;
  }

  readUrlState() {
    const params = new URLSearchParams(window.location.search);
    const complexity = Number(params.get('complexity'));
    const seed = Number(params.get('seed'));
    if (Number.isFinite(complexity)) {
      UI.complexity.value = String(clamp(Math.round(complexity), 1, 10));
      UI.complexityValue.textContent = UI.complexity.value;
    }
    if (Number.isFinite(seed)) {
      UI.seedInput.value = String(Math.trunc(seed));
    }
  }

  buildShareUrl() {
    const url = new URL(window.location.href);
    url.searchParams.set('complexity', UI.complexity.value);
    url.searchParams.set('seed', String(Math.trunc(Number(UI.seedInput.value) || 0)));
    return url.toString();
  }

  writeUrlState() {
    const url = this.buildShareUrl();
    window.history.replaceState({}, '', url);
  }

  setStatus(text, mode = 'loading') {
    UI.statusBadge.textContent = text;
    UI.statusBadge.className = `badge ${mode}`;
  }

  setMode(text) {
    UI.modeValue.textContent = text;
  }

  setDiagnosticSummary(text) {
    if (UI.diagnosticSummary) {
      UI.diagnosticSummary.textContent = text;
    }
  }

  setDiagnosticLog(text) {
    if (UI.diagnosticLog) {
      UI.diagnosticLog.textContent = text;
    }
  }

  exposeDiagnostics(extra = {}) {
    window.__chieNoWaDiagnostics = {
      timestamp: new Date().toISOString(),
      summary: UI.diagnosticSummary?.textContent ?? '',
      report: this.diagnosticReport,
      lastXml: this.lastXml,
      lastCompatibilityXml: this.lastCompatibilityXml,
      ...extra,
    };
  }

  getAvailableLoaders() {
    const loaders = [];
    const mjModelClass = this.mujoco?.MjModel;
    if (typeof mjModelClass?.loadFromXML === 'function') {
      loaders.push({
        name: 'MjModel.loadFromXML',
        fn: (path) => mjModelClass.loadFromXML(path),
      });
    }
    if (typeof mjModelClass?.mj_loadXML === 'function') {
      loaders.push({
        name: 'MjModel.mj_loadXML',
        fn: (path) => mjModelClass.mj_loadXML(path),
      });
    }
    return loaders;
  }

  compileModelFromXml(xml, label = 'model') {
    const parseError = browserXmlError(xml);
    if (parseError) {
      throw new Error(`Browser XML parser error (${label}): ${parseError}`);
    }

    const path = `/working/${label}.xml`;
    this.mujoco.FS.writeFile(path, xml);
    const loaders = this.getAvailableLoaders();
    if (loaders.length === 0) {
      throw new Error('No MuJoCo XML loader was found. Expected MjModel.loadFromXML or MjModel.mj_loadXML.');
    }

    const attempts = [];
    for (const loader of loaders) {
      try {
        const model = loader.fn(path);
        if (model) {
          return { model, loader: loader.name, path };
        }
        attempts.push(`${loader.name}: returned null`);
      } catch (error) {
        attempts.push(`${loader.name}: ${safeErrorString(error)}`);
      }
    }

    throw new Error(`All MuJoCo XML loaders failed for ${label}.
${attempts.join('
')}`);
  }

  probeModelObjects(model, sceneCap = 4096) {
    let data = null;
    let scene = null;
    let option = null;
    let perturb = null;
    let camera = null;
    try {
      data = new this.mujoco.MjData(model);
      scene = new this.mujoco.MjvScene(model, sceneCap);
      option = new this.mujoco.MjvOption();
      perturb = new this.mujoco.MjvPerturb();
      camera = new this.mujoco.MjvCamera();
      this.mujoco.mj_forward(model, data);
      return { ok: true };
    } catch (error) {
      return { ok: false, error: safeErrorString(error) };
    } finally {
      camera?.delete?.();
      perturb?.delete?.();
      option?.delete?.();
      scene?.delete?.();
      data?.delete?.();
    }
  }

  async runSelfTest({ includePuzzle = true, includeCompatibility = true, reason = 'manual' } = {}) {
    if (!this.ready) {
      this.setDiagnosticSummary('MuJoCo 未初期化');
      this.setDiagnosticLog('MuJoCo の初期化前のため自己診断を実行できません。');
      return null;
    }

    const lines = [];
    const report = {
      reason,
      userAgent: navigator.userAgent,
      loaders: this.getAvailableLoaders().map((loader) => loader.name),
      minimal: null,
      puzzle: null,
      compatibility: null,
    };

    const runCase = (name, xml) => {
      const info = { ok: false, compile: null, scene: null, error: null, xmlLength: xml.length };
      try {
        const compiled = this.compileModelFromXml(xml, name);
        info.compile = compiled.loader;
        const sceneProbe = this.probeModelObjects(compiled.model, 4096);
        info.scene = sceneProbe;
        info.ok = sceneProbe.ok;
        if (!sceneProbe.ok) {
          info.error = sceneProbe.error;
        }
        compiled.model.delete();
      } catch (error) {
        info.error = safeErrorString(error);
      }
      return info;
    };

    report.minimal = runCase('diagnostic_minimal', buildMinimalDiagnosticMjcf());

    lines.push(`[reason] ${reason}`);
    lines.push(`[user-agent] ${navigator.userAgent}`);
    lines.push(`[loaders] ${report.loaders.length ? report.loaders.join(', ') : '(none)'}`);
    lines.push(`[minimal] ok=${report.minimal.ok} compile=${report.minimal.compile ?? '-'} scene=${report.minimal.scene?.ok ?? '-'} error=${report.minimal.error ?? '-'}`);

    if (includePuzzle) {
      const seed = Math.trunc(Number(UI.seedInput.value) || 0) || 1;
      const complexity = clamp(Math.round(Number(UI.complexity.value) || 5), 1, 10);
      const spec = this.currentSpec ?? generatePuzzleSpec({ seed, complexity });
      const xml = this.lastXml || buildPuzzleMjcf(spec);
      this.lastXml = xml;
      report.puzzle = runCase('diagnostic_puzzle', xml);
      lines.push(`[puzzle] ok=${report.puzzle.ok} compile=${report.puzzle.compile ?? '-'} scene=${report.puzzle.scene?.ok ?? '-'} error=${report.puzzle.error ?? '-'}`);

      if (includeCompatibility) {
        this.lastCompatibilityXml = buildCompatibilityMjcf(xml);
        report.compatibility = runCase('diagnostic_compatibility', this.lastCompatibilityXml);
        lines.push(`[compatibility] ok=${report.compatibility.ok} compile=${report.compatibility.compile ?? '-'} scene=${report.compatibility.scene?.ok ?? '-'} error=${report.compatibility.error ?? '-'}`);
      }
    }

    let summary = '自己診断 OK';
    if (!report.minimal.ok) {
      summary = 'MuJoCo 本体/API 側が怪しい';
    } else if (report.puzzle && !report.puzzle.ok && report.compatibility?.ok) {
      summary = '生成 MJCF と MuJoCo 版の互換性が怪しい';
    } else if (report.puzzle && !report.puzzle.ok && report.puzzle.scene && !report.puzzle.scene.ok) {
      summary = 'MJCF ではなく scene 初期化が怪しい';
    } else if (report.puzzle && !report.puzzle.ok) {
      summary = '生成 MJCF 側が怪しい';
    }

    this.diagnosticReport = report;
    this.setDiagnosticSummary(summary);
    this.setDiagnosticLog(lines.join('
'));
    this.exposeDiagnostics({ reason, report });
    return report;
  }

  downloadCurrentMjcf() {
    if (!this.lastXml) {
      this.showOverlay('まだ MJCF が生成されていません。先にパズルを生成してください。', 'error', true);
      return;
    }
    const seed = Math.trunc(Number(UI.seedInput.value) || 0) || 1;
    const complexity = clamp(Math.round(Number(UI.complexity.value) || 5), 1, 10);
    triggerTextDownload(`chie_no_wa_seed${seed}_c${complexity}.xml`, this.lastXml);
    this.showOverlay('現在の MJCF を保存しました。', 'info', true);
  }

  showOverlay(message, mode = 'info', autoHide = true) {
    UI.overlayMessage.textContent = message;
    UI.overlayMessage.className = `overlay ${mode}`;
    if (autoHide) {
      clearTimeout(this.overlayTimer);
      this.overlayTimer = setTimeout(() => {
        UI.overlayMessage.className = 'overlay hidden';
      }, 2200);
    }
  }

  recenterCamera() {
    if (!this.data) return;
    const qpos = this.data.qpos;
    this.viewer.controls.target.set(qpos[0], qpos[1], qpos[2]);
  }

  disposeSimulation() {
    if (this.mjvScene) {
      this.mjvScene.delete();
      this.mjvScene = null;
    }
    if (this.mjvCamera) {
      this.mjvCamera.delete();
      this.mjvCamera = null;
    }
    if (this.mjvPerturb) {
      this.mjvPerturb.delete();
      this.mjvPerturb = null;
    }
    if (this.mjvOption) {
      this.mjvOption.delete();
      this.mjvOption = null;
    }
    if (this.data) {
      this.data.delete();
      this.data = null;
    }
    if (this.model) {
      this.model.delete();
      this.model = null;
    }
    this.viewer.clearSimulationMeshes();
  }

  dispose() {
    this.disposeSimulation();
    this.viewer.dispose();
    if (this.mujoco) {
      try {
        this.mujoco.FS.unmount('/working');
      } catch {
        // Ignore repeated unmount attempts.
      }
    }
  }

  async generateFromUi() {
    if (!this.ready) return;
    const seed = Math.trunc(Number(UI.seedInput.value) || 0) || 1;
    const complexity = clamp(Math.round(Number(UI.complexity.value) || 5), 1, 10);

    UI.seedInput.value = String(seed);
    UI.complexity.value = String(complexity);
    UI.complexityValue.textContent = UI.complexity.value;
    this.writeUrlState();

    this.setMode('生成中');
    this.setStatus('Compiling model…', 'loading');

    try {
      this.disposeSimulation();
      this.currentSpec = generatePuzzleSpec({ seed, complexity });
      const xml = buildPuzzleMjcf(this.currentSpec);
      this.lastXml = xml;
      this.lastCompatibilityXml = buildCompatibilityMjcf(xml);
      const compiled = this.compileModelFromXml(xml, 'puzzle');
      this.model = compiled.model;
      this.data = new this.mujoco.MjData(this.model);
      if (!this.data) {
        throw new Error('Failed to create mjData.');
      }

      this.mjvScene = new this.mujoco.MjvScene(this.model, 4096);
      this.mjvOption = new this.mujoco.MjvOption();
      this.mjvPerturb = new this.mujoco.MjvPerturb();
      this.mjvCamera = new this.mujoco.MjvCamera();

      this.mujoco.mj_forward(this.model, this.data);
      this.initialQpos = Array.from(this.data.qpos);
      this.initialQvel = Array.from(this.data.qvel);
      this.elapsed = 0;
      this.accumulator = 0;
      this.solved = false;
      this.paused = false;
      this.currentProgress = 0;

      this.viewer.setGuide(this.currentSpec.pathPoints);
      this.viewer.setGuideVisible(UI.guideToggle.checked);
      this.recenterCamera();

      UI.seedEcho.textContent = String(seed);
      UI.turnCount.textContent = String(this.currentSpec.stats.turns);
      UI.gateCount.textContent = String(this.currentSpec.stats.gateCount);
      UI.wireLength.textContent = `${this.currentSpec.stats.wireLength.toFixed(2)} m`;
      UI.progressValue.textContent = '0%';
      UI.timerValue.textContent = '00:00.0';
      this.updatePauseButton();
      this.setMode('プレイ中');
      this.setStatus('Ready', 'ready');
      this.setDiagnosticSummary('直近生成 OK');
      this.setDiagnosticLog([
        `[seed] ${seed}`,
        `[complexity] ${complexity}`,
        `[xml-loader] ${compiled.loader}`,
        `[xml-length] ${xml.length}`,
        `[scene-cap] 4096`,
      ].join('\n'));
      this.exposeDiagnostics({
        report: null,
        loader: compiled.loader,
        seed,
        complexity,
        xmlLength: xml.length,
      });
      this.showOverlay('新しいパズルを生成しました。青いリングを右端の出口まで運んでください。', 'info', true);
    } catch (error) {
      console.error(error);
      const report = await this.runSelfTest({ includePuzzle: true, includeCompatibility: true, reason: 'auto-after-generate-error' });
      this.setStatus('Compile Error', 'error');
      this.setMode('エラー');
      const summary = report ? `

診断: ${UI.diagnosticSummary.textContent}` : '';
      this.showOverlay(`パズル生成に失敗しました。

${safeErrorString(error)}${summary}`, 'error', false);
    }
  }

  resetPuzzle() {
    if (!this.data || !this.initialQpos || !this.initialQvel) return;
    this.data.qpos.set(this.initialQpos);
    this.data.qvel.set(this.initialQvel);
    this.data.xfrc_applied.fill(0);
    this.mujoco.mj_forward(this.model, this.data);
    this.elapsed = 0;
    this.accumulator = 0;
    this.solved = false;
    this.paused = false;
    this.currentProgress = 0;
    UI.timerValue.textContent = '00:00.0';
    UI.progressValue.textContent = '0%';
    this.updatePauseButton();
    this.setMode('プレイ中');
    this.showOverlay('初期位置へ戻しました。', 'info', true);
  }

  togglePause() {
    if (!this.data || this.solved) return;
    this.paused = !this.paused;
    this.updatePauseButton();
    this.setMode(this.paused ? '一時停止' : 'プレイ中');
  }

  updatePauseButton() {
    UI.pauseBtn.textContent = this.paused ? '再開' : '一時停止';
  }

  getPlayerPosition() {
    if (!this.data) return new THREE.Vector3();
    const qpos = this.data.qpos;
    return new THREE.Vector3(qpos[0], qpos[1], qpos[2]);
  }

  updateCameraFollow() {
    if (!this.data || !UI.followToggle.checked) return;
    const pos = this.getPlayerPosition();
    this.viewer.controls.target.lerp(pos, 0.14);
  }

  computeWrench() {
    const force = new THREE.Vector3();
    const torque = new THREE.Vector3();

    const cameraForward = new THREE.Vector3();
    this.viewer.camera.getWorldDirection(cameraForward);
    const worldUp = new THREE.Vector3(0, 0, 1);
    const cameraRight = new THREE.Vector3().crossVectors(cameraForward, worldUp);
    if (cameraRight.lengthSq() < 1e-8) {
      cameraRight.set(1, 0, 0);
    } else {
      cameraRight.normalize();
    }
    const cameraUp = new THREE.Vector3().copy(worldUp);

    if (this.keyState.KeyW) force.add(cameraForward);
    if (this.keyState.KeyS) force.addScaledVector(cameraForward, -1);
    if (this.keyState.KeyD) force.add(cameraRight);
    if (this.keyState.KeyA) force.addScaledVector(cameraRight, -1);
    if (this.keyState.KeyR) force.add(cameraUp);
    if (this.keyState.KeyF) force.addScaledVector(cameraUp, -1);

    if (this.keyState.KeyI) torque.add(cameraRight);
    if (this.keyState.KeyK) torque.addScaledVector(cameraRight, -1);
    if (this.keyState.KeyJ) torque.add(cameraUp);
    if (this.keyState.KeyL) torque.addScaledVector(cameraUp, -1);
    if (this.keyState.KeyU) torque.add(cameraForward);
    if (this.keyState.KeyO) torque.addScaledVector(cameraForward, -1);

    if (force.lengthSq() > 0) force.normalize();
    if (torque.lengthSq() > 0) torque.normalize();

    const turboEnabled = UI.turboHintToggle.checked && (this.keyState.ShiftLeft || this.keyState.ShiftRight);
    const forceScale = turboEnabled ? 16 : 9;
    const torqueScale = turboEnabled ? 1.9 : 1.1;

    force.multiplyScalar(forceScale);
    torque.multiplyScalar(torqueScale);

    return {
      force,
      torque,
    };
  }

  applyControls() {
    if (!this.data) return;
    const wrench = this.data.xfrc_applied;
    wrench.fill(0);
    const { force, torque } = this.computeWrench();
    const base = this.playerBodyId * 6;
    wrench[base + 0] = force.x;
    wrench[base + 1] = force.y;
    wrench[base + 2] = force.z;
    wrench[base + 3] = torque.x;
    wrench[base + 4] = torque.y;
    wrench[base + 5] = torque.z;
  }

  updateProgressAndSolved() {
    if (!this.data || !this.currentSpec) return;
    const qpos = this.data.qpos;
    const pos = [qpos[0], qpos[1], qpos[2]];
    const along = polylineProgress(this.currentSpec.pathPoints, pos);
    const progress = clamp(along / this.currentSpec.stats.wireLength, 0, 1.25);
    this.currentProgress = progress;
    UI.progressValue.textContent = `${Math.round(Math.min(progress, 1) * 100)}%`;

    const exit = this.currentSpec.exitPoint;
    const dir = this.currentSpec.exitDir;
    const dx = pos[0] - exit[0];
    const dy = pos[1] - exit[1];
    const dz = pos[2] - exit[2];
    const exitDot = dx * dir[0] + dy * dir[1] + dz * dir[2];

    if (!this.solved && exitDot > this.currentSpec.exitThreshold) {
      this.solved = true;
      this.paused = true;
      this.updatePauseButton();
      this.setMode('クリア');
      this.showOverlay(`クリア！\n\n経過時間: ${formatTime(this.elapsed)}\nSeed: ${this.currentSpec.seed}\n複雑性: ${this.currentSpec.complexity}`, 'solved', false);
    }
  }

  stepSimulation(dt) {
    if (!this.data || this.paused || this.solved) return;
    this.accumulator += Math.min(dt, 0.05);

    let safety = 0;
    while (this.accumulator >= this.physicsStep && safety < 120) {
      this.applyControls();
      this.mujoco.mj_step(this.model, this.data);
      this.elapsed += this.physicsStep;
      this.accumulator -= this.physicsStep;
      safety += 1;
    }

    UI.timerValue.textContent = formatTime(this.elapsed);
    this.updateProgressAndSolved();
  }

  animate(now) {
    const dt = Math.min((now - this.lastFrameTime) / 1000, 0.05);
    this.lastFrameTime = now;

    if (this.data && this.model && this.mjvScene) {
      this.stepSimulation(dt);
      this.updateCameraFollow();
      this.viewer.sync(this.mujoco, this.model, this.data, this.mjvScene, this.mjvOption, this.mjvPerturb, this.mjvCamera);
      this.viewer.render();
    } else {
      this.viewer.render();
    }

    requestAnimationFrame(this.animate);
  }
}
