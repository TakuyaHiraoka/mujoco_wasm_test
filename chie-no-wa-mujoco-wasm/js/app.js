import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from 'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco.js';
import { buildPuzzleMjcf, generatePuzzleSpec } from './puzzle-generator.js';
import {
  buildDiagnosticsBundle,
  buildRuntimeDiagnostics,
  computePoseGeometry,
  downloadText,
  formatRuntimeDiagnostics,
  formatStaticDiagnostics,
  runStaticDiagnostics,
} from './diagnostics.js';
import { clamp } from './math.js';

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
  copyUrlBtn: document.getElementById('copyUrlBtn'),
  debugToggle: document.getElementById('debugToggle'),
  followToggle: document.getElementById('followToggle'),
  turboHintToggle: document.getElementById('turboHintToggle'),
  familyValue: document.getElementById('familyValue'),
  gapRatioValue: document.getElementById('gapRatioValue'),
  difficultyValue: document.getElementById('difficultyValue'),
  wireLengthValue: document.getElementById('wireLengthValue'),
  separationValue: document.getElementById('separationValue'),
  contactValue: document.getElementById('contactValue'),
  timerValue: document.getElementById('timerValue'),
  seedEcho: document.getElementById('seedEcho'),
  modeValue: document.getElementById('modeValue'),
  staticDiagText: document.getElementById('staticDiagText'),
  runtimeDiagText: document.getElementById('runtimeDiagText'),
  logText: document.getElementById('logText'),
  runDiagBtn: document.getElementById('runDiagBtn'),
  downloadDiagBtn: document.getElementById('downloadDiagBtn'),
  downloadSpecBtn: document.getElementById('downloadSpecBtn'),
  downloadMjcfBtn: document.getElementById('downloadMjcfBtn'),
  copyLogBtn: document.getElementById('copyLogBtn'),
  overlayMessage: document.getElementById('overlayMessage'),
};



function summarizeMujocoApi(mujoco) {
  const summary = {
    topLevelKeys: Object.keys(mujoco).filter((k) => !k.startsWith('_')).sort(),
    hasFS: !!mujoco?.FS,
    hasMEMFS: !!mujoco?.MEMFS,
    hasMjModelClass: !!mujoco?.MjModel,
    hasMjDataClass: !!mujoco?.MjData,
    hasStaticMjLoadXml: typeof mujoco?.MjModel?.mj_loadXML === 'function',
    hasLegacyModelCtor: typeof mujoco?.Model === 'function',
    hasStateCtor: typeof mujoco?.State === 'function',
    hasSimulationCtor: typeof mujoco?.Simulation === 'function',
    hasMjStep: typeof mujoco?.mj_step === 'function',
    hasMjForward: typeof mujoco?.mj_forward === 'function',
    hasMjResetData: typeof mujoco?.mj_resetData === 'function',
  };
  return summary;
}

class CapsuleGeometry extends THREE.BufferGeometry {
  constructor(radius = 1, length = 1, capSegments = 8, radialSegments = 16) {
    const path = new THREE.Path();
    path.absarc(0, -length / 2, radius, Math.PI * 1.5, 0, false);
    path.absarc(0, length / 2, radius, 0, Math.PI * 0.5, false);
    const lathe = new THREE.LatheGeometry(path.getPoints(capSegments), radialSegments);
    super();
    this.setIndex(lathe.getIndex());
    this.setAttribute('position', lathe.getAttribute('position'));
    this.setAttribute('normal', lathe.getAttribute('normal'));
    this.setAttribute('uv', lathe.getAttribute('uv'));
    this.type = 'CapsuleGeometry';
  }
}

function setLinePositions(line, a, b) {
  const array = line.geometry.attributes.position.array;
  array[0] = a.x;
  array[1] = a.y;
  array[2] = a.z;
  array[3] = b.x;
  array[4] = b.y;
  array[5] = b.z;
  line.geometry.attributes.position.needsUpdate = true;
  line.geometry.computeBoundingSphere();
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
    this.camera.position.set(-1.9, -3.4, 1.9);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);

    this.simGroup = new THREE.Group();
    this.debugGroup = new THREE.Group();
    this.scene.add(this.simGroup);
    this.scene.add(this.debugGroup);

    this.meshes = [];
    this.geometryCache = new Map();
    this.debugObjects = null;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.58));

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.18);
    keyLight.position.set(2.3, -2.8, 4.8);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    this.scene.add(keyLight);

    const rimLight = new THREE.PointLight(0x69d7ff, 1.0, 16, 2);
    rimLight.position.set(-2.4, 2.2, 2.5);
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

  getGeometry(mujoco, mjvGeom) {
    const key = JSON.stringify([
      mjvGeom.type,
      Array.from(mjvGeom.size || []),
      mjvGeom.dataid,
    ]);

    if (this.geometryCache.has(key)) {
      return this.geometryCache.get(key);
    }

    let geometry;
    if (mjvGeom.type === mujoco.mjtGeom.mjGEOM_PLANE.value) {
      geometry = new THREE.PlaneGeometry(2 * (mjvGeom.size[0] || 1000), 2 * (mjvGeom.size[1] || 1000));
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
    return geometry;
  }

  clearSimulationMeshes() {
    this.meshes.forEach((mesh) => {
      if (mesh.parent) mesh.parent.remove(mesh);
      if (mesh.material) mesh.material.dispose();
    });
    this.meshes.length = 0;
    this.geometryCache.forEach((geometry) => geometry.dispose());
    this.geometryCache.clear();
  }

  clearDebug() {
    while (this.debugGroup.children.length > 0) {
      const child = this.debugGroup.children[0];
      this.debugGroup.remove(child);
      child.traverse?.((node) => {
        if (node.geometry) node.geometry.dispose();
        if (node.material) {
          if (Array.isArray(node.material)) node.material.forEach((m) => m.dispose());
          else node.material.dispose();
        }
      });
    }
    this.debugObjects = null;
  }

  setDebugVisible(visible) {
    this.debugGroup.visible = visible;
  }

  setDebugMarkers(spec) {
    this.clearDebug();
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));

    const fixedCenter = new THREE.Mesh(
      new THREE.SphereGeometry(spec.wire.radius * 0.55, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xf5f5f5 }),
    );
    fixedCenter.position.set(0, 0, 0);

    const movingCenter = new THREE.Mesh(
      new THREE.SphereGeometry(spec.wire.radius * 0.55, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0x3be4ff }),
    );

    const fixedGapLine = new THREE.Line(
      lineGeometry.clone(),
      new THREE.LineBasicMaterial({ color: 0xffb74d }),
    );

    const movingGapLine = new THREE.Line(
      lineGeometry.clone(),
      new THREE.LineBasicMaterial({ color: 0xff4fc9 }),
    );

    const centerLinkLine = new THREE.Line(
      lineGeometry.clone(),
      new THREE.LineDashedMaterial({ color: 0x8ff7b5, dashSize: 0.08, gapSize: 0.05, opacity: 0.72, transparent: true }),
    );

    const fixedArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 0, 0),
      spec.fixed.primaryLoopRadius * 0.34,
      0x7ef6a8,
      spec.wire.radius * 2.6,
      spec.wire.radius * 1.6,
    );

    const movingArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 0),
      spec.moving.primaryLoopRadius * 0.34,
      0x80d8ff,
      spec.wire.radius * 2.6,
      spec.wire.radius * 1.6,
    );

    setLinePositions(
      fixedGapLine,
      new THREE.Vector3(...spec.fixed.gap.aLocal),
      new THREE.Vector3(...spec.fixed.gap.bLocal),
    );

    fixedArrow.position.set(0, 0, 0);
    fixedArrow.setDirection(new THREE.Vector3(0, 0, 1));

    this.debugGroup.add(fixedCenter, movingCenter, fixedGapLine, movingGapLine, centerLinkLine, fixedArrow, movingArrow);

    this.debugObjects = {
      fixedCenter,
      movingCenter,
      fixedGapLine,
      movingGapLine,
      centerLinkLine,
      fixedArrow,
      movingArrow,
    };

    this.updateMovingMarkers(spec, { pos: spec.startPose.pos, quat: spec.startPose.quat });
  }

  updateMovingMarkers(spec, pose) {
    if (!this.debugObjects) return;
    const markers = computePoseGeometry(spec, pose);

    const fixedCenter = new THREE.Vector3(...markers.fixed.holeCenter);
    const movingCenter = new THREE.Vector3(...markers.moving.holeCenter);

    this.debugObjects.fixedCenter.position.copy(fixedCenter);
    this.debugObjects.movingCenter.position.copy(movingCenter);
    setLinePositions(this.debugObjects.centerLinkLine, fixedCenter, movingCenter);
    this.debugObjects.centerLinkLine.computeLineDistances();

    setLinePositions(
      this.debugObjects.movingGapLine,
      new THREE.Vector3(...markers.moving.gapA),
      new THREE.Vector3(...markers.moving.gapB),
    );

    this.debugObjects.movingArrow.position.copy(movingCenter);
    this.debugObjects.movingArrow.setDirection(new THREE.Vector3(...markers.moving.holeNormal).normalize());
    this.debugObjects.movingArrow.setLength(
      spec.moving.primaryLoopRadius * 0.34,
      spec.wire.radius * 2.6,
      spec.wire.radius * 1.6,
    );
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

      const geometry = this.getGeometry(mujoco, mjvGeom);
      let mesh = this.meshes[i];
      if (!mesh || mesh.userData.geometry !== geometry) {
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
        mesh.userData.geometry = geometry;
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
    this.clearSimulationMeshes();
    this.clearDebug();
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
    this.currentMjcf = '';
    this.currentStaticDiagnostics = null;
    this.currentRuntimeDiagnostics = null;
    this.initialQpos = null;
    this.initialQvel = null;
    this.playerBodyId = 1;

    this.keyState = Object.create(null);
    this.eventLog = [];
    this.lastFrameTime = performance.now();
    this.runtimeDiagAccumulator = 0;
    this.accumulator = 0;
    this.physicsStep = 0.0025;
    this.elapsed = 0;
    this.paused = true;
    this.solved = false;
    this.ready = false;

    this.handleKeyDown = this.handleKeyDown.bind(this);
    this.handleKeyUp = this.handleKeyUp.bind(this);
    this.animate = this.animate.bind(this);
  }

  log(message, level = 'info') {
    const stamp = new Date().toLocaleTimeString('ja-JP', { hour12: false });
    const line = `[${stamp}] [${level}] ${message}`;
    this.eventLog.push(line);
    if (this.eventLog.length > 180) this.eventLog.shift();
    UI.logText.textContent = this.eventLog.join('\n');
  }

  async init() {
    this.bindUi();
    this.bindKeyboard();
    this.bindGlobalDebugHooks();
    this.readUrlState();
    this.setMode('初期化中');
    try {
      this.mujoco = await loadMujoco();
      try {
        this.mujoco.FS.mkdir('/working');
      } catch {
        // directory may already exist
      }
      this.mujoco.FS.mount(this.mujoco.MEMFS, { root: '.' }, '/working');
      this.ready = true;
      this.setStatus('MuJoCo Ready', 'ready');
      const apiSummary = summarizeMujocoApi(this.mujoco);
      this.log('MuJoCo WASM を初期化しました。');
      this.log(`MuJoCo API: staticLoad=${apiSummary.hasStaticMjLoadXml}, legacyModel=${apiSummary.hasLegacyModelCtor}, mjData=${apiSummary.hasMjDataClass}, mj_step=${apiSummary.hasMjStep}`);
      window.__mujocoApiSummary = apiSummary;
      await this.generateFromUi();
      this.exposeDebugApi();
      requestAnimationFrame(this.animate);
    } catch (error) {
      console.error(error);
      this.log(`MuJoCo 初期化失敗: ${error instanceof Error ? error.message : String(error)}`, 'error');
      this.setStatus('Load Error', 'error');
      this.showOverlay(`MuJoCo の初期化に失敗しました。\n\n${error instanceof Error ? error.message : String(error)}`, 'error', false);
      this.setMode('エラー');
    }
  }

  bindGlobalDebugHooks() {
    window.addEventListener('error', (event) => {
      this.log(`window.onerror: ${event.message}`, 'error');
    });
    window.addEventListener('unhandledrejection', (event) => {
      this.log(`unhandledrejection: ${String(event.reason)}`, 'error');
    });
  }

  exposeDebugApi() {
    window.ChieNoWaDebug = {
      getSpec: () => this.currentSpec,
      getMjcf: () => this.currentMjcf,
      getStaticDiagnostics: () => this.currentStaticDiagnostics,
      getRuntimeDiagnostics: () => this.currentRuntimeDiagnostics,
      runDiagnostics: () => this.runDiagnosticsNow(),
      downloadDiagnostics: () => this.downloadDiagnostics(),
      downloadSpec: () => this.downloadSpec(),
      downloadMjcf: () => this.downloadMjcf(),
      getLogText: () => this.eventLog.join('\n'),
      reset: () => this.resetPuzzle(),
    };
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

    UI.copyUrlBtn.addEventListener('click', async () => {
      const url = this.buildShareUrl();
      try {
        await navigator.clipboard.writeText(url);
        this.showOverlay('現在の URL をコピーしました。', 'info', true);
      } catch {
        window.prompt('URL をコピーしてください', url);
      }
    });

    UI.debugToggle.addEventListener('change', () => {
      this.viewer.setDebugVisible(UI.debugToggle.checked);
    });

    UI.runDiagBtn.addEventListener('click', () => {
      this.runDiagnosticsNow();
      this.showOverlay('診断を再計算しました。', 'info', true);
    });

    UI.downloadDiagBtn.addEventListener('click', () => this.downloadDiagnostics());
    UI.downloadSpecBtn.addEventListener('click', () => this.downloadSpec());
    UI.downloadMjcfBtn.addEventListener('click', () => this.downloadMjcf());

    UI.copyLogBtn.addEventListener('click', async () => {
      const text = this.eventLog.join('\n');
      try {
        await navigator.clipboard.writeText(text);
        this.showOverlay('ログをコピーしました。', 'info', true);
      } catch {
        window.prompt('ログをコピーしてください', text);
      }
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
    } else if (event.code === 'KeyM' && !typing) {
      UI.debugToggle.checked = !UI.debugToggle.checked;
      this.viewer.setDebugVisible(UI.debugToggle.checked);
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
    window.history.replaceState({}, '', this.buildShareUrl());
  }

  setStatus(text, mode = 'loading') {
    UI.statusBadge.textContent = text;
    UI.statusBadge.className = `badge ${mode}`;
  }

  setMode(text) {
    UI.modeValue.textContent = text;
  }

  showOverlay(message, mode = 'info', autoHide = true) {
    UI.overlayMessage.textContent = message;
    UI.overlayMessage.className = `overlay ${mode}`;
    if (autoHide) {
      clearTimeout(this.overlayTimer);
      this.overlayTimer = setTimeout(() => {
        UI.overlayMessage.className = 'overlay hidden';
      }, 2400);
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
        // ignore repeated unmounts
      }
    }
  }



  loadModelFromXml(xmlPath) {
    if (typeof this.mujoco?.MjModel?.mj_loadXML === 'function') {
      const model = this.mujoco.MjModel.mj_loadXML(xmlPath);
      if (!model) throw new Error('MjModel.mj_loadXML returned null.');
      return model;
    }

    const apiSummary = summarizeMujocoApi(this.mujoco);
    throw new Error(
      `MuJoCo API mismatch: MjModel.mj_loadXML が見つかりません。` +
      ` loadedKeys=${apiSummary.topLevelKeys.slice(0, 24).join(',')}`
    );
  }

  createDataForModel(model) {
    if (typeof this.mujoco?.MjData === 'function') {
      const data = new this.mujoco.MjData(model);
      if (!data) throw new Error('Failed to create mjData.');
      return data;
    }

    const apiSummary = summarizeMujocoApi(this.mujoco);
    throw new Error(
      `MuJoCo API mismatch: MjData constructor が見つかりません。` +
      ` loadedKeys=${apiSummary.topLevelKeys.slice(0, 24).join(',')}`
    );
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
      this.currentStaticDiagnostics = runStaticDiagnostics(this.currentSpec);
      this.currentMjcf = buildPuzzleMjcf(this.currentSpec);

      this.mujoco.FS.writeFile('/working/puzzle.xml', this.currentMjcf);
      this.model = this.loadModelFromXml('/working/puzzle.xml');
      this.data = this.createDataForModel(this.model);

      this.mjvScene = new this.mujoco.MjvScene(this.model, 2 ** 15);
      this.mjvOption = new this.mujoco.MjvOption();
      this.mjvPerturb = new this.mujoco.MjvPerturb();
      this.mjvCamera = new this.mujoco.MjvCamera();

      this.mujoco.mj_forward(this.model, this.data);
      this.initialQpos = Array.from(this.data.qpos);
      this.initialQvel = Array.from(this.data.qvel);
      this.elapsed = 0;
      this.accumulator = 0;
      this.runtimeDiagAccumulator = 0;
      this.solved = false;
      this.paused = false;

      this.viewer.setDebugMarkers(this.currentSpec);
      this.viewer.setDebugVisible(UI.debugToggle.checked);
      this.recenterCamera();
      this.updateSummaryUi();
      this.updatePauseButton();
      this.updateRuntimeUi(true);

      this.setMode('プレイ中');
      this.setStatus('Ready', 'ready');
      this.log(`パズル生成: family=${this.currentSpec.family.id}, seed=${seed}, complexity=${complexity}`);
      this.currentSpec.generationLog.forEach((line) => this.log(`gen: ${line}`));
      this.showOverlay('新しい知恵の輪を生成しました。シアンの可動片を分離してください。', 'info', true);
    } catch (error) {
      console.error(error);
      this.setStatus('Compile Error', 'error');
      this.setMode('エラー');
      this.log(`生成失敗: ${error instanceof Error ? error.message : String(error)}`, 'error');
      this.showOverlay(`パズル生成に失敗しました。\n\n${error instanceof Error ? error.message : String(error)}`, 'error', false);
    }
  }

  updateSummaryUi() {
    if (!this.currentSpec || !this.currentStaticDiagnostics) return;
    UI.familyValue.textContent = this.currentSpec.family.label;
    UI.gapRatioValue.textContent = `${this.currentSpec.stats.avgGapRatio.toFixed(2)}x`;
    UI.difficultyValue.textContent = `${this.currentSpec.stats.estimatedDifficulty}/10`;
    UI.wireLengthValue.textContent = `${this.currentSpec.stats.totalWireLength.toFixed(2)} m`;
    UI.separationValue.textContent = '0%';
    UI.contactValue.textContent = '0';
    UI.timerValue.textContent = '00:00.0';
    UI.seedEcho.textContent = String(this.currentSpec.seed);
    UI.staticDiagText.textContent = formatStaticDiagnostics(this.currentStaticDiagnostics);
  }

  resetPuzzle() {
    if (!this.data || !this.initialQpos || !this.initialQvel) return;
    this.data.qpos.set(this.initialQpos);
    this.data.qvel.set(this.initialQvel);
    this.data.xfrc_applied.fill(0);
    this.mujoco.mj_forward(this.model, this.data);
    this.elapsed = 0;
    this.accumulator = 0;
    this.runtimeDiagAccumulator = 0;
    this.solved = false;
    this.paused = false;
    UI.timerValue.textContent = '00:00.0';
    this.updatePauseButton();
    this.updateRuntimeUi(true);
    this.setMode('プレイ中');
    this.log('パズルを初期状態に戻しました。');
    this.showOverlay('初期位置へ戻しました。', 'info', true);
  }

  togglePause() {
    if (!this.data || this.solved) return;
    this.paused = !this.paused;
    this.updatePauseButton();
    this.setMode(this.paused ? '一時停止' : 'プレイ中');
    this.updateRuntimeUi(true);
  }

  updatePauseButton() {
    UI.pauseBtn.textContent = this.paused ? '再開' : '一時停止';
  }

  getBodyState() {
    if (!this.data) {
      return {
        pos: [0, 0, 0],
        quat: [1, 0, 0, 0],
        linvel: [0, 0, 0],
        angvel: [0, 0, 0],
        ncon: 0,
      };
    }
    const qpos = this.data.qpos;
    const qvel = this.data.qvel;
    return {
      pos: [qpos[0], qpos[1], qpos[2]],
      quat: [qpos[3], qpos[4], qpos[5], qpos[6]],
      linvel: [qvel[0], qvel[1], qvel[2]],
      angvel: [qvel[3], qvel[4], qvel[5]],
      ncon: this.data.ncon ?? 0,
      paused: this.paused,
      solved: this.solved,
    };
  }

  updateCameraFollow() {
    if (!this.data || !UI.followToggle.checked) return;
    const qpos = this.data.qpos;
    const pos = new THREE.Vector3(qpos[0], qpos[1], qpos[2]);
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
    const forceScale = turboEnabled ? 16 : 8.4;
    const torqueScale = turboEnabled ? 1.8 : 1.0;

    force.multiplyScalar(forceScale);
    torque.multiplyScalar(torqueScale);

    return { force, torque };
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

  updateRuntimeUi(force = false) {
    if (!this.currentSpec) return;
    const state = this.getBodyState();
    this.currentRuntimeDiagnostics = buildRuntimeDiagnostics(this.currentSpec, state);
    if (!force && this.runtimeDiagAccumulator < 0.16) return;

    UI.runtimeDiagText.textContent = formatRuntimeDiagnostics(this.currentRuntimeDiagnostics);
    UI.separationValue.textContent = `${Math.round(Math.min(this.currentRuntimeDiagnostics.separation, 1) * 100)}%`;
    UI.contactValue.textContent = String(this.currentRuntimeDiagnostics.contactCount);
    this.viewer.updateMovingMarkers(this.currentSpec, state);
    this.runtimeDiagAccumulator = 0;
  }

  updateSolvedState() {
    if (!this.currentSpec || !this.currentRuntimeDiagnostics || this.solved) return;
    if (
      this.currentRuntimeDiagnostics.centerDistance > this.currentSpec.geometry.solveDistance
      && this.currentRuntimeDiagnostics.contactCount === 0
    ) {
      this.solved = true;
      this.paused = true;
      this.updatePauseButton();
      this.updateRuntimeUi(true);
      this.setMode('クリア');
      this.log(`クリア: ${formatTime(this.elapsed)} seed=${this.currentSpec.seed}`);
      this.showOverlay(`クリア！\n\n経過時間: ${formatTime(this.elapsed)}\nfamily: ${this.currentSpec.family.id}\nseed: ${this.currentSpec.seed}`, 'solved', false);
    }
  }

  runDiagnosticsNow() {
    if (!this.currentSpec) return null;
    this.currentStaticDiagnostics = runStaticDiagnostics(this.currentSpec);
    UI.staticDiagText.textContent = formatStaticDiagnostics(this.currentStaticDiagnostics);
    this.updateRuntimeUi(true);
    this.log('静的診断と実行診断を再計算しました。');
    return {
      static: this.currentStaticDiagnostics,
      runtime: this.currentRuntimeDiagnostics,
    };
  }

  downloadDiagnostics() {
    if (!this.currentSpec) return;
    if (!this.currentStaticDiagnostics) this.currentStaticDiagnostics = runStaticDiagnostics(this.currentSpec);
    if (!this.currentRuntimeDiagnostics) this.updateRuntimeUi(true);
    const bundle = buildDiagnosticsBundle({
      spec: this.currentSpec,
      staticDiag: this.currentStaticDiagnostics,
      runtimeDiag: this.currentRuntimeDiagnostics,
      logLines: this.eventLog,
    });
    downloadText(
      `chie-no-wa-diagnostics-seed-${this.currentSpec.seed}.json`,
      `${JSON.stringify(bundle, null, 2)}\n`,
    );
    this.log('診断 JSON をダウンロードしました。');
  }

  downloadSpec() {
    if (!this.currentSpec) return;
    downloadText(
      `chie-no-wa-spec-seed-${this.currentSpec.seed}.json`,
      `${JSON.stringify(this.currentSpec, null, 2)}\n`,
    );
    this.log('spec JSON をダウンロードしました。');
  }

  downloadMjcf() {
    if (!this.currentMjcf || !this.currentSpec) return;
    downloadText(`chie-no-wa-seed-${this.currentSpec.seed}.xml`, `${this.currentMjcf}\n`);
    this.log('MJCF XML をダウンロードしました。');
  }

  stepSimulation(dt) {
    if (!this.data || this.paused || this.solved) return;
    this.accumulator += Math.min(dt, 0.05);

    let safety = 0;
    while (this.accumulator >= this.physicsStep && safety < 140) {
      this.applyControls();
      this.mujoco.mj_step(this.model, this.data);
      this.elapsed += this.physicsStep;
      this.accumulator -= this.physicsStep;
      safety += 1;
    }

    UI.timerValue.textContent = formatTime(this.elapsed);
  }

  animate(now) {
    const dt = Math.min((now - this.lastFrameTime) / 1000, 0.05);
    this.lastFrameTime = now;
    this.runtimeDiagAccumulator += dt;

    if (this.data && this.model && this.mjvScene) {
      this.stepSimulation(dt);
      this.updateCameraFollow();
      this.updateRuntimeUi(false);
      this.updateSolvedState();
      this.viewer.sync(this.mujoco, this.model, this.data, this.mjvScene, this.mjvOption, this.mjvPerturb, this.mjvCamera);
      this.viewer.render();
    } else {
      this.viewer.render();
    }

    requestAnimationFrame(this.animate);
  }
}
