import * as THREE from 'three';
import { UI } from './ui.js';
import { createI18n } from './i18n.js';
import { MujocoThreeBridge } from './viewer.js';
import {
  createDataForModel,
  loadModelFromXml,
  loadMujocoBindings,
  mountWorkingFilesystem,
  summarizeMujocoApi,
  unmountWorkingFilesystem,
} from './mujoco-loader.js';
import { buildPuzzleMjcf, generatePuzzleSpec, localizeFamily } from './puzzle-generator.js';
import {
  buildDiagnosticsBundle,
  buildRuntimeDiagnostics,
  downloadText,
  formatRuntimeDiagnostics,
  formatStaticDiagnostics,
  runStaticDiagnostics,
} from './diagnostics.js';
import { clamp } from './math.js';

function formatTime(seconds) {
  const whole = Math.floor(seconds);
  const minutes = Math.floor(whole / 60);
  const secs = whole % 60;
  const dec = Math.floor((seconds - whole) * 10);
  return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${dec}`;
}

function formatLogTimestamp(timestamp, locale) {
  const languageTag = locale === 'en' ? 'en-US' : 'ja-JP';
  return new Date(timestamp).toLocaleTimeString(languageTag, { hour12: false });
}

export class PuzzleApp {
  constructor({ locale = 'ja' } = {}) {
    this.i18n = createI18n(locale);
    this.viewer = new MujocoThreeBridge(UI.canvasWrap);

    this.mujoco = null;
    this.mujocoSourceUrl = '';
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
    this.solveConditionHold = 0;

    this.currentStatus = null;
    this.currentMode = null;
    this.overlayTimer = null;

    this.handleKeyDown = this.handleKeyDown.bind(this);
    this.handleKeyUp = this.handleKeyUp.bind(this);
    this.animate = this.animate.bind(this);

    this.i18n.onChange(() => {
      this.refreshLocalizedUi();
      this.writeUrlState();
    });
  }

  get locale() {
    return this.i18n.locale;
  }

  t(path, params = {}) {
    return this.i18n.t(path, params);
  }

  appendLogEntry({ level = 'info', key = null, params = null, text = null }) {
    this.eventLog.push({
      timestamp: Date.now(),
      level,
      key,
      params,
      text,
    });

    if (this.eventLog.length > 180) {
      this.eventLog.shift();
    }

    this.renderLog();
  }

  log(message, level = 'info') {
    this.appendLogEntry({ level, text: message });
  }

  logKey(key, params = {}, level = 'info') {
    this.appendLogEntry({ level, key, params });
  }

  renderLog() {
    if (!this.eventLog.length) {
      UI.logText.textContent = this.t('log.waiting');
      return;
    }

    UI.logText.textContent = this.eventLog
      .map((entry) => {
        const stamp = formatLogTimestamp(entry.timestamp, this.locale);
        const body = entry.key ? this.t(`log.${entry.key}`, entry.params ?? {}) : entry.text;
        return `[${stamp}] [${entry.level}] ${body}`;
      })
      .join('\n');
  }

  setStatus(statusKey, badgeMode = 'loading', params = {}) {
    this.currentStatus = { statusKey, badgeMode, params };
    UI.statusBadge.textContent = this.t(`status.${statusKey}`, params);
    UI.statusBadge.className = `badge ${badgeMode}`;
  }

  setMode(modeKey, params = {}) {
    this.currentMode = { modeKey, params };
    UI.modeValue.textContent = this.t(`mode.${modeKey}`, params);
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

  showOverlayKey(key, params = {}, mode = 'info', autoHide = true) {
    this.showOverlay(this.t(`overlay.${key}`, params), mode, autoHide);
  }

  refreshLocalizedUi() {
    document.documentElement.lang = this.locale;
    this.i18n.apply(document);
    UI.languageSelect.value = this.locale;

    if (this.currentStatus) {
      this.setStatus(
        this.currentStatus.statusKey,
        this.currentStatus.badgeMode,
        this.currentStatus.params,
      );
    } else {
      this.setStatus('loading', 'loading');
    }

    if (this.currentMode) {
      this.setMode(this.currentMode.modeKey, this.currentMode.params);
    } else {
      this.setMode('notStarted');
    }

    this.updatePauseButton();
    this.renderLog();

    if (this.currentSpec && this.currentStaticDiagnostics) {
      this.updateSummaryUi();
    }

    if (this.currentSpec) {
      this.updateRuntimeUi(true);
    }
  }

  async init() {
    this.bindUi();
    this.bindKeyboard();
    this.bindGlobalDebugHooks();
    this.readUrlState();
    this.refreshLocalizedUi();
    this.setMode('initializing');

    try {
      const loadResult = await loadMujocoBindings((message, level = 'info') => this.log(message, level));
      this.mujoco = loadResult.mujoco;
      this.mujocoSourceUrl = loadResult.sourceUrl;

      mountWorkingFilesystem(this.mujoco);

      this.ready = true;
      this.setStatus('ready', 'ready');

      const apiSummary = summarizeMujocoApi(this.mujoco);
      this.logKey('mujocoInitialized', { source: this.mujocoSourceUrl });
      this.logKey('mujocoApiSummary', {
        staticLoad: apiSummary.hasStaticMjLoadXml,
        legacyModel: apiSummary.hasLegacyModelCtor,
        mjData: apiSummary.hasMjDataClass,
        mjStep: apiSummary.hasMjStep,
      });

      window.__mujocoApiSummary = apiSummary;

      await this.generateFromUi();
      this.exposeDebugApi();
      requestAnimationFrame(this.animate);
    } catch (error) {
      console.error(error);
      this.logKey(
        'mujocoInitFailed',
        { message: error instanceof Error ? error.message : String(error) },
        'error',
      );
      this.setStatus('loadError', 'error');
      this.setMode('error');
      this.showOverlayKey(
        'initFailed',
        { message: error instanceof Error ? error.message : String(error) },
        'error',
        false,
      );
    }
  }

  bindGlobalDebugHooks() {
    window.addEventListener('error', (event) => {
      this.logKey('windowError', { message: event.message }, 'error');
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.logKey('unhandledRejection', { reason: String(event.reason) }, 'error');
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
      getLogText: () => UI.logText.textContent,
      getMujocoSourceUrl: () => this.mujocoSourceUrl,
      reset: () => this.resetPuzzle(),
      runSelfCheck: (options = {}) => this.runSelfCheck(options),
      setLocale: (locale) => this.i18n.setLocale(locale),
    };
  }

  bindUi() {
    UI.languageSelect.addEventListener('change', () => {
      this.i18n.setLocale(UI.languageSelect.value);
    });

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
        this.showOverlayKey('urlCopied', {}, 'info', true);
      } catch {
        window.prompt(this.t('prompt.copyUrl'), url);
      }
    });

    UI.debugToggle.addEventListener('change', () => {
      this.viewer.setDebugVisible(UI.debugToggle.checked);
    });

    UI.runDiagBtn.addEventListener('click', () => {
      this.runDiagnosticsNow();
      this.showOverlayKey('diagnosticsRecomputed', {}, 'info', true);
    });

    UI.downloadDiagBtn.addEventListener('click', () => this.downloadDiagnostics());
    UI.downloadSpecBtn.addEventListener('click', () => this.downloadSpec());
    UI.downloadMjcfBtn.addEventListener('click', () => this.downloadMjcf());

    UI.copyLogBtn.addEventListener('click', async () => {
      const text = UI.logText.textContent;

      try {
        await navigator.clipboard.writeText(text);
        this.showOverlayKey('logCopied', {}, 'info', true);
      } catch {
        window.prompt(this.t('prompt.copyLog'), text);
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
    const lang = params.get('lang');

    if (lang) {
      this.i18n.setLocale(lang);
    }

    if (Number.isFinite(complexity)) {
      UI.complexity.value = String(clamp(Math.round(complexity), 1, 10));
      UI.complexityValue.textContent = UI.complexity.value;
    }

    if (Number.isFinite(seed)) {
      UI.seedInput.value = String(Math.trunc(seed));
    } else if (!UI.seedInput.value) {
      UI.seedInput.value = '1';
    }
  }

  buildShareUrl() {
    const url = new URL(window.location.href);
    url.searchParams.set('complexity', UI.complexity.value);
    url.searchParams.set('seed', String(Math.trunc(Number(UI.seedInput.value) || 0)));
    url.searchParams.set('lang', this.locale);
    return url.toString();
  }

  writeUrlState() {
    window.history.replaceState({}, '', this.buildShareUrl());
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
    clearTimeout(this.overlayTimer);
    this.disposeSimulation();
    this.viewer.dispose();

    if (this.mujoco) {
      unmountWorkingFilesystem(this.mujoco);
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

    this.setMode('generating');
    this.setStatus('compiling', 'loading');

    try {
      this.disposeSimulation();

      this.currentSpec = generatePuzzleSpec({ seed, complexity });
      this.currentStaticDiagnostics = runStaticDiagnostics(this.currentSpec);
      this.currentMjcf = buildPuzzleMjcf(this.currentSpec);

      this.mujoco.FS.writeFile('/working/puzzle.xml', this.currentMjcf);
      this.model = loadModelFromXml({
        mujoco: this.mujoco,
        xmlPath: '/working/puzzle.xml',
        xmlText: this.currentMjcf,
        log: (message, level = 'info') => this.log(message, level),
      });
      this.data = createDataForModel(this.mujoco, this.model);

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
      this.solveConditionHold = 0;

      this.viewer.setDebugMarkers(this.currentSpec);
      this.viewer.setDebugVisible(UI.debugToggle.checked);
      this.recenterCamera();
      this.updateSummaryUi();
      this.updatePauseButton();
      this.updateRuntimeUi(true);

      this.setMode('playing');
      this.setStatus('ready', 'ready');

      this.logKey('generatePuzzle', {
        family: this.currentSpec.family.id,
        seed,
        complexity,
      });

      this.currentSpec.generationLog.forEach((line) => this.log(`gen: ${line}`));
      this.showOverlayKey('generated', {}, 'info', true);
    } catch (error) {
      console.error(error);
      this.setStatus('compileError', 'error');
      this.setMode('error');

      this.logKey(
        'generateFailed',
        { message: error instanceof Error ? error.message : String(error) },
        'error',
      );

      this.showOverlayKey(
        'generateFailed',
        { message: error instanceof Error ? error.message : String(error) },
        'error',
        false,
      );
    }
  }

  updateSummaryUi() {
    if (!this.currentSpec || !this.currentStaticDiagnostics) return;

    const family = localizeFamily(this.currentSpec.family, this.locale);

    UI.familyValue.textContent = family.label;
    UI.gapRatioValue.textContent = `${this.currentSpec.stats.avgGapRatio.toFixed(2)}x`;
    UI.difficultyValue.textContent = `${this.currentSpec.stats.estimatedDifficulty}/10`;
    UI.wireLengthValue.textContent = `${this.currentSpec.stats.totalWireLength.toFixed(2)} m`;
    UI.separationValue.textContent = '0%';
    UI.contactValue.textContent = '0';
    UI.timerValue.textContent = '00:00.0';
    UI.seedEcho.textContent = String(this.currentSpec.seed);
    UI.staticDiagText.textContent = formatStaticDiagnostics(this.currentStaticDiagnostics, {
      locale: this.locale,
      familyLabel: family.label,
    });
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
    this.solveConditionHold = 0;

    UI.timerValue.textContent = '00:00.0';
    this.updatePauseButton();
    this.updateRuntimeUi(true);
    this.setMode('playing');

    this.logKey('reset');
    this.showOverlayKey('resetDone', {}, 'info', true);
  }

  togglePause() {
    if (!this.data || this.solved) return;

    this.paused = !this.paused;
    this.updatePauseButton();
    this.setMode(this.paused ? 'paused' : 'playing');
    this.updateRuntimeUi(true);
  }

  updatePauseButton() {
    UI.pauseBtn.textContent = this.paused ? this.t('button.resume') : this.t('button.pause');
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

    const turboEnabled =
      UI.turboHintToggle.checked && (this.keyState.ShiftLeft || this.keyState.ShiftRight);

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

    if (!force && this.runtimeDiagAccumulator < 0.16) {
      return;
    }

    UI.runtimeDiagText.textContent = formatRuntimeDiagnostics(this.currentRuntimeDiagnostics, {
      locale: this.locale,
    });
    UI.separationValue.textContent = `${Math.round(Math.min(this.currentRuntimeDiagnostics.separation, 1) * 100)}%`;
    UI.contactValue.textContent = String(this.currentRuntimeDiagnostics.contactCount);
    this.viewer.updateMovingMarkers(this.currentSpec, state);
    this.runtimeDiagAccumulator = 0;
  }

  updateSolvedState(dt = this.physicsStep) {
    if (!this.currentSpec || !this.currentRuntimeDiagnostics || this.solved) return;

    const diag = this.currentRuntimeDiagnostics;
    const geom = this.currentSpec.geometry;

    const solveCandidate =
      diag.threadingCrossings === 0 &&
      diag.threadingGapCrossings === 0 &&
      diag.threadingScore < (geom.solveThreadingScoreMax ?? 0.02) &&
      diag.interPieceClearance > geom.solveClearance &&
      diag.mutualHole < (geom.solveMutualHoleMax ?? 0.03) &&
      diag.centerDistance > (geom.solveCenterDistanceMin ?? geom.solveDistance) &&
      diag.contactCount === 0 &&
      diag.speed < 0.9 &&
      diag.angularSpeed < 2.2;

    this.solveConditionHold = solveCandidate ? this.solveConditionHold + Math.max(0, dt) : 0;

    if (this.solveConditionHold < (geom.solveHoldDuration ?? 0.35)) return;

    this.solved = true;
    this.paused = true;
    this.updatePauseButton();
    this.updateRuntimeUi(true);
    this.setMode('solved');

    this.logKey('solved', {
      time: formatTime(this.elapsed),
      seed: this.currentSpec.seed,
      clearance: diag.interPieceClearance.toFixed(4),
      threading: `${diag.threadingCrossings}/${diag.threadingGapCrossings}`,
    });

    const family = localizeFamily(this.currentSpec.family, this.locale);
    this.showOverlayKey(
      'solved',
      {
        time: formatTime(this.elapsed),
        family: family.label,
        seed: this.currentSpec.seed,
      },
      'solved',
      false,
    );
  }

  runSelfCheck({ seeds = [1, 2, 3, 4, 5], complexities = [1, 4, 7, 10] } = {}) {
    const results = [];

    for (const complexity of complexities) {
      for (const seed of seeds) {
        const spec = generatePuzzleSpec({ seed, complexity });
        const staticDiag = runStaticDiagnostics(spec);
        const runtimeDiag = buildRuntimeDiagnostics(spec, {
          pos: spec.startPose.pos,
          quat: spec.startPose.quat,
          linvel: [0, 0, 0],
          angvel: [0, 0, 0],
          ncon: 0,
          paused: false,
          solved: false,
        });

        results.push({
          seed,
          complexity,
          startSolvedLike:
            runtimeDiag.threadingCrossings === 0 &&
            runtimeDiag.threadingGapCrossings === 0,
          startSeparation: runtimeDiag.separation,
          startThreading: runtimeDiag.threadingScore,
          solveClearance: spec.geometry.solveClearance,
          startClearance: spec.startPoseStats.clearance,
          staticWarnings: staticDiag.checks
            .filter((check) => check.status === 'warn' || check.status === 'fail')
            .map((check) => check.code),
        });
      }
    }

    const bad = results.filter(
      (row) =>
        row.startSolvedLike ||
        row.solveClearance <= row.startClearance ||
        row.startSeparation > 0.9,
    );

    const summary = {
      total: results.length,
      badCount: bad.length,
      bad,
      results,
    };

    this.logKey('selfCheckSummary', {
      total: summary.total,
      badCount: summary.badCount,
    });

    return summary;
  }

  runDiagnosticsNow() {
    if (!this.currentSpec) return null;

    this.currentStaticDiagnostics = runStaticDiagnostics(this.currentSpec);
    this.updateSummaryUi();
    this.updateRuntimeUi(true);
    this.logKey('diagnosticsRecomputed');

    return {
      static: this.currentStaticDiagnostics,
      runtime: this.currentRuntimeDiagnostics,
    };
  }

  downloadDiagnostics() {
    if (!this.currentSpec) return;

    if (!this.currentStaticDiagnostics) {
      this.currentStaticDiagnostics = runStaticDiagnostics(this.currentSpec);
    }

    if (!this.currentRuntimeDiagnostics) {
      this.updateRuntimeUi(true);
    }

    const bundle = buildDiagnosticsBundle({
      spec: this.currentSpec,
      staticDiag: this.currentStaticDiagnostics,
      runtimeDiag: this.currentRuntimeDiagnostics,
      logLines: UI.logText.textContent.split('\n'),
    });

    downloadText(
      `chie-no-wa-diagnostics-seed-${this.currentSpec.seed}.json`,
      `${JSON.stringify(bundle, null, 2)}\n`,
    );

    this.logKey('downloadedDiagnostics');
  }

  downloadSpec() {
    if (!this.currentSpec) return;

    downloadText(
      `chie-no-wa-spec-seed-${this.currentSpec.seed}.json`,
      `${JSON.stringify(this.currentSpec, null, 2)}\n`,
    );

    this.logKey('downloadedSpec');
  }

  downloadMjcf() {
    if (!this.currentMjcf || !this.currentSpec) return;

    downloadText(
      `chie-no-wa-seed-${this.currentSpec.seed}.xml`,
      `${this.currentMjcf}\n`,
    );

    this.logKey('downloadedMjcf');
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
      this.updateSolvedState(dt);
      this.viewer.sync(
        this.mujoco,
        this.model,
        this.data,
        this.mjvScene,
        this.mjvOption,
        this.mjvPerturb,
        this.mjvCamera,
      );
      this.viewer.render();
    } else {
      this.viewer.render();
    }

    requestAnimationFrame(this.animate);
  }
}
