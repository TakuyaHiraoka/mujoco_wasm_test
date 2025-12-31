import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import load_mujoco from "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js";
import { ik3LinkPlanar, clamp, wrapToPi } from "./ik.js";

/**
 * MuJoCo WASM Drawing Arm (Kinematic only)
 *
 * - No dynamics (mj_step) is used.
 * - We directly update qpos towards IK targets and call mj_forward.
 * - 2D: user draws on the right canvas.
 * - 3D: the arm follows the path and drops tiny "ink" spheres on the canvas surface.
 */

// =============================
// DOM
// =============================
const simCanvas = /** @type {HTMLCanvasElement} */ (document.getElementById("simCanvas"));
const drawCanvas = /** @type {HTMLCanvasElement} */ (document.getElementById("drawCanvas"));
const statusEl = document.getElementById("status");

const btnClear = document.getElementById("btnClear");
const btnHome = document.getElementById("btnHome");

const chkFollow = /** @type {HTMLInputElement} */ (document.getElementById("chkFollow"));
const chkShowTarget = /** @type {HTMLInputElement} */ (document.getElementById("chkShowTarget"));
const chkShowExec = /** @type {HTMLInputElement} */ (document.getElementById("chkShowExec"));
const chkShowInk3D = /** @type {HTMLInputElement} */ (document.getElementById("chkShowInk3D"));

const ctx2d = drawCanvas.getContext("2d");

// =============================
// Workspace mapping (2D canvas <-> world XY)
// =============================
// この範囲は models/arm_pen.xml のキャンバス(geom)内の、アームが届く領域に合わせています。
const workspace = {
  xMin: 0.03,
  xMax: 0.43,
  yMin: -0.15,
  yMax: 0.15,
};

let drawCssW = 1;
let drawCssH = 1;

function canvasCssToWorld(px, py) {
  const u = px / drawCssW;
  const v = py / drawCssH;
  const x = workspace.xMin + u * (workspace.xMax - workspace.xMin);
  const y = workspace.yMax - v * (workspace.yMax - workspace.yMin); // y反転
  return { x, y };
}

function worldToCanvasCss(x, y) {
  const u = (x - workspace.xMin) / (workspace.xMax - workspace.xMin);
  const v = (workspace.yMax - y) / (workspace.yMax - workspace.yMin);
  return { px: u * drawCssW, py: v * drawCssH };
}

// =============================
// Target / executed strokes
// =============================
/** @type {Array<Array<{x:number,y:number}>>} */
let targetStrokes = [];
/** @type {Array<Array<{x:number,y:number}>>} */
let executedStrokes = [];

/** @type {Array<{x:number,y:number,penDown:boolean}>} */
let commandQueue = [];

let isPointerDown = false;
/** @type {Array<{x:number,y:number}> | null} */
let currentTargetStroke = null;
/** @type {Array<{x:number,y:number,penDown:boolean}>} */
let currentStrokeCmds = [];

let followWhileDrawing = true;
let showTarget = true;
let showExecuted = true;
let showInk3D = true;

// =============================
// MuJoCo state
// =============================
let mujoco = null;
let model = null;
let data = null;

const PEN_SITE_NAME = "pen_tip_site";
let penSiteId = -1;

function toMjtEnumInt(v) {
  // Some mujoco-js builds expose enums as embind objects, not plain numbers.
  if (typeof v === "number") return v;
  if (v && typeof v === "object") {
    if (typeof v.value === "number") return v.value;
    if (typeof v.value === "function") {
      const n = v.value();
      if (typeof n === "number") return n;
    }
    if (typeof v.valueOf === "function") {
      const n = v.valueOf();
      if (typeof n === "number") return n;
    }
    if ("__value" in v && typeof v.__value === "number") return v.__value;
  }
  return NaN;
}

function lookupSiteIdByName(name) {
  if (!mujoco || !model) return -1;
  try {
    if (mujoco.mj_name2id && mujoco.mjtObj && mujoco.mjtObj.mjOBJ_SITE !== undefined) {
      const t = toMjtEnumInt(mujoco.mjtObj.mjOBJ_SITE);
      if (!Number.isFinite(t)) throw new Error("mjtObj.mjOBJ_SITE not numeric");
      return mujoco.mj_name2id(model, t, name);
    }
  } catch (e) {
    // ignore, fallback
  }
  return -1;
}

function getSiteXpos(siteId) {
  const i = 3 * siteId;
  return {
    x: data.site_xpos[i + 0],
    y: data.site_xpos[i + 1],
    z: data.site_xpos[i + 2],
  };
}

function getPenWorld() {
  if (!data) return { x: 0, y: 0, z: 0 };
  if (penSiteId >= 0) return getSiteXpos(penSiteId);
  // If site lookup fails but there is exactly 1 site, this fallback is still correct.
  return { x: data.site_xpos[0], y: data.site_xpos[1], z: data.site_xpos[2] };
}

// =============================
// IK / kinematic control params
// =============================
const L1 = 0.20;
const L2 = 0.20;
const L3 = 0.06;
const MAX_REACH = L1 + L2 + L3 - 0.005;

const PEN_UP = 0.03;
const PEN_DOWN = -0.012;

const TARGET_EPS = 0.010;
const MOVE_EPS = 0.012;

const JOINT_MAX_RATE = 8.0; // rad/s
const PEN_MAX_RATE = 0.20; // m/s

function angleDiff(a, b) {
  // smallest signed diff (a-b) in (-pi,pi]
  return wrapToPi(a - b);
}

function clampToReach(x, y) {
  const r = Math.hypot(x, y);
  if (r < 1e-9) return { x, y };
  if (r <= MAX_REACH) return { x, y };
  const s = MAX_REACH / r;
  return { x: x * s, y: y * s };
}

function pickIKSolution(x, y) {
  // Choose end-effector orientation to face the target direction.
  const phi = Math.atan2(y, x);

  const qDown = ik3LinkPlanar(x, y, L1, L2, L3, phi, "down");
  const qUp = ik3LinkPlanar(x, y, L1, L2, L3, phi, "up");
  if (!qDown && !qUp) return null;
  if (qDown && !qUp) return qDown;
  if (qUp && !qDown) return qUp;

  // pick the one closest to current pose to avoid elbow flipping
  const qcur = [data.qpos[0], data.qpos[1], data.qpos[2]];
  const cost = (q) =>
    Math.abs(angleDiff(q[0], qcur[0])) +
    Math.abs(angleDiff(q[1], qcur[1])) +
    Math.abs(angleDiff(q[2], qcur[2]));

  return cost(qDown) <= cost(qUp) ? qDown : qUp;
}

// =============================
// UI wiring
// =============================
chkFollow.addEventListener("change", () => {
  followWhileDrawing = chkFollow.checked;
});
chkShowTarget.addEventListener("change", () => {
  showTarget = chkShowTarget.checked;
});
chkShowExec.addEventListener("change", () => {
  showExecuted = chkShowExec.checked;
});
chkShowInk3D.addEventListener("change", () => {
  showInk3D = chkShowInk3D.checked;
  if (inkMesh) inkMesh.visible = showInk3D;
});

btnClear.addEventListener("click", () => {
  targetStrokes = [];
  executedStrokes = [];
  commandQueue = [];
  isPointerDown = false;
  currentTargetStroke = null;
  currentStrokeCmds = [];
  clearInk3D();
});

btnHome.addEventListener("click", () => {
  commandQueue = [];
  currentStrokeCmds = [];
  // Home pose
  if (data && mujoco && model) {
    data.qpos[0] = 0.0;
    data.qpos[1] = 0.8;
    data.qpos[2] = -0.8;
    data.qpos[3] = PEN_UP;
    mujoco.mj_forward(model, data);
  }
});

// =============================
// Pointer input on draw canvas
// =============================
function getPointerPosCss(e) {
  const rect = drawCanvas.getBoundingClientRect();
  return { px: e.clientX - rect.left, py: e.clientY - rect.top };
}

function pushTargetPoint(px, py) {
  if (!currentTargetStroke) return;

  // decimate
  const pts = currentTargetStroke;
  const last = pts[pts.length - 1];
  const minDistPx = 2.5;
  if (last) {
    const d = Math.hypot(px - last.x, py - last.y);
    if (d < minDistPx) return;
  }

  const isFirst = pts.length === 0;
  pts.push({ x: px, y: py });

  let { x, y } = canvasCssToWorld(px, py);
  ({ x, y } = clampToReach(x, y));

  if (isFirst) {
    // pen-up move to start, then pen-down at start
    const move = { x, y, penDown: false };
    const down = { x, y, penDown: true };
    currentStrokeCmds.push(move, down);
    if (followWhileDrawing) commandQueue.push(move, down);
  } else {
    const down = { x, y, penDown: true };
    currentStrokeCmds.push(down);
    if (followWhileDrawing) commandQueue.push(down);
  }
}

drawCanvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  drawCanvas.setPointerCapture(e.pointerId);

  isPointerDown = true;
  currentStrokeCmds = [];
  currentTargetStroke = [];
  targetStrokes.push(currentTargetStroke);

  const { px, py } = getPointerPosCss(e);
  pushTargetPoint(px, py);
});

drawCanvas.addEventListener("pointermove", (e) => {
  if (!isPointerDown) return;
  e.preventDefault();
  const { px, py } = getPointerPosCss(e);
  pushTargetPoint(px, py);
});

function endStroke(e) {
  if (!isPointerDown) return;
  e.preventDefault();

  isPointerDown = false;

  const lastStroke = currentTargetStroke;
  if (lastStroke && lastStroke.length > 0) {
    const lastPt = lastStroke[lastStroke.length - 1];
    let { x, y } = canvasCssToWorld(lastPt.x, lastPt.y);
    ({ x, y } = clampToReach(x, y));

    if (!followWhileDrawing) {
      for (const cmd of currentStrokeCmds) commandQueue.push(cmd);
    }

    // lift the pen at the end
    commandQueue.push({ x, y, penDown: false });
  }

  currentTargetStroke = null;
  currentStrokeCmds = [];
}

drawCanvas.addEventListener("pointerup", endStroke);
drawCanvas.addEventListener("pointercancel", endStroke);

// =============================
// Resize handling
// =============================
let renderer = null;
let scene = null;
let camera = null;
let controls = null;

function resizeAll() {
  // 2D draw canvas
  {
    const rect = drawCanvas.getBoundingClientRect();
    drawCssW = Math.max(1, Math.floor(rect.width));
    drawCssH = Math.max(1, Math.floor(rect.height));

    const dpr = window.devicePixelRatio || 1;
    drawCanvas.width = Math.floor(drawCssW * dpr);
    drawCanvas.height = Math.floor(drawCssH * dpr);

    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // Three canvas
  if (renderer && camera) {
    const rect = simCanvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

window.addEventListener("resize", resizeAll);

// =============================
// Three.js rendering (MuJoCo geoms + 3D ink)
// =============================
/** @type {Array<{geomIndex:number, mesh:THREE.Mesh}>} */
let geomMeshes = [];

function mujocoGeomToThreeGeometry(geomType, sizeVec3) {
  // MuJoCo mjtGeom (common subset)
  // 2: sphere, 3: capsule, 5: cylinder, 6: box
  switch (geomType) {
    case 2: {
      const r = sizeVec3[0];
      return new THREE.SphereGeometry(r, 20, 14);
    }
    case 3: {
      const r = sizeVec3[0];
      const half = sizeVec3[1];
      const g = new THREE.CapsuleGeometry(r, 2 * half, 8, 16);
      // Three capsule is along Y; MuJoCo capsules are typically along Z.
      g.rotateX(Math.PI / 2);
      return g;
    }
    case 5: {
      const r = sizeVec3[0];
      const half = sizeVec3[1];
      const g = new THREE.CylinderGeometry(r, r, 2 * half, 20);
      g.rotateX(Math.PI / 2);
      return g;
    }
    case 6: {
      const hx = sizeVec3[0];
      const hy = sizeVec3[1];
      const hz = sizeVec3[2];
      return new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz);
    }
    default:
      return null;
  }
}

function createThreeSceneFromModel() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e0f12);

  camera = new THREE.PerspectiveCamera(45, 1.0, 0.01, 10.0);
  camera.position.set(0.45, -0.55, 0.45);
  camera.lookAt(new THREE.Vector3(0.25, 0, 0.02));

  const hemi = new THREE.HemisphereLight(0xffffff, 0x222233, 0.9);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 0.7);
  dir.position.set(1.0, -1.0, 1.2);
  scene.add(dir);

  renderer = new THREE.WebGLRenderer({ canvas: simCanvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  controls = new OrbitControls(camera, simCanvas);
  controls.target.set(0.25, 0, 0.02);
  controls.update();

  // MuJoCo geoms
  geomMeshes = [];
  for (let i = 0; i < model.ngeom; i++) {
    const geomType = model.geom_type[i];
    const size = model.geom_size.subarray(3 * i, 3 * i + 3);
    const rgba = model.geom_rgba.subarray(4 * i, 4 * i + 4);

    const geometry = mujocoGeomToThreeGeometry(geomType, size);
    if (!geometry) continue;

    const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
    const material = new THREE.MeshStandardMaterial({
      color,
      transparent: rgba[3] < 0.999,
      opacity: rgba[3],
      metalness: 0.0,
      roughness: 0.85,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false;
    scene.add(mesh);
    geomMeshes.push({ geomIndex: i, mesh });
  }

  // 3D ink dots
  createInk3D();
}

function updateThreeTransformsFromData() {
  const gxpos = data.geom_xpos;
  const gxmat = data.geom_xmat;

  for (const { geomIndex, mesh } of geomMeshes) {
    const p = 3 * geomIndex;
    const r = 9 * geomIndex;

    const px = gxpos[p + 0];
    const py = gxpos[p + 1];
    const pz = gxpos[p + 2];

    // MuJoCo: row-major 3x3
    const m00 = gxmat[r + 0], m01 = gxmat[r + 1], m02 = gxmat[r + 2];
    const m10 = gxmat[r + 3], m11 = gxmat[r + 4], m12 = gxmat[r + 5];
    const m20 = gxmat[r + 6], m21 = gxmat[r + 7], m22 = gxmat[r + 8];

    mesh.matrix.set(
      m00, m01, m02, px,
      m10, m11, m12, py,
      m20, m21, m22, pz,
      0,   0,   0,   1
    );
    mesh.matrixWorldNeedsUpdate = true;
  }
}

// ===== 3D ink (instanced spheres) =====
const CANVAS_SURFACE_Z = 0.0;
const INK_Z_OFFSET = 0.001;
const INK_RADIUS = 0.0026;
const MAX_INK_DOTS = 60000;

/** @type {THREE.InstancedMesh | null} */
let inkMesh = null;
let inkCount = 0;
let tmpMat4 = new THREE.Matrix4();

function createInk3D() {
  const geom = new THREE.SphereGeometry(INK_RADIUS, 12, 10);
  const mat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.95, metalness: 0.0 });

  inkMesh = new THREE.InstancedMesh(geom, mat, MAX_INK_DOTS);
  inkMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  inkMesh.count = 0;
  inkMesh.visible = showInk3D;
  scene.add(inkMesh);
  inkCount = 0;
}

function clearInk3D() {
  if (!inkMesh) return;
  inkCount = 0;
  inkMesh.count = 0;
  inkMesh.instanceMatrix.needsUpdate = true;
}

function addInkDotWorld(x, y) {
  if (!inkMesh) return;
  if (inkCount >= MAX_INK_DOTS) return;
  tmpMat4.identity();
  tmpMat4.setPosition(x, y, CANVAS_SURFACE_Z + INK_Z_OFFSET);
  inkMesh.setMatrixAt(inkCount, tmpMat4);
  inkCount++;
  inkMesh.count = inkCount;
  inkMesh.instanceMatrix.needsUpdate = true;
}

// =============================
// 2D overlay drawing
// =============================
function drawStrokes(ctx, strokes) {
  for (const stroke of strokes) {
    if (!stroke || stroke.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(stroke[0].x, stroke[0].y);
    for (let i = 1; i < stroke.length; i++) ctx.lineTo(stroke[i].x, stroke[i].y);
    ctx.stroke();
  }
}

function redraw2DOverlay(penWorld, targetCmd) {
  // background
  ctx2d.clearRect(0, 0, drawCssW, drawCssH);
  ctx2d.fillStyle = "#ffffff";
  ctx2d.fillRect(0, 0, drawCssW, drawCssH);

  // frame
  ctx2d.strokeStyle = "rgba(0,0,0,0.25)";
  ctx2d.lineWidth = 1;
  ctx2d.setLineDash([]);
  ctx2d.strokeRect(0.5, 0.5, drawCssW - 1, drawCssH - 1);

  if (showTarget) {
    ctx2d.strokeStyle = "rgba(0,0,0,0.22)";
    ctx2d.lineWidth = 2;
    ctx2d.setLineDash([6, 5]);
    drawStrokes(ctx2d, targetStrokes);
  }

  if (showExecuted) {
    ctx2d.strokeStyle = "rgba(0,0,0,0.92)";
    ctx2d.lineWidth = 2.2;
    ctx2d.setLineDash([]);
    drawStrokes(ctx2d, executedStrokes);
  }

  if (targetCmd) {
    const p = worldToCanvasCss(targetCmd.x, targetCmd.y);
    ctx2d.fillStyle = "rgba(255,0,0,0.85)";
    ctx2d.beginPath();
    ctx2d.arc(p.px, p.py, 4.2, 0, Math.PI * 2);
    ctx2d.fill();
  }

  if (penWorld) {
    const p = worldToCanvasCss(penWorld.x, penWorld.y);
    ctx2d.fillStyle = "rgba(0,120,255,0.85)";
    ctx2d.beginPath();
    ctx2d.arc(p.px, p.py, 4.2, 0, Math.PI * 2);
    ctx2d.fill();
  }

  // text
  ctx2d.fillStyle = "rgba(0,0,0,0.55)";
  ctx2d.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  ctx2d.fillText(`queue: ${commandQueue.length} | ink: ${inkCount}`, 10, 18);
}

// =============================
// Kinematic follow loop
// =============================
let lastFrameTime = performance.now();
let accumulator = 0;
let fpsCounter = { frames: 0, last: performance.now(), fps: 0 };

let execPenDown = false;
let currentExecStroke = null;
let lastInkCanvasPt = null;

function recordExecutedPoint(penWorld, penIsDown) {
  if (penIsDown) {
    if (!execPenDown) {
      currentExecStroke = [];
      executedStrokes.push(currentExecStroke);
      execPenDown = true;
      lastInkCanvasPt = null;
    }

    const { px, py } = worldToCanvasCss(penWorld.x, penWorld.y);
    const last = currentExecStroke[currentExecStroke.length - 1];
    const minDistPx = 2.0;

    if (!last || Math.hypot(px - last.x, py - last.y) >= minDistPx) {
      currentExecStroke.push({ x: px, y: py });

      // 3D ink: also decimate by the same criterion (in canvas space)
      if (!lastInkCanvasPt || Math.hypot(px - lastInkCanvasPt.x, py - lastInkCanvasPt.y) >= minDistPx) {
        addInkDotWorld(penWorld.x, penWorld.y);
        lastInkCanvasPt = { x: px, y: py };
      }
    }
  } else {
    execPenDown = false;
    currentExecStroke = null;
    lastInkCanvasPt = null;
  }
}

function stepOnce() {
  if (!model || !data) return;

  const cmd = commandQueue.length > 0 ? commandQueue[0] : null;
  const penDesired = cmd && cmd.penDown ? PEN_DOWN : PEN_UP;

  // IK target
  let qTarget = null;
  if (cmd) {
    let { x, y } = clampToReach(cmd.x, cmd.y);
    qTarget = pickIKSolution(x, y);
    if (!qTarget) {
      // unreachable (should be rare with clamp) -> drop
      commandQueue.shift();
      qTarget = null;
    }
  }
  if (!qTarget) qTarget = [data.qpos[0], data.qpos[1], data.qpos[2]];

  const simDt = model.opt.timestep || 0.002;

  // Slew qpos towards target (no dynamics)
  const maxDq = JOINT_MAX_RATE * simDt;
  for (let i = 0; i < 3; i++) {
    const diff = angleDiff(qTarget[i], data.qpos[i]);
    data.qpos[i] = wrapToPi(data.qpos[i] + clamp(diff, -maxDq, maxDq));
  }

  const maxDp = PEN_MAX_RATE * simDt;
  data.qpos[3] = clamp(data.qpos[3] + clamp(penDesired - data.qpos[3], -maxDp, maxDp), -0.02, 0.05);

  mujoco.mj_forward(model, data);

  const penWorld = getPenWorld();

  // Dequeue when reached
  if (cmd) {
    const err = Math.hypot(penWorld.x - cmd.x, penWorld.y - cmd.y);

    if (cmd.penDown) {
      if (err < TARGET_EPS) {
        // skip multiple nearby points
        do {
          commandQueue.shift();
        } while (
          commandQueue.length > 0 &&
          commandQueue[0].penDown &&
          Math.hypot(penWorld.x - commandQueue[0].x, penWorld.y - commandQueue[0].y) < TARGET_EPS
        );
      }
    } else {
      // pen-up move / finish
      if (err < MOVE_EPS && data.qpos[3] > PEN_UP * 0.7) {
        commandQueue.shift();
      }
    }
  }

  // Record executed strokes (use actual pen_z)
  const penIsDown = data.qpos[3] < (PEN_UP + PEN_DOWN) * 0.5;
  recordExecutedPoint(penWorld, penIsDown);

  return { penWorld, targetCmd: cmd };
}

function animate(now) {
  requestAnimationFrame(animate);

  const dt = (now - lastFrameTime) / 1000;
  lastFrameTime = now;

  accumulator += Math.min(0.05, dt);

  const simDt = model ? model.opt.timestep : 0.002;
  let snapshot = null;
  while (model && data && accumulator >= simDt) {
    snapshot = stepOnce();
    accumulator -= simDt;
  }

  if (model && data) {
    updateThreeTransformsFromData();
    renderer.render(scene, camera);

    const penWorld = snapshot?.penWorld ?? getPenWorld();
    const targetCmd = snapshot?.targetCmd ?? (commandQueue.length > 0 ? commandQueue[0] : null);
    redraw2DOverlay(penWorld, targetCmd);

    fpsCounter.frames++;
    const t = performance.now();
    if (t - fpsCounter.last > 500) {
      fpsCounter.fps = (fpsCounter.frames * 1000) / (t - fpsCounter.last);
      fpsCounter.frames = 0;
      fpsCounter.last = t;
      statusEl.textContent = `fps ${fpsCounter.fps.toFixed(1)} | queue ${commandQueue.length} | ink ${inkCount}`;
    }
  }
}

// =============================
// Boot
// =============================
async function boot() {
  try {
    statusEl.textContent = "loading mujoco-js…";
    mujoco = await load_mujoco();

    statusEl.textContent = "loading model…";

    mujoco.FS.mkdir("/working");
    mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");

    const xmlText = await (await fetch("./models/arm_pen.xml")).text();
    mujoco.FS.writeFile("/working/arm_pen.xml", xmlText);

    model = mujoco.MjModel.loadFromXML("/working/arm_pen.xml");
    data = new mujoco.MjData(model);

    // Resolve pen site ID (fallback to 0 if lookup fails)
    penSiteId = lookupSiteIdByName(PEN_SITE_NAME);
    if (penSiteId < 0) {
      console.warn(`[warn] site "${PEN_SITE_NAME}" not found via mj_name2id. Using site 0 fallback.`);
    }

    // Home pose
    data.qpos[0] = 0.0;
    data.qpos[1] = 0.8;
    data.qpos[2] = -0.8;
    data.qpos[3] = PEN_UP;
    mujoco.mj_forward(model, data);

    statusEl.textContent = "init renderer…";
    createThreeSceneFromModel();
    resizeAll();

    statusEl.textContent = "ready";
    requestAnimationFrame(animate);
  } catch (err) {
    console.error(err);
    statusEl.textContent = "failed to load (see console)";
  }
}

resizeAll();
boot();
