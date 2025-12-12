import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import load_mujoco from "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js";
import { Midi } from "https://esm.sh/@tonejs/midi@2.0.28"; // MIDIパース用（ブラウザ向けに解決してくれる）

const UI = {
  status: document.getElementById("status"),
  midiFile: document.getElementById("midiFile"),
  playPause: document.getElementById("playPause"),
  stop: document.getElementById("stop"),
  speed: document.getElementById("speed"),
  speedValue: document.getElementById("speedValue"),
  useSoundfont: document.getElementById("useSoundfont"),
};

const FIRST_MIDI = 21; // A0
const N_KEYS = 88;

function isBlackMidi(midiNote) {
  const m = midiNote % 12;
  return m === 1 || m === 3 || m === 6 || m === 8 || m === 10;
}

// --------------------
// Three.js
// --------------------
const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(0.0, -1.3, 0.55);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.18, 0.02);
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0x222222, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(1.2, -0.6, 2.0);
scene.add(dir);

// --------------------
// Piano geometry layout
// --------------------
function buildKeyInfos() {
  const whitePitch = 0.023;
  let whiteIndex = 0;
  const infos = [];

  for (let i = 0; i < N_KEYS; i++) {
    const midi = FIRST_MIDI + i;
    const black = isBlackMidi(midi);

    let x;
    if (black) x = (whiteIndex - 0.5) * whitePitch;
    else { x = whiteIndex * whitePitch; whiteIndex += 1; }

    infos.push({
      midi,
      black,
      x,
      width:  black ? 0.014 : 0.022,
      length: black ? 0.090 : 0.140,
      height: 0.020,
      pivotZ: black ? 0.030 : 0.022,
      // 見た目で分かるように少し深め
      pressAngle: black ? -0.10 : -0.12
    });
  }

  const centerX = (51 * whitePitch) / 2;
  for (const k of infos) k.x -= centerX;
  return infos;
}
const keyInfos = buildKeyInfos();

// --------------------
// MJCF生成（鍵盤のみ）
// --------------------
function mjcfPiano(keys) {
  const xs = keys.map(k => k.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const baseW = (maxX - minX) + 0.12;
  const baseL = 0.24;
  const baseH = 0.04;

  let xml = "";
  xml += `<mujoco model="midi_piano">\n`;
  xml += `  <compiler angle="radian" coordinate="local"/>\n`;
  xml += `  <option timestep="0.005" gravity="0 0 -9.81"/>\n`;
  xml += `  <default>\n`;
  xml += `    <joint type="hinge" damping="0.05" limited="true"/>\n`;
  xml += `    <geom type="box" density="300" contype="0" conaffinity="0"/>\n`;
  xml += `  </default>\n`;
  xml += `  <worldbody>\n`;
  xml += `    <light pos="0 -1 2" dir="0 1 -1" diffuse="1 1 1"/>\n`;
  xml += `    <body name="keyboard" pos="0 0 0">\n`;
  xml += `      <geom name="base" type="box" size="${(baseW/2).toFixed(4)} ${(baseL/2).toFixed(4)} ${(baseH/2).toFixed(4)}" pos="0 ${(baseL/2).toFixed(4)} 0" rgba="0.25 0.25 0.25 1"/>\n`;

  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    const hx = (k.width / 2).toFixed(4);
    const hy = (k.length / 2).toFixed(4);
    const hz = (k.height / 2).toFixed(4);
    const rgba = k.black ? "0.08 0.08 0.08 1" : "0.95 0.95 0.95 1";

    xml += `      <body name="key_${i}" pos="${k.x.toFixed(4)} 0 ${k.pivotZ.toFixed(4)}">\n`;
    xml += `        <joint name="key_joint_${i}" axis="1 0 0" range="${k.pressAngle.toFixed(4)} 0"/>\n`;
    // pivot を後端にして、geom中心は前方へ hy
    xml += `        <geom name="key_geom_${i}" type="box" size="${hx} ${hy} ${hz}" pos="0 ${hy} 0" rgba="${rgba}"/>\n`;
    xml += `      </body>\n`;
  }

  xml += `    </body>\n`;
  xml += `  </worldbody>\n`;
  xml += `</mujoco>\n`;
  return xml;
}

// --------------------
// MuJoCo init
// --------------------
UI.status.textContent = "Loading MuJoCo WASM…";
const MUJOCO_CDN_BASE = "https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/";
const mujoco = await load_mujoco({
  locateFile: (p) => `${MUJOCO_CDN_BASE}${p}`,
});

mujoco.FS.mkdir("/working");
mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");

mujoco.FS.writeFile("/working/piano.xml", mjcfPiano(keyInfos));
const model = mujoco.MjModel.loadFromXML("/working/piano.xml");
const data  = new mujoco.MjData(model);
mujoco.mj_forward(model, data);

// safe getter
const val = (x) => (typeof x === "function" ? x() : x);
const ngeom = val(model.ngeom);
const geom_type = val(model.geom_type);
const geom_size = val(model.geom_size);
const geom_rgba = val(model.geom_rgba);

const mjGEOM_PLANE = 0;
const mjGEOM_BOX = 6;

// geom mesh build
const meshes = new Array(ngeom);
for (let gi = 0; gi < ngeom; gi++) {
  const t = geom_type[gi];
  if (t !== mjGEOM_BOX && t !== mjGEOM_PLANE) continue;

  const r = geom_rgba[4*gi+0], g = geom_rgba[4*gi+1], b = geom_rgba[4*gi+2], a = geom_rgba[4*gi+3];
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(r, g, b),
    transparent: a < 0.999,
    opacity: a,
    roughness: 0.9,
    metalness: 0.1,
    side: THREE.DoubleSide,
  });

  let mesh;
  if (t === mjGEOM_PLANE) {
    mesh = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), mat);
  } else {
    const sx = geom_size[3*gi+0] * 2;
    const sy = geom_size[3*gi+1] * 2;
    const sz = geom_size[3*gi+2] * 2;
    mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
  }
  meshes[gi] = mesh;
  scene.add(mesh);
}

const matWhite = new THREE.MeshStandardMaterial({ color: 0xf0f0f0 });
const matBlack = new THREE.MeshStandardMaterial({ color: 0x111111 });
const matOn    = new THREE.MeshStandardMaterial({ color: 0x40ff40 });

// gi=0 は base, gi=1..88 は key_geom_0..87 の順（生成順）
function updateKeyMaterials(noteCounts) {
  for (let i = 0; i < N_KEYS; i++) {
    const gi = 1 + i;
    const mesh = meshes[gi];
    if (!mesh) continue;
    const on = noteCounts[i] > 0;
    if (on) mesh.material = matOn;
    else mesh.material = keyInfos[i].black ? matBlack : matWhite;
  }
}

// --------------------
// MIDI → events
// --------------------
let events = []; // {time, keyIndex, on, velocity}
let noteCounts = new Int16Array(N_KEYS);

function buildEventsFromMidi(midiObj) {
  const ev = [];
  // トラック全体を混ぜる（ドラムch=9は除外）
  for (const tr of midiObj.tracks) {
    if (tr.channel === 9) continue;
    for (const n of tr.notes) {
      const keyIndex = n.midi - FIRST_MIDI;
      if (keyIndex < 0 || keyIndex >= N_KEYS) continue;
      ev.push({ time: n.time, keyIndex, on: true,  velocity: n.velocity ?? 0.8 });
      ev.push({ time: n.time + n.duration, keyIndex, on: false, velocity: n.velocity ?? 0.8 });
    }
  }
  ev.sort((a,b) => (a.time - b.time) || ((a.on===b.on)?0:(a.on? -1 : 1))); // 同時刻はON→OFF
  return ev;
}

// --------------------
// Audio (simple synth + optional soundfont)
// --------------------
let audioCtx = null;

// simple synth voices: keyIndex -> voice list
const synthVoices = Array.from({ length: N_KEYS }, () => []);
function keyToFreq(keyIndex) {
  const midi = FIRST_MIDI + keyIndex;
  return 440 * Math.pow(2, (midi - 69) / 12);
}
function ensureAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

let sfInstrument = null; // Soundfont instrument
async function ensureSoundfontPiano() {
  const ac = ensureAudioCtx();
  if (sfInstrument) return sfInstrument;

  const Soundfont = globalThis.Soundfont;
  if (!Soundfont) throw new Error("soundfont-player が読み込まれていません");

  UI.status.textContent = "Loading SoundFont piano…";
  // FluidR3_GM の例（軽め） :contentReference[oaicite:1]{index=1}
  sfInstrument = await Soundfont.instrument(ac, "acoustic_grand_piano", { soundfont: "FluidR3_GM" });
  UI.status.textContent = "Ready";
  return sfInstrument;
}

function noteOnAudio(keyIndex, velocity) {
  const ac = ensureAudioCtx();
  const useSF = UI.useSoundfont.checked;

  if (useSF && sfInstrument) {
    // soundfont-player は midi 番号で play できる :contentReference[oaicite:2]{index=2}
    const midi = FIRST_MIDI + keyIndex;
    const player = sfInstrument.play(midi, ac.currentTime, { gain: 0.7 * velocity });
    synthVoices[keyIndex].push({ kind: "sf", player });
    return;
  }

  // simple synth
  const osc = ac.createOscillator();
  const gain = ac.createGain();

  osc.type = "triangle";
  osc.frequency.value = keyToFreq(keyIndex);

  const now = ac.currentTime;
  const amp = 0.08 * velocity;

  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.linearRampToValueAtTime(amp, now + 0.01);

  osc.connect(gain).connect(ac.destination);
  osc.start(now);

  synthVoices[keyIndex].push({ kind: "osc", osc, gain });
}

function noteOffAudio(keyIndex) {
  const ac = audioCtx;
  if (!ac) return;

  const voices = synthVoices[keyIndex];
  while (voices.length) {
    const v = voices.pop();
    if (v.kind === "sf") {
      try { v.player.stop(); } catch {}
    } else {
      const now = ac.currentTime;
      try {
        v.gain.gain.cancelScheduledValues(now);
        v.gain.gain.setValueAtTime(Math.max(v.gain.gain.value, 0.0001), now);
        v.gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.03);
        v.osc.stop(now + 0.04);
      } catch {}
    }
  }
}

function stopAllAudio() {
  for (let i = 0; i < N_KEYS; i++) noteOffAudio(i);
  if (sfInstrument) {
    try { sfInstrument.stop(); } catch {}
  }
}

// --------------------
// Playback
// --------------------
let playing = false;
let songTime = 0;
let eventIdx = 0;

// 物理遅れゼロにするため、角度を qpos に直接入れる（少しだけ滑らかに）
const keyAngles = new Float32Array(N_KEYS);
const KEY_RESP = 70; // 大きいほど追従速い（見た目と音を揃えたいので速め）

function resetAll() {
  playing = false;
  songTime = 0;
  eventIdx = 0;
  noteCounts.fill(0);
  keyAngles.fill(0);
  stopAllAudio();

  // 角度をゼロに
  for (let i = 0; i < N_KEYS; i++) {
    data.qpos[i] = 0;
    data.qvel[i] = 0;
  }
  mujoco.mj_forward(model, data);
  updateKeyMaterials(noteCounts);
}

UI.speed.addEventListener("input", () => {
  UI.speedValue.textContent = `${Number(UI.speed.value).toFixed(2)}x`;
});

UI.playPause.addEventListener("click", async () => {
  try {
    const ac = ensureAudioCtx();
    if (ac.state === "suspended") await ac.resume();

    // SoundFont ON ならここでロード（初回だけ時間かかる）
    if (UI.useSoundfont.checked) {
      await ensureSoundfontPiano();
    } else {
      UI.status.textContent = "Ready";
    }

    playing = !playing;
    UI.playPause.textContent = playing ? "⏸ Pause" : "▶︎ Play";

    // pause時は音を止める（再開時は次のnote-onから）
    if (!playing) stopAllAudio();
  } catch (e) {
    console.error(e);
    UI.status.textContent = `Audio init error: ${e?.message ?? e}`;
  }
});

UI.stop.addEventListener("click", () => {
  resetAll();
  UI.playPause.textContent = "▶︎ Play";
});

UI.midiFile.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  try {
    UI.status.textContent = `Parsing MIDI… (${file.name})`;
    const buf = await file.arrayBuffer();
    const midi = new Midi(buf);
    events = buildEventsFromMidi(midi);

    UI.status.textContent = `Loaded: ${file.name} / events=${events.length}`;
    resetAll();
    playing = true;
    UI.playPause.textContent = "⏸ Pause";
  } catch (err) {
    console.error(err);
    UI.status.textContent = `MIDI parse error: ${err?.message ?? err}`;
  }
});

// 初期デモ（Cメジャースケール）
(function initDemo() {
  const notes = [60,62,64,65,67,69,71,72];
  const dur = 0.22, gap = 0.02;
  let t = 0;
  events = [];
  for (const n of notes) {
    const k = n - FIRST_MIDI;
    if (k < 0 || k >= N_KEYS) continue;
    events.push({ time: t, keyIndex: k, on:true, velocity:0.9 });
    events.push({ time: t+dur, keyIndex: k, on:false, velocity:0.9 });
    t += dur + gap;
  }
  events.sort((a,b)=>a.time-b.time);
  UI.status.textContent = "Ready (demo scale)";
})();

// --------------------
// Render update from geom_xpos/xmat
// --------------------
function updateMeshesFromGeom() {
  const gx = data.geom_xpos;
  const gm = data.geom_xmat;

  for (let gi = 0; gi < ngeom; gi++) {
    const m = meshes[gi];
    if (!m) continue;

    m.position.set(gx[3*gi+0], gx[3*gi+1], gx[3*gi+2]);

    const r00 = gm[9*gi+0], r01 = gm[9*gi+1], r02 = gm[9*gi+2];
    const r10 = gm[9*gi+3], r11 = gm[9*gi+4], r12 = gm[9*gi+5];
    const r20 = gm[9*gi+6], r21 = gm[9*gi+7], r22 = gm[9*gi+8];

    const mat4 = new THREE.Matrix4().set(
      r00, r01, r02, 0,
      r10, r11, r12, 0,
      r20, r21, r22, 0,
      0,   0,   0,   1
    );
    m.setRotationFromMatrix(mat4);
  }
}

// --------------------
// Main loop
// --------------------
function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

let last = performance.now();

function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  const realDt = Math.min(0.05, (now - last) / 1000);
  last = now;

  if (playing) {
    const speed = Number(UI.speed.value);
    songTime += realDt * speed;

    // events を songTime まで適用
    while (eventIdx < events.length && events[eventIdx].time <= songTime) {
      const ev = events[eventIdx++];
      const i = ev.keyIndex;

      if (ev.on) {
        const prev = noteCounts[i];
        noteCounts[i] = prev + 1;
        if (prev === 0) noteOnAudio(i, ev.velocity ?? 0.8);
      } else {
        const prev = noteCounts[i];
        noteCounts[i] = Math.max(0, prev - 1);
        if (prev === 1 && noteCounts[i] === 0) noteOffAudio(i);
      }
    }

    // 鍵盤角度を即時追従（わずかにスムージング）
    const alpha = 1 - Math.exp(-KEY_RESP * realDt);
    for (let i = 0; i < N_KEYS; i++) {
      const target = noteCounts[i] > 0 ? keyInfos[i].pressAngle : 0;
      keyAngles[i] += (target - keyAngles[i]) * alpha;

      data.qpos[i] = keyAngles[i];
      data.qvel[i] = 0;
    }
    mujoco.mj_forward(model, data);

    updateKeyMaterials(noteCounts);

    // 曲終わり
    if (eventIdx >= events.length) {
      playing = false;
      UI.playPause.textContent = "▶︎ Play";
      stopAllAudio();
    }
  } else {
    // not playing: forward & render only
    mujoco.mj_forward(model, data);
  }

  updateMeshesFromGeom();
  controls.update();
  renderer.render(scene, camera);
}

animate();
