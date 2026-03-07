import {
  add,
  clamp,
  distance,
  formatQuat,
  formatVec,
  lerp,
  normalize,
  orthonormalBasis,
  polylineLength,
  quatFromUnitVectors,
  rotateBasisAroundAxis,
  scale,
  sub,
} from './math.js';

class Rng {
  constructor(seed = 1) {
    this.state = seed >>> 0;
    if (this.state === 0) this.state = 1;
  }

  next() {
    let t = this.state += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  float(min = 0, max = 1) {
    return lerp(min, max, this.next());
  }

  int(min, max) {
    return Math.floor(this.float(min, max + 1));
  }

  pick(array) {
    return array[Math.floor(this.next() * array.length)];
  }

  sign() {
    return this.next() < 0.5 ? -1 : 1;
  }
}

function pushPoint(points, point) {
  const last = points[points.length - 1];
  if (!last || distance(last, point) > 1e-6) {
    points.push(point);
  }
}

function collectSegments(points) {
  const segments = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    const a = points[i];
    const b = points[i + 1];
    const delta = sub(b, a);
    const length = distance(a, b);
    if (length < 1e-6) continue;
    segments.push({ index: i, a, b, dir: normalize(delta), length });
  }
  return segments;
}

function buildPath(seed, complexity) {
  const rng = new Rng(seed);
  const modules = 3 + Math.floor(complexity * 0.9);
  const lateralLimit = 0.22 + complexity * 0.045;
  const verticalLimit = 0.16 + complexity * 0.03;
  const points = [[0, 0, 0]];

  let x = 0;
  let y = 0;
  let z = 0;

  for (let i = 0; i < modules; i += 1) {
    x += 0.6 + rng.float(0.0, 0.18 + complexity * 0.02);
    pushPoint(points, [x, y, z]);

    if (i === modules - 1) break;

    const dy = rng.sign() * rng.float(0.14, 0.2 + complexity * 0.03);
    y = clamp(y + dy, -lateralLimit, lateralLimit);
    pushPoint(points, [x, y, z]);

    if (complexity >= 3) {
      const dz = rng.sign() * rng.float(0.08, 0.14 + complexity * 0.025);
      z = clamp(z + dz, -verticalLimit, verticalLimit);
      pushPoint(points, [x, y, z]);
    }

    if (complexity >= 7 && rng.next() < 0.45) {
      const dy2 = rng.sign() * rng.float(0.06, 0.12 + complexity * 0.02);
      y = clamp(y + dy2, -lateralLimit, lateralLimit);
      pushPoint(points, [x, y, z]);
    }
  }

  // Add a final straight exit section.
  x += 0.7 + complexity * 0.03;
  pushPoint(points, [x, y, z]);

  // Center the puzzle around the origin for nicer camera control.
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const zs = points.map((p) => p[2]);
  const xMid = (Math.min(...xs) + Math.max(...xs)) * 0.5;
  const yMid = (Math.min(...ys) + Math.max(...ys)) * 0.5;
  const zMid = (Math.min(...zs) + Math.max(...zs)) * 0.5;

  return points.map((p) => [p[0] - xMid, p[1] - yMid, p[2] - zMid]);
}

function buildGates(pathPoints, ring, wire, complexity, seed) {
  const rng = new Rng(seed ^ 0x9E3779B9);
  const segments = collectSegments(pathPoints);
  const candidates = segments
    .filter((segment, idx) => idx > 0 && idx < segments.length - 1 && segment.length > 0.54)
    .slice();

  const gateTarget = clamp(1 + Math.floor(complexity * 0.8), 1, candidates.length);
  const gates = [];
  const used = new Set();
  const openingWidth = ring.tubeRadius * 2.35 + 0.028;
  const openingHeight = (ring.radius + ring.tubeRadius) * 2 + 0.065;
  const frameRadius = wire.radius * 0.78;

  while (gates.length < gateTarget && used.size < candidates.length) {
    const pickIndex = rng.int(0, candidates.length - 1);
    if (used.has(pickIndex)) continue;
    used.add(pickIndex);

    const segment = candidates[pickIndex];
    const centerT = rng.float(0.38, 0.62);
    const center = add(segment.a, scale(sub(segment.b, segment.a), centerT));
    const base = orthonormalBasis(segment.dir);
    const roll = rng.float(0, Math.PI * 2);
    const rotated = rotateBasisAroundAxis(base.u, base.v, roll);

    gates.push({
      center,
      tangent: segment.dir,
      u: rotated.u,
      v: rotated.v,
      openingWidth,
      openingHeight,
      frameRadius,
      segmentIndex: segment.index,
    });
  }

  return gates.sort((a, b) => a.segmentIndex - b.segmentIndex);
}

function buildRingSegments(ring) {
  const points = [];
  for (let i = 0; i < ring.segmentCount; i += 1) {
    const angle = (i / ring.segmentCount) * Math.PI * 2;
    points.push([
      Math.cos(angle) * ring.radius,
      Math.sin(angle) * ring.radius,
      0,
    ]);
  }

  const geoms = [];
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    geoms.push({ a, b });
  }
  return geoms;
}

function buildGateFrameSegments(gate) {
  const { center, u, v, openingWidth, openingHeight, frameRadius } = gate;
  const halfW = openingWidth * 0.5;
  const halfH = openingHeight * 0.5;

  const left = add(center, scale(u, -(halfW + frameRadius)));
  const right = add(center, scale(u, +(halfW + frameRadius)));
  const top = add(center, scale(v, +(halfH + frameRadius)));
  const bottom = add(center, scale(v, -(halfH + frameRadius)));

  return [
    {
      a: add(left, scale(v, -halfH)),
      b: add(left, scale(v, +halfH)),
    },
    {
      a: add(right, scale(v, -halfH)),
      b: add(right, scale(v, +halfH)),
    },
    {
      a: add(top, scale(u, -halfW)),
      b: add(top, scale(u, +halfW)),
    },
    {
      a: add(bottom, scale(u, -halfW)),
      b: add(bottom, scale(u, +halfW)),
    },
  ];
}

export function generatePuzzleSpec({ seed = 42, complexity = 5 } = {}) {
  const safeComplexity = clamp(Math.round(complexity), 1, 10);
  const pathPoints = buildPath(seed, safeComplexity);
  const wire = {
    radius: 0.032 + safeComplexity * 0.0008,
  };
  const ring = {
    radius: 0.18,
    tubeRadius: 0.028,
    segmentCount: 20,
  };
  const gates = buildGates(pathPoints, ring, wire, safeComplexity, seed);
  const ringSegments = buildRingSegments(ring);

  const firstSegment = normalize(sub(pathPoints[1], pathPoints[0]));
  const lastSegment = normalize(sub(pathPoints[pathPoints.length - 1], pathPoints[pathPoints.length - 2]));

  const startPosition = add(pathPoints[0], scale(firstSegment, 0.25));
  const startQuat = quatFromUnitVectors([0, 0, 1], firstSegment);

  const startStopper = {
    pos: add(pathPoints[0], scale(firstSegment, -0.07)),
    radius: ring.radius - ring.tubeRadius * 0.55,
  };

  const exitMarker = {
    pos: add(pathPoints[pathPoints.length - 1], scale(lastSegment, 0.22)),
    radius: ring.radius * 0.24,
  };

  const wireLength = polylineLength(pathPoints);
  const turns = Math.max(0, pathPoints.length - 2);

  return {
    seed,
    complexity: safeComplexity,
    ring,
    wire,
    ringSegments,
    pathPoints,
    gates,
    startPosition,
    startQuat,
    startStopper,
    exitPoint: pathPoints[pathPoints.length - 1],
    exitDir: lastSegment,
    exitThreshold: ring.radius + 0.18,
    exitMarker,
    stats: {
      turns,
      gateCount: gates.length,
      wireLength,
    },
  };
}

function rgbaString(r, g, b, a) {
  return `${r.toFixed(4)} ${g.toFixed(4)} ${b.toFixed(4)} ${a.toFixed(4)}`;
}

export function buildPuzzleMjcf(spec) {
  const {
    ring,
    wire,
    pathPoints,
    gates,
    ringSegments,
    startPosition,
    startQuat,
    startStopper,
    exitMarker,
  } = spec;

  const wireGeoms = [];
  for (let i = 0; i < pathPoints.length - 1; i += 1) {
    wireGeoms.push(`      <geom name="wire_${i}" type="capsule" fromto="${formatVec(pathPoints[i])} ${formatVec(pathPoints[i + 1])}" size="${wire.radius.toFixed(6)}" rgba="${rgbaString(0.76, 0.78, 0.83, 1)}" friction="0.22 0.004 0.0002"/>`);
  }

  const gateGeoms = [];
  gates.forEach((gate, gateIndex) => {
    const segments = buildGateFrameSegments(gate);
    segments.forEach((segment, segmentIndex) => {
      gateGeoms.push(`      <geom name="gate_${gateIndex}_${segmentIndex}" type="capsule" fromto="${formatVec(segment.a)} ${formatVec(segment.b)}" size="${gate.frameRadius.toFixed(6)}" rgba="${rgbaString(0.96, 0.62, 0.36, 0.92)}" friction="0.22 0.004 0.0002"/>`);
    });
  });

  const ringGeoms = ringSegments.map((segment, i) => `        <geom name="ring_${i}" type="capsule" fromto="${formatVec(segment.a)} ${formatVec(segment.b)}" size="${ring.tubeRadius.toFixed(6)}" density="380" rgba="${rgbaString(0.28, 0.86, 0.98, 1)}" friction="0.18 0.002 0.0001"/>`);

  return `
<mujoco model="wire_disentanglement">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
  <option timestep="0.0025" gravity="0 0 0" iterations="70" integrator="implicitfast"/>
  <size nconmax="4096" njmax="8192"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <map fogstart="5" fogend="12" force="0.005"/>
    <rgba haze="0.03 0.04 0.06 1"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.06 0.08 0.12" rgb2="0.00 0.00 0.00" width="512" height="3072"/>
    <material name="wireMat" rgba="0.78 0.80 0.84 1"/>
    <material name="ringMat" rgba="0.28 0.86 0.98 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="1 1 1" specular="0.2 0.2 0.2"/>
    <light pos="-2 -2 2" directional="false" diffuse="0.5 0.6 0.7"/>
    <geom name="ambient_floor" type="plane" pos="0 0 -2.0" size="8 8 0.1" rgba="0.06 0.07 0.09 1" contype="0" conaffinity="0"/>
    <geom name="start_stop" type="sphere" pos="${formatVec(startStopper.pos)}" size="${startStopper.radius.toFixed(6)}" rgba="${rgbaString(0.92, 0.38, 0.38, 1)}" friction="0.22 0.004 0.0002"/>
    <geom name="exit_marker" type="sphere" pos="${formatVec(exitMarker.pos)}" size="${exitMarker.radius.toFixed(6)}" rgba="${rgbaString(0.42, 1.0, 0.72, 0.4)}" contype="0" conaffinity="0"/>
${wireGeoms.join('\n')}
${gateGeoms.join('\n')}

    <body name="player" pos="${formatVec(startPosition)}" quat="${formatQuat(startQuat)}">
      <joint name="player_free" type="free" damping="0.35"/>
${ringGeoms.join('\n')}
    </body>
  </worldbody>
</mujoco>
`.trim();
}
