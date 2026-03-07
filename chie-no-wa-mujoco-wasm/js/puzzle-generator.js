import {
  add,
  averagePoint,
  clamp,
  clamp01,
  distance,
  dot,
  formatQuat,
  formatVec,
  length,
  lerp,
  normalize,
  pointsToSegments,
  polylineLength,
  quatFromEulerXYZ,
  radiansToDegrees,
  scale,
  sub,
  transformDirection,
  transformPoint,
  transformSegments,
  minSegmentDistance,
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

  sign() {
    return this.next() < 0.5 ? -1 : 1;
  }

  pick(values) {
    return values[Math.floor(this.next() * values.length)];
  }
}

const FAMILY_LIBRARY = [
  {
    id: 'alpha-hook',
    label: 'alpha-hook',
    description: '主ループに片側フック尾を持つ基本系。ギャップ合わせとひねりの両方が必要。',
    primaryLobes: 1,
    radialWave: 0.05,
    zLift: 0.18,
    tailA: 'hook',
    tailB: 'straight',
    twistBias: 1.0,
    tailScale: 1.0,
  },
  {
    id: 'alpha-zig',
    label: 'alpha-zig',
    description: '両端にジグザグ尾を持つ系。姿勢自由度が増え、開始姿勢の抜け道が減る。',
    primaryLobes: 1,
    radialWave: 0.02,
    zLift: 0.24,
    tailA: 'zigzag',
    tailB: 'zigzag',
    twistBias: 1.2,
    tailScale: 1.1,
  },
  {
    id: 'alpha-lobed',
    label: 'alpha-lobed',
    description: '主ループ自体に外側ローブを持つ系。見た目は素朴だが有効開口が読みづらい。',
    primaryLobes: 3,
    radialWave: 0.12,
    zLift: 0.14,
    tailA: 'straight',
    tailB: 'straight',
    twistBias: 0.95,
    tailScale: 0.9,
  },
  {
    id: 'alpha-guard',
    label: 'alpha-guard',
    description: '片側にガード状の二重尾を持つ系。開口の向きを合わせても一手で抜けにくい。',
    primaryLobes: 2,
    radialWave: 0.08,
    zLift: 0.22,
    tailA: 'guard',
    tailB: 'hook',
    twistBias: 1.35,
    tailScale: 1.15,
  },
];

function pushPoint(points, point) {
  const last = points[points.length - 1];
  if (!last || distance(last, point) > 1e-6) {
    points.push(point);
  }
}

function estimatePrimaryLoopRadius(points) {
  if (!points.length) return 0;
  return points.reduce((sum, point) => sum + Math.hypot(point[0], point[1]), 0) / points.length;
}

function buildTail(style, endpoint, out, side, normal, baseLength, handedness = 1, bias = 1) {
  const sign = handedness;
  const lengthA = baseLength * bias;
  if (style === 'straight') {
    return [
      add(endpoint, add(scale(out, lengthA * 0.45), scale(side, sign * lengthA * 0.08))),
      add(endpoint, add(scale(out, lengthA), scale(normal, lengthA * 0.10))),
    ];
  }
  if (style === 'hook') {
    return [
      add(endpoint, add(scale(out, lengthA * 0.25), scale(side, sign * lengthA * 0.12))),
      add(endpoint, add(add(scale(out, lengthA * 0.62), scale(side, sign * lengthA * 0.36)), scale(normal, lengthA * 0.18))),
      add(endpoint, add(add(scale(out, lengthA * 0.95), scale(side, sign * lengthA * 0.48)), scale(normal, lengthA * 0.28))),
    ];
  }
  if (style === 'zigzag') {
    return [
      add(endpoint, add(add(scale(out, lengthA * 0.24), scale(side, sign * lengthA * 0.16)), scale(normal, lengthA * 0.14))),
      add(endpoint, add(add(scale(out, lengthA * 0.58), scale(side, sign * lengthA * -0.12)), scale(normal, lengthA * 0.30))),
      add(endpoint, add(add(scale(out, lengthA * 0.92), scale(side, sign * lengthA * 0.22)), scale(normal, lengthA * 0.05))),
    ];
  }
  if (style === 'guard') {
    return [
      add(endpoint, add(add(scale(out, lengthA * 0.20), scale(side, sign * lengthA * 0.20)), scale(normal, lengthA * 0.08))),
      add(endpoint, add(add(scale(out, lengthA * 0.46), scale(side, sign * lengthA * 0.42)), scale(normal, lengthA * 0.18))),
      add(endpoint, add(add(scale(out, lengthA * 0.76), scale(side, sign * lengthA * 0.18)), scale(normal, lengthA * 0.34))),
      add(endpoint, add(add(scale(out, lengthA * 1.02), scale(side, sign * lengthA * 0.34)), scale(normal, lengthA * 0.46))),
    ];
  }

  return [
    add(endpoint, scale(out, baseLength * 0.5)),
    add(endpoint, scale(out, baseLength)),
  ];
}

function samplePrimaryLoop({
  radius,
  wireRadius,
  desiredGapRatio,
  samples,
  lobes,
  radialWave,
  zLift,
  phase,
  xScale,
  yScale,
}) {
  const gapWidth = desiredGapRatio * (wireRadius * 2);
  const safeGapWidth = clamp(gapWidth, wireRadius * 2.2, radius * 0.92);
  const gapAngle = 2 * Math.asin(clamp(safeGapWidth / (2 * radius), 0.04, 0.65));
  const start = gapAngle * 0.5;
  const end = Math.PI * 2 - gapAngle * 0.5;
  const points = [];

  for (let i = 0; i < samples; i += 1) {
    const u = i / (samples - 1);
    const t = lerp(start, end, u);
    const envelope = Math.sin(Math.PI * u);
    const radial = radius * (1 + radialWave * 0.5 * Math.cos(lobes * t + phase));
    const x = radial * xScale * Math.cos(t);
    const y = radial * yScale * Math.sin(t);
    const z = zLift * envelope * Math.sin((lobes + 0.65) * t + phase * 0.5);
    points.push([x, y, z]);
  }

  const loopCenter = averagePoint(points);
  const centered = points.map((point) => sub(point, loopCenter));
  const loopStart = centered[0];
  const loopEnd = centered[centered.length - 1];
  const gapCenter = scale(add(loopStart, loopEnd), 0.5);
  const gapOut = normalize(gapCenter);
  const primaryLoopRadius = estimatePrimaryLoopRadius(centered);
  const tangents = {
    start: normalize(sub(centered[1], centered[0])),
    end: normalize(sub(centered[centered.length - 1], centered[centered.length - 2])),
  };

  return {
    points: centered,
    primaryLoopRadius,
    gapAngle,
    gapWidth: distance(loopStart, loopEnd),
    loopCenterLocal: [0, 0, 0],
    loopNormalLocal: [0, 0, 1],
    gap: {
      aLocal: loopStart,
      bLocal: loopEnd,
      centerLocal: gapCenter,
      outLocal: gapOut,
    },
    tangents,
  };
}

function mirrorY(points, sign) {
  if (sign > 0) return points;
  return points.map((point) => [point[0], -point[1], point[2]]);
}

function buildPieceGeometry(pieceIR) {
  const loop = samplePrimaryLoop({
    radius: pieceIR.loopRadius,
    wireRadius: pieceIR.wireRadius,
    desiredGapRatio: pieceIR.gapRatio,
    samples: pieceIR.samples,
    lobes: pieceIR.primaryLobes,
    radialWave: pieceIR.radialWave,
    zLift: pieceIR.zLift,
    phase: pieceIR.phase,
    xScale: pieceIR.xScale,
    yScale: pieceIR.yScale,
  });

  const loopStart = loop.points[0];
  const loopEnd = loop.points[loop.points.length - 1];
  const outStart = normalize(loopStart);
  const outEnd = normalize(loopEnd);
  const sideStart = normalize([-outStart[1], outStart[0], 0]);
  const sideEnd = normalize([-outEnd[1], outEnd[0], 0]);
  const normal = [0, 0, 1];
  const tailLength = pieceIR.tailLength;

  const prefixTail = buildTail(
    pieceIR.tailStyleA,
    loopStart,
    outStart,
    sideStart,
    normal,
    tailLength * pieceIR.tailScaleA,
    pieceIR.handedness,
    pieceIR.tailBiasA,
  );

  const suffixTail = buildTail(
    pieceIR.tailStyleB,
    loopEnd,
    outEnd,
    sideEnd,
    normal,
    tailLength * pieceIR.tailScaleB,
    -pieceIR.handedness,
    pieceIR.tailBiasB,
  );

  const points = [];
  mirrorY(prefixTail, pieceIR.handedness).slice().reverse().forEach((point) => pushPoint(points, point));
  mirrorY(loop.points, pieceIR.handedness).forEach((point) => pushPoint(points, point));
  mirrorY(suffixTail, pieceIR.handedness).forEach((point) => pushPoint(points, point));

  const transformedGapA = mirrorY([loop.gap.aLocal], pieceIR.handedness)[0];
  const transformedGapB = mirrorY([loop.gap.bLocal], pieceIR.handedness)[0];
  const transformedGapCenter = mirrorY([loop.gap.centerLocal], pieceIR.handedness)[0];
  const transformedGapOut = normalize(mirrorY([loop.gap.outLocal], pieceIR.handedness)[0]);

  const piece = {
    ...pieceIR,
    pointsLocal: points,
    segmentsLocal: pointsToSegments(points),
    primaryLoopPointsLocal: mirrorY(loop.points, pieceIR.handedness),
    primaryLoopRadius: loop.primaryLoopRadius,
    loopNormalLocal: pieceIR.handedness > 0 ? [0, 0, 1] : [0, 0, 1],
    holeCenterLocal: [0, 0, 0],
    gap: {
      ...loop.gap,
      width: loop.gapWidth,
      angleDeg: radiansToDegrees(loop.gapAngle),
      aLocal: transformedGapA,
      bLocal: transformedGapB,
      centerLocal: transformedGapCenter,
      outLocal: transformedGapOut,
    },
    wireLength: polylineLength(points),
    extents: computeExtents(points),
  };

  return piece;
}

function computeExtents(points) {
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const zs = points.map((point) => point[2]);
  return {
    min: [Math.min(...xs), Math.min(...ys), Math.min(...zs)],
    max: [Math.max(...xs), Math.max(...ys), Math.max(...zs)],
  };
}

function chooseFamily(rng, complexityNorm) {
  const bucket = complexityNorm < 0.35
    ? FAMILY_LIBRARY.slice(0, 3)
    : complexityNorm < 0.7
      ? FAMILY_LIBRARY
      : [FAMILY_LIBRARY[0], FAMILY_LIBRARY[1], FAMILY_LIBRARY[3], FAMILY_LIBRARY[2]];
  return rng.pick(bucket);
}

function buildPieceIR(role, family, wireRadius, radius, complexityNorm, rng) {
  const phase = rng.float(0, Math.PI * 2);
  const gapRatio = lerp(2.15, 1.42, complexityNorm) * rng.float(0.97, 1.03);
  const tailLength = radius * lerp(0.40, 0.62, complexityNorm) * family.tailScale * rng.float(0.95, 1.08);
  return {
    role,
    familyId: family.id,
    wireRadius,
    loopRadius: radius,
    gapRatio,
    samples: 36 + Math.round(complexityNorm * 14),
    primaryLobes: family.primaryLobes,
    radialWave: family.radialWave * rng.float(0.9, 1.1),
    zLift: radius * family.zLift * rng.float(0.92, 1.08),
    phase,
    xScale: rng.float(0.96, 1.08),
    yScale: rng.float(0.94, 1.06),
    tailLength,
    tailStyleA: family.tailA,
    tailStyleB: family.tailB,
    tailScaleA: rng.float(0.88, 1.18),
    tailScaleB: rng.float(0.88, 1.18),
    tailBiasA: rng.float(0.95, 1.1),
    tailBiasB: rng.float(0.95, 1.1),
    handedness: rng.sign(),
    twistBias: family.twistBias * rng.float(0.95, 1.08),
  };
}

function buildIntrinsicPuzzle(seed, complexity) {
  const rng = new Rng(seed);
  const complexityNorm = (complexity - 1) / 9;
  const family = chooseFamily(rng, complexityNorm);
  const wireRadius = lerp(0.027, 0.022, complexityNorm);
  const fixedRadius = lerp(0.43, 0.52, complexityNorm) * rng.float(0.97, 1.03);
  const movingRadius = fixedRadius * rng.float(0.88, 0.95);

  const fixedIR = buildPieceIR('fixed', family, wireRadius, fixedRadius, complexityNorm, rng);
  const movingIR = buildPieceIR('moving', family, wireRadius, movingRadius, complexityNorm, rng);

  return {
    seed,
    complexity,
    complexityNorm,
    family,
    wireRadius,
    pieces: { fixed: fixedIR, moving: movingIR },
    nodes: [
      { id: 'F.loop0', type: 'loop', piece: 'fixed' },
      { id: 'F.gap0', type: 'gap', piece: 'fixed', on: 'F.loop0' },
      { id: 'M.loop0', type: 'loop', piece: 'moving' },
      { id: 'M.gap0', type: 'gap', piece: 'moving', on: 'M.loop0' },
    ],
    relations: [
      { type: 'threaded-through', source: 'M.loop0', target: 'F.loop0' },
      { type: 'solve-step', value: 'align-gaps' },
      { type: 'solve-step', value: 'twist-and-slide' },
    ],
  };
}

function holeContainmentScore(center, diskCenter, diskNormal, diskRadius) {
  const rel = sub(center, diskCenter);
  const axial = Math.abs(dot(rel, diskNormal));
  const radialVector = sub(rel, scale(diskNormal, dot(rel, diskNormal)));
  const radial = length(radialVector);
  const axialScore = clamp01(1 - axial / (diskRadius * 0.42));
  const radialScore = clamp01(1 - radial / (diskRadius * 1.02));
  return axialScore * radialScore;
}

function getPoseDescriptor(piece, pos, quat) {
  return {
    holeCenter: transformPoint(piece.holeCenterLocal, pos, quat),
    holeNormal: transformDirection(piece.loopNormalLocal, quat),
    gapCenter: transformPoint(piece.gap.centerLocal, pos, quat),
    gapOut: transformDirection(piece.gap.outLocal, quat),
    gapA: transformPoint(piece.gap.aLocal, pos, quat),
    gapB: transformPoint(piece.gap.bLocal, pos, quat),
    radius: piece.primaryLoopRadius,
  };
}

function scoreStartPose(spec, pos, quat) {
  const fixedPose = getPoseDescriptor(spec.fixed, [0, 0, 0], [1, 0, 0, 0]);
  const movingPose = getPoseDescriptor(spec.moving, pos, quat);
  const movingSegments = transformSegments(spec.moving.segmentsLocal, pos, quat);
  const interDistance = minSegmentDistance(spec.fixed.segmentsLocal, movingSegments).distance;
  const clearance = interDistance - 2 * spec.wire.radius;
  const mutualHole = 0.5 * (
    holeContainmentScore(movingPose.holeCenter, fixedPose.holeCenter, fixedPose.holeNormal, fixedPose.radius)
    + holeContainmentScore(fixedPose.holeCenter, movingPose.holeCenter, movingPose.holeNormal, movingPose.radius)
  );
  const normalDot = Math.abs(dot(fixedPose.holeNormal, movingPose.holeNormal));
  const orthogonality = 1 - normalDot;
  const centerDistance = distance(fixedPose.holeCenter, movingPose.holeCenter);
  const centerBand = clamp01(1 - Math.abs(centerDistance - spec.fixed.primaryLoopRadius * 0.58) / (spec.fixed.primaryLoopRadius * 0.34));
  const gapMisalignment = clamp01(0.5 * (1 - dot(fixedPose.gapOut, movingPose.gapOut)));
  const clearanceScore = clearance < 0
    ? -20 - 80 * Math.abs(clearance)
    : clamp01(clearance / (spec.wire.radius * 1.8));

  const score = mutualHole * 3.4 + orthogonality * 1.5 + centerBand * 0.8 + gapMisalignment * 0.6 + clearanceScore;

  return {
    score,
    clearance,
    interDistance,
    orthogonality,
    mutualHole,
    centerDistance,
    fixedPose,
    movingPose,
  };
}

function findStartPose(spec, rng) {
  const radius = spec.fixed.primaryLoopRadius;
  const pitchValues = [-0.14, 0, 0.14];
  const yawValues = [-0.42, -0.18, 0, 0.18, 0.42];
  const rollValues = [-0.26, 0, 0.26];
  const xValues = [0.46, 0.56, 0.66, 0.76].map((v) => v * radius);
  const yValues = [-0.16, 0, 0.16].map((v) => v * radius);
  const zValues = [-0.16, 0, 0.16].map((v) => v * radius);

  let best = null;

  for (const pitchDelta of pitchValues) {
    for (const yaw of yawValues) {
      for (const roll of rollValues) {
        const quat = quatFromEulerXYZ(Math.PI * 0.5 + pitchDelta, yaw, roll);
        for (const x of xValues) {
          for (const y of yValues) {
            for (const z of zValues) {
              const pos = [x, y, z];
              const candidate = scoreStartPose(spec, pos, quat);
              if (!best || candidate.score > best.score) {
                best = {
                  ...candidate,
                  pos,
                  quat,
                };
              }
            }
          }
        }
      }
    }
  }

  if (!best) {
    return {
      pos: [radius * 0.58, 0, 0],
      quat: quatFromEulerXYZ(Math.PI * 0.5, 0, 0),
      clearance: -Infinity,
      score: -Infinity,
    };
  }

  const refinedPos = best.pos.slice();
  let refinedClearance = best.clearance;
  let guard = 0;
  while (refinedClearance < spec.wire.radius * 0.10 && guard < 30) {
    refinedPos[0] += spec.wire.radius * 0.18;
    const rescored = scoreStartPose(spec, refinedPos, best.quat);
    refinedClearance = rescored.clearance;
    best = {
      ...best,
      ...rescored,
      pos: refinedPos.slice(),
    };
    guard += 1;
  }

  return best;
}

function estimateDifficulty(spec, startPose) {
  const complexityBase = spec.complexity * 0.65;
  const gapTightness = clamp01((2.2 - spec.stats.avgGapRatio) / 0.9) * 2.2;
  const twistiness = clamp01(spec.stats.twistProxy / 4.2) * 2.8;
  const startTightness = clamp01((spec.wire.radius * 0.65 - startPose.clearance) / (spec.wire.radius * 0.65)) * 1.8;
  return clamp(Math.round(complexityBase + gapTightness + twistiness + startTightness), 1, 10);
}

function makeGenerationLog(spec, startPose) {
  return [
    `family=${spec.family.id}`,
    `complexity=${spec.complexity}`,
    `fixedRadius=${spec.fixed.primaryLoopRadius.toFixed(3)} m`,
    `movingRadius=${spec.moving.primaryLoopRadius.toFixed(3)} m`,
    `avgGapRatio=${spec.stats.avgGapRatio.toFixed(2)}x wire diameter`,
    `twistProxy=${spec.stats.twistProxy.toFixed(2)}`,
    `startClearance=${startPose.clearance.toFixed(4)} m`,
    `startCenterDistance=${startPose.centerDistance.toFixed(3)} m`,
    `orthogonality=${(startPose.orthogonality * 100).toFixed(1)} %`,
  ];
}

function rgbaString(r, g, b, a) {
  return `${r.toFixed(4)} ${g.toFixed(4)} ${b.toFixed(4)} ${a.toFixed(4)}`;
}

function pieceGeomsXml(piece, radius, rgba, density, namePrefix) {
  return piece.segmentsLocal.map((segment, index) => (
    `      <geom name="${namePrefix}_${index}" type="capsule" fromto="${formatVec(segment.a)} ${formatVec(segment.b)}" size="${radius.toFixed(6)}" density="${density.toFixed(2)}" rgba="${rgba}" friction="0.35 0.01 0.0004"/>`
  )).join('\n');
}

export function generatePuzzleSpec({ seed = 42, complexity = 5 } = {}) {
  const safeComplexity = clamp(Math.round(complexity), 1, 10);
  const ir = buildIntrinsicPuzzle(seed, safeComplexity);

  const fixed = buildPieceGeometry(ir.pieces.fixed);
  const moving = buildPieceGeometry(ir.pieces.moving);

  const spec = {
    seed,
    complexity: safeComplexity,
    family: ir.family,
    ir,
    wire: {
      radius: ir.wireRadius,
      diameter: ir.wireRadius * 2,
    },
    fixed,
    moving,
  };

  const avgGapRatio = ((fixed.gap.width / spec.wire.diameter) + (moving.gap.width / spec.wire.diameter)) * 0.5;
  const twistProxy = (
    fixed.zLift / spec.wire.diameter
    + moving.zLift / spec.wire.diameter
    + fixed.primaryLobes * 0.4
    + moving.primaryLobes * 0.4
    + fixed.twistBias * 0.55
    + moving.twistBias * 0.55
  );

  const startPose = findStartPose(spec, new Rng(seed ^ 0x9E3779B9));
  const solveDistance = fixed.primaryLoopRadius + moving.primaryLoopRadius + Math.max(0.50, fixed.primaryLoopRadius * 0.72);

  spec.startPose = {
    pos: startPose.pos,
    quat: startPose.quat,
  };

  spec.geometry = {
    startCenterDistance: startPose.centerDistance,
    solveDistance,
  };

  spec.stats = {
    fixedWireLength: fixed.wireLength,
    movingWireLength: moving.wireLength,
    totalWireLength: fixed.wireLength + moving.wireLength,
    avgGapRatio,
    fixedGapAngleDeg: fixed.gap.angleDeg,
    movingGapAngleDeg: moving.gap.angleDeg,
    twistProxy,
    mechanismCount: ir.nodes.length + ir.relations.length,
  };

  spec.stats.estimatedDifficulty = estimateDifficulty(spec, startPose);
  spec.generationLog = makeGenerationLog(spec, startPose);
  spec.startPoseStats = {
    clearance: startPose.clearance,
    score: startPose.score,
    orthogonality: startPose.orthogonality,
    mutualHole: startPose.mutualHole,
    centerDistance: startPose.centerDistance,
  };

  return spec;
}

export function buildPuzzleMjcf(spec) {
  const fixedRgba = rgbaString(0.78, 0.81, 0.87, 1.0);
  const movingRgba = rgbaString(0.30, 0.86, 0.98, 1.0);

  return `
<mujoco model="topological_wire_disentanglement">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.0025" gravity="0 0 0" iterations="90" integrator="implicitfast"/>
  <size nconmax="8192" njmax="16384"/>

  <default>
    <geom condim="3" solref="0.004 1" solimp="0.93 0.985 0.001"/>
    <joint damping="0.18"/>
  </default>

  <visual>
    <global offwidth="1440" offheight="900"/>
    <map fogstart="5" fogend="13" force="0.005"/>
    <rgba haze="0.04 0.05 0.07 1"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.06 0.08 0.12" rgb2="0.00 0.00 0.00" width="512" height="3072"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.6" dir="0 0 -1" directional="true" diffuse="1 1 1" specular="0.2 0.2 0.2"/>
    <light pos="-2.4 -2.2 2.0" directional="false" diffuse="0.55 0.65 0.75"/>
    <geom name="ambient_floor" type="plane" pos="0 0 -2.2" size="10 10 0.1" rgba="0.05 0.06 0.08 1" contype="0" conaffinity="0"/>

${pieceGeomsXml(spec.fixed, spec.wire.radius, fixedRgba, 650, 'fixed')}

    <body name="moving" pos="${formatVec(spec.startPose.pos)}" quat="${formatQuat(spec.startPose.quat)}">
      <joint name="moving_free" type="free"/>
${pieceGeomsXml(spec.moving, spec.wire.radius, movingRgba, 420, 'moving')}
    </body>
  </worldbody>
</mujoco>
  `.trim();
}
