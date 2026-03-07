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
  minSegmentDistance,
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
} from './math.js';
import { computeThreadingMetrics } from './diagnostics.js';

function normalizeLocale(locale) {
  if (!locale) return 'ja';
  return String(locale).toLowerCase().startsWith('en') ? 'en' : 'ja';
}

class Rng {
  constructor(seed = 1) {
    this.state = seed >>> 0;
    if (this.state === 0) this.state = 1;
  }

  next() {
    let t = (this.state += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  float(min = 0, max = 1) {
    return lerp(min, max, this.next());
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
    labels: {
      ja: 'alpha-hook',
      en: 'Alpha Hook',
    },
    descriptions: {
      ja: '主ループに片側フック尾を持つ基本系。ギャップ合わせとひねりの両方が必要。',
      en: 'A basic family with a one-sided hooked tail. It usually requires both gap alignment and twisting.',
    },
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
    labels: {
      ja: 'alpha-zig',
      en: 'Alpha Zig',
    },
    descriptions: {
      ja: '両端にジグザグ尾を持つ系。姿勢自由度が増え、開始姿勢の抜け道が減る。',
      en: 'A family with zig-zag tails on both ends. It adds pose freedom and reduces trivial starting escapes.',
    },
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
    labels: {
      ja: 'alpha-lobed',
      en: 'Alpha Lobed',
    },
    descriptions: {
      ja: '主ループ自体に外側ローブを持つ系。見た目は素朴だが有効開口が読みづらい。',
      en: 'A family whose primary loop contains outward lobes. It looks simple but makes the effective opening harder to read.',
    },
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
    labels: {
      ja: 'alpha-guard',
      en: 'Alpha Guard',
    },
    descriptions: {
      ja: '片側にガード状の二重尾を持つ系。開口の向きを合わせても一手で抜けにくい。',
      en: 'A family with a guard-like double tail on one side. Even after aligning the opening, it rarely slips out in a single move.',
    },
    primaryLobes: 2,
    radialWave: 0.08,
    zLift: 0.22,
    tailA: 'guard',
    tailB: 'hook',
    twistBias: 1.35,
    tailScale: 1.15,
  },
];

export function localizeFamily(family, locale = 'ja') {
  const normalized = normalizeLocale(locale);
  return {
    ...family,
    label: family.labels?.[normalized] ?? family.id,
    description: family.descriptions?.[normalized] ?? family.id,
  };
}

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
  const scaledLength = baseLength * bias;

  if (style === 'straight') {
    return [
      add(endpoint, add(scale(out, scaledLength * 0.45), scale(side, sign * scaledLength * 0.08))),
      add(endpoint, add(scale(out, scaledLength), scale(normal, scaledLength * 0.10))),
    ];
  }

  if (style === 'hook') {
    return [
      add(endpoint, add(scale(out, scaledLength * 0.25), scale(side, sign * scaledLength * 0.12))),
      add(
        endpoint,
        add(
          add(scale(out, scaledLength * 0.62), scale(side, sign * scaledLength * 0.36)),
          scale(normal, scaledLength * 0.18),
        ),
      ),
      add(
        endpoint,
        add(
          add(scale(out, scaledLength * 0.95), scale(side, sign * scaledLength * 0.48)),
          scale(normal, scaledLength * 0.28),
        ),
      ),
    ];
  }

  if (style === 'zigzag') {
    return [
      add(
        endpoint,
        add(
          add(scale(out, scaledLength * 0.24), scale(side, sign * scaledLength * 0.16)),
          scale(normal, scaledLength * 0.14),
        ),
      ),
      add(
        endpoint,
        add(
          add(scale(out, scaledLength * 0.58), scale(side, sign * scaledLength * -0.12)),
          scale(normal, scaledLength * 0.30),
        ),
      ),
      add(
        endpoint,
        add(
          add(scale(out, scaledLength * 0.92), scale(side, sign * scaledLength * 0.22)),
          scale(normal, scaledLength * 0.05),
        ),
      ),
    ];
  }

  if (style === 'guard') {
    return [
      add(
        endpoint,
        add(
          add(scale(out, scaledLength * 0.20), scale(side, sign * scaledLength * 0.20)),
          scale(normal, scaledLength * 0.08),
        ),
      ),
      add(
        endpoint,
        add(
          add(scale(out, scaledLength * 0.46), scale(side, sign * scaledLength * 0.42)),
          scale(normal, scaledLength * 0.18),
        ),
      ),
      add(
        endpoint,
        add(
          add(scale(out, scaledLength * 0.76), scale(side, sign * scaledLength * 0.18)),
          scale(normal, scaledLength * 0.34),
        ),
      ),
      add(
        endpoint,
        add(
          add(scale(out, scaledLength * 1.02), scale(side, sign * scaledLength * 0.34)),
          scale(normal, scaledLength * 0.46),
        ),
      ),
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
  };
}

function mirrorY(points, sign) {
  if (sign > 0) return points;
  return points.map((point) => [point[0], -point[1], point[2]]);
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

function buildPieceGeometry(pieceIr) {
  const loop = samplePrimaryLoop({
    radius: pieceIr.loopRadius,
    wireRadius: pieceIr.wireRadius,
    desiredGapRatio: pieceIr.gapRatio,
    samples: pieceIr.samples,
    lobes: pieceIr.primaryLobes,
    radialWave: pieceIr.radialWave,
    zLift: pieceIr.zLift,
    phase: pieceIr.phase,
    xScale: pieceIr.xScale,
    yScale: pieceIr.yScale,
  });

  const loopStart = loop.points[0];
  const loopEnd = loop.points[loop.points.length - 1];
  const outStart = normalize(loopStart);
  const outEnd = normalize(loopEnd);
  const sideStart = normalize([-outStart[1], outStart[0], 0]);
  const sideEnd = normalize([-outEnd[1], outEnd[0], 0]);
  const normal = [0, 0, 1];
  const tailLength = pieceIr.tailLength;

  const prefixTail = buildTail(
    pieceIr.tailStyleA,
    loopStart,
    outStart,
    sideStart,
    normal,
    tailLength * pieceIr.tailScaleA,
    pieceIr.handedness,
    pieceIr.tailBiasA,
  );

  const suffixTail = buildTail(
    pieceIr.tailStyleB,
    loopEnd,
    outEnd,
    sideEnd,
    normal,
    tailLength * pieceIr.tailScaleB,
    -pieceIr.handedness,
    pieceIr.tailBiasB,
  );

  const points = [];
  mirrorY(prefixTail, pieceIr.handedness)
    .slice()
    .reverse()
    .forEach((point) => pushPoint(points, point));
  mirrorY(loop.points, pieceIr.handedness).forEach((point) => pushPoint(points, point));
  mirrorY(suffixTail, pieceIr.handedness).forEach((point) => pushPoint(points, point));

  const transformedGapA = mirrorY([loop.gap.aLocal], pieceIr.handedness)[0];
  const transformedGapB = mirrorY([loop.gap.bLocal], pieceIr.handedness)[0];
  const transformedGapCenter = mirrorY([loop.gap.centerLocal], pieceIr.handedness)[0];
  const transformedGapOut = normalize(mirrorY([loop.gap.outLocal], pieceIr.handedness)[0]);

  return {
    ...pieceIr,
    pointsLocal: points,
    segmentsLocal: pointsToSegments(points),
    primaryLoopPointsLocal: mirrorY(loop.points, pieceIr.handedness),
    primaryLoopRadius: loop.primaryLoopRadius,
    loopNormalLocal: [0, 0, 1],
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
}

function chooseFamily(rng, complexityNorm) {
  const bucket =
    complexityNorm < 0.35
      ? FAMILY_LIBRARY.slice(0, 3)
      : complexityNorm < 0.7
        ? FAMILY_LIBRARY
        : [FAMILY_LIBRARY[0], FAMILY_LIBRARY[1], FAMILY_LIBRARY[3], FAMILY_LIBRARY[2]];
  return rng.pick(bucket);
}

function buildPieceIr(role, family, wireRadius, radius, complexityNorm, rng) {
  const phase = rng.float(0, Math.PI * 2);
  const gapRatio = lerp(2.15, 1.42, complexityNorm) * rng.float(0.97, 1.03);
  const tailLength =
    radius *
    lerp(0.40, 0.62, complexityNorm) *
    family.tailScale *
    rng.float(0.95, 1.08);

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

  const fixedIr = buildPieceIr('fixed', family, wireRadius, fixedRadius, complexityNorm, rng);
  const movingIr = buildPieceIr('moving', family, wireRadius, movingRadius, complexityNorm, rng);

  return {
    seed,
    complexity,
    complexityNorm,
    family,
    wireRadius,
    pieces: {
      fixed: fixedIr,
      moving: movingIr,
    },
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
  const threading = computeThreadingMetrics(spec, { fixed: fixedPose, moving: movingPose }, movingSegments);

  const mutualHole =
    0.5 *
    (
      holeContainmentScore(
        movingPose.holeCenter,
        fixedPose.holeCenter,
        fixedPose.holeNormal,
        fixedPose.radius,
      ) +
      holeContainmentScore(
        fixedPose.holeCenter,
        movingPose.holeCenter,
        movingPose.holeNormal,
        movingPose.radius,
      )
    );

  const normalDot = Math.abs(dot(fixedPose.holeNormal, movingPose.holeNormal));
  const orthogonality = 1 - normalDot;
  const centerDistance = distance(fixedPose.holeCenter, movingPose.holeCenter);
  const centerBand = clamp01(
    1 - Math.abs(centerDistance - spec.fixed.primaryLoopRadius * 0.58) / (spec.fixed.primaryLoopRadius * 0.34),
  );
  const gapMisalignment = clamp01(0.5 * (1 - dot(fixedPose.gapOut, movingPose.gapOut)));
  const clearanceScore =
    clearance < 0
      ? -20 - 80 * Math.abs(clearance)
      : clamp01(clearance / (spec.wire.radius * 1.8));

  const score =
    threading.score * 2.4 +
    mutualHole * 2.2 +
    orthogonality * 1.2 +
    centerBand * 0.6 +
    gapMisalignment * 0.5 +
    clearanceScore * 0.4;

  return {
    score,
    clearance,
    interDistance,
    orthogonality,
    mutualHole,
    centerDistance,
    threadingCrossings: threading.nongapCrossings,
    threadingGapCrossings: threading.gapCrossings,
    threadingScore: threading.score,
    fixedPose,
    movingPose,
  };
}

function findStartPose(spec) {
  const radius = spec.fixed.primaryLoopRadius;
  const pitchValues = [-0.14, 0, 0.14];
  const yawValues = [-0.42, -0.18, 0, 0.18, 0.42];
  const rollValues = [-0.26, 0, 0.26];
  const xValues = [0.46, 0.56, 0.66, 0.76].map((value) => value * radius);
  const yValues = [-0.16, 0, 0.16].map((value) => value * radius);
  const zValues = [-0.16, 0, 0.16].map((value) => value * radius);

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
  const startTightness =
    clamp01((spec.wire.radius * 0.65 - startPose.clearance) / (spec.wire.radius * 0.65)) * 1.8;
  const startThreading = clamp01((startPose.threadingScore ?? 0) / 0.55) * 1.3;

  return clamp(
    Math.round(
      complexityBase + gapTightness + twistiness + startTightness + startThreading,
    ),
    1,
    10,
  );
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
    `startThreading=${(startPose.threadingScore ?? 0).toFixed(3)} crossings=${startPose.threadingCrossings ?? 0}`,
  ];
}

function rgbaString(r, g, b, a) {
  return `${r.toFixed(4)} ${g.toFixed(4)} ${b.toFixed(4)} ${a.toFixed(4)}`;
}

function indentLines(text, count) {
  const indent = ' '.repeat(count);
  return text
    .split('\n')
    .map((line) => `${indent}${line}`)
    .join('\n');
}

function pieceGeomsXml(piece, radius, rgba, density, namePrefix) {
  return piece.segmentsLocal
    .map((segment, index) => {
      const fromto = `${formatVec(segment.a)} ${formatVec(segment.b)}`;
      return [
        `<geom`,
        `  name="${namePrefix}-seg-${index}"`,
        `  type="capsule"`,
        `  fromto="${fromto}"`,
        `  size="${radius.toFixed(6)}"`,
        `  rgba="${rgba}"`,
        `  density="${density}"`,
        `  friction="0.60 0.02 0.002"`,
        `  solref="0.004 1"`,
        `  solimp="0.995 0.999 0.0005"`,
        `  condim="3"`,
        `/>`,
      ].join('\n');
    })
    .join('\n');
}

export function generatePuzzleSpec({ seed = 42, complexity = 5 } = {}) {
  const safeComplexity = clamp(Math.round(complexity), 1, 10);
  const ir = buildIntrinsicPuzzle(seed, safeComplexity);
  const fixed = buildPieceGeometry(ir.pieces.fixed);
  const moving = buildPieceGeometry(ir.pieces.moving);

  const spec = {
    seed,
    complexity: safeComplexity,
    family: { ...ir.family },
    ir,
    wire: {
      radius: ir.wireRadius,
      diameter: ir.wireRadius * 2,
    },
    fixed,
    moving,
  };

  const avgGapRatio =
    ((fixed.gap.width / spec.wire.diameter) + (moving.gap.width / spec.wire.diameter)) * 0.5;

  const twistProxy =
    fixed.zLift / spec.wire.diameter +
    moving.zLift / spec.wire.diameter +
    fixed.primaryLobes * 0.4 +
    moving.primaryLobes * 0.4 +
    fixed.twistBias * 0.55 +
    moving.twistBias * 0.55;

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

  const startPose = findStartPose(spec);
  const solveDistance =
    fixed.primaryLoopRadius + moving.primaryLoopRadius + Math.max(0.50, fixed.primaryLoopRadius * 0.72);
  const solveClearance = Math.max(
    startPose.clearance + Math.max(spec.wire.radius * 2.2, 0.05),
    spec.wire.radius * 5.0,
    0.12,
  );

  spec.startPose = {
    pos: startPose.pos,
    quat: startPose.quat,
  };

  spec.geometry = {
    startCenterDistance: startPose.centerDistance,
    solveDistance,
    solveCenterDistanceMin: solveDistance * 0.96,
    solveClearance,
    solveMutualHoleMax: 0.03,
    solveThreadingScoreMax: 0.02,
    solveHoldDuration: 0.35,
  };

  spec.stats.estimatedDifficulty = estimateDifficulty(spec, startPose);
  spec.generationLog = makeGenerationLog(spec, startPose);
  spec.startPoseStats = {
    clearance: startPose.clearance,
    score: startPose.score,
    orthogonality: startPose.orthogonality,
    mutualHole: startPose.mutualHole,
    centerDistance: startPose.centerDistance,
    threadingCrossings: startPose.threadingCrossings ?? 0,
    threadingGapCrossings: startPose.threadingGapCrossings ?? 0,
    threadingScore: startPose.threadingScore ?? 0,
  };

  return spec;
}

export function buildPuzzleMjcf(spec) {
  const fixedRgba = rgbaString(0.78, 0.81, 0.87, 1.0);
  const movingRgba = rgbaString(0.30, 0.86, 0.98, 1.0);

  const fixedXml = indentLines(pieceGeomsXml(spec.fixed, spec.wire.radius, fixedRgba, 650, 'fixed'), 4);
  const movingXml = indentLines(pieceGeomsXml(spec.moving, spec.wire.radius, movingRgba, 420, 'moving'), 6);

  return [
    `<mujoco model="chie-no-wa">`,
    `  <compiler angle="radian" autolimits="true"/>`,
    `  <size njmax="12000" nconmax="2000"/>`,
    `  <option timestep="0.0025" gravity="0 0 0" iterations="80" integrator="Euler"/>`,
    `  <default>`,
    `    <joint damping="0.25"/>`,
    `    <geom condim="3" friction="0.60 0.02 0.002" solref="0.004 1" solimp="0.995 0.999 0.0005"/>`,
    `  </default>`,
    `  <visual>`,
    `    <map znear="0.001" zfar="40"/>`,
    `  </visual>`,
    `  <worldbody>`,
    `    <light pos="2 -2 4" dir="-0.3 0.3 -1"/>`,
    `    <geom name="floor" type="plane" pos="0 0 -2.5" size="6 6 0.2" rgba="0.03 0.04 0.06 1" contype="0" conaffinity="0"/>`,
    fixedXml,
    `    <body name="player" pos="${formatVec(spec.startPose.pos)}" quat="${formatQuat(spec.startPose.quat)}">`,
    `      <freejoint name="player-free"/>`,
    movingXml,
    `    </body>`,
    `  </worldbody>`,
    `</mujoco>`,
  ].join('\n');
}
