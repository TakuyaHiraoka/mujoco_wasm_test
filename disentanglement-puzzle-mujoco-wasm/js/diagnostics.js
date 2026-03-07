import {
  clamp,
  cross,
  distance,
  dot,
  formatQuat,
  formatVec,
  length,
  minSegmentDistance,
  radiansToDegrees,
  scale,
  sub,
  transformDirection,
  transformPoint,
  transformSegments,
} from './math.js';

const LOCALE_TEXT = {
  ja: {
    checks: {
      fixedGapRatio: '固定片 gap / 線径比',
      movingGapRatio: '可動片 gap / 線径比',
      interPieceClearance: '開始時の片間クリアランス',
      fixedSelfClearance: '固定片の自己クリアランス',
      movingSelfClearance: '可動片の自己クリアランス',
      orthogonality: '開始姿勢の主ループ直交性',
      mutualHole: '相互 hole 食い込みスコア',
      startThreading: '開始時の主ループ貫通数',
      solveClearanceMargin: 'クリア閾値 clearance 余裕',
      irMechanism: 'IR メカニズム',
    },
    details: {
      gapTooTight: '狭すぎて理論上の通しが難しい可能性。',
      gapWide: '広く、抜け道が増えやすい。',
      gapWideMild: 'やや広め。',
      gapGood: '古典的 gap-loop 系として妥当範囲。',
      gapOk: '妥当。',
      intersecting: '初期交差あり。',
      nearContact: '接触直前。ブラウザ差で不安定化しやすい。',
      noInitialContact: '初期接触なし。',
      selfIntersecting: '自己交差。',
      selfTight: 'タイト。',
      selfClear: '自己交差なし。',
      orthogonalNatural: '直交に近く、alpha 系の噛み合わせとして自然。',
      orthogonalSkewed: 'やや偏りあり。',
      orthogonalPoor: '直交性が低く、開始姿勢が不自然。',
      mutualStrong: '両ループが十分に噛み合う配置。',
      mutualWeak: '弱め。簡単化の恐れ。',
      mutualPoor: 'ループ噛み合いが弱い。',
      threadingStrong: '少なくとも一方の主ループ面を相手片が貫いており、知恵の輪らしい噛み合いがある。',
      threadingGapOnly: '貫通は gap 近傍のみ。やや抜けやすい。',
      threadingWeak: '開始時の位相的噛み合いが弱い可能性。',
      solveMarginGood: '開始状態より十分大きい clearance が必要。',
      solveMarginSmall: '余裕が小さい。早すぎるクリアに注意。',
      solveMarginInvalid: '開始状態で既に閾値を満たしている。判定が緩すぎる。',
      irIntrinsic: 'Loop + Gap ノードから幾何を起こす intrinsic 表現を使用。',
    },
    runtimeLabels: {
      separation: 'separation',
      interPieceClearance: 'interPieceClearance',
      threadingCrossings: 'threadingCrossings',
      threadingGapCrossings: 'threadingGapCrossings',
      threadingScore: 'threadingScore',
      mutualHole: 'mutualHole',
      centerDistance: 'centerDistance',
      loopNormalAngle: 'loopNormalAngle',
      gapAlignment: 'gapAlignment',
      speed: 'speed',
      angularSpeed: 'angSpeed',
      contacts: 'contacts',
      state: 'state',
      warnings: 'warnings',
    },
    warnings: {
      nonFiniteCenterDistance: 'centerDistance が非有限値です。',
      highSpeed: '線速度が大きいです。推力を下げると安定します。',
      highAngularSpeed: '角速度が大きいです。',
      manyContacts: '接触数が多いです。局所的に詰まっている可能性があります。',
      nongapThreadingRemaining: 'まだ主ループ面の貫通が残っています。未クリアです。',
      gapThreadingRemaining: 'まだ gap 近傍に貫通が残っています。あと少し通し切る必要があります。',
      separationTooHigh: '分離度が高すぎます。位相貫通が残っている可能性があります。',
    },
  },
  en: {
    checks: {
      fixedGapRatio: 'Fixed gap / wire ratio',
      movingGapRatio: 'Moving gap / wire ratio',
      interPieceClearance: 'Start inter-piece clearance',
      fixedSelfClearance: 'Fixed piece self-clearance',
      movingSelfClearance: 'Moving piece self-clearance',
      orthogonality: 'Starting loop orthogonality',
      mutualHole: 'Mutual hole overlap score',
      startThreading: 'Starting loop threading count',
      solveClearanceMargin: 'Solve clearance margin',
      irMechanism: 'IR mechanism',
    },
    details: {
      gapTooTight: 'Too tight; theoretical disentanglement may be impossible.',
      gapWide: 'Wide enough that escape shortcuts become more likely.',
      gapWideMild: 'Slightly wide.',
      gapGood: 'A plausible range for a classic gap-loop puzzle.',
      gapOk: 'Acceptable.',
      intersecting: 'Starts with interpenetration.',
      nearContact: 'Starts almost in contact. Browser differences may make it unstable.',
      noInitialContact: 'No initial contact detected.',
      selfIntersecting: 'Self-intersection detected.',
      selfTight: 'Tight but still valid.',
      selfClear: 'No self-intersection.',
      orthogonalNatural: 'Close to orthogonal, which is natural for an alpha-style entanglement.',
      orthogonalSkewed: 'Somewhat skewed.',
      orthogonalPoor: 'Poor orthogonality; the start pose may feel unnatural.',
      mutualStrong: 'The two loops interlock strongly.',
      mutualWeak: 'Interlock is weak and may make the puzzle easier.',
      mutualPoor: 'Loop interlock is too weak.',
      threadingStrong: 'At least one primary loop plane is threaded by the opposite piece, so the topology feels puzzle-like.',
      threadingGapOnly: 'Threading only occurs near the gap, which may make the puzzle easier.',
      threadingWeak: 'The starting topology may be too weakly entangled.',
      solveMarginGood: 'Solving requires a clearance meaningfully larger than the starting state.',
      solveMarginSmall: 'The margin is small. Watch out for premature solve states.',
      solveMarginInvalid: 'The starting state already satisfies the solve threshold. The solve condition is too loose.',
      irIntrinsic: 'Uses an intrinsic Loop + Gap graph to generate geometry.',
    },
    runtimeLabels: {
      separation: 'separation',
      interPieceClearance: 'interPieceClearance',
      threadingCrossings: 'threadingCrossings',
      threadingGapCrossings: 'threadingGapCrossings',
      threadingScore: 'threadingScore',
      mutualHole: 'mutualHole',
      centerDistance: 'centerDistance',
      loopNormalAngle: 'loopNormalAngle',
      gapAlignment: 'gapAlignment',
      speed: 'speed',
      angularSpeed: 'angSpeed',
      contacts: 'contacts',
      state: 'state',
      warnings: 'warnings',
    },
    warnings: {
      nonFiniteCenterDistance: 'centerDistance is non-finite.',
      highSpeed: 'Linear speed is high. Lower thrust for more stable manipulation.',
      highAngularSpeed: 'Angular speed is high.',
      manyContacts: 'Contact count is high. The pieces may be locally jammed.',
      nongapThreadingRemaining: 'A non-gap threading crossing still remains. The puzzle is not solved yet.',
      gapThreadingRemaining: 'A gap-adjacent threading crossing remains. It needs a little more clearance.',
      separationTooHigh: 'Separation looks too high while topology still remains. Check the progress heuristic.',
    },
  },
};

function normalizeLocale(locale) {
  if (!locale) return 'ja';
  return String(locale).toLowerCase().startsWith('en') ? 'en' : 'ja';
}

function pickText(locale, section, code) {
  const normalized = normalizeLocale(locale);
  return LOCALE_TEXT[normalized][section][code] ?? LOCALE_TEXT.ja[section][code] ?? code;
}

function statusIcon(status) {
  if (status === 'pass') return '✔';
  if (status === 'warn') return '⚠';
  if (status === 'fail') return '✖';
  return 'ℹ';
}

function holeContainmentScore(center, diskCenter, diskNormal, diskRadius) {
  const rel = sub(center, diskCenter);
  const axial = Math.abs(dot(rel, diskNormal));
  const radialVector = sub(rel, scale(diskNormal, dot(rel, diskNormal)));
  const radial = length(radialVector);
  const axialScore = clamp(1 - axial / (diskRadius * 0.42), 0, 1);
  const radialScore = clamp(1 - radial / (diskRadius * 1.02), 0, 1);
  return axialScore * radialScore;
}

function wrapAngle(angle) {
  let wrapped = angle;
  while (wrapped <= -Math.PI) wrapped += Math.PI * 2;
  while (wrapped > Math.PI) wrapped -= Math.PI * 2;
  return wrapped;
}

function buildDiskFrame(normal, gapOut) {
  let u = sub(gapOut, scale(normal, dot(gapOut, normal)));

  if (length(u) < 1e-8) {
    const helper = Math.abs(normal[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
    u = cross(helper, normal);
  }

  u = normalizeVector(u);
  const v = normalizeVector(cross(normal, u));

  return { u, v };
}

function normalizeVector(v) {
  const len = length(v);
  if (len < 1e-8) return [0, 0, 0];
  return scale(v, 1 / len);
}

function segmentPoint(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

function evaluateDiskThreading(
  segments,
  { center, normal, radius, gapOut, gapHalfAngleRad },
  wireRadius,
) {
  const { u, v } = buildDiskFrame(normal, gapOut);
  const planeTolerance = wireRadius * 0.18 + 1e-7;
  const gapMarginRad = Math.asin(
    clamp((wireRadius * 1.15) / Math.max(radius, wireRadius * 1.4), 0, 0.72),
  );
  const deepInteriorRadius = Math.max(radius - wireRadius * 0.45, wireRadius * 1.15);

  let crossings = 0;
  let gapCrossings = 0;
  let nearCrossings = 0;
  let maxPenetration = 0;

  for (const segment of segments) {
    const da = dot(sub(segment.a, center), normal);
    const db = dot(sub(segment.b, center), normal);

    if (Math.abs(da) <= planeTolerance && Math.abs(db) <= planeTolerance) {
      continue;
    }

    const straddlesPlane =
      (da <= planeTolerance && db >= -planeTolerance) ||
      (da >= -planeTolerance && db <= planeTolerance);

    if (!straddlesPlane) continue;

    const denominator = da - db;
    if (Math.abs(denominator) < 1e-8) continue;

    const t = da / denominator;
    if (t < -0.02 || t > 1.02) continue;

    const point = segmentPoint(segment.a, segment.b, clamp(t, 0, 1));
    const rel = sub(point, center);
    const axial = dot(rel, normal);
    const radialVec = sub(rel, scale(normal, axial));
    const radial = length(radialVec);

    if (radial > radius + wireRadius * 0.9) continue;

    const theta = radial > 1e-8 ? Math.atan2(dot(radialVec, v), dot(radialVec, u)) : 0;
    const throughGap = Math.abs(wrapAngle(theta)) <= gapHalfAngleRad + gapMarginRad;
    const penetration = clamp(
      (deepInteriorRadius - radial) / Math.max(deepInteriorRadius, 1e-6),
      0,
      1,
    );

    if (radial <= deepInteriorRadius) {
      if (throughGap) {
        gapCrossings += 1;
      } else {
        crossings += 1;
      }
      maxPenetration = Math.max(maxPenetration, penetration);
    } else {
      nearCrossings += 1;
    }
  }

  const score = clamp(
    crossings * 0.58 + gapCrossings * 0.18 + nearCrossings * 0.08 + maxPenetration * 0.9,
    0,
    1,
  );

  return {
    crossings,
    gapCrossings,
    nearCrossings,
    maxPenetration,
    score,
  };
}

function computeSelfClearance(piece, wireRadius) {
  const lastIndex = piece.segmentsLocal.length - 1;
  const gapNeighborhood = 3;

  const skip = (i, j) => {
    if (Math.abs(i - j) <= 2) return true;

    const acrossIntentionalGap =
      (i <= gapNeighborhood && j >= lastIndex - gapNeighborhood) ||
      (j <= gapNeighborhood && i >= lastIndex - gapNeighborhood);

    return acrossIntentionalGap;
  };

  const result = minSegmentDistance(piece.segmentsLocal, piece.segmentsLocal, skip);
  return result.distance - 2 * wireRadius;
}

function makeCheck(code, status, value, detailCode = null) {
  return { code, status, value, detailCode };
}

export function computePoseGeometry(spec, pose) {
  const pos = pose.pos ?? spec.startPose.pos;
  const quat = pose.quat ?? spec.startPose.quat;

  return {
    fixed: {
      holeCenter: spec.fixed.holeCenterLocal,
      holeNormal: spec.fixed.loopNormalLocal,
      gapCenter: spec.fixed.gap.centerLocal,
      gapOut: spec.fixed.gap.outLocal,
      gapA: spec.fixed.gap.aLocal,
      gapB: spec.fixed.gap.bLocal,
      radius: spec.fixed.primaryLoopRadius,
    },
    moving: {
      holeCenter: transformPoint(spec.moving.holeCenterLocal, pos, quat),
      holeNormal: transformDirection(spec.moving.loopNormalLocal, quat),
      gapCenter: transformPoint(spec.moving.gap.centerLocal, pos, quat),
      gapOut: transformDirection(spec.moving.gap.outLocal, quat),
      gapA: transformPoint(spec.moving.gap.aLocal, pos, quat),
      gapB: transformPoint(spec.moving.gap.bLocal, pos, quat),
      radius: spec.moving.primaryLoopRadius,
    },
  };
}

export function computeThreadingMetrics(spec, markers, movingSegments) {
  const fixedDisk = evaluateDiskThreading(
    movingSegments,
    {
      center: markers.fixed.holeCenter,
      normal: markers.fixed.holeNormal,
      radius: markers.fixed.radius,
      gapOut: markers.fixed.gapOut,
      gapHalfAngleRad: (spec.fixed.gap.angleDeg * Math.PI / 180) * 0.5,
    },
    spec.wire.radius,
  );

  const movingDisk = evaluateDiskThreading(
    spec.fixed.segmentsLocal,
    {
      center: markers.moving.holeCenter,
      normal: markers.moving.holeNormal,
      radius: markers.moving.radius,
      gapOut: markers.moving.gapOut,
      gapHalfAngleRad: (spec.moving.gap.angleDeg * Math.PI / 180) * 0.5,
    },
    spec.wire.radius,
  );

  const nongapCrossings = fixedDisk.crossings + movingDisk.crossings;
  const gapCrossings = fixedDisk.gapCrossings + movingDisk.gapCrossings;
  const nearCrossings = fixedDisk.nearCrossings + movingDisk.nearCrossings;
  const maxPenetration = Math.max(fixedDisk.maxPenetration, movingDisk.maxPenetration);
  const score = clamp(Math.max(fixedDisk.score, movingDisk.score), 0, 1);

  return {
    fixedDisk,
    movingDisk,
    nongapCrossings,
    gapCrossings,
    nearCrossings,
    maxPenetration,
    score,
  };
}

export function runStaticDiagnostics(spec) {
  const pose = {
    pos: spec.startPose.pos,
    quat: spec.startPose.quat,
  };

  const movingSegments = transformSegments(spec.moving.segmentsLocal, pose.pos, pose.quat);
  const inter = minSegmentDistance(spec.fixed.segmentsLocal, movingSegments).distance - 2 * spec.wire.radius;
  const fixedSelf = computeSelfClearance(spec.fixed, spec.wire.radius);
  const movingSelf = computeSelfClearance(spec.moving, spec.wire.radius);
  const markers = computePoseGeometry(spec, pose);
  const threading = computeThreadingMetrics(spec, markers, movingSegments);

  const angle = radiansToDegrees(
    Math.acos(clamp(Math.abs(dot(markers.fixed.holeNormal, markers.moving.holeNormal)), -1, 1)),
  );

  const mutualHole =
    0.5 *
    (
      holeContainmentScore(
        markers.moving.holeCenter,
        markers.fixed.holeCenter,
        markers.fixed.holeNormal,
        markers.fixed.radius,
      ) +
      holeContainmentScore(
        markers.fixed.holeCenter,
        markers.moving.holeCenter,
        markers.moving.holeNormal,
        markers.moving.radius,
      )
    );

  const fixedGapRatio = spec.fixed.gap.width / spec.wire.diameter;
  const movingGapRatio = spec.moving.gap.width / spec.wire.diameter;
  const solveClearanceMargin = (spec.geometry?.solveClearance ?? inter) - inter;

  const checks = [];

  checks.push(
    makeCheck(
      'fixedGapRatio',
      fixedGapRatio < 1.1 ? 'fail' : fixedGapRatio > 2.6 ? 'warn' : 'pass',
      `${fixedGapRatio.toFixed(2)}x`,
      fixedGapRatio < 1.1 ? 'gapTooTight' : fixedGapRatio > 2.6 ? 'gapWide' : 'gapGood',
    ),
  );

  checks.push(
    makeCheck(
      'movingGapRatio',
      movingGapRatio < 1.1 ? 'fail' : movingGapRatio > 2.6 ? 'warn' : 'pass',
      `${movingGapRatio.toFixed(2)}x`,
      movingGapRatio < 1.1 ? 'gapTooTight' : movingGapRatio > 2.6 ? 'gapWideMild' : 'gapOk',
    ),
  );

  checks.push(
    makeCheck(
      'interPieceClearance',
      inter < 0 ? 'fail' : inter < spec.wire.radius * 0.12 ? 'warn' : 'pass',
      `${inter.toFixed(4)} m`,
      inter < 0 ? 'intersecting' : inter < spec.wire.radius * 0.12 ? 'nearContact' : 'noInitialContact',
    ),
  );

  checks.push(
    makeCheck(
      'fixedSelfClearance',
      fixedSelf < 0 ? 'fail' : fixedSelf < spec.wire.radius * 0.08 ? 'warn' : 'pass',
      `${fixedSelf.toFixed(4)} m`,
      fixedSelf < 0 ? 'selfIntersecting' : fixedSelf < spec.wire.radius * 0.08 ? 'selfTight' : 'selfClear',
    ),
  );

  checks.push(
    makeCheck(
      'movingSelfClearance',
      movingSelf < 0 ? 'fail' : movingSelf < spec.wire.radius * 0.08 ? 'warn' : 'pass',
      `${movingSelf.toFixed(4)} m`,
      movingSelf < 0 ? 'selfIntersecting' : movingSelf < spec.wire.radius * 0.08 ? 'selfTight' : 'selfClear',
    ),
  );

  checks.push(
    makeCheck(
      'orthogonality',
      Math.abs(angle - 90) < 20 ? 'pass' : Math.abs(angle - 90) < 35 ? 'warn' : 'fail',
      `${angle.toFixed(1)}°`,
      Math.abs(angle - 90) < 20
        ? 'orthogonalNatural'
        : Math.abs(angle - 90) < 35
          ? 'orthogonalSkewed'
          : 'orthogonalPoor',
    ),
  );

  checks.push(
    makeCheck(
      'mutualHole',
      mutualHole > 0.34 ? 'pass' : mutualHole > 0.18 ? 'warn' : 'fail',
      mutualHole.toFixed(2),
      mutualHole > 0.34 ? 'mutualStrong' : mutualHole > 0.18 ? 'mutualWeak' : 'mutualPoor',
    ),
  );

  checks.push(
    makeCheck(
      'startThreading',
      threading.nongapCrossings >= 1 ? 'pass' : threading.gapCrossings >= 1 ? 'warn' : 'fail',
      `${threading.nongapCrossings} (gap=${threading.gapCrossings})`,
      threading.nongapCrossings >= 1
        ? 'threadingStrong'
        : threading.gapCrossings >= 1
          ? 'threadingGapOnly'
          : 'threadingWeak',
    ),
  );

  checks.push(
    makeCheck(
      'solveClearanceMargin',
      solveClearanceMargin > Math.max(spec.wire.radius * 1.2, 0.03)
        ? 'pass'
        : solveClearanceMargin > 0
          ? 'warn'
          : 'fail',
      `${solveClearanceMargin.toFixed(4)} m`,
      solveClearanceMargin > Math.max(spec.wire.radius * 1.2, 0.03)
        ? 'solveMarginGood'
        : solveClearanceMargin > 0
          ? 'solveMarginSmall'
          : 'solveMarginInvalid',
    ),
  );

  checks.push(
    makeCheck(
      'irMechanism',
      'info',
      `${spec.family.id} / nodes=${spec.ir.nodes.length}, relations=${spec.ir.relations.length}`,
      'irIntrinsic',
    ),
  );

  return {
    checks,
    values: {
      fixedGapRatio,
      movingGapRatio,
      interPieceClearance: inter,
      fixedSelfClearance: fixedSelf,
      movingSelfClearance: movingSelf,
      orthogonalityAngleDeg: angle,
      mutualHole,
      threadingCrossings: threading.nongapCrossings,
      threadingGapCrossings: threading.gapCrossings,
      threadingScore: threading.score,
      solveClearanceMargin,
    },
    summary: {
      familyId: spec.family.id,
      avgGapRatio: spec.stats.avgGapRatio,
      twistProxy: spec.stats.twistProxy,
      estimatedDifficulty: spec.stats.estimatedDifficulty,
    },
  };
}

export function buildRuntimeDiagnostics(spec, pose) {
  const markers = computePoseGeometry(spec, pose);
  const centerDistance = distance(markers.fixed.holeCenter, markers.moving.holeCenter);
  const movingSegments = transformSegments(
    spec.moving.segmentsLocal,
    pose.pos ?? spec.startPose.pos,
    pose.quat ?? spec.startPose.quat,
  );
  const segmentDistance = minSegmentDistance(spec.fixed.segmentsLocal, movingSegments).distance;
  const interPieceClearance = segmentDistance - 2 * spec.wire.radius;
  const threading = computeThreadingMetrics(spec, markers, movingSegments);

  const mutualHole =
    0.5 *
    (
      holeContainmentScore(
        markers.moving.holeCenter,
        markers.fixed.holeCenter,
        markers.fixed.holeNormal,
        markers.fixed.radius,
      ) +
      holeContainmentScore(
        markers.fixed.holeCenter,
        markers.moving.holeCenter,
        markers.moving.holeNormal,
        markers.moving.radius,
      )
    );

  const startClearance = spec.startPoseStats?.clearance ?? 0;
  const clearanceTarget = Math.max(
    spec.geometry.solveClearance ?? Math.max(spec.wire.radius * 2.4, 0.03),
    startClearance + Math.max(spec.wire.radius * 1.2, 0.03),
  );

  const clearanceProgress = clamp(
    (interPieceClearance - startClearance) /
      Math.max(clearanceTarget - startClearance, spec.wire.radius * 1.1, 1e-6),
    0,
    1,
  );

  const startThreadingScore = Math.max(spec.startPoseStats?.threadingScore ?? 0.55, 0.12);
  const threadingProgress = 1 - clamp(threading.score / startThreadingScore, 0, 1);
  const distanceProgress = clamp(
    (centerDistance - spec.geometry.startCenterDistance) /
      Math.max(spec.geometry.solveDistance - spec.geometry.startCenterDistance, spec.wire.radius * 4.0, 1e-6),
    0,
    1,
  );

  let separation = clamp(
    0.5 * threadingProgress + 0.25 * clearanceProgress + 0.25 * distanceProgress,
    0,
    1,
  );

  if (threading.nongapCrossings > 0) {
    separation = Math.min(separation, 0.68);
  } else if (threading.gapCrossings > 0 || threading.score > 0.08) {
    separation = Math.min(separation, 0.88);
  }

  const normalAngle = radiansToDegrees(
    Math.acos(clamp(Math.abs(dot(markers.fixed.holeNormal, markers.moving.holeNormal)), -1, 1)),
  );
  const gapAlignment = radiansToDegrees(
    Math.acos(clamp(dot(markers.fixed.gapOut, markers.moving.gapOut), -1, 1)),
  );
  const speed = length(pose.linvel ?? [0, 0, 0]);
  const angularSpeed = length(pose.angvel ?? [0, 0, 0]);

  const warnings = [];

  if (!Number.isFinite(centerDistance)) warnings.push('nonFiniteCenterDistance');
  if (speed > 5) warnings.push('highSpeed');
  if (angularSpeed > 14) warnings.push('highAngularSpeed');
  if ((pose.ncon ?? 0) > 18) warnings.push('manyContacts');

  if (threading.nongapCrossings > 0) {
    warnings.push('nongapThreadingRemaining');
  } else if (threading.gapCrossings > 0) {
    warnings.push('gapThreadingRemaining');
  }

  if (separation > 0.9 && (threading.nongapCrossings > 0 || threading.score > 0.08)) {
    warnings.push('separationTooHigh');
  }

  return {
    centerDistance,
    segmentDistance,
    interPieceClearance,
    mutualHole,
    clearanceProgress,
    distanceProgress,
    threadingProgress,
    threadingCrossings: threading.nongapCrossings,
    threadingGapCrossings: threading.gapCrossings,
    threadingNearCrossings: threading.nearCrossings,
    threadingScore: threading.score,
    maxThreadingPenetration: threading.maxPenetration,
    separation,
    normalAngleDeg: normalAngle,
    gapAlignmentDeg: gapAlignment,
    speed,
    angularSpeed,
    contactCount: pose.ncon ?? 0,
    paused: !!pose.paused,
    solved: !!pose.solved,
    warnings,
  };
}

export function formatStaticDiagnostics(
  diag,
  { locale = 'ja', familyLabel = null } = {},
) {
  const summaryLine =
    `${familyLabel ?? diag.summary.familyId} | ` +
    `gap=${diag.summary.avgGapRatio.toFixed(2)}x | ` +
    `twist=${diag.summary.twistProxy.toFixed(2)} | ` +
    `diff=${diag.summary.estimatedDifficulty}/10`;

  const lines = [summaryLine];

  for (const check of diag.checks) {
    lines.push(`${statusIcon(check.status)} ${pickText(locale, 'checks', check.code)}: ${check.value}`);
    if (check.detailCode) {
      lines.push(`  ${pickText(locale, 'details', check.detailCode)}`);
    }
  }

  return lines.join('\n');
}

export function formatRuntimeDiagnostics(diag, { locale = 'ja' } = {}) {
  const labels = LOCALE_TEXT[normalizeLocale(locale)].runtimeLabels;
  const lines = [
    `${labels.separation}=${(diag.separation * 100).toFixed(1)} %`,
    `${labels.interPieceClearance}=${diag.interPieceClearance.toFixed(4)} m`,
    `${labels.threadingCrossings}=${diag.threadingCrossings}`,
    `${labels.threadingGapCrossings}=${diag.threadingGapCrossings}`,
    `${labels.threadingScore}=${diag.threadingScore.toFixed(3)}`,
    `${labels.mutualHole}=${diag.mutualHole.toFixed(3)}`,
    `${labels.centerDistance}=${diag.centerDistance.toFixed(3)} m`,
    `${labels.loopNormalAngle}=${diag.normalAngleDeg.toFixed(1)} °`,
    `${labels.gapAlignment}=${diag.gapAlignmentDeg.toFixed(1)} °`,
    `${labels.speed}=${diag.speed.toFixed(3)} m/s`,
    `${labels.angularSpeed}=${diag.angularSpeed.toFixed(3)} rad/s`,
    `${labels.contacts}=${diag.contactCount}`,
    `${labels.state}=paused=${diag.paused ? 'true' : 'false'} solved=${diag.solved ? 'true' : 'false'}`,
  ];

  if (diag.warnings.length) {
    lines.push('');
    lines.push(`${labels.warnings}:`);
    diag.warnings.forEach((warningCode) => {
      lines.push(`- ${pickText(locale, 'warnings', warningCode)}`);
    });
  }

  return lines.join('\n');
}

export function buildDiagnosticsBundle({ spec, staticDiag, runtimeDiag, logLines }) {
  return {
    generatedAt: new Date().toISOString(),
    spec: {
      seed: spec.seed,
      complexity: spec.complexity,
      family: spec.family,
      wire: spec.wire,
      geometry: spec.geometry,
      stats: spec.stats,
      startPoseStats: spec.startPoseStats,
      startPose: {
        pos: formatVec(spec.startPose.pos),
        quat: formatQuat(spec.startPose.quat),
      },
      ir: spec.ir,
      generationLog: spec.generationLog,
    },
    staticDiagnostics: staticDiag,
    runtimeDiagnostics: runtimeDiag,
    logLines,
  };
}

export function downloadText(filename, text) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
