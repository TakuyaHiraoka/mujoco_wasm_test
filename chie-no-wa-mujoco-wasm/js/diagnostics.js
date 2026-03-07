import {
  clamp,
  distance,
  dot,
  formatQuat,
  formatVec,
  length,
  minSegmentDistance,
  radiansToDegrees,
  transformDirection,
  transformPoint,
  transformSegments,
  sub,
  scale,
} from './math.js';

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

function computeSelfClearance(piece, wireRadius) {
  const lastIndex = piece.segmentsLocal.length - 1;
  const gapNeighborhood = 3;
  const skip = (i, j) => {
    if (Math.abs(i - j) <= 2) return true;
    const acrossIntentionalGap = (i <= gapNeighborhood && j >= lastIndex - gapNeighborhood)
      || (j <= gapNeighborhood && i >= lastIndex - gapNeighborhood);
    return acrossIntentionalGap;
  };
  const result = minSegmentDistance(piece.segmentsLocal, piece.segmentsLocal, skip);
  return result.distance - 2 * wireRadius;
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

function makeCheck(label, status, value, detail) {
  return { label, status, value, detail };
}

export function runStaticDiagnostics(spec) {
  const pose = { pos: spec.startPose.pos, quat: spec.startPose.quat };
  const movingSegments = transformSegments(spec.moving.segmentsLocal, pose.pos, pose.quat);
  const inter = minSegmentDistance(spec.fixed.segmentsLocal, movingSegments).distance - 2 * spec.wire.radius;
  const fixedSelf = computeSelfClearance(spec.fixed, spec.wire.radius);
  const movingSelf = computeSelfClearance(spec.moving, spec.wire.radius);
  const markers = computePoseGeometry(spec, pose);
  const angle = radiansToDegrees(Math.acos(clamp(Math.abs(dot(markers.fixed.holeNormal, markers.moving.holeNormal)), -1, 1)));
  const mutualHole = 0.5 * (
    holeContainmentScore(markers.moving.holeCenter, markers.fixed.holeCenter, markers.fixed.holeNormal, markers.fixed.radius)
    + holeContainmentScore(markers.fixed.holeCenter, markers.moving.holeCenter, markers.moving.holeNormal, markers.moving.radius)
  );

  const checks = [];
  const fixedGapRatio = spec.fixed.gap.width / spec.wire.diameter;
  const movingGapRatio = spec.moving.gap.width / spec.wire.diameter;

  checks.push(makeCheck(
    '固定片 gap / 線径比',
    fixedGapRatio < 1.10 ? 'fail' : fixedGapRatio > 2.60 ? 'warn' : 'pass',
    `${fixedGapRatio.toFixed(2)}x`,
    fixedGapRatio < 1.10 ? '狭すぎて理論上の通しが難しい可能性。' : fixedGapRatio > 2.60 ? '広く、抜け道が増えやすい。' : '古典的 gap-loop 系として妥当範囲。',
  ));

  checks.push(makeCheck(
    '可動片 gap / 線径比',
    movingGapRatio < 1.10 ? 'fail' : movingGapRatio > 2.60 ? 'warn' : 'pass',
    `${movingGapRatio.toFixed(2)}x`,
    movingGapRatio < 1.10 ? '狭すぎる可能性。' : movingGapRatio > 2.60 ? 'やや広め。' : '妥当。',
  ));

  checks.push(makeCheck(
    '開始時の片間クリアランス',
    inter < 0 ? 'fail' : inter < spec.wire.radius * 0.12 ? 'warn' : 'pass',
    `${inter.toFixed(4)} m`,
    inter < 0 ? '初期交差あり。' : inter < spec.wire.radius * 0.12 ? '接触直前。ブラウザ差で不安定化しやすい。' : '初期接触なし。',
  ));

  checks.push(makeCheck(
    '固定片の自己クリアランス',
    fixedSelf < 0 ? 'fail' : fixedSelf < spec.wire.radius * 0.08 ? 'warn' : 'pass',
    `${fixedSelf.toFixed(4)} m`,
    fixedSelf < 0 ? '自己交差。' : fixedSelf < spec.wire.radius * 0.08 ? 'タイト。' : '自己交差なし。',
  ));

  checks.push(makeCheck(
    '可動片の自己クリアランス',
    movingSelf < 0 ? 'fail' : movingSelf < spec.wire.radius * 0.08 ? 'warn' : 'pass',
    `${movingSelf.toFixed(4)} m`,
    movingSelf < 0 ? '自己交差。' : movingSelf < spec.wire.radius * 0.08 ? 'タイト。' : '自己交差なし。',
  ));

  checks.push(makeCheck(
    '開始姿勢の主ループ直交性',
    Math.abs(angle - 90) < 20 ? 'pass' : Math.abs(angle - 90) < 35 ? 'warn' : 'fail',
    `${angle.toFixed(1)}°`,
    Math.abs(angle - 90) < 20 ? '直交に近く、alpha 系の噛み合わせとして自然。' : Math.abs(angle - 90) < 35 ? 'やや偏りあり。' : '直交性が低く、開始姿勢が不自然。',
  ));

  checks.push(makeCheck(
    '相互 hole 食い込みスコア',
    mutualHole > 0.34 ? 'pass' : mutualHole > 0.18 ? 'warn' : 'fail',
    mutualHole.toFixed(2),
    mutualHole > 0.34 ? '両ループが十分に噛み合う配置。' : mutualHole > 0.18 ? '弱め。簡単化の恐れ。' : 'ループ噛み合いが弱い。',
  ));

  checks.push(makeCheck(
    'IR メカニズム',
    'info',
    `${spec.family.id} / nodes=${spec.ir.nodes.length}, relations=${spec.ir.relations.length}`,
    'Loop + Gap ノードから幾何を起こす intrinsic 表現を使用。',
  ));

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
    },
    summary: `${spec.family.id} | gap=${spec.stats.avgGapRatio.toFixed(2)}x | twist=${spec.stats.twistProxy.toFixed(2)} | diff=${spec.stats.estimatedDifficulty}/10`,
  };
}

export function buildRuntimeDiagnostics(spec, pose) {
  const markers = computePoseGeometry(spec, pose);
  const centerDistance = distance(markers.fixed.holeCenter, markers.moving.holeCenter);
  const separation = clamp(
    (centerDistance - spec.geometry.startCenterDistance) / (spec.geometry.solveDistance - spec.geometry.startCenterDistance),
    0,
    1.25,
  );
  const normalAngle = radiansToDegrees(Math.acos(clamp(Math.abs(dot(markers.fixed.holeNormal, markers.moving.holeNormal)), -1, 1)));
  const gapAlignment = radiansToDegrees(Math.acos(clamp(dot(markers.fixed.gapOut, markers.moving.gapOut), -1, 1)));
  const speed = length(pose.linvel ?? [0, 0, 0]);
  const angularSpeed = length(pose.angvel ?? [0, 0, 0]);

  const warnings = [];
  if (!Number.isFinite(centerDistance)) warnings.push('centerDistance が非有限値です。');
  if (speed > 5) warnings.push('線速度が大きいです。推力を下げると安定します。');
  if (angularSpeed > 14) warnings.push('角速度が大きいです。');
  if ((pose.ncon ?? 0) > 18) warnings.push('接触数が多いです。局所的に詰まっている可能性があります。');

  return {
    centerDistance,
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

export function formatStaticDiagnostics(diag) {
  const lines = [diag.summary];
  for (const check of diag.checks) {
    lines.push(`${statusIcon(check.status)} ${check.label}: ${check.value}`);
    if (check.detail) {
      lines.push(`   ${check.detail}`);
    }
  }
  return lines.join('\n');
}

export function formatRuntimeDiagnostics(diag) {
  const lines = [
    `separation=${(diag.separation * 100).toFixed(1)} %`,
    `centerDistance=${diag.centerDistance.toFixed(3)} m`,
    `loopNormalAngle=${diag.normalAngleDeg.toFixed(1)} °`,
    `gapAlignment=${diag.gapAlignmentDeg.toFixed(1)} °`,
    `speed=${diag.speed.toFixed(3)} m/s`,
    `angSpeed=${diag.angularSpeed.toFixed(3)} rad/s`,
    `contacts=${diag.contactCount}`,
    `paused=${diag.paused ? 'true' : 'false'} solved=${diag.solved ? 'true' : 'false'}`,
  ];
  if (diag.warnings.length) {
    lines.push('');
    lines.push('warnings:');
    diag.warnings.forEach((warning) => lines.push(`- ${warning}`));
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
